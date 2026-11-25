"""Pythonic model API over the popso C library."""

import ctypes
import math
import numbers
import os
import threading

from ._lib import INCUMBENT_CALLBACK, lib

INF = float("inf")

CONTINUOUS = 0
INTEGER = 1
BINARY = 2
SEMICONTINUOUS = 3  # x = 0 or lb <= x <= ub
SEMIINTEGER = 4     # x = 0 or lb <= x <= ub, integer

SOS1 = 1
SOS2 = 2

MINIMIZE = 1
MAXIMIZE = -1

# Basis status (Result.var_basis / Result.constr_basis).
BASIC = 0
AT_LOWER = 1
AT_UPPER = 2
AT_ZERO = 3  # nonbasic free variable (or free constraint) at zero

_VTYPES = {
    CONTINUOUS: CONTINUOUS, INTEGER: INTEGER, BINARY: BINARY, SEMICONTINUOUS: SEMICONTINUOUS,
    SEMIINTEGER: SEMIINTEGER,
    "C": CONTINUOUS, "I": INTEGER, "B": BINARY, "S": SEMICONTINUOUS, "N": SEMIINTEGER,
    "continuous": CONTINUOUS, "integer": INTEGER, "binary": BINARY, "semicontinuous": SEMICONTINUOUS,
    "semiinteger": SEMIINTEGER,
}
_SENSES = {MINIMIZE: MINIMIZE, MAXIMIZE: MAXIMIZE, "min": MINIMIZE, "max": MAXIMIZE,
           "minimize": MINIMIZE, "maximize": MAXIMIZE}

PARAMS = ("time_limit", "threads", "mip_gap", "mip_abs_gap", "seed", "deterministic", "verbosity",
          "mem_limit", "node_limit", "presolve", "racing", "lp_method", "duals")
_INT_PARAMS = {"threads", "seed", "verbosity", "mem_limit", "node_limit", "racing", "lp_method", "duals"}
# lp_method also accepts these names.
LP_METHODS = {"auto": 0, "simplex": 1, "barrier": 2}
_BOOL_PARAMS = {"deterministic", "presolve"}


class PopsoError(Exception):
    """An error reported by the popso library (code is the POPSO_ERROR_* value)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _enc(s):
    return None if s is None else str(s).encode("utf-8")


def _dec(b):
    return None if b is None else b.decode("utf-8", "replace")


def version():
    """Version of the loaded popso library."""
    return _dec(lib.popso_version())


def _darray(values):
    return (ctypes.c_double * len(values))(*[float(v) for v in values])


def _iarray(values):
    return (ctypes.c_int * len(values))(*[int(v) for v in values])


# ---------------------------------------------------------------- expressions

class LinExpr:
    """A linear expression sum(coef * var) + constant. Build it with operators on Vars."""

    __slots__ = ("terms", "constant", "_model")

    def __init__(self, terms=None, constant=0.0, model=None):
        self.terms = dict(terms) if terms else {}  # variable index -> coefficient
        self.constant = float(constant)
        self._model = model

    @staticmethod
    def _of(x):
        if isinstance(x, LinExpr):
            return x
        if isinstance(x, Var):
            return LinExpr({x.index: 1.0}, 0.0, x.model)
        if isinstance(x, numbers.Real):
            return LinExpr(None, float(x))
        return None

    def _combine(self, other, sign):
        o = LinExpr._of(other)
        if o is None:
            return NotImplemented
        model = self._model or o._model
        if self._model is not None and o._model is not None and self._model is not o._model:
            raise ValueError("cannot combine variables of different models")
        terms = dict(self.terms)
        for j, c in o.terms.items():
            terms[j] = terms.get(j, 0.0) + sign * c
        return LinExpr(terms, self.constant + sign * o.constant, model)

    def __add__(self, other):
        return self._combine(other, 1.0)

    def __radd__(self, other):
        return self._combine(other, 1.0)

    def __sub__(self, other):
        return self._combine(other, -1.0)

    def __rsub__(self, other):
        r = self._combine(other, -1.0)
        return NotImplemented if r is NotImplemented else -r

    def __mul__(self, k):
        if isinstance(k, (Var, LinExpr)):
            return QuadExpr._product(self, LinExpr._of(k))
        if not isinstance(k, numbers.Real):
            return NotImplemented
        k = float(k)
        return LinExpr({j: c * k for j, c in self.terms.items()}, self.constant * k, self._model)

    __rmul__ = __mul__

    def __pow__(self, e):
        if e != 2:
            return NotImplemented  # only squares
        return QuadExpr._product(self, self)

    def __truediv__(self, k):
        if not isinstance(k, numbers.Real):
            return NotImplemented
        return self * (1.0 / float(k))

    def __neg__(self):
        return self * -1.0

    def __pos__(self):
        return self

    def _relation(self, other, kind):
        o = LinExpr._of(other)
        if o is None:
            return NotImplemented
        d = self - o  # d <= 0, d >= 0 or d == 0
        rhs = -d.constant
        d = LinExpr(d.terms, 0.0, d._model)
        if kind == "le":
            return Constraint(d, -INF, rhs)
        if kind == "ge":
            return Constraint(d, rhs, INF)
        return Constraint(d, rhs, rhs)

    def __le__(self, other):
        return self._relation(other, "le")

    def __ge__(self, other):
        return self._relation(other, "ge")

    def __eq__(self, other):
        return self._relation(other, "eq")

    __hash__ = None

    def value(self, result):
        """Value of the expression at a solve result's solution."""
        x = result.x
        return self.constant + sum(c * x[j] for j, c in self.terms.items())

    def __repr__(self):
        parts = ["%g*%s" % (c, self._model.var(j).name if self._model else "x%d" % j) for j, c in self.terms.items()]
        if self.constant or not parts:
            parts.append("%g" % self.constant)
        return " + ".join(parts)


class QuadExpr:
    """A quadratic expression: sum(coef * x_i * x_j) + a linear part. Built by multiplying variables or
    linear expressions (x * y, x * x, x ** 2, (x + y) * (x - 2 * z)). As an objective it must be convex
    (concave when maximizing); compared with <=, >= it gives a quadratic constraint, which must be
    convex (x ** 2 + y ** 2 <= 4, not >=; == is not allowed)."""

    __slots__ = ("quad", "linear")

    def __init__(self, quad=None, linear=None):
        self.quad = dict(quad) if quad else {}  # (i, j) with i >= j -> coefficient of x_i * x_j
        self.linear = linear if linear is not None else LinExpr()

    @staticmethod
    def _product(a, b):
        if a._model is not None and b._model is not None and a._model is not b._model:
            raise ValueError("cannot combine variables of different models")
        q = {}
        for i, ci in a.terms.items():
            for j, cj in b.terms.items():
                key = (max(i, j), min(i, j))
                q[key] = q.get(key, 0.0) + ci * cj
        model = a._model or b._model
        lin = LinExpr(None, a.constant * b.constant, model)
        if a.constant != 0.0:
            lin = lin + LinExpr({j: c * a.constant for j, c in b.terms.items()}, 0.0, model)
        if b.constant != 0.0:
            lin = lin + LinExpr({j: c * b.constant for j, c in a.terms.items()}, 0.0, model)
        return QuadExpr(q, lin)

    @property
    def _model(self):
        return self.linear._model

    @property
    def constant(self):
        return self.linear.constant

    def _combine(self, other, sign):
        if isinstance(other, QuadExpr):
            q = dict(self.quad)
            for k, c in other.quad.items():
                q[k] = q.get(k, 0.0) + sign * c
            lin = self.linear._combine(other.linear, sign)
            return QuadExpr(q, lin)
        o = LinExpr._of(other)
        if o is None:
            return NotImplemented
        return QuadExpr(self.quad, self.linear._combine(o, sign))

    def __add__(self, other):
        return self._combine(other, 1.0)

    __radd__ = __add__

    def __sub__(self, other):
        return self._combine(other, -1.0)

    def __rsub__(self, other):
        r = self._combine(other, -1.0)
        return NotImplemented if r is NotImplemented else -r

    def __mul__(self, k):
        if not isinstance(k, numbers.Real):
            return NotImplemented  # degree 2 at most
        k = float(k)
        return QuadExpr({key: c * k for key, c in self.quad.items()}, self.linear * k)

    __rmul__ = __mul__

    def __truediv__(self, k):
        if not isinstance(k, numbers.Real):
            return NotImplemented
        return self * (1.0 / float(k))

    def __neg__(self):
        return self * -1.0

    def __pos__(self):
        return self

    def _relation(self, other, kind):
        if isinstance(other, QuadExpr):
            d = self - other
        else:
            o = LinExpr._of(other)
            if o is None:
                return NotImplemented
            d = self - o  # d <= 0 or d >= 0
        rhs = -d.constant
        d = QuadExpr(d.quad, LinExpr(d.linear.terms, 0.0, d._model))
        return Constraint(d, -INF, rhs) if kind == "le" else Constraint(d, rhs, INF)

    def __le__(self, other):
        return self._relation(other, "le")

    def __ge__(self, other):
        return self._relation(other, "ge")

    def __eq__(self, other):
        raise TypeError("a quadratic equality is not convex (use <= or >=)")

    __hash__ = None

    def value(self, result):
        """Value of the expression at a solve result's solution."""
        x = result.x
        return self.linear.value(result) + sum(c * x[i] * x[j] for (i, j), c in self.quad.items())

    def __repr__(self):
        name = (lambda j: self._model.var(j).name) if self._model else (lambda j: "x%d" % j)
        parts = ["%g*%s*%s" % (c, name(i), name(j)) if i != j else "%g*%s^2" % (c, name(i))
                 for (i, j), c in self.quad.items()]
        return " + ".join(parts + [repr(self.linear)])


class Var:
    """A model variable (a handle: index + model). Supports arithmetic into LinExpr."""

    __slots__ = ("model", "index")

    def __init__(self, model, index):
        self.model = model
        self.index = index

    def __hash__(self):
        return hash((id(self.model), self.index))

    def _same(self, other):
        return isinstance(other, Var) and other.model is self.model and other.index == self.index

    def _expr(self):
        return LinExpr({self.index: 1.0}, 0.0, self.model)

    def __add__(self, o):
        return self._expr() + o

    def __radd__(self, o):
        return o + self._expr()

    def __sub__(self, o):
        return self._expr() - o

    def __rsub__(self, o):
        return o - self._expr()

    def __mul__(self, k):
        return self._expr() * k

    __rmul__ = __mul__

    def __pow__(self, e):
        return self._expr() ** e

    def __truediv__(self, k):
        return self._expr() / k

    def __neg__(self):
        return -self._expr()

    def __pos__(self):
        return self._expr()

    def __le__(self, o):
        return self._expr() <= o

    def __ge__(self, o):
        return self._expr() >= o

    def __eq__(self, o):
        c = self._expr() == o
        if isinstance(c, Constraint):
            c._identity = self._same(o)  # lets `x == y` / `x in list` behave as identity in a bool context
        return c

    def __ne__(self, o):
        return not self._same(o)

    # Attributes read from / written to the library.
    @property
    def name(self):
        return _dec(lib.popso_get_var_name(self.model._h, self.index))

    def _get(self):
        lb, ub, obj, vt = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_int()
        self.model._check(lib.popso_get_var(self.model._h, self.index, ctypes.byref(lb), ctypes.byref(ub),
                                            ctypes.byref(obj), ctypes.byref(vt)))
        return lb.value, ub.value, obj.value, vt.value

    @property
    def lb(self):
        return self._get()[0]

    @lb.setter
    def lb(self, v):
        self.model._check(lib.popso_set_var_bounds(self.model._h, self.index, float(v), self.ub))

    @property
    def ub(self):
        return self._get()[1]

    @ub.setter
    def ub(self, v):
        self.model._check(lib.popso_set_var_bounds(self.model._h, self.index, self.lb, float(v)))

    @property
    def obj(self):
        return self._get()[2]

    @obj.setter
    def obj(self, v):
        self.model._check(lib.popso_set_obj_coef(self.model._h, self.index, float(v)))

    @property
    def vtype(self):
        return self._get()[3]

    @vtype.setter
    def vtype(self, t):
        self.model._check(lib.popso_set_var_type(self.model._h, self.index, _vtype(t)))

    def __repr__(self):
        return "<popso.Var %s>" % self.name


class Constraint:
    """lb <= expr <= ub, produced by comparisons of expressions; pass it to Model.add_constr."""

    __slots__ = ("expr", "lb", "ub", "_identity")

    def __init__(self, expr, lb, ub):
        self.expr = expr
        self.lb = lb
        self.ub = ub
        self._identity = None

    def __bool__(self):
        if self._identity is not None:
            return self._identity
        raise TypeError("a constraint has no truth value (chained comparisons such as 1 <= x <= 2 are not "
                        "supported; use add_constr(expr, lb=1, ub=2))")

    def __repr__(self):
        return "<popso.Constraint %g <= %r <= %g>" % (self.lb, self.expr, self.ub)


class Constr:
    """A row of a model."""

    __slots__ = ("model", "index")

    def __init__(self, model, index):
        self.model = model
        self.index = index

    @property
    def name(self):
        return _dec(lib.popso_get_row_name(self.model._h, self.index))

    def _bounds(self):
        lb, ub = ctypes.c_double(), ctypes.c_double()
        self.model._check(lib.popso_get_row(self.model._h, self.index, ctypes.byref(lb), ctypes.byref(ub)))
        return lb.value, ub.value

    @property
    def lb(self):
        return self._bounds()[0]

    @property
    def ub(self):
        return self._bounds()[1]

    def set_bounds(self, lb, ub):
        self.model._check(lib.popso_set_row_bounds(self.model._h, self.index, float(lb), float(ub)))

    def __repr__(self):
        return "<popso.Constr %s>" % self.name


def _vtype(t):
    try:
        return _VTYPES[t]
    except (KeyError, TypeError):
        raise ValueError("invalid variable type %r (use popso.CONTINUOUS, INTEGER, BINARY, SEMICONTINUOUS or "
                         "SEMIINTEGER)" % (t,)) from None


# ---------------------------------------------------------------- parameters and results

class Params:
    """Solver parameters as attributes: m.params.time_limit = 10 (see popso.PARAMS)."""

    def __init__(self, model):
        object.__setattr__(self, "_model", model)

    def __getattr__(self, name):
        if name not in PARAMS:
            raise AttributeError("unknown parameter %r (known: %s)" % (name, ", ".join(PARAMS)))
        return self._model.get_param(name)

    def __setattr__(self, name, value):
        if name not in PARAMS:
            raise AttributeError("unknown parameter %r (known: %s)" % (name, ", ".join(PARAMS)))
        self._model.set_param(name, value)

    def __repr__(self):
        return "Params(%s)" % ", ".join("%s=%r" % (n, getattr(self, n)) for n in PARAMS)


class Result:
    """Outcome of Model.solve().

    status: "OPTIMAL", "INFEASIBLE", "TIME_LIMIT", ...; objective / x are None without a solution.
    Index it with a Var, a variable name or an index: result[x], result["x"], result[0].

    duals (one per constraint) and reduced_costs (one per variable), else None: LPs and continuous
    QPs solved to optimality; for a MILP with a solution and the parameter duals=2, the duals of an
    optimal basis of the LP with the integer variables fixed at the solution (fixed_mip_duals is then
    True). If the solve stopped early, the solution's continuous part need not be optimal for that
    fixed LP: the duals then belong to the fixed LP's optimum and need not be complementary to the
    returned continuous values (the solution itself is not changed). Duals whose recovery fails a
    consistency check (their basis does not reproduce the solution's objective) or is interrupted
    are dropped (None).
    d = c + Q x - A^T y; when minimizing, a dual is >= 0 at an active lower and <= 0 at an active
    upper bound (opposite signs when maximizing). result.dual(c) and result.reduced_cost(x) look
    single values up. var_basis / constr_basis: the optimal basis of LP and fixed-MIP duals
    (popso.BASIC, AT_LOWER, AT_UPPER, AT_ZERO per variable / constraint), else None.

    objectives: the value of every objective at the solution (one entry for an ordinary model; with
    Model.add_objective the first is `objective`), None without a solution. With several objectives
    `bound` is finite only when the first objective alone has the highest priority, and no duals are
    returned.
    """

    def __init__(self, model, status, objective, bound, gap, nodes, lp_iterations, time, x, duals=None,
                 reduced_costs=None, fixed_mip_duals=False, var_basis=None, constr_basis=None, objectives=None):
        self._model = model
        self.objectives = objectives
        self.status = status
        self.objective = objective
        self.bound = bound
        self.gap = gap
        self.nodes = nodes
        self.lp_iterations = lp_iterations
        self.time = time
        self.x = x
        self.duals = duals
        self.reduced_costs = reduced_costs
        self.fixed_mip_duals = fixed_mip_duals
        self.var_basis = var_basis
        self.constr_basis = constr_basis

    @property
    def has_solution(self):
        return self.x is not None

    @property
    def optimal(self):
        return self.status == "OPTIMAL"

    def __getitem__(self, key):
        if self.x is None:
            raise ValueError("no solution (status %s)" % self.status)
        if isinstance(key, QuadExpr):
            return key.value(self)
        if isinstance(key, (LinExpr, Var)):
            return LinExpr._of(key).value(self)
        return self.x[self._model._var_index(key)]

    def value(self, key):
        return self[key]

    def dual(self, constr):
        """Dual value of a constraint (a Constr or a row index)."""
        if self.duals is None:
            raise ValueError("no dual values (LPs and continuous QPs solved to optimality, MILPs with duals=2; "
                             "status %s)" % self.status)
        return self.duals[constr.index if isinstance(constr, Constr) else int(constr)]

    def reduced_cost(self, key):
        """Reduced cost of a variable (a Var, a name or an index)."""
        if self.reduced_costs is None:
            raise ValueError("no reduced costs (LPs and continuous QPs solved to optimality, MILPs with duals=2; "
                             "status %s)" % self.status)
        return self.reduced_costs[self._model._var_index(key)]

    def as_dict(self):
        """{variable name: value} (empty without a solution)."""
        if self.x is None:
            return {}
        return {self._model.var(j).name: v for j, v in enumerate(self.x)}

    def __repr__(self):
        return "<popso.Result %s objective=%r bound=%r nodes=%d time=%.3fs>" % (
            self.status, self.objective, self.bound, self.nodes, self.time)


# ---------------------------------------------------------------- model

class Model:
    """An LP/MILP/QP/MIQP model: min/max c^T x (+ 1/2 x^T Q x) + constant s.t. lb <= A x <= ub, bounds and
    integrality."""

    def __init__(self, name=None):
        h = lib.popso_create()
        if not h:
            raise MemoryError("popso: cannot create a model")
        self._h = h
        self._callback = None      # the ctypes callback object (kept alive while set)
        self._callback_error = None  # exception raised by the incumbent callback during solve()
        self.params = Params(self)
        if name:
            self.name = name

    # -- lifetime
    def close(self):
        """Frees the native model (also done when the object is garbage collected)."""
        h, self._h = getattr(self, "_h", None), None
        if h and lib is not None:
            lib.popso_free(h)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _check(self, code):
        if code != 0:
            raise PopsoError(code, _dec(lib.popso_last_error(self._h)))

    # -- model data
    @property
    def name(self):
        return _dec(lib.popso_get_model_name(self._h))

    @name.setter
    def name(self, value):
        self._check(lib.popso_set_model_name(self._h, _enc(value)))

    @property
    def num_vars(self):
        return lib.popso_num_vars(self._h)

    @property
    def num_constrs(self):
        return lib.popso_num_rows(self._h)

    @property
    def num_nonzeros(self):
        return lib.popso_num_nonzeros(self._h)

    @property
    def num_sos(self):
        return lib.popso_num_sos(self._h)

    @property
    def num_indicators(self):
        return lib.popso_num_indicators(self._h)

    @property
    def vars(self):
        return [Var(self, j) for j in range(self.num_vars)]

    @property
    def constrs(self):
        return [Constr(self, i) for i in range(self.num_constrs)]

    def var(self, key):
        """The variable with this name or index."""
        return Var(self, self._var_index(key))

    def _var_index(self, key):
        if isinstance(key, Var):
            if key.model is not self:
                raise ValueError("variable of another model")
            return key.index
        if isinstance(key, str):
            j = lib.popso_find_var(self._h, _enc(key))
            if j < 0:
                raise KeyError(key)
            return j
        j = int(key)
        if not 0 <= j < self.num_vars:
            raise IndexError("variable index %d out of range" % j)
        return j

    def read_mps(self, filename):
        """Replaces the model by the contents of an MPS file."""
        self._check(lib.popso_read_mps(self._h, _enc(os.fspath(filename))))
        return self

    def read_lp(self, filename):
        """Replaces the model by the contents of a CPLEX LP-format file."""
        self._check(lib.popso_read_lp(self._h, _enc(os.fspath(filename))))
        return self

    def write_mps(self, filename):
        self._check(lib.popso_write_mps(self._h, _enc(os.fspath(filename))))

    def add_var(self, lb=0.0, ub=INF, obj=0.0, vtype=CONTINUOUS, name=None):
        """Adds a variable and returns it (vtype: CONTINUOUS, INTEGER, BINARY, SEMICONTINUOUS, SEMIINTEGER
        or "C"/"I"/"B"/"S"/"N"). A semi-continuous variable is 0 or in [lb, ub]."""
        idx = ctypes.c_int()
        self._check(lib.popso_add_var(self._h, float(lb), float(ub), float(obj), _vtype(vtype), _enc(name),
                                      ctypes.byref(idx)))
        return Var(self, idx.value)

    def add_vars(self, count, lb=0.0, ub=INF, obj=0.0, vtype=CONTINUOUS, names=None):
        """Adds count variables; lb/ub/obj/vtype may be scalars or sequences. Returns the list of Vars."""
        def seq(v, conv):
            if isinstance(v, (numbers.Real, str)):
                return [conv(v)] * count
            v = [conv(a) for a in v]
            if len(v) != count:
                raise ValueError("expected %d values" % count)
            return v
        first = self.num_vars
        names_arr = None
        if names is not None:
            names = [_enc(n) for n in names]
            if len(names) != count:
                raise ValueError("expected %d names" % count)
            names_arr = (ctypes.c_char_p * count)(*names)
        self._check(lib.popso_add_vars(self._h, count, _darray(seq(lb, float)), _darray(seq(ub, float)),
                                       _darray(seq(obj, float)), _iarray(seq(vtype, _vtype)), names_arr))
        return [Var(self, first + k) for k in range(count)]

    def _coeffs(self, lhs):
        """(indices, values, constant) of a Var, LinExpr, {var: coef} dict or (indices, values) pair."""
        if isinstance(lhs, (Var, LinExpr)):
            e = LinExpr._of(lhs)
            if e._model is not None and e._model is not self:
                raise ValueError("expression of another model")
            return list(e.terms.keys()), list(e.terms.values()), e.constant
        if isinstance(lhs, dict):
            idx, val = [], []
            for k, v in lhs.items():
                idx.append(self._var_index(k))
                val.append(float(v))
            return idx, val, 0.0
        if isinstance(lhs, tuple) and len(lhs) == 2:
            idx = [int(k) for k in lhs[0]]  # range-checked by the library
            val = [float(v) for v in lhs[1]]
            if len(idx) != len(val):
                raise ValueError("indices and values differ in length")
            return idx, val, 0.0
        raise TypeError("expected a constraint, expression, dict or (indices, values) pair, got %r" % (lhs,))

    def add_constr(self, lhs, lb=None, ub=None, name=None):
        """Adds a row and returns it.

        add_constr(x + 2*y <= 5), add_constr({x: 1, y: 2}, ub=5), add_constr(([0, 1], [1.0, 2.0]), lb=1, ub=5)
        or add_constr(x + y, lb=1, ub=3). Without lb/ub a side is unbounded.
        """
        if isinstance(lhs, Constraint):
            if lb is not None or ub is not None:
                raise ValueError("lb/ub cannot be combined with a comparison")
            lhs, lb, ub = lhs.expr, lhs.lb, lhs.ub
        quad = None
        if isinstance(lhs, QuadExpr):
            if lhs._model is not None and lhs._model is not self:
                raise ValueError("expression of another model")
            quad, lhs = lhs.quad, lhs.linear
        idx, val, const = self._coeffs(lhs)
        lb = -INF if lb is None else float(lb) - const
        ub = INF if ub is None else float(ub) - const
        out = ctypes.c_int()
        n = len(idx)
        self._check(lib.popso_add_row(self._h, n, _iarray(idx), _darray(val), lb, ub, _enc(name), ctypes.byref(out)))
        if quad:
            terms = [(i, j, c) for (i, j), c in quad.items() if c != 0.0]
            self._check(lib.popso_add_qrow_terms(self._h, out.value, len(terms), _iarray([t[0] for t in terms]),
                                                 _iarray([t[1] for t in terms]), _darray([t[2] for t in terms])))
        return Constr(self, out.value)

    def get_qrow_terms(self, row):
        """A row's quadratic terms as a list of (i, j, coef) meaning coef * x_i * x_j, i >= j (empty for a
        linear row); row is a Constr or a row index."""
        r = row.index if isinstance(row, Constr) else int(row)
        n = lib.popso_num_qrow_terms(self._h, r)
        if n < 0:
            raise IndexError("row %d out of range" % r)
        a, b, v = (ctypes.c_int * n)(), (ctypes.c_int * n)(), (ctypes.c_double * n)()
        self._check(lib.popso_get_qrow_terms(self._h, r, n, a, b, v))
        return [(a[k], b[k], v[k]) for k in range(n)]

    def add_sos(self, vars, weights=None, type=SOS1, name=None):
        """Adds an SOS1 (at most one nonzero) or SOS2 (at most two nonzeros, adjacent in the order of the
        weights) constraint over the variables; weights default to 1, 2, ... Returns its index."""
        idx = [self._var_index(v) for v in vars]
        w = None if weights is None else _darray([float(a) for a in weights])
        if weights is not None and len(weights) != len(idx):
            raise ValueError("one weight per variable needed")
        out = ctypes.c_int()
        self._check(lib.popso_add_sos(self._h, int(type), len(idx), _iarray(idx), w, _enc(name), ctypes.byref(out)))
        return out.value

    def add_indicator(self, binvar, value, lhs, lb=None, ub=None, name=None):
        """Adds an indicator constraint: if the binary variable binvar equals value (0 or 1), the
        constraint must hold. lhs, lb and ub as in add_constr, e.g. add_indicator(b, 1, x + y <= 5).
        Returns its index."""
        if isinstance(lhs, Constraint):
            if lb is not None or ub is not None:
                raise ValueError("lb/ub cannot be combined with a comparison")
            lhs, lb, ub = lhs.expr, lhs.lb, lhs.ub
        idx, val, const = self._coeffs(lhs)
        lb = -INF if lb is None else float(lb) - const
        ub = INF if ub is None else float(ub) - const
        out = ctypes.c_int()
        self._check(lib.popso_add_indicator(self._h, self._var_index(binvar), int(value), len(idx), _iarray(idx),
                                            _darray(val), lb, ub, _enc(name), ctypes.byref(out)))
        return out.value

    def set_objective(self, expr, sense=None):
        """Sets the objective (an expression, Var, number or {var: coef} dict) and optionally the sense.
        A QuadExpr (x * y, x ** 2, ...) gives a quadratic objective, which must be convex when
        minimizing (concave when maximizing); any other objective removes the quadratic part."""
        rows, cols, qv = [], [], []
        if isinstance(expr, QuadExpr):
            if expr._model is not None and expr._model is not self:
                raise ValueError("expression of another model")
            for (i, j), c in expr.quad.items():
                if c != 0.0:
                    # c x_i x_j = 1/2 x'Qx with Q_ij = Q_ji = c, or Q_ii = 2c.
                    rows.append(i)
                    cols.append(j)
                    qv.append(2.0 * c if i == j else c)
            expr = expr.linear
        if isinstance(expr, numbers.Real):
            idx, val, const = [], [], float(expr)
        else:
            idx, val, const = self._coeffs(expr)
        self._check(lib.popso_set_objective(self._h, len(idx), _iarray(idx), _darray(val)))
        self._check(lib.popso_set_objective_constant(self._h, const))
        self._check(lib.popso_set_q(self._h, len(qv), _iarray(rows), _iarray(cols), _darray(qv)))
        if sense is not None:
            self.sense = sense

    def set_objective_attributes(self, priority=0, weight=1.0, abs_tol=1e-6, rel_tol=0.0):
        """Priority, weight and tolerances of the model's (first) objective when further objectives are
        added with add_objective; see there."""
        self._check(lib.popso_set_objective_attributes(self._h, int(priority), float(weight), float(abs_tol),
                                                       float(rel_tol)))

    def add_objective(self, expr, priority=0, weight=1.0, abs_tol=1e-6, rel_tol=0.0, name=None):
        """Adds a further linear objective (an expression, Var, number or {var: coef} dict) and returns its
        index (1, 2, ...; the objective of set_objective is number 0).

        All objectives are optimized in the model's sense (a negative weight turns one around), in stages
        by descending priority. Objectives of one priority are blended: the stage optimizes the sum of
        weight * objective. A later stage may make an earlier stage's value worse by at most
        max(abs_tol, rel_tol * |value|), with the largest tolerances of that stage's objectives. The time
        limit covers all stages. Result.objectives holds the value of every objective."""
        if isinstance(expr, QuadExpr):
            raise ValueError("further objectives must be linear")
        if isinstance(expr, numbers.Real):
            idx, val, const = [], [], float(expr)
        else:
            idx, val, const = self._coeffs(expr)
        index = ctypes.c_int(-1)
        self._check(lib.popso_add_objective(self._h, len(idx), _iarray(idx), _darray(val), const, int(priority),
                                            float(weight), float(abs_tol), float(rel_tol), _enc(name),
                                            ctypes.byref(index)))
        return index.value

    def clear_objectives(self):
        """Removes the objectives added with add_objective (the first objective stays)."""
        self._check(lib.popso_clear_objectives(self._h))

    @property
    def num_objectives(self):
        return lib.popso_num_objectives(self._h)

    def get_q(self):
        """The objective's quadratic part as the lower triangle of Q (objective c'x + 1/2 x'Qx):
        a list of (row, column, value) with row >= column."""
        n = lib.popso_num_q_nonzeros(self._h)
        if n < 0:
            self._check(1)
        r, c, v = (ctypes.c_int * n)(), (ctypes.c_int * n)(), (ctypes.c_double * n)()
        self._check(lib.popso_get_q(self._h, n, r, c, v))
        return [(r[k], c[k], v[k]) for k in range(n)]

    @property
    def sense(self):
        return lib.popso_get_objective_sense(self._h)

    @sense.setter
    def sense(self, s):
        try:
            s = _SENSES[s]
        except (KeyError, TypeError):
            raise ValueError("invalid sense %r (use popso.MINIMIZE / MAXIMIZE or 'min' / 'max')" % (s,)) from None
        self._check(lib.popso_set_objective_sense(self._h, s))

    # -- parameters
    def set_param(self, name, value):
        if isinstance(value, bool):
            value = 1.0 if value else 0.0
        if value is None and name in ("time_limit",):
            value = INF
        if name == "lp_method" and isinstance(value, str):
            if value not in LP_METHODS:
                raise ValueError("lp_method must be one of %s" % ", ".join(LP_METHODS))
            value = LP_METHODS[value]
        self._check(lib.popso_set_param(self._h, _enc(name), float(value)))

    def get_param(self, name):
        v = ctypes.c_double()
        self._check(lib.popso_get_param(self._h, _enc(name), ctypes.byref(v)))
        if name in _BOOL_PARAMS:
            return bool(v.value)
        if name in _INT_PARAMS:
            return int(v.value)
        return v.value

    # -- starting solution
    def set_start(self, values):
        """Starting (possibly partial) solution: {var, name or index: value}, or one value per variable
        with None/NaN for "not given"."""
        if isinstance(values, dict):
            idx = [self._var_index(k) for k in values]
            val = [float(v) for v in values.values()]
            self._check(lib.popso_set_start(self._h, len(idx), _iarray(idx), _darray(val)))
        else:
            val = [math.nan if v is None else float(v) for v in values]
            self._check(lib.popso_set_start(self._h, len(val), None, _darray(val)))

    def read_start(self, filename):
        """Reads a starting solution from a "name value" file (the format write_solution produces)."""
        self._check(lib.popso_read_start(self._h, _enc(os.fspath(filename))))

    def clear_start(self):
        self._check(lib.popso_clear_start(self._h))

    # -- solving
    def solve(self, **params):
        """Solves the model and returns a Result. Keyword arguments set parameters first
        (e.g. solve(time_limit=10, threads=2)).

        Ctrl-C (KeyboardInterrupt) during the solve stops it like interrupt(): solve() returns the
        result with status "INTERRUPTED" and the best solution found so far. (Called from the main
        thread, the native solve runs on a helper thread so that the main thread can receive the
        signal.) An exception raised by the incumbent callback stops the solve and is re-raised here."""
        for k, v in params.items():
            if k not in PARAMS:
                raise TypeError("unknown parameter %r" % k)
            self.set_param(k, v)
        self._callback_error = None
        # The native solve keeps calling the callback it started with: keep its ctypes object alive
        # until the solve returns, even if the callback is replaced or removed meanwhile.
        callback = self._callback  # noqa: F841
        if threading.current_thread() is threading.main_thread():
            code = self._solve_interruptible()
        else:
            code = lib.popso_solve(self._h)
        err, self._callback_error = self._callback_error, None
        if err is not None:
            raise err
        self._check(code)
        return self.result()

    def _solve_interruptible(self):
        box = []
        done = threading.Event()

        def run():
            try:
                box.append(lib.popso_solve(self._h))
            finally:
                done.set()

        # Waiting on an Event (not Thread.join, whose state a KeyboardInterrupt can corrupt in
        # some Python versions).
        threading.Thread(target=run, name="popso-solve", daemon=True).start()
        interrupted = False
        try:
            while True:
                try:
                    while not done.wait(0.1):
                        if interrupted:
                            # Repeated: a request sent before the native solve started is cleared by it.
                            lib.popso_interrupt(self._h)
                    break
                except KeyboardInterrupt:
                    interrupted = True
                    lib.popso_interrupt(self._h)
        finally:
            # Any other exception escaping the wait: the native solve must not outlive this call
            # (the model could be freed under it), so stop it and wait for the helper thread.
            while not done.is_set():
                try:
                    lib.popso_interrupt(self._h)
                    done.wait(0.1)
                except KeyboardInterrupt:
                    pass
        return box[0]

    def interrupt(self):
        """Asks a solve() running in another thread to stop soon (status "INTERRUPTED", with the best
        solution so far). Safe to call from any thread; no effect when no solve is running."""
        self._check(lib.popso_interrupt(self._h))

    def set_incumbent_callback(self, fn):
        """Calls fn(objective, bound, x) for each new best solution during solve(): its objective, the
        best bound (both in the model's sense) and the values of all variables (a list indexed like
        Var.index). A true return value stops the solve (status "INTERRUPTED"). Calls come one at a
        time, possibly from solver threads; an exception stops the solve and is re-raised by solve().
        fn=None removes the callback. Not called for quadratic objectives."""
        if fn is None:
            self._check(lib.popso_set_incumbent_callback(self._h, INCUMBENT_CALLBACK(), None))
            self._callback = None
            return

        def trampoline(_data, objective, bound, xp, n):
            try:
                return 1 if fn(objective, bound, xp[:n]) else 0
            except BaseException as e:  # noqa: B036 - re-raised by solve()
                if self._callback_error is None:
                    self._callback_error = e
                return 1

        cb = INCUMBENT_CALLBACK(trampoline)
        self._check(lib.popso_set_incumbent_callback(self._h, cb, None))
        self._callback = cb

    def result(self):
        """The result of the last solve."""
        h = self._h
        status = lib.popso_get_status(h)
        if status == 0:
            raise PopsoError(5, "the model has not been solved since it was last changed")
        d = ctypes.c_double()
        ll = ctypes.c_longlong()

        def get_d(f):
            self._check(f(h, ctypes.byref(d)))
            return d.value

        def get_ll(f):
            self._check(f(h, ctypes.byref(ll)))
            return ll.value

        x = objective = objectives = None
        if lib.popso_has_solution(h):
            objective = get_d(lib.popso_get_objective)
            n = self.num_vars
            buf = (ctypes.c_double * n)()
            self._check(lib.popso_get_solution(h, buf, n))
            x = list(buf)
            objectives = []
            for k in range(self.num_objectives):
                self._check(lib.popso_get_objective_value(h, k, ctypes.byref(d)))
                objectives.append(d.value)
        duals = reduced_costs = var_basis = constr_basis = None
        has_duals = lib.popso_has_duals(h)
        if has_duals:
            rows, n = self.num_constrs, self.num_vars
            ybuf, dbuf = (ctypes.c_double * max(rows, 1))(), (ctypes.c_double * max(n, 1))()
            self._check(lib.popso_get_duals(h, ybuf, rows))
            self._check(lib.popso_get_reduced_costs(h, dbuf, n))
            duals, reduced_costs = list(ybuf)[:rows], list(dbuf)[:n]
            if lib.popso_has_basis(h):
                vb, rb = (ctypes.c_int * max(n, 1))(), (ctypes.c_int * max(rows, 1))()
                self._check(lib.popso_get_basis(h, vb, rb))
                var_basis, constr_basis = list(vb)[:n], list(rb)[:rows]
        return Result(self, _dec(lib.popso_status_string(status)), objective, get_d(lib.popso_get_bound),
                      get_d(lib.popso_get_gap), get_ll(lib.popso_get_node_count),
                      get_ll(lib.popso_get_lp_iterations), get_d(lib.popso_get_solve_time), x, duals,
                      reduced_costs, has_duals == 2, var_basis, constr_basis, objectives)

    def write_solution(self, filename):
        self._check(lib.popso_write_solution(self._h, _enc(os.fspath(filename))))

    def __repr__(self):
        return "<popso.Model %s: %d vars, %d constrs>" % (self.name or "(unnamed)", self.num_vars, self.num_constrs)


def read_mps(filename):
    """Reads an MPS file into a new Model."""
    return Model().read_mps(filename)


def read_lp(filename):
    """Reads a CPLEX LP-format file into a new Model."""
    return Model().read_lp(filename)
