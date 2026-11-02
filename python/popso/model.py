"""Pythonic model API over the popso C library."""

import ctypes
import math
import numbers
import os

from ._lib import lib

INF = float("inf")

CONTINUOUS = 0
INTEGER = 1
BINARY = 2

MINIMIZE = 1
MAXIMIZE = -1

_VTYPES = {
    CONTINUOUS: CONTINUOUS, INTEGER: INTEGER, BINARY: BINARY,
    "C": CONTINUOUS, "I": INTEGER, "B": BINARY,
    "continuous": CONTINUOUS, "integer": INTEGER, "binary": BINARY,
}
_SENSES = {MINIMIZE: MINIMIZE, MAXIMIZE: MAXIMIZE, "min": MINIMIZE, "max": MAXIMIZE,
           "minimize": MINIMIZE, "maximize": MAXIMIZE}

PARAMS = ("time_limit", "threads", "mip_gap", "mip_abs_gap", "seed", "deterministic", "verbosity",
          "mem_limit", "node_limit", "presolve", "racing")
_INT_PARAMS = {"threads", "seed", "verbosity", "mem_limit", "node_limit", "racing"}
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
        if not isinstance(k, numbers.Real):
            return NotImplemented  # only linear expressions
        k = float(k)
        return LinExpr({j: c * k for j, c in self.terms.items()}, self.constant * k, self._model)

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
        raise ValueError("invalid variable type %r (use popso.CONTINUOUS, INTEGER or BINARY)" % (t,)) from None


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
    """

    def __init__(self, model, status, objective, bound, gap, nodes, lp_iterations, time, x):
        self._model = model
        self.status = status
        self.objective = objective
        self.bound = bound
        self.gap = gap
        self.nodes = nodes
        self.lp_iterations = lp_iterations
        self.time = time
        self.x = x

    @property
    def has_solution(self):
        return self.x is not None

    @property
    def optimal(self):
        return self.status == "OPTIMAL"

    def __getitem__(self, key):
        if self.x is None:
            raise ValueError("no solution (status %s)" % self.status)
        if isinstance(key, (LinExpr, Var)):
            return LinExpr._of(key).value(self)
        return self.x[self._model._var_index(key)]

    def value(self, key):
        return self[key]

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
    """An LP/MILP model: min/max c^T x + constant s.t. lb <= A x <= ub, bounds and integrality."""

    def __init__(self, name=None):
        h = lib.popso_create()
        if not h:
            raise MemoryError("popso: cannot create a model")
        self._h = h
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

    def write_mps(self, filename):
        self._check(lib.popso_write_mps(self._h, _enc(os.fspath(filename))))

    def add_var(self, lb=0.0, ub=INF, obj=0.0, vtype=CONTINUOUS, name=None):
        """Adds a variable and returns it (vtype: CONTINUOUS, INTEGER, BINARY or "C"/"I"/"B")."""
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
        idx, val, const = self._coeffs(lhs)
        lb = -INF if lb is None else float(lb) - const
        ub = INF if ub is None else float(ub) - const
        out = ctypes.c_int()
        n = len(idx)
        self._check(lib.popso_add_row(self._h, n, _iarray(idx), _darray(val), lb, ub, _enc(name), ctypes.byref(out)))
        return Constr(self, out.value)

    def set_objective(self, expr, sense=None):
        """Sets the objective (an expression, Var, number or {var: coef} dict) and optionally the sense."""
        if isinstance(expr, numbers.Real):
            idx, val, const = [], [], float(expr)
        else:
            idx, val, const = self._coeffs(expr)
        self._check(lib.popso_set_objective(self._h, len(idx), _iarray(idx), _darray(val)))
        self._check(lib.popso_set_objective_constant(self._h, const))
        if sense is not None:
            self.sense = sense

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
        (e.g. solve(time_limit=10, threads=2))."""
        for k, v in params.items():
            if k not in PARAMS:
                raise TypeError("unknown parameter %r" % k)
            self.set_param(k, v)
        self._check(lib.popso_solve(self._h))
        return self.result()

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

        x = objective = None
        if lib.popso_has_solution(h):
            objective = get_d(lib.popso_get_objective)
            n = self.num_vars
            buf = (ctypes.c_double * n)()
            self._check(lib.popso_get_solution(h, buf, n))
            x = list(buf)
        return Result(self, _dec(lib.popso_status_string(status)), objective, get_d(lib.popso_get_bound),
                      get_d(lib.popso_get_gap), get_ll(lib.popso_get_node_count),
                      get_ll(lib.popso_get_lp_iterations), get_d(lib.popso_get_solve_time), x)

    def write_solution(self, filename):
        self._check(lib.popso_write_solution(self._h, _enc(os.fspath(filename))))

    def __repr__(self):
        return "<popso.Model %s: %d vars, %d constrs>" % (self.name or "(unnamed)", self.num_vars, self.num_constrs)


def read_mps(filename):
    """Reads an MPS file into a new Model."""
    return Model().read_mps(filename)
