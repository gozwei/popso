"""Tiny pure-Python linear modelling layer with an MPS writer.

Just enough to build the random benchmark families (generators/families.py)
without any solver library:

    m = Model("knap")
    x = [m.add_var(vtype="B", obj=p[j], name=f"x{j}") for j in range(n)]
    m.add_constr(quicksum(w[j] * x[j] for j in range(n)) <= C, "cap")
    m.sense = "max"
    m.write("knap.mps")

Variables have bounds (lb, ub; +-inf allowed), an objective coefficient and
a type: "C" continuous, "I" integer, "B" binary (integer with bounds [0, 1]).
Linear expressions support +, -, scalar * and quicksum(); comparing an
expression with <=, >= or == gives a constraint for Model.add_constr.  All
terms are moved to the left-hand side and constants to the right-hand side;
duplicate variables in one constraint are merged.

The MPS writer emits fixed-column-style free MPS: integer columns inside
MARKER INTORG/INTEND blocks, every integer column with explicit bounds
(BV for binaries, LI for an integer with an infinite upper bound), so the
file reads the same under all common MPS conventions.

The LP writer (Model.to_lp, or Model.write("x.lp"); lp_text() for models
from other sources) emits the CPLEX LP format: every variable is listed in
the objective in column order (with coefficient 0 if needed), so a reader
numbers the columns exactly as in the MPS file; bounds are explicit wherever
they differ from the LP defaults ([0, +inf), binaries [0, 1]).
"""

from __future__ import annotations

import math

INF = math.inf


class Var:
    __slots__ = ("index", "name", "lb", "ub", "obj", "vtype")
    __array_ufunc__ = None  # make numpy scalars defer to our operators

    def __init__(self, index, name, lb, ub, obj, vtype):
        self.index = index
        self.name = name
        self.lb = lb
        self.ub = ub
        self.obj = obj
        self.vtype = vtype

    def _expr(self):
        return LinExpr({self: 1.0})

    __hash__ = object.__hash__

    def __add__(self, o):
        return self._expr() + o

    __radd__ = __add__

    def __sub__(self, o):
        return self._expr() - o

    def __rsub__(self, o):
        return o + (-1.0) * self._expr()

    def __mul__(self, c):
        return self._expr() * c

    __rmul__ = __mul__

    def __neg__(self):
        return self._expr() * -1.0

    def __le__(self, o):
        return self._expr() <= o

    def __ge__(self, o):
        return self._expr() >= o

    def __eq__(self, o):
        return self._expr() == o

    def __repr__(self):
        return f"Var({self.name})"


class LinExpr:
    __slots__ = ("terms", "const")
    __array_ufunc__ = None

    def __init__(self, terms=None, const=0.0):
        self.terms = dict(terms) if terms else {}  # Var -> coefficient
        self.const = float(const)

    @staticmethod
    def _of(o):
        if isinstance(o, LinExpr):
            return o
        if isinstance(o, Var):
            return o._expr()
        return LinExpr(const=float(o))

    def __add__(self, o):
        o = LinExpr._of(o)
        t = dict(self.terms)
        for v, c in o.terms.items():
            t[v] = t.get(v, 0.0) + c
        return LinExpr(t, self.const + o.const)

    __radd__ = __add__

    def __sub__(self, o):
        return self + LinExpr._of(o) * -1.0

    def __rsub__(self, o):
        return LinExpr._of(o) + self * -1.0

    def __mul__(self, c):
        if isinstance(c, (Var, LinExpr)):
            raise TypeError("only linear expressions are supported")
        c = float(c)
        return LinExpr({v: c * a for v, a in self.terms.items()}, c * self.const)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __le__(self, o):
        return Constraint(self - o, "L")

    def __ge__(self, o):
        return Constraint(self - o, "G")

    def __eq__(self, o):
        return Constraint(self - o, "E")

    __hash__ = None


def quicksum(items):
    t, const = {}, 0.0
    for it in items:
        it = LinExpr._of(it)
        for v, c in it.terms.items():
            t[v] = t.get(v, 0.0) + c
        const += it.const
    return LinExpr(t, const)


def lin(coefs, vars_):
    """sum_j coefs[j] * vars_[j] (coefficients converted to float)."""
    return LinExpr(_merge(zip(vars_, map(float, coefs))))


def _merge(pairs):
    t = {}
    for v, c in pairs:
        t[v] = t.get(v, 0.0) + c
    return t


class Constraint:
    """expr (sense) 0, normalized to  terms (sense) rhs."""

    __slots__ = ("terms", "sense", "rhs", "name", "index")

    def __init__(self, expr, sense):
        self.terms = expr.terms
        self.sense = sense
        self.rhs = -expr.const
        self.name = None
        self.index = None

    def __bool__(self):  # guard against chained comparisons
        raise TypeError("constraints have no truth value")


class Model:
    def __init__(self, name=""):
        self.name = name
        self.sense = "min"  # or "max"
        self.vars: list[Var] = []
        self.constrs: list[Constraint] = []

    def add_var(self, lb=0.0, ub=INF, obj=0.0, vtype="C", name=None):
        vtype = str(vtype)
        if vtype not in ("C", "I", "B"):
            raise ValueError(f"bad vtype {vtype!r}")
        if vtype == "B":
            lb, ub = max(0.0, float(lb)), min(1.0, float(ub))
        v = Var(len(self.vars), name if name is not None else f"C{len(self.vars)}",
                float(lb), float(ub), float(obj), vtype)
        self.vars.append(v)
        return v

    def add_constr(self, con, name=None):
        if not isinstance(con, Constraint):
            raise TypeError("add_constr expects expr <= / >= / == rhs")
        con.index = len(self.constrs)
        con.name = name if name is not None else f"R{con.index}"
        self.constrs.append(con)
        return con

    # ---- statistics
    @property
    def num_vars(self):
        return len(self.vars)

    @property
    def num_constrs(self):
        return len(self.constrs)

    @property
    def num_nonzeros(self):
        return sum(1 for c in self.constrs for a in c.terms.values() if a != 0.0)

    @property
    def num_int_vars(self):
        return sum(1 for v in self.vars if v.vtype != "C")

    # ---- MPS / LP output
    def write(self, path):
        """Write MPS, or CPLEX LP if the file name ends in .lp."""
        text = self.to_lp() if str(path).lower().endswith(".lp") else self.to_mps()
        with open(path, "w") as f:
            f.write(text)

    def to_lp(self):
        cols = [(v.name, v.obj, v.lb, v.ub, v.vtype) for v in self.vars]
        rows = []
        for c in self.constrs:
            lo, hi = {"L": (-INF, c.rhs), "G": (c.rhs, INF), "E": (c.rhs, c.rhs)}[c.sense]
            rows.append((c.name, lo, hi, [(v.name, a) for v, a in c.terms.items()]))
        return lp_text(self.name, self.sense, 0.0, cols, rows)

    def to_mps(self):
        out = [f"NAME {self.name}"]
        if self.sense == "max":
            out.append("OBJSENSE MAX")
        elif self.sense != "min":
            raise ValueError(f"bad sense {self.sense!r}")
        out.append("ROWS")
        out.append(" N  OBJ")
        for c in self.constrs:
            out.append(f" {c.sense}  {c.name:<8}")
        # column-wise matrix (entries in row order)
        cols = [[] for _ in self.vars]
        for c in self.constrs:
            for v, a in c.terms.items():
                if a != 0.0:
                    cols[v.index].append((c.index, a))
        out.append("COLUMNS")
        in_int = False
        for v in self.vars:
            is_int = v.vtype != "C"
            if is_int != in_int:
                out.append("    MARKER    'MARKER'                 "
                           + ("'INTORG'" if is_int else "'INTEND'"))
                in_int = is_int
            entries = sorted(cols[v.index])
            if v.obj != 0.0:
                out.append(f"    {v.name:<8}  {'OBJ':<8}  {_num(v.obj)}")
            elif not entries:
                out.append(f"    {v.name:<8}  {'OBJ':<8}  0")
            for i, a in entries:
                out.append(f"    {v.name:<8}  {self.constrs[i].name:<8}  {_num(a)}")
        if in_int:
            out.append("    MARKER    'MARKER'                 'INTEND'")
        out.append("RHS")
        for c in self.constrs:
            if c.rhs != 0.0:
                out.append(f"    RHS1      {c.name:<8}  {_num(c.rhs)}")
        out.append("BOUNDS")
        for v in self.vars:
            out.extend(_bounds(v))
        out.append("ENDATA")
        return "\n".join(out) + "\n"


def _bound(kind, name, val=None):
    s = f" {kind} BND1      {name:<8}"
    return s if val is None else f"{s}  {_num(val)}"


def _bounds(v):
    lb, ub, n = v.lb, v.ub, v.name
    if v.vtype == "B" and lb == 0.0 and ub == 1.0:
        return [_bound("BV", n)]
    if lb == ub:
        return [_bound("FX", n, lb)]
    out = []
    if v.vtype != "C" and ub == INF:
        # integer with no upper bound: LI keeps ub = +inf under every reader
        out.append(_bound("LI", n, lb) if lb != -INF else _bound("MI", n))
        if lb == -INF:
            out.append(_bound("PL", n))
        return out
    if lb == -INF and ub == INF:
        return [_bound("FR", n)]
    if lb == -INF:
        out.append(_bound("MI", n))
    elif lb != 0.0:
        out.append(_bound("LO", n, lb))
    if ub != INF:
        out.append(_bound("UP", n, ub))
    return out


def _num(x):
    """Shortest exact decimal up to 15 significant digits, else %.16e."""
    x = float(x)
    if x == 0.0:
        return "0"
    s = f"{x:.15g}"
    if float(s) == x:
        return s
    return f"{x:.16e}"


# ---- CPLEX LP output

_LP_NAME_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&()/,;?@_`'{}|~")
_LP_NAME_CHARS = _LP_NAME_START | set("0123456789.")
# Words a reader may take for a keyword (at the start of a line) or a number.
_LP_RESERVED = {
    "min", "minimize", "minimise", "minimum", "max", "maximize", "maximise", "maximum",
    "st", "s.t.", "st.", "s.t", "subject", "such", "bound", "bounds", "general", "generals", "gen",
    "integer", "integers", "binary", "binaries", "bin", "semi", "semis", "semicontinuous", "sos",
    "sos1", "sos2", "end", "free", "inf", "infinity", "lazy", "user", "pwlobj", "pwl",
}


def _lp_name(name):
    if (not name or name[0] not in _LP_NAME_START or any(ch not in _LP_NAME_CHARS for ch in name)
            or name.lower() in _LP_RESERVED):
        raise ValueError(f"name {name!r} cannot be written in LP format")
    return name


def _lp_num(x):
    if x == INF:
        return "+inf"
    if x == -INF:
        return "-inf"
    return _num(x)


def _lp_terms(terms):
    """[(name, coef)] -> ["+ 3 x", "- y", ...] (coefficients written exactly)."""
    out = []
    for name, a in terms:
        a = float(a)
        sign = "-" if a < 0 or (a == 0.0 and math.copysign(1.0, a) < 0) else "+"
        mag = abs(a)
        out.append(f"{sign} {name}" if mag == 1.0 else f"{sign} {_num(mag)} {name}")
    return out


def _lp_wrap(head, parts, width=100):
    """Join parts after head, continuing on new lines (expressions may span lines)."""
    lines, cur = [], head
    for p in parts:
        if len(cur) + 1 + len(p) > width and cur.strip():
            lines.append(cur)
            cur = "   " + p
        else:
            cur = f"{cur} {p}" if cur else p
    lines.append(cur)
    return lines


def lp_text(name, sense, obj_const, cols, rows):
    """CPLEX LP text.

    cols: [(name, obj, lb, ub, vtype)] with vtype "C", "I" or "B";
    rows: [(name, lo, hi, [(col name, coef)])] with lo/hi possibly infinite.
    """
    if sense not in ("min", "max"):
        raise ValueError(f"bad sense {sense!r}")
    out = []
    if name:
        out.append(f"\\Problem name: {name}")
    out.append("Maximize" if sense == "max" else "Minimize")
    obj = _lp_terms((_lp_name(c[0]), c[1]) for c in cols)
    if obj_const:
        obj.append(("- " if obj_const < 0 else "+ ") + _num(abs(obj_const)))
    out.extend(_lp_wrap(" obj:", obj))
    out.append("Subject To")
    first_col = _lp_name(cols[0][0]) if cols else None
    for rname, lo, hi, terms in rows:
        lhs = _lp_terms((_lp_name(c), a) for c, a in terms)
        if not lhs and first_col is not None:
            lhs = [f"+ 0 {first_col}"]  # empty row
        head = f" {_lp_name(rname)}:"
        if lo == hi:
            out.extend(_lp_wrap(head, lhs + ["=", _lp_num(hi)]))
        elif lo == -INF:
            out.extend(_lp_wrap(head, lhs + ["<=", _lp_num(hi)]))
        elif hi == INF:
            out.extend(_lp_wrap(head, lhs + [">=", _lp_num(lo)]))
        else:  # ranged
            out.extend(_lp_wrap(head, [_lp_num(lo), "<="] + lhs + ["<=", _lp_num(hi)]))
    out.append("Bounds")
    for cname, _, lb, ub, vtype in cols:
        n = _lp_name(cname)
        if vtype == "B" and lb == 0.0 and ub == 1.0:
            continue
        if lb == ub:
            out.append(f" {n} = {_lp_num(lb)}")
        elif lb == -INF and ub == INF:
            out.append(f" {n} free")
        elif lb == 0.0 and ub == INF:
            continue
        elif lb == 0.0 and ub >= 0.0:
            out.append(f" {n} <= {_lp_num(ub)}")
        elif ub == INF:
            out.append(f" {n} >= {_lp_num(lb)}")
        else:
            out.append(f" {_lp_num(lb)} <= {n} <= {_lp_num(ub)}")
    for title, kind in (("Generals", "I"), ("Binaries", "B")):
        names = [_lp_name(c[0]) for c in cols if c[4] == kind]
        if names:
            out.append(title)
            out.extend(_lp_wrap("", names))
    out.append("End")
    return "\n".join(out) + "\n"
