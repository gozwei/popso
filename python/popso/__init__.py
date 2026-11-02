"""popso: Python bindings (ctypes) for the popso LP/MILP solver.

    import popso
    m = popso.Model("example")
    x = m.add_var(ub=10, vtype=popso.INTEGER, name="x")
    y = m.add_var(name="y")
    m.add_constr(x + 2 * y <= 14)
    m.add_constr(3 * x - y >= 0)
    m.set_objective(3 * x + 4 * y, sense=popso.MAXIMIZE)
    r = m.solve(time_limit=10)
    print(r.status, r.objective, r[x], r["y"])

The shared library is searched in $POPSO_LIBRARY (a file or a directory), next to this
package, in the build/ directory of a source checkout, and on the system library path.
"""

from ._lib import path as library_path
from .model import (
    BINARY,
    CONTINUOUS,
    INF,
    INTEGER,
    MAXIMIZE,
    MINIMIZE,
    PARAMS,
    Constr,
    Constraint,
    LinExpr,
    Model,
    Params,
    PopsoError,
    Result,
    Var,
    read_mps,
    version,
)

__version__ = version()

__all__ = [
    "BINARY", "CONTINUOUS", "INF", "INTEGER", "MAXIMIZE", "MINIMIZE", "PARAMS", "Constr", "Constraint",
    "LinExpr", "Model", "Params", "PopsoError", "Result", "Var", "library_path", "read_mps", "version",
]
