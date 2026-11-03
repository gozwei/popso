"""Locating and loading the popso shared library (ctypes prototypes of include/popso.h)."""

import ctypes
import ctypes.util
import os
import sys

if sys.platform == "win32":
    _LIB_NAMES = ["popso.dll", "libpopso.dll"]
elif sys.platform == "darwin":
    _LIB_NAMES = ["libpopso.dylib"]
else:
    _LIB_NAMES = ["libpopso.so"]

_HERE = os.path.dirname(os.path.abspath(__file__))


def _candidates():
    """Places to look for the library, in order."""
    env = os.environ.get("POPSO_LIBRARY")
    if env:
        if os.path.isdir(env):
            for name in _LIB_NAMES:
                yield os.path.join(env, name)
        else:
            yield env
        return  # an explicit choice is not second-guessed
    for name in _LIB_NAMES:
        yield os.path.join(_HERE, name)
    # A source checkout: python/popso/ -> <repo>/build[/Release]
    repo = os.path.dirname(os.path.dirname(_HERE))
    for sub in ("build", os.path.join("build", "Release")):
        for name in _LIB_NAMES:
            yield os.path.join(repo, sub, name)
    found = ctypes.util.find_library("popso")
    if found:
        yield found


def _load():
    tried = []
    for cand in _candidates():
        if os.path.isabs(cand) and not os.path.exists(cand):
            tried.append(cand)
            continue
        try:
            return ctypes.CDLL(cand), cand
        except OSError as e:
            tried.append("%s (%s)" % (cand, e))
    raise ImportError(
        "popso: cannot find the popso shared library. Build it with CMake (it is built by default as "
        "build/%s), then set POPSO_LIBRARY to its path or copy it into %s. Tried: %s"
        % (_LIB_NAMES[0], _HERE, "; ".join(tried) or "nothing")
    )


lib, path = _load()

_c_int = ctypes.c_int
_c_double = ctypes.c_double
_c_ll = ctypes.c_longlong
_c_char_p = ctypes.c_char_p
_vp = ctypes.c_void_p
_pd = ctypes.POINTER(ctypes.c_double)
_pi = ctypes.POINTER(ctypes.c_int)
_pll = ctypes.POINTER(ctypes.c_longlong)
_pcp = ctypes.POINTER(ctypes.c_char_p)

_PROTOTYPES = {
    "popso_version": (_c_char_p, []),
    "popso_status_string": (_c_char_p, [_c_int]),
    "popso_last_error": (_c_char_p, [_vp]),
    "popso_create": (_vp, []),
    "popso_free": (None, [_vp]),
    "popso_read_mps": (_c_int, [_vp, _c_char_p]),
    "popso_write_mps": (_c_int, [_vp, _c_char_p]),
    "popso_read_lp": (_c_int, [_vp, _c_char_p]),
    "popso_add_sos": (_c_int, [_vp, _c_int, _c_int, _pi, _pd, _c_char_p, _pi]),
    "popso_add_indicator": (_c_int, [_vp, _c_int, _c_int, _c_int, _pi, _pd, _c_double, _c_double, _c_char_p, _pi]),
    "popso_num_sos": (_c_int, [_vp]),
    "popso_num_indicators": (_c_int, [_vp]),
    "popso_set_model_name": (_c_int, [_vp, _c_char_p]),
    "popso_get_model_name": (_c_char_p, [_vp]),
    "popso_add_var": (_c_int, [_vp, _c_double, _c_double, _c_double, _c_int, _c_char_p, _pi]),
    "popso_add_vars": (_c_int, [_vp, _c_int, _pd, _pd, _pd, _pi, _pcp]),
    "popso_add_row": (_c_int, [_vp, _c_int, _pi, _pd, _c_double, _c_double, _c_char_p, _pi]),
    "popso_add_rows": (_c_int, [_vp, _c_int, _pi, _pi, _pd, _pd, _pd, _pcp]),
    "popso_set_var_bounds": (_c_int, [_vp, _c_int, _c_double, _c_double]),
    "popso_set_var_type": (_c_int, [_vp, _c_int, _c_int]),
    "popso_set_row_bounds": (_c_int, [_vp, _c_int, _c_double, _c_double]),
    "popso_set_objective_sense": (_c_int, [_vp, _c_int]),
    "popso_set_objective_constant": (_c_int, [_vp, _c_double]),
    "popso_set_obj_coef": (_c_int, [_vp, _c_int, _c_double]),
    "popso_set_objective": (_c_int, [_vp, _c_int, _pi, _pd]),
    "popso_num_vars": (_c_int, [_vp]),
    "popso_num_rows": (_c_int, [_vp]),
    "popso_num_nonzeros": (_c_ll, [_vp]),
    "popso_get_objective_sense": (_c_int, [_vp]),
    "popso_get_objective_constant": (_c_double, [_vp]),
    "popso_get_var": (_c_int, [_vp, _c_int, _pd, _pd, _pd, _pi]),
    "popso_get_row": (_c_int, [_vp, _c_int, _pd, _pd]),
    "popso_get_var_name": (_c_char_p, [_vp, _c_int]),
    "popso_get_row_name": (_c_char_p, [_vp, _c_int]),
    "popso_find_var": (_c_int, [_vp, _c_char_p]),
    "popso_set_param": (_c_int, [_vp, _c_char_p, _c_double]),
    "popso_get_param": (_c_int, [_vp, _c_char_p, _pd]),
    "popso_set_start": (_c_int, [_vp, _c_int, _pi, _pd]),
    "popso_read_start": (_c_int, [_vp, _c_char_p]),
    "popso_clear_start": (_c_int, [_vp]),
    "popso_solve": (_c_int, [_vp]),
    "popso_get_status": (_c_int, [_vp]),
    "popso_has_solution": (_c_int, [_vp]),
    "popso_get_objective": (_c_int, [_vp, _pd]),
    "popso_get_bound": (_c_int, [_vp, _pd]),
    "popso_get_gap": (_c_int, [_vp, _pd]),
    "popso_get_node_count": (_c_int, [_vp, _pll]),
    "popso_get_lp_iterations": (_c_int, [_vp, _pll]),
    "popso_get_solve_time": (_c_int, [_vp, _pd]),
    "popso_get_solution": (_c_int, [_vp, _pd, _c_int]),
    "popso_write_solution": (_c_int, [_vp, _c_char_p]),
}

for _name, (_res, _args) in _PROTOTYPES.items():
    _f = getattr(lib, _name)
    _f.restype = _res
    _f.argtypes = _args
