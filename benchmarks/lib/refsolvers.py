"""Thin wrappers around the reference solvers (HiGHS, SCIP).

Each `solve_<name>(path, time_limit, threads, want_solution)` returns a dict:
  status      one of lib.solfile statuses (optimal/infeasible/unbounded/
              inf_or_unbd/timelimit/unknown)
  objective   primal objective (None if no solution)
  bound       dual bound (None if unavailable)
  time        wall-clock seconds of the solve call (excl. reading)
  solution    {name: value} if want_solution and a solution exists
"""

from __future__ import annotations

import math
import os
import time

from lib.solfile import INF_OR_UNBD, INFEASIBLE, OPTIMAL, TIMELIMIT, UNBOUNDED, UNKNOWN

# Reference solves aim at proven optimality: tight gaps. POPSO_REF_GAP overrides both (a comparison run under the
# settings popso is run with, e.g. 1e-6 for spatial branch-and-bound, where 1e-9 is out of reach for any solver).
REL_GAP = float(os.environ.get("POPSO_REF_GAP", 1e-9))
ABS_GAP = float(os.environ.get("POPSO_REF_GAP", 1e-9))


def _num(v):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and abs(v) < 1e29 else None


def has_quadratic_objective(path):
    """True if an MPS file has a QUADOBJ / QMATRIX / QSECTION / QCMATRIX section (an LP file a '['): a quadratic
    objective or quadratic constraints."""
    import gzip

    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    lp = path.lower().endswith(".lp")
    with opener(path, "rb") as f:
        for line in f:
            if lp:
                if b"[" in line:
                    return True
            elif line[:1] in (b"Q", b"q") and line.split()[0].upper() in (b"QUADOBJ", b"QMATRIX", b"QSECTION", b"QCMATRIX"):
                return True
    return False


def solve_highs(path, time_limit, threads=1, want_solution=False, mem_mb=None):
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", REL_GAP)
    h.setOptionValue("mip_abs_gap", ABS_GAP)
    h.setOptionValue("threads", int(threads))
    st = h.readModel(str(path))
    if st == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS failed to read model")
    model = h.getModel()
    if model.hessian_.dim_ > 0 and any(t != highspy.HighsVarType.kContinuous for t in (model.lp_.integrality_ or [])):
        return _highs_miqp(h, model, time_limit, want_solution)
    t0 = time.perf_counter()
    h.run()
    elapsed = time.perf_counter() - t0
    ms = h.getModelStatus()
    MS = highspy.HighsModelStatus
    if ms == MS.kUnboundedOrInfeasible:
        # disambiguate without presolve
        h.setOptionValue("presolve", "off")
        h.clearSolver()
        t1 = time.perf_counter()
        h.run()
        elapsed += time.perf_counter() - t1
        ms = h.getModelStatus()
    status = {
        MS.kOptimal: OPTIMAL,
        MS.kInfeasible: INFEASIBLE,
        MS.kUnbounded: UNBOUNDED,
        MS.kUnboundedOrInfeasible: INF_OR_UNBD,
        MS.kTimeLimit: TIMELIMIT,
    }.get(ms, UNKNOWN)
    info = h.getInfo()
    has_sol = info.primal_solution_status == 2  # kSolutionStatusFeasible
    lp = h.getLp()
    is_mip = any(t != highspy.HighsVarType.kContinuous for t in (lp.integrality_ or []))
    obj = _num(info.objective_function_value) if has_sol else None
    if status in (INFEASIBLE, UNBOUNDED, INF_OR_UNBD):
        obj = None
    if is_mip:
        bound = _num(info.mip_dual_bound)
    else:
        bound = obj if status == OPTIMAL else None
    res = {"status": status, "objective": obj, "bound": bound, "time": elapsed,
           "raw_status": h.modelStatusToString(ms)}
    if is_mip:
        res["nodes"] = int(info.mip_node_count)
    if want_solution and has_sol and obj is not None:
        sol = h.getSolution()
        names = [h.getColName(j)[1] for j in range(lp.num_col_)]
        res["solution"] = dict(zip(names, sol.col_value))
    return res


def _highs_miqp(h, model, time_limit, want_solution):
    """HiGHS has no MIQP solver: a plain branch-and-bound in Python on HiGHS's QP solver (depth-first,
    most fractional branching, pruning with a relative 1e-9 tolerance). Small models only; an
    independent second opinion next to SCIP, whose nonlinear handling of the objective is slow on
    some MIQPs."""
    import highspy

    lp = model.lp_
    n = lp.num_col_
    ints = [j for j in range(n) if lp.integrality_[j] != highspy.HighsVarType.kContinuous]
    lp.integrality_ = [highspy.HighsVarType.kContinuous] * n
    sense = 1 if lp.sense_ == highspy.ObjSense.kMinimize else -1
    t0 = time.perf_counter()
    best, best_x, nodes, complete, failed = math.inf, None, 0, True, False
    stack = [(list(lp.col_lower_), list(lp.col_upper_))]
    while stack:
        if time.perf_counter() - t0 > time_limit:
            complete = False
            break
        lb, ub = stack.pop()
        nodes += 1
        lp.col_lower_, lp.col_upper_ = lb, ub
        h.passModel(model)
        h.run()
        ms = h.getModelStatus()
        if ms == highspy.HighsModelStatus.kInfeasible:
            continue
        if ms != highspy.HighsModelStatus.kOptimal:
            failed = True  # unbounded or unsolved relaxation: no reference
            break
        obj = sense * h.getInfo().objective_function_value
        if obj >= best - REL_GAP * max(1.0, abs(best)):
            continue
        x = list(h.getSolution().col_value)
        frac = [(abs(x[j] - round(x[j])), j) for j in ints if abs(x[j] - round(x[j])) > 1e-7]
        if not frac:
            best, best_x = obj, x
            continue
        _, j = max(frac)
        lo, hi = (list(lb), list(ub)), (list(lb), list(ub))
        lo[1][j] = math.floor(x[j])
        hi[0][j] = math.ceil(x[j])
        stack += [hi, lo] if x[j] - math.floor(x[j]) < 0.5 else [lo, hi]
    elapsed = time.perf_counter() - t0
    if failed:
        status = UNKNOWN
    elif complete:
        status = OPTIMAL if best_x is not None else INFEASIBLE
    else:
        status = TIMELIMIT
    obj = sense * best if best_x is not None and status != INFEASIBLE else None
    res = {"status": status, "objective": obj, "bound": obj if status == OPTIMAL else None, "time": elapsed,
           "raw_status": "python-bnb", "nodes": nodes}
    if want_solution and best_x is not None:
        res["solution"] = {h.getColName(j)[1]: best_x[j] for j in range(n)}
    return res


def solve_scip(path, time_limit, threads=1, want_solution=False, mem_mb=None):
    import pyscipopt

    m = pyscipopt.Model()
    m.hideOutput()
    m.readProblem(str(path))
    m.setParam("limits/time", float(time_limit))
    m.setParam("limits/gap", REL_GAP)
    m.setParam("limits/absgap", ABS_GAP)
    m.setParam("randomization/randomseedshift", 0)
    if has_quadratic_objective(path):
        # The quadratic objective becomes a nonlinear constraint on an auxiliary variable: with the
        # default feasibility tolerance 1e-6 SCIP's "optimum" undercuts the true one by ~1e-7. Quadratic
        # constraints likewise, and much more at a cone's apex (x^2 <= t^2 at t = 0 lets x reach 1e-3).
        # POPSO_REF_SCIP_FEASTOL overrides it ("default": SCIP's own 1e-6): on nonconvex models 1e-9 made SCIP
        # report 4 QPLIB instances infeasible whose published points are feasible to 1e-9.
        ft = os.environ.get("POPSO_REF_SCIP_FEASTOL", "1e-9")
        if ft != "default":
            m.setParam("numerics/feastol", float(ft))
    if mem_mb:
        m.setParam("limits/memory", float(mem_mb) * 0.8)
    t0 = time.perf_counter()
    m.optimize()
    elapsed = time.perf_counter() - t0
    raw = m.getStatus()
    status = {
        "optimal": OPTIMAL,
        "gaplimit": OPTIMAL,
        "infeasible": INFEASIBLE,
        "unbounded": UNBOUNDED,
        "inforunbd": INF_OR_UNBD,
        "timelimit": TIMELIMIT,
        "memlimit": "memlimit",
    }.get(raw, UNKNOWN)
    obj = None
    if m.getNSols() > 0 and status not in (INFEASIBLE, UNBOUNDED, INF_OR_UNBD):
        obj = _num(m.getObjVal())
    bound = None
    if status in (OPTIMAL, TIMELIMIT):
        bound = _num(m.getDualbound())
    res = {"status": status, "objective": obj, "bound": bound, "time": elapsed,
           "raw_status": raw, "nodes": int(m.getNNodes())}
    if want_solution and obj is not None:
        best = m.getBestSol()
        res["solution"] = {v.name: m.getSolVal(best, v) for v in m.getVars(transformed=False)}
    return res


SOLVERS = {"highs": solve_highs, "scip": solve_scip}


def available_solvers():
    out = []
    try:
        import highspy  # noqa: F401
        out.append("highs")
    except ImportError:
        pass
    try:
        import pyscipopt  # noqa: F401
        out.append("scip")
    except ImportError:
        pass
    return out
