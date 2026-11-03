#!/usr/bin/env python3
"""Compute reference results for the benchmark instances.

Solves every instance with HiGHS and SCIP, cross-checks that they agree
(and with the hand-derived answers of the lpsanity cases and the MIPLIB
.solu values), verifies every returned solution with our independent
Python MPS reader (lib/mps.py), and writes benchmarks/reference/solutions.json.

    .venv/bin/python benchmarks/reference.py                      # everything
    .venv/bin/python benchmarks/reference.py --jobs 6 --time-limit 600
    .venv/bin/python benchmarks/reference.py 'benchmarks/instances/tsp_*' --update
    .venv/bin/python benchmarks/reference.py --recompute          # no solving

Each (instance, solver) pair runs in its own subprocess with a hard timeout,
single-threaded by default.  With --jobs > 1 several solves run in parallel,
so reported times are slightly noisier than on an idle machine.  Each solver
process is memory-capped (--mem-limit MB, default 2048): RLIMIT_AS/DATA in
the worker where enforced, SCIP limits/memory, and an RSS watchdog in the
parent that kills the process (status "memlimit").
The machine is shared: keep --jobs <= 3.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

from lib import mps  # noqa: E402
from lib.solfile import INF_OR_UNBD, INFEASIBLE, OPTIMAL, TIMELIMIT, UNBOUNDED  # noqa: E402

DEFAULT_OUT = BENCH / "reference" / "solutions.json"
OBJ_RTOL = 1e-6


def rel_diff(a, b):
    return abs(a - b) / max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------- worker

def worker(solver, path, time_limit, threads, mem_mb):
    from lib import refsolvers
    from lib.proc import limit_self

    limit_self(mem_mb)
    try:
        res = refsolvers.SOLVERS[solver](path, time_limit, threads, want_solution=True,
                                         mem_mb=mem_mb)
    except MemoryError:
        print(json.dumps({"status": "memlimit", "objective": None, "bound": None, "time": None}))
        return
    sol = res.pop("solution", None)
    if sol is not None:
        model = mps.read_mps(path)
        v = mps.verify(model, sol)
        res["verify"] = {
            "feasible": v.feasible,
            "objective": v.objective,
            "max_viol": max(v.max_bound_viol, v.max_int_viol, v.max_row_viol, v.max_logic_viol),
            "worst": v.worst,
        }
    print(json.dumps(res))


def run_task(solver, path, time_limit, threads, mem_mb):
    import tempfile

    from lib.proc import run_capped

    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", solver, str(path),
           "--time-limit", str(time_limit), "--threads", str(threads), "--mem-limit", str(mem_mb)]
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        rc, reason, wall, peak = run_capped(cmd, time_limit * 1.5 + 60, mem_mb * 1.1 if mem_mb else None,
                                            stdout=out, stderr=err)
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    if reason == "timeout":
        return {"status": TIMELIMIT, "objective": None, "bound": None, "time": wall,
                "error": "hard timeout (killed)"}
    if reason == "memlimit":
        return {"status": "memlimit", "objective": None, "bound": None, "time": wall,
                "error": f"killed: RSS above {mem_mb} MB"}
    lines = [ln for ln in stdout.splitlines() if ln.startswith("{")]
    if rc != 0 or not lines:
        if "MemoryError" in stderr or "bad_alloc" in stderr:
            return {"status": "memlimit", "objective": None, "bound": None, "time": wall,
                    "error": "out of memory"}
        return {"status": "error", "objective": None, "bound": None, "time": wall,
                "error": (stderr or stdout)[-2000:]}
    r = json.loads(lines[-1])
    r["peak_rss_mb"] = round(peak, 1)
    return r


# ------------------------------------------------------------- consensus

def consensus(name, results, sense=1, expected=None):
    """Combine per-solver results. Returns (consensus dict, agree, notes)."""
    notes = []
    agree = True
    ok = {s: r for s, r in results.items()
          if r and "skipped" not in r and r["status"] not in ("error", "memlimit")}
    for s, r in results.items():
        if r and r.get("status") == "error":
            notes.append(f"{s}: error: {r.get('error', '')[:200]}")
    # verification of returned solutions
    for s, r in ok.items():
        v = r.get("verify")
        if v is None:
            continue
        if not v["feasible"]:
            agree = False
            notes.append(f"{s}: returned solution FAILS verification: {v['worst']}")
        if r["objective"] is not None and rel_diff(v["objective"], r["objective"]) > OBJ_RTOL:
            agree = False
            notes.append(f"{s}: reported objective {r['objective']} != recomputed {v['objective']}")
    # a solver whose solution fails verification does not get a vote
    ok = {s: r for s, r in ok.items() if (r.get("verify") or {}).get("feasible", True)}
    statuses = {s: r["status"] for s, r in ok.items()}
    opt = {s: r["objective"] for s, r in ok.items() if r["status"] == OPTIMAL}
    final = {"status": "unknown", "objective": None, "bound": None}
    if opt:
        vals = list(opt.values())
        if any(v is None for v in vals) or max(vals) - min(vals) > OBJ_RTOL * max(1.0, *map(abs, vals)):
            agree = False
            notes.append(f"optimal objectives disagree: {opt}")
        final["status"] = OPTIMAL
        # prefer highs > scip for the reported value
        for s in ("highs", "scip"):
            if s in opt and opt[s] is not None:
                final["objective"] = final["bound"] = opt[s]
                break
        for s, st in statuses.items():
            if st in (INFEASIBLE, UNBOUNDED, INF_OR_UNBD):
                agree = False
                notes.append(f"{s} says {st} but others found optimum")
    else:
        sts = set(statuses.values())
        decided = sts & {INFEASIBLE, UNBOUNDED}
        if len(decided) > 1:
            agree = False
            notes.append(f"status disagreement: {statuses}")
        if decided:
            final["status"] = decided.pop() if len(decided) == 1 else "unknown"
        elif INF_OR_UNBD in sts:
            final["status"] = INF_OR_UNBD
        elif sts == {TIMELIMIT} or TIMELIMIT in sts:
            final["status"] = TIMELIMIT
            prim = [r["objective"] for r in ok.values() if r["objective"] is not None]
            bnds = [r["bound"] for r in ok.values() if r["bound"] is not None]
            if prim:
                final["objective"] = min(prim) if sense > 0 else max(prim)
            if bnds:
                final["bound"] = max(bnds) if sense > 0 else min(bnds)
    # timelimit results must be consistent with the optimum
    if final["status"] == OPTIMAL and final["objective"] is not None:
        z = final["objective"]
        for s, r in ok.items():
            if r["status"] != TIMELIMIT:
                continue
            if r["objective"] is not None and sense * (z - r["objective"]) > OBJ_RTOL * max(1, abs(z)):
                agree = False
                notes.append(f"{s} found incumbent {r['objective']} better than optimum {z}")
            if r["bound"] is not None and sense * (r["bound"] - z) > OBJ_RTOL * max(1, abs(z)):
                agree = False
                notes.append(f"{s} bound {r['bound']} beyond optimum {z}")
    if expected is not None:
        est, eobj = expected
        if final["status"] != est or (eobj is not None and (
                final["objective"] is None or rel_diff(final["objective"], eobj) > OBJ_RTOL)):
            agree = False
            notes.append(f"hand-derived expectation {expected} not met (got {final['status']}, "
                         f"{final['objective']})")
    return final, agree, notes


# ------------------------------------------------------------------ main

def judge(n, res, sense, expected, hand_only, solu):
    """Consensus for one instance from its per-solver results `res`, plus the
    hand-derived expectations (lpsanity) and the MIPLIB .solu cross-check."""
    if n in hand_only:
        est, eobj = expected[n]
        final = {"status": est, "objective": eobj, "bound": eobj}
        agree = True
        notes = ["MPS semantics differ between readers (negative UP bound); reference is "
                 "the hand-derived answer, solver results informational"]
    else:
        final, agree, notes = consensus(n, res, sense, expected.get(n))
    if n in solu and solu[n]["status"] in (OPTIMAL, INFEASIBLE):
        # MIPLIB instance: cross-check against the official .solu value
        z, zs = solu[n].get("objective"), solu[n]["status"]
        for s_, r in res.items():
            if "skipped" in r or r.get("status") not in (OPTIMAL, INFEASIBLE, TIMELIMIT):
                continue
            if r["status"] in (OPTIMAL, INFEASIBLE) and r["status"] != zs:
                agree = False
                notes.append(f"{s_}: status {r['status']} but MIPLIB solu says {zs}")
            elif zs == OPTIMAL and r["status"] == OPTIMAL and rel_diff(r["objective"], z) > OBJ_RTOL:
                agree = False
                notes.append(f"{s_}: optimum {r['objective']} != MIPLIB solu {z}")
            elif zs == OPTIMAL and r.get("objective") is not None and \
                    sense * (z - r["objective"]) > OBJ_RTOL * max(1, abs(z)):
                agree = False
                notes.append(f"{s_}: incumbent {r['objective']} better than MIPLIB optimum {z}")
        if final["status"] not in (OPTIMAL, INFEASIBLE):
            notes.append(f"no solver proved the result; MIPLIB solu: {zs} {z}")
    return final, agree, notes


def load_solu():
    solu_path = BENCH / "reference" / "miplib_solu.json"
    if solu_path.exists():
        return json.loads(solu_path.read_text())["instances"]
    return {}


def recompute(out_path, names):
    """Re-derive status/objective/bound/agree/notes of the stored entries from
    their stored per-solver results (no solving)."""
    from generators import lp_sanity
    data = json.loads(out_path.read_text())
    expected = lp_sanity.expected_by_instance()
    hand_only = lp_sanity.hand_derived_only_instances()
    solu = load_solu()
    changed = 0
    for n, e in data["instances"].items():
        if names and n not in names:
            continue
        sense = -1 if e.get("sense") == "max" else 1
        final, agree, notes = judge(n, e["solvers"], sense, expected, hand_only, solu)
        new = {**final, "agree": agree, "notes": notes}
        old = {k: e.get(k) for k in new}
        if old != new:
            changed += 1
            print(f"  {n}: {old['status']} {old['objective']} -> {new['status']} {new['objective']}"
                  + ("" if agree else "  (DISAGREE)"))
        e.update(new)
    out_path.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")
    print(f"recomputed {out_path}: {changed} entr{'y' if changed == 1 else 'ies'} changed")
    return 0



def expand(patterns):
    files = []
    for p in patterns:
        pp = Path(p)
        if pp.is_dir():
            files += sorted(pp.glob("*.mps")) + sorted(pp.glob("*.mps.gz"))
        else:
            files += [Path(f) for f in sorted(glob.glob(p))]
    seen, out = set(), []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            out.append(f)
    return out


def repo_rel(path) -> str:
    """Path relative to the repository root (never an absolute local path)."""
    p = Path(os.path.abspath(path))
    for cand in (p, p.resolve()):
        if cand.is_relative_to(BENCH.parent):
            return cand.relative_to(BENCH.parent).as_posix()
    return os.path.relpath(p, BENCH.parent)


def inst_name(path: Path):
    n = path.name
    for ext in (".mps.gz", ".mps"):
        if n.endswith(ext):
            return n[: -len(ext)]
    return path.stem


def versions():
    v = {"python": platform.python_version(), "platform": platform.platform(),
         "machine": platform.machine()}
    try:
        import highspy
        h = highspy.Highs()
        v["highs"] = h.version() if hasattr(h, "version") else "?"
    except Exception:  # noqa: BLE001
        pass
    try:
        import pyscipopt
        v["scip"] = str(pyscipopt.Model().version())
        v["pyscipopt"] = pyscipopt.__version__
    except Exception:  # noqa: BLE001
        pass
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", nargs="*", help="files, globs or directories")
    ap.add_argument("--solvers", default="highs,scip")
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=2,
                    help="concurrent solver processes (shared machine: keep <= 3)")
    ap.add_argument("--mem-limit", type=int, default=2048,
                    help="memory cap per solver process in MB (0 = none)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--update", action="store_true",
                    help="merge into an existing output file instead of replacing it")
    ap.add_argument("--recompute", action="store_true",
                    help="no solving: re-derive the consensus of the entries in --out (all, or "
                         "the instances named by the positional arguments) from their stored "
                         "per-solver results")
    ap.add_argument("--worker", nargs=2, metavar=("SOLVER", "PATH"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.worker:
        worker(args.worker[0], args.worker[1], args.time_limit, args.threads, args.mem_limit)
        return 0
    if args.recompute:
        return recompute(Path(args.out), {inst_name(Path(p)) for p in args.instances})

    from generators import lp_sanity
    expected = lp_sanity.expected_by_instance()
    hand_only = lp_sanity.hand_derived_only_instances()

    files = expand(args.instances or [str(BENCH / "instances")])
    if not files:
        print("no instances found", file=sys.stderr)
        return 1
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    from lib.refsolvers import available_solvers
    avail = available_solvers()
    missing = [s for s in solvers if s not in avail]
    if missing:
        print(f"warning: solvers not available, skipping: {missing}", file=sys.stderr)
        solvers = [s for s in solvers if s in avail]

    out_path = Path(args.out)
    data = {"meta": {}, "instances": {}}
    if args.update and out_path.exists():
        data = json.loads(out_path.read_text())

    stats = {}
    for f in files:
        m = mps.read_mps(f)
        stats[inst_name(f)] = (m.stats(), m.sense, f)

    tasks = [(inst_name(f), s, f) for f in files for s in solvers]
    results = {n: {} for n in stats}
    print(f"{len(files)} instances x {solvers} with {args.jobs} jobs, "
          f"time limit {args.time_limit:g}s", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_task, s, f, args.time_limit, args.threads, args.mem_limit): (n, s)
                for n, s, f in tasks}
        for fut in as_completed(futs):
            n, s = futs[fut]
            r = fut.result()
            results[n][s] = r
            if "skipped" in r:
                continue
            obj = r.get("objective")
            print(f"  {n:<40} {s:<7} {r['status']:<11} "
                  f"obj={obj if obj is not None else '-':<22} t={r['time']:.2f}s", flush=True)

    solu = load_solu()
    run_info = {"date": dt.date.today().isoformat(), "time_limit": args.time_limit,
                "threads": args.threads, "jobs": args.jobs, "mem_limit_mb": args.mem_limit}
    for n, (st, sense, f) in stats.items():
        res = results[n]
        old = data["instances"].get(n) if args.update else None
        if old:
            # merge: keep the other solvers' earlier results (e.g. SCIP run after HiGHS)
            res = {**old.get("solvers", {}), **res}
        final, agree, notes = judge(n, res, sense, expected, hand_only, solu)
        entry = {
            "file": repo_rel(f),
            **st,
            "status": final["status"],
            "objective": final["objective"],
            "bound": final["bound"],
            "agree": agree,
            "notes": notes,
            "solvers": {s: r for s, r in res.items() if "skipped" not in r},
        }
        if args.update or old:
            runs = list((old or {}).get("ref_runs", []))
            runs.append({**run_info, "solvers": [s_ for s_ in solvers if s_ in results[n]]})
            entry["ref_runs"] = runs
        data["instances"][n] = entry

    if args.update and data.get("meta", {}).get("generated"):
        # keep the meta of the full run; partial runs are recorded per entry (ref_runs)
        data["meta"].setdefault("updates", []).append(
            {**run_info, "solvers": solvers, "instances": len(stats), "versions": versions()})
    else:
        data["meta"] = {
            "generated": dt.datetime.now().isoformat(timespec="seconds"),
            "time_limit": args.time_limit,
            "threads": args.threads,
            "jobs": args.jobs,
            "mem_limit_mb": args.mem_limit,
            "rel_gap": 1e-9,
            "versions": versions(),
        }
    data["instances"] = dict(sorted(data["instances"].items()))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")

    bad = {n: e for n, e in data["instances"].items() if n in stats and not e["agree"]}
    print(f"\nwrote {out_path}")
    print(f"status counts: " + ", ".join(
        f"{k}={v}" for k, v in sorted(_count(e['status'] for n, e in data['instances'].items()
                                             if n in stats).items())))
    if bad:
        print(f"\n!!! {len(bad)} instance(s) with solver disagreement / verification problems:")
        for n, e in bad.items():
            for note in e["notes"]:
                print(f"  {n}: {note}")
        return 2
    return 0


def _count(it):
    c = {}
    for x in it:
        c[x] = c.get(x, 0) + 1
    return c


if __name__ == "__main__":
    sys.exit(main())
