#!/usr/bin/env python3
"""Presolve gap vs SCIP and HiGHS: size of the presolved model, its LP bound, presolve time.

For each instance:
  - SCIP (pyscipopt): Model.presolve() (timed); rows/cols/#int from the transformed problem; nnz and LP bound from
    the transformed problem written as MPS (it may contain aggregated columns, so its nnz is an upper bound);
  - HiGHS (highspy): Highs.presolve() (timed), the presolved model (getPresolvedLp) and its LP bound;
  - ours: `milp` with MILP_PRESOLVE_ONLY=1 MILP_PRESOLVE_DUMP=<tmp.mps> (size and time from our log), LP bound of
    the dump;
  - all LP bounds are solved by HiGHS; the LP bound of the original model and the known optimum
    (benchmarks/reference) give the root gap: "closed" = (LP_presolved - LP_orig) / (opt - LP_orig) in percent
    (minimization sense). "best" is the reference (SCIP or HiGHS) with the larger closed gap.

Usage:
  .venv/bin/python benchmarks/tools/presolvegap.py --set miplib_wide_core [--solver build/milp] [--out f.json]
  .venv/bin/python benchmarks/tools/presolvegap.py benchmarks/instances/cflp_medium_1.mps ...
  .venv/bin/python benchmarks/tools/presolvegap.py --set miplib_wide_core --ours-only f.json --out g.json
  .venv/bin/python benchmarks/tools/presolvegap.py --table f.json [--table-b g.json]   # markdown table only

One solver process at a time; the references run single-threaded with a time limit per presolve.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]


def known_optima() -> dict:
    out = {}
    for f in ("miplib_solu.json", "solutions.json"):
        p = BENCH / "reference" / f
        if not p.exists():
            continue
        for n, e in json.loads(p.read_text()).get("instances", {}).items():
            z = e.get("objective")
            if e.get("status") == "optimal" and z is not None and n not in out:
                out[n] = float(z)
    return out


def highs():
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    return h


def lp_stats(lp) -> dict:
    integ = list(lp.integrality_) if len(lp.integrality_) else []
    ints = sum(1 for t in integ if int(t) != 0)
    bins = sum(1 for j, t in enumerate(integ) if int(t) != 0 and lp.col_lower_[j] > -0.5 and lp.col_upper_[j] < 1.5)
    return {"rows": lp.num_row_, "cols": lp.num_col_, "nnz": len(lp.a_matrix_.value_), "ints": ints, "bins": bins}


def lp_bound_of(h, tl: float) -> float | None:
    """LP relaxation bound of the model loaded in h, in minimization sense (None if not solved)."""
    import highspy

    sign = -1.0 if h.getLp().sense_ == highspy.ObjSense.kMaximize else 1.0
    h.setOptionValue("solve_relaxation", True)
    h.setOptionValue("time_limit", tl)
    h.run()
    if h.getModelStatus() == highspy.HighsModelStatus.kOptimal:
        return sign * float(h.getInfo().objective_function_value)
    return None


def original_side(path: str, tl: float) -> dict:
    import highspy

    h = highs()
    h.readModel(path)
    lp = h.getLp()
    sense = -1 if lp.sense_ == highspy.ObjSense.kMaximize else 1
    return {"sense": sense, "orig": lp_stats(lp), "lp_orig": lp_bound_of(h, tl)}  # lp_orig: minimization sense


def highs_side(path: str, tl: float) -> dict:
    h = highs()
    h.readModel(path)
    h.setOptionValue("time_limit", tl)
    t0 = time.perf_counter()
    h.presolve()
    res = {"time": time.perf_counter() - t0}
    p = h.getPresolvedLp()
    res["pre"] = lp_stats(p)
    g = highs()
    g.passModel(p)
    res["lp"] = lp_bound_of(g, tl)
    return res


def scip_side(path: str, tl: float, mem: int) -> dict:
    import pyscipopt

    m = pyscipopt.Model()
    m.hideOutput()
    m.setParam("limits/time", tl)
    m.setParam("limits/memory", mem)
    m.readProblem(path)
    t0 = time.perf_counter()
    m.presolve()
    res = {"time": time.perf_counter() - t0}
    res["pre"] = {"rows": m.getNConss(True), "cols": m.getNVars(True), "ints": m.getNIntVars() + m.getNBinVars(),
                  "bins": m.getNBinVars()}
    with tempfile.TemporaryDirectory(prefix="pgap_") as d:
        dump = os.path.join(d, "scip.mps")
        m.writeProblem(dump, trans=True, verbose=False)
        h = highs()
        h.readModel(dump)
        res["pre"]["nnz"] = len(h.getLp().a_matrix_.value_)
        res["lp"] = lp_bound_of(h, tl)
    return res


def ours_side(path: str, solver: list[str], tl: float, mem: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="pgap_") as d:
        dump = os.path.join(d, "pre.mps")
        env = dict(os.environ, MILP_PRESOLVE_ONLY="1", MILP_PRESOLVE_DUMP=dump)
        cp = subprocess.run(solver + [path, "--verbosity", "2", "--mem-limit", str(mem), "--threads", "1"],
                            env=env, capture_output=True, text=True, timeout=1800)
        out = cp.stdout + cp.stderr
        res = {"details": [l for l in out.splitlines() if l.startswith(("Presolve", "Clique"))]}
        mt = re.search(r"Presolve: \d+ rows, \d+ cols, \d+ nnz -> (\d+) rows, (\d+) cols, (\d+) nnz \(([\d.]+)s\)", out)
        res["time"] = float(mt.group(4)) if mt else None
        if not os.path.exists(dump):
            res["status"] = "infeasible" if "infeasible" in out.lower() else "error"
            res["log_tail"] = out[-2000:]
            return res
        h = highs()
        h.readModel(dump)
        res["pre"] = lp_stats(h.getLp())
        if mt:  # exact sizes from our log (the MPS round trip may add empty columns)
            res["pre"].update(rows=int(mt.group(1)), cols=int(mt.group(2)), nnz=int(mt.group(3)))
        res["lp"] = lp_bound_of(h, tl)  # our reduced model is a minimization
        res["status"] = "ok"
        return res


def closed(lp: float | None, lp0: float | None, opt: float | None) -> float | None:
    """Percent of the root gap closed by lp over lp0 (minimization sense)."""
    if lp is None or lp0 is None or opt is None:
        return None
    g = opt - lp0
    if abs(g) <= 1e-9 * max(1.0, abs(opt)):
        return None
    return 100.0 * (lp - lp0) / g


def name_of(path: str) -> str:
    name = Path(path).name
    for ext in (".gz", ".mps"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name


def run_one(path: str, args, solver: list[str], opts: dict, prev: dict | None) -> dict:
    name = name_of(path)
    if prev:
        rec = dict(prev)
    else:
        o = original_side(path, args.lp_time)
        s = o["sense"]
        lp0 = o["lp_orig"]
        opt = opts.get(name)
        opt = None if opt is None else s * opt
        rec = {"name": name, "file": path, "sense": s, "orig": o["orig"], "lp_orig": lp0, "opt": opt}
        for who, fn in (("scip", lambda: scip_side(path, args.ref_time, args.mem_limit)),
                        ("highs", lambda: highs_side(path, args.ref_time))):
            try:
                r = fn()
            except Exception as e:  # keep going
                print(f"{name}: {who} failed: {e}", file=sys.stderr)
                rec[who] = {"pre": None, "time": None, "lp": None, "closed": None}
                continue
            lp = r.get("lp")
            rec[who] = {"pre": r["pre"], "time": r["time"], "lp": lp, "closed": closed(lp, lp0, opt)}
    o = ours_side(path, solver, args.lp_time, args.mem_limit)
    lp = o.get("lp")
    rec["ours"] = {"pre": o.get("pre"), "time": o.get("time"), "lp": lp, "closed": closed(lp, rec["lp_orig"], rec["opt"]),
                   "status": o.get("status"), "details": o.get("details")}
    return rec


def best_ref(r: dict) -> str:
    cs = [(r[w]["closed"], w) for w in ("scip", "highs") if r.get(w) and r[w].get("closed") is not None]
    return max(cs)[1] if cs else "scip"


def fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def table(recs: list[dict], recs_b: list[dict] | None = None) -> str:
    """Markdown table: SCIP, HiGHS, ours (A) and optionally ours (B)."""
    by_b = {r["name"]: r for r in (recs_b or [])}
    sides = ["scip", "highs", "ours"] + (["B"] if recs_b else [])
    hdr = "S / H / ours" + (" A / B" if recs_b else "")
    lines = [f"| instance | rows {hdr} | cols {hdr} | nnz {hdr} | closed% {hdr} | time (s) {hdr} |",
             "|---|---|---|---|---|---|"]

    def get(r, w):
        return by_b.get(r["name"], {}).get("ours", {}) if w == "B" else (r.get(w) or {})

    for r in recs:
        cells = []
        for key in ("rows", "cols", "nnz"):
            cells.append(" / ".join(str((get(r, w).get("pre") or {}).get(key, "-")) for w in sides))
        cells.append(" / ".join(fmt(get(r, w).get("closed")) for w in sides))
        cells.append(" / ".join(fmt(get(r, w).get("time"), 2) for w in sides))
        lines.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summary(recs: list[dict], who: str, label: str) -> str:
    """Geometric means of size ratios (presolved / original) and time; mean closed gap."""
    def gm(xs):
        xs = [x for x in xs if x is not None and x > 0]
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

    side = [(r, r.get(who) or {}) for r in recs]
    side = [(r, s) for r, s in side if s.get("pre")]
    rows = [(s["pre"]["rows"] + 1) / (r["orig"]["rows"] + 1) for r, s in side]
    cols = [(s["pre"]["cols"] + 1) / (r["orig"]["cols"] + 1) for r, s in side]
    nnz = [(s["pre"]["nnz"] + 1) / (r["orig"]["nnz"] + 1) for r, s in side]
    cl = [s["closed"] for r, s in side if s.get("closed") is not None]
    tm = [s["time"] + 0.01 for r, s in side if s.get("time") is not None]
    return (f"{label}: rows {gm(rows):.3f}, cols {gm(cols):.3f}, nnz {gm(nnz):.3f} (geomean of presolved/orig), "
            f"closed gap mean {sum(cl) / max(1, len(cl)):.1f}% over {len(cl)}, "
            f"time geomean(+0.01) {gm(tm) - 0.01:.3f}s, total {sum(tm):.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", nargs="*")
    ap.add_argument("--set", help="name in benchmarks/sets/ or a file")
    ap.add_argument("--solver", default="build/milp")
    ap.add_argument("--lp-time", type=float, default=300.0)
    ap.add_argument("--ref-time", type=float, default=120.0, help="time limit of a reference presolve")
    ap.add_argument("--mem-limit", type=int, default=2048)
    ap.add_argument("--out", help="JSON output (records)")
    ap.add_argument("--table", help="print the table of a JSON file and exit")
    ap.add_argument("--table-b", help="with --table: second run (ours B) for a before/after table")
    ap.add_argument("--ours-only", help="reuse the reference side of this JSON file, only rerun ours")
    args = ap.parse_args()
    if args.table:
        recs = json.loads(Path(args.table).read_text())
        recs_b = json.loads(Path(args.table_b).read_text()) if args.table_b else None
        print(table(recs, recs_b))
        print()
        print(summary(recs, "scip", "SCIP"))
        print(summary(recs, "highs", "HiGHS"))
        print(summary(recs, "ours", "ours A" if recs_b else "ours"))
        if recs_b:
            print(summary(recs_b, "ours", "ours B"))
        return
    files = list(args.instances)
    if args.set:
        sp = Path(args.set)
        if not sp.exists():
            sp = BENCH / "sets" / (args.set + ".txt")
        files += [l.strip() for l in sp.read_text().splitlines() if l.strip() and not l.startswith("#")]
    solver = args.solver.split()
    opts = known_optima()
    prev = {}
    if args.ours_only:
        prev = {r["name"]: r for r in json.loads(Path(args.ours_only).read_text())}
    recs = []
    for f in files:
        try:
            r = run_one(f, args, solver, opts, prev.get(name_of(f)))
        except Exception as e:  # keep going
            print(f"{f}: ERROR {e}", file=sys.stderr)
            continue
        recs.append(r)
        parts = []
        for w in ("scip", "highs", "ours"):
            s = r.get(w) or {}
            p = s.get("pre") or {}
            parts.append(f"{w} {p.get('rows')}x{p.get('cols')} nnz {p.get('nnz')} closed {fmt(s.get('closed'))}% "
                         f"{fmt(s.get('time'), 2)}s")
        print(f"{r['name']:22s} " + " | ".join(parts), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(recs, indent=1))
    print()
    for w, label in (("scip", "SCIP"), ("highs", "HiGHS"), ("ours", "ours")):
        print(summary(recs, w, label))


if __name__ == "__main__":
    main()
