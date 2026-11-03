#!/usr/bin/env python3
"""Root gap closed: ours vs the open-source reference solvers.

For each instance, runs the reference solvers (SCIP via pyscipopt with a node limit of 1 by default;
HiGHS via highspy with mip_max_nodes = 1 on request) and our solver (--node-limit 1), and reports
the root LP bound, the bound after the root node (cuts, root heuristics, reduced-cost fixing,
restarts), the time to finish the root and the share of the root gap closed relative to the known
optimum (benchmarks/reference/*.json):

    closed% = (root bound - LP0) / (optimum - LP0)

with LP0 the LP relaxation of the ORIGINAL model (HiGHS): a common baseline for solvers that
presolve differently; "own%" uses our presolved root LP instead. "best ref" is the reference with
the largest closed%. Reference results are cached (--ref-cache): they do not change between our builds.

    .venv/bin/python benchmarks/tools/rootgap.py --set benchmarks/sets/miplib_wide_core.txt \\
        --solver build/popso --out /tmp/rootgap.json [--refs scip,highs --threads 4 -t 120 --mem-limit 2048]
    .venv/bin/python benchmarks/tools/rootgap.py --report /tmp/rootgap.json        # table only
    .venv/bin/python benchmarks/tools/rootgap.py --compare before.json after.json  # ours, two builds

Our side parses the "ROOTSTAT" line (search/branch_and_bound.cpp, one per root incl. restarts: the
first gives the presolved root LP, the last the final root bound, root node LP and time) and the
"cut rows by separator" line of the root cut loop (verbosity 2).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))

from run import load_references  # noqa: E402
from lib.proc import run_capped  # noqa: E402

FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf"


def read_set(path):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.append(line)
    return out


def inst_name(p):
    n = Path(p).name
    for suf in (".gz", ".mps", ".lp"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n


# ----------------------------------------------------------------------------- references


def lp0_highs(mps, tl, threads):
    """LP relaxation of the original model (HiGHS): the common baseline."""
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(tl))
    h.setOptionValue("solve_relaxation", True)
    h.setOptionValue("threads", int(threads))  # HiGHS keeps one thread pool per process: same value everywhere
    h.readModel(str(mps))
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None
    return h.getInfo().objective_function_value


def scip_root(mps, threads, tl, mem_mb):
    """SCIP (pyscipopt, single thread), node limit 1: root LP, root bound, applied cuts per separator."""
    import pyscipopt

    res = {}
    m = pyscipopt.Model()
    m.hideOutput()
    m.readProblem(str(mps))
    m.setParam("limits/nodes", 1)
    m.setParam("limits/time", float(tl))
    m.setParam("limits/memory", float(mem_mb))
    t0 = time.perf_counter()
    m.optimize()
    res["time"] = time.perf_counter() - t0
    res["status"] = m.getStatus()
    res["bound"] = m.getDualbound()
    res["incumbent"] = m.getPrimalbound() if m.getNSols() > 0 else None
    res["nodes"] = m.getNNodes()
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "stats.txt")
        m.writeStatistics(f)
        st = Path(f).read_text()
    mm = re.search(r"First LP value\s*:\s*(" + FLOAT + ")", st)
    res["root_lp"] = float(mm.group(1)) if mm else None
    mm = re.search(r"Presolved Problem\s*:.*?Variables\s*:\s*(\d+).*?Constraints\s*:\s*(\d+)", st, re.S)
    if mm:
        res["presolved"] = [int(mm.group(2)), int(mm.group(1))]
    # Applied cuts per separator (sub-separators such as aggregation > cmir replace their parent).
    cuts = {}
    mm = re.search(r"^Separators\s*:.*?\n((?:  .*\n)+)", st, re.M)
    if mm:
        parent = None
        for line in mm.group(1).splitlines():
            name, _, rest = line.partition(":")
            toks = rest.split()
            sub = name.strip().startswith(">")
            name = name.strip().lstrip(">").strip()
            if name == "cut pool" or len(toks) < 10:
                continue
            try:
                applied = int(toks[9])
            except ValueError:
                continue
            if sub:
                cuts.pop(parent, None)
            else:
                parent = name
            cuts[name] = applied
    res["cuts"] = {k: v for k, v in cuts.items() if v > 0}
    res["root_solved"] = res["status"] in ("optimal", "infeasible")
    m.freeProb()
    return res


def highs_root(mps, threads, tl, mem_mb):
    """HiGHS (highspy), mip_max_nodes = 1: root bound only (no root LP or cut statistics)."""
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(tl))
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("mip_max_nodes", 1)
    h.readModel(str(mps))
    t0 = time.perf_counter()
    h.run()
    ms = h.getModelStatus()
    return {"time": time.perf_counter() - t0, "status": h.modelStatusToString(ms),
            "bound": h.getInfo().mip_dual_bound, "root_lp": None, "cuts": {},
            "root_solved": ms in (highspy.HighsModelStatus.kOptimal, highspy.HighsModelStatus.kInfeasible)}


REFS = {"scip": scip_root, "highs": highs_root}


# ----------------------------------------------------------------------------- ours

ROOTSTAT = re.compile(
    r"ROOTSTAT bound=(\S+) lp=(\S+) nodelp=(\S+) rounds=(\d+) cuts=(\d+) rows=(\d+) time=(\S+)")


def ours_root(solver, mps, threads, tl, mem_mb, extra_env=None):
    cmd = shlex.split(solver) + [str(mps), "--node-limit", "1", "--threads", str(threads), "--time-limit",
                                 str(tl), "--mem-limit", str(mem_mb), "--verbosity", "2"]
    with tempfile.TemporaryFile("w+") as out:
        # run_capped has no env argument: set it for the child via os.environ temporarily.
        saved = dict(os.environ)
        os.environ.update(extra_env or {})
        try:
            rc, reason, wall, _ = run_capped(cmd, tl + 30, mem_mb + 512, stdout=out, stderr=subprocess.STDOUT)
        finally:
            os.environ.clear()
            os.environ.update(saved)
        out.seek(0)
        log = out.read()
    res = {"wall": wall, "reason": reason, "rc": rc}
    stats = ROOTSTAT.findall(log)
    if stats:
        first, last = stats[0], stats[-1]
        res.update(root_lp=float(first[1]), bound=float(last[0]), nodelp=float(last[2]),
                   rounds=sum(int(s[3]) for s in stats), cuts=sum(int(s[4]) for s in stats), rows=int(last[5]),
                   time=float(last[6]), restarts=len(stats) - 1, root_done=True)
    else:
        res["root_done"] = False
        mm = re.findall(r"Cuts: \d+ rounds, \d+ cuts, bound (\S+) -> (\S+)", log)
        if mm:
            res["root_lp"], res["bound"] = float(mm[0][0]), float(mm[-1][1])
        mm = re.findall(r"^Bound: (\S+)", log, re.M)
        if mm and "bound" not in res:
            res["bound"] = float(mm[-1])
        res["time"] = wall
    mm = re.search(r"^Status: (\S+)", log, re.M)
    res["status"] = mm.group(1) if mm else None
    mm = re.search(r"^Presolve: .* -> (\d+) rows, (\d+) cols, (\d+) nnz", log, re.M)
    if mm:
        res["presolved"] = [int(mm.group(i)) for i in (1, 2, 3)]
    by = re.findall(r"cut rows by separator:(.*)", log)
    cuts = {}
    if by:
        tok = by[-1].split()
        for k in range(0, len(tok) - 1, 2):
            cuts[tok[k]] = int(tok[k + 1])
    res["cut_rows"] = cuts
    return res


# ----------------------------------------------------------------------------- report


def closed(bound, base, opt):
    if bound is None or base is None or opt is None or not math.isfinite(bound):
        return None
    den = opt - base
    if abs(den) <= 1e-9 * max(1.0, abs(opt)):
        return 100.0
    return max(-100.0, min(100.0, 100.0 * (bound - base) / den))


def fmt(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


def fmtb(v):
    if v is None:
        return "-"
    if not math.isfinite(v):
        return str(v)
    return f"{v:.6g}"


def sgm(vals, shift=1.0):
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v + shift) for v in vals) / len(vals)) - shift


def best_ref(e):
    """(name, result, closed%) of the reference with the largest closed%, or (None, {}, None)."""
    opt, lp0 = e.get("optimum"), e.get("lp0")
    best = (None, {}, None)
    for name, r in (e.get("refs") or {}).items():
        if not r or "error" in r:
            continue
        b = opt if r.get("root_solved") and opt is not None else r.get("bound")
        c = closed(b, lp0, opt)
        if c is not None and (best[2] is None or c > best[2]):
            best = (name, r, c)
    return best


def report(data, file=sys.stdout):
    rows = []
    for name, e in data["instances"].items():
        o, opt, lp0 = e.get("ours") or {}, e.get("optimum"), e.get("lp0")
        rname, r, rc = best_ref(e)
        ob = o.get("bound")
        rows.append(dict(
            name=name, opt=opt, lp0=lp0, rname=rname or "-", r_c=rc, r_t=r.get("time"), r_solved=r.get("root_solved"),
            o_lp=o.get("root_lp"), o_b=ob, o_c=closed(ob, lp0, opt), o_own=closed(ob, o.get("root_lp"), opt),
            o_t=o.get("time"), o_done=o.get("root_done"), o_rounds=o.get("rounds"), o_cuts=o.get("cuts"),
            o_rows=o.get("rows"), o_restarts=o.get("restarts", 0)))
    print("| instance | optimum | LP0 | best ref | ref closed% | ref time | ours root LP | ours bound | "
          "ours closed% | ours own% | ours time | rounds/cuts/rows |", file=file)
    print("|---|---|---|---|---|---|---|---|---|---|---|---|", file=file)
    for r in rows:
        rt = fmt(r["r_t"], 1) + ("*" if r["r_solved"] else "")
        ot = fmt(r["o_t"], 1) + ("" if r["o_done"] else " TL") + ("R" * (r["o_restarts"] or 0))
        print(f"| {r['name']} | {fmtb(r['opt'])} | {fmtb(r['lp0'])} | {r['rname']} | {fmt(r['r_c'])} | {rt} | "
              f"{fmtb(r['o_lp'])} | {fmtb(r['o_b'])} | {fmt(r['o_c'])} | {fmt(r['o_own'])} | {ot} | "
              f"{r['o_rounds'] or 0}/{r['o_cuts'] or 0}/{r['o_rows'] or 0} |", file=file)
    both = [r for r in rows if r["r_c"] is not None and r["o_c"] is not None]
    if both:
        n = len(both)
        behind = [r for r in both if r["o_c"] < r["r_c"] - 5.0]
        ahead = [r for r in both if r["o_c"] > r["r_c"] + 5.0]
        print(file=file)
        print(f"{n} instances: mean closed% best reference {sum(r['r_c'] for r in both) / n:.1f}, ours "
              f"{sum(r['o_c'] for r in both) / n:.1f}; ours >5 points behind on {len(behind)}, ahead on {len(ahead)}; "
              f"root time shifted geomean (1 s) reference {sgm([r['r_t'] for r in both]):.2f} s, "
              f"ours {sgm([r['o_t'] for r in both]):.2f} s", file=file)
        gap = sorted(both, key=lambda r: r["o_c"] - r["r_c"])[:15]
        print("largest deficits: " + ", ".join(f"{r['name']} {r['o_c']:.0f} vs {r['r_c']:.0f}" for r in gap), file=file)


def compare(pa, pb, file=sys.stdout):
    """Ours in two result files (e.g. before / after a change), with the best reference."""
    a, b = json.loads(Path(pa).read_text()), json.loads(Path(pb).read_text())
    print("| instance | ref closed% | A closed% | B closed% | ref time | A time | B time | A rounds/cuts | B rounds/cuts |",
          file=file)
    print("|---|---|---|---|---|---|---|---|---|", file=file)
    ca, cb, ta, tb = [], [], [], []
    for name, eb in b["instances"].items():
        ea = a["instances"].get(name)
        if not ea or not ea.get("ours") or not eb.get("ours"):
            continue
        opt, lp0 = eb.get("optimum"), eb.get("lp0")
        _, r, xr = best_ref(eb)
        oa, ob = ea["ours"], eb["ours"]
        xa, xb = closed(oa.get("bound"), lp0, opt), closed(ob.get("bound"), lp0, opt)
        flag = ""
        if xa is not None and xb is not None and abs(xa - xb) >= 2.0:
            flag = " **+**" if xb > xa else " **-**"
        print(f"| {name}{flag} | {fmt(xr)} | {fmt(xa)} | {fmt(xb)} | {fmt(r.get('time'))} | {fmt(oa.get('time'))} | "
              f"{fmt(ob.get('time'))} | {oa.get('rounds', 0)}/{oa.get('cuts', 0)} | {ob.get('rounds', 0)}/{ob.get('cuts', 0)} |",
              file=file)
        if xa is not None and xb is not None:
            ca.append(xa), cb.append(xb), ta.append(oa.get("time")), tb.append(ob.get("time"))
    if ca:
        n = len(ca)
        print(f"\n{n} instances: mean closed% A {sum(ca) / n:.1f}, B {sum(cb) / n:.1f}; better by >= 2 points: "
              f"{sum(1 for x, y in zip(ca, cb) if y >= x + 2)}, worse: {sum(1 for x, y in zip(ca, cb) if y <= x - 2)}; "
              f"root time sgm A {sgm(ta):.2f} s, B {sgm(tb):.2f} s", file=file)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", nargs="*", help="instance names or .mps paths")
    ap.add_argument("--set", help="instance list file (benchmarks/sets/*.txt)")
    ap.add_argument("--solver", default=str(ROOT / "build" / "popso"), help="our solver command")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("-t", "--time-limit", type=float, default=120.0)
    ap.add_argument("--mem-limit", type=int, default=2048, help="MB")
    ap.add_argument("--out", help="result .json (updated after each instance; existing entries are kept)")
    ap.add_argument("--refs", default="scip", help="reference solvers: comma list of scip, highs")
    ap.add_argument("--ref-cache", default=str(BENCH / "results" / "rootgap_refs.json"))
    ap.add_argument("--no-refs", action="store_true", help="only use cached reference results")
    ap.add_argument("--no-ours", action="store_true")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for our solver (repeatable)")
    ap.add_argument("--report", help="print the table of a result .json and exit")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare ours in two result .json files and exit")
    args = ap.parse_args(argv)

    if args.compare:
        compare(*args.compare)
        return 0
    if args.report:
        report(json.loads(Path(args.report).read_text()))
        return 0
    insts = list(args.instances)
    if args.set:
        insts += read_set(args.set)
    if not insts:
        ap.error("no instances")
    known = load_references(str(BENCH / "reference" / "solutions.json"), str(BENCH / "reference" / "miplib_solu.json"))
    cache_p = Path(args.ref_cache)
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    out_p = Path(args.out) if args.out else None
    data = json.loads(out_p.read_text()) if out_p and out_p.exists() else {"instances": {}}
    data["meta"] = {"solver": args.solver, "threads": args.threads, "time_limit": args.time_limit,
                    "mem_limit": args.mem_limit, "env": args.env, "refs": args.refs}
    extra_env = dict(kv.split("=", 1) for kv in args.env)
    ref_names = [r.strip() for r in args.refs.split(",") if r.strip()]

    def cached(key, fn):
        if key not in cache and not args.no_refs:
            print(f"[{key}]", file=sys.stderr, flush=True)
            try:
                cache[key] = fn()
            except Exception as ex:  # noqa: BLE001
                cache[key] = {"error": str(ex)}
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            cache_p.write_text(json.dumps(cache, indent=1))
        return cache.get(key)

    for inst in insts:
        p = Path(inst)
        if not p.suffix:
            p = BENCH / "miplib" / f"{inst}.mps"
            if not p.exists():
                p = BENCH / "instances" / f"{inst}.mps"
        elif not p.is_absolute():
            p = ROOT / p
        name = inst_name(p)
        ref = known.get(name, {})
        e = {"optimum": ref.get("objective") if ref.get("status") == "optimal" else None}
        lp0 = cached(f"lp0|{name}", lambda: lp0_highs(p, args.time_limit, args.threads))
        e["lp0"] = lp0 if isinstance(lp0, (int, float)) else None
        e["refs"] = {rn: cached(f"{rn}|{name}|{args.threads}|{args.time_limit}",
                                lambda rn=rn: REFS[rn](p, args.threads, args.time_limit, args.mem_limit))
                     for rn in ref_names}
        if not args.no_ours:
            print(f"[ours] {name}", file=sys.stderr, flush=True)
            e["ours"] = ours_root(args.solver, p, args.threads, args.time_limit, args.mem_limit, extra_env)
        elif name in data["instances"]:
            e["ours"] = data["instances"][name].get("ours")
        data["instances"][name] = e
        if out_p:
            out_p.write_text(json.dumps(data, indent=1))
    report(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
