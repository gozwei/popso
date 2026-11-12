#!/usr/bin/env python3
"""A/B comparison of two solver builds over several random seeds.

Runs benchmarks/run.py for build A and build B (interleaved per seed, so slow
drifts in machine load affect both; A first on odd-numbered seeds of the list, B first
on the others), then reports:
  * wrong answers per side (any wrong answer fails the comparison),
  * solved counts,
  * shifted geometric mean time (shift 1 s, unsolved = time limit) per side,
  * the speed ratio B/A with a 90% bootstrap confidence interval over instances,
  * per-family ratios and the instances with the largest changes.

Usage:
  .venv/bin/python benchmarks/ab.py --a build-base/popso --b build/popso --seeds 1,2,3 --sizes small,medium -t 60
  .venv/bin/python benchmarks/ab.py --a build/popso --b build/popso --b-args "--deterministic" ...

Extra arguments for each side go in --a-args / --b-args. Instance selection
options (--sizes, --families, --max-size, positional globs, --miplib, --set) are
forwarded to run.py.

With -j > 1 all sides and seeds run at once and share one pool of -j x --threads threads and
-j x --mem-limit MB (slots.py; the machine load stays as with one run.py -j at a time), so the
machine does not idle while one run's last instances finish. A pool given in POPSO_SLOTS (the
benchmark server's queue sets one for all jobs) is used instead. --sequential: one run at a time.
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BENCH = Path(__file__).resolve().parent
RUN = BENCH / "run.py"
sys.path.insert(0, str(BENCH))
from lib.primal import pgap as primal_gap  # noqa: E402


def run_side(solver, extra, seed, args, label, env_extra=""):
    cmd = [sys.executable, str(RUN), "--solver", f"{solver} --seed {seed} {extra}".strip(),
           "-t", str(args.time_limit), "--threads", str(args.threads), "-j", str(args.jobs),
           "--mem-limit", str(args.mem_limit), "--gap", str(args.gap), "--tag", f"ab-{label}-s{seed}"]
    for opt in ("sizes", "families", "max_size"):
        v = getattr(args, opt)
        if v:
            cmd += ["--" + opt.replace("_", "-"), v]
    if args.miplib:
        cmd.append("--miplib")
    if args.keep_logs:
        cmd.append("--keep-sols")
    if args.primal:
        cmd.append("--trace")
    for st in args.set:
        cmd += ["--set", st]
    if args.starts:
        cmd += ["--starts", args.starts]
    cmd += args.instances
    print(f"[ab] {label} seed {seed}: {' '.join(cmd[2:5])} ...", flush=True)
    env = dict(os.environ)
    for kv in env_extra.split():
        k, _, v = kv.partition("=")
        env[k] = v
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    m = re.search(r"^results: (.+\.json)$", out.stdout, re.M)
    if not m:
        sys.stderr.write(out.stdout[-3000:] + out.stderr[-3000:])
        raise SystemExit(f"[ab] run.py failed for {label} seed {seed}")
    return json.loads(Path(m.group(1)).read_text())


def family(name):
    return name.split("_")[0] if "_" in name else name


def sgm(times, shift=1.0):
    return math.exp(sum(math.log(t + shift) for t in times) / len(times)) - shift


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="solver binary A (baseline)")
    ap.add_argument("--b", required=True, help="solver binary B (candidate)")
    ap.add_argument("--a-args", default="")
    ap.add_argument("--b-args", default="")
    ap.add_argument("--a-env", default="", help="environment for A, e.g. 'MILP_NO_RCFIX=1'")
    ap.add_argument("--b-env", default="")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("-t", "--time-limit", type=float, default=60)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("-j", "--jobs", type=int, default=1,
                    help="parallel solver runs: run.py -j, and the size of the pool all runs share (see above)")
    ap.add_argument("--mem-limit", type=int, default=1500)
    ap.add_argument("--gap", type=float, default=1e-6)
    ap.add_argument("--sizes")
    ap.add_argument("--families")
    ap.add_argument("--max-size")
    ap.add_argument("--miplib", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="instance list (see run.py --set)")
    ap.add_argument("--out", help="write the comparison as JSON here")
    ap.add_argument("--keep-logs", action="store_true", help="keep the solver logs (run.py --keep-sols)")
    ap.add_argument("--sequential", action="store_true", help="with -j > 1: one run.py at a time (no shared pool)")
    ap.add_argument("--primal", action="store_true",
                    help="popso only: also compare the primal integral (run.py --trace): mean primal gap over the "
                         "time limit per side, B - A with a 90%% bootstrap interval")
    ap.add_argument("--starts", metavar="DIR", help="warm starts for both sides (run.py --starts)")
    ap.add_argument("instances", nargs="*")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # times[side][instance] -> list over seeds
    times = {"A": defaultdict(list), "B": defaultdict(list)}
    solved = {"A": 0, "B": 0}
    wrong = {"A": [], "B": []}
    pint = {"A": defaultdict(list), "B": defaultdict(list)}  # primal integrals per instance (--primal)
    fgap = {"A": defaultdict(list), "B": defaultdict(list)}  # final primal gaps per instance (--primal)
    plan = []
    for k, seed in enumerate(seeds):
        sides = [("A", args.a, args.a_args, args.a_env), ("B", args.b, args.b_args, args.b_env)]
        # Alternate which side runs first: an A/A run on the 12-core server (12 parallel jobs, A always first)
        # came out 0.958 [0.925, 0.991], the second run of each seed faster (warm file cache, clocks).
        if k % 2 == 1:
            sides.reverse()
        plan += [(seed, side) for side in sides]
    if args.jobs > 1 and not args.sequential:
        # All runs at once over one pool (slots.py): each run.py would otherwise idle down to its last instances
        # before the next one starts (on the 12-job server, minutes per run).
        pool = None
        if not os.environ.get("POPSO_SLOTS"):
            pool = tempfile.mkdtemp(prefix="ab-slots-", dir=BENCH / "results")
            os.environ["POPSO_SLOTS"] = f"{pool}:{args.jobs * args.threads}:{args.jobs * args.mem_limit}"
        try:
            with ThreadPoolExecutor(max_workers=len(plan)) as ex:
                futs = [ex.submit(run_side, solver, extra, seed, args, side, env)
                        for seed, (side, solver, extra, env) in plan]
                results = [f.result() for f in futs]
        finally:
            if pool:
                shutil.rmtree(pool, ignore_errors=True)
    else:
        results = [run_side(solver, extra, seed, args, side, env) for seed, (side, solver, extra, env) in plan]
    for (seed, (side, *_)), res in zip(plan, results):
        for r in res["results"]:
            name = r["instance"]
            # Solved: agreeing with the reference, or optimal and verified where there is none (QPLIB / CBLIB
            # instances without references counted as unsolved before: every time the limit, every ratio 1).
            ok = r["class"] == "ok" or (r["class"] == "noref" and r.get("status") == "optimal")
            if r["class"] == "WRONG":
                wrong[side].append((name, seed, r.get("message", "")))
            solved[side] += ok
            times[side][name].append(min(r["time"], args.time_limit) if ok else args.time_limit)
            if r.get("primal_integral") is not None:
                pint[side][name].append(r["primal_integral"])
            if r.get("ref_objective") is not None and r["class"] != "WRONG":
                z = r.get("objective_recomputed", r.get("objective"))
                fgap[side][name].append(primal_gap(z, r["ref_objective"]))

    names = sorted(set(times["A"]) & set(times["B"]))
    if not names:
        raise SystemExit("[ab] no common instances")
    # Per-instance geometric mean over seeds, then shifted geomean over instances.
    per = {n: (sgm(times["A"][n]), sgm(times["B"][n])) for n in names}
    a_all = [per[n][0] for n in names]
    b_all = [per[n][1] for n in names]
    ga, gb = sgm(a_all), sgm(b_all)
    ratio = (gb + 1) / (ga + 1)  # ratio of shifted geomeans

    rng = random.Random(0)
    boots = []
    for _ in range(2000):
        sample = [rng.choice(names) for _ in names]
        ba = sgm([per[n][0] for n in sample])
        bb = sgm([per[n][1] for n in sample])
        boots.append((bb + 1) / (ba + 1))
    boots.sort()
    lo, hi = boots[int(0.05 * len(boots))], boots[int(0.95 * len(boots))]

    print()
    print(f"instances {len(names)} x seeds {len(seeds)}   (time limit {args.time_limit:g}s, threads {args.threads})")
    for side in ("A", "B"):
        print(f"  {side}: solved {solved[side]}/{len(names) * len(seeds)}  wrong {len(wrong[side])}  "
              f"shifted geomean {sgm(a_all if side == 'A' else b_all):.3f}s")
    verdict = "FASTER" if hi < 1.0 else "SLOWER" if lo > 1.0 else "no significant difference"
    print(f"  ratio B/A of shifted geomeans: {ratio:.3f}   90% CI [{lo:.3f}, {hi:.3f}]  -> B is {verdict}")

    fam = defaultdict(list)
    for n in names:
        fam[family(n)].append(n)
    print("\nper family (B/A ratio of shifted geomeans):")
    for f in sorted(fam):
        fa = sgm([per[n][0] for n in fam[f]])
        fb = sgm([per[n][1] for n in fam[f]])
        print(f"  {f:14s} {len(fam[f]):3d} inst  A {fa:8.3f}s  B {fb:8.3f}s  ratio {(fb + 1) / (fa + 1):.3f}")

    changes = sorted(names, key=lambda n: (per[n][1] + 1) / (per[n][0] + 1))
    print("\nlargest speedups:")
    for n in changes[:5]:
        print(f"  {n:34s} A {per[n][0]:8.2f}s  B {per[n][1]:8.2f}s")
    print("largest slowdowns:")
    for n in changes[::-1][:5]:
        print(f"  {n:34s} A {per[n][0]:8.2f}s  B {per[n][1]:8.2f}s")

    primal = None
    pnames = sorted(n for n in names if pint["A"][n] and pint["B"][n])
    if args.primal and pnames:
        pa = {n: sum(pint["A"][n]) / len(pint["A"][n]) for n in pnames}
        pb = {n: sum(pint["B"][n]) / len(pint["B"][n]) for n in pnames}
        ma, mb = sum(pa.values()) / len(pnames), sum(pb.values()) / len(pnames)
        diffs = []
        for _ in range(2000):
            sample = [rng.choice(pnames) for _ in pnames]
            diffs.append(sum(pb[n] - pa[n] for n in sample) / len(sample))
        diffs.sort()
        plo, phi = diffs[int(0.05 * len(diffs))], diffs[int(0.95 * len(diffs))]
        primal = {"A": ma, "B": mb, "diff": mb - ma, "ci90": [plo, phi], "instances": len(pnames)}
        print(f"\nprimal integral / time limit (mean over {len(pnames)} instances with a reference objective):")
        print(f"  A {ma:.4f}  B {mb:.4f}  B - A {mb - ma:+.4f}   90% CI [{plo:+.4f}, {phi:+.4f}]")
        ch = sorted(pnames, key=lambda n: pb[n] - pa[n])
        print("  largest improvements: " + ", ".join(f"{n} {pa[n]:.3f}->{pb[n]:.3f}" for n in ch[:5] if pb[n] < pa[n]))
        print("  largest losses:       " + ", ".join(f"{n} {pa[n]:.3f}->{pb[n]:.3f}" for n in ch[::-1][:5] if pb[n] > pa[n]))

    final = None
    fnames = sorted(n for n in names if fgap["A"][n] and fgap["B"][n])
    if args.primal and fnames:
        # The primal gap of the final solution (0 at the reference objective, 1 without a solution), mean over
        # seeds: what a time-limited run delivers. Instances differing by less than 1e-6 count as equal.
        fa = {n: sum(fgap["A"][n]) / len(fgap["A"][n]) for n in fnames}
        fb = {n: sum(fgap["B"][n]) / len(fgap["B"][n]) for n in fnames}
        ma, mb = sum(fa.values()) / len(fnames), sum(fb.values()) / len(fnames)
        diffs = []
        for _ in range(2000):
            sample = [rng.choice(fnames) for _ in fnames]
            diffs.append(sum(fb[n] - fa[n] for n in sample) / len(sample))
        diffs.sort()
        flo, fhi = diffs[int(0.05 * len(diffs))], diffs[int(0.95 * len(diffs))]
        b_better = sum(fb[n] < fa[n] - 1e-6 for n in fnames)
        a_better = sum(fa[n] < fb[n] - 1e-6 for n in fnames)
        final = {"A": ma, "B": mb, "diff": mb - ma, "ci90": [flo, fhi], "instances": len(fnames),
                 "b_better": b_better, "a_better": a_better, "per_instance": {n: (fa[n], fb[n]) for n in fnames}}
        print(f"\nfinal primal gap (mean over {len(fnames)} instances with a reference objective):")
        print(f"  A {ma:.5f}  B {mb:.5f}  B - A {mb - ma:+.5f}   90% CI [{flo:+.5f}, {fhi:+.5f}]   "
              f"B better on {b_better}, A better on {a_better}")
        ch = sorted(fnames, key=lambda n: fb[n] - fa[n])
        print("  largest improvements: " + ", ".join(f"{n} {fa[n]:.4f}->{fb[n]:.4f}" for n in ch[:5] if fb[n] < fa[n] - 1e-6))
        print("  largest losses:       " + ", ".join(f"{n} {fa[n]:.4f}->{fb[n]:.4f}" for n in ch[::-1][:5] if fb[n] > fa[n] + 1e-6))

    for side in ("A", "B"):
        for name, seed, msg in wrong[side]:
            print(f"!!! WRONG {side} {name} seed {seed}: {msg}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "a": args.a, "b": args.b, "a_args": args.a_args, "b_args": args.b_args, "seeds": seeds,
            "ratio": ratio, "ci90": [lo, hi], "per_instance": per, "wrong": wrong, "solved": solved,
            **({"primal": primal} if primal else {}), **({"final_gap": final} if final else {})}, indent=1))
    sys.exit(1 if wrong["B"] else 0)


if __name__ == "__main__":
    main()
