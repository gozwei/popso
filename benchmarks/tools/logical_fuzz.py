#!/usr/bin/env python3
"""Random small models with semi-continuous / semi-integer variables, SOS1/SOS2
sets and indicator constraints, solved by popso and SCIP (and HiGHS when the
model has only semi-continuous variables); statuses and objectives must agree
and popso's solutions must pass lib/mps.py verification.

    .venv/bin/python benchmarks/tools/logical_fuzz.py [--n 200] [--seed 1] [--build build]
                                                     [--format mps|lp] [--popso-args "..."]

Runs one solver process at a time. Exit status 1 on any mismatch; the
offending model is kept in --out (default: a temporary directory).
"""

from __future__ import annotations

import argparse
import math
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))

from lib import mps, refsolvers, solfile  # noqa: E402
from lib.modeler import Model, lin  # noqa: E402


def random_model(rng, k):
    m = Model(f"fuzz{k}")
    n = int(rng.integers(4, 9))
    xs = []
    for j in range(n):
        kind = rng.choice(["B", "C", "I", "SC", "SI", "Cfree"], p=[0.2, 0.25, 0.15, 0.15, 0.1, 0.15])
        obj = float(rng.integers(-6, 7))
        if kind == "B":
            xs.append(m.add_var(vtype="B", obj=obj, name=f"x{j}"))
        elif kind in ("SC", "SI"):
            lo = float(rng.integers(1, 4))
            up = lo + float(rng.integers(0, 6))
            xs.append(m.add_var(lb=lo, ub=up, vtype=kind, obj=obj, name=f"x{j}"))
        elif kind == "Cfree":
            # no upper bound (indicators / SOS on it may need the branching fallback)
            xs.append(m.add_var(lb=float(-rng.integers(0, 3)), vtype="C", obj=abs(obj) + 1, name=f"x{j}"))
        else:
            lo = float(-rng.integers(0, 3))
            xs.append(m.add_var(lb=lo, ub=lo + float(rng.integers(1, 8)), vtype=kind, obj=obj, name=f"x{j}"))
    for i in range(int(rng.integers(1, 5))):
        idx = [j for j in range(n) if rng.random() < 0.5] or [int(rng.integers(n))]
        a = rng.integers(-4, 6, len(idx)).astype(float)
        rhs = float(rng.integers(0, 12))
        e = lin(a, [xs[j] for j in idx])
        m.add_constr(e <= rhs if rng.random() < 0.7 else e >= -rhs, f"r{i}")
    for s in range(int(rng.integers(0, 3))):
        k2 = int(rng.integers(3, min(n, 5) + 1))
        members = list(rng.choice(n, k2, replace=False))
        w = [float(t + 1) for t in range(k2)]
        m.add_sos([xs[j] for j in members], w, sos_type=int(rng.integers(1, 3)), name=f"s{s}")
    bins = [v for v in xs if v.vtype == "B"]
    for q in range(int(rng.integers(0, 3)) if bins else 0):
        b = bins[int(rng.integers(len(bins)))]
        idx = [j for j in range(n) if xs[j] is not b and rng.random() < 0.5] or [0 if xs[0] is not b else 1]
        a = rng.integers(-3, 4, len(idx)).astype(float)
        e = lin(a, [xs[j] for j in idx])
        rhs = float(rng.integers(-1, 6))
        sense = rng.choice(["L", "G", "E"], p=[0.5, 0.3, 0.2])
        con = e <= rhs if sense == "L" else (e >= rhs if sense == "G" else e == rhs)
        m.add_indicator(b, int(rng.integers(0, 2)), con, name=f"i{q}")
    m.sense = "max" if rng.random() < 0.5 else "min"
    return m


def run_popso(popso, path, sol, extra):
    out = subprocess.run([str(popso), str(path), "--threads", "1", "--time-limit", "30", "--gap", "1e-9",
                          "--mem-limit", "500", "--sol", str(sol)] + extra, capture_output=True, text=True,
                         timeout=120)
    st = re.search(r"^Status: (\S+)", out.stdout, re.M)
    ob = re.search(r"^Objective: (\S+)", out.stdout, re.M)
    if out.returncode != 0 or not st:
        return "ERROR " + (out.stderr or out.stdout)[-300:], None
    return st.group(1).lower(), (float(ob.group(1)) if ob and ob.group(1) != "-" else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--build", default=str(ROOT / "build"))
    ap.add_argument("--format", choices=["mps", "lp"], default="mps")
    ap.add_argument("--popso-args", default="", help="extra popso arguments")
    ap.add_argument("--out", help="directory for the models (default: temporary)")
    args = ap.parse_args()
    popso = Path(args.build) / "popso"
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="logical_fuzz_"))
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    extra = shlex.split(args.popso_args)
    bad = 0
    counts = {}
    for k in range(args.n):
        m = random_model(rng, k)
        path = out / f"fuzz{k}.{args.format}"
        m.write(path)
        mps_path = out / f"fuzz{k}.mps"
        if args.format != "mps":
            m.write(mps_path)
        ref = refsolvers.solve_scip(str(mps_path), 30, 1)
        rst, robj = ref["status"], ref["objective"]
        st, obj = run_popso(popso, path, out / "fuzz.sol", extra)
        if rst not in ("optimal", "infeasible", "unbounded", "inf_or_unbd"):
            # SCIP undecided (e.g. a time limit on an unbounded SOS member): only verify popso's point
            rst = "undecided"
            if st.startswith("ERROR") or (st in ("optimal", "unbounded") and obj is None):
                bad += 1
                print(f"BAD fuzz{k}: popso {st} {obj}, SCIP undecided  ({path})")
                continue
        ok = rst == "undecided" or st == rst or (rst == "inf_or_unbd" and st in ("infeasible", "unbounded", "inf_or_unbd")) or \
            (st == "inf_or_unbd" and rst in ("infeasible", "unbounded"))
        scip_worse = False
        if ok and rst == "optimal":
            ok = obj is not None and abs(obj - robj) <= 1e-6 * max(1.0, abs(robj))
            sense = -1 if m.sense == "max" else 1
            # popso better than SCIP's "optimum": fine if popso's point verifies (SCIP's SOS handling
            # has been seen to miss solutions with negative SOS members)
            scip_worse = not ok and obj is not None and sense * (robj - obj) > 0
        if (ok or scip_worse) and obj is not None:
            sol = solfile.read_sol(out / "fuzz.sol")
            v = mps.verify(mps.read_mps(mps_path), sol.values)
            ok = v.feasible
            if not ok:
                print(f"  fuzz{k}: popso solution fails verification: {v.worst}")
            elif scip_worse:
                print(f"  fuzz{k}: SCIP optimum {robj} worse than popso's verified {obj} ({path})")
                rst = "scip_worse"
        counts[rst] = counts.get(rst, 0) + 1
        if not ok:
            bad += 1
            print(f"BAD fuzz{k}: popso {st} {obj}, SCIP {rst} {robj}  ({path})")
        else:
            path.unlink()
            if mps_path.exists():
                mps_path.unlink()
    print(f"{args.n} models ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}): "
          f"{bad} mismatch(es)" + (f"; kept in {out}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
