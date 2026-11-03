#!/usr/bin/env python3
"""Cross-check popso's LP-format reader against its MPS reader (and HiGHS).

    lp_crosscheck.py [--build build] [--out DIR] [--generate] [--highs N] [--scip N] [instances/*.mps ...]

1. Converts every benchmark instance (default: benchmarks/instances/*.mps) to
   the CPLEX LP format with the pure-Python writer in lib/modeler.py (read
   with the independent reader lib/mps.py), then runs the unit test
   lp_reader_crosscheck_dir, which reads each pair with popso and requires
   the two models to be identical (names, order, bounds, types, numbers).
   With --generate, the generated families are instead rebuilt by
   generators/generate.py and written with Model.to_lp directly.
2. With --highs N: for the N smallest instances, HiGHS reads the MPS file and
   writes an LP file; popso solves that LP file (1 thread) and the result is
   compared with HiGHS solving the MPS file (status and objective).
   --scip N does the same with LP files written by SCIP (pyscipopt).

The instances must live in benchmarks/instances (the unit test looks up the
MPS original there by file stem). The writer emits ranged rows as
"lo <= expr <= hi"; HiGHS reads that form, SCIP's LP reader does not.

Runs one process at a time. Exit status 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
ROOT = BENCH.parent
sys.path.insert(0, str(BENCH))

from lib import mps as mpsmod  # noqa: E402
from lib.modeler import lp_text  # noqa: E402


def mps_to_lp(path: Path) -> str:
    m = mpsmod.read_mps(str(path))
    cols = []
    for j, name in enumerate(m.col_names):
        lb, ub = m.lb[j], m.ub[j]
        # popso maps |value| >= 1e30 to infinity; write it as inf.
        lb = -math.inf if lb <= -1e30 else lb
        ub = math.inf if ub >= 1e30 else ub
        vtype = ("B" if (lb, ub) == (0.0, 1.0) else "I") if m.is_int[j] else "C"
        if m.semi[j]:
            vtype = "SI" if m.is_int[j] else "SC"
        cols.append((name, m.obj[j], lb, ub, vtype))
    rows = []
    for i, name in enumerate(m.row_names):
        lo, hi = m.row_lo[i], m.row_hi[i]
        lo = -math.inf if lo <= -1e30 else lo
        hi = math.inf if hi >= 1e30 else hi
        terms = [(m.col_names[j], v) for j, v in zip(m.row_cols[i], m.row_vals[i])]
        rows.append((name, lo, hi, terms))
    sos = [(n, t, [m.col_names[j] for j in c], w) for n, t, c, w in m.sos]
    inds = []
    for n, b, val, lo, hi, c, v in m.indicators:
        sense, rhs = ("E", lo) if lo == hi else (("L", hi) if lo == -math.inf else ("G", lo))
        if lo != hi and lo != -math.inf and hi != math.inf:
            raise ValueError(f"ranged indicator constraint {n} cannot be written in LP format")
        inds.append((n, m.col_names[b], val, sense, rhs, [(m.col_names[j], a) for j, a in zip(c, v)]))
    return lp_text(m.name, "max" if m.sense == -1 else "min", m.obj_const, cols, rows, sos, inds)


def run_popso(popso: Path, model: Path, time_limit: float):
    out = subprocess.run([str(popso), str(model), "--threads", "1", "--time-limit", str(time_limit),
                          "--mem-limit", "1000"], capture_output=True, text=True, timeout=time_limit + 60)
    status = re.search(r"^Status: (\S+)", out.stdout, re.M)
    obj = re.search(r"^Objective: (\S+)", out.stdout, re.M)
    if out.returncode != 0 or not status:
        return "ERROR: " + (out.stderr.strip() or out.stdout.strip())[:300], None
    o = float(obj.group(1)) if obj and obj.group(1) != "-" else None
    return status.group(1), o


def write_lp_scip(mps: Path, lp_file: Path):
    from pyscipopt import Model

    m = Model()
    m.hideOutput()
    m.readProblem(str(mps))
    m.writeProblem(str(lp_file), verbose=False)


def solver_check(writer: str, popso: Path, instances, n: int, outdir: Path, time_limit: float) -> int:
    """`writer` ("HiGHS" or "SCIP") converts MPS -> LP; popso solves the LP
    file; HiGHS solving the MPS file is the reference."""
    import highspy

    bad = 0
    smallest = sorted(instances, key=lambda p: p.stat().st_size)[:n]
    for p in smallest:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("time_limit", time_limit)
        h.readModel(str(p))
        lp_file = outdir / f"{p.stem}.{writer.lower()}.lp"
        if writer == "HiGHS":
            h.writeModel(str(lp_file))
        else:
            write_lp_scip(p, lp_file)
        h.run()
        hs = h.modelStatusToString(h.getModelStatus())
        hobj = h.getInfo().objective_function_value if hs == "Optimal" else None
        st, obj = run_popso(popso, lp_file, time_limit)
        if hobj is None:
            expect = {"Infeasible": {"INFEASIBLE", "INF_OR_UNBD"},
                      "Unbounded": {"UNBOUNDED", "INF_OR_UNBD"},
                      "Primal infeasible or unbounded": {"INFEASIBLE", "UNBOUNDED", "INF_OR_UNBD"}}
            ok = st in expect.get(hs, set())
            note = f"HiGHS {hs}"
        else:
            ok = st == "OPTIMAL" and obj is not None and abs(obj - hobj) <= 1e-6 * max(1.0, abs(hobj))
            note = f"HiGHS {hobj:.10g}"
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'} {p.stem:<32} popso(LP by {writer}) {st} {obj}  {note}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", nargs="*", type=Path)
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--out", type=Path, help="directory for the LP files (default: a temporary directory)")
    ap.add_argument("--highs", type=int, default=0, metavar="N", help="also cross-check N instances via HiGHS-written LP")
    ap.add_argument("--generate", action="store_true",
                    help="write generated families with Model.to_lp straight from the generators "
                         "(instead of converting their MPS files)")
    ap.add_argument("--scip", type=int, default=0, metavar="N", help="also cross-check N instances via SCIP-written LP")
    ap.add_argument("--time-limit", type=float, default=30.0)
    args = ap.parse_args()

    instances = args.instances or sorted((BENCH / "instances").glob("*.mps"))
    tmp = None
    if args.out:
        outdir = args.out
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="popso_lp_")
        outdir = Path(tmp.name)
    lpdir = outdir / "lp"
    lpdir.mkdir(exist_ok=True)
    for old in lpdir.glob("*.lp"):
        old.unlink()
    generated = set()
    if args.generate:
        from generators import families
        from generators.generate import CONFIG, instance_name, rng_for

        wanted = {p.stem for p in instances}
        for family, size, seed, params in CONFIG:
            name = instance_name(family, size, seed)
            if name not in wanted:
                continue
            m = families.BUILDERS[family](rng_for(family, size, seed), **params)
            m.name = name
            m.write(lpdir / f"{name}.lp")  # Model.to_lp
            generated.add(name)
    for p in instances:
        if p.stem not in generated:
            (lpdir / f"{p.stem}.lp").write_text(mps_to_lp(p))
    print(f"wrote {len(instances)} LP files to {lpdir} ({len(generated)} from the generators)", flush=True)

    env = dict(os.environ, POPSO_LP_CROSSCHECK_DIR=str(lpdir))
    res = subprocess.run([str(args.build / "popso_tests"), "lp_reader_crosscheck_dir"], env=env)
    bad = int(res.returncode != 0)
    if args.highs:
        print(f"HiGHS-written LP files ({args.highs} smallest instances):")
        bad += solver_check("HiGHS", args.build / "popso", instances, args.highs, outdir, args.time_limit)
    if args.scip:
        print(f"SCIP-written LP files ({args.scip} smallest instances):")
        bad += solver_check("SCIP", args.build / "popso", instances, args.scip, outdir, args.time_limit)
    print("LP cross-check:", "OK" if bad == 0 else f"{bad} FAILURE(S)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
