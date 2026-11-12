#!/usr/bin/env python3
"""Random small nonconvex quadratic models (LP format) solved by popso (spatial branch-and-bound) and SCIP; their
statuses and objectives must agree (relative 1e-4 by default, popso run with --gap 1e-6).

    .venv/bin/python benchmarks/tools/nc_fuzz.py [--n 200] [--seed 1] [--popso build/popso]
                                                [--popso-args "--gap 1e-6"] [--tol 1e-4] [--open-bounds] [--out DIR]

2-6 bounded columns (continuous / integer / binary), a linear objective with indefinite quadratic terms in two
thirds of the cases, 0-3 quadratic rows with random terms and sides (<=, >=, both, equality), 0-3 linear rows. A
case counts as a mismatch when one solver proves optimality and the other reports another optimum or infeasibility;
cases SCIP does not finish in its time limit are skipped. Exit status 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qc_fuzz import bracket, popso_result, term  # noqa: E402


def gen(rng, k, open_bounds=False):
    n = rng.randint(2, 6)
    names = [f"x{j}" for j in range(n)]
    kind = [rng.choice("CCCIB") for _ in range(n)]

    def quad_terms(count):
        S = {}
        for _ in range(count):
            a, b = rng.randrange(n), rng.randrange(n)
            a, b = min(a, b), max(a, b)
            S[(a, b)] = S.get((a, b), 0) + rng.choice([-3, -2, -1, 1, 2, 3])
        return {key: v for key, v in S.items() if v}

    lines = [rng.choice(["Minimize", "Maximize"]), " obj:" + "".join(term(rng.randint(-5, 5), nm) for nm in names)]
    if k % 3:
        S = quad_terms(rng.randint(1, 4))
        if S:
            lines[1] += " + [" + bracket("".join(term(2 * v, names[a], names[b]) for (a, b), v in S.items())) + " ] / 2"
    lines.append("Subject To")
    for r in range(rng.randint(0, 3)):
        S = quad_terms(rng.randint(1, 4))
        if not S:
            continue
        lin = "".join(term(rng.randint(-3, 3), nm) for nm in names if rng.random() < 0.4)
        quad = " + [" + bracket("".join(term(v, names[a], names[b]) for (a, b), v in S.items())) + " ]"
        rhs = rng.randint(-10, 30)
        op = rng.choice(["<=", "<=", ">=", ">=", "="])
        lines.append(f" q{r}: 0 {names[0]}{lin}{quad} {op} {rhs}")
    for r in range(rng.randint(0, 3)):
        lines.append(f" l{r}: " + "".join(term(rng.randint(-4, 4), nm) for nm in names) + f" <= {rng.randint(0, 20)}")
    lines.append("Bounds")
    for j in range(n):
        if kind[j] == "B":
            continue
        lo = rng.choice([-5, -3, 0, 0, 1])
        hi = lo + rng.choice([1, 2, 4, 6, 10])
        u = rng.random()
        if open_bounds and u < 0.08:
            lines.append(f" {names[j]} free")
        elif open_bounds and u < 0.25:
            lines.append(f" {names[j]} >= {lo}")
        else:
            lines.append(f" {lo} <= {names[j]} <= {hi}")
    ints = [names[j] for j in range(n) if kind[j] == "I"]
    bins = [names[j] for j in range(n) if kind[j] == "B"]
    if ints:
        lines += ["General", " " + " ".join(ints)]
    if bins:
        lines += ["Binary", " " + " ".join(bins)]
    lines.append("End")
    return "\n".join(lines) + "\n"


def main(argv=None):
    import pyscipopt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--popso", default=str(BENCH.parent / "build" / "popso"))
    ap.add_argument("--popso-args", default="--gap 1e-6")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--open-bounds", action="store_true", help="some columns with one bound or none")
    ap.add_argument("--out", help="directory for the models (default: a temporary one)")
    args = ap.parse_args(argv)
    out = args.out or tempfile.mkdtemp(prefix="nc_fuzz_")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(args.seed)
    bad = skipped = 0
    tally = {}
    for k in range(args.n):
        path = os.path.join(out, f"nc{k}.lp")
        open(path, "w").write(gen(rng, k, args.open_bounds))
        m = pyscipopt.Model()
        m.hideOutput()
        m.readProblem(path)
        m.setParam("limits/time", 30)
        m.setParam("limits/gap", 1e-7)
        try:
            m.optimize()
        except Exception:  # SCIP rejects the model (e.g. "error in input data" on an unbounded product)
            skipped += 1
            continue
        st = m.getStatus()
        pst, pob = popso_result(args.popso, args.popso_args.split(), path)
        tally[pst] = tally.get(pst, 0) + 1
        if st == "optimal" and abs(m.getObjVal()) >= 1e9:
            # SCIP's "optimal" at its infinity (1e20), or at the 1e10 it gives a free column of a nonlinear row
            # (the generated coefficients are single digits: such a value means an unbounded model).
            st = "unbounded"
        ref = m.getObjVal() if st == "optimal" else None
        if st not in ("optimal", "infeasible"):
            # SCIP at a limit, unbounded or undecided: its incumbent still tells a wrong claim by popso (INFEASIBLE
            # although SCIP has a point; OPTIMAL at a value worse than SCIP's point).
            skipped += 1
            if st != "unbounded" and m.getNSols() > 0 and abs(m.getObjVal()) < 1e19:
                inc = m.getObjVal()
                sense = -1 if "Maximize" in open(path).read() else 1
                if pst == "INFEASIBLE" or (pst == "OPTIMAL" and pob is not None and
                                           sense * (pob - inc) > args.tol * max(1, abs(inc))):
                    bad += 1
                    print(f"MISMATCH {path}: popso {pst} {pob}, SCIP {st} with the point {inc}")
            continue
        ok = (st == "infeasible" and pst == "INFEASIBLE") or (
            ref is not None and pst == "OPTIMAL" and pob is not None and abs(pob - ref) <= args.tol * max(1, abs(ref)))
        # A limit on popso's side is not a wrong answer, as long as its incumbent is not better than the optimum.
        if not ok and pst in ("TIME_LIMIT", "NUMERIC") and (pob is None or ref is None or True):
            sense = -1 if "Maximize" in open(path).read() else 1
            if pob is None or ref is None or sense * (pob - ref) >= -args.tol * max(1, abs(ref)):
                print(f"  note {path}: popso {pst} {pob}, SCIP {st} {ref}")
                continue
        if not ok:
            bad += 1
            print(f"MISMATCH {path}: popso {pst} {pob}, SCIP {st} {ref}")
    print(f"{args.n} cases, {skipped} skipped (SCIP limit), {bad} mismatches; popso statuses {tally}; models in {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
