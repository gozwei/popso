#!/usr/bin/env python3
"""Generate the benchmark instance set into benchmarks/instances/.

    .venv/bin/python benchmarks/generators/generate.py            # all
    .venv/bin/python benchmarks/generators/generate.py --only tsp,mkp
    .venv/bin/python benchmarks/generators/generate.py --extra    # also the
                      # 'xhard' instances that are too big/slow to commit

Instances are named <family>_<size>_<seed>.mps.  Generation is
deterministic: the RNG seed is derived from (family, size, seed) only.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from generators import families, lp_sanity  # noqa: E402

# (family, size, seed, params). Sizes: tiny < small < medium < large < hard.
# 'xhard' entries are only generated with --extra and are not committed.
CONFIG = [
    # ---- pure binary combinatorial
    ("knapsack", "tiny", 1, dict(n=10, kind="uncorr")),
    ("knapsack", "small", 1, dict(n=50, kind="weak")),
    ("knapsack", "medium", 1, dict(n=200, kind="strong")),
    ("knapsack", "large", 1, dict(n=1000, kind="strong", R=10000)),
    ("mkp", "tiny", 1, dict(n=10, k=2)),
    ("mkp", "small", 1, dict(n=30, k=5)),
    ("mkp", "medium", 1, dict(n=60, k=5)),
    ("mkp", "large", 1, dict(n=100, k=5)),
    ("mkp", "hard", 1, dict(n=100, k=10)),
    ("setcover", "tiny", 1, dict(rows=10, cols=20, density=0.3)),
    ("setcover", "small", 1, dict(rows=50, cols=100, density=0.1)),
    ("setcover", "medium", 1, dict(rows=200, cols=500, density=0.05)),
    ("setcover", "large", 1, dict(rows=400, cols=1000, density=0.03)),
    ("setcover", "hard", 1, dict(rows=500, cols=1000, density=0.02)),
    ("setpack", "tiny", 1, dict(rows=10, cols=20, density=0.3)),
    ("setpack", "small", 1, dict(rows=50, cols=100, density=0.1)),
    ("setpack", "medium", 1, dict(rows=100, cols=200, density=0.05)),
    ("setpack", "large", 1, dict(rows=200, cols=400, density=0.03)),
    ("setpack", "hard", 1, dict(rows=250, cols=500, density=0.025)),
    ("binpack", "tiny", 1, dict(n=6)),
    ("binpack", "small", 1, dict(n=12)),
    ("binpack", "medium", 1, dict(n=25)),
    ("binpack", "large", 1, dict(n=40)),
    ("binpack", "hard", 1, dict(n=45)),
    ("gap", "tiny", 1, dict(agents=3, jobs=8)),
    ("gap", "small", 1, dict(agents=5, jobs=20)),
    ("gap", "medium", 1, dict(agents=5, jobs=40)),
    ("gap", "large", 1, dict(agents=10, jobs=60)),
    ("gap", "hard", 1, dict(agents=10, jobs=70)),
    # ---- routing / network
    ("cflp", "tiny", 1, dict(fac=4, cust=8)),
    ("cflp", "small", 1, dict(fac=10, cust=30)),
    ("cflp", "medium", 1, dict(fac=20, cust=60)),
    ("cflp", "large", 1, dict(fac=40, cust=100)),
    ("cflp", "hard", 1, dict(fac=50, cust=120)),
    ("netdesign", "tiny", 1, dict(nodes=5, extra_arcs=4, commodities=3)),
    ("netdesign", "small", 1, dict(nodes=8, extra_arcs=10, commodities=8)),
    ("netdesign", "medium", 1, dict(nodes=12, extra_arcs=25, commodities=15)),
    ("netdesign", "large", 1, dict(nodes=20, extra_arcs=60, commodities=40)),
    ("netdesign", "hard", 1, dict(nodes=25, extra_arcs=80, commodities=50)),
    ("tsp", "tiny", 1, dict(n=5)),
    ("tsp", "small", 1, dict(n=10)),
    ("tsp", "medium", 1, dict(n=20)),
    ("tsp", "large", 1, dict(n=35)),
    ("tsp", "hard", 1, dict(n=50)),
    ("cvrp", "tiny", 1, dict(n=6)),
    ("cvrp", "small", 1, dict(n=8)),
    ("cvrp", "medium", 1, dict(n=10)),
    ("cvrp", "large", 1, dict(n=12)),
    # ---- scheduling / lot sizing
    ("lotsize", "tiny", 1, dict(items=2, periods=4)),
    ("lotsize", "small", 1, dict(items=4, periods=8)),
    ("lotsize", "medium", 1, dict(items=8, periods=12)),
    ("lotsize", "large", 1, dict(items=10, periods=15)),
    ("lotsize", "hard", 1, dict(items=12, periods=20)),
    ("jobshop", "tiny", 1, dict(jobs=3, machines=3)),
    ("jobshop", "small", 1, dict(jobs=4, machines=4)),
    ("jobshop", "medium", 1, dict(jobs=6, machines=6)),
    ("jobshop", "large", 1, dict(jobs=8, machines=8)),
    ("jobshop", "hard", 1, dict(jobs=10, machines=5)),
    ("timeidx", "tiny", 1, dict(jobs=4, pmax=5)),
    ("timeidx", "small", 1, dict(jobs=10, pmax=8)),
    ("timeidx", "medium", 1, dict(jobs=20, machines=2, pmax=15)),
    ("timeidx", "large", 1, dict(jobs=40, machines=3, pmax=20)),
    ("timeidx", "hard", 1, dict(jobs=60, machines=4, pmax=25)),
    # ---- general mixed integer
    ("randmilp", "tiny", 1, dict(cols=10, rows=6, density=0.4)),
    ("randmilp", "small", 1, dict(cols=40, rows=25, density=0.15)),
    ("randmilp", "medium", 1, dict(cols=100, rows=60, density=0.06)),
    ("randmilp", "large", 1, dict(cols=150, rows=100, density=0.05)),
    ("randmilp", "hard", 1, dict(cols=200, rows=120, density=0.04, frac_eq=0.05)),
    ("unbdmilp", "small", 1, dict(cols=30, rows=20, density=0.2)),
    ("unbdmilp", "medium", 1, dict(cols=150, rows=100, density=0.05)),
    # ---- infeasible MILPs
    ("pigeon", "tiny", 1, dict(holes=4)),
    ("pigeon", "small", 1, dict(holes=7)),
    ("pigeon", "medium", 1, dict(holes=10)),
    ("pigeon", "large", 1, dict(holes=14)),
    ("subsetsum", "small", 1, dict(n=12, lo=1000, hi=2000)),
    ("subsetsum", "medium", 1, dict(n=22, lo=10000, hi=20000, side_rows=3)),
    ("transinf", "small", 1, dict(sources=5, sinks=8)),
    ("transinf", "medium", 1, dict(sources=30, sinks=50)),
    # ---- pure LPs (LP engine performance)
    ("transport", "small", 1, dict(sources=10, sinks=20)),
    ("transport", "medium", 1, dict(sources=50, sinks=100)),
    ("transport", "large", 1, dict(sources=150, sinks=300)),
    ("randlp", "small", 1, dict(cols=50, rows=30, density=0.1)),
    ("randlp", "medium", 1, dict(cols=1000, rows=600, density=0.01)),
    ("randlp", "large", 1, dict(cols=8000, rows=5000, density=0.0015)),
    # ---- logical constraints (semi-continuous, SOS, indicators)
    ("semilot", "tiny", 1, dict(n=8, m=3)),
    ("semilot", "small", 1, dict(n=25, m=5)),
    ("semilot", "medium", 1, dict(n=60, m=10)),
    ("semilot", "large", 1, dict(n=150, m=20)),
    ("pwlnet", "tiny", 1, dict(sources=3, sinks=4)),
    ("pwlnet", "small", 1, dict(sources=5, sinks=10)),
    ("pwlnet", "medium", 1, dict(sources=8, sinks=20)),
    ("pwlnet", "large", 1, dict(sources=12, sinks=40, K=6)),
    ("indsched", "tiny", 1, dict(n=4)),
    ("indsched", "small", 1, dict(n=6)),
    ("indsched", "medium", 1, dict(n=8)),
    ("indsched", "large", 1, dict(n=10)),
]


def instance_name(family, size, seed):
    return f"{family}_{size}_{seed}"


def rng_for(family, size, seed):
    h = hashlib.sha256(f"{family}/{size}/{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(BENCH / "instances"))
    ap.add_argument("--only", help="comma-separated families (lpsanity for hand-written)")
    ap.add_argument("--sizes", help="comma-separated sizes to generate")
    ap.add_argument("--extra", action="store_true", help="also generate xhard instances")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    sizes = set(args.sizes.split(",")) if args.sizes else None

    if only is None or "lpsanity" in only:
        for p in lp_sanity.write_all(out):
            print(f"wrote {p.name}")

    for family, size, seed, params in CONFIG:
        if only is not None and family not in only:
            continue
        if sizes is not None and size not in sizes:
            continue
        if size == "xhard" and not args.extra:
            continue
        name = instance_name(family, size, seed)
        t0 = time.time()
        m = families.BUILDERS[family](rng_for(family, size, seed), **params)
        m.name = name
        out_file = out / f"{name}.mps"
        m.write(out_file)
        print(f"wrote {out_file.name}: {m.num_vars} vars, {m.num_constrs} cons, "
              f"{m.num_nonzeros} nnz, {m.num_int_vars} int ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
