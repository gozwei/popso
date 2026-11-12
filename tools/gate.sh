#!/bin/bash
# Development gates, run before every commit that touches the solver (Mac: inside one bench slot):
#   python3 tools/bench_slot.py -- bash tools/gate.sh [build-dir]
# 1. ctest (unit, CLI tooling, C API, Python);
# 2. identity: deterministic node / LP iteration counts that must not change unless a change is meant to alter
#    the search (Mac, 2 threads): cflp_medium_1 2100 / 124833, gap_medium_1 2205 / 72936, mas74 (2000 nodes) 2062 / 21826;
# 3. the gate instances (119, medium and smaller) in both modes: solved 119 / 119, WRONG 0;
# 4. the quadratic families in both modes (32 / 33 default, 31 / 33 deterministic; qcrand_hard_1 / portqc_hard_1 open).
cd "$(dirname "$0")/.." || exit 1
B=${1:-build}
(cd "$B" && ctest -j 2 2>&1 | grep -E "tests passed|Failed|\*\*\*")
for i in cflp_medium_1 gap_medium_1; do
  echo "$i: $("$B"/popso --threads 2 --deterministic benchmarks/instances/$i.mps | grep -E '^Nodes|^LP iter' | tr '\n' ' ')"
done
echo "mas74: $("$B"/popso --threads 2 --deterministic --node-limit 2000 benchmarks/miplib/mas74.mps | grep -E '^Nodes|^LP iter' | tr '\n' ' ')"
for mode in "" "--deterministic"; do
  .venv/bin/python benchmarks/run.py --solver "$B/popso $mode" --max-size medium -t 60 --threads 2 --mem-limit 2048 2>&1 |
    grep -E '^instances' | sed "s/^/gates ${mode:-default}: /"
  .venv/bin/python benchmarks/run.py --solver "$B/popso $mode" --families portqc,qcrand,portfolio,lsq,miqpport,miqprand,sqtrack \
    -t 60 --threads 2 --mem-limit 2048 2>&1 | grep -E '^instances' | sed "s/^/quadratic ${mode:-default}: /"
done
