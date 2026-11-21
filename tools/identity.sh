#!/bin/bash
# The gate's identity counts alone (deterministic node / LP iteration counts), for a quick check of a change
# that must not alter the search:  python3 tools/bench_slot.py -- tools/identity.sh [build-dir]
# Expected: cflp_medium_1 3068 / 129049, gap_medium_1 3439 / 96982, mas74 2103 / 21902.
cd "$(dirname "$0")/.." || exit 1
B=${1:-build}
for i in cflp_medium_1 gap_medium_1; do
  echo "$i: $("$B"/popso --threads 2 --deterministic --mem-limit 2048 benchmarks/instances/$i.mps | grep -E '^Nodes|^LP iter' | tr '\n' ' ')"
done
echo "mas74: $("$B"/popso --threads 2 --deterministic --mem-limit 2048 --node-limit 2000 benchmarks/miplib/mas74.mps | grep -E '^Nodes|^LP iter' | tr '\n' ' ')"
