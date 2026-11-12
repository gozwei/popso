#!/bin/bash
# Mirrors .github/workflows/ci.yml on Linux: gcc 15 / clang 21 Release -Werror builds with the tests and the tiny
# end-to-end runs (both modes), and the clang ASan + UBSan build with the unit and C API tests. Run it on the
# benchmark server before tagging a release: tools/remote_bench.sh -d -c HEAD -n ci<ver> -- bash tools/ci_linux.sh
export ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
for cfg in "gcc g++-15" "clang clang++-21"; do
  set -- $cfg
  echo "=== $1"
  cmake -S . -B ci-$1 -DCMAKE_BUILD_TYPE=Release -DPOPSO_BUILD_TESTS=ON -DPOPSO_WERROR=ON -DCMAKE_CXX_COMPILER=$2 > ci-$1.cfg.log 2>&1 || { echo "configure failed"; tail -20 ci-$1.cfg.log; continue; }
  nice cmake --build ci-$1 --parallel 6 > ci-$1.build.log 2>&1 || { echo "build failed"; grep -E 'error|warning' ci-$1.build.log | head -20; continue; }
  ctest --test-dir ci-$1 --output-on-failure 2>&1 | tail -4
  python benchmarks/run.py --solver ci-$1/popso --sizes tiny -t 20 --threads 2 --mem-limit 1000 --no-save --strict 2>&1 | grep -E '^instances|WRONG'
  python benchmarks/run.py --solver "ci-$1/popso --deterministic" --sizes tiny -t 20 --threads 2 --mem-limit 1000 --no-save --strict 2>&1 | grep -E '^instances|WRONG'
done
echo "=== asan"
cmake -S . -B ci-asan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPOPSO_BUILD_TESTS=ON -DPOPSO_SANITIZE=ON -DPOPSO_WERROR=ON -DCMAKE_CXX_COMPILER=clang++-21 -DCMAKE_C_COMPILER=clang-21 > ci-asan.cfg.log 2>&1 || { echo "configure failed"; tail -20 ci-asan.cfg.log; exit 1; }
nice cmake --build ci-asan --parallel 6 > ci-asan.build.log 2>&1 || { echo "build failed"; grep -E 'error|warning' ci-asan.build.log | head -20; exit 1; }
ctest --test-dir ci-asan --output-on-failure -E '^python$' 2>&1 | tail -6
