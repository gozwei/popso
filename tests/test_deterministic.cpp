// Deterministic parallel mode: repeated solves with the same seed and thread count must give
// identical results (node count, LP iterations, objective), also when other threads compete for
// the CPU (the round scheduling, sub-MIP jobs and FJ synchronisation must not depend on timing).

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "milp/core/model.hpp"
#include "milp/core/log.hpp"
#include "milp/core/timer.hpp"
#include "milp/search/branch_and_bound.hpp"
#include "milp/search/mip_problem.hpp"
#include "milp/solver/solver.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

// Multi-dimensional knapsack (maximize): small, but needs a few thousand nodes.
Model random_mkp(std::mt19937& rng, int n, int m) {
  ModelBuilder b;
  b.model().sense = ObjSense::Maximize;
  // Plain modulo instead of std::uniform_int_distribution: the distributions are implementation-
  // defined, the mt19937 sequence is not, so every standard library builds the same instances.
  auto w = [](std::mt19937& g) { return 10 + static_cast<int>(g() % 51); };
  auto noise = [](std::mt19937& g) { return static_cast<int>(g() % 11); };
  std::vector<std::vector<int>> a(m, std::vector<int>(n));
  std::vector<double> sum(m, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) sum[i] += a[i][j] = w(rng);
  }
  // Values correlated with the weights (hard instances).
  for (int j = 0; j < n; ++j) {
    double v = noise(rng);
    for (int i = 0; i < m; ++i) v += a[i][j] / static_cast<double>(m);
    b.add_col("x" + std::to_string(j), std::round(v), 0.0, 1.0, VarType::Binary);
  }
  for (int i = 0; i < m; ++i) {
    const Index row = b.add_row("c" + std::to_string(i), -kInf, std::floor(0.5 * sum[i]));
    for (int j = 0; j < n; ++j) b.add_coef(row, j, a[i][j]);
  }
  return b.build();
}

struct Run {
  Status status;
  double obj;
  long long nodes, iters;
};

Run solve_det(const Model& m, int threads, unsigned seed) {
  SolverParams sp;
  sp.verbosity = 0;
  sp.threads = threads;
  sp.deterministic = true;
  sp.seed = seed;
  sp.mip_rel_gap = 0.0;
  sp.mip_abs_gap = 1e-9;
  sp.node_limit = 200000;
  const SolveResult r = solve(m, sp);
  return {r.status, r.has_solution ? r.objective : kInf, r.stats.nodes, r.stats.lp_iterations};
}

}  // namespace

TEST(deterministic_mode_reproducible) {
  std::mt19937 rng(2024);
  int multi = 0;
  for (int t = 0; t < 4; ++t) {
    const Model m = random_mkp(rng, 40 + 10 * t, 3);
    const int threads = 2 + t % 3;
    const Run a = solve_det(m, threads, 1 + t);
    // Second run with the CPU contended by busy threads (other timings, same result).
    std::atomic<bool> stop{false};
    std::vector<std::thread> burn;
    for (int k = 0; k < 2; ++k) {
      burn.emplace_back([&stop] {
        volatile double x = 1.0;
        while (!stop.load(std::memory_order_relaxed)) x = x * 1.0000001 + 1e-9;
      });
    }
    const Run b = solve_det(m, threads, 1 + t);
    stop = true;
    for (auto& th : burn) th.join();
    const Run c = solve_det(m, threads, 1 + t);
    CHECK(a.status == Status::Optimal);
    CHECK(a.status == b.status && a.status == c.status);
    CHECK_EQ(a.nodes, b.nodes);
    CHECK_EQ(a.nodes, c.nodes);
    CHECK_EQ(a.iters, b.iters);
    CHECK_EQ(a.iters, c.iters);
    CHECK_EQ(a.obj, b.obj);
    CHECK_EQ(a.obj, c.obj);
    multi += a.nodes > 300;  // ~350-1900 nodes on these seeds
    if (std::getenv("MILP_TEST_VERBOSE")) std::printf("  mkp %d: %lld nodes, %lld iterations\n", t, a.nodes, a.iters);
  }
  CHECK(multi >= 2);  // the tree search (rounds, jobs) was exercised
}

// Memory limit: a search whose tree outgrows a tiny cap drops stored bases, drains the tree depth
// first and still proves the optimum (same objective as without the cap), in both modes; the
// deterministic one stays reproducible.
TEST(memory_limit_drains_tree) {
  std::mt19937 rng(8);  // an instance of ~1300 nodes
  const Model m = random_mkp(rng, 90, 4);
  MipProblem p;
  CHECK(MipProblem::from_model(m, &p));
  Logger log(std::getenv("MILP_TEST_VERBOSE") ? 2 : 0);
  auto run = [&](long long cap, int threads, bool det) {
    SolverParams sp;
    sp.verbosity = log.verbosity();
    sp.threads = threads;
    sp.deterministic = det;
    sp.seed = 3;
    sp.mip_rel_gap = 0.0;
    sp.mip_abs_gap = 1e-9;
    sp.node_limit = 2000000;
    SearchOptions opts;
    opts.check_resident = false;  // the cap applies to the tree memory alone
    opts.mem_limit_bytes = cap;
    Timer timer;
    return branch_and_bound(p, sp, log, timer, opts);
  };
  static const long long cap = std::getenv("MILP_TEST_MEMCAP") ? std::atoll(std::getenv("MILP_TEST_MEMCAP")) : 200000;
  const MipResult ref = run(-1, 1, false);
  CHECK(ref.status == Status::Optimal);
  const MipResult a = run(cap, 1, false);
  CHECK(a.status == Status::Optimal);
  CHECK(std::abs(a.objective - ref.objective) <= 1e-6);
  const MipResult b = run(cap, 2, true), c = run(cap, 2, true);
  if (b.status != Status::Optimal) std::printf("  b: status %d, %lld nodes, obj %g\n", (int)b.status, b.nodes, b.objective);
  CHECK(b.status == Status::Optimal);
  CHECK(std::abs(b.objective - ref.objective) <= 1e-6);
  CHECK_EQ(b.nodes, c.nodes);
  CHECK_EQ(b.lp_iterations, c.lp_iterations);
  if (std::getenv("MILP_TEST_VERBOSE")) {
    std::printf("  mkp: %lld nodes (no cap), %lld / %lld nodes with the cap\n", ref.nodes, a.nodes, b.nodes);
  }
}
