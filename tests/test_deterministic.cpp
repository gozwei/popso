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

namespace {

// More instance types for the memory tests (same construction on every standard library).
// General-integer knapsack (maximize): integers in [0, 3], weights and values correlated.
Model random_gik(std::mt19937& rng, int n, int m) {
  ModelBuilder b;
  b.model().sense = ObjSense::Maximize;
  std::vector<std::vector<int>> a(m, std::vector<int>(n));
  std::vector<double> sum(m, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) sum[i] += a[i][j] = 20 + static_cast<int>(rng() % 61);
  }
  for (int j = 0; j < n; ++j) {
    double v = static_cast<int>(rng() % 9);
    for (int i = 0; i < m; ++i) v += a[i][j] / static_cast<double>(m);
    b.add_col("y" + std::to_string(j), std::round(v), 0.0, 3.0, VarType::Integer);
  }
  for (int i = 0; i < m; ++i) {
    const Index row = b.add_row("k" + std::to_string(i), -kInf, std::floor(0.7 * sum[i]));
    for (int j = 0; j < n; ++j) b.add_coef(row, j, a[i][j]);
  }
  return b.build();
}

// Market split with slacks (minimize, mixed): m equality knapsacks over n binaries, coefficients
// 0..99, right-hand sides half the row sums; continuous slacks in both directions are penalised.
Model random_msplit(std::mt19937& rng, int n, int m) {
  ModelBuilder b;
  for (int j = 0; j < n; ++j) b.add_col("x" + std::to_string(j), 0.0, 0.0, 1.0, VarType::Binary);
  for (int i = 0; i < m; ++i) {
    std::vector<int> a(n);
    long long sum = 0;
    for (int j = 0; j < n; ++j) sum += a[j] = static_cast<int>(rng() % 100);
    const Index sp = b.add_col("sp" + std::to_string(i), 1.0, 0.0, kInf);
    const Index sm = b.add_col("sm" + std::to_string(i), 1.0, 0.0, kInf);
    const Index row = b.add_row("e" + std::to_string(i), static_cast<double>(sum / 2), static_cast<double>(sum / 2));
    for (int j = 0; j < n; ++j) b.add_coef(row, j, a[j]);
    b.add_coef(row, sp, 1.0);
    b.add_coef(row, sm, -1.0);
  }
  return b.build();
}

MipResult solve_capped(const MipProblem& p, Logger& log, long long cap, int threads, bool det, unsigned seed) {
  SolverParams sp;
  sp.verbosity = log.verbosity();
  sp.threads = threads;
  sp.deterministic = det;
  sp.seed = seed;
  sp.mip_rel_gap = 0.0;
  sp.mip_abs_gap = 1e-9;
  sp.node_limit = 2000000;
  SearchOptions opts;
  opts.check_resident = false;  // the cap applies to the tree memory alone
  opts.mem_limit_bytes = cap;
  Timer timer;
  return branch_and_bound(p, sp, log, timer, opts);
}

// A search whose tree outgrows the cap drains it and still proves the optimum: the opportunistic
// 1-thread run and the deterministic 2- and 4-thread runs on every seed (deterministic decisions
// are taken at round barriers only, so a round must not carry the tree past the stop threshold),
// the deterministic ones reproducibly. Returns the number of capped runs that drained.
int check_memory_case(const char* name, const Model& m, long long cap, int seeds) {
  const char* only = std::getenv("MILP_TEST_MEMCASE");  // debugging: run this case only
  if (only && std::string(only) != name) return 1 + 2 * seeds;
  MipProblem p;
  CHECK(MipProblem::from_model(m, &p));
  const bool verbose = std::getenv("MILP_TEST_VERBOSE") != nullptr;
  Logger log(verbose && std::getenv("MILP_TEST_VERBOSE")[0] == '2' ? 2 : 0);
  const MipResult ref = solve_capped(p, log, -1, 1, false, 1);
  CHECK(ref.status == Status::Optimal);
  int drained = 0;
  auto check = [&](const MipResult& r, int threads, bool det, unsigned seed) {
    if (r.status != Status::Optimal || std::abs(r.objective - ref.objective) > 1e-6) {
      std::printf("  %s: %d threads%s seed %u: status %d, %lld nodes, obj %.10g (optimum %.10g)\n", name, threads,
                  det ? " deterministic" : "", seed, static_cast<int>(r.status), r.nodes, r.objective, ref.objective);
    }
    CHECK(r.status == Status::Optimal);
    CHECK(std::abs(r.objective - ref.objective) <= 1e-6);
    drained += r.mem_drains > 0;
  };
  const MipResult a = solve_capped(p, log, cap, 1, false, 1);
  check(a, 1, false, 1);
  long long nodes[2] = {0, 0};
  for (int threads : {2, 4}) {
    for (int s = 1; s <= seeds; ++s) {
      const MipResult b = solve_capped(p, log, cap, threads, true, static_cast<unsigned>(s));
      check(b, threads, true, static_cast<unsigned>(s));
      nodes[threads / 4] += b.nodes;
      if (s == 1) {  // rerun: identical
        const MipResult c = solve_capped(p, log, cap, threads, true, 1u);
        CHECK(c.status == b.status);
        CHECK_EQ(b.nodes, c.nodes);
        CHECK_EQ(b.lp_iterations, c.lp_iterations);
        CHECK_EQ(b.objective, c.objective);
      }
    }
  }
  if (verbose) {
    std::printf("  %s: %lld nodes without the cap; with %lld bytes: %lld (1 thread), mean %lld / %lld (2 / 4 threads"
                " deterministic); %d of %d runs drained\n",
                name, ref.nodes, cap, a.nodes, nodes[0] / seeds, nodes[1] / seeds, drained, 1 + 2 * seeds);
  }
  return drained;
}

}  // namespace

// Memory limit: a search whose tree outgrows a tiny cap drops stored bases, drains the tree depth
// first and still proves the optimum (same objective as without the cap), in both modes; the
// deterministic one stays reproducible. Several seeds and thread counts (the drain must not depend
// on a lucky seed) and instance types.
TEST(memory_limit_drains_tree) {
  static const long long cap = std::getenv("MILP_TEST_MEMCAP") ? std::atoll(std::getenv("MILP_TEST_MEMCAP")) : 0;
  static const int seeds = std::getenv("MILP_TEST_MEMSEEDS") ? std::atoi(std::getenv("MILP_TEST_MEMSEEDS")) : 6;
  {
    std::mt19937 rng(8);  // ~1300-1700 nodes
    const int d = check_memory_case("mkp", random_mkp(rng, 90, 4), cap > 0 ? cap : 100000, seeds);
    CHECK(d >= 1 + seeds);  // the cap binds
  }
  {
    std::mt19937 rng(5);
    const int d = check_memory_case("gik", random_gik(rng, 60, 3), cap > 0 ? cap : 100000, seeds);
    CHECK(d >= 1 + seeds);
  }
  {
    std::mt19937 rng(3);
    const int d = check_memory_case("msplit", random_msplit(rng, 24, 2), cap > 0 ? cap : 60000, seeds);
    CHECK(d >= 1 + seeds);
  }
}
