// Tests for the interior-point method, the crossover and the barrier driver.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

#include "milp/io/mps_reader.hpp"
#include "milp/lp/barrier.hpp"
#include "milp/lp/crossover.hpp"
#include "milp/lp/simplex.hpp"
#include "milp/solver/solver.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

struct Lp {
  SparseMatrix A;
  std::vector<double> c, cl, cu, rl, ru;
};

Lp make_lp(Index m, Index n, const std::vector<std::vector<double>>& rows, std::vector<double> c, std::vector<double> cl,
           std::vector<double> cu, std::vector<double> rl, std::vector<double> ru) {
  Lp lp;
  std::vector<Index> J, I;
  std::vector<double> V;
  for (Index i = 0; i < m; ++i)
    for (Index j = 0; j < n; ++j)
      if (rows[i][j] != 0.0) J.push_back(j), I.push_back(i), V.push_back(rows[i][j]);
  lp.A = SparseMatrix::from_triplets(n, m, J, I, V);
  lp.c = std::move(c);
  lp.cl = std::move(cl);
  lp.cu = std::move(cu);
  lp.rl = std::move(rl);
  lp.ru = std::move(ru);
  return lp;
}

// Random LP with a known feasible point; mixed bound types (free, one-sided, boxed) and row
// types (equality, ranges, one-sided). Costs are made dual feasible for the free directions by
// bounding most columns, so most instances have an optimum.
Lp random_lp(std::mt19937& rng, Index m, Index n, double density) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Lp lp;
  std::vector<Index> J, I;
  std::vector<double> V;
  std::vector<double> x0(static_cast<size_t>(n));
  for (Index j = 0; j < n; ++j) {
    x0[j] = 3.0 * u(rng);
    const int kind = static_cast<int>(rng() % 10);
    double lo = -kInf, hi = kInf;
    if (kind < 4) lo = x0[j] - std::abs(u(rng)) * 2, hi = x0[j] + std::abs(u(rng)) * 2;  // boxed
    else if (kind < 7) lo = x0[j] - std::abs(u(rng));                                    // lower
    else if (kind < 9) hi = x0[j] + std::abs(u(rng));                                    // upper
    lp.cl.push_back(lo);
    lp.cu.push_back(hi);
    // Cost signs that keep one-sided columns bounded in the objective direction.
    double c = u(rng);
    if (kind >= 4 && kind < 7) c = std::abs(c) + 0.1;
    else if (kind >= 7 && kind < 9) c = -std::abs(c) - 0.1;
    else if (kind == 9) c = 0.0;
    lp.c.push_back(c);
    bool any = false;
    for (Index i = 0; i < m; ++i) {
      if (static_cast<double>(rng() % 1000) < 1000.0 * density) {
        J.push_back(j), I.push_back(i), V.push_back(std::round(10 * u(rng)) / 2 + 0.25);
        any = true;
      }
    }
    if (!any && m > 0) J.push_back(j), I.push_back(static_cast<Index>(rng() % m)), V.push_back(1.0);
  }
  lp.A = SparseMatrix::from_triplets(n, m, J, I, V);
  std::vector<double> act(static_cast<size_t>(m), 0.0);
  for (Index j = 0; j < n; ++j)
    for (Index k = lp.A.begin(j); k < lp.A.end(j); ++k) act[lp.A.index[k]] += lp.A.value[k] * x0[j];
  for (Index i = 0; i < m; ++i) {
    const int kind = static_cast<int>(rng() % 4);
    if (kind == 0) lp.rl.push_back(act[i]), lp.ru.push_back(act[i]);
    else if (kind == 1) lp.rl.push_back(act[i] - std::abs(u(rng))), lp.ru.push_back(act[i] + std::abs(u(rng)));
    else if (kind == 2) lp.rl.push_back(act[i] - std::abs(u(rng))), lp.ru.push_back(kInf);
    else lp.rl.push_back(-kInf), lp.ru.push_back(act[i] + std::abs(u(rng)));
  }
  return lp;
}

LpStatus simplex_solve(const Lp& lp, double* obj) {
  DualSimplex s;
  s.load(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru);
  LpOptions o;
  const LpStatus st = s.solve(o);
  *obj = s.objective();
  return st;
}

LpStatus barrier(const Lp& lp, double* obj, BarrierStats* stats) {
  DualSimplex s;
  s.load(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru);
  BarrierOptions o;
  const LpStatus st = barrier_solve(s, lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru, o, stats);
  *obj = s.objective();
  return st;
}

bool close(double a, double b) { return std::abs(a - b) <= 1e-7 * (1.0 + std::abs(b)); }

}  // namespace

TEST(ipm_small_lp) {
  // max x + y s.t. x + 2y <= 4, 3x + y <= 6, x, y >= 0  ->  x = 1.6, y = 1.2, obj 2.8.
  Lp lp = make_lp(2, 2, {{1, 2}, {3, 1}}, {-1, -1}, {0, 0}, {kInf, kInf}, {-kInf, -kInf}, {4, 6});
  IpmOptions o;
  IpmResult r = ipm_solve(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru, o);
  REQUIRE(r.status == IpmStatus::Optimal);
  CHECK_NEAR(r.objective, -2.8, 1e-6);
  CHECK_NEAR(r.x[0], 1.6, 1e-6);
  CHECK_NEAR(r.x[1], 1.2, 1e-6);
  // Duals: d = c - A^T y with y <= 0 for <= rows in a minimization.
  CHECK(r.primal_infeas < 1e-7);
  CHECK(r.dual_infeas < 1e-7);
}

TEST(ipm_free_variables_ranges_and_equalities) {
  // min x0 - x1 + 0 x2, x0 free, x1 in [-1, 1], x2 >= 0
  //   x0 + x1 + x2 = 2;  -1 <= x0 - x1 <= 1 (range).
  Lp lp = make_lp(2, 3, {{1, 1, 1}, {1, -1, 0}}, {1, -1, 0}, {-kInf, -1, 0}, {kInf, 1, kInf}, {2, -1}, {2, 1});
  double ref;
  REQUIRE(simplex_solve(lp, &ref) == LpStatus::Optimal);
  IpmOptions o;
  IpmResult r = ipm_solve(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru, o);
  REQUIRE(r.status == IpmStatus::Optimal);
  CHECK(close(r.objective, ref));
  double obj;
  BarrierStats bs;
  REQUIRE(barrier(lp, &obj, &bs) == LpStatus::Optimal);
  CHECK(bs.used);
  CHECK(close(obj, ref));
}

TEST(ipm_degenerate_lp) {
  // Highly degenerate: many redundant constraints through the optimal vertex (x = y = 1).
  std::vector<std::vector<double>> rows;
  std::vector<double> rl, ru;
  for (int k = 0; k < 12; ++k) {
    const double a = 1.0 + k, b = 1.0 + (k % 3);
    rows.push_back({a, b});
    rl.push_back(-kInf);
    ru.push_back(a + b);
  }
  Lp lp = make_lp(12, 2, rows, {-1, -1}, {0, 0}, {kInf, kInf}, rl, ru);
  double ref, obj;
  REQUIRE(simplex_solve(lp, &ref) == LpStatus::Optimal);
  BarrierStats bs;
  REQUIRE(barrier(lp, &obj, &bs) == LpStatus::Optimal);
  CHECK(close(obj, ref));
  CHECK(close(obj, -2.0));
}

TEST(ipm_infeasible_and_unbounded) {
  {
    // x + y <= 1, x + y >= 3: infeasible.
    Lp lp = make_lp(2, 2, {{1, 1}, {1, 1}}, {1, 1}, {0, 0}, {kInf, kInf}, {-kInf, 3}, {1, kInf});
    IpmOptions o;
    IpmResult r = ipm_solve(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru, o);
    CHECK(r.status != IpmStatus::Optimal);
    double obj;
    BarrierStats bs;
    CHECK(barrier(lp, &obj, &bs) != LpStatus::Optimal);
    CHECK(!bs.used);
  }
  {
    // min -x - y, x - y <= 1, x, y >= 0: unbounded.
    Lp lp = make_lp(1, 2, {{1, -1}}, {-1, -1}, {0, 0}, {kInf, kInf}, {-kInf}, {1});
    IpmOptions o;
    IpmResult r = ipm_solve(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru, o);
    CHECK(r.status != IpmStatus::Optimal);
    double obj;
    BarrierStats bs;
    CHECK(barrier(lp, &obj, &bs) != LpStatus::Optimal);
    CHECK(!bs.used);
  }
  {
    // Empty row with an impossible right-hand side (after a fixed column): infeasible at once.
    Lp lp = make_lp(1, 2, {{1, 0}}, {1, 1}, {2, 0}, {2, 1}, {3}, {3});
    IpmOptions o;
    IpmResult r = ipm_solve(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru, o);
    CHECK(r.status == IpmStatus::PrimalInfeasible);
  }
}

TEST(ipm_random_lps_match_simplex) {
  std::mt19937 rng(2024);
  int optimal = 0;
  for (int t = 0; t < 120; ++t) {
    const Index m = 1 + static_cast<Index>(rng() % 40), n = 1 + static_cast<Index>(rng() % 60);
    Lp lp = random_lp(rng, m, n, 0.05 + 0.3 * (rng() % 100) / 100.0);
    double ref;
    const LpStatus sref = simplex_solve(lp, &ref);
    double obj;
    BarrierStats bs;
    const LpStatus st = barrier(lp, &obj, &bs);
    if (sref == LpStatus::Optimal) {
      ++optimal;
      // The IPM must reach the optimum (the instances are well scaled) and the crossover must
      // give an optimal basis with the simplex objective.
      if (bs.ipm.status != IpmStatus::Optimal) std::fprintf(stderr, "case %d m %d n %d: %s after %d it, pinf %.2g gap %.2g\n", t, m, n, to_string(bs.ipm.status), bs.ipm.iterations, bs.ipm.primal_infeas, bs.ipm.rel_gap);
      CHECK(bs.ipm.status == IpmStatus::Optimal);
      CHECK(st == LpStatus::Optimal);
      CHECK(close(obj, ref));
      if (bs.ipm.status == IpmStatus::Optimal) CHECK(close(bs.ipm.objective, ref));
    } else {
      CHECK(st != LpStatus::Optimal || !bs.used);
    }
  }
  CHECK(optimal > 60);
}

TEST(crossover_basis_is_optimal_basis) {
  // The crossover basis alone (before cleanup) must be a valid basis: m basic variables, and
  // the simplex from it must agree with a cold solve.
  std::mt19937 rng(77);
  for (int t = 0; t < 30; ++t) {
    const Index m = 5 + static_cast<Index>(rng() % 30), n = 5 + static_cast<Index>(rng() % 40);
    Lp lp = random_lp(rng, m, n, 0.2);
    double ref;
    if (simplex_solve(lp, &ref) != LpStatus::Optimal) continue;
    IpmOptions o;
    IpmResult r = ipm_solve(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru, o);
    REQUIRE(r.status == IpmStatus::Optimal);
    CrossoverStats cs;
    Basis b = crossover(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru, r.x, r.y, &cs);
    REQUIRE(static_cast<Index>(b.status.size()) == m + n);
    Index basic = 0;
    for (VarStatus s : b.status) basic += s == VarStatus::Basic;
    CHECK(basic == m);
    DualSimplex s;
    s.load(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru);
    s.set_basis(b);
    LpOptions lo;
    lo.primal_first = true;
    lo.perturb = false;
    REQUIRE(s.solve(lo) == LpStatus::Optimal);
    CHECK(close(s.objective(), ref));
  }
}

TEST(barrier_lp_method_on_benchmark_lps) {
  // Every pure LP of the benchmark set (lpsanity, transport, random LPs up to medium) gives the
  // same status and objective with --lp-method barrier as with the simplex, with and without
  // presolve (without presolve the barrier also sees fixed columns, free rows and empty rows).
  const std::string dir = std::string(MILP_SOURCE_DIR) + "/benchmarks/instances";
  int checked = 0;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    const std::string name = entry.path().filename().string();
    const bool lp = (name.rfind("lpsanity_", 0) == 0 && name.find("_mip_") == std::string::npos) ||
                    name.rfind("transport_", 0) == 0 || name.rfind("transinf_", 0) == 0 ||
                    name == "randlp_small_1.mps" || name == "randlp_medium_1.mps";
    if (!lp || entry.path().extension() != ".mps") continue;
    std::vector<std::string> warnings;
    const Model model = read_mps(entry.path().string(), &warnings);
    for (int presolve = 0; presolve < 2; ++presolve) {
      SolverParams p;
      p.verbosity = 0;
      p.threads = 1;
      p.presolve = presolve != 0;
      p.lp_method = LpMethod::Simplex;
      const SolveResult rs = solve(model, p);
      p.lp_method = LpMethod::Barrier;
      const SolveResult rb = solve(model, p);
      const bool same = rs.status == rb.status &&
                        (!rs.has_solution || std::abs(rs.objective - rb.objective) <= 1e-7 * (1.0 + std::abs(rs.objective)));
      if (!same) {
        std::fprintf(stderr, "  %s (presolve %d): simplex %s %.10g, barrier %s %.10g\n", name.c_str(), presolve,
                     to_string(rs.status), rs.objective, to_string(rb.status), rb.objective);
      }
      CHECK(same);
      if (rb.has_solution) CHECK(model.objective_value(rb.x) == rb.objective);
      ++checked;
    }
  }
  CHECK(checked >= 40);
}
