// Tests for LP folding (lp/lp_fold.hpp): the folded LP has the optimum of the original, and its expanded primal
// and dual solution is optimal for the original.

#include <cmath>
#include <random>

#include "milp/lp/lp_fold.hpp"
#include "milp/lp/simplex.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

struct Lp {
  SparseMatrix A;
  std::vector<double> c, cl, cu, rl, ru;
  Index n = 0, m = 0;
};

// k copies of a random block (m0 rows, n0 columns) and `links` rows that each sum one block column over the copies.
Lp replicated_lp(std::mt19937& rng, Index m0, Index n0, int k, int links, bool break_symmetry) {
  std::uniform_int_distribution<int> coef(-3, 3), cost(-4, 4);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<std::vector<double>> B(m0, std::vector<double>(n0, 0.0));
  for (auto& r : B)
    for (double& v : r) v = u(rng) < 0.4 ? coef(rng) : 0.0;
  std::vector<double> c0(n0), rhs(m0);
  for (double& v : c0) v = cost(rng);
  for (double& v : rhs) v = 2 + static_cast<int>(rng() % 6);
  Lp lp;
  lp.n = n0 * k, lp.m = m0 * k + links;
  std::vector<Index> J, I;
  std::vector<double> V;
  for (int b = 0; b < k; ++b) {
    for (Index i = 0; i < m0; ++i)
      for (Index j = 0; j < n0; ++j)
        if (B[i][j] != 0.0) J.push_back(b * n0 + j), I.push_back(b * m0 + i), V.push_back(B[i][j]);
  }
  for (int l = 0; l < links; ++l) {
    const Index j = static_cast<Index>(rng() % n0);
    for (int b = 0; b < k; ++b) J.push_back(b * n0 + j), I.push_back(m0 * k + l), V.push_back(1.0);
  }
  lp.A = SparseMatrix::from_triplets(lp.n, lp.m, J, I, V);
  for (int b = 0; b < k; ++b) {
    for (Index j = 0; j < n0; ++j) lp.c.push_back(c0[j]), lp.cl.push_back(0.0), lp.cu.push_back(3.0);
    for (Index i = 0; i < m0; ++i) lp.rl.push_back(-kInf), lp.ru.push_back(rhs[i]);
  }
  for (int l = 0; l < links; ++l) lp.rl.push_back(-kInf), lp.ru.push_back(1.5 * k);
  if (break_symmetry) lp.c[0] += 0.5;  // the first copy differs: still a valid (coarser or finer) partition
  // Row order: block rows of copy b come before the links only if from_triplets keeps row indices (it does).
  std::vector<double> rl2(lp.m), ru2(lp.m);
  for (int b = 0; b < k; ++b)
    for (Index i = 0; i < m0; ++i) rl2[b * m0 + i] = -kInf, ru2[b * m0 + i] = rhs[i];
  for (int l = 0; l < links; ++l) rl2[m0 * k + l] = -kInf, ru2[m0 * k + l] = 1.5 * k;
  lp.rl = rl2, lp.ru = ru2;
  return lp;
}

LpStatus solve(const SparseMatrix& A, const std::vector<double>& c, const std::vector<double>& cl, const std::vector<double>& cu,
               const std::vector<double>& rl, const std::vector<double>& ru, DualSimplex& s) {
  s.load(A, c, cl, cu, rl, ru);
  LpOptions o;
  return s.solve(o);
}

}  // namespace

TEST(lp_fold_replicated_blocks_keep_the_optimum) {
  std::mt19937 rng(11);
  int folded = 0, optimal = 0;
  for (int t = 0; t < 60; ++t) {
    const Index m0 = 2 + static_cast<Index>(rng() % 6), n0 = 3 + static_cast<Index>(rng() % 8);
    const int k = 3 + static_cast<int>(rng() % 5), links = static_cast<int>(rng() % 3);
    const Lp lp = replicated_lp(rng, m0, n0, k, links, t % 4 == 3);
    DualSimplex full;
    const LpStatus sf = solve(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru, full);
    const LpFold f = fold_lp(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru, 0.9);
    if (!f.ok) continue;
    ++folded;
    CHECK(f.num_col_classes < lp.n);
    DualSimplex fs;
    const LpStatus st = solve(f.A, f.cost, f.col_lb, f.col_ub, f.row_lb, f.row_ub, fs);
    CHECK((st == LpStatus::Optimal) == (sf == LpStatus::Optimal));
    if (st != LpStatus::Optimal || sf != LpStatus::Optimal) continue;
    ++optimal;
    CHECK(std::abs(fs.objective() - full.objective()) <= 1e-7 * (1.0 + std::abs(full.objective())));
    // The expanded point: feasible, same objective; the expanded duals: sign-feasible reduced costs.
    std::vector<double> x(lp.n), y(lp.m);
    const std::vector<double> w = fs.row_duals();
    for (Index j = 0; j < lp.n; ++j) x[j] = fs.x()[f.col_class[j]];
    for (Index i = 0; i < lp.m; ++i) y[i] = w[f.row_class[i]] / static_cast<double>(f.row_size[f.row_class[i]]);
    double obj = 0.0;
    std::vector<double> act(lp.m, 0.0), d = lp.c;
    for (Index j = 0; j < lp.n; ++j) {
      obj += lp.c[j] * x[j];
      CHECK(x[j] >= lp.cl[j] - 1e-7 && x[j] <= lp.cu[j] + 1e-7);
      for (Index p = lp.A.begin(j); p < lp.A.end(j); ++p) act[lp.A.index[p]] += lp.A.value[p] * x[j], d[j] -= lp.A.value[p] * y[lp.A.index[p]];
    }
    CHECK(std::abs(obj - full.objective()) <= 1e-7 * (1.0 + std::abs(full.objective())));
    for (Index i = 0; i < lp.m; ++i) CHECK(act[i] >= lp.rl[i] - 1e-7 && act[i] <= lp.ru[i] + 1e-7);
    for (Index j = 0; j < lp.n; ++j) {
      if (x[j] > lp.cl[j] + 1e-6) CHECK(d[j] <= 1e-6);  // not at the lower bound: no positive reduced cost
      if (x[j] < lp.cu[j] - 1e-6) CHECK(d[j] >= -1e-6);
    }
  }
  CHECK(folded > 40);
  CHECK(optimal > 20);
}

TEST(lp_fold_declines_without_symmetry) {
  // Distinct costs: every column is its own class.
  std::mt19937 rng(5);
  Lp lp = replicated_lp(rng, 4, 6, 4, 1, false);
  for (Index j = 0; j < lp.n; ++j) lp.c[j] = 1.0 + 0.01 * j;
  const LpFold f = fold_lp(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru, 0.67);
  CHECK(!f.ok);
}
