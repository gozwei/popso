// Tests for the dual simplex: warm starts, bound changes, row addition/removal.

#include <random>

#include "milp/lp/simplex.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

struct RandomLp {
  SparseMatrix A;
  std::vector<double> c, cl, cu, rl, ru;
};

// Bounded, feasible-ish random LP (all columns boxed so it is never unbounded).
RandomLp random_lp(std::mt19937& rng, Index m, Index n) {
  RandomLp lp;
  std::vector<Index> J, I;
  std::vector<double> V;
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < m; ++i)
      if (rng() % 100 < 30) J.push_back(j), I.push_back(i), V.push_back(static_cast<int>(rng() % 21) - 10);
  lp.A = SparseMatrix::from_triplets(n, m, J, I, V);
  for (Index j = 0; j < n; ++j) {
    lp.c.push_back(static_cast<int>(rng() % 21) - 10);
    lp.cl.push_back(-static_cast<double>(rng() % 5));
    lp.cu.push_back(static_cast<double>(rng() % 5));
  }
  for (Index i = 0; i < m; ++i) {
    lp.rl.push_back(-static_cast<double>(rng() % 30));
    lp.ru.push_back(static_cast<double>(rng() % 30));
  }
  return lp;
}

LpStatus cold_solve(const RandomLp& lp, double* obj) {
  DualSimplex s;
  s.load(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru);
  LpStatus st = s.solve(LpOptions{});
  *obj = s.objective();
  return st;
}

}  // namespace

TEST(lp_warm_start_after_bound_changes) {
  std::mt19937 rng(3);
  for (int t = 0; t < 200; ++t) {
    RandomLp lp = random_lp(rng, 1 + rng() % 20, 1 + rng() % 30);
    DualSimplex s;
    s.load(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru);
    REQUIRE(s.solve(LpOptions{}) == LpStatus::Optimal);
    // Tighten a few bounds and resolve warm; compare with a cold solve.
    for (int k = 0; k < 3; ++k) {
      Index j = rng() % lp.c.size();
      double v = s.x()[j];
      if (rng() % 2) lp.cu[j] = std::max(lp.cl[j], std::floor(v));
      else lp.cl[j] = std::min(lp.cu[j], std::ceil(v));
      s.set_col_bounds(j, lp.cl[j], lp.cu[j]);
    }
    LpStatus warm = s.solve(LpOptions{});
    double cold_obj;
    LpStatus cold = cold_solve(lp, &cold_obj);
    CHECK(warm == cold);
    if (warm == LpStatus::Optimal && cold == LpStatus::Optimal) CHECK_NEAR(s.objective(), cold_obj, 1e-6);
  }
}

TEST(lp_add_and_remove_rows) {
  std::mt19937 rng(5);
  for (int t = 0; t < 200; ++t) {
    const Index m = 1 + rng() % 15, n = 1 + rng() % 25;
    RandomLp lp = random_lp(rng, m, n);
    DualSimplex s;
    s.load(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru);
    REQUIRE(s.solve(LpOptions{}) == LpStatus::Optimal);
    // Add two random rows (as a "cut"), compare with cold solve of the extended LP.
    std::vector<Index> start{0}, idx;
    std::vector<double> val, lb, ub;
    std::vector<Index> J(lp.A.index.size()), I;
    for (Index j = 0; j < n; ++j)
      for (Index p = lp.A.begin(j); p < lp.A.end(j); ++p) J[p] = j;
    I = lp.A.index;
    std::vector<double> V = lp.A.value;
    for (int r = 0; r < 2; ++r) {
      for (Index j = 0; j < n; ++j) {
        if (rng() % 3) continue;
        double a = static_cast<int>(rng() % 11) - 5;
        if (a == 0) continue;
        idx.push_back(j), val.push_back(a);
        J.push_back(j), I.push_back(m + r), V.push_back(a);
      }
      start.push_back(static_cast<Index>(idx.size()));
      double act = 0;
      for (Index k = start[r]; k < start[r + 1]; ++k) act += val[k] * s.x()[idx[k]];
      lb.push_back(-kInf);
      ub.push_back(std::floor(act) - 1.0);  // cuts off the current point
      lp.rl.push_back(lb.back());
      lp.ru.push_back(ub.back());
    }
    s.add_rows(2, start, idx, val, lb, ub);
    LpStatus warm = s.solve(LpOptions{});
    RandomLp ext = lp;
    ext.A = SparseMatrix::from_triplets(n, m + 2, J, I, V);
    double cold_obj;
    LpStatus cold = cold_solve(ext, &cold_obj);
    CHECK(warm == cold);
    if (warm != LpStatus::Optimal || cold != LpStatus::Optimal) continue;
    CHECK_NEAR(s.objective(), cold_obj, 1e-6);
    // Remove basic (slack) rows and resolve: objective must not increase.
    std::vector<Index> slack;
    for (Index i = 0; i < m + 2; ++i)
      if (s.row_is_basic(i)) slack.push_back(i);
    const double before = s.objective();
    s.remove_rows(slack);
    CHECK(s.num_rows() == m + 2 - static_cast<Index>(slack.size()));
    REQUIRE(s.solve(LpOptions{}) == LpStatus::Optimal);
    CHECK_NEAR(s.objective(), before, 1e-6);
  }
}

// Steepest-edge weights carried with a basis (basic_weights / set_basic_weights): the weights the dual
// simplex maintains by updates equal the ones recomputed for the same basis, installing them does not
// change the results of later solves, also with rows added after the basis was taken, and a count that
// does not fit the basis is refused.
TEST(lp_basic_weights_carry) {
  std::mt19937 rng(11);
  int carried = 0;
  for (int t = 0; t < 300; ++t) {
    const Index m = 2 + rng() % 20, n = 2 + rng() % 30;
    RandomLp lp = random_lp(rng, m, n);
    DualSimplex s;
    s.load(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru);
    if (s.solve(LpOptions{}) != LpStatus::Optimal) continue;
    const Basis b = s.get_basis();
    std::vector<float> w, exact;
    if (!s.basic_weights(n + m, w)) continue;  // e.g. after a primal cleanup
    ++carried;
    CHECK_EQ(static_cast<Index>(w.size()), m);
    // Recomputed for the same basis (set_basis asks for exact weights at the next solve).
    DualSimplex e = s;
    e.set_basis(b);
    REQUIRE(e.solve(LpOptions{}) == LpStatus::Optimal);
    REQUIRE(e.basic_weights(n + m, exact));
    REQUIRE(exact.size() == w.size());
    for (size_t k = 0; k < w.size(); ++k) CHECK(std::abs(w[k] - exact[k]) <= 1e-3 * std::max(1.0f, exact[k]));
    // A wrong count is refused; so is a call that does not follow set_basis.
    DualSimplex r = s;
    CHECK(!r.set_basic_weights(n + m, w.data(), static_cast<Index>(w.size())));
    r.set_basis(b);
    CHECK(!r.set_basic_weights(n + m, w.data(), static_cast<Index>(w.size()) - 1));
    // Installed weights, then bound changes: same result as a cold solve.
    DualSimplex u = s;
    u.set_basis(b);
    CHECK(u.set_basic_weights(n + m, w.data(), static_cast<Index>(w.size())));
    RandomLp mod = lp;
    for (int k = 0; k < 3; ++k) {
      const Index j = rng() % n;
      const double v = s.x()[j];
      if (rng() % 2) mod.cu[j] = std::max(mod.cl[j], std::floor(v));
      else mod.cl[j] = std::min(mod.cu[j], std::ceil(v));
      u.set_col_bounds(j, mod.cl[j], mod.cu[j]);
    }
    const LpStatus warm = u.solve(LpOptions{});
    double cold_obj;
    const LpStatus cold = cold_solve(mod, &cold_obj);
    CHECK(warm == cold);
    if (warm == LpStatus::Optimal && cold == LpStatus::Optimal) CHECK_NEAR(u.objective(), cold_obj, 1e-6);
    // Rows added since (basic logicals, not covered by the weights): accepted, same result as a cold solve.
    DualSimplex x = s;
    std::vector<Index> start{0}, idx, J(lp.A.index.size()), I = lp.A.index;
    std::vector<double> val, rlb, rub, V = lp.A.value;
    for (Index j = 0; j < n; ++j)
      for (Index q = lp.A.begin(j); q < lp.A.end(j); ++q) J[q] = j;
    RandomLp ext = lp;
    for (int rr = 0; rr < 2; ++rr) {
      for (Index j = 0; j < n; ++j) {
        if (rng() % 3) continue;
        const double a = static_cast<int>(rng() % 11) - 5;
        if (a == 0) continue;
        idx.push_back(j), val.push_back(a);
        J.push_back(j), I.push_back(m + rr), V.push_back(a);
      }
      start.push_back(static_cast<Index>(idx.size()));
      double act = 0;
      for (Index k = start[rr]; k < start[rr + 1]; ++k) act += val[k] * s.x()[idx[k]];
      rlb.push_back(-kInf);
      rub.push_back(std::floor(act) - 1.0);
      ext.rl.push_back(rlb.back());
      ext.ru.push_back(rub.back());
    }
    x.add_rows(2, start, idx, val, rlb, rub);
    Basis bx = b;
    bx.status.resize(static_cast<size_t>(n + m + 2), VarStatus::Basic);
    x.set_basis(bx);
    CHECK(x.set_basic_weights(n + m, w.data(), static_cast<Index>(w.size())));
    const LpStatus xs = x.solve(LpOptions{});
    ext.A = SparseMatrix::from_triplets(n, m + 2, J, I, V);
    const LpStatus xc = cold_solve(ext, &cold_obj);
    CHECK(xs == xc);
    if (xs == LpStatus::Optimal && xc == LpStatus::Optimal) CHECK_NEAR(x.objective(), cold_obj, 1e-6);
  }
  CHECK(carried > 100);
}

TEST(lp_tableau_row_consistency) {
  std::mt19937 rng(11);
  for (int t = 0; t < 50; ++t) {
    RandomLp lp = random_lp(rng, 2 + rng() % 10, 2 + rng() % 15);
    DualSimplex s;
    s.load(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru);
    REQUIRE(s.solve(LpOptions{}) == LpStatus::Optimal);
    const Index N = s.num_cols() + s.num_rows();
    for (Index b : std::vector<Index>(s.basic_vars())) {
      std::vector<double> row;
      REQUIRE(s.tableau_row(b, row));
      // x_b + sum_{nonbasic} row_j x_j = 0 (since [A I] x = 0).
      double v = s.var_value(b);
      for (Index j = 0; j < N; ++j)
        if (s.status(j) != VarStatus::Basic) v += row[j] * s.var_value(j);
      CHECK_NEAR(v, 0.0, 1e-7);
      CHECK_NEAR(row[b], 1.0, 1e-9);
    }
  }
}

TEST(lp_tableau_row_sparse_matches_dense) {
  std::mt19937 rng(12);
  for (int t = 0; t < 50; ++t) {
    RandomLp lp = random_lp(rng, 2 + rng() % 12, 2 + rng() % 20);
    DualSimplex s;
    s.load(lp.A, lp.c, lp.cl, lp.cu, lp.rl, lp.ru);
    REQUIRE(s.solve(LpOptions{}) == LpStatus::Optimal);
    const Index N = s.num_cols() + s.num_rows();
    std::vector<double> sparse(N, 0.0);
    std::vector<Index> nz;
    for (Index b : std::vector<Index>(s.basic_vars())) {
      std::vector<double> dense;
      REQUIRE(s.tableau_row(b, dense));
      for (Index j : nz) sparse[j] = 0.0;
      REQUIRE(s.tableau_row(b, sparse, nz));
      CHECK(s.tableau_row_work() >= static_cast<long long>(nz.size()));  // at least the listed entries
      for (size_t k = 1; k < nz.size(); ++k) CHECK(nz[k - 1] < nz[k]);
      std::vector<char> listed(N, 0);
      for (Index j : nz) listed[j] = 1;
      for (Index j = 0; j < N; ++j) {
        CHECK(sparse[j] == dense[j]);  // bit-identical
        if (!listed[j]) CHECK(dense[j] == 0.0);
      }
    }
  }
}

TEST(lp_negligible_entries_do_not_skew_row_scaling) {
  // Min-max LPs (min t s.t. t + sum_k a_ik x_k >= 0, x in [0, 1], sum_k x_k = K) as in var-smallemery-m6j6, where a
  // coefficient of 8.9e-16 next to ~100 in a row scaled it by 2^22 and the dual tolerance on its logical let the
  // root LP stop 16 units above the optimum. The same LPs with such entries added must have the same optimum.
  std::mt19937 rng(97);
  int checked = 0;
  for (int t = 0; t < 60; ++t) {
    const Index rows = 5 + static_cast<Index>(rng() % 20), nx = 8 + static_cast<Index>(rng() % 30);
    const Index n = nx + 1, m = rows + 1;
    std::vector<Index> J, I, Jt, It;
    std::vector<double> V, Vt;
    for (Index i = 0; i < rows; ++i) {
      J.push_back(0), I.push_back(i), V.push_back(1.0);  // t
      for (Index k = 0; k < nx; ++k) {
        if (rng() % 100 < 60) J.push_back(1 + k), I.push_back(i), V.push_back(static_cast<double>(static_cast<int>(rng() % 4001) - 2000) / 32.0);
      }
    }
    for (Index k = 0; k < nx; ++k) J.push_back(1 + k), I.push_back(rows), V.push_back(1.0);
    Jt = J, It = I, Vt = V;
    for (Index i = 0; i < rows; ++i) {  // a negligible entry in every min-max row, on a column not in it yet
      Index k = 1 + static_cast<Index>(rng() % nx);
      bool present = false;
      for (size_t e = 0; e < J.size(); ++e) present = present || (I[e] == i && J[e] == k);
      if (present) continue;
      Jt.push_back(k), It.push_back(i), Vt.push_back(8.881784197001252e-16);
    }
    std::vector<double> c(n, 0.0), cl(n, 0.0), cu(n, 1.0), rl(m, 0.0), ru(m, kInf);
    c[0] = 1.0, cl[0] = -kInf, cu[0] = kInf;
    rl[rows] = ru[rows] = static_cast<double>(1 + rng() % (nx / 2));
    DualSimplex a, b;
    a.load(SparseMatrix::from_triplets(n, m, J, I, V), c, cl, cu, rl, ru);
    b.load(SparseMatrix::from_triplets(n, m, Jt, It, Vt), c, cl, cu, rl, ru);
    const LpStatus sa = a.solve(LpOptions{}), sb = b.solve(LpOptions{});
    REQUIRE(sa == LpStatus::Optimal);
    REQUIRE(sb == LpStatus::Optimal);
    if (!CHECK_NEAR(a.objective(), b.objective(), 1e-6 * (1.0 + std::abs(a.objective())))) {
      std::printf("  case %d: %.10g without, %.10g with the negligible entries\n", t, a.objective(), b.objective());
    }
    ++checked;
  }
  CHECK(checked == 60);
}

