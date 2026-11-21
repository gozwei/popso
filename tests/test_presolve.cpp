// Tests for presolve reductions and postsolve, plus randomized end-to-end
// comparisons of solving with and without presolve.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "milp/io/mps_reader.hpp"
#include "milp/presolve/clique_merge.hpp"
#include "milp/presolve/group_prop.hpp"
#include "milp/presolve/implied_int.hpp"
#include "milp/presolve/presolve.hpp"
#include "milp/presolve/probing.hpp"
#include "milp/search/obj_cliques.hpp"
#include "milp/solver/postsolve.hpp"
#include "milp/solver/solver.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

// Small problem builder: rows given as (col, coef) lists.
struct Builder {
  ModelBuilder b;
  Index col(double c, double lb, double ub, bool integer = false) {
    return b.add_col("c" + std::to_string(b.model().num_cols()), c, lb, ub,
                     integer ? VarType::Integer : VarType::Continuous);
  }
  Index row(double lb, double ub, const std::vector<std::pair<Index, double>>& coefs) {
    const Index r = b.add_row("r" + std::to_string(b.model().num_rows()), lb, ub);
    for (const auto& [j, v] : coefs) b.add_coef(r, j, v);
    return r;
  }
  MipProblem build() {
    MipProblem p;
    Model m = b.build();
    REQUIRE(MipProblem::from_model(m, &p));
    return p;
  }
};

PresolveOptions only(bool bt, bool coef, bool par, bool dual, bool aggr) {
  PresolveOptions o;
  o.bound_tightening = bt;
  o.coef_tightening = coef;
  o.parallel_rows = par;
  o.dual_fixing = dual;
  o.aggregation = aggr;
  o.substitution = false;
  o.probing = false;
  o.parallel_cols = false;
  o.dominated_cols = false;
  o.dual_aggregation = false;
  o.sparsify = false;
  o.group_prop = false;
  return o;
}

PresolveOptions only_subst() {
  PresolveOptions o = only(false, false, false, false, false);
  o.substitution = true;
  o.substitution_bounded = true;  // off by default (cut separators), kept for models where it is enabled
  return o;
}

PresolveOptions only_probing() {
  PresolveOptions o = only(false, false, false, false, false);
  o.probing = true;
  return o;
}

const Logger kQuiet(0);

// Reduced column index of original column j (-1 if removed).
Index reduced_index(const Presolver& pre, Index j) {
  const auto& map = pre.col_map();
  for (size_t k = 0; k < map.size(); ++k) {
    if (map[k] == j) return static_cast<Index>(k);
  }
  return -1;
}

}  // namespace

TEST(presolve_empty_rows_and_fixed_cols) {
  Builder b;
  const Index x = b.col(2.0, 3.0, 3.0);       // fixed at 3
  const Index y = b.col(1.0, 0.0, 10.0);
  b.row(1.0, 8.0, {{x, 1.0}, {y, 1.0}});       // 1 <= 3 + y <= 8
  b.row(-1.0, 1.0, {});                        // empty, feasible
  Presolver pre;
  REQUIRE(pre.run(b.build(), only(false, false, false, false, false), kQuiet) == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  // x removed; row becomes a singleton -> y in [0, 5]; then y is an empty column fixed at 0.
  CHECK_EQ(q.num_rows(), 0);
  CHECK_EQ(q.num_cols(), 0);
  CHECK_NEAR(q.offset, 6.0, 1e-12);
  const auto xs = pre.postsolve({});
  CHECK_EQ(xs[x], 3.0);
  CHECK_EQ(xs[y], 0.0);
}

TEST(presolve_empty_row_infeasible) {
  Builder b;
  const Index x = b.col(1.0, 0.0, 0.0);
  b.row(1.0, 2.0, {{x, 1.0}});
  Presolver pre;
  CHECK(pre.run(b.build(), PresolveOptions{}, kQuiet) == PresolveStatus::Infeasible);
}

TEST(presolve_empty_columns) {
  Builder b;
  const Index x = b.col(1.0, -2.0, 5.0);   // min -> lb
  const Index y = b.col(-1.0, -2.0, 5.0);  // -> ub
  const Index z = b.col(0.0, 1.0, 5.0);    // -> closest to zero
  const Index w = b.col(1.0, 0.0, 4.0);
  b.row(1.0, kInf, {{w, 1.0}, {w, 0.0}});
  Presolver pre;
  REQUIRE(pre.run(b.build(), PresolveOptions{}, kQuiet) == PresolveStatus::Reduced);
  CHECK_EQ(pre.reduced().num_cols(), 0);
  const auto xs = pre.postsolve({});
  CHECK_EQ(xs[x], -2.0);
  CHECK_EQ(xs[y], 5.0);
  CHECK_EQ(xs[z], 1.0);
  CHECK_EQ(xs[w], 1.0);
  CHECK_NEAR(pre.reduced().offset, -2.0 - 5.0 + 1.0, 1e-12);

  Builder u;
  u.col(-1.0, 0.0, kInf);  // empty, pushed to +inf
  Presolver pre2;
  CHECK(pre2.run(u.build(), PresolveOptions{}, kQuiet) == PresolveStatus::Unbounded);
}

TEST(presolve_singleton_rows_round_integers) {
  Builder b;
  const Index x = b.col(-1.0, 0.0, 10.0, true);
  const Index y = b.col(-1.0, 0.0, 10.0);
  const Index z = b.col(-1.0, 0.0, 10.0, true);
  b.row(-kInf, 5.0, {{x, 2.0}});      // x <= 2.5 -> 2
  b.row(-7.0, kInf, {{y, -2.0}});     // y <= 3.5
  b.row(-kInf, 3.0, {{x, 1.0}, {y, 1.0}, {z, 1.0}});
  Presolver pre;
  REQUIRE(pre.run(b.build(), only(false, false, false, false, false), kQuiet) == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  CHECK_EQ(q.num_rows(), 1);
  CHECK_EQ(q.col_ub[reduced_index(pre, x)], 2.0);
  CHECK_NEAR(q.col_ub[reduced_index(pre, y)], 3.5, 1e-12);

  Builder c;
  const Index v = c.col(1.0, 0.0, 10.0, true);
  c.row(3.0, 3.0, {{v, 2.0}});  // 2v = 3 has no integer solution
  Presolver pre2;
  CHECK(pre2.run(c.build(), PresolveOptions{}, kQuiet) == PresolveStatus::Infeasible);
}

TEST(presolve_bound_tightening) {
  Builder b;
  const Index x = b.col(-1.0, 0.0, 10.0, true);
  const Index y = b.col(-2.0, 0.0, 10.0, true);
  const Index z = b.col(-1.0, 0.0, kInf);
  b.row(-kInf, 3.5, {{x, 1.0}, {y, 1.0}});
  b.row(-kInf, 10.0, {{x, 1.0}, {z, 2.0}});
  Presolver pre;
  REQUIRE(pre.run(b.build(), only(true, false, false, false, false), kQuiet) == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  CHECK_EQ(q.col_ub[reduced_index(pre, x)], 3.0);
  CHECK_EQ(q.col_ub[reduced_index(pre, y)], 3.0);
  // z <= (10 - 0) / 2 = 5 (continuous, infinite before: significant).
  CHECK_NEAR(q.col_ub[reduced_index(pre, z)], 5.0, 1e-5);
  CHECK(q.col_ub[reduced_index(pre, z)] >= 5.0);
}

TEST(presolve_redundant_and_forcing_rows) {
  Builder b;
  const Index x = b.col(-1.0, 0.0, 10.0);
  const Index y = b.col(-1.0, 0.0, 10.0, true);
  const Index z = b.col(-1.0, 0.0, 5.0);
  const Index w = b.col(-1.0, 0.0, 5.0, true);
  b.row(-kInf, 30.0, {{x, 1.0}, {y, 1.0}});         // redundant
  b.row(-kInf, 15.0, {{x, 1.0}, {y, 1.0}, {z, 1.0}});
  b.row(-kInf, 0.0, {{z, 1.0}, {w, 2.0}});          // forcing: z = w = 0
  Presolver pre;
  REQUIRE(pre.run(b.build(), only(true, false, false, false, false), kQuiet) == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  CHECK_EQ(q.num_rows(), 1);
  CHECK_EQ(reduced_index(pre, z), -1);
  CHECK_EQ(reduced_index(pre, w), -1);
  const auto xs = pre.postsolve(std::vector<double>(q.num_cols(), 1.0));
  CHECK_EQ(xs[z], 0.0);
  CHECK_EQ(xs[w], 0.0);

  Builder c;
  const Index u = c.col(1.0, 0.0, 1.0, true);
  const Index v = c.col(1.0, 0.0, 1.0, true);
  c.row(2.5, kInf, {{u, 1.0}, {v, 1.0}});
  Presolver pre2;
  CHECK(pre2.run(c.build(), PresolveOptions{}, kQuiet) == PresolveStatus::Infeasible);
}

TEST(presolve_coefficient_tightening_big_m) {
  Builder b;
  const Index x = b.col(-1.0, 0.0, 10.0);
  const Index y = b.col(5.0, 0.0, 1.0, true);
  const Index z = b.col(-1.0, 0.0, 4.0);
  b.row(-kInf, 0.0, {{x, 1.0}, {y, -100.0}});  // x <= 100 y  ->  x <= 10 y
  b.row(1.0, kInf, {{z, 1.0}, {y, 50.0}});      // z + 50 y >= 1  ->  z + y >= 1
  Presolver pre;
  REQUIRE(pre.run(b.build(), only(false, true, false, false, false), kQuiet) == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  REQUIRE(q.num_rows() == 2);
  const Index ry = reduced_index(pre, y);
  REQUIRE(ry >= 0);
  std::vector<double> coefs;
  for (Index p = q.A.begin(ry); p < q.A.end(ry); ++p) coefs.push_back(q.A.value[p]);
  REQUIRE(coefs.size() == 2);
  CHECK_NEAR(coefs[0], -10.0, 1e-6);
  CHECK(coefs[0] <= -10.0);  // never stronger than valid
  CHECK_NEAR(coefs[1], 1.0, 1e-6);
  CHECK(coefs[1] >= 1.0);
  CHECK_NEAR(q.row_lb[1], 1.0, 1e-6);
}

TEST(presolve_parallel_rows) {
  Builder b;
  const Index x = b.col(-1.0, 0.0, 10.0);
  const Index y = b.col(-1.0, 0.0, 10.0);
  b.row(-kInf, 4.0, {{x, 1.0}, {y, 2.0}});
  b.row(-kInf, 6.0, {{x, 2.0}, {y, 4.0}});    // x + 2y <= 3 (tighter)
  b.row(-10.0, kInf, {{x, -3.0}, {y, -6.0}});  // x + 2y <= 10/3
  Presolver pre;
  REQUIRE(pre.run(b.build(), only(false, false, true, false, false), kQuiet) == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  // The row with the largest coefficients is kept: -3x - 6y >= -9.
  REQUIRE(q.num_rows() == 1);
  CHECK_NEAR(q.row_lb[0], -9.0, 1e-12);
  CHECK(q.row_ub[0] == kInf);
  CHECK_EQ(q.A.value[q.A.begin(0)], -3.0);

  Builder c;
  const Index u = c.col(1.0, 0.0, 10.0);
  const Index v = c.col(1.0, 0.0, 10.0);
  c.row(5.0, kInf, {{u, 1.0}, {v, 1.0}});
  c.row(-kInf, 8.0, {{u, 2.0}, {v, 2.0}});
  Presolver pre2;
  CHECK(pre2.run(c.build(), only(false, false, true, false, false), kQuiet) == PresolveStatus::Infeasible);
}

TEST(presolve_dual_fixing) {
  Builder b;
  const Index x = b.col(1.0, 1.0, 10.0, true);  // only in <= rows with +coef: fix at lb
  const Index y = b.col(-1.0, 0.0, 7.0);        // pushed up, only in >= rows with +coef: fix at ub
  const Index z = b.col(1.0, 0.0, 10.0);
  b.row(-kInf, 20.0, {{x, 1.0}, {z, 1.0}});
  b.row(2.0, kInf, {{y, 1.0}, {z, 1.0}});
  Presolver pre;
  REQUIRE(pre.run(b.build(), only(false, false, false, true, false), kQuiet) == PresolveStatus::Reduced);
  const auto xs = pre.postsolve(std::vector<double>(pre.reduced().num_cols(), 0.0));
  CHECK_EQ(xs[x], 1.0);
  CHECK_EQ(xs[y], 7.0);
  CHECK_EQ(reduced_index(pre, x), -1);
  CHECK_EQ(reduced_index(pre, y), -1);

  Builder c;
  const Index u = c.col(-1.0, 0.0, kInf, true);  // no up-lock, cost < 0, ub = inf
  const Index v = c.col(1.0, 0.0, 5.0);
  c.row(1.0, kInf, {{u, 1.0}, {v, 1.0}});
  Presolver pre2;
  CHECK(pre2.run(c.build(), only(false, false, false, true, false), kQuiet) == PresolveStatus::Unbounded);
}

TEST(presolve_doubleton_aggregation) {
  Builder b;
  const Index x = b.col(1.0, 0.0, 9.0);         // continuous, eliminated
  const Index y = b.col(1.0, 0.0, 10.0, true);
  const Index z = b.col(-1.0, 0.0, 10.0, true);
  b.row(1.0, 1.0, {{x, 1.0}, {y, -2.0}});        // x = 1 + 2y, x <= 9 -> y <= 4
  b.row(-kInf, 12.0, {{x, 1.0}, {z, 1.0}});      // 1 + 2y + z <= 12
  Presolver pre;
  REQUIRE(pre.run(b.build(), only(false, false, false, false, true), kQuiet) == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  CHECK_EQ(reduced_index(pre, x), -1);
  const Index ry = reduced_index(pre, y);
  REQUIRE(ry >= 0);
  CHECK_EQ(q.col_ub[ry], 4.0);
  CHECK_NEAR(q.cost[ry], 3.0, 1e-12);  // x + y = 1 + 3y
  CHECK_NEAR(q.offset, 1.0, 1e-12);
  REQUIRE(q.num_rows() == 1);
  CHECK_NEAR(q.row_ub[0], 11.0, 1e-12);
  std::vector<double> xr(q.num_cols(), 0.0);
  xr[ry] = 3.0;
  const auto xs = pre.postsolve(xr);
  CHECK_NEAR(xs[x], 7.0, 1e-12);
  CHECK_NEAR(xs[y], 3.0, 0);
}

TEST(presolve_integer_doubleton_aggregation) {
  Builder b;
  const Index x = b.col(1.0, 0.0, 9.0, true);
  const Index y = b.col(-1.0, 0.0, 10.0, true);
  const Index z = b.col(1.0, 0.0, 10.0, true);
  b.row(1.0, 1.0, {{x, 1.0}, {y, -2.0}});    // x = 1 + 2y stays integral -> eliminated, y <= 4
  b.row(-kInf, 12.0, {{x, 1.0}, {z, 3.0}});
  b.row(1.0, 1.0, {{y, 2.0}, {z, 3.0}});     // neither 3/2 nor 2/3 integral: kept
  Presolver pre;
  REQUIRE(pre.run(b.build(), only(false, false, false, false, true), kQuiet) == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  CHECK_EQ(reduced_index(pre, x), -1);
  const Index ry = reduced_index(pre, y), rz = reduced_index(pre, z);
  REQUIRE(ry >= 0);
  REQUIRE(rz >= 0);
  CHECK_EQ(q.col_ub[ry], 4.0);
  CHECK_EQ(q.num_rows(), 2);
  std::vector<double> xr(q.num_cols(), 0.0);
  xr[ry] = 2.0;
  CHECK_EQ(pre.postsolve(xr)[x], 5.0);
}

TEST(presolve_column_singleton_substitution) {
  Builder b;
  const Index x1 = b.col(2.0, 0.0, 1.0, true);
  const Index x2 = b.col(3.0, 0.0, 1.0, true);
  const Index s = b.col(1.0, 0.0, 1.0, true);  // only in the partitioning row: a slack
  const Index c = b.col(0.5, 0.0, 4.0);         // continuous singleton with a 2x coefficient
  b.row(1.0, 1.0, {{x1, 1.0}, {x2, 1.0}, {s, 1.0}});
  b.row(3.0, 3.0, {{x1, 1.0}, {x2, 1.0}, {c, 2.0}});
  Presolver pre;
  REQUIRE(pre.run(b.build(), only_subst(), kQuiet) == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  CHECK_EQ(reduced_index(pre, s), -1);
  CHECK_EQ(reduced_index(pre, c), -1);
  REQUIRE(q.num_cols() == 2);
  // s = 1 - x1 - x2 in [0, 1] -> x1 + x2 <= 1 (s <= 1 is implied); c = (3 - x1 - x2) / 2 in [0, 4] -> x1 + x2 <= 3.
  CHECK_NEAR(q.offset, 1.0 + 0.75, 1e-12);
  CHECK_NEAR(q.cost[reduced_index(pre, x1)], 2.0 - 1.0 - 0.25, 1e-12);
  bool found = false;
  for (Index i = 0; i < q.num_rows(); ++i) found = found || (q.row_lb[i] <= 0.0 && q.row_ub[i] == 1.0);
  CHECK(found);
  std::vector<double> xr(2, 0.0);
  xr[reduced_index(pre, x2)] = 1.0;
  const auto xs = pre.postsolve(xr);
  CHECK_EQ(xs[s], 0.0);
  CHECK_NEAR(xs[c], 1.0, 1e-12);
  // An integer singleton whose coefficient does not divide the others stays.
  Builder d;
  const Index y1 = d.col(1.0, 0.0, 3.0, true);
  const Index y2 = d.col(1.0, 0.0, 3.0, true);
  d.row(1.0, 1.0, {{y1, 2.0}, {y2, 3.0}});
  Presolver pre2;
  REQUIRE(pre2.run(d.build(), only_subst(), kQuiet) == PresolveStatus::Reduced);
  CHECK(reduced_index(pre2, y1) >= 0);
  CHECK(reduced_index(pre2, y2) >= 0);
}

TEST(presolve_implied_free_substitution) {
  Builder b;
  const Index x = b.col(1.0, -kInf, kInf);  // free: x = 4 - y - z
  const Index y = b.col(1.0, 0.0, 2.0);
  const Index z = b.col(-1.0, 0.0, 2.0);
  const Index w = b.col(0.0, -1.0, 10.0);  // only in inequalities: stays
  b.row(4.0, 4.0, {{x, 1.0}, {y, 1.0}, {z, 1.0}});
  b.row(-kInf, 3.0, {{x, 1.0}, {y, 2.0}});
  b.row(2.0, 3.0, {{w, 1.0}, {y, 1.0}, {z, 1.0}});
  b.row(-kInf, 5.0, {{w, 1.0}, {x, 1.0}});
  Presolver pre;
  REQUIRE(pre.run(b.build(), only_subst(), kQuiet) == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  CHECK_EQ(reduced_index(pre, x), -1);
  CHECK(reduced_index(pre, w) >= 0);
  CHECK_NEAR(q.offset, 4.0, 1e-12);
  CHECK_EQ(q.num_cols(), 3);
  std::vector<double> xr(q.num_cols(), 0.0);
  xr[reduced_index(pre, y)] = 1.0;
  xr[reduced_index(pre, z)] = 0.5;
  xr[reduced_index(pre, w)] = 0.5;
  CHECK_NEAR(pre.postsolve(xr)[x], 2.5, 1e-12);
}

TEST(presolve_long_equation_singleton_substitution) {
  // Objective-defining row z = sum_k c_k x_k with 1500 entries and a free singleton z (cost 1): z is
  // substituted (the row is longer than kMaxSubstRow) and the row disappears.
  Builder b;
  const int n = 1500;
  std::vector<std::pair<Index, double>> row;
  std::vector<Index> xs;
  for (int k = 0; k < n; ++k) {
    xs.push_back(b.col(0.0, 0.0, 1.0, true));
    row.push_back({xs.back(), (k < 10 ? 1.0 : 150.0) * (1 + k % 7)});  // z: coefficient 1 << 1e-2 * 1050
  }
  const Index z = b.col(1.0, -kInf, kInf);
  row.push_back({z, -1.0});
  b.row(0.0, 0.0, row);
  std::vector<std::pair<Index, double>> cover;
  for (int k = 0; k < 10; ++k) cover.push_back({xs[k], 1.0});
  b.row(3.0, kInf, cover);
  const Model model = b.b.build();
  MipProblem p;
  REQUIRE(MipProblem::from_model(model, &p));
  Presolver pre;
  REQUIRE(pre.run(p, only_subst(), kQuiet) == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  CHECK_EQ(reduced_index(pre, z), -1);
  CHECK_EQ(q.num_rows(), 1);
  CHECK_EQ(q.cost[reduced_index(pre, xs[5])], 6.0);
  std::vector<double> xr(q.num_cols(), 0.0);
  xr[reduced_index(pre, xs[1])] = 1.0;
  xr[reduced_index(pre, xs[6])] = 1.0;
  CHECK_NEAR(pre.postsolve(xr)[z], 2.0 + 7.0, 1e-12);
  // Solve end to end: the three cheapest of x0..x9 (coefficients 1, 2, .., 7, 1, 2, 3) -> 4.
  SolverParams params;
  params.verbosity = 0;
  params.threads = 1;
  const SolveResult r = solve(model, params);
  REQUIRE(r.status == Status::Optimal);
  CHECK_NEAR(r.objective, 4.0, 1e-9);
}

TEST(presolve_singleton_does_not_block_substitution) {
  // 2 y + 3 w + c = 5: the integer singleton y does not qualify (2 does not divide 3); the continuous c
  // (in one more row) is substituted instead.
  Builder b;
  const Index y = b.col(1.0, 0.0, 3.0, true);
  const Index w = b.col(1.0, 0.0, 3.0, true);
  const Index c = b.col(1.0, 0.0, 10.0);
  const Index v = b.col(1.0, 0.0, 10.0);
  b.row(5.0, 5.0, {{y, 2.0}, {w, 3.0}, {c, 1.0}});
  b.row(1.0, kInf, {{c, 1.0}, {v, 1.0}});
  Presolver pre;
  REQUIRE(pre.run(b.build(), only_subst(), kQuiet) == PresolveStatus::Reduced);
  CHECK_EQ(reduced_index(pre, c), -1);
  CHECK(reduced_index(pre, y) >= 0);
  std::vector<double> xr(pre.reduced().num_cols(), 0.0);
  xr[reduced_index(pre, y)] = 1.0;
  xr[reduced_index(pre, w)] = 1.0;
  CHECK_NEAR(pre.postsolve(xr)[c], 0.0, 1e-12);
}

TEST(presolve_substitution_implied_by_other_rows) {
  // x + y - z = 1 with x in [0, 3] (cost 1): the equation alone gives x in [-9, 11], but x + w <= 3 (w in [0, 1])
  // and x - u >= 0 (u in [0, 2]) imply x in [0, 3]: x is implied free, substituted with its cost, and the equation
  // is removed. y and z (cost, bounds not implied) cannot be substituted. Off (subst_implied): x has a cost and
  // bounds of its own, so nothing is substituted.
  for (bool implied : {true, false}) {
    Builder b;
    const Index x = b.col(1.0, 0.0, 3.0);
    const Index y = b.col(1.0, 0.0, 10.0);
    const Index z = b.col(-1.0, 0.0, 10.0);
    const Index w = b.col(-1.0, 0.0, 1.0);
    const Index u = b.col(0.5, 0.0, 2.0);
    b.row(1.0, 1.0, {{x, 1.0}, {y, 1.0}, {z, -1.0}});
    b.row(-kInf, 3.0, {{x, 1.0}, {w, 1.0}});
    b.row(0.0, kInf, {{x, 1.0}, {u, -1.0}});
    b.row(-kInf, 15.0, {{y, 1.0}, {z, 1.0}});
    const MipProblem p = b.build();
    PresolveOptions o = only_subst();
    o.substitution_bounded = false;  // the default
    o.substitution_implied = implied;
    Presolver pre;
    REQUIRE(pre.run(p, o, kQuiet) == PresolveStatus::Reduced);
    const MipProblem& q = pre.reduced();
    CHECK_EQ(reduced_index(pre, x) < 0, implied);
    CHECK_EQ(q.num_rows(), implied ? 3 : 4);
    CHECK_EQ(pre.stats().subst_implied, implied ? 1 : 0);
    if (!implied) continue;
    // x = 1 - y + z: its cost goes to the offset and to y and z.
    CHECK_NEAR(q.offset, 1.0, 1e-12);
    CHECK_NEAR(q.cost[reduced_index(pre, y)], 0.0, 1e-12);
    CHECK_NEAR(q.cost[reduced_index(pre, z)], 0.0, 1e-12);
    // x + w <= 3 -> -y + z + w <= 2 and x - u >= 0 -> -y + z - u >= -1 keep x in [0, 3].
    std::vector<double> xr(q.num_cols(), 0.0);
    xr[reduced_index(pre, y)] = 2.0;
    xr[reduced_index(pre, z)] = 3.0;
    xr[reduced_index(pre, w)] = 1.0;
    xr[reduced_index(pre, u)] = 1.0;
    CHECK(q.max_violation(xr) <= 1e-12);
    const auto xs = pre.postsolve(xr);
    CHECK_NEAR(xs[x], 2.0, 1e-12);
    CHECK(p.max_violation(xs) <= 1e-12);
    xr[reduced_index(pre, z)] = 5.0;  // x = 4: violates the first of the two rows in the reduced problem as well
    CHECK(q.max_violation(xr) > 0.5);
  }
}

TEST(presolve_substitution_fill_limit) {
  // x + y1 + y2 + y3 = 4 with x in [0, 5] (continuous, no cost) also in x - z1 <= 5 and x - z2 >= -1: substituting
  // x keeps the equation (as y1 + y2 + y3 <= 4: x >= 0 is not implied) and adds y1..y3 to both rows, 6 entries
  // for x's 3: net +3. The y's (costs, bounds not implied) are not substituted. Allowed with the default
  // substitution_fill (8), also with probing (after its first pass: a round more), not with 0 or 2; x integer
  // (with integer y's and z's), or x with a variable bound x <= 5 w (w binary): not at all.
  for (int variant = 0; variant < 6; ++variant) {
    const int fill = variant == 1 ? 0 : (variant == 2 ? 2 : 8);
    const bool integer = variant == 4, vbound = variant == 5;
    Builder b;
    const Index x = b.col(0.0, 0.0, 5.0, integer);
    std::vector<Index> y, z;
    for (int k = 0; k < 3; ++k) y.push_back(b.col(1.0, 0.0, 2.0, integer));
    for (int k = 0; k < 2; ++k) z.push_back(b.col(-1.0, 0.0, 1.0, integer));
    b.row(4.0, 4.0, {{x, 1.0}, {y[0], 1.0}, {y[1], 1.0}, {y[2], 1.0}});
    b.row(-kInf, 5.0, {{x, 1.0}, {z[0], -1.0}});
    b.row(-1.0, kInf, {{x, 1.0}, {z[1], -1.0}});
    b.row(-kInf, 3.0, {{y[0], 1.0}, {z[0], 1.0}});
    b.row(-kInf, 3.0, {{y[1], 1.0}, {z[1], 1.0}});
    b.row(-kInf, 3.0, {{y[2], 1.0}, {z[0], 1.0}});
    if (vbound) b.row(-kInf, 0.0, {{x, 1.0}, {b.col(1.0, 0.0, 1.0, true), -5.0}});
    const MipProblem p = b.build();
    PresolveOptions o = only_subst();
    o.substitution_bounded = false;
    o.substitution_fill = fill;
    o.probing = variant == 3;
    Presolver pre;
    REQUIRE(pre.run(p, o, kQuiet) == PresolveStatus::Reduced);
    const MipProblem& q = pre.reduced();
    const bool subst = fill >= 3 && !integer && !vbound;
    CHECK_EQ(reduced_index(pre, x) < 0, subst);
    CHECK_EQ(pre.stats().subst_fill, subst ? 1 : 0);
    CHECK_EQ(q.num_rows(), vbound ? 7 : 6);
    CHECK_EQ(q.A.nnz(), p.A.nnz() + (subst ? 3 : 0));
    if (!subst) continue;
    // The equation stays as y1 + y2 + y3 <= 4 (x >= 0); x <= 5 is implied by it.
    std::vector<double> xr(q.num_cols(), 0.0);
    for (Index k : y) xr[reduced_index(pre, k)] = 2.0;  // x = -2
    CHECK(q.max_violation(xr) > 1.0);
    xr[reduced_index(pre, y[0])] = 0.0;  // x = 0
    CHECK(q.max_violation(xr) <= 1e-12);
    const auto xs = pre.postsolve(xr);
    CHECK_NEAR(xs[x], 0.0, 1e-12);
    CHECK(p.max_violation(xs) <= 1e-12);
  }
}

TEST(presolve_dual_bound_keeps_integral_value) {
  // makespan C >= s_i + p_i with C's cost > 0: dual bound strengthening gives C <= max(ub_s + p) = 19 exactly
  // (in the first round): no 1e-9 relative margin above an integral value.
  Builder b;
  const Index s1 = b.col(0.0, 0.0, 10.0, true);
  const Index s2 = b.col(0.0, 2.0, 12.0, true);
  const Index cmax = b.col(1.0, 0.0, kInf);
  b.row(4.0, kInf, {{cmax, 1.0}, {s1, -1.0}});
  b.row(7.0, kInf, {{cmax, 1.0}, {s2, -1.0}});
  b.row(-kInf, 15.0, {{s1, 1.0}, {s2, 1.0}});
  b.row(-3.0, 3.0, {{s1, 1.0}, {s2, -1.0}});  // locks s both ways (no dual fixing)
  PresolveOptions o = only(false, false, false, true, false);
  o.implied_int = false;
  o.max_rounds = 1;  // later rounds keep shrinking the bounds
  Presolver pre;
  REQUIRE(pre.run(b.build(), o, kQuiet) == PresolveStatus::Reduced);
  const Index rc = reduced_index(pre, cmax);
  REQUIRE(rc >= 0);
  const double ub = pre.reduced().col_ub[rc];
  CHECK_EQ(ub, 19.0);
}

TEST(implied_integers_primal_chain) {
  // Inventory chain s_t = s_{t-1} + x_t - d_t (integers x, integral d): every s_t is implied integer by the
  // iterated primal rule; u = 0.5 x_1 + s_1 is not (non-integral ratio), nor is f in an inequality only.
  Builder b;
  std::vector<Index> x, st;
  for (int t = 0; t < 4; ++t) x.push_back(b.col(1.0, 0.0, 10.0, true));
  for (int t = 0; t < 4; ++t) st.push_back(b.col(0.5, 0.0, 30.5));
  const Index u = b.col(1.0, 0.0, 100.0);
  const Index f = b.col(1.0, 0.0, 100.0);
  for (int t = 0; t < 4; ++t) {
    std::vector<std::pair<Index, double>> r = {{st[t], 1.0}, {x[t], -1.0}};
    if (t > 0) r.push_back({st[t - 1], -1.0});
    b.row(-(t + 1.0), -(t + 1.0), r);
  }
  b.row(0.0, 0.0, {{u, 1.0}, {x[0], -0.5}, {st[0], -1.0}});
  b.row(1.5, kInf, {{f, 1.0}, {x[1], 1.0}, {u, 1.0}});
  const MipProblem p = b.build();
  ImpliedIntStats is;
  const std::vector<char> impl = detect_implied_integers(p, false, 1e-9, &is);
  for (int t = 0; t < 4; ++t) CHECK(impl[st[t]]);
  CHECK(!impl[u]);
  CHECK(!impl[f]);
  for (int t = 0; t < 4; ++t) CHECK(!impl[x[t]]);
  CHECK_EQ(is.primal, 4);
  CHECK_EQ(is.dual, 0);
  // mark_implied_integers rounds the bounds (s_t <= 30).
  MipProblem q = p;
  CHECK_EQ(mark_implied_integers(q, false), 4);
  CHECK_EQ(q.col_ub[st[0]], 30.0);
  CHECK(q.is_implied_int(st[3]) && !q.is_implied_int(u) && !q.is_int[st[3]]);
}

TEST(implied_integers_dual_rule_makespan) {
  // C >= s_i + p_i (integral p, integer s): only inequalities, so only the dual rule finds C. With a
  // fractional p (2.5) the dual rule does not apply.
  for (int frac = 0; frac < 2; ++frac) {
    Builder b;
    const Index s1 = b.col(0.0, 0.0, 10.0, true);
    const Index s2 = b.col(0.0, 0.0, 10.0, true);
    const Index cmax = b.col(1.0, 0.0, kInf);
    b.row(frac ? 2.5 : 3.0, kInf, {{cmax, 1.0}, {s1, -1.0}});
    b.row(4.0, kInf, {{cmax, 1.0}, {s2, -1.0}});
    b.row(-kInf, 1.0, {{s1, 1.0}, {s2, -1.0}});
    const MipProblem p = b.build();
    CHECK(!detect_implied_integers(p, false)[cmax]);
    ImpliedIntStats is;
    CHECK_EQ(detect_implied_integers(p, true, 1e-9, &is)[cmax] != 0, frac == 0);
    CHECK_EQ(is.dual, frac ? 0 : 1);
    // Through presolve: the flag reaches the reduced problem (unless presolve removes C).
    Presolver pre;
    PresolveOptions o = only(false, false, false, false, false);
    REQUIRE(pre.run(p, o, kQuiet) == PresolveStatus::Reduced);
    const Index rc = reduced_index(pre, cmax);
    REQUIRE(rc >= 0);
    CHECK_EQ(pre.reduced().is_implied_int(rc), frac == 0);
    CHECK_EQ(pre.stats().implied_int, frac ? 0 : 1);
    o.implied_int_dual = false;
    Presolver pre2;
    REQUIRE(pre2.run(p, o, kQuiet) == PresolveStatus::Reduced);
    CHECK(!pre2.reduced().is_implied_int(reduced_index(pre2, cmax)));
  }
}

TEST(implied_integers_fractional_start_stays_feasible) {
  // An implied integer is never required to be integral: a start with a fractional makespan (feasible)
  // crushes to a feasible point of the reduced problem, and the solve finds the optimum 4 (s = 0, w = 1).
  Builder b;
  const Index s1 = b.col(0.0, 0.0, 10.0, true);
  const Index s2 = b.col(0.0, 0.0, 10.0, true);
  const Index cmax = b.col(1.0, 0.0, 40.0);
  const Index w = b.col(0.0, 0.0, 1.0, true);
  b.row(3.0, kInf, {{cmax, 1.0}, {s1, -1.0}});
  b.row(4.0, kInf, {{cmax, 1.0}, {s2, -1.0}});
  b.row(1.0, kInf, {{s1, 1.0}, {s2, 1.0}, {w, 1.0}});
  b.row(-kInf, 1.0, {{s1, 1.0}, {w, 1.0}});
  const Model model = b.b.build();
  MipProblem p;
  REQUIRE(MipProblem::from_model(model, &p));
  Presolver pre;
  PresolveOptions o = only(false, false, false, false, false);
  REQUIRE(pre.run(p, o, kQuiet) == PresolveStatus::Reduced);
  REQUIRE(pre.stats().implied_int >= 1);
  const std::vector<double> start = {1.0, 2.0, 7.25, 0.0};
  CHECK(p.max_violation(start) <= 1e-9);
  CHECK(pre.reduced().max_violation(pre.crush(start)) <= 1e-9);
  SolverParams params;
  params.verbosity = 0;
  params.threads = 1;
  const SolveResult r = solve(model, params, start);
  REQUIRE(r.status == Status::Optimal);
  CHECK_NEAR(r.objective, 4.0, 1e-9);
}

TEST(presolve_parallel_columns) {
  Builder b;
  const Index x1 = b.col(2.0, 0.0, 1.0, true);
  const Index x2 = b.col(2.0, 0.0, 1.0, true);   // same column and cost as x1: merged into y in [0, 2]
  const Index x3 = b.col(-2.0, 0.0, 1.0, true);  // A_3 = -A_1, c_3 = -c_1: merged too (y = x1 - x3)
  const Index x4 = b.col(3.0, 0.0, 1.0, true);   // other cost: kept
  const Index c1 = b.col(1.0, 0.0, 2.0);
  const Index c2 = b.col(0.5, 1.0, 3.0);         // A = 0.5 A_c1, c = 0.5 c_c1
  b.row(-kInf, 1.5, {{x1, 1.0}, {x2, 1.0}, {x3, -1.0}, {x4, 1.0}, {c1, 2.0}, {c2, 1.0}});
  b.row(1.0, kInf, {{x1, 2.0}, {x2, 2.0}, {x3, -2.0}, {x4, 2.0}, {c1, -2.0}, {c2, -1.0}});
  PresolveOptions o = only(false, false, false, false, false);
  o.parallel_cols = true;
  Presolver pre;
  REQUIRE(pre.run(b.build(), o, kQuiet) == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  CHECK_EQ(q.num_cols(), 3);
  CHECK_EQ(pre.stats().parallel_cols, 3);
  const Index ry = reduced_index(pre, x1), rc = reduced_index(pre, c1);
  REQUIRE(ry >= 0);
  REQUIRE(rc >= 0);
  CHECK_EQ(q.col_lb[ry], -1.0);
  CHECK_EQ(q.col_ub[ry], 2.0);
  CHECK_NEAR(q.col_lb[rc], 0.5, 1e-12);
  CHECK_NEAR(q.col_ub[rc], 3.5, 1e-12);
  for (double y : {-1.0, 0.0, 1.0, 2.0}) {
    std::vector<double> xr(q.num_cols(), 0.0);
    xr[ry] = y;
    xr[rc] = 2.25;
    const auto xs = pre.postsolve(xr);
    CHECK_EQ(xs[x1] + xs[x2] - xs[x3], y);
    for (Index j : {x1, x2, x3}) CHECK(xs[j] == 0.0 || xs[j] == 1.0);
    CHECK_NEAR(xs[c1] + 0.5 * xs[c2], 2.25, 1e-12);
    CHECK(xs[c1] >= 0.0 && xs[c1] <= 2.0 && xs[c2] >= 1.0 && xs[c2] <= 3.0);
  }
}

TEST(presolve_dual_aggregation) {
  // Facilities y0 (aggregated row x0k <= 4 y0) and y1 (rows x1k <= y1) cover four clients: x0k + x1k >= 1.
  PresolveOptions o = only(false, false, false, false, false);
  o.dual_aggregation = true;
  {
    Builder b;
    const Index y0 = b.col(3.0, 0.0, 1.0, true), y1 = b.col(4.0, 0.0, 1.0, true);
    std::vector<Index> x0, x1;
    for (int k = 0; k < 4; ++k) x0.push_back(b.col(0.0, 0.0, 1.0, true)), x1.push_back(b.col(k == 0 ? -0.5 : 0.0, 0.0, 1.0, true));
    std::vector<std::pair<Index, double>> agg{{y0, -4.0}};
    for (int k = 0; k < 4; ++k) agg.push_back({x0[k], 1.0});
    b.row(-kInf, 0.0, agg);
    for (int k = 0; k < 4; ++k) b.row(0.0, kInf, {{y1, 1.0}, {x1[k], -1.0}});  // x1k <= y1 as a >= row
    for (int k = 0; k < 4; ++k) b.row(1.0, kInf, {{x0[k], 1.0}, {x1[k], 1.0}});
    Presolver pre;
    REQUIRE(pre.run(b.build(), o, kQuiet) == PresolveStatus::Reduced);
    CHECK_EQ(pre.stats().dual_aggregations, 8);
    CHECK_EQ(pre.reduced().num_cols(), 2);
    // y1's cost absorbs x10's -0.5.
    CHECK_NEAR(pre.reduced().cost[reduced_index(pre, y1)], 3.5, 1e-12);
    std::vector<double> xr(2, 0.0);
    xr[reduced_index(pre, y0)] = 1.0;
    const auto xs = pre.postsolve(xr);
    for (int k = 0; k < 4; ++k) CHECK_EQ(xs[x0[k]], 1.0), CHECK_EQ(xs[x1[k]], 0.0);
  }
  {
    // Not aggregated: x0 has a positive cost; x1's row does not hold at y = 1 for all x (2 y covers 3 columns);
    // x4 has a second up-lock; x5's row is an equation.
    Builder b;
    const Index y = b.col(1.0, 0.0, 1.0, true);
    const Index x0 = b.col(1.0, 0.0, 1.0, true);
    b.row(-kInf, 0.0, {{x0, 1.0}, {y, -1.0}});
    const Index x1 = b.col(0.0, 0.0, 1.0, true), x2 = b.col(0.0, 0.0, 1.0, true), x3 = b.col(0.0, 0.0, 1.0, true);
    b.row(-kInf, 0.0, {{x1, 1.0}, {x2, 1.0}, {x3, 1.0}, {y, -2.0}});
    const Index x4 = b.col(0.0, 0.0, 1.0, true);
    b.row(-kInf, 0.0, {{x4, 1.0}, {y, -1.0}});
    b.row(-kInf, 1.0, {{x4, 1.0}, {x0, 1.0}});
    const Index x5 = b.col(0.0, 0.0, 1.0, true), z = b.col(0.0, 0.0, 1.0, true);
    b.row(0.0, 0.0, {{x5, 1.0}, {z, 1.0}, {y, -1.0}});
    for (Index x : {x0, x1, x2, x3, x4, x5}) b.row(1.0, kInf, {{x, 1.0}, {z, 1.0}});
    Presolver pre;
    REQUIRE(pre.run(b.build(), o, kQuiet) == PresolveStatus::Reduced);
    CHECK_EQ(pre.stats().dual_aggregations, 0);
  }
}

TEST(presolve_dominated_columns) {
  Builder b;
  const Index z = b.col(0.0, 0.0, 50.0, true);
  const Index x1 = b.col(0.0, 0.0, 8.0, true);   // dominated by x2: cheaper capacity use, same in the link row
  const Index x2 = b.col(0.0, 0.0, 50.0, true);
  const Index y = b.col(1.0, 0.0, 10.0);         // continuous: dominated by w (unbounded, cheaper)
  const Index w = b.col(0.5, 0.0, kInf);
  const Index v = b.col(-1.0, 0.0, 10.0);
  b.row(-kInf, 10000.0, {{x1, 1200.0}, {x2, 11.0}});
  b.row(-31.0, 0.0, {{z, -1.0}, {x1, 1.0}, {x2, 1.0}});  // x1 > 0 needs x2 = 50, then x1 <= z - 50 <= 0
  b.row(3.0, kInf, {{y, 1.0}, {w, 1.0}, {v, -1.0}});
  b.row(-kInf, 4.0, {{v, 1.0}, {z, 0.1}});
  PresolveOptions o = only(false, false, false, false, false);
  o.dominated_cols = true;
  Presolver pre;
  REQUIRE(pre.run(b.build(), o, kQuiet) == PresolveStatus::Reduced);
  CHECK(pre.stats().dominated >= 2);
  CHECK_EQ(reduced_index(pre, x1), -1);
  CHECK_EQ(reduced_index(pre, y), -1);
  const auto xs = pre.postsolve(std::vector<double>(pre.reduced().num_cols(), 0.0));
  CHECK_EQ(xs[x1], 0.0);
  CHECK_EQ(xs[y], 0.0);
}

TEST(presolve_dominated_columns_set_cover) {
  // x1 covers a subset of x2's rows at no lower cost: with x2 = 1 every row of x1 is covered, so x1 = 0.
  Builder b;
  const Index x1 = b.col(3.0, 0.0, 1.0, true);
  const Index x2 = b.col(2.0, 0.0, 1.0, true);
  const Index x3 = b.col(1.0, 0.0, 1.0, true);
  const Index x4 = b.col(1.0, 0.0, 1.0, true);
  b.row(1.0, kInf, {{x1, 1.0}, {x2, 1.0}, {x3, 1.0}});
  b.row(1.0, kInf, {{x1, 1.0}, {x2, 1.0}, {x4, 1.0}});
  b.row(1.0, kInf, {{x2, 1.0}, {x3, 1.0}, {x4, 1.0}});
  PresolveOptions o = only(false, false, false, false, false);
  o.dominated_cols = true;
  Presolver pre;
  REQUIRE(pre.run(b.build(), o, kQuiet) == PresolveStatus::Reduced);
  CHECK_EQ(reduced_index(pre, x1), -1);
  const auto xs = pre.postsolve(std::vector<double>(pre.reduced().num_cols(), 1.0));
  CHECK_EQ(xs[x1], 0.0);
}

TEST(presolve_probing_continuous_bounds) {
  // Disjunction s2 >= s1 + 10 or s1 >= s2 + 20 (big-M on y): both settings of y imply cmax >= 30.
  Builder b;
  const Index s1 = b.col(0.0, 0.0, 1000.0);
  const Index s2 = b.col(0.0, 0.0, 1000.0);
  const Index y = b.col(0.0, 0.0, 1.0, true);
  const Index cmax = b.col(1.0, 20.0, 1000.0);
  b.row(-990.0, kInf, {{s2, 1.0}, {s1, -1.0}, {y, -1000.0}});  // s2 - s1 >= 10 - 1000 (1 - y)
  b.row(20.0, kInf, {{s1, 1.0}, {s2, -1.0}, {y, 1000.0}});     // s1 - s2 >= 20 - 1000 y
  b.row(10.0, kInf, {{cmax, 1.0}, {s1, -1.0}});
  b.row(20.0, kInf, {{cmax, 1.0}, {s2, -1.0}});
  Presolver pre;
  REQUIRE(pre.run(b.build(), only_probing(), kQuiet) == PresolveStatus::Reduced);
  const Index rc = reduced_index(pre, cmax);
  REQUIRE(rc >= 0);
  CHECK(pre.reduced().col_lb[rc] >= 30.0 - 1e-6);
}

TEST(presolve_gcd_tightening) {
  Builder b;
  const Index x = b.col(-1.0, 0.0, 10.0, true);
  const Index y = b.col(-1.0, 0.0, 10.0, true);
  const Index z = b.col(-1.0, 0.0, 10.0, true);
  b.row(-kInf, 5.0, {{x, 2.0}, {y, 4.0}});
  b.row(-7.0, kInf, {{y, -3.0}, {z, -6.0}});
  PresolveOptions o = only(false, true, false, false, false);
  Presolver pre;
  REQUIRE(pre.run(b.build(), o, kQuiet) == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  REQUIRE(q.num_rows() == 2);
  CHECK_EQ(q.row_ub[0], 4.0);
  CHECK_EQ(q.row_lb[1], -6.0);
  Builder c;
  const Index u = c.col(0.0, 0.0, 10.0, true);
  const Index v = c.col(0.0, 0.0, 10.0, true);
  c.row(3.0, 3.0, {{u, 2.0}, {v, 4.0}});
  Presolver pre2;
  CHECK(pre2.run(c.build(), o, kQuiet) == PresolveStatus::Infeasible);
}

TEST(presolve_dual_bound_strengthening) {
  // y (cost >= 0) is only locked downwards by 25 y >= 10 x1 + 20 x2 + 15 x3: y never needs to exceed 2.
  // z (cost <= 0) only locked upwards by z + x1 <= 7.5 (x1 <= 1): z >= 6.5 is always possible.
  Builder b;
  const Index x1 = b.col(-1.0, 0.0, 1.0, true);
  const Index x2 = b.col(-2.0, 0.0, 1.0, true);
  const Index x3 = b.col(-3.0, 0.0, 1.0, true);
  const Index y = b.col(1.0, 0.0, kInf, true);
  const Index z = b.col(-1.0, 0.0, 100.0);
  b.row(0.0, kInf, {{y, 25.0}, {x1, -10.0}, {x2, -20.0}, {x3, -15.0}});
  b.row(-kInf, 7.5, {{z, 1.0}, {x1, 1.0}});
  b.row(-kInf, 2.0, {{x1, 1.0}, {x2, 1.0}, {x3, 1.0}, {z, 0.0}});
  PresolveOptions o = only(false, false, false, true, false);
  Presolver pre;
  const Model model = b.b.build();
  MipProblem p;
  REQUIRE(MipProblem::from_model(model, &p));
  REQUIRE(pre.run(p, o, kQuiet) == PresolveStatus::Reduced);
  CHECK(pre.stats().dual_bounds >= 2);
  const Index ry = reduced_index(pre, y), rz = reduced_index(pre, z);
  if (ry >= 0) CHECK_EQ(pre.reduced().col_ub[ry], 2.0);
  if (rz >= 0) CHECK(pre.reduced().col_lb[rz] >= 6.5 - 1e-6);
  // Round trip through the solver: x2 = x3 = 1, y = 2, z = 7.5.
  SolverParams params;
  params.verbosity = 0;
  params.threads = 1;
  const SolveResult r1 = solve(model, params);
  CHECK(r1.status == Status::Optimal);
  CHECK_NEAR(r1.objective, -10.5, 1e-6);
}

TEST(presolve_clique_extension) {
  // Packing row x1 + x2 + x3 <= 1; x4 conflicts with x1, x2 (pairwise rows) and with x3 (knapsack 3 x3 + 3 x4 <= 4).
  Builder b;
  std::vector<Index> x;
  for (int k = 0; k < 6; ++k) x.push_back(b.col(-1.0 - k, 0.0, 1.0, true));
  b.row(-kInf, 1.0, {{x[0], 1.0}, {x[1], 1.0}, {x[2], 1.0}});
  b.row(-kInf, 1.0, {{x[0], 1.0}, {x[3], 1.0}});
  b.row(0.0, 1.0, {{x[1], 1.0}, {x[3], 1.0}});
  b.row(-kInf, 4.0, {{x[2], 3.0}, {x[3], 3.0}, {x[4], 1.0}});
  // Partitioning row x4 + x5 = 1 (0-based x[4], x[5]); x[0] conflicts with both: x[0] = 0.
  b.row(1.0, 1.0, {{x[4], 1.0}, {x[5], 1.0}});
  b.row(-kInf, 1.0, {{x[0], 1.0}, {x[4], 1.0}});
  b.row(-kInf, 0.0, {{x[0], 1.0}, {x[5], 1.0}, {x[1], -1.0}});  // x0 + x5 <= x1 <= 1 - x0 ... (x0 = x5 = 1 infeasible)
  MipProblem p = b.build();
  CliqueExtendStats cs;
  REQUIRE(extend_cliques(p, 1000000, &cs));
  CHECK(cs.extended >= 1);
  CHECK(cs.dominated >= 2);  // x0 + x3 <= 1 and x1 + x3 <= 1 lie in the extended row
  CHECK_EQ(cs.fixed, 1);
  CHECK_EQ(p.col_ub[x[0]], 0.0);
  // The extended row x0 + x1 + x2 + x3 <= 1 exists.
  const SparseMatrix R = p.A.transpose();
  bool found = false;
  for (Index i = 0; i < p.num_rows(); ++i) {
    if (R.end(i) - R.begin(i) == 4 && p.row_ub[i] == 1.0) found = true;
  }
  CHECK(found);
  // End to end: same optimum as without presolve.
  Builder c;
  for (int k = 0; k < 6; ++k) c.col(-1.0 - k, 0.0, 1.0, true);
  c.row(-kInf, 1.0, {{0, 1.0}, {1, 1.0}, {2, 1.0}});
  c.row(-kInf, 1.0, {{0, 1.0}, {3, 1.0}});
  c.row(0.0, 1.0, {{1, 1.0}, {3, 1.0}});
  c.row(-kInf, 4.0, {{2, 3.0}, {3, 3.0}, {4, 1.0}});
  c.row(1.0, 1.0, {{4, 1.0}, {5, 1.0}});
  SolverParams params;
  params.verbosity = 0;
  params.threads = 1;
  const Model model = c.b.build();
  const SolveResult r1 = solve(model, params);
  params.presolve = false;
  const SolveResult r2 = solve(model, params);
  CHECK(r1.status == Status::Optimal && r2.status == Status::Optimal);
  CHECK_NEAR(r1.objective, r2.objective, 1e-9);
}

TEST(presolve_probing) {
  Builder b;
  const Index x = b.col(1.0, 0.0, 1.0, true);
  const Index y = b.col(-1.0, 0.0, 1.0, true);
  const Index z = b.col(2.0, 0.0, 1.0, true);
  const Index u = b.col(1.0, 0.0, 1.0, true);
  const Index v = b.col(1.0, 0.0, 1.0, true);
  const Index f1 = b.col(1.0, 0.0, 1.0);  // continuous flows with f1 + f2 = 2 x
  const Index f2 = b.col(-1.0, 0.0, 1.0);
  b.row(-kInf, 0.0, {{x, 1.0}, {y, -1.0}});            // x <= y
  b.row(1.0, kInf, {{x, 1.0}, {y, -1.0}, {z, 2.0}});   // y = 1 & x = 0 forces z = 1 ...
  b.row(-kInf, 1.0, {{y, 1.0}, {z, 1.0}, {u, 1.0}});   // ... while y + z <= 1
  b.row(1.0, 1.0, {{x, 1.0}, {v, 1.0}, {u, 0.0}});      // v = 1 - x
  b.row(0.0, 0.0, {{f1, 1.0}, {f2, 1.0}, {x, -2.0}});
  b.row(-kInf, 3.0, {{u, 1.0}, {v, 1.0}, {z, 1.0}, {y, 1.0}});
  Presolver pre;
  REQUIRE(pre.run(b.build(), only_probing(), kQuiet) == PresolveStatus::Reduced);
  CHECK(pre.stats().probing_fixed + pre.stats().probing_equivs > 0);
  // x <= y and (y = 1 -> x = 1 or z = 1, y + z <= 1): x = y. v = 1 - x, f1 = f2 = x.
  const MipProblem& q = pre.reduced();
  CHECK(q.num_cols() <= 3);
  std::vector<double> xr(q.num_cols(), 0.0);
  const Index rx = reduced_index(pre, x);
  if (rx >= 0) xr[rx] = 1.0;
  const auto xs = pre.postsolve(xr);
  if (rx >= 0) {
    CHECK_EQ(xs[y], 1.0);
    CHECK_EQ(xs[v], 0.0);
    CHECK_NEAR(xs[f1], 1.0, 1e-9);
    CHECK_NEAR(xs[f2], 1.0, 1e-9);
  }
  // x = 1 is infeasible: probing fixes x = 0.
  Builder c;
  const Index a = c.col(-1.0, 0.0, 1.0, true);
  const Index d = c.col(0.0, 0.0, 1.0, true);
  const Index e = c.col(0.0, 0.0, 1.0, true);
  c.row(-kInf, 0.0, {{a, 1.0}, {d, -1.0}});   // a <= d
  c.row(-kInf, 0.0, {{a, 1.0}, {e, -1.0}});   // a <= e
  c.row(-kInf, 1.0, {{d, 1.0}, {e, 1.0}});    // d + e <= 1
  Presolver pre2;
  REQUIRE(pre2.run(c.build(), only_probing(), kQuiet) == PresolveStatus::Reduced);
  CHECK_EQ(reduced_index(pre2, a), -1);
  CHECK_EQ(pre2.postsolve(std::vector<double>(pre2.reduced().num_cols(), 0.0))[a], 0.0);
}

TEST(presolve_everything_removed_solve) {
  // Solved entirely by presolve; the solver must still report the optimum.
  ModelBuilder b;
  const Index x = b.add_col("x", 1.0, 0.0, 10.0, VarType::Integer);
  const Index y = b.add_col("y", 2.0, 0.0, 10.0);
  const Index r = b.add_row("r", 2.5, kInf);
  b.add_coef(r, x, 1.0);
  const Index s = b.add_row("s", 1.0, 1.0);
  b.add_coef(s, y, 4.0);
  b.model().obj_offset = 1.0;
  Model m = b.build();
  SolverParams params;
  params.verbosity = 0;
  params.threads = 1;
  SolveResult res = solve(m, params);
  REQUIRE(res.status == Status::Optimal);
  CHECK_NEAR(res.objective, 3.0 + 0.5 + 1.0, 1e-9);
  CHECK_NEAR(res.x[0], 3.0, 1e-9);
  CHECK_NEAR(res.x[1], 0.25, 1e-9);
}

namespace {

// Random small MILPs seeded with structures that trigger every reduction.
Model random_model(std::mt19937& rng) {
  auto uni = [&](int lo, int hi) { return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)); };
  ModelBuilder b;
  const bool big = uni(0, 4) == 0;
  const int n = big ? uni(10, 16) : uni(2, 9), m = big ? uni(6, 14) : uni(1, 8);
  const double denom = uni(0, 3) == 0 ? uni(3, 7) : 1.0;  // non-dyadic coefficients stress numerics
  const bool anchored = uni(0, 4) != 0;  // rows built around a point within the column bounds
  std::vector<double> x0;
  for (int j = 0; j < n; ++j) {
    const int kind = uni(0, 9);
    VarType t = kind < 4 ? VarType::Continuous : (kind < 7 ? VarType::Integer : VarType::Binary);
    double lb = uni(-3, 2), ub = lb + uni(0, 8);
    if (t == VarType::Binary) lb = 0, ub = 1;
    if (uni(0, 9) == 0) lb = -kInf;
    if (uni(0, 9) == 0) ub = kInf;
    if (uni(0, 14) == 0 && is_finite(lb)) ub = lb;  // fixed
    if (t == VarType::Continuous && uni(0, 3) == 0 && is_finite(lb) && is_finite(ub)) {
      lb += 0.5 * uni(0, 1);
      ub = std::max(lb, ub - 0.25 * uni(0, 2));
    }
    b.add_col("x" + std::to_string(j), uni(-6, 6) * (uni(0, 4) ? 1.0 : 0.5), lb, ub, t);
    const double lo = is_finite(lb) ? lb : (is_finite(ub) ? ub - 3 : -1), hi = is_finite(ub) ? ub : lo + 3;
    double v = lo + (hi - lo) * uni(0, 4) / 4.0;
    if (t != VarType::Continuous) v = std::floor(v);
    x0.push_back(v);
  }
  for (int i = 0; i < m; ++i) {
    const int kind = uni(0, 11);
    std::vector<std::pair<Index, double>> coefs;
    if (kind == 0) {  // singleton
      coefs.push_back({uni(0, n - 1), uni(1, 4) * (uni(0, 1) ? 1.0 : -1.0)});
    } else if (kind <= 2) {  // doubleton
      Index a = uni(0, n - 1), c = uni(0, n - 1);
      if (a == c) c = (c + 1) % n;
      coefs.push_back({a, uni(1, 3) / denom});
      coefs.push_back({c, -static_cast<double>(uni(1, 4))});
    } else if (kind == 3) {  // big-M: x_a <= M x_c
      Index a = uni(0, n - 1), c = uni(0, n - 1);
      if (a == c) c = (c + 1) % n;
      coefs.push_back({a, 1.0});
      coefs.push_back({c, -static_cast<double>(uni(10, 100))});
    } else {
      for (int j = 0; j < n; ++j) {
        if (uni(0, 2) == 0) coefs.push_back({j, uni(-5, 5) / (uni(0, 1) ? denom : 1.0)});
      }
    }
    double act = 0.0, act0 = 0.0;
    for (auto& [j, v] : coefs) act += v, act0 += v * x0[j];
    double lb = -kInf, ub = kInf;
    const int sense = uni(0, 5);
    double rhs = std::round(act * 0.3) + uni(-4, 6);
    if (anchored) rhs = act0 + (sense <= 1 ? uni(0, 3) : (sense <= 3 ? 0 : (sense == 4 ? 0 : uni(0, 2))));
    if (kind == 3) {
      ub = anchored ? std::max(0.0, act0) : 0.0;
    } else if (sense <= 1) {
      ub = rhs;
    } else if (sense <= 3) {
      lb = rhs - uni(0, 6);
    } else if (sense == 4) {
      lb = ub = (kind <= 2 && !anchored ? uni(-3, 3) : rhs);
    } else {
      lb = rhs - uni(0, 8);
      ub = rhs;
    }
    const Index r = b.add_row("r" + std::to_string(i), lb, ub);
    for (auto& [j, v] : coefs) b.add_coef(r, j, v);
    if (uni(0, 6) == 0 && coefs.size() > 1) {  // parallel copy
      const double s = uni(0, 1) ? 2.0 : -0.5;
      const Index r2 = b.add_row("p" + std::to_string(i), s > 0 ? lb * s : ub * s + uni(0, 2) * s,
                                 s > 0 ? ub * s - uni(0, 2) : lb * s);
      for (auto& [j, v] : coefs) b.add_coef(r2, j, v * s);
    }
  }
  b.model().obj_offset = uni(-3, 3);
  if (uni(0, 1)) b.model().sense = ObjSense::Maximize;
  return b.build();
}

// Converts an internal (minimization) problem back into a user model.
Model to_model(const MipProblem& q) {
  Model m;
  m.obj = q.cost;
  m.obj_offset = q.offset;
  m.col_lb = q.col_lb;
  m.col_ub = q.col_ub;
  m.row_lb = q.row_lb;
  m.row_ub = q.row_ub;
  m.A = q.A;
  for (Index j = 0; j < q.num_cols(); ++j) {
    m.col_type.push_back(q.is_int[j] ? VarType::Integer : VarType::Continuous);
    m.col_names.push_back("c" + std::to_string(j));
  }
  for (Index i = 0; i < q.num_rows(); ++i) m.row_names.push_back("r" + std::to_string(i));
  return m;
}

bool compatible(Status a, Status b) {
  if (a == b) return true;
  auto unb = [](Status s) { return s == Status::Unbounded || s == Status::InfeasibleOrUnbounded; };
  auto inf = [](Status s) { return s == Status::Infeasible || s == Status::InfeasibleOrUnbounded; };
  return (unb(a) && unb(b)) || (inf(a) && inf(b));
}

PresolveStats g_stats;  // reductions applied by the randomized tests (coverage report)
int g_numeric = 0;      // cases skipped because the LP solver reported numerical trouble
int g_tolerance = 0;    // cases whose objectives differ only through the feasibility tolerances (tolerance_level)

// Two optimal objectives beyond the comparison tolerance still agree when the better point is not exactly
// feasible (it uses the 1e-6 feasibility or integrality tolerance, e.g. a binary at 1e-7 opening a big-M row)
// and gains at most 1e-4 relative: such a point does not prove that the other solve missed a solution.
// An exactly feasible better point does, and stays a failure.
bool tolerance_level(const Model& m, const SolveResult& a, const SolveResult& b) {
  if (!a.has_solution || !b.has_solution) return false;
  if (std::abs(a.objective - b.objective) > 1e-4 * std::max(1.0, std::abs(a.objective))) return false;
  const bool a_better = (m.sense == ObjSense::Minimize) == (a.objective < b.objective);
  return m.violation(a_better ? a.x : b.x).max() > 1e-9;
}

void print_coverage(const char* name) {
  std::printf("  %s reductions: %lld aggregations, %lld substitutions (%lld implied free by other rows), %lld parallel "
              "cols, %lld dominated, probing %lld fixed / %lld bounds / %lld equivs / %lld lifts, cliques %lld extended / "
              "%lld dominated / %lld fixed, %lld rows sparsified, %lld dual aggregations, %lld columns rescaled\n",
              name, g_stats.aggregations, g_stats.substitutions, g_stats.subst_implied, g_stats.parallel_cols, g_stats.dominated,
              g_stats.probing_fixed, g_stats.probing_bounds, g_stats.probing_equivs, g_stats.probing_lifts,
              g_stats.clique_extended, g_stats.clique_dominated, g_stats.clique_fixed, g_stats.sparsify_rows,
              g_stats.dual_aggregations, g_stats.int_divisors);
  if (g_numeric > 0) std::printf("  %s: %d cases skipped (numerical trouble)\n", name, g_numeric);
  if (g_tolerance > 0) std::printf("  %s: %d cases with a tolerance-level objective difference\n", name, g_tolerance);
  g_tolerance = 0;
  g_stats = PresolveStats{};
  g_numeric = 0;
}

// Solves m with and without presolve (presolve + reduced solve + postsolve, without the solver's safety fallback)
// and checks that both agree. Returns the status without presolve (NodeLimit if skipped). sparsify: 0 / 1 off / on,
// -1 off in a third of the cases. base: the presolve options before those variations (default: the defaults).
Status compare_with_presolve(const Model& m, int it, int sparsify = -1, const PresolveOptions* base = nullptr) {
  SolverParams params;
  params.verbosity = 0;
  params.threads = 1;
  params.mip_rel_gap = 0.0;
  params.mip_abs_gap = 1e-9;
  params.node_limit = 20000;  // random models may have unbounded integer domains
  params.presolve = false;
  const SolveResult a = solve(m, params);
  if (a.status == Status::NodeLimit) return a.status;
  if (a.status == Status::NumericalError) return ++g_numeric, Status::NodeLimit;  // LP trouble, not presolve
  SolveResult b;
  MipProblem p;
  Presolver pre;
  const bool valid = MipProblem::from_model(m, &p);
  PresolveOptions opt = base ? *base : PresolveOptions{};
  opt.substitution_bounded = it % 2 == 1;  // the default (off) and the opt-in variant
  opt.probing_cliques = it % 4 >= 2;       // likewise
  opt.sparsify = sparsify < 0 ? it % 3 != 2 : sparsify != 0;
  opt.int_divisors = it % 5 < 3;  // opt-in
  opt.keep_members = it % 3 == 0;  // opt-in
  opt.lift_live = (it / 2) % 2 == 0;  // opt-in
  opt.lift_dyadic = (it / 3) % 2 == 0;  // opt-in
  opt.unit_rows = it % 7 < 4;  // opt-in
  opt.sparsify_protect = (it / 4) % 2 == 0;  // the default (on) and off
  const PresolveStatus ps = valid ? pre.run(p, opt, kQuiet) : PresolveStatus::Infeasible;
  g_stats.sparsify_rows += pre.stats().sparsify_rows;
  g_stats.int_divisors += pre.stats().int_divisors;
  g_stats.aggregations += pre.stats().aggregations;
  g_stats.substitutions += pre.stats().substitutions;
  g_stats.subst_implied += pre.stats().subst_implied;
  g_stats.subst_fill += pre.stats().subst_fill;
  g_stats.probing_fixed += pre.stats().probing_fixed;
  g_stats.probing_bounds += pre.stats().probing_bounds;
  g_stats.probing_equivs += pre.stats().probing_equivs;
  g_stats.probing_lifts += pre.stats().probing_lifts;
  g_stats.parallel_cols += pre.stats().parallel_cols;
  g_stats.dominated += pre.stats().dominated;
  g_stats.dual_aggregations += pre.stats().dual_aggregations;
  g_stats.clique_extended += pre.stats().clique_extended;
  g_stats.clique_dominated += pre.stats().clique_dominated;
  g_stats.clique_fixed += pre.stats().clique_fixed;
  if (ps == PresolveStatus::Infeasible) {
    b.status = Status::Infeasible;
  } else if (ps == PresolveStatus::Unbounded) {
    b.status = Status::InfeasibleOrUnbounded;
  } else if (pre.reduced().num_cols() == 0) {
    b.status = Status::Optimal;
    b.has_solution = true;
    b.x = pre.postsolve({});
  } else {
    b = solve(to_model(pre.reduced()), params);
    if (b.has_solution) b.x = pre.postsolve(b.x);
  }
  if (b.status == Status::NodeLimit) return b.status;
  if (b.status == Status::NumericalError) return ++g_numeric, Status::NodeLimit;
  if (b.has_solution) b.objective = m.objective_value(b.x);
  const bool ok_status = compatible(a.status, b.status);
  CHECK(ok_status);
  // 2e-6: a point at the edge of the 1e-6 feasibility tolerance may be 1e-6 better (MSVC: -1.00000000003e-6 for 0).
  bool same_obj = a.status != Status::Optimal ||
                  std::abs(a.objective - b.objective) <= 2e-6 * std::max(1.0, std::abs(a.objective));
  const bool near_obj = !same_obj && tolerance_level(m, a, b);
  if (near_obj) ++g_tolerance, same_obj = true;
  const char* dump = std::getenv("MILP_FUZZ_DUMP");  // directory for failing cases
  const char* dump_case = std::getenv("MILP_FUZZ_DUMP_CASE");  // with MILP_FUZZ_DUMP: also this case (other platforms)
  const bool violated = a.status == Status::Optimal && b.has_solution && m.violation(b.x).max() > 1e-6;
  if (dump && (!ok_status || !same_obj || violated || (dump_case && std::atoi(dump_case) == it))) {
    write_mps(m, std::string(dump) + "/case" + std::to_string(it) + ".mps");
  }
  if (!ok_status) {
    std::fprintf(stderr, "  case %d: status %s (no presolve) vs %s (presolve)\n", it, to_string(a.status),
                 to_string(b.status));
    return a.status;
  }
  if (a.status == Status::Optimal) {
    REQUIRE(b.has_solution);
    const double tol = (near_obj ? 1e-4 : 2e-6) * std::max(1.0, std::abs(a.objective));  // see same_obj
    if (!CHECK_NEAR(a.objective, b.objective, tol)) std::fprintf(stderr, "  case %d: objective mismatch\n", it);
    const auto v = m.violation(b.x);
    if (!CHECK(v.max() <= 1e-6)) std::fprintf(stderr, "  case %d: violation %g\n", it, v.max());
    // The full pipeline (with its fallback) must agree as well.
    params.presolve = true;
    const SolveResult c = solve(m, params);
    CHECK(c.status == Status::Optimal);
    if (!near_obj && std::abs(a.objective - c.objective) > tol && tolerance_level(m, a, c)) {
      ++g_tolerance;
    } else if (!CHECK_NEAR(a.objective, c.objective, tol)) {
      std::fprintf(stderr, "  case %d: objective mismatch (full pipeline)\n", it);
      if (dump) write_mps(m, std::string(dump) + "/case" + std::to_string(it) + ".mps");
    }
  }
  return a.status;
}

// MILP_FUZZ_SEED / MILP_FUZZ_ITERS override the defaults for longer fuzzing runs.
int fuzz_iters(int def) {
  const char* env = std::getenv("MILP_FUZZ_ITERS");
  return env ? std::atoi(env) : def;
}

unsigned fuzz_seed(unsigned def) {
  const char* env = std::getenv("MILP_FUZZ_SEED");
  return env ? static_cast<unsigned>(std::atoi(env)) : def;
}

// Random binary-heavy models with the structures probing and substitution target: set partitioning rows with
// slack singletons, linking rows sum x_e = k y (continuous x_e), integer doubletons, knapsacks and big-M rows.
Model random_structured_model(std::mt19937& rng) {
  auto uni = [&](int lo, int hi) { return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)); };
  ModelBuilder b;
  const int nb = uni(3, 10), nc = uni(0, 6), ni = uni(0, 3);
  std::vector<Index> bin, cont, gen;
  for (int j = 0; j < nb; ++j) bin.push_back(b.add_col("b" + std::to_string(j), uni(-5, 5), 0, 1, VarType::Binary));
  for (int j = 0; j < nc; ++j) {
    const double ub = uni(0, 3) == 0 ? kInf : uni(1, 4);
    cont.push_back(b.add_col("c" + std::to_string(j), uni(-3, 3) * (uni(0, 2) ? 1.0 : 0.0), uni(0, 3) ? 0 : -2, ub));
  }
  for (int j = 0; j < ni; ++j) {
    gen.push_back(b.add_col("g" + std::to_string(j), uni(-3, 3), uni(-2, 0), uni(1, 6), VarType::Integer));
  }
  auto pick = [&](const std::vector<Index>& v) { return v[uni(0, static_cast<int>(v.size()) - 1)]; };
  const int m = uni(2, 9);
  int r = 0;
  auto add_row = [&](double lo, double hi, const std::vector<std::pair<Index, double>>& coefs) {
    const Index i = b.add_row("r" + std::to_string(r++), lo, hi);
    std::vector<Index> seen;
    for (const auto& [j, v] : coefs) {
      if (std::find(seen.begin(), seen.end(), j) != seen.end()) continue;
      seen.push_back(j);
      b.add_coef(i, j, v);
    }
  };
  for (int i = 0; i < m; ++i) {
    const int kind = uni(0, 7);
    if (kind == 0) {  // set partitioning / packing / covering
      std::vector<std::pair<Index, double>> c;
      for (int k = uni(2, 4); k > 0; --k) c.push_back({pick(bin), 1.0});
      if (!cont.empty() && uni(0, 1)) c.push_back({pick(cont), 1.0});  // slack-like
      const int s = uni(0, 2);
      add_row(s == 2 ? -kInf : 1.0, s == 1 ? kInf : 1.0, c);
    } else if (kind == 1 && !cont.empty()) {  // sum x_e = k y
      std::vector<std::pair<Index, double>> c;
      int k = 0;
      for (int t = uni(1, 3); t > 0; --t) c.push_back({pick(cont), 1.0}), ++k;
      c.push_back({pick(bin), -static_cast<double>(uni(1, k + 1))});
      add_row(0.0, uni(0, 2) ? 0.0 : kInf, c);
    } else if (kind == 2) {  // integer doubleton
      const Index x = gen.empty() || uni(0, 1) ? pick(bin) : pick(gen), y = pick(bin);
      if (x == y) continue;
      add_row(uni(-1, 2), uni(0, 1) ? kInf : 2.0, {{x, uni(1, 2) * (uni(0, 1) ? 1.0 : -1.0)}, {y, uni(1, 3) * 1.0}});
    } else if (kind == 3) {  // implications x_a <= x_b, x_a + x_b <= 1
      const Index x = pick(bin), y = pick(bin);
      if (x == y) continue;
      if (uni(0, 1)) add_row(-kInf, 0.0, {{x, 1.0}, {y, -1.0}}); else add_row(-kInf, 1.0, {{x, 1.0}, {y, 1.0}});
    } else if (kind == 4 && !cont.empty()) {  // big-M: x_c <= M y
      add_row(-kInf, 0.0, {{pick(cont), 1.0}, {pick(bin), -static_cast<double>(uni(1, 6))}});
    } else if (kind == 5 && !cont.empty()) {  // continuous equation (substitution candidates)
      std::vector<std::pair<Index, double>> c;
      for (int t = uni(2, 4); t > 0; --t) c.push_back({uni(0, 1) ? pick(cont) : pick(bin), uni(-3, 3) * 0.5 + 0.25});
      add_row(uni(-1, 3), uni(-1, 3) + uni(0, 1) * 100.0, c);
      // Equality half of the time.
    } else {  // knapsack
      std::vector<std::pair<Index, double>> c;
      double tot = 0.0;
      for (int t = uni(2, 5); t > 0; --t) {
        const double a = uni(1, 7);
        c.push_back({uni(0, 3) || gen.empty() ? pick(bin) : pick(gen), a});
        tot += a;
      }
      add_row(uni(0, 1) ? -kInf : 1.0, std::floor(tot * uni(2, 8) / 10.0), c);
    }
  }
  // Equalities on some continuous rows.
  Model mod = b.build();
  for (Index i = 0; i < mod.num_rows(); ++i) {
    if (uni(0, 3) == 0 && is_finite(mod.row_lb[i])) mod.row_ub[i] = mod.row_lb[i];
  }
  // Parallel copies of some columns (same or negated coefficients, cost proportional or not).
  for (int copies = uni(0, 2); copies > 0 && mod.num_cols() > 0; --copies) {
    const Index src = uni(0, mod.num_cols() - 1);
    const double sgn = uni(0, 2) ? 1.0 : -1.0;
    std::vector<Index> J, I;
    std::vector<double> V;
    for (Index j = 0; j < mod.num_cols(); ++j) {
      for (Index p = mod.A.begin(j); p < mod.A.end(j); ++p) {
        J.push_back(j), I.push_back(mod.A.index[p]), V.push_back(mod.A.value[p]);
      }
    }
    for (Index p = mod.A.begin(src); p < mod.A.end(src); ++p) {
      J.push_back(mod.num_cols()), I.push_back(mod.A.index[p]), V.push_back(sgn * mod.A.value[p]);
    }
    mod.A = SparseMatrix::from_triplets(mod.num_cols() + 1, mod.num_rows(), J, I, V);
    mod.obj.push_back(uni(0, 2) ? sgn * mod.obj[src] : mod.obj[src] + uni(-2, 2));
    mod.col_lb.push_back(uni(0, 1) ? mod.col_lb[src] : 0.0);
    mod.col_ub.push_back(uni(0, 1) ? mod.col_ub[src] : 2.0);
    mod.col_type.push_back(mod.col_type[src]);
    mod.col_names.push_back("p" + std::to_string(copies));
  }
  if (uni(0, 1)) mod.sense = ObjSense::Maximize;
  return mod;
}

}  // namespace

TEST(presolve_random_milps_match_no_presolve) {
  std::mt19937 rng(fuzz_seed(12345u));
  const int iters = fuzz_iters(1500);
  int optimal = 0, infeasible = 0, other = 0;
  for (int it = 0; it < iters; ++it) {
    const Status st = compare_with_presolve(random_model(rng), it);
    if (st == Status::NodeLimit) continue;
    if (st == Status::Optimal) {
      ++optimal;
    } else if (st == Status::Infeasible) {
      ++infeasible;
    } else {
      ++other;
    }
  }
  std::printf("  random presolve: %d optimal, %d infeasible, %d other\n", optimal, infeasible, other);
  print_coverage("random");
  CHECK(optimal > iters / 5);
  CHECK(infeasible > iters / 30);
}

TEST(presolve_random_structured_milps_match_no_presolve) {
  std::mt19937 rng(fuzz_seed(777u));
  const int iters = fuzz_iters(1500);
  int optimal = 0, infeasible = 0, other = 0;
  for (int it = 0; it < iters; ++it) {
    const Status st = compare_with_presolve(random_structured_model(rng), it);
    if (st == Status::NodeLimit) continue;
    if (st == Status::Optimal) {
      ++optimal;
    } else if (st == Status::Infeasible) {
      ++infeasible;
    } else {
      ++other;
    }
  }
  std::printf("  structured presolve: %d optimal, %d infeasible, %d other\n", optimal, infeasible, other);
  print_coverage("structured");
  CHECK(optimal > iters / 5);
}

TEST(presolve_roundoff_regressions) {
  // Wrong answers of the presolve found by the sparsify fuzz test (minimized; also without sparsify).
  // (1) GCD tightening: 5 x8 + 5 x5 in [13.9999999999972, 14.9999999999972] (the coefficient tightening's safety
  // margin on a row with -3.5, carried through a parallel-row merge and two substitutions) was declared infeasible:
  // the lower side was rounded up to 15, the upper one stayed just below it. (2) The same with a margin from a
  // propagated continuous bound. (3) 0.1 - 0.1 = 1.4e-17 left by a substitution for a free integer column became
  // the pivot of the next one (optimum 160.25, presolved 27.25).
  const char* cases[] = {
      "NAME c13\nOBJSENSE\n  MAX\nROWS\n N OBJ\n E r0\n L r5\n L r6\nCOLUMNS\n"
      "    MARK0000 'MARKER' 'INTORG'\n"
      "    x10 OBJ 5\n    x10 r0 -8\n    x10 r6 -8\n    x9 OBJ 2\n    x9 r0 2\n    x9 r5 -1\n    x9 r6 2\n"
      "    x8 OBJ 5\n    x8 r0 5\n    x5 OBJ -5\n    x5 r0 5\n    x2 OBJ 2\n    x2 r0 7\n    x2 r5 -3.5\n    x2 r6 7\n"
      "    MARK0001 'MARKER' 'INTEND'\n"
      "RHS\n    RHS r0 24\n    RHS r5 -8.5\n    RHS r6 10\n"
      "BOUNDS\n BV BND x10\n UI BND x9 6\n BV BND x8\n UI BND x5 5\n UI BND x2 4\nENDATA\n",
      "NAME c21\nROWS\n N OBJ\n E r0\n L r3\n L r4\n L r5\n L r7\nCOLUMNS\n"
      "    x9 OBJ -4\n    x9 r0 5\n    x9 r3 -2.25\n    x9 r4 0.5\n    x9 r5 -0.075\n    x9 r7 -0.75\n"
      "    MARK0000 'MARKER' 'INTORG'\n"
      "    x8 OBJ -2\n    x8 r5 -4\n    x8 r7 -3\n    x7 OBJ 1\n    x7 r0 -1.75\n    x7 r3 -5.25\n    x7 r4 -0.175\n"
      "    x7 r5 -0.175\n    x7 r7 -1.75\n    x5 OBJ -3\n    x5 r5 3\n"
      "    MARK0001 'MARKER' 'INTEND'\n"
      "RHS\n    RHS r0 21.25\n    RHS r3 -6.5625\n    RHS r4 2.125\n    RHS r5 -12.31875\n    RHS r7 -12.1875\n"
      "RANGES\n    RNG r3 6\n    RNG r4 3\n    RNG r5 1\n    RNG r7 1\n"
      "BOUNDS\n UP BND x9 7\n UI BND x8 7\n BV BND x7\n BV BND x5\nENDATA\n",
      "NAME c22\nOBJSENSE\n  MAX\nROWS\n N OBJ\n E r1\n E r2\n E r4\nCOLUMNS\n"
      "    x7 OBJ 3\n    x7 r1 -5\n"
      "    MARK0000 'MARKER' 'INTORG'\n"
      "    x3 OBJ 4\n    x3 r1 -7\n    x3 r2 1\n    x3 r4 0.1\n"
      "    MARK0001 'MARKER' 'INTEND'\n"
      "    x2 OBJ 5\n    x2 r1 8\n    x2 r2 -1.5\n    x2 r4 -0.15\n"
      "    MARK0002 'MARKER' 'INTORG'\n"
      "    x1 OBJ 5\n    x1 r1 1.25\n    x1 r4 1\n"
      "    MARK0003 'MARKER' 'INTEND'\n"
      "RHS\n    RHS r1 -31.75\n    RHS r2 -0.375\n    RHS r4 0.9625\n"
      "BOUNDS\n UP BND x7 7\n LI BND x3 0\n BV BND x1\nENDATA\n",
  };
  const double opt[] = {12.0, -23.0, 160.25};
  for (int k = 0; k < 3; ++k) {
    const Model m = read_mps_string(cases[k]);
    CHECK(compare_with_presolve(m, 0) == Status::Optimal);
    SolverParams sp;
    sp.verbosity = 0;
    sp.threads = 1;
    const SolveResult r = solve(m, sp);
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, opt[k], 1e-6);
  }
}

namespace {

// Random set packing / partitioning models with conflicts hidden in knapsacks and equations (clique extension,
// clique fixings, dominated packing rows).
Model random_clique_model(std::mt19937& rng) {
  auto uni = [&](int lo, int hi) { return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)); };
  ModelBuilder b;
  const int n = uni(5, 12);
  std::vector<Index> x;
  for (int j = 0; j < n; ++j) x.push_back(b.add_col("x" + std::to_string(j), uni(-6, 3), 0, 1, VarType::Binary));
  const Index c = b.add_col("c", uni(-2, 2), 0, uni(1, 3));
  auto pick = [&] { return x[uni(0, n - 1)]; };
  const int m = uni(3, 10);
  for (int i = 0; i < m; ++i) {
    const int kind = uni(0, 5);
    std::vector<std::pair<Index, double>> coefs;
    auto add = [&](Index j, double v) {
      for (const auto& e : coefs) {
        if (e.first == j) return;
      }
      coefs.push_back({j, v});
    };
    double lb = -kInf, ub = kInf;
    if (kind <= 1) {  // packing (sometimes partitioning)
      for (int k = uni(2, 5); k > 0; --k) add(pick(), 1.0);
      ub = 1.0;
      if (uni(0, 3) == 0) lb = 1.0;
    } else if (kind == 2) {  // pairwise conflict
      add(pick(), 1.0);
      add(pick(), 1.0);
      ub = 1.0;
    } else if (kind == 3) {  // knapsack with large coefficients
      double tot = 0.0;
      for (int k = uni(2, 5); k > 0; --k) {
        const double a = uni(2, 9);
        add(pick(), a);
        tot += a;
      }
      if (uni(0, 1)) add(c, uni(1, 3));
      ub = std::floor(tot * uni(3, 7) / 10.0);
    } else if (kind == 4) {  // x_a + x_b <= x_c
      add(pick(), 1.0);
      add(pick(), 1.0);
      add(pick(), -1.0);
      ub = 0.0;
    } else {  // covering
      for (int k = uni(2, 4); k > 0; --k) add(pick(), 1.0);
      lb = 1.0;
    }
    const Index r = b.add_row("r" + std::to_string(i), lb, ub);
    for (const auto& [j, v] : coefs) b.add_coef(r, j, v);
  }
  if (uni(0, 1)) b.model().sense = ObjSense::Maximize;
  return b.build();
}

}  // namespace

TEST(presolve_random_clique_models_match_no_presolve) {
  std::mt19937 rng(fuzz_seed(4242u));
  const int iters = fuzz_iters(1000);
  int optimal = 0, infeasible = 0;
  for (int it = 0; it < iters; ++it) {
    const Status st = compare_with_presolve(random_clique_model(rng), it);
    optimal += st == Status::Optimal;
    infeasible += st == Status::Infeasible;
  }
  std::printf("  clique presolve: %d optimal, %d infeasible\n", optimal, infeasible);
  const long long ext = g_stats.clique_extended, fixed = g_stats.clique_fixed;
  print_coverage("clique");
  CHECK(optimal > iters / 3);
  CHECK(ext > 0);
  CHECK(fixed > 0);
}

// The search's answer on the presolved problem, mapped back (solver/postsolve.hpp). binkar10_1: penalty
// columns (cost 1e6) are column singletons of equations s = z - a; presolve substitutes them out and
// their bound s >= 0 becomes the row z - a >= 0, which an LP point may violate within the feasibility
// tolerance. Postsolve clamps s into its bound, which moves that violation into the objective: the
// postsolved incumbent came out 1e6 * 2.5e-8 worse than the search's, beyond the gap, as "optimal".
namespace {

struct PenaltyModel {
  MipProblem p;
  Index z1 = -1, z2 = -1, a1 = -1, a2 = -1, s1 = -1, s2 = -1;
};

// min -a1 - a2 + 1e6 (s1 + s2): s_k = z_k - a_k >= 0, a1 + a2 <= 1.5, z binary, a in [0, 1]. Optimum -1.
PenaltyModel penalty_model() {
  Builder b;
  PenaltyModel m;
  m.z1 = b.col(0.0, 0.0, 1.0, true);
  m.z2 = b.col(0.0, 0.0, 1.0, true);
  m.a1 = b.col(-1.0, 0.0, 1.0);
  m.a2 = b.col(-1.0, 0.0, 1.0);
  m.s1 = b.col(1e6, 0.0, kInf);
  m.s2 = b.col(1e6, 0.0, kInf);
  b.row(0.0, 0.0, {{m.z1, 1.0}, {m.a1, -1.0}, {m.s1, -1.0}});
  b.row(0.0, 0.0, {{m.z2, 1.0}, {m.a2, -1.0}, {m.s2, -1.0}});
  b.row(-kInf, 1.5, {{m.a1, 1.0}, {m.a2, 1.0}});
  m.p = b.build();
  return m;
}

// Presolves m (substitution only: s1, s2 go) and returns the search's answer at the reduced point
// z = (z1, z2), a = (a1, a2), claimed optimal.
MipResult penalty_answer(const PenaltyModel& m, Presolver& pre, double z1, double z2, double a1, double a2) {
  PresolveOptions o = only(false, false, false, false, false);
  o.substitution = true;
  REQUIRE(pre.run(m.p, o, kQuiet) == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  REQUIRE(reduced_index(pre, m.s1) == -1);
  REQUIRE(reduced_index(pre, m.s2) == -1);
  REQUIRE(q.num_cols() == 4);
  std::vector<double> xr(4, 0.0);
  xr[reduced_index(pre, m.z1)] = z1;
  xr[reduced_index(pre, m.z2)] = z2;
  xr[reduced_index(pre, m.a1)] = a1;
  xr[reduced_index(pre, m.a2)] = a2;
  MipResult mr;
  mr.status = Status::Optimal;
  mr.has_solution = true;
  mr.x = xr;
  mr.objective = mr.bound = q.objective(xr);
  return mr;
}

}  // namespace

namespace {

// Random models with implied-integer structure: continuous columns defined by equations over integers
// (chains through other continuous columns), makespan-like inequalities, some fractional coefficients
// (no implied integer) and integral or half-integral costs.
Model random_implint_model(std::mt19937& rng) {
  auto uni = [&](int lo, int hi) { return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)); };
  ModelBuilder b;
  const int ni = uni(2, 6), nc = uni(1, 4);
  std::vector<Index> xi, xc;
  for (int j = 0; j < ni; ++j) {
    xi.push_back(b.add_col("i" + std::to_string(j), uni(-3, 3), uni(-2, 0), uni(1, 4), VarType::Integer));
  }
  for (int j = 0; j < nc; ++j) {
    const double lb = uni(0, 4) == 0 ? -kInf : (uni(0, 3) == 0 ? -1.5 : 0.0);
    const double ub = uni(0, 2) == 0 ? kInf : uni(5, 20) + 0.5 * uni(0, 1);
    xc.push_back(b.add_col("c" + std::to_string(j), uni(-1, 3) * (uni(0, 3) ? 1.0 : 0.5), lb, ub));
  }
  auto pick = [&](const std::vector<Index>& v) { return v[uni(0, static_cast<int>(v.size()) - 1)]; };
  int r = 0;
  auto add_row = [&](double lo, double hi, const std::vector<std::pair<Index, double>>& coefs) {
    const Index i = b.add_row("r" + std::to_string(r++), lo, hi);
    std::vector<Index> seen;
    for (const auto& [j, v] : coefs) {
      if (std::find(seen.begin(), seen.end(), j) != seen.end()) continue;
      seen.push_back(j);
      b.add_coef(i, j, v);
    }
  };
  for (int k = 0; k < nc; ++k) {
    const Index c = xc[k];
    const double a = uni(0, 2) == 0 ? 2.0 : 1.0;
    const int kind = uni(0, 3);
    std::vector<std::pair<Index, double>> coefs = {{c, a}};
    for (int t = uni(1, 3); t > 0; --t) coefs.push_back({pick(xi), uni(-2, 2) * a * (uni(0, 5) ? 1.0 : 0.5)});
    if (k > 0 && uni(0, 1)) coefs.push_back({xc[k - 1], -a});  // chain
    const double side = a * uni(-3, 3) + (uni(0, 6) ? 0.0 : 0.5);
    if (kind <= 1) {
      add_row(side, side, coefs);
    } else if (kind == 2) {  // makespan: c >= ... (several rows)
      add_row(side, kInf, coefs);
      add_row(side - a * uni(0, 2), kInf, {{c, a}, {pick(xi), -a}});
    } else {
      add_row(-kInf, side + a * 4, coefs);
    }
  }
  for (int t = uni(0, 3); t > 0; --t) {  // knapsacks on the integers
    std::vector<std::pair<Index, double>> coefs;
    for (int q = uni(2, 4); q > 0; --q) coefs.push_back({pick(xi), static_cast<double>(uni(1, 5))});
    add_row(-kInf, uni(2, 10), coefs);
  }
  if (uni(0, 1)) b.model().sense = ObjSense::Maximize;
  return b.build();
}

}  // namespace

TEST(postsolve_result_reoptimizes_objective_lost_to_clamping) {
  const PenaltyModel m = penalty_model();
  Presolver pre;
  // An optimal LP point within the tolerance: a1 = d puts s1 = -d (z1 - a1 >= 0 violated by d), a2 = 1 - d
  // puts s2 = d; objective -1, the optimum (-1e6 d and +1e6 d cancel).
  const double d = 1e-7;
  MipResult mr = penalty_answer(m, pre, 0.0, 1.0, d, 1.0 - d);
  CHECK(pre.reduced().max_violation(mr.x) <= 1e-6);
  CHECK_NEAR(mr.objective, -1.0, 1e-6);
  // Postsolve clamps s1 to 0 and keeps s2 = d: -1 + 1e6 d, feasible within the tolerance.
  const std::vector<double> xs = pre.postsolve(mr.x);
  CHECK_NEAR(m.p.objective(xs), -0.9, 1e-6);
  CHECK(m.p.max_violation(xs) <= 1e-6);
  SolverParams params;
  params.mip_rel_gap = 1e-6;
  const Timer timer;
  REQUIRE(postsolve_result(m.p, pre, params, timer, kQuiet, &mr));
  CHECK(mr.has_solution);
  CHECK_NEAR(mr.objective, -1.0, 1e-9);
  CHECK_NEAR(m.p.objective(mr.x), mr.objective, 1e-12);
  CHECK(m.p.max_violation(mr.x) <= 1e-9);
  CHECK(mr.bound <= mr.objective);
}

TEST(postsolve_result_repairs_or_rejects_infeasible_points) {
  // As above with d = 3e-6: the clamp leaves the postsolved point 3e-6 outside the row of s1, beyond the
  // tolerance. Repaired by the LP when that keeps the search's objective (relative 1e-6) ...
  const PenaltyModel m = penalty_model();
  const double d = 3e-6;
  SolverParams params;
  const Timer timer;
  Presolver pre;
  MipResult mr = penalty_answer(m, pre, 0.0, 1.0, d, 1.0 - d);
  CHECK(m.p.max_violation(pre.postsolve(mr.x)) > 1e-6);
  REQUIRE(postsolve_result(m.p, pre, params, timer, kQuiet, &mr));
  CHECK_NEAR(mr.objective, -1.0, 1e-9);
  CHECK(m.p.max_violation(mr.x) <= 1e-9);
  // ... else dropped (the caller solves without presolve): z = (0, 0), a1 = d claims -(1e6 + 1) d, the
  // repaired point is worth 0.
  Presolver pre2;
  MipResult mr2 = penalty_answer(m, pre2, 0.0, 0.0, d, 0.0);
  CHECK_NEAR(mr2.objective, -3.0, 1e-4);
  CHECK(!postsolve_result(m.p, pre2, params, timer, kQuiet, &mr2));
  CHECK(!mr2.has_solution);
}

TEST(presolve_random_implied_integer_models) {
  // Presolve on / off agree, and an independent reference without objective integrality (costs divided by 3,
  // so the objective is never integral) gives the same optimum: rounding the bound by the implied integers'
  // objective integrality must not cut off the optimum.
  std::mt19937 rng(fuzz_seed(2026u));
  const int iters = fuzz_iters(800);
  int optimal = 0;
  long long implied = 0;
  SolverParams params;
  params.verbosity = 0;
  params.threads = 1;
  params.mip_rel_gap = 0.0;
  params.mip_abs_gap = 1e-9;
  params.node_limit = 20000;
  for (int it = 0; it < iters; ++it) {
    const Model m = random_implint_model(rng);
    const Status st = compare_with_presolve(m, it);
    MipProblem p;
    REQUIRE(MipProblem::from_model(m, &p));
    for (char c : detect_implied_integers(p)) implied += c;
    if (st != Status::Optimal) continue;
    ++optimal;
    const SolveResult a = solve(m, params);
    Model m3 = m;
    for (double& c : m3.obj) c /= 3.0;
    const SolveResult ref = solve(m3, params);
    REQUIRE(a.status == Status::Optimal);
    if (ref.status != Status::Optimal) continue;  // node limit
    if (!CHECK_NEAR(a.objective, 3.0 * ref.objective, 1e-6 * std::max(1.0, std::abs(a.objective)))) {
      std::fprintf(stderr, "  case %d: objective %.9g vs reference %.9g\n", it, a.objective, 3.0 * ref.objective);
    }
  }
  std::printf("  implied-integer models: %d optimal, %lld implied integers\n", optimal, implied);
  print_coverage("implint");
  CHECK(optimal > iters / 4);
  CHECK(implied > iters / 2);
}

namespace {

// Random pure-integer models small enough to enumerate (4-10 binaries, at most two integers in [0, 3]): set
// packing / partitioning / covering rows, knapsacks, implications, big-M rows, cardinality rows and rows with
// mixed signs; mostly one-sided (the rows probing lifts).
MipProblem random_enum_model(std::mt19937& rng) {
  auto uni = [&](int lo, int hi) { return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)); };
  Builder b;
  std::vector<Index> bin, gen, all;
  for (int j = uni(4, 10); j > 0; --j) bin.push_back(b.col(uni(-5, 5), 0.0, 1.0, true));
  for (int j = uni(0, 2); j > 0; --j) gen.push_back(b.col(uni(-3, 3), 0.0, uni(1, 3), true));
  all = bin;
  all.insert(all.end(), gen.begin(), gen.end());
  std::vector<double> ub(all.size() + 1, 1.0);
  for (Index g : gen) ub[g] = b.b.model().col_ub[g];
  auto pick = [&](const std::vector<Index>& v) { return v[uni(0, static_cast<int>(v.size()) - 1)]; };
  auto row = [&](double lo, double hi, const std::vector<std::pair<Index, double>>& c) {
    std::vector<std::pair<Index, double>> u;
    for (const auto& e : c) {
      if (std::none_of(u.begin(), u.end(), [&](const auto& f) { return f.first == e.first; })) u.push_back(e);
    }
    if (u.size() >= 2) b.row(lo, hi, u);
  };
  for (int i = uni(2, 9); i > 0; --i) {
    const int kind = uni(0, 5);
    std::vector<std::pair<Index, double>> c;
    if (kind == 0) {  // set packing / partitioning / covering
      for (int k = uni(2, 5); k > 0; --k) c.push_back({pick(bin), 1.0});
      const int s = uni(0, 3);
      row(s == 2 ? 1.0 : -kInf, s == 3 ? kInf : 1.0, c);
    } else if (kind == 1) {  // knapsack (or its covering counterpart)
      double tot = 0.0;
      for (int k = uni(2, 6); k > 0; --k) {
        const double a = uni(1, 9);
        c.push_back({pick(all), a});
        tot += a;
      }
      const double rhs = std::floor(tot * uni(3, 8) / 10.0);
      if (uni(0, 3)) row(-kInf, rhs, c); else row(rhs, kInf, c);
    } else if (kind == 2) {  // x <= y, x + y <= 1, x + y >= 1
      const int t = uni(0, 2);
      row(t == 2 ? 1.0 : -kInf, t == 0 ? 0.0 : (t == 1 ? 1.0 : kInf), {{pick(bin), 1.0}, {pick(bin), t == 0 ? -1.0 : 1.0}});
    } else if (kind == 3 && !gen.empty()) {  // big-M: g <= M x
      row(-kInf, 0.0, {{pick(gen), 1.0}, {pick(bin), -static_cast<double>(uni(2, 9))}});
    } else if (kind == 4) {  // cardinality
      for (int k = uni(3, 6); k > 0; --k) c.push_back({pick(bin), 1.0});
      if (uni(0, 1)) row(-kInf, 2.0, c); else row(2.0, kInf, c);
    } else {  // mixed signs, one-sided (sometimes an equation)
      double lo = 0.0, hi = 0.0;
      for (int k = uni(2, 5); k > 0; --k) {
        const Index j = pick(all);
        double a = uni(-4, 4);
        if (a == 0.0) a = 1.0;
        c.push_back({j, a});
        (a > 0 ? hi : lo) += a * ub[j];
      }
      const double rhs = std::round(lo + (hi - lo) * uni(2, 7) / 10.0);
      const int s = uni(0, 5);
      row(s >= 3 ? rhs : -kInf, s == 5 || s < 3 ? rhs : kInf, c);
    }
  }
  return b.build();
}

// The integer points within the bounds of p satisfying its rows (absolute tolerance 1e-9).
std::vector<std::vector<double>> enumerate_feasible(const MipProblem& p) {
  const Index n = p.num_cols();
  const SparseMatrix R = p.A.transpose();
  std::vector<std::vector<double>> out;
  std::vector<double> x(n);
  for (Index j = 0; j < n; ++j) x[j] = p.col_lb[j];
  for (;;) {
    bool ok = true;
    for (Index i = 0; i < p.num_rows() && ok; ++i) {
      double act = 0.0;
      for (Index k = R.begin(i); k < R.end(i); ++k) act += R.value[k] * x[R.index[k]];
      ok = act >= p.row_lb[i] - 1e-9 && act <= p.row_ub[i] + 1e-9;
    }
    if (ok) out.push_back(x);
    Index j = 0;
    while (j < n && x[j] >= p.col_ub[j]) x[j] = p.col_lb[j], ++j;
    if (j == n) break;
    x[j] += 1.0;
  }
  return out;
}

// A small model with one-of groups and rows over them, to enumerate: groups of binaries with a partitioning or
// packing row each (some binaries in two such rows, some in none), integer columns linked to a group by an
// equation  sum_k a_k x_k - y = c  (a start time), precedence-like rows between integer columns, capacity rows
// over members of several groups against an integer level with a cost, and a few rows with mixed signs.
MipProblem random_group_model(std::mt19937& rng) {
  auto uni = [&](int lo, int hi) { return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)); };
  Builder b;
  std::vector<std::vector<Index>> groups;
  std::vector<Index> bin, gen;
  const int ng = uni(1, 3);
  for (int g = 0; g < ng; ++g) {
    groups.emplace_back();
    for (int k = uni(2, 4); k > 0; --k) {
      groups.back().push_back(b.col(uni(0, 3) ? 0.0 : uni(-2, 2), 0.0, 1.0, true));
      bin.push_back(groups.back().back());
    }
  }
  for (int k = uni(0, 2); k > 0; --k) bin.push_back(b.col(uni(-2, 2), 0.0, 1.0, true));
  auto pick = [&](const std::vector<Index>& v) { return v[uni(0, static_cast<int>(v.size()) - 1)]; };
  for (int g = 0; g < ng; ++g) {
    std::vector<std::pair<Index, double>> c;
    for (Index j : groups[g]) c.push_back({j, 1.0});
    if (uni(0, 4) == 0) {  // a member of another group or a loose binary joins the row
      const Index j = pick(bin);
      if (std::none_of(c.begin(), c.end(), [&](const auto& e) { return e.first == j; })) c.push_back({j, 1.0});
    }
    b.row(uni(0, 3) ? 1.0 : -kInf, 1.0, c);
  }
  // Linked integer columns.
  for (int g = 0; g < ng; ++g) {
    if (uni(0, 3) == 0) continue;
    const Index y = b.col(uni(0, 2) ? 0.0 : uni(-2, 2), uni(0, 1), uni(2, 6), true);
    gen.push_back(y);
    std::vector<std::pair<Index, double>> c;
    for (Index j : groups[g]) {
      if (uni(0, 5)) c.push_back({j, static_cast<double>(uni(-1, 6))});
    }
    c.erase(std::remove_if(c.begin(), c.end(), [](const auto& e) { return e.second == 0.0; }), c.end());
    if (c.empty()) continue;
    c.push_back({y, -1.0});
    const int s = uni(0, 5);
    const double rhs = uni(-1, 1);
    b.row(s == 4 ? -kInf : rhs, s == 5 ? kInf : rhs, c);
  }
  // Levels with a cost and capacity rows against them.
  for (int k = uni(0, 2); k > 0; --k) {
    const Index lvl = b.col(uni(1, 5), 0.0, uni(1, 4), true);
    gen.push_back(lvl);
    for (int r = uni(1, 3); r > 0; --r) {
      std::vector<std::pair<Index, double>> c;
      for (Index j : bin) {
        if (uni(0, 2) == 0) c.push_back({j, static_cast<double>(uni(1, 3))});
      }
      if (c.empty()) continue;
      c.push_back({lvl, -1.0});
      b.row(-kInf, 0.0, c);
    }
  }
  // Precedence-like rows between the integer columns, and rows with mixed signs over everything.
  std::vector<Index> all = bin;
  all.insert(all.end(), gen.begin(), gen.end());
  for (int r = uni(0, 3); r > 0 && gen.size() >= 2; --r) {
    const Index u = pick(gen), v = pick(gen);
    if (u != v) b.row(-kInf, -static_cast<double>(uni(0, 2)), {{u, 1.0}, {v, -1.0}});
  }
  for (int r = uni(0, 3); r > 0; --r) {
    std::vector<std::pair<Index, double>> c;
    for (int k = uni(2, 5); k > 0; --k) {
      const Index j = pick(all);
      double a = uni(-3, 4);
      if (a == 0.0) a = 2.0;
      if (std::none_of(c.begin(), c.end(), [&](const auto& e) { return e.first == j; })) c.push_back({j, a});
    }
    if (c.size() < 2) continue;
    const double rhs = uni(-2, 5);
    const int s = uni(0, 4);
    b.row(s >= 3 ? rhs : -kInf, s == 4 || s < 3 ? rhs : kInf, c);
  }
  return b.build();
}

}  // namespace

TEST(group_prop_keeps_feasible_points) {
  // One-of group propagation and shaving on enumerable models: every feasible point lies within the derived
  // bounds, and "infeasible" only where there is none.
  std::mt19937 rng(fuzz_seed(77123u));
  const int iters = fuzz_iters(6000);
  long long groups = 0, fixed = 0, bounds = 0, shaved = 0, infeasible = 0, with_points = 0;
  for (int it = 0; it < iters; ++it) {
    const MipProblem p = it % 3 == 0 ? random_enum_model(rng) : random_group_model(rng);
    const auto pts = enumerate_feasible(p);
    with_points += !pts.empty();
    for (int mode = 0; mode < 2; ++mode) {
      GroupPropOptions go;
      go.shave = mode == 1;
      const GroupPropResult gr = group_propagate(p, go);
      if (gr.infeasible) {
        if (mode == 1) ++infeasible;
        if (!CHECK(pts.empty())) std::fprintf(stderr, "  case %d mode %d: infeasible, %zu feasible points\n", it, mode, pts.size());
        continue;
      }
      REQUIRE(gr.lb.size() == static_cast<size_t>(p.num_cols()) && gr.ub.size() == gr.lb.size());
      if (mode == 1) groups += gr.groups, fixed += gr.fixed, bounds += gr.tightened, shaved += gr.shaved;
      bool ok = true;
      for (Index j = 0; j < p.num_cols() && ok; ++j) ok = gr.lb[j] >= p.col_lb[j] && gr.ub[j] <= p.col_ub[j];
      CHECK(ok);  // bounds only tighten
      for (const auto& x : pts) {
        bool in = true;
        for (Index j = 0; j < p.num_cols() && in; ++j) in = x[j] >= gr.lb[j] && x[j] <= gr.ub[j];
        if (!CHECK(in)) {
          std::fprintf(stderr, "  case %d mode %d: a feasible point is outside the derived bounds\n", it, mode);
          break;
        }
      }
    }
  }
  std::printf("  group propagation: %lld groups, %lld members fixed, %lld bounds, %lld shaving steps, %lld infeasible, "
              "%lld of %d models with a feasible point\n", groups, fixed, bounds, shaved, infeasible, with_points, iters);
  CHECK(groups > 0 && fixed > 0 && bounds > 0 && shaved > 0 && infeasible > 0 && with_points > iters / 10);
}

TEST(presolve_probing_equivalence_roundoff_regression) {
  // Fuzz seed 22, case 1184 of the sparsify models: probing reported the continuous column x5, which both probes
  // leave within 2.5e-10 of 0.5, as "x5 = 0.5 + 3.4e-11 x8"; the substitution put 3.4e-11 into a row (optimum
  // -1.5, the presolved solve returned a point violating a row by 1.33).
  const Model m = read_mps_string(
      "NAME\n"
      "OBJSENSE\n"
      "    MAX\n"
      "ROWS\n"
      " N  OBJ\n"
      " E  r0\n"
      " E  r1\n"
      " G  r2\n"
      " L  r3\n"
      " G  r4\n"
      "COLUMNS\n"
      "    MARKER0  'MARKER'  'INTORG'\n"
      "    x9  OBJ  -5\n"
      "    x8  r1  -5\n"
      "    x8  r2  2.5\n"
      "    x7  OBJ  -4\n"
      "    x7  r4  -4\n"
      "    x6  OBJ  -3\n"
      "    x6  r4  4\n"
      "    MARKER1  'MARKER'  'INTEND'\n"
      "    x5  OBJ  -1\n"
      "    x5  r0  8\n"
      "    x5  r1  1\n"
      "    x5  r2  -0.5\n"
      "    x5  r3  -4\n"
      "    x5  r4  8\n"
      "    MARKER2  'MARKER'  'INTORG'\n"
      "    x4  OBJ  -3\n"
      "    x4  r1  2\n"
      "    x4  r2  -1\n"
      "    x4  r4  4\n"
      "    x3  OBJ  5\n"
      "    x3  r1  7\n"
      "    x3  r2  -3.5\n"
      "    x2  OBJ  2\n"
      "    x2  r0  4\n"
      "    x2  r3  -2\n"
      "    x1  OBJ  -2\n"
      "    x1  r0  4\n"
      "    x1  r3  -2\n"
      "    x1  r4  4\n"
      "    MARKER3  'MARKER'  'INTEND'\n"
      "RHS\n"
      "    RHS  r0  20\n"
      "    RHS  r1  4.5\n"
      "    RHS  r2  -3.25\n"
      "    RHS  r3  -7\n"
      "    RHS  r4  19\n"
      "RANGES\n"
      "    RNG  r4  3\n"
      "BOUNDS\n"
      " BV BND  x9\n"
      " UP BND  x8  7\n"
      " BV BND  x7\n"
      " BV BND  x6\n"
      " UP BND  x5  2\n"
      " UP BND  x4  2\n"
      " BV BND  x3\n"
      " UP BND  x2  6\n"
      " UP BND  x1  2\n"
      "ENDATA\n");
  CHECK(compare_with_presolve(m, 0) == Status::Optimal);
}

TEST(group_prop_start_time_windows) {
  // Two jobs, each with three start options (one-of rows), start columns S = sum t x_t, job 2 after job 1
  // finishes (duration 2): S2 >= S1 + 2. Plain propagation fixes nothing; the windows leave job 1 the starts
  // {0, 1} and job 2 the starts {2, 3}.
  Builder b;
  std::vector<Index> x1, x2;
  for (int t = 0; t < 3; ++t) x1.push_back(b.col(0.0, 0.0, 1.0, true));  // starts 0, 1, 2
  for (int t = 0; t < 3; ++t) x2.push_back(b.col(0.0, 0.0, 1.0, true));  // starts 1, 2, 3
  const Index s1 = b.col(0.0, 0.0, 10.0, true), s2 = b.col(0.0, 0.0, 10.0, true);
  b.row(1.0, 1.0, {{x1[0], 1.0}, {x1[1], 1.0}, {x1[2], 1.0}});
  b.row(1.0, 1.0, {{x2[0], 1.0}, {x2[1], 1.0}, {x2[2], 1.0}});
  b.row(0.0, 0.0, {{x1[1], 1.0}, {x1[2], 2.0}, {s1, -1.0}});                 // start 0 has no entry
  b.row(0.0, 0.0, {{x2[0], 1.0}, {x2[1], 2.0}, {x2[2], 3.0}, {s2, -1.0}});
  b.row(-kInf, -2.0, {{s1, 1.0}, {s2, -1.0}});
  const MipProblem p = b.build();
  const GroupPropResult gr = group_propagate(p);
  REQUIRE(!gr.infeasible);
  CHECK(gr.groups == 2);
  CHECK(gr.lb[s1] == 0.0 && gr.ub[s1] == 1.0);
  CHECK(gr.lb[s2] == 2.0 && gr.ub[s2] == 3.0);
  CHECK(gr.ub[x1[2]] == 0.0 && gr.ub[x1[0]] == 1.0 && gr.ub[x1[1]] == 1.0);
  CHECK(gr.ub[x2[0]] == 0.0 && gr.ub[x2[1]] == 1.0 && gr.ub[x2[2]] == 1.0);
}

TEST(group_prop_shaves_a_level) {
  // Three jobs with one option each of 2 units and one of 1 unit at the same time slot, level L >= usage, cost on
  // L: every choice needs at least 3 units in the slot, the LP and plain propagation allow L = 0.
  Builder b;
  const Index L = b.col(1.0, 0.0, 9.0, true);
  std::vector<std::pair<Index, double>> cap;
  for (int j = 0; j < 3; ++j) {
    const Index a = b.col(0.0, 0.0, 1.0, true), c = b.col(0.0, 0.0, 1.0, true);
    b.row(1.0, 1.0, {{a, 1.0}, {c, 1.0}});
    cap.push_back({a, 2.0});
    cap.push_back({c, 1.0});
  }
  cap.push_back({L, -1.0});
  b.row(-kInf, 0.0, cap);
  const MipProblem p = b.build();
  const GroupPropResult gr = group_propagate(p);
  REQUIRE(!gr.infeasible);
  CHECK(gr.lb[L] == 3.0);
}

TEST(probing_lifting_keeps_feasible_points) {
  // Probing results on enumerable models: every feasible point satisfies the derived bounds, equivalences and
  // conflicts and each lifted row (liftings of different rows hold together: at most one per row).
  std::mt19937 rng(fuzz_seed(4242u));
  const int iters = fuzz_iters(2000);
  long long lifts = 0, fills = 0, conflicts = 0, equivs = 0, infeasible = 0;
  for (int it = 0; it < iters; ++it) {
    const MipProblem p = random_enum_model(rng);
    const auto pts = enumerate_feasible(p);
    // Mode 0: one candidate at a time. Modes 1 and 2: batches of 4 candidates on 1 and on 4 threads (the same
    // result on any thread count); mode 3: batches of 3 on 2 threads, the helpers used for every batch.
    ProbingResult batched1;
    for (int mode = 0; mode < 4; ++mode) {
    ProbingOptions po;
    po.lift = true;
    po.max_conflicts = 100000;
    po.batch = mode == 0 ? 0 : mode == 3 ? 3 : 4;
    po.threads = mode == 2 ? 4 : mode == 3 ? 2 : 1;
    if (mode >= 2) po.par_min = 0;
    const ProbingResult pr = probe_binaries(p, 10000000, po);
    if (mode == 1) batched1 = pr;
    if (mode == 2) {
      bool same = pr.infeasible == batched1.infeasible && pr.probes == batched1.probes && pr.work == batched1.work &&
                  pr.lb == batched1.lb && pr.ub == batched1.ub && pr.conflicts == batched1.conflicts &&
                  pr.equiv.size() == batched1.equiv.size() && pr.lifts.size() == batched1.lifts.size();
      for (size_t t = 0; same && t < pr.equiv.size(); ++t) {
        same = pr.equiv[t].y == batched1.equiv[t].y && pr.equiv[t].x == batched1.equiv[t].x &&
               pr.equiv[t].v0 == batched1.equiv[t].v0 && pr.equiv[t].v1 == batched1.equiv[t].v1;
      }
      for (size_t t = 0; same && t < pr.lifts.size(); ++t) {
        same = pr.lifts[t].row == batched1.lifts[t].row && pr.lifts[t].col == batched1.lifts[t].col &&
               pr.lifts[t].delta == batched1.lifts[t].delta && pr.lifts[t].one == batched1.lifts[t].one;
      }
      if (!CHECK(same)) std::fprintf(stderr, "  case %d: batched probing differs between 1 and 4 threads\n", it);
    }
    if (pr.infeasible) {
      if (mode == 0) ++infeasible;
      if (!CHECK(pts.empty())) std::fprintf(stderr, "  case %d mode %d: probing infeasible, %zu feasible points\n", it, mode, pts.size());
      continue;
    }
    bool ok = true;
    for (const auto& x : pts) {
      for (Index j = 0; j < p.num_cols(); ++j) ok = ok && x[j] >= pr.lb[j] - 1e-9 && x[j] <= pr.ub[j] + 1e-9;
      for (const ProbingResult::Equiv& e : pr.equiv) ok = ok && std::abs(x[e.y] - (e.v0 + (e.v1 - e.v0) * x[e.x])) <= 1e-9;
      for (const auto& [a, c] : pr.conflicts) ok = ok && !(x[a >> 1] == (a & 1) && x[c >> 1] == (c & 1));
    }
    if (!CHECK(ok)) std::fprintf(stderr, "  case %d: bounds / equivalences / conflicts cut off a feasible point\n", it);
    const SparseMatrix R = p.A.transpose();
    std::vector<char> lifted(p.num_rows(), 0);
    for (const ProbingResult::Lift& l : pr.lifts) {
      CHECK(!lifted[l.row]);
      lifted[l.row] = 1;
      // The row in <= form with the lifting applied.
      const double s = l.upper ? 1.0 : -1.0;
      std::vector<double> a(p.num_cols(), 0.0);
      for (Index k = R.begin(l.row); k < R.end(l.row); ++k) a[R.index[k]] = s * R.value[k];
      CHECK(l.fill == (a[l.col] == 0.0));
      double rhs = l.upper ? p.row_ub[l.row] : -p.row_lb[l.row];
      a[l.col] += l.one ? l.delta : -l.delta;
      if (!l.one) rhs -= l.delta;
      bool row_ok = true;
      for (const auto& x : pts) {
        double act = 0.0;
        for (Index j = 0; j < p.num_cols(); ++j) act += a[j] * x[j];
        row_ok = row_ok && act <= rhs + 1e-9;
      }
      if (!CHECK(row_ok)) std::fprintf(stderr, "  case %d: lifting of row %d by col %d cuts off a feasible point\n", it, l.row, l.col);
      if (mode == 0) ++lifts, fills += l.fill;
    }
    if (mode == 0) conflicts += static_cast<long long>(pr.conflicts.size()), equivs += static_cast<long long>(pr.equiv.size());
    }
  }
  std::printf("  probing: %lld liftings (%lld new entries), %lld conflicts, %lld equivalences, %lld infeasible\n", lifts,
              fills, conflicts, equivs, infeasible);
  CHECK(lifts > iters / 4);
  CHECK(fills > 0);
  CHECK(conflicts > iters);
}

TEST(presolve_probing_round_trip_enumerated) {
  // Presolve with probing (liftings, passes, kept implications; clique extension): the feasible points of the
  // original problem map exactly onto those of the reduced one (crush), and back (postsolve).
  std::mt19937 rng(fuzz_seed(777777u));
  const int iters = fuzz_iters(1500);
  long long lifts = 0, compared = 0;
  for (int it = 0; it < iters; ++it) {
    const MipProblem p = random_enum_model(rng);
    const auto pts = enumerate_feasible(p);
    PresolveOptions o = only_probing();
    o.probing_cliques = it % 2 == 1;
    Presolver pre;
    const PresolveStatus st = pre.run(p, o, kQuiet);
    if (st == PresolveStatus::Infeasible) {
      if (!CHECK(pts.empty())) std::fprintf(stderr, "  case %d: presolve infeasible, %zu feasible points\n", it, pts.size());
      continue;
    }
    REQUIRE(st == PresolveStatus::Reduced);
    lifts += pre.stats().probing_lifts;
    const MipProblem& q = pre.reduced();
    std::set<std::vector<double>> image;
    bool ok = true;
    for (const auto& x : pts) {
      const std::vector<double> z = pre.crush(x);
      ok = ok && q.max_violation(z) <= 1e-9;
      image.insert(z);
    }
    if (!CHECK(ok)) std::fprintf(stderr, "  case %d: presolve cut off a feasible point\n", it);
    if (q.num_cols() > 14) continue;
    const auto red = enumerate_feasible(q);
    ++compared;
    if (!CHECK_EQ(red.size(), image.size())) {
      std::fprintf(stderr, "  case %d: %zu reduced vs %zu original points\n", it, red.size(), image.size());
    }
    bool back = true;
    for (const auto& z : red) back = back && p.max_violation(pre.postsolve(z)) <= 1e-9;
    if (!CHECK(back)) std::fprintf(stderr, "  case %d: a reduced point postsolves to an infeasible one\n", it);
  }
  std::printf("  probing round trip: %lld liftings, %lld compared exactly\n", lifts, compared);
  CHECK(lifts > iters / 10);
  CHECK(compared > iters / 2);
}

namespace {

// Presolves p with o and checks the round trip against enumeration: every feasible point of p crushes onto a
// feasible reduced point, and (reduced problems of at most 14 columns) the reduced feasible points are exactly
// those images and postsolve to feasible points of p. Returns false if presolve found p infeasible.
bool check_round_trip(const MipProblem& p, const PresolveOptions& o, int it, Presolver& pre, long long* compared) {
  const auto pts = enumerate_feasible(p);
  const PresolveStatus st = pre.run(p, o, kQuiet);
  if (st == PresolveStatus::Infeasible) {
    if (!CHECK(pts.empty())) std::fprintf(stderr, "  case %d: presolve infeasible, %zu feasible points\n", it, pts.size());
    return false;
  }
  REQUIRE(st == PresolveStatus::Reduced);
  const MipProblem& q = pre.reduced();
  std::set<std::vector<double>> image;
  bool ok = true;
  for (const auto& x : pts) {
    const std::vector<double> z = pre.crush(x);
    ok = ok && q.max_violation(z) <= 1e-9;
    image.insert(z);
  }
  if (!CHECK(ok)) std::fprintf(stderr, "  case %d: presolve cut off a feasible point\n", it);
  if (q.num_cols() > 14) return true;
  const auto red = enumerate_feasible(q);
  ++*compared;
  if (!CHECK_EQ(red.size(), image.size())) {
    std::fprintf(stderr, "  case %d: %zu reduced vs %zu original points\n", it, red.size(), image.size());
  }
  bool back = true;
  for (const auto& z : red) back = back && p.max_violation(pre.postsolve(z)) <= 1e-9;
  if (!CHECK(back)) std::fprintf(stderr, "  case %d: a reduced point postsolves to an infeasible one\n", it);
  return true;
}

// Random models around equations whose columns also appear in pair rows a_j x_j + a_k x_k <= / >= c with the side
// set so that the row implies a bound of x_j exactly, cuts it by 1 / a_j, or misses it by 1 / a_j (what
// implied_by_rows looks for), plus a few general rows around a point x0. Integer columns (binaries, [0, 3],
// [-1, 2]); with `mixed`, also continuous columns (some unbounded on one side), fractional coefficients and sides.
Model random_subst_model(std::mt19937& rng, bool mixed) {
  auto uni = [&](int lo, int hi) { return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)); };
  ModelBuilder b;
  std::vector<double> lb, ub, x0;
  const int n = mixed ? uni(4, 10) : uni(4, 8);
  for (int j = 0; j < n; ++j) {
    const int kind = uni(0, mixed ? 6 : 2);
    double l = 0.0, u = 1.0;
    if (kind == 1) u = 3.0;
    if (kind == 2) l = -1.0, u = 2.0;
    if (kind >= 3) l = uni(-2, 0) * 0.5, u = l + uni(1, 8) * 0.5;
    const VarType t = kind >= 3 ? VarType::Continuous : (kind == 0 ? VarType::Binary : VarType::Integer);
    const double cost = t == VarType::Continuous && uni(0, 1) == 0 ? 0.0 : uni(-4, 4) * (mixed && uni(0, 2) == 0 ? 0.5 : 1.0);
    b.add_col("x" + std::to_string(j), cost, l, u, t);
    lb.push_back(l), ub.push_back(u);
    x0.push_back(kind >= 3 ? l + (u - l) * uni(0, 4) / 4.0 : uni(static_cast<int>(l), static_cast<int>(u)));
  }
  auto pick = [&] { return uni(0, n - 1); };
  int r = 0;
  auto unique = [](const std::vector<std::pair<Index, double>>& c) {
    std::vector<std::pair<Index, double>> u;
    for (const auto& e : c) {
      if (e.second != 0.0 && std::none_of(u.begin(), u.end(), [&](const auto& f) { return f.first == e.first; })) {
        u.push_back(e);
      }
    }
    return u;
  };
  auto row = [&](double lo, double hi, const std::vector<std::pair<Index, double>>& c) {
    if (c.size() < 2) return;
    const Index i = b.add_row("r" + std::to_string(r++), lo, hi);
    for (const auto& [j, v] : c) b.add_coef(i, j, v);
  };
  auto act = [&](const std::vector<std::pair<Index, double>>& c) {
    double v = 0.0;
    for (const auto& [j, a] : c) v += a * x0[j];
    return v;
  };
  const double fracs[] = {1.0, 1.0, 1.0, 2.0, 0.5, 0.25};
  std::vector<Index> eq_cols;
  for (int k = uni(1, mixed ? 4 : 3); k > 0; --k) {  // equations through x0
    std::vector<std::pair<Index, double>> c;
    for (int t = uni(2, 4); t > 0; --t) {
      c.push_back({pick(), (uni(0, 1) ? 1.0 : -1.0) * (mixed ? fracs[uni(0, 5)] : (uni(0, 4) ? 1.0 : 2.0))});
    }
    c = unique(c);
    row(act(c), act(c), c);
    for (const auto& e : c) eq_cols.push_back(e.first);
  }
  for (int k = uni(2, 7); k > 0; --k) {  // pair rows implying (or nearly implying) a bound of x_j
    const Index j = pick();
    Index k2 = pick();
    if (k2 == j) k2 = (j + 1) % n;
    const double aj = (uni(0, 3) ? 1.0 : 2.0) * (uni(0, 1) ? 1.0 : -1.0);
    const double ak = (uni(0, 1) ? 1.0 : -1.0) * uni(1, 2) * (mixed && uni(0, 2) == 0 ? 0.5 : 1.0);
    const int side = uni(0, 4);  // cuts the bound (side 0), implies it exactly (1, 2), misses it (3, 4)
    const double slack = (side == 0 ? -1.0 : (side <= 2 ? 0.0 : 1.0)) * (mixed && uni(0, 1) ? 0.5 : 1.0);
    const double kmin = ak > 0 ? ak * lb[k2] : ak * ub[k2], kmax = ak > 0 ? ak * ub[k2] : ak * lb[k2];
    if (uni(0, 1)) {  // implies x_j <= ub_j: aj x_j + ak x_k <= aj ub_j + min(ak x_k) (aj > 0)
      if (aj > 0) row(-kInf, aj * ub[j] + kmin + slack, {{j, aj}, {k2, ak}});
      else row(aj * ub[j] + kmax - slack, kInf, {{j, aj}, {k2, ak}});
    } else {  // implies x_j >= lb_j
      if (aj > 0) row(aj * lb[j] + kmax - slack, kInf, {{j, aj}, {k2, ak}});
      else row(-kInf, aj * lb[j] + kmin + slack, {{j, aj}, {k2, ak}});
    }
  }
  for (int k = uni(1, 3); k > 0; --k) {  // general rows around x0
    std::vector<std::pair<Index, double>> c;
    for (int t = uni(2, 4); t > 0; --t) {
      // Mixed: half of the entries from the equations' columns (substitutions that add entries to these rows).
      const Index j = mixed && !eq_cols.empty() && uni(0, 1) ? eq_cols[uni(0, static_cast<int>(eq_cols.size()) - 1)] : pick();
      c.push_back({j, static_cast<double>(uni(-3, 3))});
    }
    c = unique(c);
    const double v = act(c);
    if (uni(0, 1)) row(-kInf, v + uni(0, 2), c); else row(v - uni(0, 2), kInf, c);
  }
  if (mixed && uni(0, 3) == 0) {  // a column unbounded on one side
    Model m = b.build();
    const Index j = uni(0, n - 1);
    if (m.col_type[j] == VarType::Continuous) {
      if (uni(0, 1)) m.col_lb[j] = -kInf; else m.col_ub[j] = kInf;
    }
    if (uni(0, 1)) m.sense = ObjSense::Maximize;
    return m;
  }
  if (uni(0, 1)) b.model().sense = ObjSense::Maximize;
  return b.build();
}

}  // namespace

TEST(presolve_substitution_implied_bounds_in_place) {
  // x <= y and x >= y (two rows) imply x's bounds from y's and y's from x's (both [0, 3]). Once x is substituted out
  // of x + p - q = 2 (its bounds implied by them), y >= 0 is still implied (y >= 2 - p + q) but y <= 3 is not
  // (2 - p + q reaches 4, and y + s - 2 t = 1 gives y <= 5): it must stay when y is substituted. The reduced
  // problem has exactly the original's points.
  Builder b;
  const Index x = b.col(1.0, 0.0, 3.0, true), y = b.col(-1.0, 0.0, 3.0, true);
  std::vector<Index> v;
  for (int k = 0; k < 4; ++k) v.push_back(b.col(k - 1.5, 0.0, 2.0, true));
  b.row(2.0, 2.0, {{x, 1.0}, {v[0], 1.0}, {v[1], -1.0}});
  b.row(1.0, 1.0, {{y, 1.0}, {v[2], 1.0}, {v[3], -2.0}});
  b.row(-kInf, 0.0, {{x, 1.0}, {y, -1.0}});
  b.row(0.0, kInf, {{x, 1.0}, {y, -1.0}});
  b.row(-kInf, 3.0, {{v[0], 1.0}, {v[1], 1.0}});
  b.row(-kInf, 3.0, {{v[2], 1.0}, {v[3], 1.0}});
  const MipProblem p = b.build();
  PresolveOptions o = only_subst();
  o.substitution_bounded = false;
  Presolver pre;
  long long compared = 0;
  REQUIRE(check_round_trip(p, o, 0, pre, &compared));
  CHECK_EQ(compared, 1);
  CHECK_EQ(reduced_index(pre, x), -1);
  CHECK(pre.stats().subst_implied >= 1);
  CHECK_EQ(enumerate_feasible(pre.reduced()).size(), enumerate_feasible(p).size());
}

TEST(presolve_substitution_round_trip_enumerated) {
  // Substitution with bounds implied by other rows (on, off; fill allowance on or off; alone or with bound
  // tightening, coefficient tightening and aggregation) on enumerable models: the feasible points map exactly onto
  // those of the reduced problem (crush) and back (postsolve). Pure integer: the fill allowance (continuous columns
  // only) is covered by the MILP test below.
  std::mt19937 rng(fuzz_seed(13579u));
  const int iters = fuzz_iters(2000);
  long long by_rows = 0, compared = 0;
  for (int it = 0; it < iters; ++it) {
    MipProblem p;
    REQUIRE(MipProblem::from_model(random_subst_model(rng, false), &p));
    PresolveOptions o = only(it % 2 == 0, it % 4 == 1, false, false, it % 3 == 0);
    o.substitution = true;
    o.substitution_implied = it % 5 != 4;
    o.substitution_fill = it % 7 == 6 ? 0 : 8;
    o.clique_extension = it % 2 == 1;
    Presolver pre;
    if (!check_round_trip(p, o, it, pre, &compared)) continue;
    by_rows += pre.stats().subst_implied;
    CHECK_EQ(pre.stats().subst_fill, 0);
  }
  std::printf("  substitution round trip: %lld implied free by other rows, %lld compared exactly\n", by_rows, compared);
  CHECK(by_rows > iters / 10);
  CHECK(compared > iters / 2);
}

TEST(presolve_random_substitution_milps_match_no_presolve) {
  // Mixed models around equations, with the default presolve and, in a third of the cases, without bound tightening
  // and probing (the bounds propagation tightens make most columns implied free; substitutions that add nonzeros
  // wait for probing): more substitutions that keep their row and grow the matrix.
  std::mt19937 rng(fuzz_seed(86420u));
  const int iters = fuzz_iters(2000);
  int optimal = 0;
  PresolveOptions plain;
  plain.bound_tightening = false;
  plain.probing = false;
  for (int it = 0; it < iters; ++it) {
    const Model m = random_subst_model(rng, true);
    optimal += compare_with_presolve(m, it, -1, it % 3 == 2 ? &plain : nullptr) == Status::Optimal;
  }
  const long long subst = g_stats.substitutions, by_rows = g_stats.subst_implied, grew = g_stats.subst_fill;
  std::printf("  substitution models: %d optimal, %lld kept rows grew\n", optimal, grew);
  print_coverage("substitution");
  CHECK(optimal > iters / 3);
  CHECK(subst > iters / 10);
  CHECK(by_rows > iters / 40);
  CHECK(grew > iters / 100);
}

// Facility covering models: binary facilities y_i, binary assignments x_ij with x_ij <= y_i (aggregated
// sum_j x_ij <= |J| y_i, disaggregated, or sometimes with a smaller capacity), client rows sum_i x_ij >= 1 or
// == 1, costs of either sign on the assignments and some side rows.
Model random_vub_model(std::mt19937& rng) {
  auto uni = [&](int lo, int hi) { return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)); };
  ModelBuilder b;
  const int nf = uni(2, 4), nc = uni(2, 5);
  std::vector<Index> y(nf);
  std::vector<std::vector<Index>> x(nf, std::vector<Index>(nc, -1));
  for (int i = 0; i < nf; ++i) y[i] = b.add_col("y" + std::to_string(i), uni(1, 6), 0.0, 1.0, VarType::Binary);
  int r = 0;
  auto row = [&](double lo, double hi, const std::vector<std::pair<Index, double>>& c) {
    const Index k = b.add_row("r" + std::to_string(r++), lo, hi);
    for (const auto& [j, v] : c) b.add_coef(k, j, v);
  };
  for (int i = 0; i < nf; ++i) {
    std::vector<std::pair<Index, double>> agg;
    for (int j = 0; j < nc; ++j) {
      if (uni(0, 4) == 0) continue;
      x[i][j] = b.add_col("x" + std::to_string(i) + "_" + std::to_string(j), uni(0, 3) == 0 ? uni(1, 3) : -uni(0, 2) * 0.5,
                          0.0, 1.0, VarType::Binary);
      agg.push_back({x[i][j], static_cast<double>(uni(1, 2))});
    }
    const int kind = uni(0, 3);
    double cap = 0.0;
    for (const auto& e : agg) cap += e.second;
    if (kind == 3) cap = std::max(1.0, cap - uni(1, 2));  // a capacity: not a variable upper bound
    if (kind == 0) {
      for (const auto& e : agg) row(-kInf, 0.0, {{e.first, 1.0}, {y[i], -1.0}});
    } else if (!agg.empty()) {
      agg.push_back({y[i], -cap});
      if (kind == 2) {
        for (auto& e : agg) e.second = -e.second;
        row(0.0, kInf, agg);
      } else {
        row(-kInf, 0.0, agg);
      }
    }
  }
  for (int j = 0; j < nc; ++j) {
    std::vector<std::pair<Index, double>> c;
    for (int i = 0; i < nf; ++i) {
      if (x[i][j] >= 0) c.push_back({x[i][j], 1.0});
    }
    if (c.empty()) continue;
    if (uni(0, 3) == 0) row(1.0, 1.0, c);
    else row(1.0, kInf, c);
  }
  for (int t = uni(0, 2); t > 0; --t) {  // side rows on facilities and assignments
    std::vector<std::pair<Index, double>> c;
    for (int i = 0; i < nf; ++i) {
      if (uni(0, 1)) c.push_back({y[i], static_cast<double>(uni(1, 3))});
      const int j = uni(0, nc - 1);
      if (x[i][j] >= 0 && uni(0, 2) == 0) c.push_back({x[i][j], static_cast<double>(uni(-2, 2))});
    }
    std::vector<std::pair<Index, double>> u;
    for (const auto& e : c) {
      if (e.second != 0.0 && std::none_of(u.begin(), u.end(), [&](const auto& f) { return f.first == e.first; })) u.push_back(e);
    }
    if (u.empty()) continue;
    if (uni(0, 1)) row(-kInf, uni(2, 6), u);
    else row(1.0, kInf, u);
  }
  return b.build();
}

TEST(presolve_random_vub_milps_match_no_presolve) {
  // Dual aggregation with the default presolve, and alone with bound tightening (fewer other reductions first).
  std::mt19937 rng(fuzz_seed(24680u));
  const int iters = fuzz_iters(1500);
  int optimal = 0;
  PresolveOptions alone = only(true, false, false, false, false);
  alone.dual_aggregation = true;
  for (int it = 0; it < iters; ++it) {
    optimal += compare_with_presolve(random_vub_model(rng), it, -1, it % 2 == 0 ? &alone : nullptr) == Status::Optimal;
  }
  const long long agg = g_stats.dual_aggregations;
  std::printf("  variable-upper-bound models: %d optimal\n", optimal);
  print_coverage("vub");
  CHECK(optimal > iters / 3);
  CHECK(agg > iters);
}

TEST(presolve_complement_keeps_the_group_member) {
  // x0 + x1 + x2 = 1 with complements y_k = 1 - x_k used in big-M rows: the aggregation keeps the x_k, so the
  // partitioning row survives as it is (without the rule the first complement pair may leave y in it).
  for (bool keep : {true, false}) {
    Builder b;
    std::vector<Index> x, y;
    for (int k = 0; k < 3; ++k) x.push_back(b.col(k + 1.0, 0.0, 1.0, true));
    for (int k = 0; k < 3; ++k) y.push_back(b.col(0.0, 0.0, 1.0, true));
    const Index w = b.col(-1.0, 0.0, 10.0);
    for (int k = 0; k < 3; ++k) b.row(1.0, 1.0, {{y[k], 1.0}, {x[k], 1.0}});  // y listed first
    b.row(1.0, 1.0, {{x[0], 1.0}, {x[1], 1.0}, {x[2], 1.0}});
    for (int k = 0; k < 3; ++k) {
      b.row(-kInf, 2.0 * k + 1.0, {{w, 1.0}, {y[k], -10.0}});
      b.row(-kInf, 12.0, {{w, 1.0}, {y[k], 3.0}, {x[(k + 1) % 3], 1.0}});
    }
    const Model m = b.b.build();
    MipProblem p;
    REQUIRE(MipProblem::from_model(m, &p));
    PresolveOptions o = only(false, false, false, false, true);
    o.keep_members = keep;
    Presolver pre;
    REQUIRE(pre.run(p, o, kQuiet) == PresolveStatus::Reduced);
    CHECK(pre.stats().aggregations == 3);
    int kept = 0;
    for (int k = 0; k < 3; ++k) kept += reduced_index(pre, x[k]) >= 0;
    if (keep) CHECK(kept == 3);
    SolverParams params;
    params.verbosity = 0;
    params.threads = 1;
    const SolveResult r = solve(m, params);
    REQUIRE(r.status == Status::Optimal);
    params.presolve = false;
    const SolveResult r0 = solve(m, params);
    REQUIRE(r0.status == Status::Optimal);
    CHECK_NEAR(r.objective, r0.objective, 1e-9);
  }
}

TEST(presolve_probing_does_not_lift_a_redundant_row) {
  // R: y - 3 t + 100 a <= 100 (a = 1: y <= 3 t). Probing fixes a = 0 (a = 1 forces c = 1 and c = 0), which
  // makes R redundant (y <= 50); the same pass finds the lift of R by c (c = 1: y <= 5 and a = 0, so R has a
  // slack of 98). Lifted, the row stays as y - 3 t + 98 c <= 100; with the rule it is left alone.
  for (bool live : {true, false}) {
    Builder b;
    const Index c = b.col(1.0, 0.0, 1.0, true);  // probed before a: the lift is found while a is still free
    const Index a = b.col(1.0, 0.0, 1.0, true);
    const Index y = b.col(-1.0, 0.0, 50.0);
    const Index t = b.col(2.0, 1.0, 5.0, true);
    b.row(-kInf, 100.0, {{y, 1.0}, {t, -3.0}, {a, 100.0}});
    b.row(-kInf, 0.0, {{a, 1.0}, {c, -1.0}});
    b.row(-kInf, 1.0, {{a, 1.0}, {c, 1.0}});
    b.row(-kInf, 50.0, {{y, 1.0}, {c, 45.0}});
    const Model m = b.b.build();
    MipProblem p;
    REQUIRE(MipProblem::from_model(m, &p));
    PresolveOptions o = only_probing();
    o.lift_live = live;
    Presolver pre;
    REQUIRE(pre.run(p, o, kQuiet) == PresolveStatus::Reduced);
    CHECK(pre.stats().probing_fixed >= 1);
    CHECK(reduced_index(pre, a) < 0);
    CHECK((pre.stats().probing_lifts == 0) == live);
    SolverParams params;
    params.verbosity = 0;
    params.threads = 1;
    const SolveResult r = solve(m, params);
    params.presolve = false;
    const SolveResult r0 = solve(m, params);
    REQUIRE(r.status == Status::Optimal);
    REQUIRE(r0.status == Status::Optimal);
    CHECK_NEAR(r.objective, r0.objective, 1e-9);
  }
}

TEST(presolve_dyadic_rows_tighten_exactly_and_become_unit_rows) {
  // x + 0.25 y <= 1 over binaries: coefficient tightening gives 0.25 on x and the side 0.25. With the rounding
  // margin of non-integral rows the coefficient is 0.25 + 2e-13 and the row is no clique; exact, it is
  // 0.25 (x + y) <= 0.25, which unit_rows writes as x + y <= 1. Likewise 238 (u + v + w) >= 238 becomes
  // u + v + w >= 1.
  for (int mode = 0; mode < 3; ++mode) {
    Builder b;
    const Index x = b.col(-3.0, 0.0, 1.0, true);
    const Index y = b.col(-1.0, 0.0, 1.0, true);
    const Index u = b.col(2.0, 0.0, 1.0, true), v = b.col(3.0, 0.0, 1.0, true), w = b.col(4.0, 0.0, 1.0, true);
    b.row(-kInf, 1.0, {{x, 1.0}, {y, 0.25}});
    b.row(238.0, kInf, {{u, 238.0}, {v, 238.0}, {w, 238.0}});
    b.row(-kInf, 2.0, {{x, 1.0}, {y, 1.0}, {u, 1.0}, {v, 1.0}});
    // A row with continuous columns keeps its coefficients (divided by 4 it would be held to four times the
    // feasibility tolerance in its original form).
    const Index c1 = b.col(-0.5, 0.0, 2.0), c2 = b.col(-0.25, 0.0, 2.0);
    b.row(-kInf, 6.0, {{c1, 4.0}, {c2, 4.0}});
    b.row(-kInf, 3.0, {{c1, 1.0}, {c2, 2.0}, {x, 1.0}});
    const Model m = b.b.build();
    MipProblem p;
    REQUIRE(MipProblem::from_model(m, &p));
    PresolveOptions o = only(false, true, false, false, false);
    o.lift_dyadic = mode >= 1;
    o.unit_rows = mode == 2;
    Presolver pre;
    REQUIRE(pre.run(p, o, kQuiet) == PresolveStatus::Reduced);
    const MipProblem& q = pre.reduced();
    const SparseMatrix R = q.A.transpose();
    REQUIRE(q.num_rows() == 5);
    CHECK(R.value[R.begin(3)] == 4.0);
    CHECK(q.row_ub[3] == 6.0);
    // Row 0: equal coefficients exactly when the tightening is exact; unit form with unit_rows.
    REQUIRE(R.end(0) - R.begin(0) == 2);
    const double a0 = R.value[R.begin(0)], a1 = R.value[R.begin(0) + 1];
    CHECK((a0 == a1) == (mode >= 1));
    CHECK((a0 == 1.0 && a1 == 1.0 && q.row_ub[0] == 1.0) == (mode == 2));
    if (mode == 1) CHECK(a0 == 0.25 && q.row_ub[0] == 0.25);
    // Row 1: the covering row.
    CHECK((R.value[R.begin(1)] == 1.0 && q.row_lb[1] == 1.0) == (mode == 2));
    SolverParams params;
    params.verbosity = 0;
    params.threads = 1;
    const SolveResult r = solve(m, params);
    REQUIRE(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -1.0 - 0.75, 1e-9);  // x = 1, y = 0, u = 1; c1 = 1.5, c2 = 0
  }
}

TEST(presolve_integer_divisor_rescales_a_column) {
  // w = 10 (u - v) + 30 with u, v in [0, 5]: w is 0 modulo 10, so w = 10 w' (w in [-5, 65]: w' <= 6).
  // z = w + 5 t inherits a divisor of 5 once w is rescaled. Minimize z >= 37: z = 40.
  for (bool on : {true, false}) {
    Builder b;
    const Index u = b.col(0.0, 0.0, 5.0, true);
    const Index v = b.col(0.0, 0.0, 5.0, true);
    const Index w = b.col(0.0, -5.0, 65.0, true);
    const Index t = b.col(0.0, 0.0, 3.0, true);
    const Index z = b.col(1.0, 37.0, 200.0, true);
    b.row(30.0, 30.0, {{w, 1.0}, {u, -10.0}, {v, 10.0}});
    b.row(0.0, 0.0, {{z, 1.0}, {w, -1.0}, {t, -5.0}});
    b.row(-kInf, 7.0, {{u, 1.0}, {v, 1.0}, {t, 1.0}});
    const Model m = b.b.build();
    MipProblem p;
    REQUIRE(MipProblem::from_model(m, &p));
    PresolveOptions o = only(true, true, false, false, false);
    o.int_divisors = on;
    Presolver pre;
    REQUIRE(pre.run(p, o, kQuiet) == PresolveStatus::Reduced);
    CHECK(pre.stats().int_divisors == (on ? 2 : 0));
    const Index kw = reduced_index(pre, w), kz = reduced_index(pre, z);
    REQUIRE(kw >= 0);
    REQUIRE(kz >= 0);
    if (on) {
      CHECK_NEAR(pre.reduced().col_lb[kw], 3.0, 1e-12);  // w >= 37 - 5 * 3 = 22 from the second row
      CHECK_NEAR(pre.reduced().col_ub[kw], 6.0, 1e-12);
      CHECK_NEAR(pre.reduced().col_lb[kz], 8.0, 1e-12);  // z = 5 z', z >= 37
    }
    // A point of the original space survives crush and postsolve.
    std::vector<double> x(5, 0.0);
    x[u] = 3.0, x[v] = 2.0, x[w] = 40.0, x[t] = 0.0, x[z] = 40.0;
    const std::vector<double> back = pre.postsolve(pre.crush(x));
    for (Index j = 0; j < 5; ++j) CHECK_NEAR(back[j], x[j], 1e-9);
    SolverParams params;
    params.verbosity = 0;
    params.threads = 1;
    const SolveResult r = solve(m, params);
    REQUIRE(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, 40.0, 1e-9);
  }
}

// Integer columns defined by equations whose other coefficients share a divisor (with an offset), chains of
// them, a one-of group valued by one of them, and side rows.
Model random_divisor_model(std::mt19937& rng) {
  auto uni = [&](int lo, int hi) { return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)); };
  ModelBuilder b;
  int r = 0;
  auto row = [&](double lo, double hi, const std::vector<std::pair<Index, double>>& c) {
    const Index k = b.add_row("r" + std::to_string(r++), lo, hi);
    for (const auto& [j, v] : c) b.add_coef(k, j, v);
  };
  std::vector<Index> ints;
  const int nb = uni(2, 4);
  for (int i = 0; i < nb; ++i) {
    ints.push_back(b.add_col("u" + std::to_string(i), uni(-3, 3), uni(-2, 0), uni(2, 6), VarType::Integer));
  }
  std::vector<Index> der;
  for (int k = uni(1, 4); k > 0; --k) {
    const int g = uni(2, 5) * (uni(0, 3) == 0 ? uni(2, 3) : 1);
    const int aw = uni(0, 3) == 0 ? uni(2, 3) : 1;
    const bool wide = uni(0, 2) == 0;
    const Index w = b.add_col("w" + std::to_string(der.size()), uni(-3, 3), wide ? -200.0 : uni(-20, 0),
                              wide ? 200.0 : uni(5, 60), VarType::Integer);
    std::vector<std::pair<Index, double>> c{{w, uni(0, 1) ? aw : -aw}};
    // Terms: base columns with multiples of aw g, or an earlier derived column with coefficient aw (a chain: the
    // divisor shows only after that column was rescaled).
    for (int t = uni(1, 3); t > 0; --t) {
      const bool chain = !der.empty() && uni(0, 2) == 0;
      const Index j = chain ? der[uni(0, static_cast<int>(der.size()) - 1)] : ints[uni(0, nb - 1)];
      if (std::any_of(c.begin(), c.end(), [&](const auto& e) { return e.first == j; })) continue;
      const int m = chain ? aw : aw * g * uni(1, 2);
      c.push_back({j, uni(0, 1) ? m : -m});
    }
    if (c.size() < 2) c.push_back({ints[0], static_cast<double>(aw * g)});
    const double rhs = aw * uni(-3, 12);
    row(rhs, rhs, c);
    der.push_back(w);
  }
  if (uni(0, 1)) {  // a one-of group whose value is a derived column
    const Index w = der[uni(0, static_cast<int>(der.size()) - 1)];
    const int K = uni(3, 12), k0 = uni(-2, 2);
    std::vector<std::pair<Index, double>> one, val{{w, -1.0}};
    for (int k = 0; k <= K; ++k) {
      const Index x = b.add_col("x" + std::to_string(k), uni(-4, 4), 0.0, 1.0, VarType::Binary);
      one.push_back({x, 1.0});
      if (k0 + k != 0) val.push_back({x, static_cast<double>(k0 + k)});
    }
    row(1.0, uni(0, 3) == 0 ? kInf : 1.0, one);
    row(0.0, 0.0, val);
  }
  Index y = -1;
  if (uni(0, 2) == 0) y = b.add_col("y", uni(-2, 2), 0.0, uni(1, 9), VarType::Continuous);
  std::vector<Index> all = ints;
  all.insert(all.end(), der.begin(), der.end());
  for (int t = uni(1, 3); t > 0; --t) {  // side rows
    std::vector<std::pair<Index, double>> c;
    for (Index j : all) {
      if (uni(0, 2) == 0) c.push_back({j, static_cast<double>(uni(1, 3) * (uni(0, 1) ? 1 : -1))});
    }
    if (y >= 0 && uni(0, 1)) c.push_back({y, uni(0, 1) ? 1.0 : -1.5});
    if (c.empty()) continue;
    const int kind = uni(0, 2);
    if (kind == 0) row(-kInf, uni(-5, 30), c);
    else if (kind == 1) row(uni(-30, 5), kInf, c);
    else row(uni(-10, 0), uni(0, 25), c);
  }
  return b.build();
}

TEST(presolve_random_divisor_milps_match_no_presolve) {
  std::mt19937 rng(fuzz_seed(86420u));
  const int iters = fuzz_iters(1500);
  int optimal = 0;
  PresolveOptions alone = only(true, true, false, false, false);
  for (int it = 0; it < iters; ++it) {
    optimal += compare_with_presolve(random_divisor_model(rng), it, -1, it % 3 == 0 ? &alone : nullptr) == Status::Optimal;
  }
  const long long rescaled = g_stats.int_divisors;
  std::printf("  divisor models: %d optimal\n", optimal);
  print_coverage("divisor");
  CHECK(optimal > iters / 5);
  CHECK(rescaled > iters / 2);
}

TEST(presolve_probing_lifts_big_m) {
  // y <= 100 x where x = 1 forces w = 1 and with it y <= 5 (y + 10 w <= 15): lifted to y <= 5 x (with x = 0,
  // y = 0 already). Liftings off: the coefficient stays.
  for (bool lift : {true, false}) {
    Builder b;
    const Index x = b.col(1.0, 0.0, 1.0, true);
    const Index w = b.col(1.0, 0.0, 1.0, true);
    const Index y = b.col(-1.0, 0.0, 100.0);
    const Index z = b.col(-1.0, 0.0, 1.0, true);
    b.row(-kInf, 0.0, {{y, 1.0}, {x, -100.0}});
    b.row(-kInf, 15.0, {{y, 1.0}, {w, 10.0}, {z, 1.0}});
    b.row(-kInf, 0.0, {{x, 1.0}, {w, -1.0}});
    PresolveOptions o = only_probing();
    o.probing_lift = lift;
    Presolver pre;
    REQUIRE(pre.run(b.build(), o, kQuiet) == PresolveStatus::Reduced);
    const MipProblem& q = pre.reduced();
    const Index rx = reduced_index(pre, x), ry = reduced_index(pre, y);
    REQUIRE(rx >= 0 && ry >= 0);
    // The row of exactly y and x: its coefficient ratio.
    const SparseMatrix R = q.A.transpose();
    double ratio = 0.0;
    for (Index i = 0; i < q.num_rows(); ++i) {
      double ax = 0.0, ay = 0.0;
      for (Index k = R.begin(i); k < R.end(i); ++k) {
        if (R.index[k] == rx) ax = R.value[k];
        if (R.index[k] == ry) ay = R.value[k];
      }
      if (ax != 0.0 && ay != 0.0 && R.end(i) - R.begin(i) == 2) ratio = ax / ay;
    }
    CHECK_NEAR(ratio, lift ? -5.0 : -100.0, 1e-9);
    CHECK_EQ(pre.stats().probing_lifts > 0, lift);
  }
}

namespace {

// Random MILPs with the structures liftings work on: knapsacks and set packing over binaries, big-M rows of
// continuous and general integer columns, implications and partitioning rows.
Model random_lifting_model(std::mt19937& rng) {
  auto uni = [&](int lo, int hi) { return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)); };
  ModelBuilder b;
  std::vector<Index> bin, oth;
  for (int j = uni(4, 12); j > 0; --j) bin.push_back(b.add_col("b" + std::to_string(j), uni(-6, 4), 0, 1, VarType::Binary));
  for (int j = uni(1, 4); j > 0; --j) {
    const bool integer = uni(0, 1);
    oth.push_back(b.add_col("o" + std::to_string(j), uni(-4, 1), 0, uni(2, 20), integer ? VarType::Integer : VarType::Continuous));
  }
  auto pick = [&](const std::vector<Index>& v) { return v[uni(0, static_cast<int>(v.size()) - 1)]; };
  int r = 0;
  auto row = [&](double lo, double hi, const std::vector<std::pair<Index, double>>& c) {
    const Index i = b.add_row("r" + std::to_string(r++), lo, hi);
    std::vector<Index> seen;
    for (const auto& [j, v] : c) {
      if (std::find(seen.begin(), seen.end(), j) != seen.end()) continue;
      seen.push_back(j);
      b.add_coef(i, j, v);
    }
  };
  for (int i = uni(3, 10); i > 0; --i) {
    const int kind = uni(0, 4);
    std::vector<std::pair<Index, double>> c;
    if (kind == 0) {
      for (int k = uni(2, 5); k > 0; --k) c.push_back({pick(bin), 1.0});
      row(uni(0, 3) ? -kInf : 1.0, 1.0, c);
    } else if (kind == 1) {
      double tot = 0.0;
      for (int k = uni(2, 6); k > 0; --k) {
        const double a = uni(1, 9) * (uni(0, 4) ? 1.0 : 0.5);
        c.push_back({uni(0, 3) ? pick(bin) : pick(oth), a});
        tot += a;
      }
      row(-kInf, std::floor(tot * uni(3, 7) / 10.0), c);
    } else if (kind == 2) {
      row(-kInf, 0.0, {{pick(oth), 1.0}, {pick(bin), -static_cast<double>(uni(5, 50))}});
    } else if (kind == 3) {
      const bool imp = uni(0, 1);
      row(-kInf, imp ? 0.0 : 1.0, {{pick(bin), 1.0}, {pick(bin), imp ? -1.0 : 1.0}});
    } else {
      for (int k = uni(2, 4); k > 0; --k) c.push_back({pick(oth), static_cast<double>(uni(1, 3))});
      c.push_back({pick(bin), static_cast<double>(uni(1, 8))});
      row(-kInf, uni(4, 15), c);
    }
  }
  if (uni(0, 1)) b.model().sense = ObjSense::Maximize;
  return b.build();
}

}  // namespace

TEST(presolve_random_lifting_milps_match_no_presolve) {
  std::mt19937 rng(fuzz_seed(31337u));
  const int iters = fuzz_iters(1000);
  int optimal = 0;
  for (int it = 0; it < iters; ++it) optimal += compare_with_presolve(random_lifting_model(rng), it) == Status::Optimal;
  std::printf("  lifting models: %d optimal\n", optimal);
  const long long lifts = g_stats.probing_lifts;
  print_coverage("lifting");
  CHECK(optimal > iters / 2);
  CHECK(lifts > iters / 10);
}

TEST(presolve_sparsify_cancels_entries) {
  // 2 y1 + 3 y2 + 5 y3 = 10 (general integers: not substituted, 3 / 2 is fractional) taken from
  // 2 y1 + 3 y2 + 5 y3 + 2 y4 <= 12 leaves 2 y4 <= 2: y4 <= 1, fixed there by its cost. Off: y4 <= 6.
  for (bool on : {true, false}) {
    Builder b;
    const Index y1 = b.col(1.0, 0.0, 10.0, true), y2 = b.col(1.0, 0.0, 10.0, true), y3 = b.col(1.0, 0.0, 10.0, true);
    const Index y4 = b.col(-1.0, 0.0, 10.0, true);
    b.row(10.0, 10.0, {{y1, 2.0}, {y2, 3.0}, {y3, 5.0}});
    b.row(-kInf, 12.0, {{y1, 2.0}, {y2, 3.0}, {y3, 5.0}, {y4, 2.0}});
    PresolveOptions o;
    o.sparsify = on;
    Presolver pre;
    REQUIRE(pre.run(b.build(), o, kQuiet) == PresolveStatus::Reduced);
    CHECK_EQ(pre.stats().sparsify_rows, on ? 1LL : 0LL);
    const Index r4 = reduced_index(pre, y4);
    if (on) {
      CHECK(r4 < 0);
      CHECK_EQ(pre.postsolve(pre.reduced().col_lb)[y4], 1.0);
    } else {
      REQUIRE(r4 >= 0);
      CHECK_EQ(pre.reduced().col_ub[r4], 6.0);
    }
  }
}

TEST(presolve_sparsify_keeps_structure) {
  // Only sparsify (and the basic row / column passes).
  PresolveOptions o = only(false, false, false, false, false);
  o.clique_extension = false;
  o.sparsify = true;
  {
    // The set partitioning equation x1 + x2 + x3 = 1 is not used: x1 + x2 + x4 <= 1 would become x4 - x3 <= 0.
    Builder b;
    const Index x1 = b.col(-1.0, 0.0, 1.0, true), x2 = b.col(-1.0, 0.0, 1.0, true), x3 = b.col(-1.0, 0.0, 1.0, true);
    const Index x4 = b.col(-1.0, 0.0, 1.0, true), z = b.col(1.0, 0.0, 5.0);
    b.row(1.0, 1.0, {{x1, 1.0}, {x2, 1.0}, {x3, 1.0}});
    b.row(-kInf, 1.0, {{x1, 1.0}, {x2, 1.0}, {x4, 1.0}});
    b.row(-kInf, 4.0, {{x1, 2.0}, {x2, 2.0}, {x4, 1.0}, {z, 1.0}});
    Presolver pre;
    REQUIRE(pre.run(b.build(), o, kQuiet) == PresolveStatus::Reduced);
    CHECK_EQ(pre.stats().sparsify_rows, 0LL);
  }
  {
    // A set packing row is kept: x1 + x2 + x3 - y + w = 1 (no partitioning row) would turn x1 + x2 + x3 <= 1 into
    // y - w <= 0 and lose the clique. The mixed row 3 x1 + 3 x2 + 3 x3 + z <= 7 becomes 3 y - 3 w + z <= 4.
    Builder b;
    const Index x1 = b.col(-1.0, 0.0, 1.0, true), x2 = b.col(-1.0, 0.0, 1.0, true), x3 = b.col(-1.0, 0.0, 1.0, true);
    const Index y = b.col(1.0, 0.0, 1.0, true), w = b.col(1.0, 0.0, 1.0, true), z = b.col(-1.0, 0.0, 5.0);
    b.row(1.0, 1.0, {{x1, 1.0}, {x2, 1.0}, {x3, 1.0}, {y, -1.0}, {w, 1.0}});
    b.row(-kInf, 1.0, {{x1, 1.0}, {x2, 1.0}, {x3, 1.0}});
    b.row(-kInf, 7.0, {{x1, 3.0}, {x2, 3.0}, {x3, 3.0}, {z, 1.0}});
    Presolver pre;
    REQUIRE(pre.run(b.build(), o, kQuiet) == PresolveStatus::Reduced);
    CHECK_EQ(pre.stats().sparsify_rows, 1LL);
    CHECK_EQ(pre.stats().sparsify_removed, 1LL);
    CHECK_EQ(pre.reduced().A.nnz(), 11);
  }
  {
    // A row of integer columns with integral coefficients stays integral: 0.5 y1 + 0.5 y2 + 0.25 y3 = 1 is not added
    // to y1 + y2 + y4 <= 3 (lambda = -2 gives -0.5 y3), but it is to y1 + y2 + z <= 3 (z continuous): -0.5 y3 + z <= 1.
    Builder b;
    const Index y1 = b.col(1.0, 0.0, 4.0, true), y2 = b.col(1.0, 0.0, 4.0, true), y3 = b.col(1.0, 0.0, 4.0, true);
    const Index y4 = b.col(-1.0, 0.0, 4.0, true), z = b.col(-1.0, 0.0, 4.0);
    b.row(1.0, 1.0, {{y1, 0.5}, {y2, 0.5}, {y3, 0.25}});
    b.row(-kInf, 3.0, {{y1, 1.0}, {y2, 1.0}, {y4, 1.0}});
    b.row(-kInf, 3.0, {{y1, 1.0}, {y2, 1.0}, {z, 1.0}});
    Presolver pre;
    REQUIRE(pre.run(b.build(), o, kQuiet) == PresolveStatus::Reduced);
    CHECK_EQ(pre.stats().sparsify_rows, 1LL);
    const MipProblem& q = pre.reduced();
    REQUIRE(q.num_rows() == 3);
    const SparseMatrix R = q.A.transpose();
    CHECK_EQ(R.end(1) - R.begin(1), 3);  // unchanged
    CHECK_EQ(R.end(2) - R.begin(2), 2);
    CHECK_EQ(q.row_ub[2], 1.0);
  }
}

TEST(presolve_sparsify_protects_variable_bounds) {
  // Equations with a continuous column that has a variable bound are not used (sparsify_protect); off, they are.
  for (bool protect : {true, false}) {
    PresolveOptions o = only(false, false, false, false, false);
    o.clique_extension = false;
    o.sparsify = true;
    o.sparsify_protect = protect;
    {
      // Flow balance x1 + x2 - x3 = 0 with x1 <= 5 y1, x2 <= 5 y2: x1 + x2 + x4 = 2 would become x3 + x4 = 2.
      Builder b;
      const Index x1 = b.col(1.0, 0.0, 5.0), x2 = b.col(1.0, 0.0, 5.0), x3 = b.col(1.0, 0.0, 5.0);
      const Index x4 = b.col(1.0, 0.0, 5.0), x5 = b.col(-1.0, 0.0, 5.0), x6 = b.col(-1.0, 0.0, 5.0);
      const Index y1 = b.col(1.0, 0.0, 1.0, true), y2 = b.col(1.0, 0.0, 1.0, true);
      b.row(0.0, 0.0, {{x1, 1.0}, {x2, 1.0}, {x3, -1.0}});
      b.row(-kInf, 0.0, {{x1, 1.0}, {y1, -5.0}});
      b.row(-kInf, 0.0, {{x2, 1.0}, {y2, -5.0}});
      b.row(2.0, 2.0, {{x1, 1.0}, {x2, 1.0}, {x4, 1.0}});
      b.row(-kInf, 7.0, {{x3, 1.0}, {x5, 1.0}});
      b.row(-3.0, kInf, {{x3, 1.0}, {x6, -1.0}});
      Presolver pre;
      REQUIRE(pre.run(b.build(), o, kQuiet) == PresolveStatus::Reduced);
      CHECK_EQ(pre.stats().sparsify_rows, protect ? 0LL : 1LL);
    }
    {
      // Arc total z = f1 + f2 + f3 and the capacity row f1 + f2 + f3 <= 7 y (y general integer): the capacity row
      // would become z <= 7 y, hiding the flows from the cut separators.
      Builder b;
      const Index f1 = b.col(1.0, 0.0, kInf), f2 = b.col(2.0, 0.0, kInf), f3 = b.col(3.0, 0.0, kInf);
      const Index z = b.col(0.0, 0.0, kInf), y = b.col(4.0, 0.0, 10.0, true);
      b.row(0.0, 0.0, {{f1, 1.0}, {f2, 1.0}, {f3, 1.0}, {z, -1.0}});
      b.row(-kInf, 0.0, {{f1, 1.0}, {f2, 1.0}, {f3, 1.0}, {y, -7.0}});
      b.row(5.0, kInf, {{f1, 1.0}, {f2, 1.0}});
      Presolver pre;
      REQUIRE(pre.run(b.build(), o, kQuiet) == PresolveStatus::Reduced);
      CHECK_EQ(pre.stats().sparsify_rows, protect ? 0LL : 1LL);
    }
  }
}

namespace {

// Random pure-integer models small enough to enumerate (4-9 binaries, at most two integers in [0, 3]) built around
// 1-3 equations (general coefficients, now and then set partitioning) and rows containing most of an equation's
// columns with proportional coefficients (multiples +-1, 2, -2, 1/2, 3; sometimes one coefficient off by 1) and a few
// others: the rows sparsify combines.
MipProblem random_sparsify_model(std::mt19937& rng) {
  auto uni = [&](int lo, int hi) { return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)); };
  Builder b;
  std::vector<Index> all;
  std::vector<double> ub, x0;
  for (int j = uni(4, 9); j > 0; --j) all.push_back(b.col(uni(-5, 5), 0.0, 1.0, true)), ub.push_back(1.0);
  for (int j = uni(0, 2); j > 0; --j) {
    ub.push_back(uni(1, 3));
    all.push_back(b.col(uni(-3, 3), 0.0, ub.back(), true));
  }
  for (size_t j = 0; j < all.size(); ++j) x0.push_back(uni(0, static_cast<int>(ub[j])));
  auto pick = [&] { return all[uni(0, static_cast<int>(all.size()) - 1)]; };
  auto act = [&](const std::vector<std::pair<Index, double>>& c) {
    double v = 0.0;
    for (const auto& [j, a] : c) v += a * x0[j];
    return v;
  };
  auto unique = [](const std::vector<std::pair<Index, double>>& c) {
    std::vector<std::pair<Index, double>> u;
    for (const auto& e : c) {
      if (e.second != 0.0 && std::none_of(u.begin(), u.end(), [&](const auto& f) { return f.first == e.first; })) {
        u.push_back(e);
      }
    }
    return u;
  };
  std::vector<std::vector<std::pair<Index, double>>> eqs;
  for (int k = uni(1, 3); k > 0; --k) {
    std::vector<std::pair<Index, double>> c;
    const bool part = uni(0, 4) == 0;
    for (int t = uni(2, 5); t > 0; --t) c.push_back({pick(), part ? 1.0 : (uni(0, 1) ? 1.0 : -1.0) * uni(1, 3)});
    c = unique(c);
    if (c.size() < 2) continue;
    const double rhs = part ? 1.0 : act(c);  // feasible at x0 (or a set partitioning row)
    b.row(rhs, rhs, c);
    eqs.push_back(c);
  }
  if (eqs.empty()) return b.build();
  for (int i = uni(2, 7); i > 0; --i) {
    const auto& e = eqs[uni(0, static_cast<int>(eqs.size()) - 1)];
    const double mult[] = {1.0, -1.0, 2.0, -2.0, 0.5, 3.0};
    const double s = mult[uni(0, 5)];
    const int len = static_cast<int>(e.size());
    const int skip = len > 2 && uni(0, 1) ? uni(0, len - 1) : -1;
    std::vector<std::pair<Index, double>> c;
    for (int t = 0; t < len; ++t) {
      if (t != skip) c.push_back({e[t].first, s * e[t].second + (uni(0, 5) == 0 ? 1.0 : 0.0)});
    }
    for (int t = uni(0, 2); t > 0; --t) c.push_back({pick(), static_cast<double>(uni(-3, 4))});
    c = unique(c);
    if (c.size() < 2) continue;
    const double v = act(c);
    const int sense = uni(0, 4);
    b.row(sense == 0 ? -kInf : v - uni(0, 2), sense == 1 ? kInf : (sense == 2 ? v : v + uni(0, 2)), c);
  }
  return b.build();
}

}  // namespace

TEST(presolve_sparsify_round_trip_enumerated) {
  // Presolve with sparsify and the primal reductions (bounds, coefficients, parallel rows, aggregation, substitution,
  // probing in half of the cases, clique extension), or sparsify alone: the feasible points of the original problem
  // map exactly onto those of the reduced one (crush), and back (postsolve).
  std::mt19937 rng(fuzz_seed(24680u));
  const int iters = fuzz_iters(2000);
  long long combos = 0, compared = 0;
  for (int it = 0; it < iters; ++it) {
    const MipProblem p = random_sparsify_model(rng);
    const auto pts = enumerate_feasible(p);
    PresolveOptions o = only(true, true, true, false, true);
    o.substitution = true;
    o.probing = it % 2 == 0;
    if (it % 4 == 1) {
      o = only(false, false, false, false, false);
      o.clique_extension = false;
    }
    o.sparsify = true;
    Presolver pre;
    const PresolveStatus st = pre.run(p, o, kQuiet);
    if (st == PresolveStatus::Infeasible) {
      if (!CHECK(pts.empty())) std::fprintf(stderr, "  case %d: presolve infeasible, %zu feasible points\n", it, pts.size());
      continue;
    }
    REQUIRE(st == PresolveStatus::Reduced);
    combos += pre.stats().sparsify_rows;
    const MipProblem& q = pre.reduced();
    std::set<std::vector<double>> image;
    bool ok = true;
    for (const auto& x : pts) {
      const std::vector<double> z = pre.crush(x);
      ok = ok && q.max_violation(z) <= 1e-9;
      image.insert(z);
    }
    if (!CHECK(ok)) std::fprintf(stderr, "  case %d: presolve cut off a feasible point\n", it);
    if (q.num_cols() > 14) continue;
    const auto red = enumerate_feasible(q);
    ++compared;
    if (!CHECK_EQ(red.size(), image.size())) {
      std::fprintf(stderr, "  case %d: %zu reduced vs %zu original points\n", it, red.size(), image.size());
    }
    bool back = true;
    for (const auto& z : red) back = back && p.max_violation(pre.postsolve(z)) <= 1e-9;
    if (!CHECK(back)) std::fprintf(stderr, "  case %d: a reduced point postsolves to an infeasible one\n", it);
  }
  std::printf("  sparsify round trip: %lld rows combined, %lld compared exactly\n", combos, compared);
  CHECK(combos > iters / 4);
  CHECK(compared > iters / 2);
}

namespace {

// Random MILPs around equations shared with other rows: binary, integer and continuous columns (some unbounded),
// general and fractional coefficients, rows with multiples of most of an equation's entries plus a few others.
Model random_sparsify_milp(std::mt19937& rng) {
  auto uni = [&](int lo, int hi) { return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)); };
  ModelBuilder b;
  std::vector<Index> cols;
  std::vector<double> x0;
  for (int j = uni(4, 12); j > 0; --j) {
    const int kind = uni(0, 2);
    const double ub = kind == 0 ? 1.0 : uni(2, 8);
    cols.push_back(b.add_col("x" + std::to_string(j), uni(-5, 5), 0.0, kind == 0 || uni(0, 6) ? ub : kInf,
                             kind == 0 ? VarType::Binary : (kind == 1 ? VarType::Integer : VarType::Continuous)));
    x0.push_back(kind == 2 ? uni(0, 4 * static_cast<int>(ub)) * 0.25 : uni(0, static_cast<int>(ub)));
  }
  auto pick = [&] { return uni(0, static_cast<int>(cols.size()) - 1); };
  int r = 0;
  // A row around x0: activity - lo <= a x <= activity + hi (infinite: no side).
  auto row = [&](double lo, double hi, const std::vector<std::pair<int, double>>& c) {
    std::vector<std::pair<int, double>> u;
    for (const auto& e : c) {
      if (e.second != 0.0 && std::none_of(u.begin(), u.end(), [&](const auto& f) { return f.first == e.first; })) {
        u.push_back(e);
      }
    }
    if (u.size() < 2) return false;
    double act = 0.0;
    for (const auto& [j, v] : u) act += v * x0[j];
    const Index i = b.add_row("r" + std::to_string(r++), is_finite(lo) ? act - lo : -kInf, is_finite(hi) ? act + hi : kInf);
    for (const auto& [j, v] : u) b.add_coef(i, cols[j], v);
    return true;
  };
  std::vector<std::vector<std::pair<int, double>>> eqs;
  for (int k = uni(1, 3); k > 0; --k) {
    std::vector<std::pair<int, double>> c;
    for (int t = uni(2, 6); t > 0; --t) {
      c.push_back({pick(), uni(1, 8) * (uni(0, 1) ? 1.0 : -1.0) * (uni(0, 3) ? 1.0 : 0.25)});
    }
    if (row(0.0, 0.0, c)) eqs.push_back(c);
  }
  if (eqs.empty()) return b.build();
  for (int i = uni(2, 10); i > 0; --i) {
    const auto& e = eqs[uni(0, static_cast<int>(eqs.size()) - 1)];
    const double mult[] = {1.0, -1.0, 2.0, -0.5, 3.0, 0.1};
    const double s = mult[uni(0, 5)];
    std::vector<std::pair<int, double>> c;
    for (const auto& [j, v] : e) {
      if (uni(0, 4)) c.push_back({j, s * v});
    }
    for (int t = uni(0, 3); t > 0; --t) c.push_back({pick(), static_cast<double>(uni(-4, 4))});
    const int sense = uni(0, 3);
    row(sense == 0 ? kInf : uni(0, 3), sense == 1 ? kInf : (sense == 2 ? 0.0 : uni(0, 3)), c);
  }
  if (uni(0, 1)) b.model().sense = ObjSense::Maximize;
  return b.build();
}

}  // namespace

TEST(presolve_random_sparsify_milps_match_no_presolve) {
  std::mt19937 rng(fuzz_seed(97531u));
  const int iters = fuzz_iters(1000);
  int optimal = 0;
  // In half of the cases without the substitutions through other rows and with fill: they remove equations that
  // sparsify would add to other rows.
  PresolveOptions no_subst5;
  no_subst5.substitution_implied = false;
  no_subst5.substitution_fill = 0;
  for (int it = 0; it < iters; ++it) {
    optimal += compare_with_presolve(random_sparsify_milp(rng), it, it % 4 != 3, it % 2 == 0 ? &no_subst5 : nullptr) ==
               Status::Optimal;
  }
  const long long combos = g_stats.sparsify_rows;
  std::printf("  sparsify models: %d optimal\n", optimal);
  print_coverage("sparsify");
  CHECK(optimal > iters / 3);
  CHECK(combos > iters / 4);
}

namespace {

// Random bin-packing / knapsack models small enough to enumerate (6-12 binaries): rows a x_S + C y <= C (y = 1
// empties the bin S; C = sum a, or a little more or less), knapsacks, covering rows y + x_S >= 1 and packing
// rows, objective coefficients in [-6, 3] (mostly a maximization of weighted items).
MipProblem random_binpack_model(std::mt19937& rng, int min_cols = 6, int max_cols = 12) {
  auto uni = [&](int lo, int hi) { return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)); };
  Builder b;
  std::vector<Index> bin;
  for (int j = uni(min_cols, max_cols); j > 0; --j) bin.push_back(b.col(uni(-6, 3), 0.0, 1.0, true));
  const int n = static_cast<int>(bin.size());
  auto subset = [&](Index y, int k) {
    std::vector<Index> s;
    for (int t = 0; t < 4 * k && static_cast<int>(s.size()) < k; ++t) {
      const Index j = bin[uni(0, n - 1)];
      if (j != y && std::find(s.begin(), s.end(), j) == s.end()) s.push_back(j);
    }
    return s;
  };
  for (int i = uni(2, 7 * max_cols / 12); i > 0; --i) {
    const int kind = uni(0, 5);
    const Index y = bin[uni(0, n - 1)];
    std::vector<std::pair<Index, double>> c;
    if (kind <= 2) {  // a x_S + C y <= C
      double tot = 0.0;
      for (Index j : subset(y, uni(2, 6))) {
        const double a = uni(1, 4);
        c.push_back({j, a});
        tot += a;
      }
      const double C = tot + (kind == 0 ? 0.0 : (kind == 1 ? uni(1, 2) : -uni(1, 2)));
      c.push_back({y, C});
      if (C > 0.0) b.row(-kInf, C, c);
    } else if (kind == 3) {  // knapsack
      double tot = 0.0;
      for (Index j : subset(-1, uni(3, 6))) {
        const double a = uni(1, 9);
        c.push_back({j, a});
        tot += a;
      }
      b.row(-kInf, std::floor(tot * uni(3, 7) / 10.0), c);
    } else if (kind == 4) {  // y + x_S >= 1
      c.push_back({y, 1.0});
      for (Index j : subset(y, uni(1, 4))) c.push_back({j, 1.0});
      b.row(1.0, kInf, c);
    } else {  // packing
      for (Index j : subset(-1, uni(2, 4))) c.push_back({j, 1.0});
      b.row(-kInf, 1.0, c);
    }
  }
  return b.build();
}

}  // namespace

TEST(objective_cliques_star_row) {
  // y + x_k <= 1 for every item of a x + C y <= C with C = sum a (the prefix clique of the row has only the
  // heaviest item): y = 1 forces the items' objective -12 away (c x = -1 then), so y = 0 in every point with
  // c x <= -1.5.
  Builder b;
  const Index y = b.col(-1.0, 0.0, 1.0, true);
  std::vector<std::pair<Index, double>> row;
  for (int k = 1; k <= 4; ++k) row.push_back({b.col(-3.0, 0.0, 1.0, true), static_cast<double>(k)});
  row.push_back({y, 10.0});
  b.row(-kInf, 10.0, row);
  const MipProblem p = b.build();
  ObjectiveCliques oc(p, 1000);
  CHECK(oc.lists() == 1);
  CHECK(oc.entries() == 5);
  std::vector<double> lb = p.col_lb, ub = p.col_ub;
  std::vector<Index> changed;
  CHECK(oc.propagate(-1.0 + 1e-6, lb, ub, changed, 1000000) == 0);  // y = 1 (c x = -1) is still improving
  CHECK(oc.propagate(-1.5, lb, ub, changed, 1000000) == 1);
  REQUIRE(changed.size() == 1);
  CHECK(changed[0] == y);
  CHECK(ub[y] == 0.0);
  for (Index j = 1; j < p.num_cols(); ++j) CHECK(lb[j] == 0.0 && ub[j] == 1.0);
}

TEST(objective_cliques_keep_improving_points) {
  // Objective propagation with clique implications on enumerable bin-packing / knapsack models (and the probing
  // test models): at cutoffs from just below the optimum (nothing improving left) to a few units above, within
  // random sub-boxes, every feasible point of the box with c x <= cutoff satisfies the derived bounds.
  std::mt19937 rng(fuzz_seed(31337u));
  const int iters = fuzz_iters(2000);
  long long fixings = 0, cases = 0, unions = 0;
  for (int it = 0; it < iters; ++it) {
    const MipProblem p = it % 3 == 2 ? random_enum_model(rng) : random_binpack_model(rng);
    const auto pts = enumerate_feasible(p);
    if (pts.empty()) continue;
    const Index n = p.num_cols();
    auto obj = [&](const std::vector<double>& x) {
      double v = 0.0;
      for (Index j = 0; j < n; ++j) v += p.cost[j] * x[j];
      return v;
    };
    double zmin = kInf;
    for (const auto& x : pts) zmin = std::min(zmin, obj(x));
    ObjectiveCliques oc(p, 1000000);
    for (int t = 0; t < 4; ++t) {
      // The search's prune threshold for an integral objective (incumbent zmin + t - 1 + 1e-6), or a fractional one.
      const double cutoff = t == 3 ? zmin + 1.5 : zmin + t - 1.0 + 1e-6;
      std::vector<double> lb = p.col_lb, ub = p.col_ub;
      if (rng() % 2) {  // a sub-box: one or two columns fixed at the value of a random feasible point
        const auto& x0 = pts[rng() % pts.size()];
        for (int k = 1 + static_cast<int>(rng() % 2); k > 0; --k) {
          const Index j = static_cast<Index>(rng() % static_cast<unsigned>(n));
          lb[j] = ub[j] = x0[j];
        }
      }
      const std::vector<double> lb0 = lb, ub0 = ub;
      std::vector<Index> changed;
      const long long limit = it % 5 == 4 ? 5 + rng() % 40 : 1000000000;  // sometimes cut short by the work limit
      fixings += oc.propagate(cutoff, lb, ub, changed, limit);
      ++cases;
      bool ok = true;
      for (const auto& x : pts) {
        bool in_box = obj(x) <= cutoff + 1e-9;
        for (Index j = 0; j < n && in_box; ++j) in_box = x[j] >= lb0[j] && x[j] <= ub0[j];
        if (!in_box) continue;
        for (Index j = 0; j < n; ++j) ok = ok && x[j] >= lb[j] && x[j] <= ub[j];
      }
      if (!CHECK(ok)) std::fprintf(stderr, "  case %d, cutoff %g: a fixing cuts off an improving point\n", it, cutoff);
      for (Index j : changed) CHECK(lb[j] == ub[j]);
    }
    unions += oc.unions();
  }
  std::printf("  objective cliques: %lld fixings in %lld cases, %lld exact unions\n", fixings, cases, unions);
  CHECK(fixings > cases / 4);
  CHECK(unions > 0);
}

TEST(objective_cliques_solve_matches_enumeration) {
  // End to end (presolve on and off, 1 and 2 threads): the solver's optimum on enumerable bin-packing models
  // equals the enumerated one (the search applies the propagation at the root and with every incumbent).
  std::mt19937 rng(fuzz_seed(8642u));
  const int iters = fuzz_iters(300);
  int optimal = 0, infeasible = 0;
  for (int it = 0; it < iters; ++it) {
    const MipProblem p = random_binpack_model(rng, 14, 18);
    const auto pts = enumerate_feasible(p);
    SolverParams params;
    params.verbosity = 0;
    params.threads = 1 + it % 2;
    params.mip_rel_gap = 0.0;
    params.mip_abs_gap = 1e-9;
    params.presolve = it % 4 < 2;
    const SolveResult r = solve(to_model(p), params);
    if (pts.empty()) {
      ++infeasible;
      CHECK(r.status == Status::Infeasible || r.status == Status::InfeasibleOrUnbounded);
      continue;
    }
    double zmin = kInf;
    for (const auto& x : pts) {
      double v = 0.0;
      for (Index j = 0; j < p.num_cols(); ++j) v += p.cost[j] * x[j];
      zmin = std::min(zmin, v);
    }
    if (!CHECK(r.status == Status::Optimal)) {
      std::fprintf(stderr, "  case %d: status %s\n", it, to_string(r.status));
      continue;
    }
    ++optimal;
    if (!CHECK_NEAR(r.objective, zmin, 1e-6)) std::fprintf(stderr, "  case %d: objective %g, optimum %g\n", it, r.objective, zmin);
  }
  std::printf("  objective cliques solves: %d optimal, %d infeasible\n", optimal, infeasible);
  CHECK(optimal > iters / 2);
}
