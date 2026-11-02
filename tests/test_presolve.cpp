// Tests for presolve reductions and postsolve, plus randomized end-to-end
// comparisons of solving with and without presolve.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "milp/io/mps_reader.hpp"
#include "milp/presolve/clique_merge.hpp"
#include "milp/presolve/presolve.hpp"
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
  return o;
}

PresolveOptions only_subst() {
  PresolveOptions o = only(false, false, false, false, false);
  o.substitution = true;
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

void print_coverage(const char* name) {
  std::printf("  %s reductions: %lld aggregations, %lld substitutions, %lld parallel cols, %lld dominated, probing "
              "%lld fixed / %lld bounds / %lld equivs, cliques %lld extended / %lld dominated / %lld fixed\n",
              name, g_stats.aggregations, g_stats.substitutions, g_stats.parallel_cols, g_stats.dominated,
              g_stats.probing_fixed, g_stats.probing_bounds, g_stats.probing_equivs, g_stats.clique_extended,
              g_stats.clique_dominated, g_stats.clique_fixed);
  if (g_numeric > 0) std::printf("  %s: %d cases skipped (numerical trouble)\n", name, g_numeric);
  g_stats = PresolveStats{};
  g_numeric = 0;
}

// Solves m with and without presolve (presolve + reduced solve + postsolve, without the solver's safety fallback)
// and checks that both agree. Returns the status without presolve (NodeLimit if skipped).
Status compare_with_presolve(const Model& m, int it) {
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
  const PresolveStatus ps = valid ? pre.run(p, PresolveOptions{}, kQuiet) : PresolveStatus::Infeasible;
  g_stats.aggregations += pre.stats().aggregations;
  g_stats.substitutions += pre.stats().substitutions;
  g_stats.probing_fixed += pre.stats().probing_fixed;
  g_stats.probing_bounds += pre.stats().probing_bounds;
  g_stats.probing_equivs += pre.stats().probing_equivs;
  g_stats.parallel_cols += pre.stats().parallel_cols;
  g_stats.dominated += pre.stats().dominated;
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
  const bool same_obj = a.status != Status::Optimal ||
                        std::abs(a.objective - b.objective) <= 1e-6 * std::max(1.0, std::abs(a.objective));
  const char* dump = std::getenv("MILP_FUZZ_DUMP");  // directory for failing cases
  if (dump && (!ok_status || !same_obj)) write_mps(m, std::string(dump) + "/case" + std::to_string(it) + ".mps");
  if (!ok_status) {
    std::fprintf(stderr, "  case %d: status %s (no presolve) vs %s (presolve)\n", it, to_string(a.status),
                 to_string(b.status));
    return a.status;
  }
  if (a.status == Status::Optimal) {
    REQUIRE(b.has_solution);
    const double tol = 1e-6 * std::max(1.0, std::abs(a.objective));
    if (!CHECK_NEAR(a.objective, b.objective, tol)) std::fprintf(stderr, "  case %d: objective mismatch\n", it);
    const auto v = m.violation(b.x);
    if (!CHECK(v.max() <= 1e-6)) std::fprintf(stderr, "  case %d: violation %g\n", it, v.max());
    // The full pipeline (with its fallback) must agree as well.
    params.presolve = true;
    const SolveResult c = solve(m, params);
    CHECK(c.status == Status::Optimal);
    CHECK_NEAR(a.objective, c.objective, tol);
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
