// Tests for the logical modelling features: semi-continuous / semi-integer columns, SOS1/SOS2 sets and
// indicator constraints (readers, writer, Model::violation, reformulation, and solves against optima
// found by enumeration).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "milp/core/model.hpp"
#include "milp/io/lp_reader.hpp"
#include "milp/io/mps_reader.hpp"
#include "milp/solver/reformulate.hpp"
#include "milp/solver/solver.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

Index col_of(const Model& m, const std::string& name) {
  for (Index j = 0; j < m.num_cols(); ++j)
    if (m.col_names[j] == name) return j;
  return -1;
}

SolverParams quiet(double big_m = 1e6, bool presolve = true) {
  SolverParams p;
  p.verbosity = 0;
  p.threads = 1;
  p.time_limit = 30;
  p.mip_rel_gap = 1e-9;
  p.logic_big_m = big_m;
  p.presolve = presolve;
  return p;
}

// Solves with big-M reformulation and with implications only (branching), with and without presolve.
void check_optimum(const Model& m, Status status, double obj, const char* what) {
  // The last mode: implications in a deterministic parallel search.
  const std::pair<double, bool> modes[] = {{1e6, true}, {0.0, true}, {1e6, false}, {-1.0, true}};
  for (const auto& [big_m, presolve] : modes) {
    SolverParams sp = quiet(std::max(big_m, 0.0), presolve);
    if (big_m < 0) sp.threads = 2, sp.deterministic = true;
    const SolveResult r = solve(m, sp);
    const bool ok = r.status == status && (status != Status::Optimal || std::abs(r.objective - obj) <= 1e-5 * (1 + std::abs(obj)));
    if (!ok) {
      std::fprintf(stderr, "  %s (big-M %g, presolve %d): status %s objective %.10g, expected %s %.10g\n", what, big_m,
                   presolve, to_string(r.status), r.objective, to_string(status), obj);
    }
    CHECK(ok);
    if (r.has_solution) {
      const Model::Violation v = m.violation(r.x);
      if (!(v.max() <= 1e-6)) {
        std::fprintf(stderr, "  %s (big-M %g, presolve %d): violation bound %.3g row %.3g int %.3g logical %.3g\n", what,
                     big_m, presolve, v.bound, v.row, v.integrality, v.logical);
        write_mps(m, "logical_fail.mps");
      }
      CHECK(v.max() <= 1e-6);
    }
  }
}

}  // namespace

TEST(logical_lp_reader) {
  const Model m = read_lp_string(
      "max\n obj: x + 2 y + 3 z + w + v\n"
      "st\n c1: x + y + z <= 10\n"
      " ind1: b = 1 -> x + y <= 2\n"
      " b = 0 -> 1 <= z - w <= 4\n"
      "bounds\n 0 <= x <= 8\n 2 <= z <= 6\n v <= 9\n 1 <= w <= 3\n"
      "generals\n w\n"
      "semi-continuous\n z\n w\n"
      "sos\n s1: S1:: x:1 y:2\n S2:: x:1 y:2 v:3\n"
      "end\n");
  CHECK_EQ(m.num_rows(), 1);
  CHECK(m.col_type[col_of(m, "z")] == VarType::SemiContinuous);
  CHECK(m.col_type[col_of(m, "w")] == VarType::SemiInteger);
  CHECK(m.col_type[col_of(m, "b")] == VarType::Binary);  // made binary as an indicator variable
  CHECK_EQ(m.col_lb[col_of(m, "z")], 2.0);
  CHECK_EQ(m.col_ub[col_of(m, "z")], 6.0);
  REQUIRE(m.indicators.size() == 2);
  CHECK_EQ(m.indicators[0].name, std::string("ind1"));
  CHECK_EQ(m.indicators[0].binary, col_of(m, "b"));
  CHECK_EQ(m.indicators[0].value, 1);
  CHECK_EQ(m.indicators[0].cols.size(), size_t{2});
  CHECK_EQ(m.indicators[0].ub, 2.0);
  CHECK(m.indicators[0].lb == -kInf);
  CHECK_EQ(m.indicators[1].value, 0);
  CHECK_EQ(m.indicators[1].lb, 1.0);
  CHECK_EQ(m.indicators[1].ub, 4.0);
  CHECK_EQ(m.indicators[1].coefs[1], -1.0);
  REQUIRE(m.sos.size() == 2);
  CHECK_EQ(m.sos[0].name, std::string("s1"));
  CHECK_EQ(m.sos[0].type, 1);
  CHECK_EQ(m.sos[1].type, 2);
  CHECK_EQ(m.sos[1].cols.size(), size_t{3});
  CHECK_EQ(m.sos[1].weights[2], 3.0);
  // "SOS2" section headers give the type of entries without "S2::".
  const Model m2 = read_lp_string("min\n x + y\nst\n c: x + y >= 1\nsos2\n s: x:1 y:2\nend\n");
  REQUIRE(m2.sos.size() == 1);
  CHECK_EQ(m2.sos[0].type, 2);
}

TEST(logical_mps_reader_writer) {
  const std::string text =
      "NAME t\nOBJSENSE\n    MAX\nROWS\n N obj\n L c1\n L ind1\n L ind2\n G c3\n"
      "COLUMNS\n    x obj 1 c1 1\n    x ind1 1\n    y obj 2 c1 1\n    y ind1 1\n    z obj 3 c1 1\n    z ind2 1\n"
      "    w obj -1 c3 1\n    MARKER 'MARKER' 'INTORG'\n    b c3 -1\n    k obj 1 c1 1\n    MARKER 'MARKER' 'INTEND'\n"
      "RHS\n    RHS c1 10 ind1 2\n    RHS ind2 1\n"
      "BOUNDS\n UP BND x 8\n UP BND y 8\n LO BND z 2\n SC BND z 6\n UP BND w 5\n BV BND b\n LO BND k 1\n SC BND k 3\n"
      "SOS\n S1 SOS s1 1\n    x 1\n    y 2\n S2 s2\n    x:1\n    y:2\n    w:3\n"
      "INDICATORS\n IF ind1 b 1\n IF ind2 b 0\nENDATA\n";
  const Model m = read_mps_string(text);
  CHECK_EQ(m.num_rows(), 2);  // the indicator rows are not ordinary rows
  CHECK_EQ(m.row_names[0], std::string("c1"));
  CHECK_EQ(m.row_names[1], std::string("c3"));
  CHECK(m.col_type[col_of(m, "z")] == VarType::SemiContinuous);
  CHECK(m.col_type[col_of(m, "k")] == VarType::SemiInteger);
  CHECK_EQ(m.col_ub[col_of(m, "k")], 3.0);
  REQUIRE(m.sos.size() == 2);
  CHECK_EQ(m.sos[0].name, std::string("s1"));
  CHECK_EQ(m.sos[1].name, std::string("s2"));
  CHECK_EQ(m.sos[1].type, 2);
  CHECK_EQ(m.sos[1].weights[2], 3.0);
  REQUIRE(m.indicators.size() == 2);
  CHECK_EQ(m.indicators[1].name, std::string("ind2"));
  CHECK_EQ(m.indicators[1].value, 0);
  CHECK_EQ(m.indicators[1].ub, 1.0);
  // Round trip through the writer.
  const Model w = read_mps_string(write_mps_string(m));
  CHECK_EQ(w.num_rows(), m.num_rows());
  CHECK_EQ(w.num_cols(), m.num_cols());
  for (Index j = 0; j < m.num_cols(); ++j) {
    CHECK(w.col_type[j] == m.col_type[j]);
    CHECK_EQ(w.col_lb[j], m.col_lb[j]);
    CHECK_EQ(w.col_ub[j], m.col_ub[j]);
  }
  REQUIRE(w.sos.size() == 2);
  for (size_t s = 0; s < 2; ++s) {
    CHECK_EQ(w.sos[s].type, m.sos[s].type);
    CHECK(w.sos[s].cols == m.sos[s].cols);
    CHECK(w.sos[s].weights == m.sos[s].weights);
  }
  REQUIRE(w.indicators.size() == 2);
  for (size_t q = 0; q < 2; ++q) {
    CHECK_EQ(w.indicators[q].binary, m.indicators[q].binary);
    CHECK_EQ(w.indicators[q].value, m.indicators[q].value);
    CHECK(w.indicators[q].cols == m.indicators[q].cols);
    CHECK(w.indicators[q].coefs == m.indicators[q].coefs);
    CHECK_EQ(w.indicators[q].lb, m.indicators[q].lb);
    CHECK_EQ(w.indicators[q].ub, m.indicators[q].ub);
  }
  // Both reformulations (big-M and implications) agree on the optimum.
  const SolveResult r = solve(m, quiet());
  CHECK(r.status == Status::Optimal);
  CHECK(m.violation(r.x).max() <= 1e-6);
  const SolveResult r2 = solve(w, quiet(0.0));
  CHECK(r2.status == Status::Optimal);
  CHECK_NEAR(r.objective, r2.objective, 1e-6);
}

TEST(logical_violation) {
  ModelBuilder b;
  const Index x = b.add_col("x", 0, 2, 5, VarType::SemiContinuous);
  const Index y = b.add_col("y", 0, 0, 10);
  const Index z = b.add_col("z", 0, 0, 10);
  const Index u = b.add_col("u", 0, 0, 1, VarType::Binary);
  Model m = b.build();
  m.sos.push_back({"s1", 1, {y, z}, {1, 2}});
  m.sos.push_back({"s2", 2, {x, y, z}, {1, 2, 3}});
  m.indicators.push_back({"i", u, 1, {y}, {1.0}, -kInf, 3.0});
  m.validate();
  CHECK_EQ(m.violation({0, 0, 0, 0}).max(), 0.0);
  CHECK_EQ(m.violation({3, 0, 0, 0}).max(), 0.0);
  CHECK_NEAR(m.violation({1, 0, 0, 0}).bound, 1.0, 1e-12);    // between 0 and the semi-continuous lower bound
  CHECK_NEAR(m.violation({0.5, 0, 0, 0}).bound, 0.5, 1e-12);
  CHECK_NEAR(m.violation({0, 2, 4, 0}).logical, 2.0, 1e-12);  // SOS1: the smaller of the two is nonzero
  CHECK_NEAR(m.violation({3, 0, 1, 0}).logical, 1.0, 1e-12);  // SOS2: x and z are not neighbours
  CHECK_EQ(m.violation({0, 4, 0, 0}).logical, 0.0);           // the indicator is inactive
  CHECK_NEAR(m.violation({0, 4, 0, 1}).logical, 1.0, 1e-12);  // active: y <= 3
  Model bad = m;
  bad.indicators[0].value = 2;
  CHECK_THROWS(bad.validate(), std::runtime_error);
  bad = m;
  bad.sos[0].weights.pop_back();
  CHECK_THROWS(bad.validate(), std::runtime_error);
}

TEST(logical_semicontinuous_optima) {
  // min x + 10 y  s.t.  x + y >= 1, x semi-continuous in {0} u [2, 5]  -> x = 2, objective 2.
  ModelBuilder b;
  const Index x = b.add_col("x", 1, 2, 5, VarType::SemiContinuous);
  const Index y = b.add_col("y", 10, 0, kInf);
  const Index r = b.add_row("c", 1, kInf);
  b.add_coef(r, x, 1);
  b.add_coef(r, y, 1);
  Model m = b.build();
  check_optimum(m, Status::Optimal, 2.0, "semi-continuous min");
  // An extra row x <= 0.5 forces x = 0: y = 1, objective 10.
  ModelBuilder b2;
  b2.add_col("x", 1, 2, 5, VarType::SemiContinuous);
  b2.add_col("y", 10, 0, kInf);
  const Index c2 = b2.add_row("c", 1, kInf);
  b2.add_coef(c2, 0, 1);
  b2.add_coef(c2, 1, 1);
  const Index d2 = b2.add_row("d", -kInf, 0.5);
  b2.add_coef(d2, 0, 1);
  check_optimum(b2.build(), Status::Optimal, 10.0, "semi-continuous forced to 0");
  // Semi-continuous without upper bound: max x - 0.4 t with x in {0} u [4, inf), x <= 2 t + 1, t <= 7 integer.
  ModelBuilder b3;
  b3.model().sense = ObjSense::Maximize;
  const Index x3 = b3.add_col("x", 1, 4, kInf, VarType::SemiContinuous);
  const Index t3 = b3.add_col("t", -0.4, 0, 7, VarType::Integer);
  const Index r3 = b3.add_row("c", -kInf, 1);
  b3.add_coef(r3, x3, 1);
  b3.add_coef(r3, t3, -2);
  // Best: t = 7, x = 15: 15 - 2.8 = 12.2.
  check_optimum(b3.build(), Status::Optimal, 12.2, "semi-continuous without upper bound");
  // Semi-integer: min x with x in {0} u [3, 6] integer, x >= 2.5 -> x = 3.
  ModelBuilder b4;
  const Index x4 = b4.add_col("x", 1, 3, 6, VarType::SemiInteger);
  const Index r4 = b4.add_row("c", 2.5, kInf);
  b4.add_coef(r4, x4, 1);
  check_optimum(b4.build(), Status::Optimal, 3.0, "semi-integer");
  // Infeasible: x in {0} u [3, 6], 1 <= x <= 2.
  ModelBuilder b5;
  const Index x5 = b5.add_col("x", 1, 3, 6, VarType::SemiContinuous);
  const Index r5 = b5.add_row("c", 1, 2);
  b5.add_coef(r5, x5, 1);
  check_optimum(b5.build(), Status::Infeasible, 0.0, "semi-continuous infeasible");
}

TEST(logical_sos_optima) {
  // Piecewise-linear (non-concave) f with breakpoints 0..4 and values 0, 3, 1, 4, 0 (lambda formulation,
  // SOS2 on the lambdas): max f(x) - 0.5 x with x <= 2.2. The LP relaxation mixes breakpoints 1 and 3
  // (x = 2.2: f = 3.6); with SOS2 the optimum is 2.5 at x = 1.
  const double F[5] = {0, 3, 1, 4, 0};
  ModelBuilder b;
  b.model().sense = ObjSense::Maximize;
  std::vector<Index> lam;
  for (int t = 0; t < 5; ++t) lam.push_back(b.add_col("l" + std::to_string(t), F[t] - 0.5 * t, 0, kInf));
  const Index conv = b.add_row("conv", 1, 1);
  const Index xr = b.add_row("xmax", -kInf, 2.2);
  for (int t = 0; t < 5; ++t) {
    b.add_coef(conv, lam[t], 1);
    b.add_coef(xr, lam[t], t);
  }
  Model m = b.build();
  m.sos.push_back({"pwl", 2, lam, {1, 2, 3, 4, 5}});
  // Enumerate the segments: on [t, t+1] the objective is linear, so its maximum is at an end point (<= 2.2).
  double best = -kInf;
  for (double xv : {0.0, 1.0, 2.0, 2.2}) {
    const int t = std::min(3, static_cast<int>(std::floor(xv)));
    const double a = xv - t;
    best = std::max(best, (1 - a) * F[t] + a * F[t + 1] - 0.5 * xv);
  }
  CHECK_NEAR(best, 2.5, 1e-12);
  check_optimum(m, Status::Optimal, best, "SOS2 piecewise linear");
  // SOS1: max 3a + 2b + c with a + b + c <= 4, a, b, c <= 3 (continuous): one nonzero -> a = 3: 9.
  ModelBuilder b2;
  b2.model().sense = ObjSense::Maximize;
  const Index a2 = b2.add_col("a", 3, 0, 3), bb = b2.add_col("b", 2, 0, 3), c2 = b2.add_col("c", 1, 0, 3);
  const Index r2 = b2.add_row("cap", -kInf, 4);
  for (Index j : {a2, bb, c2}) b2.add_coef(r2, j, 1);
  Model m2 = b2.build();
  m2.sos.push_back({"s", 1, {a2, bb, c2}, {1, 2, 3}});
  check_optimum(m2, Status::Optimal, 9.0, "SOS1 continuous");
  // SOS1 of binaries is a packing row: max a + b + c -> 1.
  ModelBuilder b3;
  b3.model().sense = ObjSense::Maximize;
  std::vector<Index> bin;
  for (int t = 0; t < 3; ++t) bin.push_back(b3.add_col("x" + std::to_string(t), 1, 0, 1, VarType::Binary));
  Model m3 = b3.build();
  m3.sos.push_back({"s", 1, bin, {1, 2, 3}});
  check_optimum(m3, Status::Optimal, 1.0, "SOS1 binaries");
  // SOS1 with members without upper bounds (only an implication can link them): min -a - b + 2 s ... with
  // a + b <= 5 + s, s <= 10 -> a or b up to 15: objective -15 + 20 = 5? Better s = 0: -5. Members a, b free above.
  ModelBuilder b4;
  const Index a4 = b4.add_col("a", -1, 0, kInf), b4c = b4.add_col("b", -1.5, 0, kInf), s4 = b4.add_col("s", 2, 0, 10);
  const Index r4 = b4.add_row("cap", -kInf, 5);
  b4.add_coef(r4, a4, 1);
  b4.add_coef(r4, b4c, 1);
  b4.add_coef(r4, s4, -1);
  Model m4 = b4.build();
  m4.sos.push_back({"s", 1, {a4, b4c}, {1, 2}});
  // b = 5 + s: -1.5 (5 + s) + 2 s = -7.5 + 0.5 s -> s = 0: -7.5.
  check_optimum(m4, Status::Optimal, -7.5, "SOS1 unbounded members");
}

TEST(logical_indicator_optima) {
  // min t + 3 b  s.t. x >= 5, t >= 0; indicator b = 0 -> x <= 2 (x has no upper bound: no big-M).
  // b = 0 is infeasible, so b = 1: objective 3.
  ModelBuilder bl;
  const Index x = bl.add_col("x", 0, 0, kInf);
  const Index t = bl.add_col("t", 1, 0, kInf);
  const Index b = bl.add_col("b", 3, 0, 1, VarType::Binary);
  const Index r = bl.add_row("c", 5, kInf);
  bl.add_coef(r, x, 1);
  Model m = bl.build();
  (void)t;
  m.indicators.push_back({"i", b, 0, {x}, {1.0}, -kInf, 2.0});
  check_optimum(m, Status::Optimal, 3.0, "indicator without big-M");
  // Facility: open y (cost 10) or not; if y = 0 then flow f = 0; demand f + g >= 4, g costs 4 per unit, f costs 1.
  // f <= 100 gives a big-M. Open: 10 + 4 = 14 < 16 (g = 4).
  ModelBuilder bf;
  const Index y = bf.add_col("y", 10, 0, 1, VarType::Binary);
  const Index f = bf.add_col("f", 1, 0, 100);
  const Index g = bf.add_col("g", 4, 0, kInf);
  const Index d = bf.add_row("dem", 4, kInf);
  bf.add_coef(d, f, 1);
  bf.add_coef(d, g, 1);
  Model mf = bf.build();
  mf.indicators.push_back({"close", y, 0, {f}, {1.0}, -kInf, 0.0});
  check_optimum(mf, Status::Optimal, 14.0, "indicator facility");
  // Equality indicator with the indicator variable in its own row: b = 1 -> x + b = 3 (x <= 10): max x - 4 b.
  // b = 1: x = 2 -> -2; b = 0: x = 10 -> 10.  With x <= 1.5 instead: b = 1 infeasible, b = 0: 1.5.
  ModelBuilder be;
  be.model().sense = ObjSense::Maximize;
  const Index xe = be.add_col("x", 1, 0, 10);
  const Index bev = be.add_col("b", -4, 0, 1, VarType::Binary);
  const Index fix = be.add_row("fixb", 1, 1);
  be.add_coef(fix, bev, 1);
  Model me = be.build();
  me.indicators.push_back({"eq", bev, 1, {xe, bev}, {1.0, 1.0}, 3.0, 3.0});
  check_optimum(me, Status::Optimal, -2.0, "indicator equality (b fixed to 1)");
  // Unbounded LP relaxation, bounded problem: max x - 0.1 c with b + c >= 1, b = 1 -> x <= 5, c = 1 -> x <= 7
  // (x has no upper bound, so the indicators keep slack columns that are free in the LP). Optimum 6.9.
  ModelBuilder bu;
  bu.model().sense = ObjSense::Maximize;
  const Index xu = bu.add_col("x", 1, 0, kInf);
  const Index b1 = bu.add_col("b", 0, 0, 1, VarType::Binary);
  const Index c1 = bu.add_col("c", -0.1, 0, 1, VarType::Binary);
  const Index ru = bu.add_row("one", 1, kInf);
  bu.add_coef(ru, b1, 1);
  bu.add_coef(ru, c1, 1);
  Model mu = bu.build();
  mu.indicators.push_back({"ib", b1, 1, {xu}, {1.0}, -kInf, 5.0});
  Model mu_unbd = mu;
  mu.indicators.push_back({"ic", c1, 1, {xu}, {1.0}, -kInf, 7.0});
  check_optimum(mu, Status::Optimal, 6.9, "unbounded LP relaxation, bounded problem");
  // Without the second indicator, c = 1 leaves x unbounded.
  check_optimum(mu_unbd, Status::Unbounded, 0.0, "unbounded through an inactive indicator");
}

namespace {

// Random small integer model with semi-integer columns, SOS sets and indicators, solved by enumeration.
struct RandomCase {
  Model m;
  std::vector<std::vector<double>> dom;  // values each column may take (before logical constraints)
};

RandomCase random_case(std::mt19937& rng) {
  auto uni = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
  RandomCase c;
  ModelBuilder b;
  const int n = uni(4, 6);
  for (int j = 0; j < n; ++j) {
    const int kind = j < 2 ? 0 : uni(1, 3);  // 0 binary, 1 integer, 2 semi-integer, 3 integer around 0
    std::vector<double> d;
    if (kind == 0) {
      b.add_col("x" + std::to_string(j), uni(-5, 5), 0, 1, VarType::Binary);
      d = {0, 1};
    } else if (kind == 2) {
      const int l = uni(1, 2), u = l + uni(0, 2);
      b.add_col("x" + std::to_string(j), uni(-5, 5), l, u, VarType::SemiInteger);
      d.push_back(0);
      for (int v = l; v <= u; ++v) d.push_back(v);
    } else {
      const int l = kind == 3 ? -uni(0, 2) : 0, u = uni(1, 3);
      b.add_col("x" + std::to_string(j), uni(-5, 5), l, u, VarType::Integer);
      for (int v = l; v <= u; ++v) d.push_back(v);
    }
    c.dom.push_back(d);
  }
  const int rows = uni(1, 3);
  for (int i = 0; i < rows; ++i) {
    const Index r = b.add_row("r" + std::to_string(i), -kInf, uni(2, 8));
    for (int j = 0; j < n; ++j)
      if (uni(0, 1)) b.add_coef(r, j, uni(-3, 4));
  }
  b.model().sense = uni(0, 1) ? ObjSense::Maximize : ObjSense::Minimize;
  c.m = b.build();
  Model& m = c.m;
  const int nsos = uni(0, 2);
  for (int s = 0; s < nsos; ++s) {
    SosConstraint set;
    set.type = uni(1, 2);
    std::vector<Index> perm(n);
    for (int j = 0; j < n; ++j) perm[j] = j;
    std::shuffle(perm.begin(), perm.end(), rng);
    const int k = uni(3, std::min(n, 4));
    for (int t = 0; t < k; ++t) {
      set.cols.push_back(perm[t]);
      set.weights.push_back(uni(0, 1000) + 0.001 * t);
    }
    m.sos.push_back(set);
  }
  const int nind = uni(0, 2);
  for (int q = 0; q < nind; ++q) {
    IndicatorConstraint ic;
    ic.binary = uni(0, 1);
    ic.value = uni(0, 1);
    for (int j = 2; j < n; ++j) {
      if (uni(0, 1)) {
        ic.cols.push_back(j);
        ic.coefs.push_back(uni(-2, 3));
      }
    }
    const int side = uni(0, 2);
    const double rhs = uni(0, 4);
    if (side != 1) ic.ub = rhs;
    if (side != 0) ic.lb = side == 2 ? rhs - uni(0, 2) : rhs - 1;
    m.indicators.push_back(ic);
  }
  m.validate();
  return c;
}

// Exhaustive optimum (internal minimization); returns false if infeasible.
bool enumerate(const RandomCase& c, double* best) {
  const Model& m = c.m;
  const Index n = m.num_cols();
  std::vector<size_t> idx(n, 0);
  std::vector<double> x(n);
  bool found = false;
  const double sense = m.sense == ObjSense::Maximize ? -1.0 : 1.0;
  for (;;) {
    for (Index j = 0; j < n; ++j) x[j] = c.dom[j][idx[j]];
    if (m.violation(x).max() <= 1e-9) {
      const double v = sense * m.objective_value(x);
      if (!found || v < *best) *best = v;
      found = true;
    }
    Index j = 0;
    while (j < n && ++idx[j] == c.dom[j].size()) idx[j++] = 0;
    if (j == n) break;
  }
  if (found) *best *= sense;
  return found;
}

}  // namespace

TEST(logical_random_enumeration) {
  int iters = 150;
  if (const char* e = std::getenv("MILP_FUZZ_ITERS")) iters = std::max(1, std::atoi(e) / 10);
  std::mt19937 rng(20261102);
  int feasible = 0;
  for (int it = 0; it < iters; ++it) {
    const RandomCase c = random_case(rng);
    double best = 0.0;
    const bool ok = enumerate(c, &best);
    feasible += ok;
    const std::string what = "random case " + std::to_string(it);
    check_optimum(c.m, ok ? Status::Optimal : Status::Infeasible, best, what.c_str());
  }
  CHECK(feasible > iters / 3);
}

TEST(logical_random_continuous_consistency) {
  // With continuous and semi-continuous columns (no enumeration): the big-M reformulation, the
  // implications-only one and the solve without presolve agree, and the solutions are feasible.
  int iters = 100;
  if (const char* e = std::getenv("MILP_FUZZ_ITERS")) iters = std::max(1, std::atoi(e) / 10);
  std::mt19937 rng(4242);
  for (int it = 0; it < iters; ++it) {
    RandomCase c = random_case(rng);
    for (Index j = 2; j < c.m.num_cols(); ++j) {
      if (rng() % 2 == 0) continue;
      VarType& t = c.m.col_type[j];
      t = t == VarType::SemiInteger ? VarType::SemiContinuous : VarType::Continuous;
    }
    const SolveResult ref = solve(c.m, quiet());
    const std::string what = "continuous case " + std::to_string(it);
    check_optimum(c.m, ref.status, ref.objective, what.c_str());
  }
}

TEST(logical_reformulation_extend) {
  // The auxiliary values derived from a feasible point make it feasible for the reformulation.
  std::mt19937 rng(7);
  int checked = 0;
  for (int it = 0; it < 60; ++it) {
    const RandomCase c = random_case(rng);
    for (double big_m : {1e6, 0.0}) {
      const LogicalReformulation R = reformulate_logical(c.m, big_m);
      const SolveResult r = solve(c.m, quiet(big_m));
      if (!r.has_solution) continue;
      MipProblem p;
      REQUIRE(MipProblem::from_model(R.model, &p));
      p.implications = R.implications;
      const std::vector<double> y = R.extend(r.x);
      CHECK(p.max_violation(y) <= 1e-6);
      ++checked;
    }
  }
  CHECK(checked > 20);
}

TEST(logical_start_and_indicator_errors) {
  // A complete start of the original columns is extended to the auxiliary ones.
  ModelBuilder b;
  const Index y = b.add_col("y", 10, 0, 1, VarType::Binary);
  const Index f = b.add_col("f", 1, 0, 100);
  const Index g = b.add_col("g", 4, 0, kInf);
  const Index d = b.add_row("dem", 4, kInf);
  b.add_coef(d, f, 1);
  b.add_coef(d, g, 1);
  Model m = b.build();
  m.indicators.push_back({"close", y, 0, {f}, {1.0}, -kInf, 0.0});
  const SolveResult r = solve(m, quiet(), {0.0, 0.0, 4.0});
  CHECK(r.status == Status::Optimal);
  CHECK_NEAR(r.objective, 14.0, 1e-6);
  // The indicator variable must be binary.
  Model bad = m;
  bad.indicators[0].binary = f;
  CHECK_THROWS(solve(bad, quiet()), std::runtime_error);
}
