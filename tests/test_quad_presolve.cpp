// The presolve for mixed-integer models with quadratic rows (solver/quad_presolve.hpp): alias members, fixed
// columns in quadratic terms, perspective strengthening; and the solves that go through it (solver.cpp solve_oa),
// both search modes, with the cone implications and the conic subproblems at integral points.

#include <cmath>
#include <string>
#include <vector>

#include "milp/io/lp_reader.hpp"
#include "milp/solver/quad_presolve.hpp"
#include "milp/solver/solver.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

SolveResult solve_model(const Model& m, bool det) {
  SolverParams p;
  p.verbosity = 0;
  p.threads = det ? 2 : 1;
  p.deterministic = det;
  p.mip_rel_gap = 1e-9;
  p.mip_abs_gap = 1e-9;
  return solve(m, p);
}

Index col(const Model& m, const std::string& name) {
  for (Index j = 0; j < m.num_cols(); ++j) {
    if (m.col_names[j] == name) return j;
  }
  return -1;
}

}  // namespace

TEST(quad_presolve_alias_members) {
  // A conic-format style model: the cone m1^2 <= 2 m2 m3 over columns defined by equalities, m1 = x, m2 = u / 2,
  // m3 = z (binary): x^2 <= u z. y = 1 (worth 10) needs x >= 1.5, so z = 1 (cost 3) and u = 2.25: -4.75.
  const Model m = read_lp_string(
      "min\n u + 3 z - 10 y\nst\n a: m1 - x = 0\n b: 2 m2 - u = 0\n c: m3 - z = 0\n d: x - 1.5 y >= 0\n"
      " k: [ m1^2 - 2 m2 * m3 ] <= 0\nbounds\n m1 free\n -10 <= x <= 10\n 0 <= u <= 100\nbinary\n z y\nend\n");
  Model pm;
  const QuadPresolve ps = quad_presolve(m, pm);
  CHECK(ps.changed());
  CHECK(ps.aliases.size() == 3);
  CHECK(ps.perspective == 0);
  CHECK(pm.num_cols() == m.num_cols());
  CHECK(pm.num_rows() == m.num_rows());
  // The quadratic row is over x, u and z now: x^2 - u z.
  const Index x = col(m, "x"), u = col(m, "u"), z = col(m, "z");
  CHECK(pm.qr_val.size() == 2);
  double sq = 0.0, prod = 0.0;
  for (size_t k = 0; k < pm.qr_val.size(); ++k) {
    if (pm.qr_i[k] == x && pm.qr_j[k] == x) sq = pm.qr_val[k];
    if (std::max(pm.qr_i[k], pm.qr_j[k]) == std::max(u, z) && std::min(pm.qr_i[k], pm.qr_j[k]) == std::min(u, z)) {
      prod = pm.qr_val[k];
    }
  }
  CHECK_NEAR(sq, 1.0, 1e-12);
  CHECK_NEAR(prod, -1.0, 1e-12);
  // restore() recomputes the alias columns.
  std::vector<double> pt(static_cast<size_t>(m.num_cols()), 0.0);
  pt[x] = 1.5, pt[u] = 2.25, pt[z] = 1.0, pt[col(m, "y")] = 1.0;
  ps.restore(pt);
  CHECK_NEAR(pt[col(m, "m1")], 1.5, 1e-12);
  CHECK_NEAR(pt[col(m, "m2")], 1.125, 1e-12);
  CHECK_NEAR(pt[col(m, "m3")], 1.0, 1e-12);
  CHECK(m.violation(pt).max() <= 1e-9);
  for (const bool det : {false, true}) {
    const SolveResult r = solve_model(m, det);
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -4.75, 1e-6);
    REQUIRE(r.has_solution);
    CHECK(m.violation(r.x).max() <= 1e-5);
    CHECK_NEAR(r.x[col(m, "m1")], r.x[x], 1e-9);
  }
  // A member with an offset and a negative scale (not replaced): m1 = 2 - x, so (2 - x)^2 <= u z; x <= 0.5 y' ... with x <= 0.5 the
  // member is at least 1.5: z = 1, u = 2.25 again (cost 5.25), or infeasible without z: the row forces it.
  const Model m2 = read_lp_string(
      "min\n u + 3 z\nst\n a: m1 + x = 2\n b: 2 m2 - u = 0\n c: m3 - z = 0\n k: [ m1^2 - 2 m2 * m3 ] <= 0\n"
      "bounds\n m1 free\n -10 <= x <= 0.5\n 0 <= u <= 100\nbinary\n z\nend\n");
  Model pm2;
  const QuadPresolve ps2 = quad_presolve(m2, pm2);
  CHECK(ps2.aliases.size() == 2);  // not m1: a shifted member would change how the cone's violation is measured
  for (const bool det : {false, true}) {
    const SolveResult r = solve_model(m2, det);
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, 5.25, 1e-6);
    REQUIRE(r.has_solution);
    CHECK(m2.violation(r.x).max() <= 1e-5);
  }
}

TEST(quad_presolve_perspective) {
  // x^2 <= t with x <= 2 y, x >= 0, y binary: min t - 3 x + 2 y. y = 1: x = 1.5, -0.25; y = 0: 0.
  const std::string head = "min\n t - 3 x + 2 y\nst\n v: x - 2 y <= 0\n";
  const std::string tail = "bounds\n 0 <= x <= 5\n 0 <= t <= 100\nbinary\n y\nend\n";
  const Model m = read_lp_string(head + " k: - t + [ x^2 ] <= 0\n" + tail);
  Model pm;
  const QuadPresolve ps = quad_presolve(m, pm);
  CHECK(ps.perspective == 1);
  // The row is x^2 - t y <= 0 now, without a linear part.
  CHECK(pm.qr_val.size() == 2);
  const Index k = [&] {
    for (Index i = 0; i < m.num_rows(); ++i) {
      if (m.row_names[i] == "k") return i;
    }
    return Index{-1};
  }();
  for (Index j = 0; j < pm.num_cols(); ++j) {
    for (Index e = pm.A.begin(j); e < pm.A.end(j); ++e) CHECK(pm.A.index[e] != k || pm.A.value[e] == 0.0);
  }
  for (const bool det : {false, true}) {
    const SolveResult r = solve_model(m, det);
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -0.25, 1e-6);
    REQUIRE(r.has_solution);
    CHECK(m.violation(r.x).max() <= 1e-5);
  }
  // Written with a lower side: t - x^2 >= 0.
  Model pg;
  CHECK(quad_presolve(read_lp_string(head + " k: t - [ x^2 ] >= 0\n" + tail), pg).perspective == 1);
  // Not applied: a constant in the row; x may be negative; t may be negative; no switching row.
  Model po;
  CHECK(quad_presolve(read_lp_string(head + " k: - t + [ x^2 ] <= 1\n" + tail), po).perspective == 0);
  CHECK(quad_presolve(read_lp_string(head + " k: - t + [ x^2 ] <= 0\nbounds\n -1 <= x <= 5\n 0 <= t <= 100\nbinary\n y\nend\n"), po)
            .perspective == 0);
  CHECK(quad_presolve(read_lp_string(head + " k: - t + [ x^2 ] <= 0\nbounds\n 0 <= x <= 5\n -1 <= t <= 100\nbinary\n y\nend\n"), po)
            .perspective == 0);
  CHECK(quad_presolve(read_lp_string("min\n t - 3 x + 2 y\nst\n v: x - 2 y <= 1\n k: - t + [ x^2 ] <= 0\n" + tail), po)
            .perspective == 0);
  // With t < 0 allowed the models differ at y = 0, and the solve must not use the cone: min t - 3 x + 2 y + ...
  // stays at its own optimum (y = 1, x = 1.5: -0.25; y = 0 gives t >= 0 through the row either way).
  const SolveResult r = solve_model(
      read_lp_string(head + " k: - t + [ x^2 ] <= 0\nbounds\n 0 <= x <= 5\n -1 <= t <= 100\nbinary\n y\nend\n"), false);
  CHECK(r.status == Status::Optimal);
  CHECK_NEAR(r.objective, -0.25, 1e-6);
}

TEST(quad_presolve_fixed_members) {
  // x^2 <= 2 t h with h fixed at 1/2 is x^2 <= t; with the switch x <= 2 y it becomes the cone x^2 <= t y.
  const Model m = read_lp_string("min\n t - 3 x + 2 y\nst\n v: x - 2 y <= 0\n k: [ x^2 - 2 t * h ] <= 0\nbounds\n"
                                 " 0 <= x <= 5\n 0 <= t <= 100\n h = 0.5\nbinary\n y\nend\n");
  Model pm;
  const QuadPresolve ps = quad_presolve(m, pm);
  CHECK(ps.fixed_terms == 1);
  CHECK(ps.perspective == 1);
  for (const bool det : {false, true}) {
    const SolveResult r = solve_model(m, det);
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -0.25, 1e-6);
    REQUIRE(r.has_solution);
    CHECK(m.violation(r.x).max() <= 1e-5);
  }
}

TEST(quad_presolve_cone_switched_by_a_binary) {
  // x^2 <= u z with x free, u unbounded above: z = 0 forces x = 0 (an implication in the search; no big-M exists).
  // min u + 3 z - 4 w, w binary, x >= 1.5 w - 3 (1 - s) ... kept simple: x = 1.5 w. w = 1: z = 1, u = 2.25: -0.75;
  // w = 0: 0. A second cone with z2 = 0 at the optimum has its member forced to 0: x2 = 0 costs nothing.
  const Model m = read_lp_string("min\n u + 3 z - 6 w + u2 + z2 - 0.1 x2\nst\n d: x - 1.5 w = 0\n"
                                 " k: [ x^2 - u * z ] <= 0\n k2: [ x2^2 - u2 * z2 ] <= 0\nbounds\n x free\n x2 free\n"
                                 "binary\n z w z2\nend\n");
  for (const bool det : {false, true}) {
    const SolveResult r = solve_model(m, det);
    CHECK(r.status == Status::Optimal);
    // The second cone: z2 = 1 costs 1 and x2 = a needs u2 = a^2, gain 0.1 a - a^2 <= 0.0025: not worth it.
    CHECK_NEAR(r.objective, -0.75, 1e-6);
    REQUIRE(r.has_solution);
    CHECK(m.violation(r.x).max() <= 1e-5);
    CHECK_NEAR(r.x[col(m, "x2")], 0.0, 1e-5);
  }
}
