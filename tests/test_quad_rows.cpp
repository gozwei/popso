// Convex quadratic constraints solved by outer approximation in the MILP search (solver.cpp solve_oa,
// cuts/quad_oa.cpp): known optima, infeasibility, convexity detection, binary products linearized, the MPS round trip,
// both search modes.

#include <cmath>
#include <string>
#include <vector>

#include "milp/io/lp_reader.hpp"
#include "milp/io/mps_reader.hpp"
#include "milp/solver/solver.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

SolveResult solve_lp_text(const std::string& text, bool det = false) {
  SolverParams p;
  p.verbosity = 0;
  p.threads = det ? 2 : 1;
  p.deterministic = det;
  p.mip_rel_gap = 1e-9;
  p.mip_abs_gap = 1e-9;
  return solve(read_lp_string(text), p);
}

}  // namespace

TEST(quad_rows_known_optima) {
  // max x + y s.t. x + y + (x + y)^2 + 2 y^2 <= 4: at y = 0, x + x^2 = 4, x = (sqrt(17) - 1) / 2.
  const std::string qc = "min\n - x - y\nst\n c: x + y + [ x^2 + 2 x * y + 3 y^2 ] <= 4\n c2: x - y >= -1\n"
                         "bounds\n 0 <= x <= 5\n 0 <= y <= 5\nend\n";
  // max x + y with x integer, x^2 + y^2 <= 10.5: x = 2, y = sqrt(6.5).
  const std::string mi = "min\n - x - y\nst\n c: [ x^2 + y^2 ] <= 10.5\nbounds\n -5 <= x <= 5\n -5 <= y <= 5\n"
                         "general\n x\nend\n";
  for (const bool det : {false, true}) {
    const SolveResult a = solve_lp_text(qc, det);
    CHECK(a.status == Status::Optimal);
    CHECK_NEAR(a.objective, (1.0 - std::sqrt(17.0)) / 2.0, 1e-6);
    const SolveResult b = solve_lp_text(mi, det);
    CHECK(b.status == Status::Optimal);
    CHECK_NEAR(b.objective, -2.0 - std::sqrt(6.5), 1e-6);
    REQUIRE(b.has_solution);
    CHECK_NEAR(b.x[0], 2.0, 1e-9);
    CHECK(b.bound <= b.objective + 1e-9);
  }
  // A lower side with a concave quadratic part: x^2 - ... >= reads as - x^2 - y^2 >= -10.5.
  const SolveResult c = solve_lp_text("min\n - x - y\nst\n c: - [ x^2 + y^2 ] >= -10.5\nbounds\n -5 <= x <= 5\n"
                                      " -5 <= y <= 5\ngeneral\n x\nend\n");
  CHECK(c.status == Status::Optimal);
  CHECK_NEAR(c.objective, -2.0 - std::sqrt(6.5), 1e-6);
}

TEST(quad_rows_second_order_cones) {
  // Rows whose quadratic part is indefinite but which are cones over the bounds (QuadRow::cone, solve_oa detect_cone).
  const std::string bounds = "bounds\n -10 <= x <= 10\n -10 <= y <= 10\n 0 <= t <= 100\n";
  // min t s.t. ||(x, y)|| <= t, x + y >= 2: sqrt(2); written both ways.
  const std::string soc = "min\n t\nst\n c: [ x^2 + y^2 - t^2 ] <= 0\n d: x + y >= 2\n" + bounds + "end\n";
  const std::string soc_ge = "min\n t\nst\n c: [ t^2 - x^2 - y^2 ] >= 0\n d: x + y >= 2\n" + bounds + "end\n";
  // One sheet of a hyperboloid, with linear terms: (x - 1)^2 + 1 <= (t + 2)^2, t >= -2 -> x^2 - 2x - t^2 - 4t <= 2:
  // min t - 0.5 x: x - 1 = 1/sqrt(3), t + 2 = 2/sqrt(3).
  const std::string hyp = "min\n t - 0.5 x\nst\n c: - 2 x - 4 t + [ x^2 - t^2 ] <= 2\nbounds\n -10 <= x <= 10\n"
                          " -2 <= t <= 100\nend\n";
  // Rotated, u integer: x^2 <= u v, x >= 2.5: min u + v at u = 3, v = 6.25 / 3.
  const std::string rot = "min\n u + v\nst\n c: [ x^2 - u * v ] <= 0\nbounds\n 2.5 <= x <= 10\n 0 <= u <= 100\n"
                          " 0 <= v <= 100\ngeneral\n u\nend\n";
  for (const bool det : {false, true}) {
    for (const std::string* text : {&soc, &soc_ge}) {
      const SolveResult a = solve_lp_text(*text, det);
      CHECK(a.status == Status::Optimal);
      CHECK_NEAR(a.objective, std::sqrt(2.0), 1e-6);
    }
    const SolveResult b = solve_lp_text(hyp, det);
    CHECK(b.status == Status::Optimal);
    CHECK_NEAR(b.objective, 2.0 / std::sqrt(3.0) - 2.0 - 0.5 * (1.0 + 1.0 / std::sqrt(3.0)), 1e-6);
    const SolveResult c = solve_lp_text(rot, det);
    CHECK(c.status == Status::Optimal);
    CHECK_NEAR(c.objective, 3.0 + 6.25 / 3.0, 1e-6);
    REQUIRE(c.has_solution);
    CHECK_NEAR(c.x[0], 3.0, 1e-9);  // u (columns in order of appearance: u, v, x)
  }
  // Not cones (t may be negative; the product's columns may be negative; two negative squares; a positive right
  // side: x^2 - t^2 <= 1 has two components even with t >= 0): not convex, solved by spatial branch-and-bound
  // (v0.10) since every column is bounded.
  {
    const SolveResult a = solve_lp_text("min\n t\nst\n c: [ x^2 + y^2 - t^2 ] <= 0\n d: x + y >= 2\nbounds\n -10 <= x <= 10\n"
                                        " -10 <= y <= 10\n -1 <= t <= 100\nend\n");
    CHECK(a.status == Status::Optimal);
    CHECK_NEAR(a.objective, std::sqrt(2.0), 1e-5);  // t <= -sqrt(2) is outside t >= -1
    const SolveResult b = solve_lp_text("min\n u + v\nst\n c: [ x^2 - u * v ] <= 0\nbounds\n 2.5 <= x <= 10\n -1 <= u <= 100\n"
                                        " 0 <= v <= 100\nend\n");
    CHECK(b.status == Status::Optimal);
    CHECK_NEAR(b.objective, 5.0, 1e-5);
    const SolveResult c = solve_lp_text("min\n t\nst\n c: [ x^2 - y^2 - t^2 ] <= 0\n" + bounds + "end\n");
    CHECK(c.status == Status::Optimal);
    CHECK_NEAR(c.objective, 0.0, 1e-5);
    const SolveResult d = solve_lp_text("min\n t\nst\n c: [ x^2 - t^2 ] <= 1\n" + bounds + "end\n");
    CHECK(d.status == Status::Optimal);
    CHECK_NEAR(d.objective, 0.0, 1e-5);
  }
  // The cone's tolerance is absolute in its norm form, not relative to the head: x^2 <= u v at u = 0 forces x = 0
  // (relative to the head (u + v) / 2 = 5000, x could reach 7 at v = 10000: objective -6).
  for (const bool det : {false, true}) {
    const SolveResult r = solve_lp_text("min\n - x + 0.0001 v + 20 u\nst\n c: [ 4 x^2 - 4 u * v ] <= 0\nbounds\n"
                                        " 0 <= x <= 10\n 0 <= v <= 10000\nbinary\n u\nend\n", det);
    CHECK(r.status == Status::Optimal);
    CHECK(r.objective >= -1e-3);
  }
}

TEST(quad_rows_continuous_duals) {
  // min t + 0.5 x s.t. ||(x - 2, y)|| <= t (a cone row), x + y >= 3, y <= 0.5: t = 1/sqrt(2) at (2.5, 0.5). Duals
  // from the conic method: cone row -1/sqrt(2) (upper side, minimizing), the linear row 0.5 + 1/sqrt(2), reduced costs
  // d = c - A'y - y_c grad q: 0 for t and x, -0.5 for y at its upper bound.
  const SolveResult r = solve_lp_text("min\n t + 0.5 x\nst\n c: - 4 x + [ x^2 + y^2 - t^2 ] <= -4\n lin: x + y >= 3\n"
                                      "bounds\n -10 <= x <= 10\n -10 <= y <= 0.5\n 0 <= t <= 100\nend\n");
  REQUIRE(r.status == Status::Optimal);
  REQUIRE(r.has_duals);
  const double s2 = std::sqrt(0.5);
  CHECK_NEAR(r.objective, s2 + 1.25, 1e-7);
  CHECK_NEAR(r.row_dual[0], -s2, 1e-5);
  CHECK_NEAR(r.row_dual[1], 0.5 + s2, 1e-5);
  CHECK_NEAR(r.reduced_cost[0], 0.0, 1e-5);   // t (columns in order of appearance: t, x, y)
  CHECK_NEAR(r.reduced_cost[1], 0.0, 1e-5);   // x
  CHECK_NEAR(r.reduced_cost[2], -0.5, 1e-5);  // y
}

TEST(quad_rows_infeasible_and_nonconvex) {
  const SolveResult a = solve_lp_text("min\n x\nst\n c: [ x^2 + y^2 ] <= -1\nbounds\n -5 <= x <= 5\n -5 <= y <= 5\nend\n");
  CHECK(a.status == Status::Infeasible);
  // x^2 + y^2 >= 1 is not convex (a convex function bounded from below).
  const SolveResult b = solve_lp_text("min\n x\nst\n c: [ x^2 + y^2 ] >= 1\nbounds\n -5 <= x <= 5\n -5 <= y <= 5\nend\n");
  // Solved by spatial branch-and-bound (v0.10; every column bounded): x = -5 is feasible in both.
  CHECK(b.status == Status::Optimal);
  CHECK_NEAR(b.objective, -5.0, 1e-5);
  const SolveResult c = solve_lp_text("min\n x\nst\n c: [ x * y ] <= 1\nbounds\n -5 <= x <= 5\n -5 <= y <= 5\nend\n");
  CHECK(c.status == Status::Optimal);
  CHECK_NEAR(c.objective, -5.0, 1e-5);
}

TEST(quad_rows_binary_products_linearized) {
  // Rows whose quadratic terms are products of binaries (or involve a fixed column) are linearized exactly, convex or
  // not: at most one of x, y, z (c), at least one (c2), and 3 x <= 2 through the fixed f = 2 (c3): y = 1, -2.
  const std::string text = "min\n - 3 x - 2 y - z\nst\n c: [ x * y + y * z + x * z ] <= 0\n"
                           " c2: [ x^2 + y^2 + z^2 ] >= 1\n c3: x + [ f * x ] <= 2\nbounds\n f = 2\n"
                           "binary\n x y z\nend\n";
  for (const bool det : {false, true}) {
    const SolveResult r = solve_lp_text(text, det);
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -2.0, 1e-9);
    CHECK(r.x.size() == 4);
    if (r.x.size() == 4) CHECK_NEAR(r.x[1], 1.0, 1e-9);
  }
  // A binary times a bounded column (nonconvex z (1 - x) <= 0): x = 0 forces z <= 0, so x = 1, z = 3: -1.
  const std::string bz = "min\n - z + 2 x\nst\n c: z - [ x * z ] <= 0\nbounds\n -1 <= z <= 3\nbinary\n x\nend\n";
  for (const bool det : {false, true}) {
    const SolveResult r = solve_lp_text(bz, det);
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -1.0, 1e-9);
    CHECK(r.x.size() == 2);
  }
}

TEST(quad_rows_nonconvex_binary_objective_linearized) {
  // min -x - y - z - x^2 + 2 (x y + y z + x z) over binaries (indefinite): x alone, -2; maximizing the negation, 2.
  const std::string text = "min\n - x - y - z + [ - 2 x ^ 2 + 4 x * y + 4 y * z + 4 x * z ] / 2\nst\n c: x + y + z >= 1\n"
                           "binary\n x y z\nend\n";
  std::string mx = "max\n x + y + z + [ 2 x ^ 2 - 4 x * y - 4 y * z - 4 x * z ] / 2\nst\n c: x + y + z >= 1\n"
                   "binary\n x y z\nend\n";
  for (const bool det : {false, true}) {
    const SolveResult r = solve_lp_text(text, det);
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -2.0, 1e-9);
    CHECK(r.x.size() == 3);
    const SolveResult q = solve_lp_text(mx, det);
    CHECK(q.status == Status::Optimal);
    CHECK_NEAR(q.objective, 2.0, 1e-9);
  }
}

TEST(quad_rows_mps_round_trip_solves_the_same) {
  const Model m = read_lp_string("min\n - 2 x - 3 y + z\nst\n c: x + [ x^2 + 2 x * y + 2 y^2 + z^2 ] <= 6\n"
                                 " d: x + y + z <= 3\nbounds\n 0 <= x <= 4\n -2 <= y <= 4\n 0 <= z <= 1\n"
                                 "general\n x\nend\n");
  SolverParams p;
  p.verbosity = 0;
  p.threads = 1;
  p.mip_rel_gap = 1e-9;
  p.mip_abs_gap = 1e-9;
  const SolveResult a = solve(m, p);
  const SolveResult b = solve(read_mps_string(write_mps_string(m)), p);
  CHECK(a.status == Status::Optimal);
  CHECK(b.status == Status::Optimal);
  CHECK_NEAR(a.objective, b.objective, 1e-7);
  REQUIRE(a.has_solution);
  CHECK(m.violation(a.x).max() <= 1e-6);
}
