// Spatial branch-and-bound for nonconvex quadratic models (solver.cpp solve_spatial, MipProblem::bilinear): products
// lifted to columns with McCormick rows, node-local rows and branching on the factors. Known optima, both search
// modes.

#include <cmath>
#include <string>
#include <vector>

#include "milp/io/lp_reader.hpp"
#include "milp/solver/solver.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

SolveResult solve_text(const std::string& text, bool det) {
  SolverParams p;
  p.verbosity = 0;
  p.threads = det ? 2 : 1;
  p.deterministic = det;
  p.mip_rel_gap = 1e-7;
  p.time_limit = 60.0;
  return solve(read_lp_string(text), p);
}

}  // namespace

TEST(spatial_known_optima) {
  for (const bool det : {false, true}) {
    // max x y with x + y <= 4 on [0, 3]^2: 4 at (2, 2).
    const SolveResult a = solve_text("max\n obj: [ 2 x * y ] / 2\nst\n c1: x + y <= 4\nbounds\n 0 <= x <= 3\n 0 <= y <= 3\nend\n", det);
    CHECK(a.status == Status::Optimal);
    CHECK_NEAR(a.objective, 4.0, 1e-5);
    // min -x - y with x y <= 1 on [0, 4]^2: -4.25 at (4, 1/4) or (1/4, 4).
    const SolveResult b = solve_text("min\n obj: - x - y\nst\n c1: [ x * y ] <= 1\nbounds\n 0 <= x <= 4\n 0 <= y <= 4\nend\n", det);
    CHECK(b.status == Status::Optimal);
    CHECK_NEAR(b.objective, -4.25, 1e-5);
    CHECK_NEAR(b.x[0] * b.x[1], 1.0, 1e-5);
    // An indefinite objective and a nonconvex row: min x^2 - y^2 + x y + z; y = 2, x = -1, z = 0: -5.
    const SolveResult c = solve_text("min\n obj: [ 2 x ^2 - 2 y ^2 + 2 x * y ] / 2 + z\nst\n c1: x + y + z >= 1\n"
                                     " c2: [ x ^2 + y ^2 - z ^2 ] >= 0.5\nbounds\n -2 <= x <= 2\n -2 <= y <= 2\n 0 <= z <= 3\nend\n", det);
    CHECK(c.status == Status::Optimal);
    CHECK_NEAR(c.objective, -5.0, 1e-5);
    // A nonconvex equality with an integer factor: x y = 6, x integer in [1, 4], y in [0, 5]: min x + y = 5 at
    // (2, 3) and (3, 2).
    const SolveResult d = solve_text("min\n obj: x + y\nst\n c1: [ x * y ] = 6\nbounds\n 1 <= x <= 4\n 0 <= y <= 5\ngeneral\n x\nend\n", det);
    CHECK(d.status == Status::Optimal);
    CHECK_NEAR(d.objective, 5.0, 1e-5);
    CHECK_NEAR(d.x[0], std::round(d.x[0]), 1e-9);
    // A convex row beside a nonconvex one stays a tangent-cut row: x^2 + y^2 <= 4 and x y >= 1, max x: the
    // intersection of the circle and the hyperbola, x^2 = 2 + sqrt(3).
    const SolveResult e = solve_text("max\n obj: x\nst\n c1: [ x ^2 + y ^2 ] <= 4\n c2: [ x * y ] >= 1\nbounds\n 0 <= x <= 3\n 0 <= y <= 3\nend\n", det);
    CHECK(e.status == Status::Optimal);
    CHECK_NEAR(e.objective, std::sqrt(2.0 + std::sqrt(3.0)), 1e-5);
  }
}

TEST(spatial_infeasible_and_unbounded_factors) {
  for (const bool det : {false, true}) {
    // x y >= 5 on [0, 2]^2: infeasible (the product is at most 4).
    CHECK(solve_text("min\n obj: x\nst\n c1: [ x * y ] >= 5\nbounds\n 0 <= x <= 2\n 0 <= y <= 2\nend\n", det).status == Status::Infeasible);
    // Infeasible only through the product: x + y = 1 and x y = 1 (the product is at most 1/4 on the line).
    CHECK(solve_text("min\n obj: x\nst\n c0: x + y = 1\n c1: [ x * y ] = 1\nbounds\n -3 <= x <= 3\n -3 <= y <= 3\nend\n", det).status ==
          Status::Infeasible);
    // A free factor whose bounds the linear rows imply: 0 <= x <= 2 from the rows, y in [0, 2]: max x y = 4.
    const SolveResult a = solve_text("max\n obj: [ 2 x * y ] / 2\nst\n c1: x <= 2\n c2: x >= 0\nbounds\n x free\n 0 <= y <= 2\nend\n", det);
    CHECK(a.status == Status::Optimal);
    CHECK_NEAR(a.objective, 4.0, 1e-5);
    // Free factors bounded only by a convex quadratic row (the circle of radius 2): max x y = 2 at (sqrt 2, sqrt 2),
    // and min z with z >= x y: -2. Their LP relaxation alone is unbounded.
    const SolveResult c = solve_text("max\n obj: [ 2 x * y ] / 2\nst\n c1: [ x ^2 + y ^2 ] <= 4\nbounds\n x free\n y free\nend\n", det);
    CHECK(c.status == Status::Optimal);
    CHECK_NEAR(c.objective, 2.0, 1e-5);
    const SolveResult e = solve_text("min\n obj: z\nst\n c1: z - [ x * y ] >= 0\n c2: [ x ^2 + y ^2 ] <= 4\nbounds\n x free\n y free\n z free\nend\n", det);
    CHECK(e.status == Status::Optimal);
    CHECK_NEAR(e.objective, -2.0, 1e-5);
    // Factors without any bound in a model that is unbounded: no optimum is claimed (2 s limit).
    {
      SolverParams p;
      p.verbosity = 0;
      p.threads = det ? 2 : 1;
      p.deterministic = det;
      p.time_limit = 2.0;
      const SolveResult u = solve(read_lp_string("max\n obj: [ 2 x * y ] / 2\nst\n c1: x + y <= 2\nbounds\n x free\n y free\nend\n"), p);
      CHECK(u.status != Status::Optimal && u.status != Status::Infeasible);
    }
  }
}
