#pragma once
// A light presolve for mixed-integer models with quadratic rows, run before the convexity check and the cone
// detection of the outer approximation (solver.cpp, solve_oa). The model keeps its columns and rows, so a point
// of the presolved model is a point of the model (after restore(), below) and a start needs no mapping; only
// the quadratic rows, their linear parts and sides, and column bounds change:
//
//  1. Bounds of the columns in quadratic rows, tightened by activity propagation over the linear rows (the cone
//     detection needs the signs of the head columns; tangents and big-M values need finite ranges).
//  2. Alias members: a continuous column c of a quadratic row whose only linear row is an equality with one other
//     column, v c + a x = b, is replaced by s x + o in the quadratic rows (s = -a / v, o = b / v), and its bounds
//     pass to x. The column and its equality stay in the model (c = s x + o still holds), outside the quadratic
//     rows. Conic benchmark format conversions define every cone member this way (CBLIB b1bigflowc, achtziger
//     *flowc: x1551^2 <= 2 x1550 x1549 with x1549 equal to a binary and x1551 to a flow): the search then sees
//     cones over the binaries and flows themselves, where it saw free columns tied to them by rows.
//  3. Fixed columns in quadratic terms become linear terms or constants (uflquad-nopsc: x^2 <= 2 t h, h = 1/2).
//  4. Perspective strengthening: sum_k q_k x_k^2 <= c t (q_k, c > 0, t >= 0, no constant) where one binary y
//     switches every x_k off (x_k >= 0 and a row x_k <= u y) becomes the rotated cone sum_k q_k x_k^2 <= c t y. The
//     two agree at y = 1; at y = 0 every x_k is 0 and both hold for every t >= 0; in between the cone is tighter.
//     (Günlük, Linderoth: perspective reformulations of MINLPs with indicator variables, 2010.)

#include <vector>

#include "milp/core/model.hpp"

namespace milp {

struct QuadPresolve {
  struct Alias {
    Index col, of;
    double scale, offset;  // col = scale * of + offset
  };
  std::vector<Alias> aliases;
  int bounds = 0, fixed_terms = 0, perspective = 0;

  bool changed() const { return bounds + fixed_terms + perspective > 0 || !aliases.empty(); }
  // The alias columns of x recomputed from their definitions (exactly, so the model's own quadratic rows hold
  // as the presolved ones do).
  void restore(std::vector<double>& x) const;
};

// Fills `out` with the presolved copy of `m` (which has integer columns, quadratic rows and no logical
// constraints) and returns what was done.
QuadPresolve quad_presolve(const Model& m, Model& out);

}  // namespace milp
