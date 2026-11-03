#pragma once
// Reformulation of the logical modelling features of a Model (semi-continuous
// and semi-integer columns, SOS1/SOS2 sets, indicator constraints) into a
// plain MILP with auxiliary binary columns, solved by the ordinary pipeline
// (presolve, cuts, heuristics, branch-and-bound).
//
// Design choice: everything becomes binaries and linear rows, so presolve,
// propagation, cutting planes, heuristics and branching all apply unchanged;
// branching on an SOS set or a semi-continuous disjunction is branching on
// its binaries, and row propagation does the SOS / indicator fixings.
//
//  - semi-continuous x in {0} u [l, u] (l > 0): binary z with
//        l z <= x <= U z            (U = u, or the bound implied by the rows)
//    (mirrored for u < 0; if 0 lies in [l, u] the column is ordinary);
//  - SOS1 {x_1..x_k}: binaries z_i,  L_i z_i <= x_i <= U_i z_i,  sum z_i <= 1
//    (members that are binaries serve as their own z_i; an SOS1 of binaries
//    is the single row sum x_i <= 1);
//  - SOS2 (members ordered by weight): segment binaries y_1..y_{k-1},
//    sum y <= 1, L_i (y_{i-1} + y_i) <= x_i <= U_i (y_{i-1} + y_i);
//  - indicator  b = v -> lo <= a x <= hi:  big-M rows with M from the
//    activity bounds of a x, e.g. a x <= hi + M (1 - b) for v = 1.
//
// Bounds L/U/M come from the column bounds tightened by activity-based bound
// propagation over the model rows. Where no finite bound exists (or it
// exceeds max_big_m), the reformulation keeps the disjunction exact instead:
// a binary z (or the indicator column) and a column c (the member itself, or
// a slack of the indicator row) with the implication "z = zval => c = 0"
// (MipProblem::implications). The search enforces implications by fixing c
// at nodes where z is fixed and by branching on z when an integral LP point
// violates one; presolve and symmetry handling are skipped for such models.
//
// Columns 0..n-1 of the result are the original columns (same order), so a
// solution of the reformulation restricted to them solves the original model.

#include <functional>
#include <string>
#include <vector>

#include "milp/core/model.hpp"
#include "milp/search/mip_problem.hpp"

namespace milp {

struct LogicalReformulation {
  Model model;  // plain MILP: no semi types, SOS or indicators
  std::vector<Implication> implications;
  Index orig_cols = 0;
  int aux_cols = 0, aux_rows = 0, big_m = 0;  // statistics (big_m: rows with a big-M coefficient)

  // Values of the auxiliary columns derived from a point of the original model (NaN = unknown,
  // also where the point violates a logical constraint). Returns a point of the reformulation.
  std::vector<double> extend(const std::vector<double>& x) const;

  std::vector<std::function<double(const std::vector<double>&)>> aux_value;  // one per auxiliary column
};

// Throws std::runtime_error if an indicator column is not binary.
LogicalReformulation reformulate_logical(const Model& m, double max_big_m = 1e6);

// Column bounds implied by the rows (a few rounds of activity-based tightening), starting from
// lo / hi (one entry per column). Valid for every feasible point of the rows.
void implied_bounds(const Model& m, std::vector<double>& lo, std::vector<double>& hi, int max_rounds = 20);

}  // namespace milp
