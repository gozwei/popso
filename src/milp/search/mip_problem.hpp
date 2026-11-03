#pragma once
// Internal MIP representation used by the search: always minimization,
// integer bounds already rounded.

#include <string>
#include <vector>

#include "milp/core/model.hpp"
#include "milp/core/sparse.hpp"
#include "milp/core/types.hpp"

namespace milp {

// Logical implication kept out of the LP: x[z] == zval forces x[col] == 0. Used for logical
// constraints that have no finite big-M reformulation (solver/reformulate.hpp); the search
// branches on z when an integral LP point violates one, and fixes col at nodes where z is fixed.
struct Implication {
  Index z = -1;
  double zval = 1.0;
  Index col = -1;
};

struct MipProblem {
  SparseMatrix A;  // CSC
  std::vector<double> cost, col_lb, col_ub, row_lb, row_ub;
  std::vector<char> is_int;
  double offset = 0.0;  // added to cost^T x
  double display_sense = 1.0;  // -1 if the user model maximizes (for logging only)
  std::vector<Implication> implications;  // usually empty; presolve and symmetry handling are off otherwise

  Index num_cols() const { return static_cast<Index>(cost.size()); }
  Index num_rows() const { return static_cast<Index>(row_lb.size()); }

  // Build from a user model; returns false if integer bounds are inconsistent.
  static bool from_model(const Model& m, MipProblem* out);

  double objective(const std::vector<double>& x) const;
  // Max violation of bounds, rows, integrality and implications (|x[col]| where x[z] rounds to zval).
  double max_violation(const std::vector<double>& x, double* int_viol = nullptr) const;
  // Fixes col at 0 in [lb, ub] for every implication whose z is fixed at zval there, and z at the
  // other value where col's domain excludes 0 (until nothing changes). Returns false if the bounds
  // become contradictory. Changed columns are appended to `changed` if given.
  bool apply_implications(std::vector<double>& lb, std::vector<double>& ub, std::vector<Index>* changed = nullptr) const;
};

}  // namespace milp
