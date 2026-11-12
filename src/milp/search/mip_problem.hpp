#pragma once
// Internal MIP representation used by the search: always minimization,
// integer bounds already rounded.

#include <string>
#include <utility>
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

// Convex quadratic row kept out of the LP (v0.8, outer approximation in the search): lo <= a x + sum_k c_k x_i x_j <= hi
// with one finite side, the quadratic part convex in its direction (PSD for a finite hi, NSD for a finite lo). The
// search represents it by gradient cuts (cuts/quad_oa.hpp) and checks every candidate point against it.
struct QuadRow {
  std::vector<Index> idx;  // linear part
  std::vector<double> val;
  std::vector<Index> qi, qj;  // monomials c x_i x_j (i >= j; no 1/2)
  std::vector<double> qc;
  double lo = -kInf, hi = kInf;
  Index epigraph = -1;  // >= 0: the objective's epigraph row 1/2 x' Q x - eta <= 0, eta this column
  double activity(const double* x) const;
  // Sum of the absolute values of the terms at x: the scale of the row's violation (separation and acceptance
  // tolerances are relative to max(1, it)). The epigraph row: |eta| instead, the objective's scale (its violation is
  // an objective error, which the node closing test measures against the objective; dense Q terms that cancel would
  // make the terms' sum far larger).
  double magnitude(const double* x) const;
  // The violation at x relative to the row's scale (<= 0: satisfied), as max_violation and the separation use it: a
  // cone row by its norm form g(x) (the squared form's tolerance lets a column reach sqrt(tol) at the cone's apex:
  // x^2 <= t^2 + 1e-9 at t = 0 allows |x| = 3e-5) relative to its constants only, max(1, |h0|, sqrt(gamma), |b|), as
  // a linear row is to its side (relative to the head h x + h0, a rotated cone x^2 <= u v written as
  // (u - v)^2 + 4 x^2 <= (u + v)^2 let x reach 1.5 at u = 0, v = 6575: CBLIB clay0304h 6% below its optimum); any
  // other row by the distance beyond its side relative to max(1, magnitude).
  double violation(const double* x) const;
  // Gradient at x as (column, value) pairs, summed per column (`grad` is a dense scratch of num_cols zeros, left zero).
  void gradient(const double* x, std::vector<double>& grad, std::vector<std::pair<Index, double>>& out) const;
  // Second-order cone form (solve_oa sets it for a row whose quadratic part is not convex but which is a cone over its
  // columns' bounds, e.g. x^2 + y^2 <= t^2 with t >= 0, or x^2 <= u v with u, v >= 0): the row holds exactly where
  //   g(x) = sqrt(cone_gamma + sum_m (cone_a[m] x + cone_b[m])^2) - (cone_h x + cone_h0) <= 0,
  // and g is convex, so its tangents are the cuts (those of the indefinite quadratic would not be valid).
  bool cone = false;
  std::vector<std::vector<std::pair<Index, double>>> cone_a;
  std::vector<double> cone_b;
  std::vector<std::pair<Index, double>> cone_h;
  double cone_gamma = 0.0, cone_h0 = 0.0;
  // A valid cut at x, sum coef x <= rhs: the tangent plane of the row's convex function (the row's own, or g of the cone
  // form). False where there is none (zero gradient, the cone's apex, values that are not finite).
  bool linearize(const double* x, std::vector<double>& grad, std::vector<std::pair<Index, double>>& coef,
                 double& rhs) const;
};

// A product kept exact by spatial branching (v0.10): w = x * y (x == y: a square). The LP holds McCormick rows of
// the product over the root's bounds as model rows; the search adds those of a node's bounds as rows of that node's
// LP and branches on x or y (continuous columns too) where an LP point still violates the product.
struct Bilinear {
  Index w = -1, x = -1, y = -1;
  // An integer factor counts at its rounded value: a binary at 1e-8 (integral within the tolerance) times a factor
  // at 5e8 is 0, not 5, and a point that needs the 5 in some row is no solution.
  bool x_int = false, y_int = false;
  // |w - x y| relative to max(1, |x y|).
  double violation(const double* v) const;
};

struct MipProblem {
  SparseMatrix A;  // CSC
  std::vector<double> cost, col_lb, col_ub, row_lb, row_ub;
  std::vector<char> is_int;
  // Implied integers (set by presolve, presolve/implied_int.hpp): continuous columns that are integral in
  // every feasible point with integral integers (primal rule), or that can be moved to an integral value in
  // any feasible point without loss (dual rule). Empty, or one flag per column (columns beyond its size have
  // none). Never branched on and not required for feasibility; used for objective integrality and treated
  // as integers by c-MIR / Gomory cuts. Their bounds are integral (or infinite) when presolve sets them.
  std::vector<char> implied_int;
  bool is_implied_int(Index j) const {
    return static_cast<size_t>(j) < implied_int.size() && implied_int[static_cast<size_t>(j)] != 0;
  }
  double offset = 0.0;  // added to cost^T x
  double display_sense = 1.0;  // -1 if the user model maximizes (for logging only)
  std::vector<Implication> implications;  // usually empty; presolve and symmetry handling are off otherwise
  // Convex quadratic rows (outer approximation; usually empty): presolve, symmetry handling and restarts are off
  // otherwise, and max_violation counts them.
  std::vector<QuadRow> quad_rows;
  std::vector<Bilinear> bilinear;  // products enforced by spatial branching; usually empty
  // Rows that hold products as columns (a nonconvex quadratic row with each c x_i x_j replaced by c w): max_violation
  // judges them with the products recomputed from the factors, activity + sum c (x_i x_j - w), so the errors the
  // single products are allowed cannot add up past the row's tolerance (QPLIB_3816: 1.003e-6 on a row).
  struct LiftedRow {
    Index row = -1;
    std::vector<std::pair<Index, double>> terms;  // (index into bilinear, coefficient)
  };
  std::vector<LiftedRow> lifted_rows;
  // Rows outside the LP or products: presolve, symmetry handling, restarts and conflict analysis are off.
  bool nonlinear() const { return !quad_rows.empty() || !bilinear.empty(); }
  // Quadratic objective 1/2 x^T Q x (lower triangle, CSC; sign-adjusted for minimization). Empty
  // (no entries) for a MILP; branch_and_bound() and presolve only handle that case (MIQPs are
  // solved by miqp_solve, see qp/miqp.hpp).
  SparseMatrix Q;
  bool quadratic() const { return Q.nnz() > 0; }

  Index num_cols() const { return static_cast<Index>(cost.size()); }
  Index num_rows() const { return static_cast<Index>(row_lb.size()); }

  // Build from a user model; returns false if integer bounds are inconsistent.
  static bool from_model(const Model& m, MipProblem* out);

  double objective(const std::vector<double>& x) const;  // offset + c^T x (+ 1/2 x^T Q x)
  // Max violation of bounds, rows, quadratic rows, integrality and implications (|x[col]| where x[z] rounds to zval).
  // lin_viol: the same without the quadratic rows.
  double max_violation(const std::vector<double>& x, double* int_viol = nullptr, double* lin_viol = nullptr) const;
  // Fixes col at 0 in [lb, ub] for every implication whose z is fixed at zval there, and z at the
  // other value where col's domain excludes 0 (until nothing changes). Returns false if the bounds
  // become contradictory. Changed columns are appended to `changed` if given.
  bool apply_implications(std::vector<double>& lb, std::vector<double>& ub, std::vector<Index>* changed = nullptr) const;
};

}  // namespace milp
