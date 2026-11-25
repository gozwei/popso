#pragma once
// The user-facing MILP / (MI)QP model:
//
//   min/max  c^T x + 1/2 x^T Q x + offset
//   s.t.     row_lb <= A x <= row_ub
//            col_lb <=   x <= col_ub
//            x_j integer for j in integer columns
//            logical constraints (below)
//
// Q is symmetric and stored as its lower triangle (q_row >= q_col) in triplet
// form; duplicate entries are summed (q_lower() assembles the matrix). An
// off-diagonal entry (i, j) stands for both Q_ij and Q_ji, so it contributes
// Q_ij x_i x_j to the objective, a diagonal entry 1/2 Q_jj x_j^2. The solver
// requires a convex objective (Q positive semidefinite when minimizing,
// negative semidefinite when maximizing).
//
// Rows are ranged; equality rows have row_lb == row_ub, free rows (other
// than the objective) are allowed. The constraint matrix is stored by column
// (CSC). Use ModelBuilder to assemble a model incrementally.
//
// Logical modelling features (the solver reformulates them into plain MILP
// rows and auxiliary binaries, see solver/reformulate.hpp):
//  - semi-continuous / semi-integer columns (VarType::SemiContinuous /
//    SemiInteger): x = 0 or col_lb <= x <= col_ub (integral for SemiInteger);
//  - SOS1 / SOS2 sets: at most one member / at most two members that are
//    adjacent in the order of their weights may be nonzero;
//  - indicator constraints: if column `binary` equals `value` (0 or 1), then
//    lb <= sum coefs * x[cols] <= ub must hold.

#include <string>
#include <vector>

#include "milp/core/sparse.hpp"
#include "milp/core/types.hpp"

namespace milp {

struct SosConstraint {
  std::string name;
  int type = 1;                // 1 or 2
  std::vector<Index> cols;     // members
  std::vector<double> weights;  // one per member; they order the set (SOS2 adjacency)
};

struct IndicatorConstraint {
  std::string name;
  Index binary = -1;  // indicator column (binary)
  int value = 1;      // the row is enforced when x[binary] == value
  std::vector<Index> cols;
  std::vector<double> coefs;
  double lb = -kInf, ub = kInf;
};

// A further linear objective of a multi-objective model (Model::objectives). The first objective is the
// model's own (Model::obj, obj_offset) with the attributes Model::obj_priority etc.
//
// Objectives are optimized in the model's sense (a negative weight turns one around), in stages by descending
// priority. Objectives of one priority form one stage and are blended: the stage optimizes the sum of
// weight * objective. A later stage may not make an earlier stage's blended value worse than the value found
// by more than max(abs_tol, rel_tol * |value|), with abs_tol / rel_tol the largest of that stage's members.
struct Objective {
  std::string name;
  std::vector<Index> cols;
  std::vector<double> coefs;  // one per entry of cols; duplicates are summed
  double offset = 0.0;
  int priority = 0;
  double weight = 1.0;
  double abs_tol = 1e-6, rel_tol = 0.0;
};

struct Model {
  std::string name;
  ObjSense sense = ObjSense::Minimize;
  double obj_offset = 0.0;

  // Multiple objectives: the further ones, and the attributes of the first (obj, obj_offset). Empty: an
  // ordinary model. Linear objectives only (no Q with further objectives).
  std::vector<Objective> objectives;
  std::string obj_name;
  int obj_priority = 0;
  double obj_weight = 1.0;
  double obj_abs_tol = 1e-6, obj_rel_tol = 0.0;

  // Columns.
  std::vector<std::string> col_names;
  std::vector<double> obj;
  std::vector<double> col_lb;
  std::vector<double> col_ub;
  std::vector<VarType> col_type;

  // Rows.
  std::vector<std::string> row_names;
  std::vector<double> row_lb;
  std::vector<double> row_ub;

  SparseMatrix A;  // CSC, num_major = num_cols, num_minor = num_rows

  // Quadratic objective (lower triangle triplets, see above). Empty: a linear objective.
  std::vector<Index> q_row, q_col;
  std::vector<double> q_val;

  // Quadratic constraint terms: entry k adds qr_val[k] * x[qr_i[k]] * x[qr_j[k]] to the activity of row qr_row[k]
  // (a monomial coefficient, no 1/2 factor: as in LP-format constraint brackets and MPS QCMATRIX, whose
  // symmetric listing gives both (i, j) and (j, i)). qr_i >= qr_j; duplicates are summed. Empty: linear rows.
  std::vector<Index> qr_row, qr_i, qr_j;
  std::vector<double> qr_val;

  // Logical constraints.
  std::vector<SosConstraint> sos;
  std::vector<IndicatorConstraint> indicators;

  Index num_cols() const { return static_cast<Index>(obj.size()); }
  Index num_rows() const { return static_cast<Index>(row_lb.size()); }
  Index num_nonzeros() const { return A.nnz(); }
  Index num_integers() const;
  Index num_semi() const;  // semi-continuous + semi-integer columns
  // Semi-continuous / semi-integer columns, SOS sets or indicator constraints present.
  bool has_logical() const { return !sos.empty() || !indicators.empty() || num_semi() > 0; }
  bool is_mip() const { return num_integers() > 0 || has_logical(); }
  bool has_quadratic() const;  // some nonzero Q entry
  bool has_quadratic_rows() const;  // some nonzero quadratic constraint term
  // Adds v * x_i * x_j to row `row`'s activity (stored with i >= j).
  void add_qrow(Index row, Index i, Index j, double v);
  // Merges duplicate quadratic constraint terms (summed, in their input order) and drops zeros; the terms end sorted
  // by (row, i, j). The readers call it (MPS QCMATRIX lists every off-diagonal pair twice).
  void compact_qrows();
  // Row activities of x (linear and quadratic terms).
  std::vector<double> row_activity(const std::vector<double>& x) const;

  // Adds v to Q_ij (and Q_ji): stored at the lower triangle position.
  void add_q(Index i, Index j, double v);
  // Q's lower triangle as a CSC num_cols x num_cols matrix (duplicates summed, zeros dropped).
  SparseMatrix q_lower() const;

  // Objective value of x in the original sense (including offset and the quadratic term).
  double objective_value(const std::vector<double>& x) const;
  // Multiple objectives: their number (1 + objectives.size()) and the value of objective k at x (k = 0: the
  // model's own, as objective_value; k >= 1: objectives[k - 1], with its offset, without its weight).
  int num_objectives() const { return 1 + static_cast<int>(objectives.size()); }
  double objective_value(int k, const std::vector<double>& x) const;

  // Maximum absolute violation of bounds, rows, integrality and logical
  // constraints (semi-continuous domains, SOS sets, active indicator rows;
  // a member counts as nonzero above 1e-6, an indicator column is rounded).
  struct Violation {
    double bound = 0.0, row = 0.0, integrality = 0.0, logical = 0.0;
    double max() const;
  };
  Violation violation(const std::vector<double>& x) const;

  // Throws std::runtime_error on inconsistent dimensions / NaNs / bad
  // logical constraints (indices, sizes, SOS type, indicator value).
  void validate() const;

  // Removes the rows with del[i] != 0 (row data and matrix entries).
  void remove_rows(const std::vector<char>& del);
};

// Incremental builder; rows and columns may be added in any order, nonzeros
// are collected as triplets and assembled by build().
class ModelBuilder {
 public:
  Index add_col(std::string name, double obj, double lb, double ub, VarType type = VarType::Continuous);
  Index add_row(std::string name, double lb, double ub);
  void add_coef(Index row, Index col, double value);
  Model& model() { return m_; }
  Model build();

 private:
  Model m_;
  std::vector<Index> trip_col_, trip_row_;
  std::vector<double> trip_val_;
};

}  // namespace milp
