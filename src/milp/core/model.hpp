#pragma once
// The user-facing MILP model:
//
//   min/max  c^T x + offset
//   s.t.     row_lb <= A x <= row_ub
//            col_lb <=   x <= col_ub
//            x_j integer for j in integer columns
//            logical constraints (below)
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

struct Model {
  std::string name;
  ObjSense sense = ObjSense::Minimize;
  double obj_offset = 0.0;

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

  // Objective value of x in the original sense (including offset).
  double objective_value(const std::vector<double>& x) const;

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
