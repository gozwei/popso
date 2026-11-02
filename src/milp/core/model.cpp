#include "milp/core/model.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace milp {

Index Model::num_integers() const {
  Index n = 0;
  for (VarType t : col_type) n += is_integral_type(t) ? 1 : 0;
  return n;
}

double Model::objective_value(const std::vector<double>& x) const {
  double v = obj_offset;
  for (Index j = 0; j < num_cols(); ++j) v += obj[j] * x[j];
  return v;
}

double Model::Violation::max() const { return std::max({bound, row, integrality}); }

Model::Violation Model::violation(const std::vector<double>& x) const {
  Violation v;
  std::vector<double> act(num_rows(), 0.0);
  for (Index j = 0; j < num_cols(); ++j) {
    v.bound = std::max({v.bound, col_lb[j] - x[j], x[j] - col_ub[j]});
    if (is_integral_type(col_type[j])) v.integrality = std::max(v.integrality, std::abs(x[j] - std::round(x[j])));
    for (Index p = A.begin(j); p < A.end(j); ++p) act[A.index[p]] += A.value[p] * x[j];
  }
  for (Index i = 0; i < num_rows(); ++i) v.row = std::max({v.row, row_lb[i] - act[i], act[i] - row_ub[i]});
  return v;
}

void Model::validate() const {
  const size_t n = obj.size(), m = row_lb.size();
  if (col_lb.size() != n || col_ub.size() != n || col_type.size() != n || col_names.size() != n)
    throw std::runtime_error("model: column arrays have inconsistent sizes");
  if (row_ub.size() != m || row_names.size() != m) throw std::runtime_error("model: row arrays have inconsistent sizes");
  if (A.num_major != static_cast<Index>(n) || A.num_minor != static_cast<Index>(m) || A.start.size() != n + 1)
    throw std::runtime_error("model: matrix dimensions do not match");
  for (size_t j = 0; j < n; ++j) {
    if (std::isnan(obj[j]) || std::isnan(col_lb[j]) || std::isnan(col_ub[j]))
      throw std::runtime_error("model: NaN in column " + col_names[j]);
  }
  for (double v : A.value)
    if (!std::isfinite(v)) throw std::runtime_error("model: non-finite matrix coefficient");
}

Index ModelBuilder::add_col(std::string name, double obj, double lb, double ub, VarType type) {
  if (type == VarType::Binary) {
    lb = std::max(lb, 0.0);
    ub = std::min(ub, 1.0);
  }
  m_.col_names.push_back(std::move(name));
  m_.obj.push_back(obj);
  m_.col_lb.push_back(lb);
  m_.col_ub.push_back(ub);
  m_.col_type.push_back(type);
  return m_.num_cols() - 1;
}

Index ModelBuilder::add_row(std::string name, double lb, double ub) {
  m_.row_names.push_back(std::move(name));
  m_.row_lb.push_back(lb);
  m_.row_ub.push_back(ub);
  return m_.num_rows() - 1;
}

void ModelBuilder::add_coef(Index row, Index col, double value) {
  trip_row_.push_back(row);
  trip_col_.push_back(col);
  trip_val_.push_back(value);
}

Model ModelBuilder::build() {
  m_.A = SparseMatrix::from_triplets(m_.num_cols(), m_.num_rows(), trip_col_, trip_row_, trip_val_);
  trip_col_.clear();
  trip_row_.clear();
  trip_val_.clear();
  Model out = std::move(m_);
  m_ = Model{};
  out.validate();
  return out;
}

}  // namespace milp
