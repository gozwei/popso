#include "milp/core/model.hpp"

#include <algorithm>
#include <cmath>
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

Index Model::num_semi() const {
  Index n = 0;
  for (VarType t : col_type) n += is_semi_type(t) ? 1 : 0;
  return n;
}

double Model::Violation::max() const { return std::max({bound, row, integrality, logical}); }

namespace {

// Smallest violation of an SOS set by x: the largest |x| of a member that has to be zero
// (SOS1: all but the largest; SOS2: all outside the best pair of neighbours by weight).
double sos_violation(const SosConstraint& s, const std::vector<double>& x) {
  const size_t k = s.cols.size();
  if (k <= static_cast<size_t>(s.type)) return 0.0;
  std::vector<size_t> ord(k);
  for (size_t t = 0; t < k; ++t) ord[t] = t;
  if (s.weights.size() == k)
    std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return s.weights[a] < s.weights[b]; });
  std::vector<double> a(k);
  for (size_t t = 0; t < k; ++t) a[t] = std::abs(x[s.cols[ord[t]]]);
  if (s.type == 1) {
    double m1 = 0.0, m2 = 0.0;
    for (double v : a) {
      if (v > m1) m2 = m1, m1 = v;
      else m2 = std::max(m2, v);
    }
    return m2;
  }
  std::vector<double> pre(k + 1, 0.0), suf(k + 1, 0.0);  // pre[t] = max a[0..t-1], suf[t] = max a[t..]
  for (size_t t = 0; t < k; ++t) pre[t + 1] = std::max(pre[t], a[t]);
  for (size_t t = k; t-- > 0;) suf[t] = std::max(suf[t + 1], a[t]);
  double best = kInf;
  for (size_t t = 0; t + 1 < k; ++t) best = std::min(best, std::max(pre[t], suf[t + 2]));
  return best;
}

}  // namespace

Model::Violation Model::violation(const std::vector<double>& x) const {
  Violation v;
  std::vector<double> act(num_rows(), 0.0);
  for (Index j = 0; j < num_cols(); ++j) {
    double bv = std::max(col_lb[j] - x[j], x[j] - col_ub[j]);
    if (is_semi_type(col_type[j])) bv = std::min(bv, std::abs(x[j]));  // x = 0 is allowed too
    v.bound = std::max(v.bound, bv);
    if (is_integral_type(col_type[j])) v.integrality = std::max(v.integrality, std::abs(x[j] - std::round(x[j])));
    for (Index p = A.begin(j); p < A.end(j); ++p) act[A.index[p]] += A.value[p] * x[j];
  }
  for (Index i = 0; i < num_rows(); ++i) v.row = std::max({v.row, row_lb[i] - act[i], act[i] - row_ub[i]});
  for (const SosConstraint& s : sos) v.logical = std::max(v.logical, sos_violation(s, x));
  for (const IndicatorConstraint& c : indicators) {
    if (std::abs(x[c.binary] - c.value) >= 0.5) continue;
    double a = 0.0;
    for (size_t t = 0; t < c.cols.size(); ++t) a += c.coefs[t] * x[c.cols[t]];
    v.logical = std::max({v.logical, c.lb - a, a - c.ub});
  }
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
  auto check_col = [&](Index j, const std::string& what) {
    if (j < 0 || static_cast<size_t>(j) >= n) throw std::runtime_error("model: column index out of range in " + what);
  };
  for (const SosConstraint& s : sos) {
    const std::string what = "SOS '" + s.name + "'";
    if (s.type != 1 && s.type != 2) throw std::runtime_error("model: " + what + " must be of type 1 or 2");
    if (s.weights.size() != s.cols.size()) throw std::runtime_error("model: " + what + ": one weight per member needed");
    for (Index j : s.cols) check_col(j, what);
    for (double w : s.weights)
      if (!std::isfinite(w)) throw std::runtime_error("model: non-finite weight in " + what);
  }
  for (const IndicatorConstraint& c : indicators) {
    const std::string what = "indicator constraint '" + c.name + "'";
    check_col(c.binary, what);
    if (c.value != 0 && c.value != 1) throw std::runtime_error("model: " + what + ": the value must be 0 or 1");
    if (c.coefs.size() != c.cols.size()) throw std::runtime_error("model: " + what + ": one coefficient per column needed");
    for (Index j : c.cols) check_col(j, what);
    for (double a : c.coefs)
      if (!std::isfinite(a)) throw std::runtime_error("model: non-finite coefficient in " + what);
    if (std::isnan(c.lb) || std::isnan(c.ub)) throw std::runtime_error("model: NaN side in " + what);
  }
}

void Model::remove_rows(const std::vector<char>& del) {
  const Index m = num_rows();
  std::vector<Index> map(m, -1);
  Index k = 0;
  for (Index i = 0; i < m; ++i) {
    if (del[i]) continue;
    map[i] = k;
    if (k != i) row_names[k] = std::move(row_names[i]);  // (a self-move would empty the name)
    row_lb[k] = row_lb[i];
    row_ub[k] = row_ub[i];
    ++k;
  }
  row_names.resize(k);
  row_lb.resize(k);
  row_ub.resize(k);
  std::vector<Index> J, I;
  std::vector<double> V;
  for (Index j = 0; j < num_cols(); ++j) {
    for (Index p = A.begin(j); p < A.end(j); ++p) {
      if (map[A.index[p]] < 0) continue;
      J.push_back(j);
      I.push_back(map[A.index[p]]);
      V.push_back(A.value[p]);
    }
  }
  A = SparseMatrix::from_triplets(num_cols(), k, J, I, V);
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
