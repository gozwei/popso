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
  for (size_t k = 0; k < q_val.size(); ++k) {
    const Index i = q_row[k], j = q_col[k];
    v += (i == j ? 0.5 : 1.0) * q_val[k] * x[i] * x[j];
  }
  return v;
}

double Model::objective_value(int k, const std::vector<double>& x) const {
  if (k <= 0) return objective_value(x);
  const Objective& o = objectives[static_cast<size_t>(k - 1)];
  double v = o.offset;
  for (size_t t = 0; t < o.cols.size(); ++t) v += o.coefs[t] * x[static_cast<size_t>(o.cols[t])];
  return v;
}

bool Model::has_quadratic() const {
  for (double v : q_val)
    if (v != 0.0) return true;
  return false;
}

bool Model::has_quadratic_rows() const {
  for (double v : qr_val)
    if (v != 0.0) return true;
  return false;
}

void Model::add_qrow(Index row, Index i, Index j, double v) {
  if (i < j) std::swap(i, j);
  qr_row.push_back(row);
  qr_i.push_back(i);
  qr_j.push_back(j);
  qr_val.push_back(v);
}

void Model::compact_qrows() {
  const size_t k = qr_val.size();
  if (k == 0) return;
  std::vector<size_t> ord(k);
  for (size_t t = 0; t < k; ++t) ord[t] = t;
  std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
    if (qr_row[a] != qr_row[b]) return qr_row[a] < qr_row[b];
    if (qr_i[a] != qr_i[b]) return qr_i[a] < qr_i[b];
    return qr_j[a] < qr_j[b];
  });
  std::vector<Index> row, ii, jj;
  std::vector<double> val;
  for (size_t t = 0; t < k; ++t) {
    const size_t e = ord[t];
    if (!row.empty() && row.back() == qr_row[e] && ii.back() == qr_i[e] && jj.back() == qr_j[e]) {
      val.back() += qr_val[e];
    } else {
      if (!val.empty() && val.back() == 0.0) row.pop_back(), ii.pop_back(), jj.pop_back(), val.pop_back();
      row.push_back(qr_row[e]), ii.push_back(qr_i[e]), jj.push_back(qr_j[e]), val.push_back(qr_val[e]);
    }
  }
  if (!val.empty() && val.back() == 0.0) row.pop_back(), ii.pop_back(), jj.pop_back(), val.pop_back();
  qr_row.swap(row), qr_i.swap(ii), qr_j.swap(jj), qr_val.swap(val);
}

std::vector<double> Model::row_activity(const std::vector<double>& x) const {
  std::vector<double> act(num_rows(), 0.0);
  for (Index j = 0; j < num_cols(); ++j) {
    for (Index p = A.begin(j); p < A.end(j); ++p) act[A.index[p]] += A.value[p] * x[j];
  }
  for (size_t k = 0; k < qr_val.size(); ++k) act[qr_row[k]] += qr_val[k] * x[qr_i[k]] * x[qr_j[k]];
  return act;
}

void Model::add_q(Index i, Index j, double v) {
  if (i < j) std::swap(i, j);
  q_row.push_back(i);
  q_col.push_back(j);
  q_val.push_back(v);
}

SparseMatrix Model::q_lower() const {
  return SparseMatrix::from_triplets(num_cols(), num_cols(), q_col, q_row, q_val);
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
  for (size_t k = 0; k < qr_val.size(); ++k) act[qr_row[k]] += qr_val[k] * x[qr_i[k]] * x[qr_j[k]];
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
  if (q_row.size() != q_val.size() || q_col.size() != q_val.size())
    throw std::runtime_error("model: quadratic objective arrays have inconsistent sizes");
  for (size_t k = 0; k < q_val.size(); ++k) {
    if (q_row[k] < 0 || q_col[k] < 0 || static_cast<size_t>(q_row[k]) >= n || static_cast<size_t>(q_col[k]) >= n)
      throw std::runtime_error("model: column index out of range in the quadratic objective");
    if (q_row[k] < q_col[k]) throw std::runtime_error("model: quadratic objective entries must be in the lower triangle");
    if (!std::isfinite(q_val[k])) throw std::runtime_error("model: non-finite quadratic objective coefficient");
  }
  if (qr_row.size() != qr_val.size() || qr_i.size() != qr_val.size() || qr_j.size() != qr_val.size())
    throw std::runtime_error("model: quadratic constraint arrays have inconsistent sizes");
  for (size_t k = 0; k < qr_val.size(); ++k) {
    if (qr_row[k] < 0 || static_cast<size_t>(qr_row[k]) >= m)
      throw std::runtime_error("model: row index out of range in a quadratic constraint term");
    if (qr_i[k] < 0 || qr_j[k] < 0 || static_cast<size_t>(qr_i[k]) >= n || static_cast<size_t>(qr_j[k]) >= n)
      throw std::runtime_error("model: column index out of range in a quadratic constraint term");
    if (qr_i[k] < qr_j[k]) throw std::runtime_error("model: quadratic constraint terms must have i >= j");
    if (!std::isfinite(qr_val[k])) throw std::runtime_error("model: non-finite quadratic constraint coefficient");
  }
  auto check_col = [&](Index j, const std::string& what) {
    if (j < 0 || static_cast<size_t>(j) >= n) throw std::runtime_error("model: column index out of range in " + what);
  };
  if (!objectives.empty()) {
    if (has_quadratic()) throw std::runtime_error("model: several objectives need a linear first objective");
    if (!std::isfinite(obj_weight) || !(obj_abs_tol >= 0.0) || !(obj_rel_tol >= 0.0) || !std::isfinite(obj_abs_tol) ||
        !std::isfinite(obj_rel_tol))
      throw std::runtime_error("model: invalid weight or tolerance of the first objective");
  }
  for (const Objective& o : objectives) {
    const std::string what = "objective '" + o.name + "'";
    if (o.coefs.size() != o.cols.size()) throw std::runtime_error("model: " + what + ": one coefficient per column needed");
    for (Index j : o.cols) check_col(j, what);
    for (double a : o.coefs)
      if (!std::isfinite(a)) throw std::runtime_error("model: non-finite coefficient in " + what);
    if (!std::isfinite(o.offset) || !std::isfinite(o.weight)) throw std::runtime_error("model: non-finite offset or weight in " + what);
    if (!(o.abs_tol >= 0.0) || !(o.rel_tol >= 0.0) || !std::isfinite(o.abs_tol) || !std::isfinite(o.rel_tol))
      throw std::runtime_error("model: " + what + ": tolerances must be finite and not negative");
  }
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
  size_t t = 0;  // quadratic terms of the rows kept
  for (size_t q = 0; q < qr_val.size(); ++q) {
    if (map[qr_row[q]] < 0) continue;
    qr_row[t] = map[qr_row[q]], qr_i[t] = qr_i[q], qr_j[t] = qr_j[q], qr_val[t] = qr_val[q];
    ++t;
  }
  qr_row.resize(t), qr_i.resize(t), qr_j.resize(t), qr_val.resize(t);
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
