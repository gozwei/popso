#include "milp/search/mip_problem.hpp"

#include <algorithm>
#include <cmath>

namespace milp {

bool MipProblem::from_model(const Model& m, MipProblem* out) {
  MipProblem& p = *out;
  const double sense = m.sense == ObjSense::Maximize ? -1.0 : 1.0;
  p.A = m.A;
  p.cost.resize(m.num_cols());
  for (Index j = 0; j < m.num_cols(); ++j) p.cost[j] = sense * m.obj[j];
  p.offset = sense * m.obj_offset;
  p.display_sense = sense;
  p.col_lb = m.col_lb;
  p.col_ub = m.col_ub;
  p.row_lb = m.row_lb;
  p.row_ub = m.row_ub;
  p.is_int.assign(m.num_cols(), 0);
  bool ok = true;
  for (Index j = 0; j < m.num_cols(); ++j) {
    if (!is_integral_type(m.col_type[j])) continue;
    p.is_int[j] = 1;
    p.col_lb[j] = std::ceil(p.col_lb[j] - 1e-6);
    p.col_ub[j] = std::floor(p.col_ub[j] + 1e-6);
    if (p.col_lb[j] > p.col_ub[j]) ok = false;
  }
  for (Index j = 0; j < m.num_cols(); ++j) ok = ok && p.col_lb[j] <= p.col_ub[j];
  for (Index i = 0; i < m.num_rows(); ++i) ok = ok && p.row_lb[i] <= p.row_ub[i];
  return ok;
}

double MipProblem::objective(const std::vector<double>& x) const {
  double v = offset;
  for (Index j = 0; j < num_cols(); ++j) v += cost[j] * x[j];
  return v;
}

double MipProblem::max_violation(const std::vector<double>& x, double* int_viol) const {
  double viol = 0.0, iv = 0.0;
  std::vector<double> act(num_rows(), 0.0);
  for (Index j = 0; j < num_cols(); ++j) {
    viol = std::max({viol, col_lb[j] - x[j], x[j] - col_ub[j]});
    if (is_int[j]) iv = std::max(iv, std::abs(x[j] - std::round(x[j])));
    for (Index p = A.begin(j); p < A.end(j); ++p) act[A.index[p]] += A.value[p] * x[j];
  }
  for (Index i = 0; i < num_rows(); ++i) viol = std::max({viol, row_lb[i] - act[i], act[i] - row_ub[i]});
  for (const Implication& im : implications) {
    if (std::abs(x[im.z] - im.zval) < 0.5) viol = std::max(viol, std::abs(x[im.col]));
  }
  if (int_viol) *int_viol = iv;
  return std::max(viol, iv);
}

bool MipProblem::apply_implications(std::vector<double>& lb, std::vector<double>& ub, std::vector<Index>* changed) const {
  for (bool again = true; again;) {
    again = false;
    for (const Implication& im : implications) {
      const Index z = im.z, c = im.col;
      const bool z_fixed = lb[z] == ub[z] && std::abs(lb[z] - im.zval) < 0.5;
      if (z_fixed) {
        if (lb[c] > 0.0 || ub[c] < 0.0) return false;
        if (lb[c] != 0.0 || ub[c] != 0.0) {
          lb[c] = ub[c] = 0.0;
          if (changed) changed->push_back(c);
        }
      } else if ((lb[c] > 0.0 || ub[c] < 0.0) && lb[z] <= im.zval && im.zval <= ub[z]) {
        // col cannot be 0: z must take another value (z is binary here).
        const double other = im.zval >= 0.5 ? 0.0 : 1.0;
        if (other < lb[z] || other > ub[z]) return false;
        lb[z] = ub[z] = other;
        if (changed) changed->push_back(z);
        again = true;
      }
    }
  }
  return true;
}

}  // namespace milp
