#include "milp/search/mip_problem.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace milp {

double QuadRow::activity(const double* x) const {
  double s = 0.0;
  for (size_t k = 0; k < idx.size(); ++k) s += val[k] * x[idx[k]];
  for (size_t k = 0; k < qc.size(); ++k) s += qc[k] * x[qi[k]] * x[qj[k]];
  return s;
}

double QuadRow::magnitude(const double* x) const {
  if (epigraph >= 0) return std::abs(x[epigraph]);
  double s = 0.0;
  for (size_t k = 0; k < idx.size(); ++k) s += std::abs(val[k] * x[idx[k]]);
  for (size_t k = 0; k < qc.size(); ++k) s += std::abs(qc[k] * x[qi[k]] * x[qj[k]]);
  return s;
}

double QuadRow::violation(const double* x) const {
  if (cone) {
    double r2 = cone_gamma;
    for (size_t m = 0; m < cone_a.size(); ++m) {
      double t = cone_b[m];
      for (const auto& [j, a] : cone_a[m]) t += a * x[j];
      r2 += t * t;
    }
    double hx = cone_h0, scale = std::max({1.0, std::abs(cone_h0), std::sqrt(cone_gamma)});
    for (const auto& [j, h] : cone_h) hx += h * x[j];
    for (const double b : cone_b) scale = std::max(scale, std::abs(b));
    return (std::sqrt(r2) - hx) / scale;
  }
  const double a = activity(x), scale = std::max(1.0, magnitude(x));
  return std::max((lo - a) / scale, (a - hi) / scale);
}

namespace {

// Adds v to grad[j], listing j in out on its first nonzero.
inline void grad_add(std::vector<double>& grad, std::vector<std::pair<Index, double>>& out, Index j, double v) {
  if (grad[j] == 0.0 && v != 0.0) out.emplace_back(j, 0.0);
  grad[j] += v;
}

// Moves the sums from grad into out (grad left zero) and drops the zeros.
void grad_collect(std::vector<double>& grad, std::vector<std::pair<Index, double>>& out) {
  for (auto& [j, v] : out) {
    v = grad[j];
    grad[j] = 0.0;
  }
  out.erase(std::remove_if(out.begin(), out.end(), [](const auto& e) { return e.second == 0.0; }), out.end());
}

}  // namespace

void QuadRow::gradient(const double* x, std::vector<double>& grad, std::vector<std::pair<Index, double>>& out) const {
  out.clear();
  for (size_t k = 0; k < idx.size(); ++k) grad_add(grad, out, idx[k], val[k]);
  for (size_t k = 0; k < qc.size(); ++k) {
    grad_add(grad, out, qi[k], qc[k] * x[qj[k]]);
    grad_add(grad, out, qj[k], qc[k] * x[qi[k]]);
  }
  grad_collect(grad, out);
}

bool QuadRow::linearize(const double* x, std::vector<double>& grad, std::vector<std::pair<Index, double>>& coef,
                        double& rhs) const {
  double gval = 0.0;  // the convex function's value at x, with the cut grad (z - x) <= -gval
  if (cone) {
    coef.clear();
    std::vector<double> y(cone_a.size());
    double r2 = cone_gamma;
    for (size_t m = 0; m < cone_a.size(); ++m) {
      double s = cone_b[m];
      for (const auto& [j, a] : cone_a[m]) s += a * x[j];
      y[m] = s;
      r2 += s * s;
    }
    const double r = std::sqrt(r2);
    if (!(r > 1e-12) || !std::isfinite(r)) return false;  // the apex: g is not differentiable there
    for (size_t m = 0; m < cone_a.size(); ++m) {
      for (const auto& [j, a] : cone_a[m]) grad_add(grad, coef, j, a * y[m] / r);
    }
    double hx = cone_h0;
    for (const auto& [j, h] : cone_h) {
      grad_add(grad, coef, j, -h);
      hx += h * x[j];
    }
    grad_collect(grad, coef);
    gval = r - hx;
  } else {
    gradient(x, grad, coef);
    // The upper side's g = activity - hi; a lower side is negated into the same form.
    const bool upper = is_finite(hi);
    const double sign = upper ? 1.0 : -1.0;
    for (auto& e : coef) e.second *= sign;
    gval = sign * (activity(x) - (upper ? hi : lo));
  }
  if (coef.empty()) return false;
  double gx = 0.0;
  for (const auto& [j, v] : coef) gx += v * x[j];
  rhs = gx - gval;
  return std::isfinite(rhs);
}

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
  if (m.has_quadratic()) {
    p.Q = m.q_lower();
    for (double& v : p.Q.value) v *= sense;
  }
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
  for (Index j = 0; j < Q.num_major; ++j)
    for (Index k = Q.begin(j); k < Q.end(j); ++k) v += (Q.index[k] == j ? 0.5 : 1.0) * Q.value[k] * x[Q.index[k]] * x[j];
  return v;
}

double Bilinear::violation(const double* v) const {
  const double prod = (x_int ? std::round(v[x]) : v[x]) * (y_int ? std::round(v[y]) : v[y]);
  const double r = std::abs(v[w] - prod) / std::max(1.0, std::abs(prod));
  // A product that overflows (factors around 1e154 and beyond, from a relaxation without bounds) gives inf / inf:
  // never a satisfied product (std::max would skip the NaN, and such a point was once reported OPTIMAL at 1.8e308).
  return std::isfinite(r) ? r : kInf;
}

double MipProblem::max_violation(const std::vector<double>& x, double* int_viol, double* lin_viol) const {
  double viol = 0.0, iv = 0.0;
  std::vector<double> act(num_rows(), 0.0);
  for (Index j = 0; j < num_cols(); ++j) {
    viol = std::max({viol, col_lb[j] - x[j], x[j] - col_ub[j]});
    if (is_int[j]) iv = std::max(iv, std::abs(x[j] - std::round(x[j])));
    for (Index p = A.begin(j); p < A.end(j); ++p) act[A.index[p]] += A.value[p] * x[j];
  }
  for (const LiftedRow& lr : lifted_rows) {
    double e = 0.0;
    for (const auto& [t, c] : lr.terms) e += c * (x[bilinear[t].x] * x[bilinear[t].y] - x[bilinear[t].w]);
    act[lr.row] = std::isfinite(e) ? act[lr.row] + e : std::numeric_limits<double>::quiet_NaN();
  }
  for (Index i = 0; i < num_rows(); ++i) {
    const double v = std::max(row_lb[i] - act[i], act[i] - row_ub[i]);
    viol = v <= viol ? viol : std::isnan(v) ? kInf : v;
  }
  for (const Implication& im : implications) {
    if (std::abs(x[im.z] - im.zval) < 0.5) viol = std::max(viol, std::abs(x[im.col]));
  }
  if (lin_viol) *lin_viol = std::max(viol, iv);
  // (A violation that is not a number counts as infinite: std::max would drop it.)
  for (const QuadRow& q : quad_rows) {
    const double v = q.violation(x.data());
    viol = v <= viol ? viol : std::isnan(v) ? kInf : v;
  }
  for (const Bilinear& b : bilinear) viol = std::max(viol, b.violation(x.data()));
  if (std::isnan(viol)) viol = kInf;
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
