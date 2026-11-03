#include "milp/solver/reformulate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace milp {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kNonzero = 1e-9;  // |x| above this counts as nonzero when deriving auxiliary values

// Segment (pair of neighbours t, t+1) of an SOS2 point: -1 if all members are zero, -2 if the point
// violates the set or a value is unknown. `ord` lists the members in weight order.
int sos2_segment(const std::vector<Index>& ord, const std::vector<double>& x) {
  const int k = static_cast<int>(ord.size());
  int first = -1, last = -1;
  for (int t = 0; t < k; ++t) {
    const double v = x[ord[t]];
    if (std::isnan(v)) return -2;
    if (std::abs(v) > kNonzero) {
      if (first < 0) first = t;
      last = t;
    }
  }
  if (first < 0) return -1;
  if (last - first > 1) return -2;
  return first < k - 1 ? first : first - 1;
}

}  // namespace

void implied_bounds(const Model& m, std::vector<double>& lo, std::vector<double>& hi, int max_rounds) {
  const SparseMatrix R = m.A.transpose();  // row-wise
  const Index nr = m.num_rows();
  bool changed = false;
  auto tighten = [&](Index j, double b, bool upper) {
    if (!std::isfinite(b) || std::abs(b) > 1e15) return;
    const bool integral = is_integral_type(m.col_type[j]);
    if (upper) {
      if (integral) b = std::floor(b + 1e-6);
      if (b >= hi[j] - 1e-6 * std::max(1.0, std::abs(b))) return;
      if (b < lo[j] - 1e-6 * std::max(1.0, std::abs(b))) return;  // contradiction: leave it to the solver
      hi[j] = std::max(b, lo[j]);
    } else {
      if (integral) b = std::ceil(b - 1e-6);
      if (b <= lo[j] + 1e-6 * std::max(1.0, std::abs(b))) return;
      if (b > hi[j] + 1e-6 * std::max(1.0, std::abs(b))) return;
      lo[j] = std::min(b, hi[j]);
    }
    changed = true;
  };
  for (int round = 0; round < max_rounds; ++round) {
    changed = false;
    for (Index i = 0; i < nr; ++i) {
      const double L = m.row_lb[i], U = m.row_ub[i];
      if (!std::isfinite(L) && !std::isfinite(U)) continue;
      double minf = 0.0, maxf = 0.0;
      int min_inf = 0, max_inf = 0;
      auto contrib = [&](double a, Index j, double* cmin, double* cmax) {
        *cmin = a > 0 ? a * lo[j] : a * hi[j];
        *cmax = a > 0 ? a * hi[j] : a * lo[j];
      };
      for (Index k = R.begin(i); k < R.end(i); ++k) {
        double cmin, cmax;
        contrib(R.value[k], R.index[k], &cmin, &cmax);
        if (std::isfinite(cmin)) minf += cmin;
        else ++min_inf;
        if (std::isfinite(cmax)) maxf += cmax;
        else ++max_inf;
      }
      for (Index k = R.begin(i); k < R.end(i); ++k) {
        const double a = R.value[k];
        const Index j = R.index[k];
        if (std::abs(a) < 1e-9) continue;
        double cmin, cmax;
        contrib(a, j, &cmin, &cmax);
        if (std::isfinite(U)) {  // a x_j <= U - (minimal activity of the others)
          double rest = 0.0;
          bool ok = true;
          if (min_inf == 0) rest = minf - cmin;
          else if (min_inf == 1 && !std::isfinite(cmin)) rest = minf;
          else ok = false;
          if (ok) tighten(j, (U - rest) / a, a > 0);
        }
        if (std::isfinite(L)) {  // a x_j >= L - (maximal activity of the others)
          double rest = 0.0;
          bool ok = true;
          if (max_inf == 0) rest = maxf - cmax;
          else if (max_inf == 1 && !std::isfinite(cmax)) rest = maxf;
          else ok = false;
          if (ok) tighten(j, (L - rest) / a, a < 0);
        }
      }
    }
    if (!changed) break;
  }
}

std::vector<double> LogicalReformulation::extend(const std::vector<double>& x) const {
  std::vector<double> out(x.begin(), x.end());
  out.resize(static_cast<size_t>(orig_cols), kNaN);
  for (const auto& f : aux_value) out.push_back(f(x));
  return out;
}

LogicalReformulation reformulate_logical(const Model& in, double max_big_m) {
  in.validate();
  LogicalReformulation R;
  Model& m = R.model;
  const Index n = in.num_cols();
  R.orig_cols = n;
  m.name = in.name;
  m.sense = in.sense;
  m.obj_offset = in.obj_offset;
  m.col_names = in.col_names;
  m.obj = in.obj;
  m.col_lb = in.col_lb;
  m.col_ub = in.col_ub;
  m.col_type = in.col_type;
  m.row_names = in.row_names;
  m.row_lb = in.row_lb;
  m.row_ub = in.row_ub;
  std::vector<Index> J, I;
  std::vector<double> V;
  for (Index j = 0; j < n; ++j) {
    for (Index p = in.A.begin(j); p < in.A.end(j); ++p) {
      J.push_back(j);
      I.push_back(in.A.index[p]);
      V.push_back(in.A.value[p]);
    }
  }

  // Domains: the hull {0} u [l, u] for semi columns, tightened by the rows.
  std::vector<double> lo(n), hi(n);
  for (Index j = 0; j < n; ++j) {
    lo[j] = in.col_lb[j];
    hi[j] = in.col_ub[j];
    if (is_semi_type(in.col_type[j])) {
      if (lo[j] > hi[j]) lo[j] = hi[j] = 0.0;
      lo[j] = std::min(lo[j], 0.0);
      hi[j] = std::max(hi[j], 0.0);
    }
  }
  const std::vector<double> lo0 = lo, hi0 = hi;
  implied_bounds(in, lo, hi);
  for (Index j = 0; j < n; ++j) {
    // A little slack on derived continuous bounds: points within the feasibility tolerance stay feasible.
    if (is_integral_type(in.col_type[j])) continue;
    if (hi[j] < hi0[j]) hi[j] = std::min(hi0[j], hi[j] + 1e-6 * std::max(1.0, std::abs(hi[j])));
    if (lo[j] > lo0[j]) lo[j] = std::max(lo0[j], lo[j] - 1e-6 * std::max(1.0, std::abs(lo[j])));
  }
  auto usable = [&](double v) { return std::isfinite(v) && std::abs(v) <= max_big_m; };

  auto add_col = [&](const std::string& name, double lb, double ub, VarType t,
                     std::function<double(const std::vector<double>&)> value) {
    m.col_names.push_back(name);
    m.obj.push_back(0.0);
    m.col_lb.push_back(lb);
    m.col_ub.push_back(ub);
    m.col_type.push_back(t);
    R.aux_value.push_back(std::move(value));
    ++R.aux_cols;
    return m.num_cols() - 1;
  };
  auto add_row = [&](const std::string& name, double lb, double ub, const std::vector<std::pair<Index, double>>& terms) {
    const Index i = m.num_rows();
    m.row_names.push_back(name);
    m.row_lb.push_back(lb);
    m.row_ub.push_back(ub);
    for (const auto& [j, a] : terms) {
      if (a == 0.0) continue;
      J.push_back(j);
      I.push_back(i);
      V.push_back(a);
    }
    ++R.aux_rows;
  };
  auto nonzero_flag = [](Index c) {
    return [c](const std::vector<double>& x) {
      if (std::isnan(x[c])) return kNaN;
      return std::abs(x[c]) > kNonzero ? 1.0 : 0.0;
    };
  };

  // ---- semi-continuous / semi-integer columns
  for (Index j = 0; j < n; ++j) {
    const VarType t = in.col_type[j];
    if (!is_semi_type(t)) continue;
    const bool integral = t == VarType::SemiInteger;
    m.col_type[j] = integral ? VarType::Integer : VarType::Continuous;
    double l = in.col_lb[j], u = in.col_ub[j];
    if (integral) {
      l = std::ceil(l - 1e-9);
      u = std::floor(u + 1e-9);
    }
    if (l > u) {  // only x = 0 is left
      m.col_lb[j] = m.col_ub[j] = 0.0;
      continue;
    }
    if (l <= 0.0 && u >= 0.0) continue;  // 0 lies in [l, u]: an ordinary column
    const std::string& nm = in.col_names[j];
    m.col_lb[j] = std::min(l, 0.0);
    m.col_ub[j] = std::max(u, 0.0);
    const Index z = add_col("_sc_" + nm, 0.0, 1.0, VarType::Binary, nonzero_flag(j));
    if (l > 0.0) {
      add_row("_sc_" + nm + "_lo", 0.0, kInf, {{j, 1.0}, {z, -l}});  // x >= l z
      const double U = std::min(u, hi[j]);
      if (usable(U)) {
        add_row("_sc_" + nm + "_up", -kInf, 0.0, {{j, 1.0}, {z, -U}});  // x <= U z
        ++R.big_m;
      } else {
        R.implications.push_back({z, 0.0, j});
      }
    } else {  // u < 0: x in [l, u] u {0}
      add_row("_sc_" + nm + "_up", -kInf, 0.0, {{j, 1.0}, {z, -u}});  // x <= u z
      const double L = std::max(l, lo[j]);
      if (usable(L)) {
        add_row("_sc_" + nm + "_lo", 0.0, kInf, {{j, 1.0}, {z, -L}});  // x >= L z
        ++R.big_m;
      } else {
        R.implications.push_back({z, 0.0, j});
      }
    }
  }

  // ---- SOS1 / SOS2
  auto is_binary_col = [&](Index c) {
    return is_integral_type(in.col_type[c]) && !is_semi_type(in.col_type[c]) && in.col_lb[c] >= 0.0 &&
           in.col_ub[c] <= 1.0;
  };
  for (size_t s = 0; s < in.sos.size(); ++s) {
    const SosConstraint& set = in.sos[s];
    const size_t k = set.cols.size();
    if (k <= static_cast<size_t>(set.type)) continue;  // never violated
    std::vector<size_t> perm(k);
    for (size_t t = 0; t < k; ++t) perm[t] = t;
    std::stable_sort(perm.begin(), perm.end(), [&](size_t a, size_t b) { return set.weights[a] < set.weights[b]; });
    std::vector<Index> ord(k);
    for (size_t t = 0; t < k; ++t) ord[t] = set.cols[perm[t]];
    const std::string base = "_sos" + std::to_string(s) + (set.name.empty() ? std::string() : "_" + set.name);
    // Links x = 0 unless the binary sum `ind` is 1:  L * ind <= x <= U * ind, or an implication on a
    // per-member binary z (z = ind) where no usable bound exists.
    auto link = [&](Index c, const std::vector<Index>& ind, const std::string& nm,
                    std::function<double(const std::vector<double>&)> zval) {
      if (lo[c] >= 0.0 && hi[c] <= 0.0) return;  // always zero
      const bool bigm = (hi[c] <= 0.0 || usable(hi[c])) && (lo[c] >= 0.0 || usable(lo[c]));
      if (bigm) {
        if (hi[c] > 0.0) {
          std::vector<std::pair<Index, double>> terms{{c, 1.0}};
          for (Index y : ind) terms.emplace_back(y, -hi[c]);
          add_row(nm + "_up", -kInf, 0.0, terms);
          ++R.big_m;
        }
        if (lo[c] < 0.0) {
          std::vector<std::pair<Index, double>> terms{{c, 1.0}};
          for (Index y : ind) terms.emplace_back(y, -lo[c]);
          add_row(nm + "_lo", 0.0, kInf, terms);
          ++R.big_m;
        }
        return;
      }
      Index z = ind.size() == 1 ? ind[0] : -1;
      if (z < 0) {
        z = add_col(nm + "_z", 0.0, 1.0, VarType::Binary, std::move(zval));
        std::vector<std::pair<Index, double>> terms{{z, 1.0}};
        for (Index y : ind) terms.emplace_back(y, -1.0);
        add_row(nm + "_z", 0.0, 0.0, terms);  // z = sum ind
      }
      R.implications.push_back({z, 0.0, c});
    };
    if (set.type == 1) {
      bool all_binary = true;
      for (Index c : ord) all_binary = all_binary && is_binary_col(c);
      std::vector<std::pair<Index, double>> card;
      for (size_t t = 0; t < k; ++t) {
        const Index c = ord[t];
        if (all_binary) {
          card.emplace_back(c, 1.0);
          continue;
        }
        if (lo[c] >= 0.0 && hi[c] <= 0.0) continue;
        const std::string nm = base + "_" + std::to_string(t);
        const Index z = is_binary_col(c) ? c : add_col(nm, 0.0, 1.0, VarType::Binary, nonzero_flag(c));
        if (z != c) link(c, {z}, nm, nullptr);
        card.emplace_back(z, 1.0);
      }
      if (card.size() > 1) add_row(base, -kInf, 1.0, card);
      continue;
    }
    // SOS2: segment binaries y_t (members t and t+1 may be nonzero).
    std::vector<Index> y(k - 1);
    std::vector<std::pair<Index, double>> card;
    for (size_t t = 0; t + 1 < k; ++t) {
      y[t] = add_col(base + "_y" + std::to_string(t), 0.0, 1.0, VarType::Binary,
                     [ord, t](const std::vector<double>& x) {
                       const int seg = sos2_segment(ord, x);
                       if (seg == -2) return kNaN;
                       return seg == static_cast<int>(t) ? 1.0 : 0.0;
                     });
      card.emplace_back(y[t], 1.0);
    }
    add_row(base, -kInf, 1.0, card);
    for (size_t t = 0; t < k; ++t) {
      std::vector<Index> ind;
      if (t > 0) ind.push_back(y[t - 1]);
      if (t + 1 < k) ind.push_back(y[t]);
      link(ord[t], ind, base + "_" + std::to_string(t), [ord, t](const std::vector<double>& x) {
        const int seg = sos2_segment(ord, x);
        if (seg == -2) return kNaN;
        return seg == static_cast<int>(t) || seg + 1 == static_cast<int>(t) ? 1.0 : 0.0;
      });
    }
  }

  // ---- indicator constraints
  for (size_t q = 0; q < in.indicators.size(); ++q) {
    const IndicatorConstraint& c = in.indicators[q];
    const Index b = c.binary;
    const std::string nm = "_ind" + std::to_string(q) + (c.name.empty() ? std::string() : "_" + c.name);
    if (!is_binary_col(b))
      throw std::runtime_error("indicator constraint '" + c.name + "': the indicator variable '" + in.col_names[b] +
                               "' must be binary");
    std::vector<std::pair<Index, double>> terms;
    for (size_t t = 0; t < c.cols.size(); ++t) terms.emplace_back(c.cols[t], c.coefs[t]);
    if (lo[b] == hi[b]) {  // fixed (by its bounds or the rows)
      if (std::abs(lo[b] - c.value) < 0.5) add_row(nm, c.lb, c.ub, terms);  // always active
      continue;
    }
    double amin = 0.0, amax = 0.0;
    for (const auto& [j, a] : terms) {
      amin += a > 0 ? a * lo[j] : a * hi[j];
      amax += a > 0 ? a * hi[j] : a * lo[j];
    }
    if (std::isnan(amin)) amin = -kInf;  // 0 * inf
    if (std::isnan(amax)) amax = kInf;
    // The row is enforced when delta = 0: delta = 1 - b (value 1) or b (value 0).
    const double sgn = c.value == 1 ? -1.0 : 1.0;  // delta = (value == 1) + sgn * b
    const double d0 = c.value == 1 ? 1.0 : 0.0;
    bool slack_up = false, slack_lo = false;
    if (std::isfinite(c.ub) && amax > c.ub) {
      const double M = amax - c.ub;
      if (usable(M)) {  // a x <= ub + M delta
        std::vector<std::pair<Index, double>> t2 = terms;
        t2.emplace_back(b, -M * sgn);
        add_row(nm + "_up", -kInf, c.ub + M * d0, t2);
        ++R.big_m;
      } else {
        slack_up = true;
      }
    }
    if (std::isfinite(c.lb) && amin < c.lb) {
      const double M = c.lb - amin;
      if (usable(M)) {  // a x >= lb - M delta
        std::vector<std::pair<Index, double>> t2 = terms;
        t2.emplace_back(b, M * sgn);
        add_row(nm + "_lo", c.lb - M * d0, kInf, t2);
        ++R.big_m;
      } else {
        slack_lo = true;
      }
    }
    if (slack_up || slack_lo) {
      // a x - s within the fallback sides; s = 0 when the indicator is active.
      const double rlo = slack_lo ? c.lb : -kInf, rhi = slack_up ? c.ub : kInf;
      const Index s = add_col(nm + "_s", slack_lo ? -kInf : 0.0, slack_up ? kInf : 0.0, VarType::Continuous,
                              [terms, rlo, rhi](const std::vector<double>& x) {
                                double act = 0.0;
                                for (const auto& [j, a] : terms) act += a * x[j];
                                if (std::isnan(act)) return kNaN;
                                return act - std::clamp(act, rlo, rhi);
                              });
      std::vector<std::pair<Index, double>> t2 = terms;
      t2.emplace_back(s, -1.0);
      add_row(nm + "_s", rlo, rhi, t2);
      R.implications.push_back({b, static_cast<double>(c.value), s});
    }
  }

  m.A = SparseMatrix::from_triplets(m.num_cols(), m.num_rows(), J, I, V);
  m.validate();
  return R;
}

}  // namespace milp
