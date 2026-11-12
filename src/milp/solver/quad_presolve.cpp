#include "milp/solver/quad_presolve.hpp"

#include <cmath>
#include <cstdlib>
#include <map>
#include <utility>

#include "milp/solver/reformulate.hpp"

namespace milp {

namespace {

bool integral(VarType t) { return t == VarType::Integer || t == VarType::Binary; }

struct Row {  // a quadratic row: sum quad[(i, j)] x_i x_j (i >= j) + sum lin[j] x_j, sides shifted by -shift
  std::map<std::pair<Index, Index>, double> quad;
  std::map<Index, double> lin;
  double shift = 0.0;  // constant moved out of the activity
  bool touched = false;
};

bool knob_on(const char* name) {
  const char* e = std::getenv(name);
  return !(e && *e == '0');
}

}  // namespace

void QuadPresolve::restore(std::vector<double>& x) const {
  for (const Alias& a : aliases) x[a.col] = a.scale * x[a.of] + a.offset;
}

QuadPresolve quad_presolve(const Model& m, Model& out) {
  QuadPresolve ps;
  out = m;
  const Index n = m.num_cols(), nr = m.num_rows();
  std::vector<char> is_quad(static_cast<size_t>(nr), 0), member(static_cast<size_t>(n), 0);
  for (size_t k = 0; k < m.qr_val.size(); ++k) {
    if (m.qr_val[k] == 0.0) continue;
    is_quad[m.qr_row[k]] = 1;
    member[m.qr_i[k]] = member[m.qr_j[k]] = 1;
  }
  const SparseMatrix R = m.A.transpose();  // row-wise

  // 1. Bounds implied by the linear rows, for the columns of the quadratic rows. Off unless
  // MILP_QUAD_PRESOLVE_BOUNDS=1: time-neutral on CBLIB (3 seeds, 1.002 [0.976, 1.031], 344 vs 343 of 471 solved)
  // with single models two to three times slower (classical_50_1, robust_50_4, shortfall_50_2).
  const char* bounds_env = std::getenv("MILP_QUAD_PRESOLVE_BOUNDS");
  if (bounds_env && *bounds_env == '1') {
    Model lin = m;
    lin.q_row.clear(), lin.q_col.clear(), lin.q_val.clear();
    lin.remove_rows(is_quad);
    std::vector<double> lo = m.col_lb, hi = m.col_ub;
    implied_bounds(lin, lo, hi);
    for (Index j = 0; j < n; ++j) {
      if (!member[j]) continue;
      if (lo[j] > out.col_lb[j]) out.col_lb[j] = lo[j], ++ps.bounds;
      if (hi[j] < out.col_ub[j]) out.col_ub[j] = hi[j], ++ps.bounds;
    }
  }

  // 2. Alias members.
  std::vector<Index> alias_of(static_cast<size_t>(n), -1);  // index into ps.aliases
  if (knob_on("MILP_QUAD_PRESOLVE_ALIAS")) {
    std::vector<char> in_q(static_cast<size_t>(n), 0), target(static_cast<size_t>(n), 0);
    for (size_t k = 0; k < m.q_val.size(); ++k) {
      if (m.q_val[k] != 0.0) in_q[m.q_row[k]] = in_q[m.q_col[k]] = 1;
    }
    for (Index c = 0; c < n; ++c) {
      if (!member[c] || in_q[c] || target[c] || m.col_type[c] != VarType::Continuous) continue;
      Index row = -1;
      double v = 0.0;
      int lin_rows = 0;
      for (Index k = m.A.begin(c); k < m.A.end(c); ++k) {
        if (m.A.value[k] == 0.0 || is_quad[m.A.index[k]]) continue;
        ++lin_rows, row = m.A.index[k], v = m.A.value[k];
      }
      if (lin_rows != 1 || m.row_lb[row] != m.row_ub[row] || !std::isfinite(m.row_lb[row])) continue;
      Index x = -1;
      double a = 0.0;
      int others = 0;
      for (Index k = R.begin(row); k < R.end(row); ++k) {
        if (R.value[k] == 0.0 || R.index[k] == c) continue;
        ++others, x = R.index[k], a = R.value[k];
      }
      if (others != 1 || alias_of[x] >= 0) continue;
      const double s = -a / v, o = m.row_lb[row] / v;
      if (!std::isfinite(s) || !std::isfinite(o) || std::abs(s) < 1e-9 || std::abs(s) > 1e9) continue;
      // Without a constant only: a shifted member changes how the row's violation is measured (a cone with
      // constants is judged relative to them, or stops being a cone and is judged relative to its terms), and
      // points accepted for the rewritten row violated the model's cone by 3e-6 to 1e-5 (CBLIB slay*).
      if (o != 0.0) continue;
      // c's bounds as bounds of x (an integer x keeps integral bounds; a contradiction is left to the solver).
      double xl = (out.col_lb[c] - o) / s, xu = (out.col_ub[c] - o) / s;
      if (s < 0.0) std::swap(xl, xu);
      if (integral(m.col_type[x])) xl = std::ceil(xl - 1e-9), xu = std::floor(xu + 1e-9);
      if (xl > out.col_lb[x] && xl <= out.col_ub[x]) out.col_lb[x] = xl, ++ps.bounds;
      if (xu < out.col_ub[x] && xu >= out.col_lb[x]) out.col_ub[x] = xu, ++ps.bounds;
      alias_of[c] = static_cast<Index>(ps.aliases.size());
      target[x] = 1;
      ps.aliases.push_back({c, x, s, o});
    }
  }

  // The quadratic rows with the aliases replaced (3.: and fixed columns folded).
  std::vector<Index> slot(static_cast<size_t>(nr), -1);
  std::vector<Row> rows;
  std::vector<Index> row_of;
  for (Index i = 0; i < nr; ++i) {
    if (!is_quad[i]) continue;
    slot[i] = static_cast<Index>(rows.size());
    rows.emplace_back();
    row_of.push_back(i);
  }
  struct Term {
    Index col;
    double scale, offset;
  };
  auto term = [&](Index j) {
    if (alias_of[j] < 0) return Term{j, 1.0, 0.0};
    const QuadPresolve::Alias& al = ps.aliases[alias_of[j]];
    return Term{al.of, al.scale, al.offset};
  };
  for (size_t r = 0; r < rows.size(); ++r) {
    Row& q = rows[r];
    for (Index k = R.begin(row_of[r]); k < R.end(row_of[r]); ++k) {
      if (R.value[k] == 0.0) continue;
      const Term t = term(R.index[k]);
      q.lin[t.col] += R.value[k] * t.scale;
      q.shift += R.value[k] * t.offset;
      q.touched = q.touched || t.col != R.index[k];
    }
  }
  for (size_t k = 0; k < m.qr_val.size(); ++k) {
    const double v = m.qr_val[k];
    if (v == 0.0) continue;
    Row& q = rows[slot[m.qr_row[k]]];
    const Term a = term(m.qr_i[k]), b = term(m.qr_j[k]);
    q.touched = q.touched || a.col != m.qr_i[k] || b.col != m.qr_j[k];
    q.quad[{std::max(a.col, b.col), std::min(a.col, b.col)}] += v * a.scale * b.scale;
    if (b.offset != 0.0) q.lin[a.col] += v * a.scale * b.offset;
    if (a.offset != 0.0) q.lin[b.col] += v * b.scale * a.offset;
    q.shift += v * a.offset * b.offset;
  }
  // (Kept only in rows that the perspective step below turns into cones again: a cone with a fixed member is
  // judged by its norm form, the folded quadratic relative to its terms, which is looser.)
  const bool fold = knob_on("MILP_QUAD_PRESOLVE_FIXED");
  std::vector<Row> unfolded(rows.size());
  std::vector<int> folded(rows.size(), 0);
  for (size_t r = 0; r < rows.size(); ++r) {
    Row& q = rows[r];
    unfolded[r] = q;
    for (auto it = q.quad.begin(); it != q.quad.end();) {
      const Index i = it->first.first, j = it->first.second;
      const double v = it->second;
      const bool fi = out.col_lb[i] == out.col_ub[i], fj = out.col_lb[j] == out.col_ub[j];
      if (v != 0.0 && !(fold && (fi || fj))) {
        ++it;
        continue;
      }
      if (v != 0.0) {
        if (fi && fj) q.shift += v * out.col_lb[i] * out.col_lb[j];
        else if (fi) q.lin[j] += v * out.col_lb[i];
        else q.lin[i] += v * out.col_lb[j];
        ++folded[r];
        q.touched = true;
      }
      it = q.quad.erase(it);
    }
  }

  // 4. Perspective strengthening.
  if (knob_on("MILP_QUAD_PRESOLVE_PERSPECTIVE")) {
    // switch_of[x]: a binary y with a row a x + b y <= 0 (a > 0, b < 0) or its mirror image, for x >= 0.
    std::vector<Index> switch_of(static_cast<size_t>(n), -1);
    for (Index i = 0; i < nr; ++i) {
      if (is_quad[i]) continue;
      Index c[2];
      double a[2];
      int cnt = 0;
      for (Index k = R.begin(i); k < R.end(i) && cnt < 3; ++k) {
        if (R.value[k] == 0.0) continue;
        if (cnt < 2) c[cnt] = R.index[k], a[cnt] = R.value[k];
        ++cnt;
      }
      if (cnt != 2) continue;
      for (int side = 0; side < 2; ++side) {  // a x <= 0 (side 0: the upper side; 1: the lower side, negated)
        const double rhs = side == 0 ? m.row_ub[i] : -m.row_lb[i], sg = side == 0 ? 1.0 : -1.0;
        if (rhs != 0.0) continue;
        for (int p = 0; p < 2; ++p) {
          const Index x = c[p], y = c[1 - p];
          if (!(sg * a[p] > 0.0 && sg * a[1 - p] < 0.0)) continue;
          if (m.col_type[y] != VarType::Binary && !(m.col_type[y] == VarType::Integer && out.col_lb[y] == 0.0 && out.col_ub[y] == 1.0)) continue;
          if (out.col_lb[y] != 0.0 || out.col_ub[y] != 1.0 || !(out.col_lb[x] >= 0.0)) continue;
          if (switch_of[x] < 0) switch_of[x] = y;
        }
      }
    }
    for (size_t r = 0; r < rows.size(); ++r) {
      Row& q = rows[r];
      const Index i = row_of[r];
      const bool up = std::isfinite(m.row_ub[i]), dn = std::isfinite(m.row_lb[i]);
      if (up == dn) continue;
      const double dir = up ? 1.0 : -1.0;
      if ((up ? m.row_ub[i] : m.row_lb[i]) - q.shift != 0.0) continue;
      Index y = -1, t = -1;
      double ct = 0.0;
      bool ok = !q.quad.empty();
      for (const auto& [key, v] : q.quad) {
        const Index x = key.first;
        if (key.first != key.second || !(dir * v > 0.0) || switch_of[x] < 0 || (y >= 0 && switch_of[x] != y)) {
          ok = false;
          break;
        }
        y = switch_of[x];
      }
      if (!ok) continue;
      for (const auto& [j, v] : q.lin) {
        if (v == 0.0) continue;
        if (t >= 0 || !(dir * v < 0.0)) {
          ok = false;
          break;
        }
        t = j, ct = v;
      }
      if (!ok || t < 0 || t == y || !(out.col_lb[t] >= 0.0) || out.col_lb[t] == out.col_ub[t]) continue;
      if (q.quad.count({t, t}) || q.quad.count({y, y})) continue;
      q.lin.erase(t);
      q.quad[{std::max(t, y), std::min(t, y)}] = ct;
      q.touched = true;
      ++ps.perspective;
      ps.fixed_terms += folded[r];
      folded[r] = 0;
    }
  }
  for (size_t r = 0; r < rows.size(); ++r) {
    if (folded[r] > 0) rows[r] = unfolded[r];
  }

  // The presolved model: the linear rows as they are, the quadratic rows from `rows`.
  bool any = false;
  for (const Row& q : rows) any = any || q.touched;
  if (!any) return ps;
  std::vector<Index> J, I;
  std::vector<double> V;
  for (Index j = 0; j < n; ++j) {
    for (Index k = m.A.begin(j); k < m.A.end(j); ++k) {
      if (!is_quad[m.A.index[k]]) J.push_back(j), I.push_back(m.A.index[k]), V.push_back(m.A.value[k]);
    }
  }
  out.qr_row.clear(), out.qr_i.clear(), out.qr_j.clear(), out.qr_val.clear();
  for (size_t r = 0; r < rows.size(); ++r) {
    const Row& q = rows[r];
    const Index i = row_of[r];
    for (const auto& [j, v] : q.lin) {
      if (v != 0.0) J.push_back(j), I.push_back(i), V.push_back(v);
    }
    for (const auto& [key, v] : q.quad) {
      if (v != 0.0) out.add_qrow(i, key.first, key.second, v);
    }
    if (std::isfinite(out.row_lb[i])) out.row_lb[i] -= q.shift;
    if (std::isfinite(out.row_ub[i])) out.row_ub[i] -= q.shift;
  }
  out.A = SparseMatrix::from_triplets(n, nr, J, I, V);
  out.compact_qrows();
  return ps;
}

}  // namespace milp
