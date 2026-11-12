#include "milp/conic/relaxation.hpp"

#include <algorithm>
#include <cmath>
#include <map>

namespace milp {

namespace {

// Rows of F with S = F'F for the symmetric PSD matrix S (k x k, dense row-major) by Cholesky with diagonal pivoting,
// stopping when the largest remaining diagonal is below tol * the largest initial one. Each row of F is a dense
// vector over the k columns. False if a remaining diagonal is clearly negative (S not PSD).
bool pivoted_factor(std::vector<double> S, Index k, std::vector<std::vector<double>>& F) {
  F.clear();
  double dmax0 = 0.0;
  for (Index i = 0; i < k; ++i) dmax0 = std::max(dmax0, S[static_cast<size_t>(i) * k + i]);
  if (dmax0 <= 0.0) {
    for (Index i = 0; i < k; ++i) {
      if (S[static_cast<size_t>(i) * k + i] < -1e-9) return false;
    }
    return true;
  }
  const double tol = 1e-12 * dmax0;
  std::vector<char> done(static_cast<size_t>(k), 0);
  for (Index step = 0; step < k; ++step) {
    Index piv = -1;
    double d = tol;
    for (Index i = 0; i < k; ++i) {
      if (!done[i] && S[static_cast<size_t>(i) * k + i] > d) d = S[static_cast<size_t>(i) * k + i], piv = i;
    }
    if (piv < 0) break;
    // The column of L at the pivot: L(:, step) = S(:, piv) / sqrt(d); F's row is that column.
    std::vector<double> col(static_cast<size_t>(k), 0.0);
    const double sd = std::sqrt(d);
    for (Index i = 0; i < k; ++i) {
      if (!done[i]) col[i] = S[static_cast<size_t>(i) * k + piv] / sd;
    }
    done[piv] = 1;
    for (Index i = 0; i < k; ++i) {
      if (done[i] || col[i] == 0.0) continue;
      for (Index j = 0; j < k; ++j) {
        if (!done[j]) S[static_cast<size_t>(i) * k + j] -= col[i] * col[j];
      }
    }
    F.push_back(std::move(col));
  }
  for (Index i = 0; i < k; ++i) {
    if (!done[i] && S[static_cast<size_t>(i) * k + i] < -1e-7 * dmax0) return false;
  }
  return true;
}

}  // namespace

bool conic_relaxation(const MipProblem& p, ConicProblem& out, std::string* why, ConicMap* map) {
  const Index n = p.num_cols();
  out = ConicProblem{};
  out.n = n;
  out.c = p.cost;
  out.offset = p.offset;
  // Rows as (column, value) lists: equalities, orthant rows (G v <= h), then cones.
  std::vector<std::vector<std::pair<Index, double>>> eq, orth;
  std::vector<double> eq_rhs, orth_rhs;
  ConicMap mp;
  auto org = [&](bool is_eq, int8_t kind, Index idx, double sign) {
    (is_eq ? mp.a_kind : mp.g_kind).push_back(kind);
    (is_eq ? mp.a_index : mp.g_index).push_back(idx);
    (is_eq ? mp.a_sign : mp.g_sign).push_back(sign);
  };
  struct ConeRows {
    std::vector<std::vector<std::pair<Index, double>>> rows;
    std::vector<double> rhs;
  };
  std::vector<ConeRows> cones;
  // Work limits: the dense factorizations (k^3 / 3 flops per non-diagonal row of k columns) and the factors' size
  // (a dense k x k matrix of full rank gives a dense cone block in every KKT factorization).
  double flops = 0.0;
  long long factor_nnz = 0;
  for (Index j = 0; j < n; ++j) {
    const double lb = p.col_lb[j], ub = p.col_ub[j];
    if (lb == ub) {
      eq.push_back({{j, 1.0}}), eq_rhs.push_back(lb), org(true, 1, j, 1.0);
      continue;
    }
    if (is_finite(lb)) orth.push_back({{j, -1.0}}), orth_rhs.push_back(-lb), org(false, 1, j, -1.0);
    if (is_finite(ub)) orth.push_back({{j, 1.0}}), orth_rhs.push_back(ub), org(false, 1, j, 1.0);
  }
  {
    const SparseMatrix At = p.A.transpose();
    for (Index i = 0; i < p.num_rows(); ++i) {
      std::vector<std::pair<Index, double>> r;
      for (Index k = At.begin(i); k < At.end(i); ++k) r.emplace_back(At.index[k], At.value[k]);
      const double lo = p.row_lb[i], hi = p.row_ub[i];
      if (lo == hi) {
        eq.push_back(r), eq_rhs.push_back(lo), org(true, 0, i, 1.0);
        continue;
      }
      if (is_finite(hi)) orth.push_back(r), orth_rhs.push_back(hi), org(false, 0, i, 1.0);
      if (is_finite(lo)) {
        for (auto& e : r) e.second = -e.second;
        orth.push_back(std::move(r)), orth_rhs.push_back(-lo), org(false, 0, i, -1.0);
      }
    }
  }
  // The objective's quadratic part goes to P (1/2 x'Px, lower triangle): the MipProblem's own Q (an MIQP), and the
  // epigraph row 1/2 x'Qx - eta <= 0 of an outer approximation (eta's cost moves to P; eta is fixed at 0 here and
  // set to 1/2 x'Qx by the caller).
  std::vector<Index> PJ, PI;
  std::vector<double> PV;
  if (p.quadratic()) {
    for (Index j = 0; j < p.Q.num_major; ++j) {
      for (Index k = p.Q.begin(j); k < p.Q.end(j); ++k) PJ.push_back(j), PI.push_back(p.Q.index[k]), PV.push_back(p.Q.value[k]);
    }
  }
  for (const QuadRow& q : p.quad_rows) {
    ConeRows cr;
    if (q.epigraph >= 0 && !q.cone) {
      // Monomials c x_i x_j of 1/2 x'Qx: P_ii = 2 c, P_ij = c.
      for (size_t t = 0; t < q.qc.size(); ++t) {
        PJ.push_back(q.qj[t]), PI.push_back(q.qi[t]), PV.push_back(q.qi[t] == q.qj[t] ? 2.0 * q.qc[t] : q.qc[t]);
      }
      out.c[q.epigraph] = 0.0;
      eq.push_back({{q.epigraph, 1.0}}), eq_rhs.push_back(0.0), org(true, 2, q.epigraph, 1.0);
      continue;
    }
    if (q.cone) {
      // s = (h0 + h x, sqrt(gamma), b_m + a_m x) = h_vec - G x.
      std::vector<std::pair<Index, double>> r0;
      for (const auto& [j, v] : q.cone_h) r0.emplace_back(j, -v);
      cr.rows.push_back(std::move(r0)), cr.rhs.push_back(q.cone_h0);
      if (q.cone_gamma > 0.0) cr.rows.push_back({}), cr.rhs.push_back(std::sqrt(q.cone_gamma));
      for (size_t m = 0; m < q.cone_a.size(); ++m) {
        std::vector<std::pair<Index, double>> r;
        for (const auto& [j, v] : q.cone_a[m]) r.emplace_back(j, -v);
        cr.rows.push_back(std::move(r)), cr.rhs.push_back(q.cone_b[m]);
      }
      cones.push_back(std::move(cr));
      mp.cone_quad.push_back(static_cast<Index>(&q - p.quad_rows.data()));
      mp.cone_dir.push_back(is_finite(q.hi) ? 1.0 : -1.0);
      mp.cone_norm.push_back(1);
      continue;
    }
    // a'x + x'Sx <= hi in the row's direction (a lower side negated).
    const double sgn = is_finite(q.hi) ? 1.0 : -1.0;
    const double hi = sgn * (sgn > 0 ? q.hi : q.lo);
    std::map<Index, Index> local;
    std::vector<Index> cols;
    auto loc = [&](Index j) {
      auto [it, added] = local.emplace(j, static_cast<Index>(cols.size()));
      if (added) cols.push_back(j);
      return it->second;
    };
    for (size_t k = 0; k < q.qc.size(); ++k) loc(q.qi[k]), loc(q.qj[k]);
    const Index k = static_cast<Index>(cols.size());
    std::vector<std::vector<double>> F;
    bool diagonal = true;
    for (size_t t = 0; t < q.qc.size(); ++t) diagonal = diagonal && q.qi[t] == q.qj[t];
    if (diagonal) {
      std::vector<double> d(static_cast<size_t>(k), 0.0);
      for (size_t t = 0; t < q.qc.size(); ++t) d[local[q.qi[t]]] += sgn * q.qc[t];
      for (Index i = 0; i < k; ++i) {
        if (d[i] < -1e-12) {
          if (why) *why = "a quadratic row is not convex";
          return false;
        }
        if (d[i] > 0.0) {
          std::vector<double> row(static_cast<size_t>(k), 0.0);
          row[i] = std::sqrt(d[i]);
          F.push_back(std::move(row));
        }
      }
    } else {
      flops += static_cast<double>(k) * k * k / 3.0;
      if (flops > 3e9) {
        if (why) *why = "a dense quadratic row is too large for the conic form";
        return false;
      }
      std::vector<double> S(static_cast<size_t>(k) * k, 0.0);
      for (size_t t = 0; t < q.qc.size(); ++t) {
        const Index a = local[q.qi[t]], b = local[q.qj[t]];
        if (a == b) {
          S[static_cast<size_t>(a) * k + a] += sgn * q.qc[t];
        } else {
          S[static_cast<size_t>(a) * k + b] += 0.5 * sgn * q.qc[t];
          S[static_cast<size_t>(b) * k + a] += 0.5 * sgn * q.qc[t];
        }
      }
      if (!pivoted_factor(std::move(S), k, F)) {
        if (why) *why = "a quadratic row is not positive semidefinite";
        return false;
      }
    }
    for (const auto& f : F) {
      for (double v : f) factor_nnz += v != 0.0;
    }
    if (factor_nnz > 5000000) {
      if (why) *why = "the quadratic rows' factors are too large for the conic form";
      return false;
    }
    // u = hi - a'x: s0 = (1 + hi)/2 - a'x/2, s1 = (hi - 1)/2 - a'x/2, s_r = (F x)_r.
    std::vector<std::pair<Index, double>> ra;
    for (size_t t = 0; t < q.idx.size(); ++t) ra.emplace_back(q.idx[t], 0.5 * sgn * q.val[t]);
    cr.rows.push_back(ra), cr.rhs.push_back(0.5 * (1.0 + hi));
    cr.rows.push_back(ra), cr.rhs.push_back(0.5 * (hi - 1.0));
    for (const auto& f : F) {
      std::vector<std::pair<Index, double>> r;
      for (Index i = 0; i < k; ++i) {
        if (f[i] != 0.0) r.emplace_back(cols[i], -f[i]);
      }
      cr.rows.push_back(std::move(r)), cr.rhs.push_back(0.0);
    }
    cones.push_back(std::move(cr));
    mp.cone_quad.push_back(static_cast<Index>(&q - p.quad_rows.data()));
    mp.cone_dir.push_back(sgn);
    mp.cone_norm.push_back(0);
  }
  // Assemble A (equalities) and G (orthant rows, then the cones).
  auto assemble = [&](const std::vector<std::vector<std::pair<Index, double>>>& rows, SparseMatrix& M) {
    std::vector<Index> J, I;
    std::vector<double> V;
    for (size_t i = 0; i < rows.size(); ++i) {
      for (const auto& [j, v] : rows[i]) J.push_back(j), I.push_back(static_cast<Index>(i)), V.push_back(v);
    }
    M = SparseMatrix::from_triplets(n, static_cast<Index>(rows.size()), J, I, V);
  };
  assemble(eq, out.A);
  out.b = eq_rhs;
  std::vector<std::vector<std::pair<Index, double>>> g = orth;
  out.h = orth_rhs;
  out.l = static_cast<Index>(orth.size());
  for (ConeRows& cr : cones) {
    out.cones.push_back(static_cast<Index>(cr.rows.size()));
    for (size_t t = 0; t < cr.rows.size(); ++t) g.push_back(std::move(cr.rows[t])), out.h.push_back(cr.rhs[t]);
  }
  assemble(g, out.G);
  if (!PV.empty()) out.P = SparseMatrix::from_triplets(n, n, PJ, PI, PV);
  if (map) *map = std::move(mp);
  return true;
}

// Cones that fixed columns flatten: the head equal to a member (plus or minus) over the free columns and the
// constants, as in a rotated cone x^2 <= c u z with z fixed at 0, whose norm form is ||(x, w u)|| <= w u; or a head
// fixed at 0. Such a cone has no interior (the interior-point method fails on it: CBLIB
// achtziger_stolpe06-6.2flowc, "numerical error" for every integer assignment) and is linear: every other member
// is 0. Members over one free column are fixed that way and the row is removed from q; a row this does not settle
// stays as it is. False if a member cannot be 0 (the fixings are infeasible).
bool flatten_fixed_cones(MipProblem& q) {
  auto fixed = [&](Index j) { return q.col_lb[j] == q.col_ub[j]; };
  // expr = constant + its free terms (merged by column); returns the constant.
  auto split = [&](const std::vector<std::pair<Index, double>>& e, double c0, std::map<Index, double>& free_terms) {
    for (const auto& [j, v] : e) {
      if (fixed(j)) c0 += v * q.col_lb[j];
      else free_terms[j] += v;
    }
    for (auto it = free_terms.begin(); it != free_terms.end();) it = it->second == 0.0 ? free_terms.erase(it) : std::next(it);
    return c0;
  };
  std::vector<QuadRow> kept;
  for (QuadRow& r : q.quad_rows) {
    bool flat = false, settled = false;
    if (r.cone && r.cone_gamma == 0.0) {
      std::map<Index, double> hf;
      const double hc = split(r.cone_h, r.cone_h0, hf);
      size_t same = r.cone_a.size();
      for (size_t m = 0; m < r.cone_a.size() && same == r.cone_a.size(); ++m) {
        for (const double sg : {1.0, -1.0}) {
          std::map<Index, double> d = hf;
          double dc = hc;
          for (const auto& [j, v] : r.cone_a[m]) {
            if (fixed(j)) dc -= sg * v * q.col_lb[j];
            else d[j] -= sg * v;
          }
          dc -= sg * r.cone_b[m];
          bool zero = std::abs(dc) <= 1e-12;
          for (const auto& [j, v] : d) zero = zero && std::abs(v) <= 1e-12;
          if (zero) same = m;
        }
      }
      if (hf.empty() && std::abs(hc) <= 1e-12) same = r.cone_a.size() + 1;  // the head itself is 0: every member is
      flat = same != r.cone_a.size();
      if (flat) {
        settled = true;
        std::vector<std::pair<Index, double>> fixes;
        for (size_t m = 0; m < r.cone_a.size() && settled; ++m) {
          if (m == same) continue;
          std::map<Index, double> mf;
          const double mc = split(r.cone_a[m], r.cone_b[m], mf);
          if (mf.empty()) {
            if (std::abs(mc) > 1e-9) return false;
          } else if (mf.size() == 1) {
            fixes.emplace_back(mf.begin()->first, -mc / mf.begin()->second);
          } else {
            settled = false;
          }
        }
        for (size_t k = 0; k < fixes.size() && settled; ++k) {
          const auto [j, v] = fixes[k];
          if (fixed(j)) settled = std::abs(q.col_lb[j] - v) <= 1e-9;  // fixed by an earlier member or row
          else if (v < q.col_lb[j] - 1e-9 || v > q.col_ub[j] + 1e-9) return false;
          else q.col_lb[j] = q.col_ub[j] = std::clamp(v, q.col_lb[j], q.col_ub[j]);
        }
      }
    }
    if (!(flat && settled)) kept.push_back(std::move(r));
  }
  q.quad_rows = std::move(kept);
  return true;
}

}  // namespace milp
