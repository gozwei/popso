#include "milp/cuts/cut.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>

#include "milp/search/propagation.hpp"

namespace milp {

namespace {

bool is_binary(const MipProblem& p, Index j) { return p.is_int[j] && p.col_lb[j] == 0.0 && p.col_ub[j] == 1.0; }

}  // namespace

bool CutData::add_vb(Index x, const VarBound& vb, bool upper) {
  std::vector<VarBound>& list = upper ? vub[x] : vlb[x];
  // Bound values at y = 0 / 1 (binary y): a new bound dominated by one with the same y is skipped,
  // one it dominates is replaced.
  const double s = upper ? 1.0 : -1.0;
  for (VarBound& o : list) {
    if (o.y != vb.y) continue;
    const bool new_weaker = s * vb.constant >= s * o.constant - 1e-9 &&
                            s * (vb.constant + vb.coef) >= s * (o.constant + o.coef) - 1e-9;
    if (new_weaker) return false;
    const bool new_stronger = s * vb.constant <= s * o.constant && s * (vb.constant + vb.coef) <= s * (o.constant + o.coef);
    if (new_stronger) {
      o = vb;
      return true;
    }
  }
  if (list.size() >= kMaxVbPerCol) return false;
  list.push_back(vb);
  ++num_implied_vbs;
  return true;
}

void CutData::build(const MipProblem& p) {
  const Index n = p.num_cols();
  vlb.assign(n, {});
  vub.assign(n, {});
  num_implied_vbs = 0;
  if (const char* v = std::getenv("MILP_NO_IMPLIED_VB")) implied_vbs = !(*v && *v != '0');
  if (const char* v = std::getenv("MILP_NO_PROBE_VB")) probe_vbs = !(*v && *v != '0');
  const SparseMatrix rows = p.A.transpose();
  for (Index i = 0; i < p.num_rows(); ++i) {
    if (rows.end(i) - rows.begin(i) != 2) continue;
    Index j = rows.index[rows.begin(i)], k = rows.index[rows.begin(i) + 1];
    double a = rows.value[rows.begin(i)], c = rows.value[rows.begin(i) + 1];
    if (p.is_int[j]) std::swap(j, k), std::swap(a, c);
    // Need j continuous, k integer and not fixed.
    if (p.is_int[j] || !p.is_int[k] || p.col_lb[k] == p.col_ub[k]) continue;
    // a x + c y <= up  ->  a > 0: x <= up/a - (c/a) y (VUB), a < 0: VLB.
    if (is_finite(p.row_ub[i])) {
      VarBound vb{k, -c / a, p.row_ub[i] / a};
      (a > 0 ? vub : vlb)[j].push_back(vb);
    }
    if (is_finite(p.row_lb[i])) {
      VarBound vb{k, -c / a, p.row_lb[i] / a};
      (a > 0 ? vlb : vub)[j].push_back(vb);
    }
  }
  if (!implied_vbs) return;
  // Variable bounds of continuous columns implied by single longer rows (SCIP / HiGHS keep such
  // implications in their variable bound stores): fixing a binary y of the row to 0 / 1 bounds
  // a x <= b - minact(rest) for each continuous x, e.g. x1 + x2 <= 180 y with x2 >= 0 gives
  // x1 <= 180 y. Used by the c-MIR bound substitution and the flow cover relaxation.
  constexpr Index kMaxRowLen = 64;
  long long work = 0;
  const long long work_limit = 20 * static_cast<long long>(p.A.index.size()) + 100000;
  for (Index i = 0; i < p.num_rows() && work < work_limit; ++i) {
    const Index rb = rows.begin(i), re = rows.end(i);
    if (re - rb < 3 || re - rb > kMaxRowLen) continue;
    Index nbin = 0, ncont = 0;
    for (Index k = rb; k < re; ++k) {
      const Index j = rows.index[k];
      nbin += is_binary(p, j);
      ncont += !p.is_int[j] && p.col_lb[j] < p.col_ub[j];
    }
    if (nbin == 0 || ncont == 0) continue;
    for (double sign : {1.0, -1.0}) {  // the row side as  sum (sign * a) x <= b
      const double b = sign > 0 ? p.row_ub[i] : -p.row_lb[i];
      if (!is_finite(b)) continue;
      double minact = 0.0;
      bool finite = true;
      for (Index k = rb; k < re && finite; ++k) {
        const Index j = rows.index[k];
        const double a = sign * rows.value[k];
        const double bnd = a > 0 ? p.col_lb[j] : p.col_ub[j];
        finite = is_finite(bnd);
        minact += a * bnd;
      }
      if (!finite) continue;
      work += static_cast<long long>(nbin) * ncont;
      for (Index ky = rb; ky < re; ++ky) {
        const Index y = rows.index[ky];
        if (!is_binary(p, y)) continue;
        const double ay = sign * rows.value[ky];
        const double ymin = std::min(ay, 0.0);
        for (Index kx = rb; kx < re; ++kx) {
          const Index x = rows.index[kx];
          if (p.is_int[x] || p.col_lb[x] == p.col_ub[x]) continue;
          const double ax = sign * rows.value[kx];
          const double xmin = ax > 0 ? ax * p.col_lb[x] : ax * p.col_ub[x];
          const double rest = minact - ymin - xmin;  // minimal activity without x and y
          const double s0 = b - rest, s1 = b - rest - ay;  // ax x <= s_v when y = v
          if (ax > 0) {
            const double ub = p.col_ub[x];
            const double c0 = std::min(s0 / ax, ub), c1 = std::min(s1 / ax, ub);
            if (is_finite(c0) && is_finite(c1) && std::min(c0, c1) < ub - 1e-6) add_vb(x, VarBound{y, c1 - c0, c0}, true);
          } else {
            const double lb = p.col_lb[x];
            const double c0 = std::max(s0 / ax, lb), c1 = std::max(s1 / ax, lb);
            if (is_finite(c0) && is_finite(c1) && std::max(c0, c1) > lb + 1e-6) add_vb(x, VarBound{y, c1 - c0, c0}, false);
          }
        }
      }
    }
  }
}

void CutData::probe(const MipProblem& p, const double* x, long long work_limit) {
  probed = true;
  const Index n = p.num_cols();
  auto binary = [&](Index j) { return p.is_int[j] && p.col_lb[j] == 0.0 && p.col_ub[j] == 1.0; };
  std::vector<Index> order;
  for (Index j = 0; j < n; ++j) {
    if (binary(j)) order.push_back(j);
  }
  if (order.empty()) return;
  if (x) {
    // Fractional binaries first (most fractional first), then the others in index order.
    std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) {
      const double fa = std::min(x[a], 1.0 - x[a]), fb = std::min(x[b], 1.0 - x[b]);
      return (fa > 1e-6 ? fa : 0.0) > (fb > 1e-6 ? fb : 0.0);
    });
  }
  constexpr size_t kMaxImps = 500000, kMaxPairs = 4000000;
  const long long nnz = static_cast<long long>(p.A.index.size());
  long long budget = work_limit;
  DomainPropagator prop(p);
  prop.work_limit = std::max<long long>(2 * nnz, 100000);  // per probe
  prop.continuous = true;  // e.g. time windows: a chain of arcs implies other arcs through the times
  std::vector<double> lb = p.col_lb, ub = p.col_ub;
  std::vector<double> l0(n), u0(n), l1(n), u1(n);  // bounds after y = 0 / 1, valid where st == y
  std::vector<Index> st0(n, -1), st1(n, -1), t0, t1, seed(1);
  std::vector<uint64_t> pairs;
  for (Index y : order) {
    if (budget <= 0 || pairs.size() >= kMaxPairs) break;
    bool ok[2];
    for (int v = 0; v < 2; ++v) {
      std::vector<Index>& t = v ? t1 : t0;
      t.clear();
      seed[0] = y;
      lb[y] = ub[y] = v;
      ok[v] = prop.propagate(lb, ub, seed, t);
      budget -= prop.last_work + 1;
      std::vector<double>& l = v ? l1 : l0;
      std::vector<double>& u = v ? u1 : u0;
      std::vector<Index>& st = v ? st1 : st0;
      for (Index j : t) l[j] = lb[j], u[j] = ub[j], st[j] = y;
      for (Index j : t) lb[j] = p.col_lb[j], ub[j] = p.col_ub[j];
      lb[y] = 0.0, ub[y] = 1.0;
    }
    if (!ok[0] || !ok[1]) continue;  // y is fixed (left to presolve)
    for (int side = 0; side < 2; ++side) {
      for (Index j : side ? t1 : t0) {
        if (j == y || (side == 1 && st0[j] == y)) continue;  // each column once
        const double lo = p.col_lb[j], hi = p.col_ub[j];
        const double c0u = st0[j] == y ? u0[j] : hi, c1u = st1[j] == y ? u1[j] : hi;
        const double c0l = st0[j] == y ? l0[j] : lo, c1l = st1[j] == y ? l1[j] : lo;
        if (binary(j)) {
          // y = v implies x_j = w: literals (y = v) and (x_j = 1 - w) conflict.
          for (int v = 0; v < 2; ++v) {
            const double cu = v ? c1u : c0u, cl = v ? c1l : c0l;
            if (cu > 0.5 && cl < 0.5) continue;
            const uint64_t a = static_cast<uint64_t>(v ? 2 * y : 2 * y + 1);
            const uint64_t b = static_cast<uint64_t>(cu < 0.5 ? 2 * j : 2 * j + 1);
            pairs.push_back(std::min(a, b) << 32 | std::max(a, b));
          }
          continue;
        }
        if (!p.is_int[j]) {
          // Continuous: a variable bound for the c-MIR / flow cover bound substitution (the propagated
          // bounds keep the propagator's feasibility slack, so they are valid but slightly weak).
          if (!probe_vbs) continue;
          if (is_finite(c0u) && is_finite(c1u) && std::min(c0u, c1u) < hi - 1e-6 * (1.0 + std::abs(hi))) {
            add_vb(j, VarBound{y, c1u - c0u, c0u}, true);
          }
          if (is_finite(c0l) && is_finite(c1l) && std::max(c0l, c1l) > lo + 1e-6 * (1.0 + std::abs(lo))) {
            add_vb(j, VarBound{y, c1l - c0l, c0l}, false);
          }
          continue;
        }
        if (probe_imps.size() >= kMaxImps) continue;
        if (is_finite(c0u) && is_finite(c1u) && std::min(c0u, c1u) < hi - 1e-6) probe_imps.push_back({j, y, c0u, c1u, true});
        if (is_finite(c0l) && is_finite(c1l) && std::max(c0l, c1l) > lo + 1e-6) probe_imps.push_back({j, y, c0l, c1l, false});
      }
    }
  }
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  conflict_start.assign(2 * static_cast<size_t>(n) + 1, 0);
  for (uint64_t q : pairs) ++conflict_start[(q >> 32) + 1], ++conflict_start[(q & 0xffffffffu) + 1];
  for (size_t k = 1; k < conflict_start.size(); ++k) conflict_start[k] += conflict_start[k - 1];
  conflict_lit.assign(2 * pairs.size(), 0);
  std::vector<Index> pos(conflict_start.begin(), conflict_start.end() - 1);
  for (uint64_t q : pairs) {
    const Index a = static_cast<Index>(q >> 32), b = static_cast<Index>(q & 0xffffffffu);
    conflict_lit[pos[a]++] = b;
    conflict_lit[pos[b]++] = a;
  }
}

bool clean_cut(Cut& c, const std::vector<double>& lb, const std::vector<double>& ub, Index max_nnz) {
  if (!std::isfinite(c.rhs)) return false;
  // Cheap density test before sorting (dense Gomory candidates on wide models, neos-960392: the
  // sort dominated the root cut loop): merge in a scatter array and count the entries that survive
  // the cleaning below; reject if more than max_nnz.
  if (static_cast<Index>(c.idx.size()) > max_nnz && !lb.empty()) {
    thread_local std::vector<double> acc;
    thread_local std::vector<char> seen;
    thread_local std::vector<Index> list;
    if (acc.size() < lb.size()) acc.resize(lb.size(), 0.0), seen.resize(lb.size(), 0);
    list.clear();
    for (size_t k = 0; k < c.idx.size(); ++k) {
      const Index j = c.idx[k];
      if (!seen[j]) seen[j] = 1, list.push_back(j);
      acc[j] += c.val[k];
    }
    double maxabs = 0.0;
    for (Index j : list) {
      if (lb[j] != ub[j]) maxabs = std::max(maxabs, std::abs(acc[j]));
    }
    // Slightly below the exact threshold so that summation order differences cannot reject a cut
    // that the exact path below would keep.
    const double tiny = maxabs * 1e-6 * (1.0 - 1e-6);
    Index cnt = 0;
    for (Index j : list) {
      cnt += lb[j] != ub[j] && std::abs(acc[j]) >= tiny;
      acc[j] = 0.0, seen[j] = 0;
    }
    if (cnt > max_nnz) return false;
  }
  // Sort and merge duplicates.
  std::vector<Index> order(c.idx.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](Index a, Index b) { return c.idx[a] < c.idx[b]; });
  std::vector<Index> idx;
  std::vector<double> val;
  idx.reserve(order.size());
  val.reserve(order.size());
  for (Index k : order) {
    if (!std::isfinite(c.val[k])) return false;
    if (!idx.empty() && idx.back() == c.idx[k]) {
      val.back() += c.val[k];
    } else {
      idx.push_back(c.idx[k]);
      val.push_back(c.val[k]);
    }
  }
  // Move fixed variables and exact zeros out.
  double rhs = c.rhs, maxabs = 0.0;
  size_t w = 0;
  for (size_t k = 0; k < idx.size(); ++k) {
    const Index j = idx[k];
    if (val[k] == 0.0) continue;
    if (lb[j] == ub[j]) {
      rhs -= val[k] * lb[j];
      continue;
    }
    maxabs = std::max(maxabs, std::abs(val[k]));
    idx[w] = j, val[w] = val[k], ++w;
  }
  idx.resize(w), val.resize(w);
  if (idx.empty() || maxabs < 1e-9) return false;
  // Remove coefficients below maxabs * 1e-6 (bounds dynamism), relaxing the rhs:
  // a x_j >= a lb_j for a > 0 and >= a ub_j for a < 0.
  const double tiny = maxabs * 1e-6;
  w = 0;
  for (size_t k = 0; k < idx.size(); ++k) {
    const Index j = idx[k];
    const double a = val[k];
    if (std::abs(a) < tiny) {
      const double b = a > 0 ? lb[j] : ub[j];
      if (!is_finite(b)) return false;
      rhs -= a * b;
      continue;
    }
    idx[w] = j, val[w] = a, ++w;
  }
  idx.resize(w), val.resize(w);
  if (static_cast<Index>(idx.size()) > max_nnz) return false;
  // Scale by a power of two (exact) so that max |a| is in [0.5, 1).
  int e = 0;
  std::frexp(maxabs, &e);
  for (double& v : val) v = std::ldexp(v, -e);
  rhs = std::ldexp(rhs, -e);
  if (!std::isfinite(rhs) || std::abs(rhs) > 1e9) return false;
  // Safety margin against rounding errors in the derivation.
  double sum = std::abs(rhs);
  for (size_t k = 0; k < idx.size(); ++k) {
    const double b = std::max(std::abs(lb[idx[k]]), std::abs(ub[idx[k]]));
    if (is_finite(b)) sum += std::abs(val[k]) * std::min(b, 1e6);
  }
  rhs += 1e-12 * sum + 1e-10;
  c.idx = std::move(idx);
  c.val = std::move(val);
  c.rhs = rhs;
  return true;
}

bool CutPool::add(Cut c, const double* x) {
  if (!clean_cut(c, p_.col_lb, p_.col_ub, opt_.max_nnz)) return false;
  double norm = 0.0, objpar = 0.0, cnorm = 0.0;
  for (size_t k = 0; k < c.idx.size(); ++k) {
    norm += c.val[k] * c.val[k];
    objpar += c.val[k] * p_.cost[c.idx[k]];
  }
  if (cnorm_ < 0.0) {
    cnorm_ = 0.0;  // once per pool (was recomputed for every candidate: O(n) each)
    for (double v : p_.cost) cnorm_ += v * v;
  }
  cnorm = cnorm_;
  norm = std::sqrt(norm);
  const double viol = c.activity(x) - c.rhs;
  if (viol < opt_.min_violation * std::max(1.0, std::abs(c.rhs)) || norm <= 0.0) return false;
  c.efficacy = viol / norm;
  if (c.efficacy < opt_.min_efficacy) return false;
  objpar = cnorm > 0 ? std::abs(objpar) / (norm * std::sqrt(cnorm)) : 0.0;
  c.score = c.efficacy * (1.0 + 0.1 * objpar);
  c.objpar = objpar;
  Index nint = 0;
  for (Index j : c.idx) nint += p_.is_int[j];
  c.intsupport = static_cast<double>(nint) / static_cast<double>(c.idx.size());
  cuts_.push_back(std::move(c));
  return true;
}

namespace {

double norm2(const Cut& c) {
  double s = 0.0;
  for (double v : c.val) s += v * v;
  return std::sqrt(s);
}

}  // namespace

std::vector<Cut> CutPool::select() {
  double good = -kInf;
  if (opt_.hybrid && !cuts_.empty()) {
    double max_eff = 0.0;
    for (const Cut& c : cuts_) max_eff = std::max(max_eff, c.efficacy);
    double best = 0.0;
    for (Cut& c : cuts_) {
      c.score = c.efficacy / max_eff + opt_.obj_weight * c.objpar + opt_.int_weight * c.intsupport;
      best = std::max(best, c.score);
    }
    good = opt_.good_score * best;
  }
  std::sort(cuts_.begin(), cuts_.end(), [](const Cut& a, const Cut& b) { return a.score > b.score; });
  std::vector<Cut> sel;
  std::vector<double> norms;
  // Parallelism |a.b| / (|a| |b|) to the selected cuts, with the dot products accumulated over
  // column occurrence lists of the selected cuts (cost: shared columns only). The lists are
  // contiguous per column: a linked list over one shared array was dominated by cache misses
  // (h80x6320d: ~2/3 of the root cut loop with ~800 candidates of a few hundred nonzeros).
  struct Occ {
    int cut;
    double val;
  };
  std::vector<std::vector<Occ>> col_occ(cuts_.empty() ? 0 : p_.num_cols());
  std::vector<double> dots;
  std::vector<int> touched;
  for (Cut& c : cuts_) {
    if (static_cast<int>(sel.size()) >= opt_.max_cuts) break;
    const double nc = norm2(c);
    for (size_t k = 0; k < c.idx.size(); ++k) {
      const double v = c.val[k];
      for (const Occ& o : col_occ[c.idx[k]]) {
        const int s = o.cut;
        if (dots[s] == 0.0) touched.push_back(s);
        dots[s] += v * o.val;
        if (dots[s] == 0.0) dots[s] = 1e-300;  // keep it marked as touched
      }
    }
    bool ok = true;
    const double maxpar = c.score >= good ? opt_.max_parallelism : opt_.max_parallelism_weak;
    for (int s : touched) {
      ok = ok && std::abs(dots[s]) / (nc * norms[s]) <= maxpar;
      dots[s] = 0.0;
    }
    touched.clear();
    if (!ok) continue;
    const int id = static_cast<int>(sel.size());
    for (size_t k = 0; k < c.idx.size(); ++k) col_occ[c.idx[k]].push_back({id, c.val[k]});
    sel.push_back(std::move(c));
    norms.push_back(nc);
    dots.push_back(0.0);
  }
  cuts_.clear();
  return sel;
}

}  // namespace milp
