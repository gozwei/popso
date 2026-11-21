#include "milp/search/conflict.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <queue>

#include "milp/cuts/cut.hpp"

namespace milp {

namespace {
constexpr double kBigBound = 1e12;  // larger bounds are treated as infinite

double bound_or_inf(double b) { return std::abs(b) > kBigBound ? (b > 0 ? kInf : -kInf) : b; }

// Tolerance of a conflict: points violating it by at most this much are kept
// (solutions are accepted with small row violations).
double conflict_tol(double rhs) { return 1e-6 * (1.0 + std::abs(rhs)); }

// Minimal activity of c over [lb, ub]; ninf counts infinite contributions.
double min_activity(const Conflict& c, const std::vector<double>& lb, const std::vector<double>& ub, int* ninf,
                    size_t* inf_pos) {
  double act = 0.0;
  *ninf = 0;
  for (size_t k = 0; k < c.idx.size(); ++k) {
    const double a = c.val[k];
    const double b = bound_or_inf(a > 0 ? lb[c.idx[k]] : ub[c.idx[k]]);
    if (!is_finite(b)) {
      ++*ninf;
      *inf_pos = k;
    } else {
      act += a * b;
    }
  }
  return act;
}
}  // namespace

bool derive_conflict(const DualSimplex& lp, const std::vector<double>& mu, double lambda, const std::vector<double>& cost,
                     double cutoff, const std::vector<char>& is_int, const std::vector<double>& glb,
                     const std::vector<double>& gub, const ConflictOptions& opt, Conflict& out) {
  const Index n = lp.num_cols(), m = lp.num_rows();
  if (static_cast<Index>(mu.size()) != m) return false;
  std::vector<double> g(n, 0.0);
  double rhs = 0.0;
  if (lambda > 0.0) {
    if (!is_finite(cutoff)) return false;
    for (Index j = 0; j < n; ++j) g[j] = lambda * cost[j];
    rhs = lambda * cutoff;
  }
  double maxmu = 0.0;
  for (double v : mu) maxmu = std::max(maxmu, std::abs(v));
  if (!(maxmu > 0.0) || !std::isfinite(maxmu)) return false;
  const SparseMatrix& R = lp.scaled_row_matrix();  // entry * row_unscale(row): the model's coefficient, exactly
  for (Index i = 0; i < m; ++i) {
    const double u = mu[i];
    if (std::abs(u) <= 1e-12 * maxmu) continue;  // dropping a multiplier keeps the aggregation valid
    // Row i: row_lb <= a_i x <= row_ub; the logical s_i = -a_i x has bounds [-row_ub, -row_lb].
    const double b = u > 0 ? -lp.var_lower(n + i) : -lp.var_upper(n + i);
    if (!is_finite(b) || std::abs(b) > kBigBound) continue;
    rhs += u * b;
    const double us = lp.row_unscale(i);
    for (Index k = R.begin(i); k < R.end(i); ++k) g[R.index[k]] += u * (R.value[k] * us);
  }
  double gmax = 0.0;
  for (double v : g) gmax = std::max(gmax, std::abs(v));
  if (!(gmax > 0.0) || !std::isfinite(gmax) || !std::isfinite(rhs)) return false;
  // Variables whose bound relevant for the proof is the same locally and
  // globally can be replaced by it (a x_j >= a gb) without weakening the proof
  // at this node: always for continuous ones, for integers if the proof would
  // be too dense otherwise (they are kept when possible to propagate them).
  Cut c;
  for (int pass = 0; pass < 2; ++pass) {
    const bool relax_ints = pass == 1 || opt.relax_ints;
    c.idx.clear();
    c.val.clear();
    double r = rhs;
    for (Index j = 0; j < n; ++j) {
      const double a = g[j];
      if (a == 0.0) continue;
      const double gb = bound_or_inf(a > 0 ? glb[j] : gub[j]);
      if (!is_int[j] || relax_ints) {
        const double lbnd = a > 0 ? lp.col_lower(j) : lp.col_upper(j);
        if (is_finite(gb) && lbnd == (a > 0 ? glb[j] : gub[j])) {
          r -= a * gb;
          continue;
        }
      }
      if (std::abs(a) <= 1e-11 * gmax && !is_finite(gb)) continue;  // round-off on a free column
      c.idx.push_back(j);
      c.val.push_back(a);
    }
    c.rhs = r;
    if (relax_ints || static_cast<Index>(c.idx.size()) <= opt.max_nnz) break;
  }
  if (c.idx.empty()) {
    // All variables substituted: the proof is a contradiction 0 <= rhs < 0
    // under global bounds (only possible if the node's subtree is globally infeasible).
    return false;
  }
  if (!clean_cut(c, glb, gub, opt.max_nnz)) return false;
  out.idx = std::move(c.idx);
  out.val = std::move(c.val);
  out.rhs = c.rhs;
  out.cutoff = lambda > 0.0 ? cutoff : kInf;
  // It must prove the node infeasible under the node's bounds.
  double act = 0.0;
  for (size_t k = 0; k < out.idx.size(); ++k) {
    const Index j = out.idx[k];
    const double b = bound_or_inf(out.val[k] > 0 ? lp.col_lower(j) : lp.col_upper(j));
    if (!is_finite(b)) return false;
    act += out.val[k] * b;
  }
  return act > out.rhs + conflict_tol(out.rhs) + opt.min_violation * (1.0 + std::abs(out.rhs));
}

bool infeasibility_conflict(const DualSimplex& lp, const std::vector<double>& y, const std::vector<double>& cost,
                            const std::vector<char>& is_int, const std::vector<double>& glb,
                            const std::vector<double>& gub, const ConflictOptions& opt, Conflict& out) {
  if (derive_conflict(lp, y, 0.0, cost, kInf, is_int, glb, gub, opt, out)) return true;
  std::vector<double> ny(y.size());
  for (size_t i = 0; i < y.size(); ++i) ny[i] = -y[i];
  return derive_conflict(lp, ny, 0.0, cost, kInf, is_int, glb, gub, opt, out);
}

bool Conflict::violated_by(const std::vector<double>& x, double tol) const {
  if (is_clause()) {
    for (size_t k = 0; k < idx.size(); ++k) {
      const double v = x[idx[k]];
      if (upper[k] ? v > val[k] + tol : v < val[k] - tol) return false;  // literal false: satisfied
    }
    return true;
  }
  double act = 0.0;
  for (size_t k = 0; k < idx.size(); ++k) act += val[k] * x[idx[k]];
  return act > rhs + tol * (1.0 + std::abs(rhs));
}

namespace {
// Decision level of a bound implied by conflict c through its literal / term
// `pos`: the highest level among the bounds of the other columns it relies on.
int reason_level(const Conflict& c, const BoundTrail& tr, size_t pos) {
  int lv = 0;
  for (size_t k = 0; k < c.idx.size(); ++k) {
    if (k == pos) continue;
    const bool up = c.is_clause() ? c.upper[k] != 0 : c.val[k] < 0;
    lv = std::max(lv, tr.bound_level(c.idx[k], up));
  }
  return lv;
}

// Clause propagation: if all literals but one hold, the last one is negated.
bool propagate_clause(const Conflict& c, std::vector<double>& lb, std::vector<double>& ub,
                      std::vector<Index>* tightened, bool* changed, BoundTrail* trail) {
  size_t open = c.idx.size(), nopen = 0;
  for (size_t k = 0; k < c.idx.size(); ++k) {
    const Index j = c.idx[k];
    const double v = c.val[k];
    if (c.upper[k] ? ub[j] <= v : lb[j] >= v) continue;  // holds
    if (c.upper[k] ? lb[j] > v : ub[j] < v) return true;               // false: satisfied
    open = k;
    if (++nopen > 1) return true;
  }
  if (nopen == 0) return false;
  const Index j = c.idx[open];
  const double v = c.val[open];
  if (c.upper[open]) {
    lb[j] = v + 1.0;  // not x <= v
    if (trail) trail->push(j, false, lb[j], BoundTrail::kConflict, -1, &c, reason_level(c, *trail, open));
  } else {
    ub[j] = v - 1.0;  // not x >= v
    if (trail) trail->push(j, true, ub[j], BoundTrail::kConflict, -1, &c, reason_level(c, *trail, open));
  }
  if (tightened) tightened->push_back(j);
  if (changed) *changed = true;
  return true;
}
}  // namespace

bool propagate_conflict(const Conflict& c, const std::vector<char>& is_int, std::vector<double>& lb,
                        std::vector<double>& ub, std::vector<Index>* tightened, bool* changed, BoundTrail* trail,
                        double idle_room) {
  if (c.is_clause()) return propagate_clause(c, lb, ub, tightened, changed, trail);
  int ninf = 0;
  size_t inf_pos = 0;
  const double minact = min_activity(c, lb, ub, &ninf, &inf_pos);
  const double tol = conflict_tol(c.rhs);
  if (ninf == 0 && minact > c.rhs + tol) return false;
  if (ninf >= 2) return true;
  // The slack covers the largest |a| (u - l) the columns can have (idle_room, from bounds that contain the current
  // ones, with a term for the rounding of a x at a bound): the loop below would move no bound, since each implied
  // bound lies at or beyond the column's own (the margin covers the rounding of the slack there).
  const bool known_idle =
      ninf == 0 && (c.rhs + tol) - minact - 1e-12 * (std::abs(c.rhs + tol) + std::abs(minact)) >= idle_room;
#ifndef MILP_PROP_CHECK
  if (known_idle) return true;
#endif
  for (size_t k = 0; k < c.idx.size(); ++k) {
    if (ninf == 1 && k != inf_pos) continue;
    const Index j = c.idx[k];
    if (!is_int[j]) continue;
    const double a = c.val[k];
    const double own = ninf == 1 ? 0.0 : a * (a > 0 ? lb[j] : ub[j]);
    // a x_j <= rhs + tol - (minact - own)
    const double slack = c.rhs + tol - (minact - own);
    const double v = slack / a;
    if (known_idle && (a > 0 ? std::floor(v + 1e-9 * (1.0 + std::abs(v))) < ub[j] - 0.5
                             : std::ceil(v - 1e-9 * (1.0 + std::abs(v))) > lb[j] + 0.5)) {
      std::abort();  // MILP_PROP_CHECK only
    }
    if (a > 0) {
      const double nu = std::floor(v + 1e-9 * (1.0 + std::abs(v)));
      if (nu < ub[j] - 0.5) {
        if (nu < lb[j] - 0.5) return false;
        ub[j] = nu;
        if (trail) trail->push(j, true, nu, BoundTrail::kConflict, -1, &c, reason_level(c, *trail, k));
        if (tightened) tightened->push_back(j);
        if (changed) *changed = true;
      }
    } else {
      const double nl = std::ceil(v - 1e-9 * (1.0 + std::abs(v)));
      if (nl > lb[j] + 0.5) {
        if (nl > ub[j] + 0.5) return false;
        lb[j] = nl;
        if (trail) trail->push(j, false, nl, BoundTrail::kConflict, -1, &c, reason_level(c, *trail, k));
        if (tightened) tightened->push_back(j);
        if (changed) *changed = true;
      }
    }
  }
  return true;
}

namespace {
// Which local bound of the k-th column can make the conflict propagate: 0 the lower, 1 the upper.
int trigger_bit(const Conflict& c, size_t k) { return c.is_clause() ? (c.upper[k] ? 1 : 0) : (c.val[k] < 0 ? 1 : 0); }
}  // namespace

void ConflictPool::add_occ(Index id) {
  const Entry& e = entries_[id];
  const Conflict& c = *e.c;
  if (c.is_clause()) {
    // Clauses are watched by two literals that do not hold (one for a unit clause).
    wocc_[c.idx[e.w0]].push_back(2 * id + trigger_bit(c, e.w0));
    if (e.w1 != e.w0) wocc_[c.idx[e.w1]].push_back(2 * id + trigger_bit(c, e.w1));
    return;
  }
  // Per column one list of the conflicts triggered by its lower bound and one of those triggered by its upper
  // bound, each in entry order (entries are added in increasing order). Two lists, so an addition appends (one
  // list with the first group in front needed an insertion in the middle for it: 2.5% of the thread time on
  // markshare_4_0, which learns a conflict at most nodes).
  for (size_t k = 0; k < c.idx.size(); ++k) {
    occ_[2 * static_cast<size_t>(c.idx[k]) + static_cast<size_t>(trigger_bit(c, k))].push_back(id);
  }
}

void ConflictPool::add(std::shared_ptr<const Conflict> c, long long now) {
  const Index id = static_cast<Index>(entries_.size());
  const int w1 = c->idx.size() > 1 ? 1 : 0;
  entries_.push_back({std::move(c), now, 0, w1});
  Entry& e = entries_.back();
  if (e.c->is_clause()) {
    e.wk0 = 2 * e.c->idx[0] + trigger_bit(*e.c, 0);
    e.wk1 = 2 * e.c->idx[w1] + trigger_bit(*e.c, w1);
  }
  add_occ(id);
  queued_.push_back(0);
  ++alive_;
  if (alive_ > capacity_) shrink();
}

void ConflictPool::shrink() {
  // Keep the 3/4 of the capacity that propagated most recently (newer first on ties).
  std::vector<Index> ids;
  for (Index i = 0; i < static_cast<Index>(entries_.size()); ++i) {
    if (entries_[i].c) ids.push_back(i);
  }
  std::sort(ids.begin(), ids.end(), [&](Index a, Index b) {
    if (entries_[a].last_used != entries_[b].last_used) return entries_[a].last_used > entries_[b].last_used;
    return a > b;
  });
  ids.resize(std::min(ids.size(), capacity_ * 3 / 4 + 1));
  std::sort(ids.begin(), ids.end());
  std::vector<Entry> kept;
  kept.reserve(ids.size());
  for (Index i : ids) kept.push_back(std::move(entries_[i]));
  entries_ = std::move(kept);
  for (auto& o : occ_) o.clear();
  for (auto& o : wocc_) o.clear();
  for (Index id = 0; id < static_cast<Index>(entries_.size()); ++id) add_occ(id);
  queued_.assign(entries_.size(), 0);
  alive_ = entries_.size();
}

bool ConflictPool::watches(const Entry& e, Index j, int bit) const {
  const Index key = 2 * j + bit;
  return e.wk0 == key || e.wk1 == key;
}

bool ConflictPool::propagate_watched(Index id, std::vector<double>& lb, std::vector<double>& ub,
                                     std::vector<Index>& tightened, bool& changed, BoundTrail* trail, long long& work) {
  Entry& e = entries_[id];
  const Conflict& c = *e.c;
  const int n = static_cast<int>(c.idx.size());
  auto holds = [&](int k) { return c.upper[k] ? ub[c.idx[k]] <= c.val[k] : lb[c.idx[k]] >= c.val[k]; };
  work += 4;
  if (n > 1) {
    // Move watches away from literals that hold.
    for (int* w : {&e.w0, &e.w1}) {
      if (!holds(*w)) continue;
      work += n;
      for (int k = 0; k < n; ++k) {
        if (k == e.w0 || k == e.w1 || holds(k)) continue;
        *w = k;
        const Index key = 2 * c.idx[k] + trigger_bit(c, k);
        (w == &e.w0 ? e.wk0 : e.wk1) = key;
        wocc_[c.idx[k]].push_back(2 * id + trigger_bit(c, k));
        break;
      }
    }
  }
  const bool h0 = holds(e.w0), h1 = holds(e.w1);
  if (h0 && h1) return false;  // every literal holds
  if (!h0 && !h1) return true;
  const int k = h0 ? e.w1 : e.w0;  // the only literal that does not hold: negate it
  const Index j = c.idx[k];
  const double v = c.val[k];
  if (c.upper[k]) {
    if (lb[j] > v) return true;  // already false
    lb[j] = v + 1.0;
    if (trail) trail->push(j, false, lb[j], BoundTrail::kConflict, -1, &c, reason_level(c, *trail, k));
  } else {
    if (ub[j] < v) return true;
    ub[j] = v - 1.0;
    if (trail) trail->push(j, true, ub[j], BoundTrail::kConflict, -1, &c, reason_level(c, *trail, k));
  }
  tightened.push_back(j);
  changed = true;
  return true;
}

bool ConflictPool::propagate(const std::vector<char>& is_int, const std::vector<double>& glb,
                             const std::vector<double>& gub, std::vector<double>& lb, std::vector<double>& ub,
                             const std::vector<Index>& seeds, std::vector<Index>& tightened, long long now,
                             BoundTrail* trail) {
  last_work_ = last_clause_work_ = last_clause_hits_ = 0;
  if (alive_ == 0 || (work_limit < 0 && clause_work_limit < 0)) return true;
  // Linear conflicts and clauses have separate work limits (clauses are cheap
  // to propagate with watched literals); work_limit < 0 disables the linear ones.
  const long long limit = work_limit > 0 ? work_limit : work_limit < 0 ? -1 : 200000;
  const long long climit = clause_work_limit > 0 ? clause_work_limit : clause_work_limit < 0 ? -1 : 200000;
  long long& work = last_work_;
  long long& cwork = last_clause_work_;
  queue_.clear();
  // A conflict can only propagate through column j if the bound of j that
  // enters its minimal activity is tighter than globally.
  auto enqueue_col = [&](Index j) {
    const bool lb_tight = lb[j] > glb[j], ub_tight = ub[j] < gub[j];
    const bool lin = work <= limit, cl = cwork <= climit;
    if ((!lb_tight && !ub_tight) || (!lin && !cl)) return;
    if (lin) {
      // Only the list of a tight bound is read; with both tight, the two lists merged in entry order.
      const std::vector<Index>& o0 = occ_[2 * static_cast<size_t>(j)], &o1 = occ_[2 * static_cast<size_t>(j) + 1];
      const size_t n0 = o0.size(), n1 = o1.size(), n = n0 + n1;
      auto push = [&](Index id) {
        if (!queued_[id]) queued_[id] = 1, queue_.push_back(id);
      };
      if (lb_tight && ub_tight) {
        size_t a = 0, b = 0;
        while (a < n0 && b < n1) push(o0[a] <= o1[b] ? o0[a++] : o1[b++]);
        while (a < n0) push(o0[a++]);
        while (b < n1) push(o1[b++]);
      } else if (lb_tight) {
        for (size_t a = 0; a < n0; ++a) push(o0[a]);
      } else {
        for (size_t b = 0; b < n1; ++b) push(o1[b]);
      }
      work += static_cast<long long>(n) / 4;
    }
    if (cl) {
      std::vector<Index>& o = wocc_[j];
      size_t keep = 0;
      for (Index e : o) {
        const Index id = e >> 1;
        if (!watches(entries_[id], j, static_cast<int>(e & 1))) continue;  // moved watch
        o[keep++] = e;
        if (!queued_[id] && ((e & 1) ? ub_tight : lb_tight)) queued_[id] = 1, queue_.push_back(id);
      }
      cwork += 2 * static_cast<long long>(o.size());  // a watch check is a cache miss or two
      o.resize(keep);
    }
  };
  for (Index j : seeds) enqueue_col(j);
  bool feasible = true;
  for (size_t head = 0; head < queue_.size(); ++head) {
    const Index id = queue_[head];
    queued_[id] = 0;
    if (!feasible) continue;  // drain the queue flags
    Entry& e = entries_[id];
    const size_t before = tightened.size();
    bool changed = false;
    bool ok;
    if (e.c->is_clause()) {
      if (cwork > climit) continue;
      ok = propagate_watched(id, lb, ub, tightened, changed, trail, cwork);
      if (!ok || changed) ++last_clause_hits_;
    } else {
      if (work > limit) continue;
      work += static_cast<long long>(e.c->idx.size()) + 4;
      if (e.idle_room < 0.0) {
        // Once per conflict, from the global bounds (they only tighten, and the local ones lie within them) of
        // its integer columns (the others are not tightened).
        double span = 0.0, own = 0.0;
        for (size_t k = 0; k < e.c->idx.size(); ++k) {
          if (!is_int[e.c->idx[k]]) continue;
          const double gl = glb[e.c->idx[k]], gu = gub[e.c->idx[k]], a = std::abs(e.c->val[k]);
          if (std::abs(gl) > kBigBound || std::abs(gu) > kBigBound) {
            span = kInf;
            break;
          }
          span = std::max(span, a * (gu - gl));
          own = std::max(own, a * std::max(std::abs(gl), std::abs(gu)));
        }
        e.idle_room = span + 1e-12 * own;
      }
      ok = propagate_conflict(*e.c, is_int, lb, ub, &tightened, &changed, trail, e.idle_room);
    }
    if (trail && (!ok || changed)) trail->hold(e.c);
    if (!ok) {
      e.last_used = now;
      culprit_ = e.c.get();
      if (trail) trail->fail_conflict(culprit_);
      feasible = false;
      continue;
    }
    if (changed) {
      e.last_used = now;
      for (size_t t = before; t < tightened.size(); ++t) enqueue_col(tightened[t]);
    }
  }
  return feasible;
}

namespace {

// Implication graph walk for analyze_propagation_conflict. Literals are bounds
// (col, upper, val): x_col <= val if upper, else x_col >= val. A literal is
// identified with its source, the earliest trail entry that implies it (none
// if the global bound does); several requirements on the same source keep the
// tightest value.
class PropAnalyzer {
 public:
  PropAnalyzer(const BoundTrail& tr, const SparseMatrix& rows, const std::vector<double>& row_lb,
               const std::vector<double>& row_ub, const std::vector<char>& is_int, const std::vector<double>& glb,
               const std::vector<double>& gub)
      : tr_(tr), rows_(rows), row_lb_(row_lb), row_ub_(row_ub), is_int_(is_int), glb_(glb), gub_(gub) {
    // The work arrays live in a per-thread store between calls (an analysis runs at every node closed by
    // propagation: their allocation was a few percent of the thread time on models with very cheap nodes).
    Scratch& sc = scratch();
    need_.swap(sc.need), state_.swap(sc.state), touched_.swap(sc.touched), lits_.swap(sc.lits);
    lvl_cnt_.swap(sc.lvl_cnt), terms_.swap(sc.terms), cands_.swap(sc.cands), order_.swap(sc.order);
    touched_.clear(), lits_.clear(), terms_.clear(), cands_.clear(), order_.clear();
    need_.assign(tr.e.size(), 0.0);
    state_.assign(tr.e.size(), kFree);
  }
  ~PropAnalyzer() {
    Scratch& sc = scratch();
    need_.swap(sc.need), state_.swap(sc.state), touched_.swap(sc.touched), lits_.swap(sc.lits);
    lvl_cnt_.swap(sc.lvl_cnt), terms_.swap(sc.terms), cands_.swap(sc.cands), order_.swap(sc.order);
  }
  PropAnalyzer(const PropAnalyzer&) = delete;
  PropAnalyzer& operator=(const PropAnalyzer&) = delete;

  // Explains the failure, then derives up to max_clauses clauses: the first
  // unique implication point (UIP) cut of the last decision level, then the
  // UIP cuts of the next lower levels (each with the deeper levels resolved).
  int run(Index max_lits, int max_clauses, std::vector<Conflict>& out) {
    const int end = static_cast<int>(tr_.e.size());
    int max_level = tr_.level;
    for (const BoundTrail::Entry& e : tr_.e) max_level = std::max(max_level, e.level);
    lvl_cnt_.assign(static_cast<size_t>(max_level) + 1, 0);
    bool ok = true;
    switch (tr_.fail) {
      case BoundTrail::kFailRow:
        ok = explain_row(tr_.fail_row, tr_.fail_upper, -1, false, 0.0, end);
        break;
      case BoundTrail::kFailConflict:
        ok = explain_conflict(*tr_.fail_conf, -1, false, 0.0, end);
        break;
      case BoundTrail::kFailCrossing: {
        const Index j = tr_.fail_col;
        int s;
        const double l = bound_at(j, false, end, &s), u = bound_at(j, true, end, &s);
        if (!(l > u)) return 0;
        ok = add_lit(j, true, u, end) && add_lit(j, false, is_int_[j] ? u + 1.0 : l, end);
        break;
      }
      default:
        return 0;
    }
    if (!ok) return 0;
    int found = 0, level = std::numeric_limits<int>::max();
    bool fresh = true;  // literals changed since the last clause
    for (bool first = true; found < max_clauses; first = false) {
      if (!first) {
        // Resolve the levels of the previous clause completely (decisions stay).
        while (peek() && tr_.e[top()].level >= level) {
          fresh |= tr_.e[top()].kind != BoundTrail::kDecision;
          if (!pop_resolve()) return found;
        }
      }
      if (!peek()) break;
      level = tr_.e[top()].level;
      while (peek() && tr_.e[top()].level == level && lvl_cnt_[level] > 1 &&
             tr_.e[top()].kind != BoundTrail::kDecision) {
        if (!pop_resolve()) return found;
        fresh = true;
      }
      if (!fresh) continue;
      fresh = false;
      // Literals on continuous columns cannot be kept in a clause: replace them by their reasons.
      if (resolve_cont_ && !resolve_continuous()) return found;
      if (first && minimize_) minimize(level);
      Conflict c;
      if (emit(max_lits, c)) out.push_back(std::move(c)), ++found;
      else if (first_only_) break;  // too long at the deepest level: the lower levels are not tried
    }
    return found;
  }

 private:
  enum State : char { kFree, kActive, kResolved, kKept };

  // Drops stale heap entries; true if an active literal is on top.
  bool peek() {
    while (!heap_.empty()) {
      if (state_[top()] == kActive) return true;
      heap_.pop();
    }
    return false;
  }
  int top() const { return heap_.top().second; }
  // Replaces active implied literals on continuous columns by their reasons (latest first).
  bool resolve_continuous() {
    for (;;) {
      int best = -1;
      for (int src : touched_) {
        const BoundTrail::Entry& e = tr_.e[src];
        if (state_[src] == kActive && !is_int_[e.col] && e.kind != BoundTrail::kDecision) best = std::max(best, src);
      }
      if (best < 0) return true;
      if (!resolve(best)) return false;
    }
  }
 public:
  bool resolve_cont_ = false;
  bool first_only_ = false;  // stop when the clause of the deepest level cannot be emitted
  bool minimize_ = false;    // drop literals of the first clause that its other literals imply
  long long min_work_ = 0, min_dropped_ = 0;
  long long explain_work_ = 0;  // row and clause entries read by the explanations
 private:
  // Clause minimization: an implied literal below the top level is dropped when its reason holds under the
  // clause's other literals alone (those before it on the trail, with the values the clause states). A dropped
  // literal is implied by the remaining ones (by induction along the trail), so the shorter clause is valid.
  void minimize(int top_level) {
    order_.clear();
    for (int src : touched_) {
      const BoundTrail::Entry& e = tr_.e[src];
      if (state_[src] == kActive && e.kind != BoundTrail::kDecision && e.level < top_level) order_.push_back(src);
    }
    std::sort(order_.begin(), order_.end(), std::greater<int>());
    const long long limit = 20000 + 50 * static_cast<long long>(touched_.size());
    for (int src : order_) {
      if (min_work_ > limit) break;
      if (!redundant(src)) continue;
      state_[src] = kResolved;
      --lvl_cnt_[tr_.e[src].level];
      ++min_dropped_;
    }
  }
  // The tightest value the clause's literals on this bound state, among those before trail position t.
  bool clause_bound(Index j, bool up, int t, double* n) {
    int k = tr_.last[2 * static_cast<size_t>(j) + up];
    while (k >= t) k = tr_.e[k].prev;
    bool found = false;
    for (; k >= 0; k = tr_.e[k].prev) {
      ++min_work_;
      if (state_[k] != kActive && state_[k] != kKept) continue;
      if (!found || implies(up, need_[k], *n)) *n = need_[k];
      found = true;
    }
    return found;
  }
  bool redundant(int src) {
    const BoundTrail::Entry& e = tr_.e[src];
    const bool up = e.upper != 0;
    const double v = need_[src];
    if (!is_int_[e.col]) return false;
    if (e.kind != BoundTrail::kRow) {
      const Conflict& c = *e.conf;
      if (!c.is_clause()) return false;
      min_work_ += static_cast<long long>(c.idx.size());
      for (size_t k = 0; k < c.idx.size(); ++k) {
        const bool lup = c.upper[k] != 0;
        if (c.idx[k] == e.col && lup == !up) continue;
        if (implies(lup, global(c.idx[k], lup), c.val[k])) continue;
        double n;
        if (!clause_bound(c.idx[k], lup, src, &n) || !implies(lup, n, c.val[k])) return false;
      }
      return true;
    }
    // Row reason: sum_k g_k x_k <= R with every other column at the bound the clause (else the model) gives
    // must still exclude the next value of this column.
    double a = 0.0;
    for (Index k = rows_.begin(e.row); k < rows_.end(e.row); ++k) {
      if (rows_.index[k] == e.col) a = rows_.value[k];
    }
    if (a == 0.0) return false;
    const bool upper_side = (a > 0) == up;
    const double sgn = upper_side ? 1.0 : -1.0;
    const double R = upper_side ? row_ub_[e.row] : -row_lb_[e.row];
    if (!is_finite(R) || std::abs(R) > kBigBound) return false;
    min_work_ += rows_.end(e.row) - rows_.begin(e.row);
    double M = 0.0, gj = 0.0;
    for (Index k = rows_.begin(e.row); k < rows_.end(e.row); ++k) {
      const Index col = rows_.index[k];
      const double g = sgn * rows_.value[k];
      if (col == e.col) {
        gj += g;
        continue;
      }
      const bool kup = g < 0;
      double b = global(col, kup), n;
      if (clause_bound(col, kup, src, &n) && implies(kup, n, b)) b = n;
      if (!is_finite(b)) return false;
      M += g * b;
    }
    if (up ? !(gj > 0) : !(gj < 0)) return false;
    const double margin = gj * (up ? v + 1.0 : v - 1.0) - (R - M);
    return margin >= 1e-6 * (1.0 + std::abs(R) + std::abs(M));
  }
  std::vector<int> order_;
  // Pops the latest literal: a decision is kept, an implied bound is replaced by its reason.
  bool pop_resolve() {
    const int src = top();
    heap_.pop();
    return resolve(src);
  }
  bool resolve(int src) {
    const BoundTrail::Entry& e = tr_.e[src];
    if (e.kind == BoundTrail::kDecision) {
      state_[src] = kKept;
      return true;
    }
    if (++work_ > 100000) return false;
    state_[src] = kResolved;
    --lvl_cnt_[e.level];
    if (e.kind == BoundTrail::kRow) {
      // The side of the row that implies this bound: row_ub for x <= with a > 0 or x >= with a < 0.
      double a = 0.0;
      for (Index k = rows_.begin(e.row); k < rows_.end(e.row); ++k) {
        if (rows_.index[k] == e.col) a = rows_.value[k];
      }
      if (a == 0.0) return false;
      return explain_row(e.row, (a > 0) == (e.upper != 0), e.col, e.upper != 0, need_[src], src);
    }
    return explain_conflict(*e.conf, e.col, e.upper != 0, need_[src], src);
  }
  // The clause of the current literals (active or kept); of several literals
  // on the same bound only the tightest is needed.
  bool emit(Index max_lits, Conflict& out) {
    lits_.clear();
    for (int src : touched_) {
      if (state_[src] != kActive && state_[src] != kKept) continue;
      const BoundTrail::Entry& e = tr_.e[src];
      lits_.push_back({2 * e.col + e.upper, need_[src]});
    }
    std::sort(lits_.begin(), lits_.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
      if (a.first != b.first) return a.first < b.first;
      return (a.first & 1) ? a.second < b.second : a.second > b.second;  // tightest first
    });
    out.idx.clear(), out.val.clear(), out.upper.clear();
    for (size_t k = 0; k < lits_.size(); ++k) {
      if (k > 0 && lits_[k].first == lits_[k - 1].first) continue;
      const Index j = lits_[k].first >> 1;
      if (!is_int_[j] || static_cast<Index>(out.idx.size()) >= max_lits) return false;
      out.idx.push_back(j);
      out.val.push_back(lits_[k].second);
      out.upper.push_back(static_cast<char>(lits_[k].first & 1));
    }
    out.rhs = 0.0;
    out.cutoff = cutoff_;
    return !out.idx.empty();
  }

  double global(Index j, bool up) const {
    const double b = up ? gub_[j] : glb_[j];
    return std::abs(b) > kBigBound ? (b > 0 ? kInf : -kInf) : b;
  }
  // Value of a bound before trail position t; *src: entry that set it (-1: global).
  double bound_at(Index j, bool up, int t, int* src) const {
    int k = tr_.last[2 * static_cast<size_t>(j) + up];
    while (k >= t) k = tr_.e[k].prev;
    *src = k;
    return k >= 0 ? tr_.e[k].val : global(j, up);
  }
  static bool implies(bool up, double b, double v) { return up ? b <= v : b >= v; }
  // Earliest entry implying the literal (-1: global; -2: does not hold before t).
  int source(Index j, bool up, double v, int t) const {
    if (implies(up, global(j, up), v)) return -1;
    int k = tr_.last[2 * static_cast<size_t>(j) + up];
    while (k >= t) k = tr_.e[k].prev;
    int best = -2;
    while (k >= 0 && implies(up, tr_.e[k].val, v)) best = k, k = tr_.e[k].prev;
    return best;
  }
  bool add_lit(Index j, bool up, double v, int t) {
    const int src = source(j, up, v, t);
    if (src == -2) return false;
    if (src == -1) return true;
    switch (state_[src]) {
      case kFree:
        touched_.push_back(src);
        need_[src] = v;
        state_[src] = kActive;
        ++lvl_cnt_[tr_.e[src].level];
        heap_.emplace(tr_.e[src].level, src);
        return true;
      case kResolved:
        return false;  // cannot happen: explanations only use entries before the explained one
      default:
        if (!implies(up, need_[src], v)) need_[src] = v;
        return true;
    }
  }

  // Explains, from sum_k g_k x_k <= R with bounds before trail position t,
  // either its infeasibility (skip < 0) or the bound x_skip <= v (up) or
  // x_skip >= v (!up). The bounds of the other columns are relaxed towards
  // their global bounds as long as the implication still holds.
  bool explain_linear(double R, Index skip, bool up, double v, int t) {
    double M = 0.0, gj = 0.0;
    cands_.clear();
    explain_work_ += static_cast<long long>(terms_.size());
    for (const auto& [k, g] : terms_) {
      if (k == skip) {
        gj += g;
        continue;
      }
      const bool kup = g < 0;
      int src;
      const double b = bound_at(k, kup, t, &src);
      if (!is_finite(b)) return false;
      M += g * b;
      if (src >= 0) cands_.push_back({src, k, kup, b, g});
    }
    double margin = 0.0;
    if (skip < 0) {
      margin = M - R;
    } else if (is_int_[skip]) {
      if (up ? !(gj > 0) : !(gj < 0)) return false;
      // The first excluded value must exceed what the row allows: g_j * next > R - M.
      margin = gj * (up ? v + 1.0 : v - 1.0) - (R - M);
    }
    const double tol = 1e-6 * (1.0 + std::abs(R) + std::abs(M));
    // The bounds must imply the failure / the bound (up to the propagators' tolerances).
    if ((skip < 0 || is_int_[skip]) && margin < -tol) return false;
    double budget = margin - tol;
    std::sort(cands_.begin(), cands_.end(), [](const Cand& a, const Cand& b) { return a.src > b.src; });
    for (const Cand& c : cands_) {
      double val = c.b;
      const double ag = std::abs(c.g);
      if (budget > 0.0) {
        const double gb = global(c.k, c.up);
        const double full = is_finite(gb) ? ag * std::abs(c.b - gb) : kInf;
        if (full <= budget) {
          budget -= full;
          continue;
        }
        if (is_int_[c.k]) {
          const double steps = std::floor(budget / ag);
          if (steps >= 1.0) {
            val = c.up ? c.b + steps : c.b - steps;
            budget -= steps * ag;
          }
        }
      }
      if (!add_lit(c.k, c.up, val, t)) return false;
    }
    return true;
  }
  bool explain_row(Index i, bool upper_side, Index skip, bool up, double v, int t) {
    terms_.clear();
    const double sgn = upper_side ? 1.0 : -1.0;
    const double R = upper_side ? row_ub_[i] : -row_lb_[i];
    if (!is_finite(R) || std::abs(R) > kBigBound) return false;
    for (Index k = rows_.begin(i); k < rows_.end(i); ++k) terms_.emplace_back(rows_.index[k], sgn * rows_.value[k]);
    return explain_linear(R, skip, up, v, t);
  }
  bool explain_conflict(const Conflict& c, Index skip, bool up, double v, int t) {
    cutoff_ = std::min(cutoff_, c.cutoff);
    if (c.is_clause()) {
      explain_work_ += static_cast<long long>(c.idx.size());
      // The bound on `skip` negates its literal; all other literals held.
      for (size_t k = 0; k < c.idx.size(); ++k) {
        if (c.idx[k] == skip && (c.upper[k] != 0) == !up) continue;
        if (!add_lit(c.idx[k], c.upper[k] != 0, c.val[k], t)) return false;
      }
      return true;
    }
    terms_.clear();
    for (size_t k = 0; k < c.idx.size(); ++k) terms_.emplace_back(c.idx[k], c.val[k]);
    return explain_linear(c.rhs + conflict_tol(c.rhs), skip, up, v, t);
  }

  struct Cand {
    int src;
    Index k;
    bool up;
    double b, g;
  };
  struct Scratch {
    std::vector<double> need;
    std::vector<char> state;
    std::vector<int> touched, lvl_cnt, order;
    std::vector<std::pair<int, double>> lits;
    std::vector<std::pair<Index, double>> terms;
    std::vector<Cand> cands;
  };
  static Scratch& scratch() {
    thread_local Scratch s;
    return s;
  }
  const BoundTrail& tr_;
  const SparseMatrix& rows_;
  const std::vector<double>&row_lb_, &row_ub_;
  const std::vector<char>& is_int_;
  const std::vector<double>&glb_, &gub_;
  std::vector<double> need_;  // per trail entry: the value needed of the literal it is the source of
  std::vector<char> state_;   // per trail entry: State
  std::vector<int> touched_;
  std::vector<std::pair<int, double>> lits_;
  std::vector<int> lvl_cnt_;  // active or kept literals per level
  long long work_ = 0;
  std::priority_queue<std::pair<int, int>> heap_;  // active literals (level, source entry), latest first
  std::vector<std::pair<Index, double>> terms_;
  std::vector<Cand> cands_;
  double cutoff_ = kInf;
};

}  // namespace

int analyze_propagation_conflict(const BoundTrail& trail, const SparseMatrix& rows, const std::vector<double>& row_lb,
                                 const std::vector<double>& row_ub, const std::vector<char>& is_int,
                                 const std::vector<double>& glb, const std::vector<double>& gub, Index max_lits,
                                 int max_clauses, std::vector<Conflict>& out, bool first_only, long long* stats,
                                 bool minimize) {
  if (!trail.usable()) return 0;
  PropAnalyzer a(trail, rows, row_lb, row_ub, is_int, glb, gub);
  static const bool resolve_cont = [] {
    const char* v = std::getenv("MILP_CONF_RESOLVE_CONT");
    return v ? std::atoi(v) != 0 : false;
  }();
  a.resolve_cont_ = resolve_cont;
  a.first_only_ = first_only;
  a.minimize_ = minimize;
  const int n = a.run(max_lits, max_clauses, out);
  if (stats) stats[0] += a.min_work_ + a.explain_work_, stats[1] += a.min_dropped_;
  return n;
}

}  // namespace milp
