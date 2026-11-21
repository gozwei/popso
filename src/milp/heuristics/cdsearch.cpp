#include "milp/heuristics/cdsearch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <random>

#include "milp/core/interrupt.hpp"
#include "milp/core/timer.hpp"
#include "milp/search/conflict.hpp"
#include "milp/search/trail.hpp"

namespace milp {

namespace {

constexpr double kFeasTol = 1e-6;
constexpr double kMaxBound = 1e9;  // integer bounds beyond this: out of scope

// Luby sequence 1 1 2 1 1 2 4 ... (i >= 1).
long long luby(long long i) {
  long long k = 1;
  while ((1LL << k) - 1 < i) ++k;
  while ((1LL << k) - 1 != i) {
    i -= (1LL << (k - 1)) - 1;
    k = 1;
    while ((1LL << k) - 1 < i) ++k;
  }
  return 1LL << (k - 1);
}

class Cds {
 public:
  Cds(const MipProblem& p, const CdsLimits& lim, CdsStats& st)
      : p_(p), lim_(lim), st_(st), rows_(p.A.transpose()), pool_(p.num_cols(), std::max<size_t>(lim.max_clauses, 100)) {}

  bool run(std::vector<double>& sol) {
    const Index n = p_.num_cols();
    lb_ = p_.col_lb, ub_ = p_.col_ub;
    tr_.init(n, static_cast<size_t>(1) << 40);
    tr_.chrono = true;
    pool_.work_limit = -1;  // only clauses are stored
    pool_.clause_work_limit = 20000000;
    // Root propagation of every row.
    const Index m = p_.num_rows();
    minact_.assign(static_cast<size_t>(m), 0.0), maxact_.assign(static_cast<size_t>(m), 0.0);
    span_.assign(static_cast<size_t>(m), 0.0);
    rq_.assign(static_cast<size_t>(m), 0);
    rlb_ = lb_, rub_ = ub_;
    recompute_activities();
    for (Index i = 0; i < m; ++i) rq_[i] = 1, rqueue_.push_back(i);
    if (!propagate()) {
      st_.infeasible = true;
      return false;
    }
    commit_root();
    // Activities: a small static part (longer columns first, random ties), then bumps.
    std::mt19937 rng(lim_.seed * 2654435761u + 12345u);
    act_.resize(static_cast<size_t>(n));
    phase_.resize(static_cast<size_t>(n));
    pos_.assign(static_cast<size_t>(n), -1);
    Index maxlen = 1;
    for (Index j = 0; j < n; ++j) maxlen = std::max(maxlen, p_.A.end(j) - p_.A.begin(j));
    for (Index j = 0; j < n; ++j) {
      const double len = static_cast<double>(p_.A.end(j) - p_.A.begin(j)) / static_cast<double>(maxlen);
      act_[j] = 1e-3 * len + 1e-6 * std::uniform_real_distribution<double>(0.0, 1.0)(rng);
      static const int ph = std::getenv("MILP_CDS_PHASE") ? std::atoi(std::getenv("MILP_CDS_PHASE")) : 0;
      const double dflt = ph == 1 ? rub_[j] : ph == 2 ? (rng() & 1 ? rub_[j] : rlb_[j]) : ph == 3 ? (p_.cost[j] < 0 ? rub_[j] : rlb_[j]) : rlb_[j];
      const double h = lim_.hint && static_cast<Index>(lim_.hint->size()) >= n ? std::round((*lim_.hint)[j]) : dflt;
      phase_[j] = std::clamp(h, rlb_[j], rub_[j]);
      if (rlb_[j] < rub_[j]) heap_push(j);
    }
    long long restart_no = 1, since_restart = 0;
    long long until = luby(restart_no) * lim_.restart_unit;
    Timer timer;
    for (long long step = 0;; ++step) {
      if (st_.effort >= lim_.effort) return false;
      if ((step & 63) == 0) {
        if (timer.elapsed() >= lim_.time_limit || (lim_.stop && lim_.stop->load(std::memory_order_relaxed)) ||
            interrupted()) {
          return false;
        }
      }
      if (st_.check_failed) return false;
      if (since_restart >= until && tr_.level > 0) {
        backtrack(0);
        commit_root();
        ++st_.restarts;
        since_restart = 0;
        until = luby(++restart_no) * lim_.restart_unit;
      }
      const Index j = pick();
      bool ok = true;
      if (j < 0) {
        // Every column is fixed: the point.
        if (p_.max_violation(lb_) <= kFeasTol) {
          sol = lb_;
          return true;
        }
        st_.effort += p_.A.nnz();
        if (tr_.level == 0) return false;  // the root itself (propagation at its work limit): give up
        tr_.fail = BoundTrail::kNone;      // no analysis: the decisions are the clause
        ok = false;
      } else {
        decide(j);
        ok = propagate();
      }
      while (!ok) {
        ++st_.conflicts;
        ++since_restart;
        if (tr_.level == 0) {
          st_.infeasible = true;
          return false;
        }
        const long long long_before = st_.long_clauses;
        if (!learn()) return false;
        if (st_.effort >= lim_.effort) return false;
        if (st_.long_clauses > long_before && since_restart >= until && tr_.level > 0) {
          // Also here: a run of clauses too long to keep can go back and forth between two decisions.
          backtrack(0);
          commit_root();
          ++st_.restarts;
          since_restart = 0;
          until = luby(++restart_no) * lim_.restart_unit;
        }
        ok = propagate();
      }
    }
  }

 private:
  // --- heap of unfixed columns by activity (max first) ---
  bool before(Index a, Index b) const { return act_[a] > act_[b] || (act_[a] == act_[b] && a < b); }
  void sift_up(int i) {
    const Index j = heap_[i];
    while (i > 0) {
      const int par = (i - 1) / 2;
      if (!before(j, heap_[par])) break;
      heap_[i] = heap_[par], pos_[heap_[i]] = i;
      i = par;
    }
    heap_[i] = j, pos_[j] = i;
  }
  void sift_down(int i) {
    const Index j = heap_[i];
    const int sz = static_cast<int>(heap_.size());
    for (;;) {
      int c = 2 * i + 1;
      if (c >= sz) break;
      if (c + 1 < sz && before(heap_[c + 1], heap_[c])) ++c;
      if (!before(heap_[c], j)) break;
      heap_[i] = heap_[c], pos_[heap_[i]] = i;
      i = c;
    }
    heap_[i] = j, pos_[j] = i;
  }
  void heap_push(Index j) {
    if (pos_[j] >= 0) return;
    heap_.push_back(j);
    sift_up(static_cast<int>(heap_.size()) - 1);
  }
  Index heap_pop() {
    const Index j = heap_[0];
    pos_[j] = -1;
    const Index last = heap_.back();
    heap_.pop_back();
    if (!heap_.empty()) heap_[0] = last, sift_down(0);
    return j;
  }
  // The unfixed column with the highest activity (-1: all fixed).
  Index pick() {
    while (!heap_.empty()) {
      const Index j = heap_pop();
      if (lb_[j] < ub_[j]) return j;
    }
    return -1;
  }
  void bump(Index j) {
    act_[j] += inc_;
    if (pos_[j] >= 0) sift_up(pos_[j]);
    if (act_[j] > 1e100) {
      for (double& a : act_) a *= 1e-100;
      inc_ *= 1e-100;
    }
  }

  // One bound of column j towards its saved value, at a new level.
  void decide(Index j) {
    const double l = lb_[j], u = ub_[j];
    const double t = std::clamp(phase_[j], l, u);
    ++tr_.level;
    st_.max_level = std::max(st_.max_level, tr_.level);
    lvl_start_.push_back(tr_.e.size());
    if (t <= l) change(j, true, l, BoundTrail::kDecision, -1);
    else change(j, false, t, BoundTrail::kDecision, -1);
    if (lb_[j] < ub_[j]) heap_push(j);
    ++st_.decisions;
    st_.effort += 16;
  }

  // --- propagation over the rows with running activities ---
  // Every bound change goes through the trail and moves the minimal and maximal activity of the column's rows
  // (all bounds are finite); a row is visited only when it fails or when its slack is below the largest
  // |a_j| (u_j - l_j) it had at the root, i.e. when it may tighten a bound. The rounding of the implied bounds is
  // DomainPropagator's, so the analysis explains them the same way.
  void recompute_activities() {
    for (Index i = 0; i < p_.num_rows(); ++i) {
      double mn = 0.0, mx = 0.0, sp = 0.0;
      for (Index k = rows_.begin(i); k < rows_.end(i); ++k) {
        const Index j = rows_.index[k];
        const double a = rows_.value[k];
        if (a > 0) mn += a * lb_[j], mx += a * ub_[j];
        else mn += a * ub_[j], mx += a * lb_[j];
        sp = std::max(sp, std::abs(a) * (ub_[j] - lb_[j]));
      }
      minact_[i] = mn, maxact_[i] = mx, span_[i] = sp;
    }
    st_.effort += p_.A.nnz();
  }
  // Bound `up` of column j moved from `old` to `v`: activities, and (unless undoing) the rows to visit.
  void apply_delta(Index j, bool up, double old, double v, bool queue) {
    const double d = v - old;
    if (d == 0.0) return;
    st_.effort += p_.A.end(j) - p_.A.begin(j);
    for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
      const Index i = p_.A.index[k];
      const double a = p_.A.value[k];
      if ((a > 0) != up) minact_[i] += a * d;
      else maxact_[i] += a * d;
      if (!queue || rq_[i]) continue;
      const double mn = minact_[i], mx = maxact_[i], rlo = p_.row_lb[i], rhi = p_.row_ub[i];
      const bool hi = is_finite(rhi) && rhi - mn < span_[i], lo = is_finite(rlo) && mx - rlo < span_[i];
      if (hi || lo) rq_[i] = 1, rqueue_.push_back(i);  // includes every failing row (slack below zero)
    }
  }
  // A new bound, through the trail.
  void change(Index j, bool up, double v, BoundTrail::Kind kind, Index row) {
    tr_.push(j, up, v, kind, row);
    double& b = up ? ub_[j] : lb_[j];
    apply_delta(j, up, b, v, true);
    b = v;
    changed_.push_back(j);
  }
  // False: the row fails or crosses a column's bounds (recorded in the trail).
  bool visit_row(Index i) {
    const double rlo = p_.row_lb[i], rhi = p_.row_ub[i];
    {
      const double mn = minact_[i], mx = maxact_[i];
      const double scale = 1.0 + std::max(std::abs(mn), std::abs(mx));
      if (mn > rhi + kFeasTol * scale) {
        tr_.fail_row_at(i, true);
        return false;
      }
      if (mx < rlo - kFeasTol * scale) {
        tr_.fail_row_at(i, false);
        return false;
      }
      if (!(is_finite(rhi) && rhi - mn < span_[i]) && !(is_finite(rlo) && mx - rlo < span_[i])) return true;
    }
    st_.effort += rows_.end(i) - rows_.begin(i);
    for (Index k = rows_.begin(i); k < rows_.end(i); ++k) {
      const Index j = rows_.index[k];
      const double l = lb_[j], u = ub_[j];
      if (l >= u) continue;
      const double a = rows_.value[k];
      const double mn = minact_[i], mx = maxact_[i];  // current: earlier columns of this visit included
      const double rmin = a > 0 ? mn - a * l : mn - a * u, rmax = a > 0 ? mx - a * u : mx - a * l;
      const double slack = kFeasTol * (1.0 + std::abs(rmin) + std::abs(rmax));
      double nl = l, nu = u;
      if (is_finite(rhi)) {
        const double bound = (rhi - rmin) / a;
        if (a > 0) nu = std::min(nu, std::floor(bound + slack / a + 1e-9));
        else nl = std::max(nl, std::ceil(bound - slack / -a - 1e-9));
      }
      if (is_finite(rlo)) {
        const double bound = (rlo - rmax) / a;
        if (a > 0) nl = std::max(nl, std::ceil(bound - slack / a - 1e-9));
        else nu = std::min(nu, std::floor(bound + slack / -a + 1e-9));
      }
      if (nl > l) change(j, false, nl, BoundTrail::kRow, i);
      if (nu < u) change(j, true, nu, BoundTrail::kRow, i);
      if (nl > nu) {
        tr_.fail_crossing(j);
        return false;
      }
    }
    return true;
  }
  void clear_row_queue() {
    for (Index i : rqueue_) rq_[i] = 0;
    rqueue_.clear();
    changed_.clear();
  }
  // Rows until nothing is queued, then the clauses seeded with the columns the rows changed; again while the
  // clauses change bounds. False: a failure (recorded in the trail).
  bool propagate() {
    for (;;) {
      for (size_t head = 0; head < rqueue_.size(); ++head) {
        const Index i = rqueue_[head];
        rq_[i] = 0;
        if (!visit_row(i)) {
          for (size_t h = head + 1; h < rqueue_.size(); ++h) rq_[rqueue_[h]] = 0;
          rqueue_.clear();
          changed_.clear();
          return false;
        }
      }
      rqueue_.clear();
      if (pool_.size() == 0 || changed_.empty()) {
        changed_.clear();
        return true;
      }
      seeds_.swap(changed_);
      changed_.clear();
      ctight_.clear();
      const size_t from = tr_.e.size();
      const bool ok = pool_.propagate(p_.is_int, rlb_, rub_, lb_, ub_, seeds_, ctight_, now_, &tr_);
      st_.effort += pool_.last_clause_work() + 1;
      // The clauses wrote the bounds themselves: their trail entries give the activity changes.
      for (size_t k = from; k < tr_.e.size(); ++k) {
        const BoundTrail::Entry& e = tr_.e[k];
        const double old = e.prev >= 0 ? tr_.e[e.prev].val : (e.upper ? rub_[e.col] : rlb_[e.col]);
        apply_delta(e.col, e.upper != 0, old, e.val, true);
      }
      if (!ok) {
        clear_row_queue();
        return false;
      }
      if (rqueue_.empty()) return true;
    }
  }

  // Level at which the bound literal became true (0: at the root).
  int lit_level(Index j, bool up, double v) const {
    int lv = 0;
    for (int k = tr_.last[2 * static_cast<size_t>(j) + (up ? 1 : 0)]; k >= 0; k = tr_.e[k].prev) {
      const BoundTrail::Entry& e = tr_.e[k];
      if (!(up ? e.val <= v : e.val >= v)) return lv;
      lv = e.level;
    }
    return (up ? rub_[j] <= v : rlb_[j] >= v) ? 0 : lv;
  }

  // After a failure: learns a clause and jumps back; seeds_ are then the clause's columns. False: stop
  // (infeasibility proven).
  bool learn() {
    out_.clear();
    Conflict c;
    int n = 0;
    // MILP_CDS_MIN=1 (experiment): clause minimization. Shorter clauses on a third of the instances without a
    // point (bab6 928 -> 411 literals, bnatt500 101 -> 49), but no new point at 5e9 work.
    static const bool kMinEnv = std::getenv("MILP_CDS_MIN") && std::atoi(std::getenv("MILP_CDS_MIN")) != 0;
    static const Index kLongEnv = std::getenv("MILP_CDS_LONG") ? std::atoi(std::getenv("MILP_CDS_LONG")) : -1;
    long long ms[2] = {0, 0};
    if (tr_.fail != BoundTrail::kNone) {
      n = analyze_propagation_conflict(tr_, rows_, p_.row_lb, p_.row_ub, p_.is_int, rlb_, rub_,
                                       std::max(lim_.max_lits, kLongEnv >= 0 ? kLongEnv : lim_.long_lits), 1, out_, true, ms, kMinEnv || lim_.minimize);
      st_.effort += 4 * static_cast<long long>(tr_.e.size()) + 64 + 6 * ms[0];  // set up per trail entry
      st_.dropped += ms[1];
    }
    if (n > 0) {
      c = std::move(out_[0]);
    } else {
      ++st_.decision_clauses;
      if (static_cast<Index>(lvl_start_.size()) > lim_.max_lits) {
        // No clause of a usable length: the last decision is undone and its column takes the other value next
        // time. Nothing is learned, so this can repeat; a search that keeps ending here is given up.
        const BoundTrail::Entry& e = tr_.e[lvl_start_.back()];
        const Index j = e.col;
        const double other = e.upper ? e.val + 1.0 : e.val - 1.0;
        tr_.fail = BoundTrail::kNone;
        tr_.fail_conf = nullptr;
        bump(j);
        backtrack(tr_.level - 1);
        phase_[j] = std::clamp(other, rlb_[j], rub_[j]);
        return st_.decision_clauses <= 1000;
      }
      // Not all decisions hold.
      for (size_t s : lvl_start_) {
        const BoundTrail::Entry& e = tr_.e[s];
        c.idx.push_back(e.col), c.val.push_back(e.val), c.upper.push_back(static_cast<char>(e.upper));
      }
    }
    tr_.fail = BoundTrail::kNone;
    tr_.fail_conf = nullptr;
    // Levels: the highest one, how many literals it has, and the next lower one.
    int top = 0, second = 0, cnt = 0;
    size_t ktop = 0;
    for (size_t k = 0; k < c.idx.size(); ++k) {
      const int lv = lit_level(c.idx[k], c.upper[k] != 0, c.val[k]);
      if (lv > top) second = top, top = lv, cnt = 1, ktop = k;
      else if (lv == top) ++cnt;
      else if (lv > second) second = lv;
    }
    st_.effort += 30 * static_cast<long long>(c.idx.size());  // literals: sorted, levels, watches
    if (top == 0) {  // holds at the root: no point exists
      st_.infeasible = true;
      return false;
    }
    if (static_cast<Index>(c.idx.size()) > lim_.max_lits) {
      // Too long to keep. It still says where to go: back to the level where all but its last literal hold, and
      // there the other side of that literal, as a decision (the clause would have implied it).
      ++st_.long_clauses;
      for (Index j : c.idx) bump(j);
      inc_ /= 0.95;
      const Index j = c.idx[ktop];
      changed_.clear();
      if (cnt > 1) {
        backtrack(top - 1);
        return true;
      }
      backtrack(second);
      const bool up = c.upper[ktop] != 0;
      const double v = up ? c.val[ktop] + 1.0 : c.val[ktop] - 1.0;
      if (v < lb_[j] || v > ub_[j]) return true;
      ++tr_.level;
      st_.max_level = std::max(st_.max_level, tr_.level);
      lvl_start_.push_back(tr_.e.size());
      change(j, !up, v, BoundTrail::kDecision, -1);
      if (lb_[j] < ub_[j]) heap_push(j);
      return true;
    }
    int back = second;
    if (cnt > 1) ++st_.not_asserting, back = top - 1;
    for (Index j : c.idx) bump(j);
    inc_ /= 0.95;
    ++st_.learned;
    st_.lits += static_cast<long long>(c.idx.size());
    backtrack(back);
    changed_ = c.idx;  // the clause is looked at through its columns
    if (c.idx.size() == 1) {
      // A unit clause: a new root bound.
      ++st_.units;
      if (tr_.level != 0) backtrack(0);
      commit_root();
      const Index j = c.idx[0];
      if (c.upper[0]) {
        const double v = std::max(rlb_[j], c.val[0] + 1.0);
        apply_delta(j, false, lb_[j], v, true);
        lb_[j] = rlb_[j] = v;
      } else {
        const double v = std::min(rub_[j], c.val[0] - 1.0);
        apply_delta(j, true, ub_[j], v, true);
        ub_[j] = rub_[j] = v;
      }
      if (rlb_[j] > rub_[j]) {
        st_.infeasible = true;
        return false;
      }
      return true;
    }
    pool_.add(std::make_shared<const Conflict>(std::move(c)), ++now_);
    return true;
  }

  // Undoes the levels above L.
  void backtrack(int L) {
    if (tr_.level <= L) return;
    const size_t keep = lvl_start_[static_cast<size_t>(L)];
    st_.effort += 8 * static_cast<long long>(tr_.e.size() - keep);
    for (size_t k = tr_.e.size(); k-- > keep;) {
      const BoundTrail::Entry& e = tr_.e[k];
      const Index j = e.col;
      if (lb_[j] == ub_[j]) phase_[j] = lb_[j];
      tr_.last[2 * static_cast<size_t>(j) + e.upper] = e.prev;
      const double v = e.prev >= 0 ? tr_.e[e.prev].val : (e.upper ? rub_[j] : rlb_[j]);
      apply_delta(j, e.upper != 0, e.val, v, false);
      (e.upper ? ub_ : lb_)[j] = v;
      heap_push(j);
    }
    tr_.e.resize(keep);
    lvl_start_.resize(static_cast<size_t>(L));
    tr_.level = L;
    tr_.overflow = false;
    if (L == 0 && lim_.check) check_root();
  }

  // The bounds at level 0 become the root bounds; the trail starts empty.
  void commit_root() {
    rlb_ = lb_, rub_ = ub_;
    tr_.clear();
    tr_.chrono = true;
    lvl_start_.clear();
    recompute_activities();  // exact again (the running sums drift with fractional coefficients)
  }

  // At level 0 every bound is the root bound or the latest trail entry's.
  void check_root() {
    for (Index j = 0; j < p_.num_cols(); ++j) {
      const int kl = tr_.last[2 * static_cast<size_t>(j)], ku = tr_.last[2 * static_cast<size_t>(j) + 1];
      const double l = kl >= 0 ? tr_.e[kl].val : rlb_[j], u = ku >= 0 ? tr_.e[ku].val : rub_[j];
      if (lb_[j] != l || ub_[j] != u || lb_[j] < rlb_[j] || ub_[j] > rub_[j]) st_.check_failed = true;
    }
    for (const BoundTrail::Entry& e : tr_.e) {
      if (e.level != 0) st_.check_failed = true;
    }
    for (Index i = 0; i < p_.num_rows(); ++i) {
      double mn = 0.0, mx = 0.0;
      for (Index k = rows_.begin(i); k < rows_.end(i); ++k) {
        const Index j = rows_.index[k];
        const double a = rows_.value[k];
        if (a > 0) mn += a * lb_[j], mx += a * ub_[j];
        else mn += a * ub_[j], mx += a * lb_[j];
      }
      if (std::abs(mn - minact_[i]) > 1e-7 * (1.0 + std::abs(mn)) || std::abs(mx - maxact_[i]) > 1e-7 * (1.0 + std::abs(mx))) {
        st_.check_failed = true;
      }
    }
  }

  const MipProblem& p_;
  const CdsLimits& lim_;
  CdsStats& st_;
  SparseMatrix rows_;  // CSR
  std::vector<double> minact_, maxact_, span_;
  std::vector<char> rq_;
  std::vector<Index> rqueue_, changed_;
  ConflictPool pool_;
  BoundTrail tr_;
  std::vector<double> lb_, ub_, rlb_, rub_;  // current and root bounds
  std::vector<size_t> lvl_start_;            // trail position of the decision of level L (at L - 1)
  std::vector<double> act_, phase_;
  std::vector<Index> heap_;
  std::vector<int> pos_;
  double inc_ = 1.0;
  long long now_ = 0;
  std::vector<Index> seeds_, ctight_;
  std::vector<Conflict> out_;
};

}  // namespace

bool conflict_search(const MipProblem& p, const CdsLimits& limits, std::vector<double>& sol, CdsStats* stats) {
  CdsStats local;
  CdsStats& st = stats ? *stats : local;
  st = CdsStats{};
  if (p.quadratic() || !p.implications.empty() || p.nonlinear() || p.num_cols() == 0) return false;
  for (Index j = 0; j < p.num_cols(); ++j) {
    if (!p.is_int[j] || !(p.col_lb[j] >= -kMaxBound) || !(p.col_ub[j] <= kMaxBound)) return false;
  }
  st.applicable = true;
  Cds s(p, limits, st);
  return s.run(sol);
}

}  // namespace milp
