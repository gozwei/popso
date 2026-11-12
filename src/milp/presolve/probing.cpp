#include "milp/presolve/probing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "milp/core/interrupt.hpp"

namespace milp {

namespace {

constexpr double kFeasTol = 1e-6;
constexpr double kBigBound = 1e12;  // larger bounds are treated as infinite

class Prober {
 public:
  Prober(const MipProblem& p, long long work_limit, const ProbingOptions& opt)
      : p_(p), rows_(p.A.transpose()), limit_(work_limit), lb_(p.col_lb), ub_(p.col_ub) {
    trail_cap_ = 1000000 + 20 * static_cast<size_t>(p.num_cols());
    queued_.assign(p.num_rows(), 0);
    lift_ = opt.lift;
    max_conf_ = opt.max_conflicts;
    if (max_conf_ > 0) {
      conf_.resize(2 * static_cast<size_t>(p.num_cols()));
      last_lit_.assign(p.num_cols(), -1);
      no_imp_.assign(p.num_cols(), -1);
      if (opt.seeds) {
        for (const auto& [a, b] : *opt.seeds) {
          if (num_conf_ >= max_conf_) break;
          if (is_binary(a >> 1) && is_binary(b >> 1) && (a >> 1) != (b >> 1)) add_conflict(a, b);
        }
      }
    }
    if (lift_) {
      touch_.assign(p.num_rows(), -1);
      best_.assign(p.num_rows(), -1);
      redundant_.assign(p.num_rows(), 0);
      rowmax_.assign(p.num_rows(), 0.0);
      for (Index i = 0; i < p.num_rows(); ++i) {
        for (Index k = rows_.begin(i); k < rows_.end(i); ++k) rowmax_[i] = std::max(rowmax_[i], std::abs(rows_.value[k]));
      }
    }
    stamp_.assign(p.num_cols(), -1);
    first_.assign(p.num_cols(), -1);
    lb0_.resize(p.num_cols());
    ub0_.resize(p.num_cols());
    row_int_.assign(p.num_rows(), 1);
    for (Index i = 0; i < p.num_rows(); ++i) {
      for (Index k = rows_.begin(i); k < rows_.end(i); ++k) {
        const Index j = rows_.index[k];
        const double b = std::max(std::abs(p.col_lb[j]), std::abs(p.col_ub[j]));
        if (!(b < (p.is_int[j] ? kBigBound : 1e7))) row_int_[i] = 0;
      }
    }
  }

  ProbingResult run() {
    ProbingResult res;
    const Index n = p_.num_cols();
    std::vector<Index> cand;
    for (Index j = 0; j < n; ++j) {
      if (is_binary(j)) cand.push_back(j);
    }
    std::stable_sort(cand.begin(), cand.end(),
                     [&](Index a, Index b) { return p_.A.end(a) - p_.A.begin(a) > p_.A.end(b) - p_.A.begin(b); });
    std::vector<char> skip(n, 0);
    range_.assign(p_.num_rows(), 0.0);
    for (Index j = 0; j < n; ++j) {
      for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
        const double r = std::abs(p_.A.value[k]) * (ub_[j] - lb_[j]);
        double& rg = range_[p_.A.index[k]];
        rg = std::max(rg, std::isfinite(r) && r < kBigBound ? r : kInf);
      }
    }
    init_activities();
    long long idle = 0;  // work since the last useful probe
    // Longest columns first. When 10% of the budget (MILP_PROBE_IDLE, fraction) passed without a useful probe, the
    // probing continues from the other end of the order (shortest columns first), then stops after another idle stretch:
    // 30n20b8's 3241 fixings are all among its shorter columns (2745 long-column probes found none;
    // SCIP and HiGHS find them by continuing to probe). MILP_PROBE_SWITCH=0: stop as before.
    static const bool allow_switch = [] { const char* v = std::getenv("MILP_PROBE_SWITCH"); return !(v && v[0] == '0'); }();
    size_t lo = 0, hi = cand.size();
    bool reversed = false;
    while (lo < hi) {
      if (work_ > limit_ || ((res.probes & 15) == 0 && interrupted())) break;
      static const double idle_frac = [] { const char* v = std::getenv("MILP_PROBE_IDLE"); return v ? std::atof(v) : 0.1; }();
      if (idle > static_cast<long long>(idle_frac * static_cast<double>(limit_))) {
        if (reversed || !allow_switch) break;
        reversed = true;
        idle = 0;
      }
      const Index x = reversed ? cand[--hi] : cand[lo++];
      if (skip[x] || lb_[x] == ub_[x]) continue;
      if (updates_ > 4 * p_.A.nnz() + 100000) init_activities();
      ++res.probes;
      const long long work0 = work_;
      const size_t found = res.equiv.size();
      // x = 0: remember the resulting bounds of every changed column.
      start_touch();
      set_bounds(x, 0.0, 0.0);
      const bool f0 = propagate();
      collect_ = false;
      lift0_.clear();
      if (f0 && lift_) find_lifts(x, false, lift0_);
      fix0_.clear();
      for (const Change& c : trail_) {
        if (stamp_[c.j] != x && c.j != x && max_conf_ > 0 && fixed_binary(c.j)) fix0_.push_back(c.j);
        stamp_[c.j] = x;
        lb0_[c.j] = lb_[c.j];
        ub0_[c.j] = ub_[c.j];
      }
      undo();
      start_touch();
      set_bounds(x, 1.0, 1.0);
      const bool f1 = propagate();
      collect_ = false;
      if (f0 && f1 && lift_) {
        // Liftings only while x is free (a fixing makes them moot); the best one per row is kept.
        lift1_.clear();
        find_lifts(x, true, lift1_);
        for (const auto* v : {&lift0_, &lift1_}) {
          for (const ProbingResult::Lift& l : *v) keep_lift(l, res);
        }
      }
      if (f0 && f1 && num_conf_ < max_conf_) {
        // x = v fixed the binary y to w: literals (x = v) and (y = 1 - w) conflict. A binary fixed to the same value
        // by both settings is fixed globally below instead.
        for (Index j : fix0_) {
          if (num_conf_ >= max_conf_) break;
          const Index w = lb0_[j] > 0.5 ? 1 : 0;
          if (lb_[j] == ub_[j] && lb_[j] == lb0_[j]) continue;
          add_conflict(2 * x, 2 * j + 1 - w, &res);
        }
        for (const Change& c : trail_) {
          const Index j = c.j;
          if (j == x || num_conf_ >= max_conf_ || !fixed_binary(j)) continue;
          const Index w = lb_[j] > 0.5 ? 1 : 0;
          if (stamp_[j] == x && lb0_[j] == ub0_[j] && lb0_[j] == lb_[j]) continue;
          if (last_lit_[j] == 2 * x + 1) continue;  // the trail may hold j several times
          last_lit_[j] = 2 * x + 1;
          add_conflict(2 * x + 1, 2 * j + 1 - w, &res);
        }
      }
      if (!f0 && !f1) {
        res.infeasible = true;
        break;
      }
      if (f0 && f1) {
        // Implications of both settings: union of the two domains; equivalent binaries.
        std::vector<Change> global;
        for (const Change& c : trail_) {
          const Index j = c.j;
          if (j == x || stamp_[j] != x) continue;
          stamp_[j] = -1;  // each column once (the trail may hold it several times)
          const double l = std::min(lb0_[j], lb_[j]), u = std::max(ub0_[j], ub_[j]);
          const double rl = root_lb(j), ru = root_ub(j);
          if (l > rl || u < ru) global.push_back({j, std::max(l, rl), std::min(u, ru)});
          double v0, v1;
          if (!skip[j] && fixed_value(j, lb0_[j], ub0_[j], &v0) && fixed_value(j, lb_[j], ub_[j], &v1) && v0 != v1) {
            res.equiv.push_back({j, x, v0, v1});
            skip[j] = 1;
          }
        }
        undo();
        idle = !global.empty() || res.equiv.size() > found ? 0 : idle + work_ - work0;
        for (const Change& g : global) set_bounds(g.j, g.lb, g.ub);
        if (!global.empty() && !propagate()) {
          res.infeasible = true;
          break;
        }
        commit();
      } else {
        undo();
        idle = 0;
        const double v = f0 ? 0.0 : 1.0;
        set_bounds(x, v, v);
        if (!propagate()) {
          res.infeasible = true;
          break;
        }
        commit();
      }
    }
    res.lb = lb_;
    res.ub = ub_;
    res.work = work_;
    return res;
  }

 private:
  // Liftings of the rows touched by the current probe x = v (bounds now those implied by it), see
  // probing.hpp. One-sided rows only (an equation or a ranged row shares a_x between both sides), not
  // redundant under the bounds before the probe; delta > 1e-6 max(1, |b|) and > 1e-3 max |a|.
  void find_lifts(Index x, bool one, std::vector<ProbingResult::Lift>& out) {
    for (Index i : touched_) {
      const bool up = std::isfinite(p_.row_ub[i]);
      if (up == std::isfinite(p_.row_lb[i]) || redundant_[i]) continue;
      if (up ? nmax_[i] > 0 : nmin_[i] > 0) continue;  // unbounded activity
      const double s = up ? 1.0 : -1.0;
      const double b = up ? p_.row_ub[i] : -p_.row_lb[i];
      const double need = std::max(1e-6 * std::max(1.0, std::abs(b)), 1e-3 * rowmax_[i]);
      // Cheap test on the incremental activities (they may drift a little; the exact sum decides).
      if (!(b - (up ? amax_[i] : -amin_[i]) > 0.5 * need)) continue;
      const Index rb = rows_.begin(i), re = rows_.end(i);
      work_ += re - rb;
      double act = 0.0, act0 = 0.0, mag = std::abs(b);
      bool ok = true, has_x = false, exact = mag < 1e9 && b == std::round(b);
      for (Index k = rb; k < re && ok; ++k) {
        const Index j = rows_.index[k];
        const double a = s * rows_.value[k];
        has_x = has_x || j == x;
        const double h = a > 0 ? ub_[j] : lb_[j], h0 = a > 0 ? root_ub(j) : root_lb(j);
        ok = std::abs(h) < kBigBound && std::abs(h0) < kBigBound;
        act += a * h;
        act0 += a * h0;
        mag += std::abs(a * h0);
        exact = exact && a == std::round(a) && h == std::round(h) && h0 == std::round(h0) && std::abs(a * h0) < 1e9;
      }
      if (!ok) continue;
      if (act0 <= b + 1e-9 * (1.0 + std::abs(b))) {
        redundant_[i] = 1;  // also later in this pass: the global bounds only tighten (presolve removes it)
        continue;
      }
      // Margin against rounding errors in the activity (none when all data are moderate integers).
      const double safety = exact ? 0.0 : 1e-10 * mag;
      double delta = b - act - safety;
      if (!(delta > need)) continue;
      // A margin-sized distance to an integer is rounding error (continuous bounds relaxed by the propagation, the
      // safety margin): integral data stay integral (the error, <= 1e-8 |delta|, is far below the tolerances).
      const double r = std::round(delta);
      if (std::abs(delta - r) <= 1e-8 * std::max(1.0, std::abs(delta))) delta = r;
      out.push_back({i, x, delta, up, one, !has_x, delta / rowmax_[i]});
    }
  }

  // Keeps l if it is the best lifting of its row so far (largest delta / max |a|; on ties no new entry).
  void keep_lift(const ProbingResult::Lift& l, ProbingResult& res) {
    Index& b = best_[l.row];
    if (b < 0) {
      b = static_cast<Index>(res.lifts.size());
      res.lifts.push_back(l);
      return;
    }
    const ProbingResult::Lift& o = res.lifts[b];
    if (l.score > o.score * (1.0 + 1e-9) || (l.score >= o.score * (1.0 - 1e-9) && o.fill && !l.fill)) res.lifts[b] = l;
  }

  void start_touch() {
    touched_.clear();
    collect_ = lift_;
    ++touch_id_;
    ++phase_;
  }

  struct Change {
    Index j;
    double lb, ub;  // previous bounds (in the trail) or new bounds (global list)
  };

  bool is_binary(Index j) const { return p_.is_int[j] && lb_[j] == 0.0 && ub_[j] == 1.0; }
  // Whether [l, u] pins column j to one value (up to rounding errors for continuous columns).
  bool fixed_value(Index j, double l, double u, double* v) const {
    if (l == u) return *v = l, true;
    if (p_.is_int[j] || !(u - l <= 1e-8 * (1.0 + std::abs(l)))) return false;
    const double r = std::round(l);
    *v = std::abs(r - l) <= 1e-8 * (1.0 + std::abs(l)) ? r : 0.5 * (l + u);
    return true;
  }
  // Bounds before the current probe: the oldest trail entry of j, else the current bounds.
  double root_lb(Index j) const { return first_[j] >= 0 ? trail_[first_[j]].lb : lb_[j]; }
  double root_ub(Index j) const { return first_[j] >= 0 ? trail_[first_[j]].ub : ub_[j]; }

  // Activity contributions of a x_j for x_j in [l, u] (bounds beyond kBigBound count as infinite).
  static void contrib(double a, double l, double u, double* lo, double* hi, int* lo_inf, int* hi_inf) {
    const double bl = a > 0 ? l : u, bh = a > 0 ? u : l;
    if (std::abs(bl) < kBigBound) *lo = a * bl, *lo_inf = 0; else *lo = 0.0, *lo_inf = 1;
    if (std::abs(bh) < kBigBound) *hi = a * bh, *hi_inf = 0; else *hi = 0.0, *hi_inf = 1;
  }

  // Row activities from scratch (they drift through incremental updates).
  void init_activities() {
    amin_.assign(p_.num_rows(), 0.0);
    amax_.assign(p_.num_rows(), 0.0);
    nmin_.assign(p_.num_rows(), 0);
    nmax_.assign(p_.num_rows(), 0);
    for (Index j = 0; j < p_.num_cols(); ++j) {
      for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
        double lo, hi;
        int li, hi_i;
        contrib(p_.A.value[k], lb_[j], ub_[j], &lo, &hi, &li, &hi_i);
        const Index i = p_.A.index[k];
        amin_[i] += lo, amax_[i] += hi, nmin_[i] += li, nmax_[i] += hi_i;
      }
    }
    updates_ = 0;
  }

  // Bounds of j change from [l0, u0] to the current ones: updates the activities of its rows.
  void update_activities(Index j, double l0, double u0, bool enqueue) {
    // Only the contributions of a changed bound are updated (same arithmetic as contrib()).
    const double l1 = lb_[j], u1 = ub_[j];
    const bool lch = l1 != l0, uch = u1 != u0;
    auto term = [](double a, double b, double* v, int* inf) {
      if (std::abs(b) < kBigBound) *v = a * b, *inf = 0; else *v = 0.0, *inf = 1;
    };
    for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
      const Index i = p_.A.index[k];
      const double a = p_.A.value[k];
      const bool pos = a > 0;
      double v0, v1;
      int i0, i1;
      if (pos ? lch : uch) {  // minimal contribution: a l (a > 0) or a u
        term(a, pos ? l0 : u0, &v0, &i0);
        term(a, pos ? l1 : u1, &v1, &i1);
        amin_[i] += v1 - v0, nmin_[i] += i1 - i0;
      }
      if (pos ? uch : lch) {  // maximal contribution
        term(a, pos ? u0 : l0, &v0, &i0);
        term(a, pos ? u1 : l1, &v1, &i1);
        amax_[i] += v1 - v0, nmax_[i] += i1 - i0;
      }
      if (enqueue && !queued_[i]) queued_[i] = 1, queue_.push_back(i);
      if (collect_ && touch_[i] != touch_id_) touch_[i] = touch_id_, touched_.push_back(i);
    }
    updates_ += p_.A.end(j) - p_.A.begin(j);
  }

  void set_bounds(Index j, double l, double u) {
    if (first_[j] < 0) first_[j] = static_cast<Index>(trail_.size());
    trail_.push_back({j, lb_[j], ub_[j]});
    lb_[j] = std::max(lb_[j], l);
    ub_[j] = std::min(ub_[j], u);
    work_ += p_.A.end(j) - p_.A.begin(j);
    update_activities(j, trail_.back().lb, trail_.back().ub, true);
    // A binary just fixed: its literal is true (its conflicts are propagated).
    if (num_conf_ > 0 && lb_[j] == ub_[j] && trail_.back().lb == 0.0 && trail_.back().ub == 1.0 && p_.is_int[j]) {
      const Index lit = 2 * j + (lb_[j] > 0.5 ? 1 : 0);
      if (!conf_[lit].empty()) lits_.push_back(lit);
    }
  }

  void undo() {
    for (size_t t = trail_.size(); t-- > 0;) {
      const Index j = trail_[t].j;
      const double l = lb_[j], u = ub_[j];
      lb_[j] = trail_[t].lb;
      ub_[j] = trail_[t].ub;
      update_activities(j, l, u, false);
      first_[j] = -1;
    }
    trail_.clear();
    clear_queue();
  }

  // Whether row i might tighten a bound or be infeasible (cheap test on the incremental activities; the
  // row is then processed exactly). range_[i] bounds every |a_ij| (u_j - l_j).
  bool may_propagate(Index i) const {
    const double rlo = p_.row_lb[i], rhi = p_.row_ub[i];
    const double tol = 1e-6 * (1.0 + std::abs(amin_[i]) + std::abs(amax_[i]));
    if (std::isfinite(rhi) && (nmin_[i] == 1 || (nmin_[i] == 0 && rhi - amin_[i] < range_[i] + tol))) return true;
    if (std::isfinite(rlo) && (nmax_[i] == 1 || (nmax_[i] == 0 && amax_[i] - rlo < range_[i] + tol))) return true;
    return false;
  }

  // Keeps the current bounds as the new root.
  void commit() {
    for (const Change& c : trail_) first_[c.j] = -1;
    trail_.clear();
  }

  void clear_queue() {
    for (Index i : queue_) queued_[i] = 0;
    queue_.clear();
    lits_.clear();
  }

  // Processes the row queue and the literals that became true (their conflicts); false if a row or a conflict is
  // proven infeasible.
  bool propagate() {
    for (;;) {
      for (size_t head = 0; head < queue_.size(); ++head) {
        // One probe's chain of bound changes is capped: the bounds reached so far are valid, the rest is skipped.
        if (trail_.size() > trail_cap_) {
          clear_queue();
          return true;
        }
        const Index i = queue_[head];
        queued_[i] = 0;
        if (!process_row(i)) {
          for (size_t h = head + 1; h < queue_.size(); ++h) queued_[queue_[h]] = 0;
          queue_.clear();
          lits_.clear();
          return false;
        }
      }
      queue_.clear();
      if (lits_.empty()) return true;
      for (size_t h = 0; h < lits_.size(); ++h) {
        const Index lit = lits_[h];
        work_ += static_cast<long long>(conf_[lit].size());
        for (size_t t = 0; t < conf_[lit].size(); ++t) {
          const Index o = conf_[lit][t], k = o >> 1;
          const double v = (o & 1) ? 0.0 : 1.0;  // the literal (k, o & 1) is false: x_k = 1 - (o & 1)
          if (lb_[k] == ub_[k]) {
            if (lb_[k] != v) {
              lits_.clear();
              clear_queue();
              return false;
            }
            continue;
          }
          set_bounds(k, v, v);  // queues the rows and the literal of k
          no_imp_[k] = phase_;
        }
      }
      lits_.clear();
    }
  }

  // Whether j is a binary fixed by the current probe (free before it) whose implication is worth keeping: not one
  // fixed through the implication graph (the graph already leads to it).
  bool fixed_binary(Index j) const {
    return p_.is_int[j] && lb_[j] == ub_[j] && root_lb(j) == 0.0 && root_ub(j) == 1.0 && no_imp_[j] != phase_;
  }

  void add_conflict(Index a, Index b, ProbingResult* res = nullptr) {
    conf_[a].push_back(b);
    conf_[b].push_back(a);
    ++num_conf_;
    if (res) res->conflicts.push_back({std::min(a, b), std::max(a, b)});
  }

  bool process_row(Index i) {
    ++work_;
    if (!may_propagate(i)) return true;
    const Index rb = rows_.begin(i), re = rows_.end(i);
    work_ += re - rb;
    double minact = 0.0, maxact = 0.0;
    int mininf = 0, maxinf = 0;
    if (row_int_[i]) {
      // Rows of columns with moderate finite bounds: the incremental activities are accurate (drift ~1e-16 relative, far
      // below the rounding margins; they are recomputed periodically).
      minact = amin_[i], maxact = amax_[i], mininf = nmin_[i], maxinf = nmax_[i];
    } else {
      for (Index k = rb; k < re; ++k) {
        const Index j = rows_.index[k];
        const double a = rows_.value[k];
        const double l = lb_[j] < -kBigBound ? -kInf : lb_[j], u = ub_[j] > kBigBound ? kInf : ub_[j];
        const double lo = a > 0 ? l : u, hi = a > 0 ? u : l;
        if (std::isfinite(lo)) minact += a * lo; else ++mininf;
        if (std::isfinite(hi)) maxact += a * hi; else ++maxinf;
      }
    }
    const double rlo = p_.row_lb[i], rhi = p_.row_ub[i];
    const double scale = 1.0 + std::max(std::abs(minact), std::abs(maxact));
    if ((mininf == 0 && minact > rhi + kFeasTol * scale) || (maxinf == 0 && maxact < rlo - kFeasTol * scale)) {
      return false;
    }
    const bool use_hi = std::isfinite(rhi) && mininf <= 1, use_lo = std::isfinite(rlo) && maxinf <= 1;
    if (!use_hi && !use_lo) return true;
    // A column with a finite domain can only be tightened by a side whose slack is below |a| (u - l).
    const double slack_hi = mininf == 0 ? rhi - minact : -kInf, slack_lo = maxinf == 0 ? maxact - rlo : -kInf;
    const double gate_hi = slack_hi - 1e-9 * (1.0 + std::abs(slack_hi)), gate_lo = slack_lo - 1e-9 * (1.0 + std::abs(slack_lo));
    for (Index k = rb; k < re; ++k) {
      const Index j = rows_.index[k];
      if (lb_[j] == ub_[j]) continue;
      const double a = rows_.value[k];
      const double l = lb_[j] < -kBigBound ? -kInf : lb_[j], u = ub_[j] > kBigBound ? kInf : ub_[j];
      if (std::isfinite(l) && std::isfinite(u)) {
        const double span = std::abs(a) * (u - l);
        if (!(use_hi && mininf == 0 && span > gate_hi) && !(use_lo && maxinf == 0 && span > gate_lo)) continue;
      }
      const bool integer = p_.is_int[j];
      // A continuous column whose domain is down to rounding errors counts as fixed. The rule below accepts any
      // step that lands within 1e-8 of the opposite bound, so two such columns in two rows trade steps of 1e-18
      // without end (neos-5104907-jarama, a sub-MIP's probing: columns in [3.5e-13, 9.6e-12], one probe's trail
      // grew to 13 GB).
      if (!integer && u - l <= 1e-9 * (1.0 + std::max(std::abs(l), std::abs(u)))) continue;
      const double lo = a > 0 ? l : u, hi = a > 0 ? u : l;
      // Residual activities without j.
      const bool rmin_ok = mininf - (std::isfinite(lo) ? 0 : 1) == 0;
      const bool rmax_ok = maxinf - (std::isfinite(hi) ? 0 : 1) == 0;
      const double rmin = std::isfinite(lo) ? minact - a * lo : minact;
      const double rmax = std::isfinite(hi) ? maxact - a * hi : maxact;
      // Integer bounds are rounded with a feasibility tolerance; continuous ones are exact implications up to
      // rounding errors (relaxed by those) and only significant changes are kept.
      const double mag = (1.0 + std::abs(rmin) + std::abs(rmax)) / std::abs(a);
      auto slack = [&](double side) { return integer ? kFeasTol * mag + 1e-9 : 1e-11 * (mag + std::abs(side / a)); };
      double nl = -kInf, nu = kInf;
      if (use_hi && rmin_ok && std::isfinite(rmin) && std::abs(rmin) < kBigBound) {
        const double v = (rhi - rmin) / a;
        if (a > 0) nu = v + slack(rhi); else nl = v - slack(rhi);
      }
      if (use_lo && rmax_ok && std::isfinite(rmax) && std::abs(rmax) < kBigBound) {
        const double v = (rlo - rmax) / a, sl = slack(rlo);
        if (a > 0) nl = std::max(nl, v - sl); else nu = std::min(nu, v + sl);
      }
      if (integer) {
        nl = std::ceil(nl);
        nu = std::floor(nu);
      } else {
        // Insignificant changes would only keep the queue busy: a change must be 0.1% of the column's domain
        // when probing started (a threshold that does not shrink with the domain, so chains of ever smaller
        // steps stop; neos-1122047).
        const double l0 = p_.col_lb[j], u0 = p_.col_ub[j];
        const double range = std::isfinite(l0) && std::isfinite(u0) && u0 - l0 < kBigBound ? u0 - l0 : kInf;
        const double min_gain = 1e-3 * std::min(range, 1e6);
        if (!(nl > l + std::max(min_gain, 1e-7 * (1.0 + std::abs(l))) || nl > u - 1e-8 * (1.0 + std::abs(u)))) nl = -kInf;
        if (!(nu < u - std::max(min_gain, 1e-7 * (1.0 + std::abs(u))) || nu < l + 1e-8 * (1.0 + std::abs(l)))) nu = kInf;
        if (std::abs(nl) >= kBigBound) nl = -kInf;
        if (std::abs(nu) >= kBigBound) nu = kInf;
        // Crossing within rounding errors: the column is fixed.
        if (nl > nu && nl - nu <= 1e-9 * (1.0 + std::abs(nl))) nl = nu = 0.5 * (nl + nu);
      }
      nl = std::max(nl, l);
      nu = std::min(nu, u);
      if (nl > nu) {
        if (integer || nl - nu > kFeasTol * (1.0 + std::abs(nl))) return false;
        // Fixed inside the current domain: a midpoint below the lower bound (or above the upper one) would leave
        // the bounds crossed, and such errors add up over a long row (neos-5104907-jarama: 92 columns >= 0 with
        // bounds [1e-9, -5e-8], the row "proven" infeasible in the second probing pass).
        nl = nu = std::clamp(0.5 * (nl + nu), lb_[j], ub_[j]);
      }
      if (nl > lb_[j] || nu < ub_[j]) {
        set_bounds(j, nl, nu);
        // Activities of this row changed; it was re-queued by set_bounds.
      }
    }
    return true;
  }

  const MipProblem& p_;
  SparseMatrix rows_;  // CSR copy of A
  long long limit_, work_ = 0;
  size_t trail_cap_ = 0;
  std::vector<double> lb_, ub_, lb0_, ub0_;
  std::vector<Index> stamp_, first_;
  std::vector<Change> trail_;
  std::vector<char> queued_;
  std::vector<char> row_int_;  // incremental activities usable: finite, moderate bounds (continuous: <= 1e7)
  std::vector<Index> queue_;
  std::vector<double> amin_, amax_, range_;  // incremental row activities; max entry range per row
  std::vector<int> nmin_, nmax_;
  long long updates_ = 0;
  // Liftings: rows whose activities the current probe changed (touch_ stamps them with touch_id_), the
  // candidates of the x = 0 / x = 1 probes, the index of each row's kept lifting in the result.
  bool lift_ = false, collect_ = false;
  long long touch_id_ = 0;
  std::vector<long long> touch_;
  std::vector<Index> touched_, best_;
  std::vector<double> rowmax_;  // max |a| per row
  std::vector<ProbingResult::Lift> lift0_, lift1_;
  // Implication graph: conflicting literals (2 j + v: x_j = v) per literal; literals that became true and await
  // propagation; binaries fixed by the x = 0 probe; last literal whose conflict with j was added (trail duplicates).
  size_t max_conf_ = 0, num_conf_ = 0;
  std::vector<std::vector<Index>> conf_;
  std::vector<Index> lits_, fix0_;
  std::vector<Index> last_lit_;
  // Current probe x = v (stamps of no_imp_); phase in which a column was fixed through the implication graph.
  long long phase_ = 0;
  std::vector<long long> no_imp_;
  std::vector<char> redundant_;  // rows found redundant under the current global bounds (no lifting)
};

}  // namespace

ProbingResult probe_binaries(const MipProblem& p, long long work_limit, const ProbingOptions& opt) {
  Prober pr(p, work_limit, opt);
  return pr.run();
}

}  // namespace milp
