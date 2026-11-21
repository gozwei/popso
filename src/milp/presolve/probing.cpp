#include "milp/presolve/probing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>

#include "milp/core/interrupt.hpp"

namespace milp {

namespace {

constexpr double kFeasTol = 1e-6;
constexpr double kBigBound = 1e12;  // larger bounds are treated as infinite

class Prober {
 public:
  struct Change {
    Index j;
    double lb, ub;  // previous bounds (in the trail) or new bounds (global list)
  };

  Prober(const MipProblem& p, long long work_limit, const ProbingOptions& opt)
      : p_(p), rows_(p.A.transpose()), limit_(work_limit), lb_(p.col_lb), ub_(p.col_ub) {
    trail_cap_ = 1000000 + 20 * static_cast<size_t>(p.num_cols());
    rs_.assign(p.num_rows(), RowState{});
    for (Index i = 0; i < p.num_rows(); ++i) rs_[i].rlo = p.row_lb[i], rs_[i].rhi = p.row_ub[i];
    lift_ = opt.lift;
    idle_opt_ = opt.idle_frac;
    lift_dyadic_ = opt.lift_dyadic;
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
    for (Index i = 0; i < p.num_rows(); ++i) {
      for (Index k = rows_.begin(i); k < rows_.end(i); ++k) {
        const Index j = rows_.index[k];
        const double b = std::max(std::abs(p.col_lb[j]), std::abs(p.col_ub[j]));
        if (!(b < (p.is_int[j] ? kBigBound : 1e7))) rs_[i].row_int = 0;
      }
    }
    // Exact rows: integer columns with finite integer bounds and integer coefficients, small enough that every
    // partial sum of the activities is an integer below 2^50. Their incremental updates carry no rounding error,
    // so undoing a probe may put back the values saved at the row's first change instead of reversing every
    // update (the same bits either way), and the columns of such rows are not walked at an undo.
    // The tightening loop's copy of the rows, without the columns fixed for good (compact_row).
    pidx_ = rows_.index;
    pval_ = rows_.value;
    for (Index i = 0; i < p.num_rows(); ++i) {
      rs_[i].pend = rows_.end(i);
      for (Index k = rows_.begin(i); k < rows_.end(i); ++k) {
        if (p.col_lb[rows_.index[k]] == p.col_ub[rows_.index[k]]) count_dead(i);
      }
    }
    for (Index j = 0; j < p.num_cols() && int_bounds_; ++j) {
      if (!p.is_int[j]) continue;
      const double l = p.col_lb[j], u = p.col_ub[j];
      int_bounds_ = (!std::isfinite(l) || l == std::round(l)) && (!std::isfinite(u) || u == std::round(u));
    }
    col_exact_.assign(p.num_cols(), 1);
    for (Index i = 0; i < p.num_rows(); ++i) {
      double mag = 0.0;
      bool exact = true;
      for (Index k = rows_.begin(i); k < rows_.end(i) && exact; ++k) {
        const Index j = rows_.index[k];
        const double a = rows_.value[k], l = p.col_lb[j], u = p.col_ub[j];
        exact = p.is_int[j] && a == std::round(a) && std::isfinite(l) && std::isfinite(u) && l == std::round(l) &&
                u == std::round(u);
        mag += std::abs(a) * std::max(std::abs(l), std::abs(u));
        exact = exact && mag <= 1125899906842624.0;
      }
      rs_[i].exact = exact;
    }
    // Only whole blocks: a row counts as exact only if every column of it has exact rows alone (and so on), so
    // that a saved row is never also reversed entry by entry (a model with inexact rows mixed in paid for both:
    // gfd-schedulen, neos-4722843-widden 5-7% more probing time).
    std::vector<Index> stack;
    for (Index i = 0; i < p.num_rows(); ++i) {
      if (!rs_[i].exact) stack.push_back(i);
    }
    while (!stack.empty()) {
      const Index i = stack.back();
      stack.pop_back();
      for (Index k = rows_.begin(i); k < rows_.end(i); ++k) {
        const Index j = rows_.index[k];
        if (!col_exact_[j]) continue;
        col_exact_[j] = 0;
        for (Index t = p.A.begin(j); t < p.A.end(j); ++t) {
          RowState& r = rs_[p.A.index[t]];
          if (r.exact) r.exact = 0, stack.push_back(p.A.index[t]);
        }
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
    for (RowState& r : rs_) r.range = 0.0;
    for (Index j = 0; j < n; ++j) {
      for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
        const double r = std::abs(p_.A.value[k]) * (ub_[j] - lb_[j]);
        double& rg = rs_[p_.A.index[k]].range;
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
      static const double idle_env = [] { const char* v = std::getenv("MILP_PROBE_IDLE"); return v ? std::atof(v) : 0.1; }();
      const double idle_frac = idle_opt_ >= 0.0 ? idle_opt_ : idle_env;
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


  // ---- Batched probing (MILP_PROBE_PAR = B >= 2; ProbingOptions::batch) ------------------------------------
  // B probers hold the same global bounds and implications. A batch gives the next B candidates one prober each
  // (slot k to prober k, whatever the thread count: the result does not depend on the threads); each probes its
  // candidate without committing anything, then prober 0 applies the results in candidate order (as the
  // sequential loop does, propagating each fixing) and the others take over its bound changes and conflicts.
  // A candidate of a batch does not see what the earlier ones of the same batch found; every derivation holds
  // under any valid global bounds, so the results stay sound (and a candidate fixed meanwhile is dropped).
  struct ProbeOut {
    Index x = -1;
    bool f0 = true, f1 = true;
    long long work = 0;
    std::vector<std::pair<Index, Index>> conf;  // conflicting literal pairs found
    std::vector<ProbingResult::Lift> lifts;
    std::vector<ProbingResult::Equiv> equiv;
    std::vector<double> gl, gu;  // global bounds implied by both settings, of the columns gj
    std::vector<Index> gj;
    void clear() {
      f0 = f1 = true, work = 0;
      conf.clear(), lifts.clear(), equiv.clear(), gl.clear(), gu.clear(), gj.clear();
    }
  };

  // The two probes of x, undone; what they imply goes to out (the body of run()'s loop without its commits).
  void probe_one(Index x, ProbeOut& out) {
    out.clear();
    out.x = x;
    if (updates_ > 4 * p_.A.nnz() + 100000) init_activities();
    const long long work0 = work_;
    start_touch();
    set_bounds(x, 0.0, 0.0);
    out.f0 = propagate();
    collect_ = false;
    lift0_.clear();
    if (out.f0 && lift_) find_lifts(x, false, lift0_);
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
    out.f1 = propagate();
    collect_ = false;
    if (out.f0 && out.f1) {
      if (lift_) {
        lift1_.clear();
        find_lifts(x, true, lift1_);
        out.lifts = lift0_;
        out.lifts.insert(out.lifts.end(), lift1_.begin(), lift1_.end());
      }
      if (max_conf_ > 0) {
        for (Index j : fix0_) {
          const Index w = lb0_[j] > 0.5 ? 1 : 0;
          if (lb_[j] == ub_[j] && lb_[j] == lb0_[j]) continue;
          out.conf.push_back({2 * x, 2 * j + 1 - w});
        }
        for (const Change& c : trail_) {
          const Index j = c.j;
          if (j == x || !fixed_binary(j)) continue;
          const Index w = lb_[j] > 0.5 ? 1 : 0;
          if (stamp_[j] == x && lb0_[j] == ub0_[j] && lb0_[j] == lb_[j]) continue;
          if (last_lit_[j] == 2 * x + 1) continue;
          last_lit_[j] = 2 * x + 1;
          out.conf.push_back({2 * x + 1, 2 * j + 1 - w});
        }
      }
      for (const Change& c : trail_) {
        const Index j = c.j;
        if (j == x || stamp_[j] != x) continue;
        stamp_[j] = -1;
        const double l = std::min(lb0_[j], lb_[j]), u = std::max(ub0_[j], ub_[j]);
        const double rl = root_lb(j), ru = root_ub(j);
        if (l > rl || u < ru) out.gj.push_back(j), out.gl.push_back(std::max(l, rl)), out.gu.push_back(std::min(u, ru));
        double v0, v1;
        if (fixed_value(j, lb0_[j], ub0_[j], &v0) && fixed_value(j, lb_[j], ub_[j], &v1) && v0 != v1) {
          out.equiv.push_back({j, x, v0, v1});
        }
      }
    }
    undo();
    out.work = work_ - work0;
  }

  // Prober 0: tightens j to [l, u] intersected with its bounds; false if they cross (beyond rounding errors of a
  // continuous column, which leave the bounds as they are).
  bool tighten_global(Index j, double l, double u, bool* changed) {
    const double nl = std::max(l, lb_[j]), nu = std::min(u, ub_[j]);
    if (nl > nu) return !p_.is_int[j] && nl - nu <= 1e-6 * (1.0 + std::abs(nl));
    if (nl > lb_[j] || nu < ub_[j]) set_bounds(j, nl, nu), *changed = true;
    return true;
  }

  // Prober 0, after a propagation on top of the committed bounds: the columns changed, with their new bounds,
  // are appended to the list the other probers take over; then the bounds are committed.
  void commit_to(std::vector<Change>& sync) {
    for (size_t t = 0; t < trail_.size(); ++t) {
      const Index j = trail_[t].j;
      if (first_[j] == static_cast<Index>(t)) sync.push_back({j, lb_[j], ub_[j]});
    }
    commit();
  }

  // The other probers: the bound changes and conflicts prober 0 committed in the last batch.
  void take_over(const std::vector<Change>& sync, const std::vector<std::pair<Index, Index>>& conf) {
    for (const Change& c : sync) {
      if (c.lb > lb_[c.j] || c.ub < ub_[c.j]) set_bounds(c.j, c.lb, c.ub);
    }
    clear_queue();
    commit();
    for (const auto& [a, b] : conf) add_conflict(a, b);
  }

  static ProbingResult run_batched(const MipProblem& p, long long work_limit, const ProbingOptions& opt, int B) {
    std::vector<std::unique_ptr<Prober>> pr(B);
    for (int k = 0; k < B; ++k) pr[k] = std::make_unique<Prober>(p, work_limit, opt);
    Prober& m = *pr[0];
    ProbingResult res;
    const Index n = p.num_cols();
    std::vector<Index> cand;
    for (Index j = 0; j < n; ++j) {
      if (m.is_binary(j)) cand.push_back(j);
    }
    std::stable_sort(cand.begin(), cand.end(),
                     [&](Index a, Index b) { return p.A.end(a) - p.A.begin(a) > p.A.end(b) - p.A.begin(b); });
    std::vector<char> skip(n, 0);
    for (int k = 0; k < B; ++k) {
      Prober& q = *pr[k];
      for (RowState& r : q.rs_) r.range = 0.0;
      for (Index j = 0; j < n; ++j) {
        for (Index t = p.A.begin(j); t < p.A.end(j); ++t) {
          const double r = std::abs(p.A.value[t]) * (q.ub_[j] - q.lb_[j]);
          double& rg = q.rs_[p.A.index[t]].range;
          rg = std::max(rg, std::isfinite(r) && r < kBigBound ? r : kInf);
        }
      }
      q.init_activities();
    }
    // Helper threads (B - 1 at most). Between batches they spin for a moment before they sleep on the condition
    // variable: a probe often takes less than a wake-up (mzzv11: 100 microseconds).
    const int T = std::max(1, std::min(B, opt.threads));
    std::mutex mu;
    std::condition_variable cv;
    std::function<void(int)> job;
    std::atomic<int> next{0}, gen{0}, finished{0};
    std::atomic<bool> stop{false};
    int njobs = 0;
    auto work = [&] {
      for (;;) {
        const int k = next.fetch_add(1, std::memory_order_relaxed);
        if (k >= njobs) break;
        job(k);
      }
    };
    std::vector<std::thread> crew;
    for (int t = 1; t < T; ++t) {
      crew.emplace_back([&] {
        int seen = 0;
        for (;;) {
          int spins = 0;
          while (gen.load(std::memory_order_acquire) == seen && !stop.load(std::memory_order_acquire)) {
            if (++spins < 4000) {
              std::this_thread::yield();
            } else {
              std::unique_lock<std::mutex> lk(mu);
              cv.wait(lk, [&] { return gen.load(std::memory_order_acquire) != seen || stop.load(std::memory_order_acquire); });
            }
          }
          if (stop.load(std::memory_order_acquire)) return;
          seen = gen.load(std::memory_order_acquire);
          work();
          finished.fetch_add(1, std::memory_order_release);
        }
      });
    }
    auto run_jobs = [&](int count, bool parallel) {
      if (!parallel || T == 1) {
        for (int k = 0; k < count; ++k) job(k);
        return;
      }
      njobs = count;
      next.store(0, std::memory_order_relaxed);
      finished.store(0, std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> lk(mu);
        gen.fetch_add(1, std::memory_order_release);
      }
      cv.notify_all();
      work();
      while (finished.load(std::memory_order_acquire) != T - 1) std::this_thread::yield();
    };
    std::vector<ProbeOut> out(B);
    std::vector<Change> sync;
    std::vector<std::pair<Index, Index>> sync_conf;
    std::vector<Index> xs;
    long long idle = 0, total = 0, last_batch = 1 << 30;
    static const bool allow_switch = [] { const char* v = std::getenv("MILP_PROBE_SWITCH"); return !(v && v[0] == '0'); }();
    static const double idle_env = [] { const char* v = std::getenv("MILP_PROBE_IDLE"); return v ? std::atof(v) : 0.1; }();
    const double idle_frac = opt.idle_frac >= 0.0 ? opt.idle_frac : idle_env;
    static const long long kParMinEnv = [] { const char* v = std::getenv("MILP_PROBE_PAR_MIN"); return v ? std::atoll(v) : 3000LL; }();
    const long long kParMin = opt.par_min >= 0 ? opt.par_min : kParMinEnv;
    size_t lo = 0, hi = cand.size();
    bool reversed = false, done = false;
    job = [&](int k) {
      if (k > 0) pr[k]->take_over(sync, sync_conf);  // prober 0 made these changes itself
      if (k < static_cast<int>(xs.size())) pr[k]->probe_one(xs[k], out[k]);
    };
    while (!done) {
      if (total > work_limit || interrupted()) break;
      if (idle > static_cast<long long>(idle_frac * static_cast<double>(work_limit))) {
        if (reversed || !allow_switch) break;
        reversed = true;
        idle = 0;
      }
      xs.clear();
      while (lo < hi && static_cast<int>(xs.size()) < B) {
        const Index x = reversed ? cand[--hi] : cand[lo++];
        if (skip[x] || m.lb_[x] == m.ub_[x]) continue;
        xs.push_back(x);
      }
      if (xs.empty()) break;
      // Small probes run on the calling thread: waking the helpers costs more than they save.
      run_jobs(B, last_batch >= kParMin * static_cast<long long>(xs.size()));
      sync.clear();
      sync_conf.clear();
      last_batch = 0;
      for (size_t k = 0; k < xs.size() && !done; ++k) {
        const ProbeOut& o = out[k];
        const Index x = o.x;
        total += o.work;
        last_batch += o.work;
        ++res.probes;
        if (m.lb_[x] == m.ub_[x]) continue;  // fixed by an earlier result of this batch
        if (!o.f0 && !o.f1) {
          res.infeasible = true;
          done = true;
          break;
        }
        if (o.f0 && o.f1) {
          for (const auto& [a, b] : o.conf) {
            if (m.num_conf_ >= m.max_conf_) break;
            m.add_conflict(a, b, &res);
            sync_conf.push_back({a, b});
          }
          for (const ProbingResult::Lift& l : o.lifts) m.keep_lift(l, res);
          bool useful = false, changed = false, ok = true;
          for (const ProbingResult::Equiv& e : o.equiv) {
            if (skip[e.y] || m.lb_[e.y] == m.ub_[e.y]) continue;
            res.equiv.push_back(e);
            skip[e.y] = 1;
            useful = true;
          }
          for (size_t t = 0; t < o.gj.size() && ok; ++t) ok = m.tighten_global(o.gj[t], o.gl[t], o.gu[t], &changed);
          idle = changed || useful ? 0 : idle + o.work;
          if (!ok || (changed && !m.propagate())) {
            res.infeasible = true;
            done = true;
            break;
          }
          if (changed) m.commit_to(sync);
        } else {
          idle = 0;
          const double v = o.f0 ? 0.0 : 1.0;
          m.set_bounds(x, v, v);
          if (!m.propagate()) {
            res.infeasible = true;
            done = true;
            break;
          }
          m.commit_to(sync);
        }
      }
    }
    {
      std::lock_guard<std::mutex> lk(mu);
      stop.store(true, std::memory_order_release);
    }
    cv.notify_all();
    for (std::thread& t : crew) t.join();
    res.lb = m.lb_;
    res.ub = m.ub_;
    res.work = total;
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
      if (up ? rs_[i].nmax > 0 : rs_[i].nmin > 0) continue;  // unbounded activity
      const double s = up ? 1.0 : -1.0;
      const double b = up ? p_.row_ub[i] : -p_.row_lb[i];
      const double need = std::max(1e-6 * std::max(1.0, std::abs(b)), 1e-3 * rowmax_[i]);
      // Cheap test on the incremental activities (they may drift a little; the exact sum decides).
      if (!(b - (up ? rs_[i].amax : -rs_[i].amin) > 0.5 * need)) continue;
      const Index rb = rows_.begin(i), re = rows_.end(i);
      work_ += re - rb;
      double act = 0.0, act0 = 0.0, mag = std::abs(b);
      // Exact arithmetic: integral data, or (lift_dyadic) multiples of 1/1024 below 1e9 with integral bounds, whose
      // products and sums are exact in a double.
      const double unit = lift_dyadic_ ? 1024.0 : 1.0;
      bool ok = true, has_x = false, exact = mag < 1e9 && b * unit == std::round(b * unit);
      for (Index k = rb; k < re && ok; ++k) {
        const Index j = rows_.index[k];
        const double a = s * rows_.value[k];
        has_x = has_x || j == x;
        const double h = a > 0 ? ub_[j] : lb_[j], h0 = a > 0 ? root_ub(j) : root_lb(j);
        ok = std::abs(h) < kBigBound && std::abs(h0) < kBigBound;
        act += a * h;
        act0 += a * h0;
        mag += std::abs(a * h0);
        exact = exact && a * unit == std::round(a * unit) && h == std::round(h) && h0 == std::round(h0) &&
                std::abs(a * h0) < 1e9;
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
    if (++touch_id_ == std::numeric_limits<int>::max()) {
      for (RowState& r : rs_) r.touch = -1;
      touch_id_ = 0;
    }
    ++phase_;
  }

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
    for (RowState& r : rs_) r.amin = r.amax = 0.0, r.nmin = r.nmax = 0;
    for (Index j = 0; j < p_.num_cols(); ++j) {
      for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
        double lo, hi;
        int li, hi_i;
        contrib(p_.A.value[k], lb_[j], ub_[j], &lo, &hi, &li, &hi_i);
        RowState& r = rs_[p_.A.index[k]];
        r.amin += lo, r.amax += hi, r.nmin += li, r.nmax += hi_i;
      }
    }
    updates_ = 0;
  }

  // Bounds of j change from [l0, u0] to the current ones: updates the activities of its rows.
  void update_activities(Index j, double l0, double u0, bool enqueue) {
    // Only the contributions of a changed bound are updated (same arithmetic as contrib()).
    const double l1 = lb_[j], u1 = ub_[j];
    const bool lch = l1 != l0, uch = u1 != u0;
    const bool save = enqueue && snap_on_;
    auto term = [](double a, double b, double* v, int* inf) {
      if (std::abs(b) < kBigBound) *v = a * b, *inf = 0; else *v = 0.0, *inf = 1;
    };
    for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
      const Index i = p_.A.index[k];
      RowState& r = rs_[i];
      const double a = p_.A.value[k];
      const bool pos = a > 0;
      double v0, v1;
      int i0, i1;
      if (save && r.exact && r.snap != snap_id_) {  // first change of an exact row since the trail was empty
        r.snap = snap_id_;
        snaps_.push_back({i, r.nmin, r.nmax, r.amin, r.amax});
      }
      if (pos ? lch : uch) {  // minimal contribution: a l (a > 0) or a u
        term(a, pos ? l0 : u0, &v0, &i0);
        term(a, pos ? l1 : u1, &v1, &i1);
        r.amin += v1 - v0, r.nmin += i1 - i0;
      }
      if (pos ? uch : lch) {  // maximal contribution
        term(a, pos ? u0 : l0, &v0, &i0);
        term(a, pos ? u1 : l1, &v1, &i1);
        r.amax += v1 - v0, r.nmax += i1 - i0;
      }
      if (enqueue && !r.queued) r.queued = 1, queue_.push_back(i);
      if (collect_ && r.touch != touch_id_) r.touch = touch_id_, touched_.push_back(i);
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
      if (snap_on_ && col_exact_[j]) {
        const Index len = p_.A.end(j) - p_.A.begin(j);
        updates_ += len;  // counted as the reversed updates were
        snap_entries_ += len;
      } else {
        update_activities(j, l, u, false);
      }
      first_[j] = -1;
    }
    for (const Snap& sn : snaps_) {
      RowState& r = rs_[sn.i];
      r.amin = sn.amin, r.amax = sn.amax, r.nmin = sn.nmin, r.nmax = sn.nmax;
    }
    snap_rows_ += static_cast<long long>(snaps_.size());
    trail_.clear();
    new_snaps();
    clear_queue();
    // Saving pays when a probe changes its rows several times each. When nearly every changed entry is a row's
    // first change (short chains: gfd-schedulen, 5% more probing time with the saving), the saving is switched
    // off for a while and the undo reverses the updates as before; for exact rows both give the same bits, so
    // the switch changes no result. The trail is empty here, so no saved values are pending.
    if (++snap_undos_ >= (snap_on_ ? 256 : 8192)) {
      snap_on_ = !snap_on_ || 2 * snap_rows_ <= snap_entries_;
      snap_undos_ = 0, snap_rows_ = 0, snap_entries_ = 0;
    }
  }

  // The trail is empty again: the next change of an exact row saves its activities anew.
  void new_snaps() {
    snaps_.clear();
    if (++snap_id_ == std::numeric_limits<int>::max()) {
      for (RowState& r : rs_) r.snap = -1;
      snap_id_ = 0;
    }
  }

  // Whether row i might tighten a bound or be infeasible (cheap test on the incremental activities; the
  // row is then processed exactly). range_[i] bounds every |a_ij| (u_j - l_j).
  bool may_propagate(Index i) const {
    const RowState& r = rs_[i];
    const double rlo = r.rlo, rhi = r.rhi;
    const double tol = 1e-6 * (1.0 + std::abs(r.amin) + std::abs(r.amax));
    if (std::isfinite(rhi) && (r.nmin == 1 || (r.nmin == 0 && rhi - r.amin < r.range + tol))) return true;
    if (std::isfinite(rlo) && (r.nmax == 1 || (r.nmax == 0 && r.amax - rlo < r.range + tol))) return true;
    return false;
  }

  // Keeps the current bounds as the new root.
  void commit() {
    for (size_t t = 0; t < trail_.size(); ++t) {
      const Index j = trail_[t].j;
      if (first_[j] == static_cast<Index>(t) && lb_[j] == ub_[j]) {  // fixed for good by these changes
        for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) count_dead(p_.A.index[k]);
      }
    }
    for (const Change& c : trail_) first_[c.j] = -1;
    trail_.clear();
    new_snaps();
  }

  // Drops the columns fixed for good from row i's copy for the tightening loop, keeping the order of the others
  // (the loop skips fixed columns anyway, so it does the same steps in the same order; ex10: 24% of the entries
  // it walked were such columns, neos8 60%). Columns fixed only by the current probe stay.
  void count_dead(Index i) {
    if (rs_[i].dead < 255) ++rs_[i].dead;
  }

  void compact_row(Index i) {
    RowState& r = rs_[i];
    Index w = rows_.begin(i);
    for (Index k = w; k < r.pend; ++k) {
      const Index j = pidx_[k];
      if (root_lb(j) == root_ub(j)) continue;
      pidx_[w] = j, pval_[w] = pval_[k], ++w;
    }
    r.pend = w;
    r.dead = 0;
  }

  void clear_queue() {
    for (Index i : queue_) rs_[i].queued = 0;
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
        rs_[i].queued = 0;
        if (!process_row(i)) {
          for (size_t h = head + 1; h < queue_.size(); ++h) rs_[queue_[h]].queued = 0;
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

  // Whether the bound computation of process_row for an integer column with finite bounds [l, u] is certain to
  // change nothing, decided without its three divisions. Nearly every column that passes the row's gate is this
  // case (mzzv11: 153 of 155 million computations, ex10 66 of 68 million): the side's slack equals the column's
  // span, as for a free binary in x_1 + ... + x_k <= 1 with the others at 0, where the gate (slack below span
  // plus a 1e-9 margin) lets it through and the computation returns u again.
  // Claim, for a side in use: if span <= slack then the computed bound is not inside [l, u]. With exact
  // arithmetic the bound is u + (slack - span) / |a| >= u (or <= l). The computed value differs from that by at
  // most a few rounding errors of the magnitudes involved, (|side| + |activity| + |a b|) / |a| * 1e-15, and the
  // computation adds a tolerance of 1e-6 (1 + |rmin| + |rmax|) / |a| before rounding down (up): when the
  // magnitudes are below 1e8 (1 + |rmin| + |rmax|), the tolerance is over ten times the error, so the rounded
  // bound is >= u (<= l) since u and l are integers (int_bounds_: checked once for the start bounds; every later
  // bound of an integer column is a rounded value or one of these). A side not in use gives no bound at all.
  bool no_change_for_sure(bool integer, double a, double l, double u, double lo, double hi, double minact,
                          double maxact, int mininf, int maxinf, bool use_hi, bool use_lo, double rlo, double rhi,
                          double rmin, double rmax, double slack_hi, double slack_lo) const {
    if (!integer || !int_bounds_ || !std::isfinite(l) || !std::isfinite(u)) return false;
    const double span = std::abs(a) * (u - l);
    const double room = 1e8 * (1.0 + std::abs(rmin) + std::abs(rmax));
    if (use_hi && mininf == 0 &&
        !(span <= slack_hi && std::abs(rhi) + std::abs(minact) + std::abs(a * lo) <= room)) {
      return false;
    }
    if (use_lo && maxinf == 0 &&
        !(span <= slack_lo && std::abs(rlo) + std::abs(maxact) + std::abs(a * hi) <= room)) {
      return false;
    }
    return true;
  }

  bool process_row(Index i) {
    ++work_;
    if (!may_propagate(i)) return true;
    const Index rb = rows_.begin(i), re = rows_.end(i);
    work_ += re - rb;
    double minact = 0.0, maxact = 0.0;
    int mininf = 0, maxinf = 0;
    if (rs_[i].row_int) {
      // Rows of columns with moderate finite bounds: the incremental activities are accurate (drift ~1e-16 relative, far
      // below the rounding margins; they are recomputed periodically).
      minact = rs_[i].amin, maxact = rs_[i].amax, mininf = rs_[i].nmin, maxinf = rs_[i].nmax;
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
    const double rlo = rs_[i].rlo, rhi = rs_[i].rhi;
    const double scale = 1.0 + std::max(std::abs(minact), std::abs(maxact));
    if ((mininf == 0 && minact > rhi + kFeasTol * scale) || (maxinf == 0 && maxact < rlo - kFeasTol * scale)) {
      return false;
    }
    const bool use_hi = std::isfinite(rhi) && mininf <= 1, use_lo = std::isfinite(rlo) && maxinf <= 1;
    if (!use_hi && !use_lo) return true;
    // A column with a finite domain can only be tightened by a side whose slack is below |a| (u - l).
    const double slack_hi = mininf == 0 ? rhi - minact : -kInf, slack_lo = maxinf == 0 ? maxact - rlo : -kInf;
    const double gate_hi = slack_hi - 1e-9 * (1.0 + std::abs(slack_hi)), gate_lo = slack_lo - 1e-9 * (1.0 + std::abs(slack_lo));
    if (rs_[i].dead >= 4 && (4 * rs_[i].dead >= rs_[i].pend - rb || rs_[i].dead == 255)) compact_row(i);
    const Index pe = rs_[i].pend;
    for (Index k = rb; k < pe; ++k) {
      const Index j = pidx_[k];
      if (lb_[j] == ub_[j]) continue;
      const double a = pval_[k];
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
#ifndef MILP_PROBE_SHORTCUT_CHECK
      if (no_change_for_sure(integer, a, l, u, lo, hi, minact, maxact, mininf, maxinf, use_hi, use_lo, rlo, rhi, rmin,
                             rmax, slack_hi, slack_lo)) {
        continue;
      }
#endif
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
#ifdef MILP_PROBE_SHORTCUT_CHECK
      if ((nl > lb_[j] || nu < ub_[j]) && no_change_for_sure(integer, a, l, u, lo, hi, minact, maxact, mininf, maxinf,
                                                              use_hi, use_lo, rlo, rhi, rmin, rmax, slack_hi, slack_lo)) {
        std::fprintf(stderr, "probing shortcut wrong: a %.17g [%.17g, %.17g] -> [%.17g, %.17g]\n", a, l, u, nl, nu);
        std::abort();
      }
#endif
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
  // Per-row state in one cache line: a probe walks the rows of the changed columns in no order, and with the
  // fields in seven separate arrays every row visit of a large model was seven cache misses (ex10, 70,000 rows:
  // the propagation's row test and the activity updates were 90% of the probing time).
  struct alignas(64) RowState {
    double amin = 0.0, amax = 0.0;  // incremental row activities
    double range = 0.0;             // max entry range |a_ij| (u_j - l_j) of the row
    double rlo = 0.0, rhi = 0.0;    // row bounds (copy of p_.row_lb / row_ub)
    int nmin = 0, nmax = 0;         // infinite contributions
    int touch = -1;                 // liftings: stamp of the probe that changed the row's activities (touch_id_)
    int pend = 0;                   // end of the row in pidx_ / pval_ (compact_row)
    int snap = -1;                  // exact rows: stamp (snap_id_) of the trail in which the activities were saved
    unsigned char dead = 0;         // columns fixed for good still in that copy (stops counting at 255)
    char exact = 0;                 // activities free of rounding errors (see the constructor)
    char queued = 0;
    char row_int = 1;  // incremental activities usable: finite, moderate bounds (continuous: <= 1e7)
  };
  static_assert(sizeof(RowState) == 64, "one cache line");
  std::vector<RowState> rs_;
  std::vector<Index> pidx_;   // rows_.index / rows_.value for the tightening loop of process_row; row i's live
  std::vector<double> pval_;  // part is [rows_.begin(i), rs_[i].pend)
  std::vector<Index> queue_;
  long long updates_ = 0;
  struct Snap {
    Index i;
    int nmin, nmax;
    double amin, amax;
  };
  std::vector<Snap> snaps_;  // activities of the exact rows changed since the trail was empty, as they were then
  int snap_id_ = 0;
  std::vector<char> col_exact_;
  bool int_bounds_ = true;  // every integer column started with integral bounds (no_change_for_sure)
  bool snap_on_ = true;  // saving in use (undo() switches it by the share of first changes)
  int snap_undos_ = 0;
  long long snap_rows_ = 0, snap_entries_ = 0;  // rows saved / column entries not walked since the last decision
  // Liftings: rows whose activities the current probe changed (touch_ stamps them with touch_id_), the
  // candidates of the x = 0 / x = 1 probes, the index of each row's kept lifting in the result.
  bool lift_ = false, collect_ = false;
  double idle_opt_ = -1.0;  // ProbingOptions::idle_frac
  bool lift_dyadic_ = false;  // ProbingOptions::lift_dyadic
  int touch_id_ = 0;
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
  if (opt.batch >= 2) return Prober::run_batched(p, work_limit, opt, opt.batch);
  Prober pr(p, work_limit, opt);
  return pr.run();
}

}  // namespace milp
