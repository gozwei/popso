#include "milp/heuristics/submip.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>

#include "milp/core/interrupt.hpp"
#include "milp/core/log.hpp"
#include "milp/core/timer.hpp"
#include "milp/presolve/presolve.hpp"
#include "milp/search/branch_and_bound.hpp"
#include "milp/search/propagation.hpp"

namespace milp {

namespace {

// Parameter override from the environment (tuning experiments), as the search's knob().
double sub_knob(const char* name, double def) {
  const char* v = std::getenv(name);
  return v ? std::atof(v) : def;
}

}  // namespace

bool solve_sub_mip(const MipProblem& p, const std::vector<double>& lb, const std::vector<double>& ub, double cutoff,
                   const SubMipLimits& limits, std::vector<double>& sol) {
  if (limits.fingerprint || limits.skip) {
    std::uint64_t h = 1469598103934665603ULL;  // FNV-1a over the integer columns' bounds
    for (Index j = 0; j < p.num_cols(); ++j) {
      if (!p.is_int[j]) continue;
      for (double v : {lb[j], ub[j]}) {
        std::uint64_t b;
        std::memcpy(&b, &v, sizeof b);
        h = (h ^ b) * 1099511628211ULL;
      }
    }
    if (limits.fingerprint) *limits.fingerprint = h;
    if (limits.skip && std::find(limits.skip->begin(), limits.skip->end(), h) != limits.skip->end()) return false;
  }
  MipProblem q = p;
  q.col_lb = lb;
  q.col_ub = ub;
  SolverParams sp;
  sp.threads = 1;
  sp.verbosity = 0;
  sp.node_limit = limits.node_limit;
  sp.time_limit = limits.time_limit;
  sp.mem_limit_mb = limits.mem_limit_mb;
  sp.mip_rel_gap = 1e-3;
  // MILP_SUBMIP_VERBOSE=<v>: the sub-MIPs' own log at verbosity v (diagnostics; interleaves with the main log).
  static const int verbose = [] {
    const char* e = std::getenv("MILP_SUBMIP_VERBOSE");
    return e ? std::atoi(e) : 0;
  }();
  Logger log(verbose);
  Timer timer;
  SearchOptions opts;
  opts.cutoff = cutoff;
  opts.sub_mip_heuristics = false;
  opts.stop_flag = limits.stop;
  opts.progress = limits.progress;
  opts.iteration_limit = limits.iteration_limit;
  opts.work_limit = limits.work_limit;
  opts.solution_limit = limits.solution_limit;
  opts.check_resident = false;
  // The sub-MIP's root cut loop: MILP_SUB_CUT_ROUNDS = k (default -1: as the main search) at most k rounds
  // (no cheap extra rounds); MILP_SUB_CUTS = 0 (default 1): none; MILP_SUB_CUTS_FREE = f (default 0: off):
  // none when at most f integer columns are left free. pg5_34's sub-MIPs (225 x 2600, 70-98 of 100 integers
  // fixed) spend 0.3-1 s of their 1.3-3 s in the root's MIR separation, then 1-25 nodes; swath's
  // (2600-4800 columns after presolve) 0.2-2.3 s in the cut probing.
  static const int sub_cut_rounds = static_cast<int>(sub_knob("MILP_SUB_CUT_ROUNDS", -1));
  static const bool sub_cuts = sub_knob("MILP_SUB_CUTS", 1) > 0;
  static const double sub_cuts_free = sub_knob("MILP_SUB_CUTS_FREE", 0);
  Index free_ints = 0;
  for (Index j = 0; j < p.num_cols(); ++j) free_ints += p.is_int[j] && lb[j] < ub[j];
  opts.root_cut_rounds = sub_cut_rounds;
  opts.root_cuts = sub_cuts && !(sub_cuts_free > 0 && static_cast<double>(free_ints) <= sub_cuts_free);
  // Presolve large problems first: the fixings usually remove most of them. Small ones are
  // solved as they are (presolve saves little there, and on timtab1 its aggregations of
  // general integers made the sub-MIPs markedly less successful).
  static const bool no_presolve = std::getenv("MILP_SUBMIP_NO_PRESOLVE") != nullptr;
  MipResult r;
  Presolver pre;
  PresolveOptions po;
  po.probing_effort = 0.2;
  // A sub-MIP's presolve also stops with the sub-MIP (FJ repair on supportcase19, 1.4M columns: 4 s past the
  // solve's end in parallel_cols).
  po.stop = limits.stop;
  po.time_limit = limits.time_limit;
  static const long long presolve_nnz = [] {
    const char* e = std::getenv("MILP_SUBMIP_PRESOLVE_NNZ");
    return e ? std::atoll(e) : 20000LL;
  }();
  // (Presolve does not know the implications of logical constraints.)
  const bool presolve = !no_presolve && p.A.nnz() >= presolve_nnz && p.implications.empty() && !p.nonlinear();
  // Unbounded (or skipped): solve the unreduced problem, which classifies it precisely.
  const PresolveStatus ps = presolve ? pre.run(q, po, log) : PresolveStatus::Unbounded;
  const double t_presolve = timer.elapsed();
  if (ps == PresolveStatus::Infeasible) {
    r.status = Status::Infeasible;
  } else if (ps == PresolveStatus::Reduced && pre.reduced().num_cols() == 0) {
    r.x = pre.postsolve({});
    r.has_solution = q.max_violation(r.x) <= 1e-6;
    r.objective = q.objective(r.x);
    r.status = r.has_solution ? Status::Optimal : Status::Infeasible;
  } else if (ps == PresolveStatus::Reduced) {
    sp.time_limit = std::max(0.0, limits.time_limit - timer.elapsed());
    r = branch_and_bound(pre.reduced(), sp, log, timer, opts);
    if (r.has_solution) {
      r.x = pre.postsolve(r.x);
      if (q.max_violation(r.x) > 1e-6) {
        r.has_solution = false;  // postsolve inexact: drop it
        if (std::getenv("MILP_SUBMIP_DEBUG")) {
          std::fprintf(stderr, "sub-MIP: postsolve violation %g\n", q.max_violation(r.x));
        }
      }
      r.objective = q.objective(r.x);
    }
  } else {
    r = branch_and_bound(q, sp, log, timer, opts);
  }
  Index fixed = 0, ints = 0;
  for (Index j = 0; j < p.num_cols(); ++j) ints += p.is_int[j], fixed += p.is_int[j] && lb[j] == ub[j];
  if (SubMipStats* st = limits.stats) {
    ++st->solves;
    st->exhausted = r.status == Status::Optimal || r.status == Status::Infeasible;
    st->lp_iterations += r.lp_iterations;
    st->lp_work += r.lp_work;
    st->dive_iterations += r.dive_iterations;
    st->fp_iterations += r.fp_iterations;
    st->dive_wins += r.dive_wins;
    st->fp_runs += r.fp_runs;
    st->fp_wins += r.fp_wins;
    st->nodes += r.nodes;
    st->fixed = fixed;
    st->ints = ints;
  }
  if (std::getenv("MILP_SUBMIP_DEBUG")) {
    std::fprintf(stderr, "sub-MIP: fixed %d/%d ints, status %s, nodes %lld, sol %d obj %.10g, %.2fs (presolve %.2fs)\n",
                 fixed, ints, to_string(r.status), r.nodes, r.has_solution, r.objective, timer.elapsed(), t_presolve);
  }
  if (!r.has_solution || r.objective >= cutoff) return false;
  sol = std::move(r.x);
  return true;
}

namespace {

bool enough_fixed(const MipProblem& p, const std::vector<double>& lb, const std::vector<double>& ub, double frac) {
  Index ints = 0, fixed = 0;
  for (Index j = 0; j < p.num_cols(); ++j) {
    if (!p.is_int[j]) continue;
    ++ints;
    fixed += lb[j] == ub[j];
  }
  return ints > 0 && fixed >= frac * ints && fixed < ints;
}

}  // namespace

bool rens(const MipProblem& p, const double* lp_x, double cutoff, const SubMipLimits& limits,
          std::vector<double>& sol, const std::vector<double>* redcost) {
  // Integral integer variables, most "certain" (largest |reduced cost|) first.
  std::vector<std::pair<double, Index>> integral;
  for (Index j = 0; j < p.num_cols(); ++j) {
    if (!p.is_int[j]) continue;
    if (std::abs(lp_x[j] - std::round(lp_x[j])) <= 1e-6) integral.emplace_back(redcost ? -std::abs((*redcost)[j]) : 0.0, j);
  }
  std::stable_sort(integral.begin(), integral.end());
  Timer timer;
  SubMipStats local;
  SubMipLimits base = limits;
  base.stats = &local;
  auto merge_stats = [&]() {
    if (!limits.stats) return;
    SubMipStats& st = *limits.stats;
    st.solves += local.solves;
    st.exhausted = local.exhausted;
    st.lp_iterations += local.lp_iterations;
    st.lp_work += local.lp_work;
    st.dive_iterations += local.dive_iterations;
    st.fp_iterations += local.fp_iterations;
    st.dive_wins += local.dive_wins;
    st.fp_runs += local.fp_runs;
    st.fp_wins += local.fp_wins;
    st.nodes += local.nodes;
    st.fixed = local.fixed;
    st.ints = local.ints;
  };
  bool quick_infeasible = true;  // every sub-MIP so far was infeasible at (or right after) its root
  for (double keep : {1.0, 0.7, 0.4}) {
    if (keep < 1.0 && !redcost) break;
    std::vector<double> lb = p.col_lb, ub = p.col_ub;
    for (Index j = 0; j < p.num_cols(); ++j) {
      if (!p.is_int[j] || std::abs(lp_x[j] - std::round(lp_x[j])) <= 1e-6) continue;
      lb[j] = std::max(lb[j], std::floor(lp_x[j]));
      ub[j] = std::min(ub[j], std::ceil(lp_x[j]));
    }
    const size_t nfix = static_cast<size_t>(keep * static_cast<double>(integral.size()));
    for (size_t k = 0; k < nfix; ++k) {
      const Index j = integral[k].second;
      lb[j] = ub[j] = std::round(lp_x[j]);
    }
    const double min_frac = keep < 1.0 ? 0.0 : limits.min_fixed_fraction;
    if (!enough_fixed(p, lb, ub, min_frac)) {
      if (std::getenv("MILP_SUBMIP_DEBUG")) std::fprintf(stderr, "RENS skipped: too few fixed\n");
      merge_stats();
      return false;
    }
    SubMipLimits lim = base;  // the retries share the time limit
    lim.time_limit = limits.time_limit - timer.elapsed();
    if (lim.time_limit <= 0.0 || interrupted()) break;
    const long long nodes0 = local.nodes;
    if (solve_sub_mip(p, lb, ub, cutoff, lim, sol)) {
      merge_stats();
      return true;
    }
    quick_infeasible = quick_infeasible && local.exhausted && local.nodes - nodes0 <= 1;
  }
  // Fix-and-propagate (as HiGHS' RENS): the plain fixings were infeasible at once (30n20b8: every
  // integral LP value fixed, or the 70% / 40% with the largest reduced costs). Fix one integer at a
  // time at its LP value, most certain first, propagate, and skip (undo) a fixing that propagation
  // proves infeasible; fractional integers are first restricted to floor/ceil the same way.
  static const bool no_prop_rens = std::getenv("MILP_NO_PROP_RENS") != nullptr;
  if (no_prop_rens || !quick_infeasible || limits.time_limit - timer.elapsed() <= 0.0) {
    merge_stats();
    return false;
  }
  DomainPropagator prop(p);
  std::vector<double> lb = p.col_lb, ub = p.col_ub, slb = lb, sub = ub;  // current / last committed
  std::vector<Index> seeds(1), tight;
  long long work = 0;
  const long long work_limit = 400LL * p.A.nnz() + 1000000;
  auto try_bounds = [&](Index j, double l, double u) {
    if (l <= lb[j] && u >= ub[j]) return true;
    lb[j] = std::max(lb[j], l);
    ub[j] = std::min(ub[j], u);
    seeds[0] = j;
    tight.clear();
    bool ok = lb[j] <= ub[j];
    if (ok && work < work_limit) {
      ok = prop.propagate(lb, ub, seeds, tight);
      work += prop.last_work;
    }
    if (ok) {
      slb[j] = lb[j], sub[j] = ub[j];
      for (Index k : tight) slb[k] = lb[k], sub[k] = ub[k];
    } else {
      lb[j] = slb[j], ub[j] = sub[j];
      for (Index k : tight) lb[k] = slb[k], ub[k] = sub[k];
    }
    return ok;
  };
  for (Index j = 0; j < p.num_cols(); ++j) {
    if (!p.is_int[j] || std::abs(lp_x[j] - std::round(lp_x[j])) <= 1e-6) continue;
    try_bounds(j, std::floor(lp_x[j]), std::ceil(lp_x[j]));
  }
  Index skipped = 0;
  for (const auto& [score, j] : integral) {
    (void)score;
    const double v = std::round(lp_x[j]);
    if (v < lb[j] || v > ub[j]) {
      ++skipped;  // propagation moved it away from its LP value: leave it to the sub-MIP
      continue;
    }
    if (!try_bounds(j, v, v)) ++skipped;
  }
  if (std::getenv("MILP_SUBMIP_DEBUG")) std::fprintf(stderr, "RENS fix-and-propagate: %d of %zu integral values not fixed, work %lld\n", skipped, integral.size(), work);
  if (!enough_fixed(p, lb, ub, 0.0)) {
    merge_stats();
    return false;
  }
  SubMipLimits lim = base;
  lim.time_limit = limits.time_limit - timer.elapsed();
  const bool ok = solve_sub_mip(p, lb, ub, cutoff, lim, sol);
  merge_stats();
  return ok;
}

namespace {

// Presolve (with light probing) proves p restricted to [lb, ub] infeasible: the LP check misses
// this when the fixings are at the LP point's own values (puriri: every sub-MIP with the integral
// LP values fixed was infeasible in presolve, its LP feasible).
bool presolve_infeasible(const MipProblem& p, const std::vector<double>& lb, const std::vector<double>& ub) {
  if (!p.implications.empty() || p.nonlinear()) return false;
  MipProblem q = p;
  q.col_lb = lb;
  q.col_ub = ub;
  Presolver pre;
  PresolveOptions po;
  po.probing_effort = 0.2;
  Logger log(0);
  return pre.run(q, po, log) == PresolveStatus::Infeasible;
}

}  // namespace

bool lp_rens(const MipProblem& p, const DualSimplex& root, double cutoff, const SubMipLimits& limits,
             std::vector<double>& sol, double target, int attempts, LpRensStats* out, bool stop_blind) {
  static const bool debug = std::getenv("MILP_SUBMIP_DEBUG") != nullptr;
  static const bool no_pcheck = std::getenv("MILP_LP_RENS_NO_PRESOLVE") != nullptr;
  Timer timer;
  const Index n = p.num_cols();
  std::vector<Index> cols;  // integer columns free in the root LP
  for (Index j = 0; j < n; ++j) {
    if (p.is_int[j] && root.col_lower(j) < root.col_upper(j)) cols.push_back(j);
  }
  const Index ints = static_cast<Index>(cols.size());
  LpRensStats local;
  LpRensStats& st = out ? *out : local;
  if (ints == 0 || attempts <= 0) return false;
  DomainPropagator prop(p);
  SubMipStats ss_total;
  // Shared budget (time; deterministic: LP iterations incl. the dives) of all attempts.
  auto used_iters = [&] { return st.dive_iterations + ss_total.lp_iterations; };
  long long dive_work = 0;
  auto used_work = [&] { return dive_work + ss_total.lp_work; };
  // A work budget reaches the dives' LP solves as iterations at the root LP's work per iteration so far.
  const long long work_per_iter = std::max<long long>(1, root.work() / std::max<long long>(1, root.iterations()));
  auto finish = [&](bool ok) {
    if (limits.stats) {
      SubMipStats& o = *limits.stats;
      o.solves += ss_total.solves;
      o.lp_iterations += ss_total.lp_iterations + st.dive_iterations;
      o.lp_work += used_work();
      o.dive_iterations += ss_total.dive_iterations;
      o.fp_iterations += ss_total.fp_iterations;
      o.dive_wins += ss_total.dive_wins;
      o.fp_runs += ss_total.fp_runs;
      o.fp_wins += ss_total.fp_wins;
      o.nodes += ss_total.nodes;
      o.exhausted = ss_total.exhausted;
      o.fixed = ss_total.fixed;
      o.ints = ss_total.ints;
    }
    return ok;
  };
  const double target0 = std::clamp(target, 0.05, 0.98);
  double rate = target0;
  // Fixing rates known too small (sub-MIP hit its limits without a solution) / too large (sub-MIP
  // exhausted: infeasible or nothing better), with the dive's own fixings of those attempts.
  double lo = 0.0, hi = 1.0;
  std::vector<Index> lo_set, hi_set;
  int narrowings = 0;
  std::vector<char> banned(n, 0);  // fixings rejected (LP, presolve, or the culprit range of a narrowing)
  std::vector<std::pair<double, Index>> cand;
  std::vector<Index> seeds, tight, order;
  constexpr int kMaxPresolveChecks = 40;  // per call (each copies and presolves the problem)
  int pchecks = 0;
  for (int a = 0; a < attempts; ++a) {
    const int left = attempts - a;
    if (limits.time_limit - timer.elapsed() <= 0.0 || interrupted() || (limits.stop && limits.stop->load())) break;
    if (limits.iteration_limit >= 0 && used_iters() >= limits.iteration_limit) break;
    if (limits.work_limit >= 0 && used_work() >= limits.work_limit) break;
    DualSimplex lp = root;
    const long long work_start = lp.work();
    std::vector<double> lb(n), ub(n);
    for (Index j = 0; j < n; ++j) lb[j] = lp.col_lower(j), ub[j] = lp.col_upper(j);
    struct Saved {
      Index j;
      double lb, ub;
    };
    std::vector<Saved> saved;
    auto set = [&](Index j, double l, double u) {
      saved.push_back({j, lb[j], ub[j]});
      lb[j] = l, ub[j] = u;
      lp.set_col_bounds(j, l, u);
    };
    auto undo = [&](size_t mark) {
      for (; saved.size() > mark; saved.pop_back()) {
        const Saved& s = saved.back();
        lb[s.j] = s.lb, ub[s.j] = s.ub;
        lp.set_col_bounds(s.j, s.lb, s.ub);
      }
    };
    auto count_fixed = [&] {
      Index k = 0;
      for (Index j : cols) k += lb[j] == ub[j];
      return k;
    };
    order.clear();  // the dive's own fixings, in order
    const Index want = std::max<Index>(1, static_cast<Index>(rate * static_cast<double>(ints)));
    const long long it_start = lp.iterations();
    // The dive may use up to half of this attempt's share of the budget.
    const long long dive_budget =
        limits.iteration_limit >= 0 ? std::max<long long>(1000, (limits.iteration_limit - used_iters()) / (2 * left))
        : limits.work_limit >= 0 ? std::max<long long>(1000, (limits.work_limit - used_work()) / (2 * left) / work_per_iter)
                                 : -1;
    const double dive_time = (limits.time_limit - timer.elapsed()) / (2.0 * left);
    const double t_dive = timer.elapsed();
    Index fixed = count_fixed();
    Index batch = std::max<Index>(1, (want - fixed) / 8);
    bool lp_ok = true;
    while (lp_ok && fixed < want) {
      const long long used = lp.iterations() - it_start;
      if ((dive_budget >= 0 && used >= dive_budget) || timer.elapsed() - t_dive > dive_time || interrupted()) break;
      const double* x = lp.x();
      const std::vector<double> d = lp.reduced_costs();
      cand.clear();
      Index integral = 0;
      for (Index j : cols) {
        if (lb[j] == ub[j] || banned[j]) continue;
        const double dist = std::abs(x[j] - std::round(x[j]));
        if (dist <= 1e-6) {
          // Most certain first: largest |reduced cost|, then the longest columns.
          cand.emplace_back(-(std::abs(d[j]) + 1e-9 * static_cast<double>(p.A.end(j) - p.A.begin(j))), j);
          ++integral;
        } else {
          cand.emplace_back(1.0 + dist, j);  // after every integral one: least fractional first
        }
      }
      if (cand.empty()) break;
      std::sort(cand.begin(), cand.end());
      const Index k = integral > 0 ? std::min({batch, want - fixed, integral}) : 1;
      const size_t mark = saved.size();
      seeds.clear();
      for (Index t = 0; t < k; ++t) {
        const Index j = cand[t].second;
        const double v = std::clamp(std::round(x[j]), lb[j], ub[j]);
        set(j, v, v);
        seeds.push_back(j);
      }
      tight.clear();
      std::vector<double> plb = lb, pub = ub;
      bool ok = prop.propagate(plb, pub, seeds, tight);
      if (ok) {
        for (Index t : tight) {
          if (plb[t] != lb[t] || pub[t] != ub[t]) set(t, plb[t], pub[t]);
        }
        LpOptions opt;
        opt.objective_cutoff = cutoff;
        if (dive_budget >= 0) opt.iteration_limit = std::max<long long>(100, dive_budget - used);
        opt.time_limit = std::max(0.0, dive_time - (timer.elapsed() - t_dive));
        const LpStatus ls = lp.solve(opt);
        ok = ls == LpStatus::Optimal && lp.objective() < cutoff;
        if (!ok && ls != LpStatus::Infeasible && ls != LpStatus::CutoffReached) {
          undo(mark);
          LpOptions ropt;
          if (dive_budget >= 0) ropt.iteration_limit = std::max<long long>(1000, dive_budget);
          // Within the call's time too (CBLIB uflquad-nopsc-30-300, 45300 rows: this re-solve ran 30 s past a 1.4 s
          // budget and the 30 s time limit).
          ropt.time_limit = std::max(0.0, limits.time_limit - timer.elapsed());
          lp_ok = lp.solve(ropt) == LpStatus::Optimal;  // limits: restore an LP point, stop the dive
          break;
        }
        // The LP accepts fixings at its own values without moving; presolve (light probing) on
        // the restricted problem catches some of what it misses.
        if (ok && !no_pcheck && ++pchecks <= kMaxPresolveChecks && presolve_infeasible(p, lb, ub)) {
          ok = false;
          ++st.presolve_rejects;
        }
      }
      if (ok) {
        for (Index t = 0; t < k; ++t) order.push_back(cand[t].second);
        fixed = count_fixed();
        batch = std::min<Index>(batch * 2, std::max<Index>(1, want / 8));
        continue;
      }
      undo(mark);
      if (k == 1) {
        banned[cand[0].second] = 1;
      } else {
        batch = std::max<Index>(1, k / 2);
      }
      LpOptions opt;  // back to the LP point of the restored bounds
      if (dive_budget >= 0) opt.iteration_limit = std::max<long long>(1000, dive_budget - (lp.iterations() - it_start));
      opt.time_limit = std::max(0.0, limits.time_limit - timer.elapsed());
      lp_ok = lp.solve(opt) == LpStatus::Optimal;
    }
    st.dive_iterations += lp.iterations() - it_start;
    dive_work += lp.work() - work_start;
    fixed = count_fixed();
    const double achieved = static_cast<double>(fixed) / static_cast<double>(ints);
    if (fixed == 0 || fixed == ints) {
      if (debug) std::fprintf(stderr, "LP-RENS attempt %d: fixed %d/%d, skipped\n", a, fixed, ints);
      break;
    }
    SubMipLimits lim = limits;
    SubMipStats ss;
    lim.stats = &ss;
    lim.min_fixed_fraction = 0.0;
    lim.time_limit = (limits.time_limit - timer.elapsed()) / static_cast<double>(left);
    if (limits.iteration_limit >= 0) {
      lim.iteration_limit = std::max<long long>(1000, (limits.iteration_limit - used_iters()) / left);
    }
    if (limits.work_limit >= 0) lim.work_limit = std::max<long long>(1000000, (limits.work_limit - used_work()) / left);
    const bool found = solve_sub_mip(p, lb, ub, cutoff, lim, sol);
    ss_total.solves += ss.solves;
    ss_total.lp_iterations += ss.lp_iterations;
    ss_total.lp_work += ss.lp_work;
    ss_total.dive_iterations += ss.dive_iterations;
    ss_total.fp_iterations += ss.fp_iterations;
    ss_total.dive_wins += ss.dive_wins;
    ss_total.fp_runs += ss.fp_runs;
    ss_total.fp_wins += ss.fp_wins;
    ss_total.nodes += ss.nodes;
    ss_total.exhausted = ss.exhausted;
    ss_total.fixed = ss.fixed;
    ss_total.ints = ss.ints;
    ++st.attempts;
    st.last_rate = achieved;
    if (debug) {
      std::fprintf(stderr,
                   "LP-RENS attempt %d: target %.2f fixed %d/%d (%.2f), dive %lld iters, %d presolve rejects, sub-MIP %s "
                   "%lld nodes, %.2fs\n",
                   a, rate, fixed, ints, achieved, st.dive_iterations, st.presolve_rejects,
                   found ? "found" : (ss.exhausted ? "exhausted" : "limit"), ss.nodes, timer.elapsed());
    }
    if (found) return finish(true);
    if (stop_blind && a == 0 && ss.exhausted) {
      st.stopped_blind = true;
      if (debug) std::fprintf(stderr, "LP-RENS: first sub-MIP exhausted, no retry\n");
      break;
    }
    // Bisection on the fixing rate between "too large" (exhausted) and "too small" (limits hit).
    if (ss.exhausted) {
      if (achieved < hi) hi = achieved, hi_set = order;
    } else if (achieved > lo) {
      lo = achieved, lo_set = order;
    }
    if (hi <= lo + 0.03) {
      // The fixings between the two (the dive's order is nearly the same) make the sub-MIP
      // infeasible: leave them free and fix others instead, from the original target again.
      if (lo_set.empty() || hi_set.empty() || ++narrowings > 3) break;
      std::vector<char> in_lo(n, 0);
      for (Index j : lo_set) in_lo[j] = 1;
      Index nb = 0;
      for (Index j : hi_set) {
        if (!in_lo[j]) banned[j] = 1, ++nb;
      }
      if (debug) std::fprintf(stderr, "LP-RENS: %d fixings between rates %.2f and %.2f left free\n", nb, lo, hi);
      if (nb == 0) break;
      hi = 1.0, hi_set.clear();
      rate = std::max(target0, lo + 0.1);
      continue;
    }
    rate = hi < 1.0 ? (lo > 0.0 ? 0.5 * (lo + hi) : 0.6 * hi) : lo + 0.5 * (1.0 - lo);
  }
  return finish(false);
}

bool rins(const MipProblem& p, const double* lp_x, const std::vector<double>& incumbent, double cutoff,
          const SubMipLimits& limits, std::vector<double>& sol, double target_fixed, double max_fixed,
          unsigned seed) {
  std::vector<double> lb = p.col_lb, ub = p.col_ub;
  Index ints = 0, fixed = 0;
  std::vector<std::pair<double, Index>> rest;  // (|lp - incumbent|, j) of the integers left free
  std::vector<Index> agree;                    // fixed by agreement (not fixed in p)
  for (Index j = 0; j < p.num_cols(); ++j) {
    if (!p.is_int[j]) continue;
    ++ints;
    const double d = std::abs(lp_x[j] - incumbent[j]);
    if (d <= 1e-6 || lb[j] == ub[j]) {
      if (lb[j] < ub[j]) agree.push_back(j);
      lb[j] = ub[j] = std::round(incumbent[j]);
      ++fixed;
    } else {
      rest.emplace_back(d, j);
    }
  }
  const Index want = static_cast<Index>(target_fixed * static_cast<double>(ints));
  if (fixed < want && !rest.empty()) {
    std::stable_sort(rest.begin(), rest.end());
    for (size_t k = 0; k < rest.size() && fixed < want; ++k, ++fixed) {
      const Index j = rest[k].second;
      lb[j] = ub[j] = std::round(incumbent[j]);
    }
  }
  const Index cap = static_cast<Index>(max_fixed * static_cast<double>(ints));
  if (max_fixed < 1.0 && fixed > cap && !agree.empty()) {
    std::mt19937 rng(seed);
    for (size_t k = 0; k < agree.size() && fixed > cap; ++k, --fixed) {  // partial Fisher-Yates shuffle
      std::swap(agree[k], agree[k + rng() % (agree.size() - k)]);
      const Index j = agree[k];
      lb[j] = p.col_lb[j], ub[j] = p.col_ub[j];
    }
  }
  if (!enough_fixed(p, lb, ub, limits.min_fixed_fraction)) return false;
  return solve_sub_mip(p, lb, ub, cutoff, limits, sol);
}

bool repair(const MipProblem& p, const std::vector<double>& x, double cutoff, const SubMipLimits& limits,
            std::vector<double>& sol) {
  const Index n = p.num_cols(), m = p.num_rows();
  std::vector<double> act(m, 0.0);
  Index ints = 0;
  for (Index j = 0; j < n; ++j) {
    ints += p.is_int[j];
    for (Index k = p.A.begin(j); k < p.A.end(j); ++k) act[p.A.index[k]] += p.A.value[k] * x[j];
  }
  const SparseMatrix rows = p.A.transpose();
  std::vector<char> is_free(n, 0), row_seen(m, 0);
  std::vector<Index> frontier;
  for (Index i = 0; i < m; ++i) {
    const double tol = 1e-6 * (1.0 + std::abs(act[i]));
    if (act[i] < p.row_lb[i] - tol || act[i] > p.row_ub[i] + tol) frontier.push_back(i), row_seen[i] = 1;
  }
  if (frontier.empty()) return false;
  // Breadth-first over rows: free the variables of the violated rows, then of their neighbour rows.
  Index free_ints = 0;
  const Index max_free = std::max<Index>(ints / 2, 1);
  for (int level = 0; level < 3 && !frontier.empty() && free_ints < max_free; ++level) {
    std::vector<Index> next;
    for (Index i : frontier) {
      for (Index k = rows.begin(i); k < rows.end(i); ++k) {
        const Index j = rows.index[k];
        if (is_free[j]) continue;
        if (level > 0 && free_ints >= max_free) break;
        is_free[j] = 1;
        free_ints += p.is_int[j];
        for (Index t = p.A.begin(j); t < p.A.end(j); ++t) {
          const Index r = p.A.index[t];
          if (!row_seen[r]) row_seen[r] = 1, next.push_back(r);
        }
      }
    }
    frontier.swap(next);
  }
  std::vector<double> lb = p.col_lb, ub = p.col_ub;
  for (Index j = 0; j < n; ++j) {
    if (p.is_int[j] && !is_free[j]) lb[j] = ub[j] = std::min(std::max(std::round(x[j]), p.col_lb[j]), p.col_ub[j]);
  }
  return solve_sub_mip(p, lb, ub, cutoff, limits, sol);
}

bool mutation(const MipProblem& p, const std::vector<double>& incumbent, double fix_fraction, unsigned seed,
              double cutoff, const SubMipLimits& limits, std::vector<double>& sol) {
  std::vector<Index> free;
  for (Index j = 0; j < p.num_cols(); ++j) {
    if (p.is_int[j] && p.col_lb[j] < p.col_ub[j]) free.push_back(j);
  }
  std::mt19937 rng(seed);
  const size_t nfix = static_cast<size_t>(fix_fraction * static_cast<double>(free.size()));
  if (nfix == 0 || nfix >= free.size()) return false;
  std::vector<double> lb = p.col_lb, ub = p.col_ub;
  for (size_t k = 0; k < nfix; ++k) {  // partial Fisher-Yates shuffle
    std::swap(free[k], free[k + rng() % (free.size() - k)]);
    const Index j = free[k];
    lb[j] = ub[j] = std::round(incumbent[j]);
  }
  return solve_sub_mip(p, lb, ub, cutoff, limits, sol);
}

bool graph_mutation(const MipProblem& p, const std::vector<double>& incumbent, double fix_fraction, unsigned seed,
                    double cutoff, const SubMipLimits& limits, std::vector<double>& sol, Index max_row) {
  const Index n = p.num_cols(), m = p.num_rows();
  std::vector<Index> ints;
  for (Index j = 0; j < n; ++j) {
    if (p.is_int[j] && p.col_lb[j] < p.col_ub[j]) ints.push_back(j);
  }
  const size_t nfree = ints.size() - static_cast<size_t>(fix_fraction * static_cast<double>(ints.size()));
  if (nfree == 0 || nfree >= ints.size()) return false;
  // Rows linking (nearly) everything (cardinality, objective-like rows) would free the whole problem at
  // distance 2: not followed. Default: 10% of the columns, at least 50 entries.
  if (max_row <= 0) max_row = std::max<Index>(50, n / 10);
  const SparseMatrix rows = p.A.transpose();
  std::vector<char> seen_col(n, 0), seen_row(m, 0), is_free(n, 0);
  std::mt19937 rng(seed);
  std::vector<Index> queue;  // columns, breadth first
  size_t freed = 0, head = 0;
  while (freed < nfree) {
    if (head == queue.size()) {
      // New start: a random integer column not reached yet (the first start too).
      Index start = -1;
      for (int tries = 0; tries < 64 && start < 0; ++tries) {
        const Index j = ints[rng() % ints.size()];
        if (!seen_col[j]) start = j;
      }
      if (start < 0) {
        for (Index j : ints) {
          if (!seen_col[j]) {
            start = j;
            break;
          }
        }
      }
      if (start < 0) break;
      seen_col[start] = 1;
      queue.push_back(start);
    }
    const Index j = queue[head++];
    if (p.is_int[j] && p.col_lb[j] < p.col_ub[j]) is_free[j] = 1, ++freed;
    for (Index t = p.A.begin(j); t < p.A.end(j); ++t) {
      const Index i = p.A.index[t];
      if (seen_row[i]) continue;
      seen_row[i] = 1;
      if (rows.end(i) - rows.begin(i) > max_row) continue;
      for (Index k = rows.begin(i); k < rows.end(i); ++k) {
        const Index c = rows.index[k];
        if (!seen_col[c]) seen_col[c] = 1, queue.push_back(c);
      }
    }
  }
  std::vector<double> lb = p.col_lb, ub = p.col_ub;
  for (Index j : ints) {
    if (!is_free[j]) lb[j] = ub[j] = std::min(std::max(std::round(incumbent[j]), p.col_lb[j]), p.col_ub[j]);
  }
  return solve_sub_mip(p, lb, ub, cutoff, limits, sol);
}

bool crossover(const MipProblem& p, const std::vector<const std::vector<double>*>& points, double cutoff,
               const SubMipLimits& limits, std::vector<double>& sol) {
  if (points.size() < 2) return false;
  std::vector<double> lb = p.col_lb, ub = p.col_ub;
  for (Index j = 0; j < p.num_cols(); ++j) {
    if (!p.is_int[j] || lb[j] == ub[j]) continue;
    const double v = std::round((*points[0])[j]);
    bool agree = true;
    for (size_t k = 1; k < points.size() && agree; ++k) agree = std::round((*points[k])[j]) == v;
    if (agree) lb[j] = ub[j] = std::min(std::max(v, p.col_lb[j]), p.col_ub[j]);
  }
  if (!enough_fixed(p, lb, ub, limits.min_fixed_fraction)) return false;
  return solve_sub_mip(p, lb, ub, cutoff, limits, sol);
}

}  // namespace milp
