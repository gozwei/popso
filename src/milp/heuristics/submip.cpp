#include "milp/heuristics/submip.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>

#include "milp/core/log.hpp"
#include "milp/core/timer.hpp"
#include "milp/presolve/presolve.hpp"
#include "milp/search/branch_and_bound.hpp"
#include "milp/search/propagation.hpp"

namespace milp {

bool solve_sub_mip(const MipProblem& p, const std::vector<double>& lb, const std::vector<double>& ub, double cutoff,
                   const SubMipLimits& limits, std::vector<double>& sol) {
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
  Logger log(0);
  Timer timer;
  SearchOptions opts;
  opts.cutoff = cutoff;
  opts.sub_mip_heuristics = false;
  opts.stop_flag = limits.stop;
  opts.progress = limits.progress;
  opts.iteration_limit = limits.iteration_limit;
  opts.check_resident = false;
  // Presolve large problems first: the fixings usually remove most of them. Small ones are
  // solved as they are (presolve saves little there, and on timtab1 its aggregations of
  // general integers made the sub-MIPs markedly less successful).
  static const bool no_presolve = std::getenv("MILP_SUBMIP_NO_PRESOLVE") != nullptr;
  MipResult r;
  Presolver pre;
  PresolveOptions po;
  po.probing_effort = 0.2;
  static const long long presolve_nnz = [] {
    const char* e = std::getenv("MILP_SUBMIP_PRESOLVE_NNZ");
    return e ? std::atoll(e) : 20000LL;
  }();
  // (Presolve does not know the implications of logical constraints.)
  const bool presolve = !no_presolve && p.A.nnz() >= presolve_nnz && p.implications.empty();
  // Unbounded (or skipped): solve the unreduced problem, which classifies it precisely.
  const PresolveStatus ps = presolve ? pre.run(q, po, log) : PresolveStatus::Unbounded;
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
    st->nodes += r.nodes;
    st->fixed = fixed;
    st->ints = ints;
  }
  if (std::getenv("MILP_SUBMIP_DEBUG")) {
    std::fprintf(stderr, "sub-MIP: fixed %d/%d ints, status %s, nodes %lld, sol %d obj %.10g, %.2fs\n", fixed, ints,
                 to_string(r.status), r.nodes, r.has_solution, r.objective, timer.elapsed());
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
    if (lim.time_limit <= 0.0) break;
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

bool rins(const MipProblem& p, const double* lp_x, const std::vector<double>& incumbent, double cutoff,
          const SubMipLimits& limits, std::vector<double>& sol, double target_fixed) {
  std::vector<double> lb = p.col_lb, ub = p.col_ub;
  Index ints = 0, fixed = 0;
  std::vector<std::pair<double, Index>> rest;  // (|lp - incumbent|, j) of the integers left free
  for (Index j = 0; j < p.num_cols(); ++j) {
    if (!p.is_int[j]) continue;
    ++ints;
    const double d = std::abs(lp_x[j] - incumbent[j]);
    if (d <= 1e-6 || lb[j] == ub[j]) {
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
