#include "milp/solver/solver.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include "milp/conic/relaxation.hpp"
#include "milp/conic/socp.hpp"
#include "milp/core/interrupt.hpp"
#include "milp/core/log.hpp"
#include "milp/core/timer.hpp"
#include "milp/heuristics/fpr.hpp"
#include "milp/heuristics/lmip.hpp"
#include "milp/io/mps_reader.hpp"
#include "milp/io/solution.hpp"
#include "milp/presolve/presolve.hpp"
#include "milp/qp/miqp.hpp"
#include "milp/qp/qp.hpp"
#include "milp/qp/qp_presolve.hpp"
#include "milp/search/branch_and_bound.hpp"
#include "milp/search/orbitope.hpp"
#include "milp/search/sst.hpp"
#include "milp/search/symmetry.hpp"
#include "milp/search/mip_problem.hpp"
#include "milp/solver/duals.hpp"
#include "milp/solver/postsolve.hpp"
#include "milp/solver/quad_presolve.hpp"
#include "milp/solver/reformulate.hpp"
#include "milp/solver/square_terms.hpp"
#include "milp/solver/start.hpp"

namespace milp {

const char* to_string(Status s) {
  switch (s) {
    case Status::Unknown: return "UNKNOWN";
    case Status::Optimal: return "OPTIMAL";
    case Status::Infeasible: return "INFEASIBLE";
    case Status::Unbounded: return "UNBOUNDED";
    case Status::InfeasibleOrUnbounded: return "INF_OR_UNBD";
    case Status::TimeLimit: return "TIME_LIMIT";
    case Status::NodeLimit: return "NODE_LIMIT";
    case Status::MemoryLimit: return "MEMORY_LIMIT";
    case Status::IterationLimit: return "ITERATION_LIMIT";
    case Status::Interrupted: return "INTERRUPTED";
    case Status::NumericalError: return "NUMERIC";
    case Status::NonConvex: return "NONCONVEX";
  }
  return "UNKNOWN";
}

const char* to_string(LpMethod m) {
  switch (m) {
    case LpMethod::Auto: return "auto";
    case LpMethod::Simplex: return "simplex";
    case LpMethod::Barrier: return "barrier";
  }
  return "auto";
}

namespace {

// Branch-and-bound, optionally as a race of differently seeded searches that
// share incumbents; the first search to finish stops the others.
MipResult search(const MipProblem& q, const SolverParams& params, const Logger& log, const Timer& timer,
                 const SearchOptions& opts) {
  const int threads = params.threads > 0 ? params.threads : static_cast<int>(std::thread::hardware_concurrency());
  const int K = std::min(params.racing, std::max(1, threads));
  if (K <= 1 || params.deterministic) return branch_and_bound(q, params, log, timer, opts);
  log.log(1, "Racing %d searches with %d threads each\n", K, std::max(1, threads / K));
  std::atomic<bool> stop{false};
  SharedIncumbent shared;
  std::atomic<int> winner{-1};
  std::vector<MipResult> res(K);
  std::vector<std::thread> pool;
  for (int k = 0; k < K; ++k) {
    pool.emplace_back(inherit_interrupt([&, k] {
      SolverParams pk = params;
      pk.threads = std::max(1, threads / K);
      pk.seed = params.seed + 7919u * static_cast<unsigned>(k);
      pk.verbosity = k == 0 ? params.verbosity : 0;
      Logger lk(pk.verbosity);
      SearchOptions ok = opts;
      ok.stop_flag = &stop;
      ok.shared = &shared;
      ok.feasibility_jump = opts.feasibility_jump && k == 0;
      res[k] = branch_and_bound(q, pk, k == 0 ? log : lk, timer, ok);
      if (res[k].status != Status::Interrupted) {
        int expected = -1;
        if (winner.compare_exchange_strong(expected, k)) stop = true;
      }
    }));
  }
  for (auto& t : pool) t.join();
  const int w = winner.load() >= 0 ? winner.load() : 0;
  MipResult out = std::move(res[w]);
  for (int k = 0; k < K; ++k) {
    if (k == w) continue;
    out.nodes += res[k].nodes;
    out.lp_iterations += res[k].lp_iterations;
    if (res[k].has_solution && (!out.has_solution || res[k].objective < out.objective)) {
      out.has_solution = true;
      out.x = res[k].x;
      out.objective = res[k].objective;
    }
    if (out.status != Status::Optimal && out.status != Status::Infeasible) out.bound = std::max(out.bound, res[k].bound);
  }
  if (out.has_solution) out.bound = std::min(out.bound, out.objective);
  return out;
}

// Symmetry-breaking rows (search/sst.hpp) for the presolved problem q: detection, then SST rows
// and fixings for the generators not handled by packing orbitopes in the search. Returns false
// (and leaves *out alone) if nothing was added. Off with --symmetry off, MILP_NO_SYMMETRY=1 or
// MILP_SST=0. MILP_SST=1 (default): leaders with conflicts in their orbit only; 2: any orbit.
// MILP_SST_ROWS=0 keeps only the fixings; MILP_SST_MIN_FIX (default 0.1): minimum share of the
// columns fixed for the reductions to be used.
bool symmetry_rows(const MipProblem& q, const SolverParams& params, const Timer& timer, const Logger& log,
                   const SearchOptions& opts, MipProblem* out) {
  const int mode = [] {
    const char* v = std::getenv("MILP_SST");
    return v ? std::atoi(v) : 1;
  }();
  if (mode <= 0 || !params.symmetry || opts.sst_done || !opts.sub_mip_heuristics || !q.implications.empty() ||
      q.nonlinear() ||
      std::getenv("MILP_NO_SYMMETRY") != nullptr) {
    return false;
  }
  if (std::find(q.is_int.begin(), q.is_int.end(), 1) == q.is_int.end()) return false;
  const double t0 = timer.elapsed();
  SymmetryOptions so;
  // Deterministic mode: the work limit only (a binding time cap would be timing dependent).
  const double rem = std::max(0.0, params.time_limit - t0);
  so.time_limit = params.deterministic ? rem : std::max(1.0, 0.03 * rem);
  const Symmetry sym = detect_symmetry(q, so);
  if (sym.empty()) return false;
  std::vector<char> used;
  if (std::getenv("MILP_NO_ORBITOPE") == nullptr) detect_packing_orbitopes(q, sym, &used);
  SstOptions sst;
  sst.mode = mode;
  sst.seed = params.seed;
  const bool rows = [] {
    const char* v = std::getenv("MILP_SST_ROWS");
    return v ? std::atoi(v) != 0 : true;
  }();
  sst.rows = rows;
  SstResult r = sst_reductions(q, sym, used, sst);
  // Only when the fixings shrink the problem substantially: a few fixings (and their rows) mostly
  // take symmetry away from orbital branching and change the root heuristics' luck (measured on
  // miplib_wide_core: neos-950242 / neos-960392 slower, decomp2 / krka / ns1208400 neutral).
  const double min_fix = [] {
    const char* v = std::getenv("MILP_SST_MIN_FIX");
    return v ? std::atof(v) : 0.1;
  }();
  bool use = !r.empty() && static_cast<double>(r.fixed_zero.size()) >= min_fix * q.num_cols();
  if (!use && !r.empty()) log.log(2, "Symmetry: %d leaders would fix %zu columns; not used\n", r.leaders, r.fixed_zero.size());
  // Otherwise SST rows on continuous / general integer orbits (SCIP's default leader types; binaries
  // stay with orbital branching and orbitopes in the search). fhnw-binpack4-4 (infeasible 2D packing,
  // identical items): rows x_l >= x_f on the positions of interchangeable items; orbital branching on
  // the binaries alone fixed 209 variables in 96k nodes and the tree did not close in 120 s; SCIP's
  // own ablations there: SST cuts only 13 s, without symmetry handling or with integer leaders only no
  // proof in 60-120 s (280-410k nodes). By default on zero-objective problems only: on 10 core models
  // they take symmetry away from orbital branching as the integer SST rows did (neos-3381206-awhea
  // 97 leaders / 5029 rows, 1.9 s -> TL; neos-2978193-inde 17 -> 50 s; neos-4738912-atrato 45 -> 93 s;
  // n5-3, neos-1171448, neos5, seymour1, neos-827175 within +-15%; 1 seed, 120 s, 4 threads).
  // MILP_SST_CONT=1: all problems, 0: off.
  const int cont_mode = [] {
    const char* v = std::getenv("MILP_SST_CONT");
    return v ? std::atoi(v) : -1;
  }();
  bool zero_obj = true;
  for (Index j = 0; j < q.num_cols() && zero_obj; ++j) zero_obj = q.cost[j] == 0.0;
  const bool cont = cont_mode > 0 || (cont_mode < 0 && zero_obj);
  if (!use && cont) {
    SstOptions s3 = sst;
    s3.mode = 3;
    s3.rows = true;
    r = sst_reductions(q, sym, used, s3);
    use = !r.rows.empty();
  }
  if (!use) return false;
  *out = apply_sst(q, r);
  log.log(1, "Symmetry: %d generators, group order ~1e%.1f; %d leaders: %zu symmetry-breaking rows, %zu columns fixed "
          "(%.2fs%s)\n", sym.num_generators(), sym.log10_order(), r.leaders, r.rows.size(), r.fixed_zero.size(),
          timer.elapsed() - t0, r.budget ? ", budget" : "");
  return true;
}

// Restart rules (read per solve, so that tests can change them).
//  - Root restarts: the first one when at least MILP_RESTART_FRAC (10%) of the integer columns, and at
//    least MILP_RESTART_MIN (10) of them, are newly fixed globally at the root (reduced-cost fixing,
//    root strong branching, conflicts); further ones at MILP_RESTART_FRAC2 (10%), at most
//    MILP_MAX_RESTARTS (4) in all. From the second on, only if the presolve after the previous restart
//    removed at least MILP_RESTART_MINRED (5%) of the integer columns (SCIP's restartminred) and the
//    restarted roots so far plus the last one's cost again stay within MILP_RESTART_WORK (1) x the first
//    root's: LP work in deterministic mode, else seconds (the budget then at least 5% of the time limit).
//    Restarted roots from the second restart on are cheaper: at most MILP_RESTART_ROUNDS (5) cut rounds
//    (the earlier roots' cut rows are carried as rows), and root dives / RENS / root LNS only when the
//    search before the restart improved the incumbent. (The first restart at 5%, as in the 2026-11-03
//    design, cost time on quick roots: binkar10_1, neos-911970 and neos17, 8-9% fixed, restarted only
//    at 5% and were 0.4-1.4 s slower in both seeds of the core A/B.)
//  - In-tree restarts: at most MILP_MAX_TREE_RESTARTS (1) (rule in branch_and_bound.cpp).
//  - Opt-in MILP_RESTART_PC=1 / 2: restarted searches start from the pseudocosts of the search before
//    (1: as history, counts capped; 2: as estimates only; see kRestartPcDefault). Off by default: on
//    restarting core instances (deterministic, 2 threads) both modes needed ~15% more LP iterations.
// The rule before repeated restarts: MILP_MAX_RESTARTS=1 MILP_RESTART_MIN=20.
struct RestartRule {
  int max_restarts = 4, max_tree_restarts = 1;
  double frac = 0.10, frac2 = 0.10, min_removed = 0.05, work = 1.0;
  Index min_cols = 10;
  int cheap_rounds = 5;
  bool pc = true;
  static RestartRule from_env() {
    auto env = [](const char* name, double def) {
      const char* v = std::getenv(name);
      return v && *v ? std::atof(v) : def;
    };
    RestartRule r;
    r.max_restarts = static_cast<int>(env("MILP_MAX_RESTARTS", r.max_restarts));
    r.max_tree_restarts = static_cast<int>(env("MILP_MAX_TREE_RESTARTS", r.max_tree_restarts));
    r.frac = env("MILP_RESTART_FRAC", r.frac);
    r.frac2 = env("MILP_RESTART_FRAC2", r.frac2);
    r.min_cols = static_cast<Index>(env("MILP_RESTART_MIN", static_cast<double>(r.min_cols)));
    r.min_removed = env("MILP_RESTART_MINRED", r.min_removed);
    r.work = env("MILP_RESTART_WORK", r.work);
    r.cheap_rounds = static_cast<int>(env("MILP_RESTART_ROUNDS", r.cheap_rounds));
    r.pc = env("MILP_RESTART_PC", kRestartPcDefault) != 0.0;
    return r;
  }
};

Index count_ints(const MipProblem& q) {
  Index n = 0;
  for (char c : q.is_int) n += c != 0;
  return n;
}

// Solves the problem with presolve; returns false if the caller should solve
// the original problem instead (presolve gave up or postsolve failed).
bool solve_presolved(const MipProblem& p, const SolverParams& params, const Logger& log, const Timer& timer,
                     MipResult* out, const SearchOptions& opts = {}) {
  const RestartRule rule = RestartRule::from_env();
  Presolver pre;
  const PresolveStatus ps = pre.run(p, PresolveOptions::from_env(), log);
  const PresolveStats& st = pre.stats();
  log.log(1, "Presolve: %d rows, %d cols, %d nnz -> %d rows, %d cols, %d nnz (%.2fs)\n", st.rows_before,
          st.cols_before, st.nnz_before, st.rows_after, st.cols_after, st.nnz_after, st.time);
  // This presolve, for MipResult::presolve: the outermost call sets it last (restarts and the
  // symmetry-breaking rows presolve again in nested calls), so it is the first presolve's.
  PresolveSummary summary;
  summary.ran = true;
  summary.rows_before = st.rows_before, summary.cols_before = st.cols_before, summary.nnz_before = st.nnz_before;
  summary.rows = st.rows_after, summary.cols = st.cols_after, summary.nnz = st.nnz_after;
  summary.time = st.time;
  if (ps == PresolveStatus::Infeasible) {
    log.log(1, "Presolve: problem is infeasible\n");
    out->status = Status::Infeasible;
    out->bound = kInf;
    out->presolve = summary;
    return true;
  }
  // Unbounded-or-infeasible: let the full search classify it precisely.
  if (ps == PresolveStatus::Unbounded) {
    log.log(1, "Presolve: problem is unbounded or infeasible\n");
    out->presolve = summary;
    return false;
  }
  const MipProblem& q = pre.reduced();
  if (opts.sub_mip_heuristics && opts.restarts_done == 0 && opts.tree_restarts_done == 0 && !opts.sst_done &&
      local_search_bench(q)) {
    out->status = Status::Unknown;  // MILP_LS_BENCH: heuristic evaluation only
    return true;
  }
  if (const char* fb = std::getenv("MILP_FPR_BENCH")) {
    // Standalone heuristic comparison on the presolved problem (FPR vs FJ vs local MIP, same work
    // budget); no search.
    heuristic_bench(q, std::atoll(fb) > 0 ? std::atoll(fb) : 1000000000LL, params.seed, log);
    out->status = Status::Unknown;
    return true;
  }
  MipResult mr;
  std::function<void(const std::vector<double>&, double)> q_hook;  // opts.on_incumbent in q's columns
  std::vector<double> q_start;  // opts.start in q's columns
  double q_start_obj = kInf;    // its objective before the crush (SearchOptions::start_objective)
  if (q.num_cols() == 0) {
    // Everything was removed; remaining rows (if any) have no columns.
    for (Index i = 0; i < q.num_rows(); ++i) {
      if (q.row_lb[i] > params.tol.primal_feas || q.row_ub[i] < -params.tol.primal_feas) {
        out->status = Status::Infeasible;
        out->bound = kInf;
        return true;
      }
    }
    mr.status = Status::Optimal;
    mr.has_solution = true;
    mr.objective = mr.bound = mr.root_bound = q.offset;
  } else {
    SearchOptions o = opts;
    if (opts.on_incumbent) {
      // Incumbents of q are reported in p's columns.
      o.on_incumbent = [&pre, outer = opts.on_incumbent](const std::vector<double>& x, double bound) {
        outer(pre.postsolve(x), bound);
      };
    }
    q_hook = o.on_incumbent;
    if (!o.start.empty() && static_cast<Index>(o.start.size()) != p.num_cols()) {
      // A start in other columns (a caller's bug; crush would read it as this problem's point) is dropped.
      log.log(1, "WARNING: start point with %zu values for %d columns ignored\n", o.start.size(), p.num_cols());
      o.start.clear();
    }
    if (!o.start.empty()) {
      o.start_objective = p.objective(o.start);  // the start's own objective (its crushed point may differ)
      o.start = pre.crush(o.start);
    }
    q_start = o.start;
    q_start_obj = o.start_objective;
    if (!o.pc_init.empty()) o.pc_init = o.pc_init.mapped(pre.col_map());
    o.allow_restart = opts.allow_restart && opts.restarts_done < rule.max_restarts;
    o.restart_frac = opts.restarts_done == 0 ? rule.frac : rule.frac2;
    o.restart_min_cols = rule.min_cols;
    o.restart_min_removed = rule.min_removed;
    {
      // After a restart: the share of the integer columns this presolve removed (incl. the fixings).
      const Index before = count_ints(p), after = count_ints(q);
      o.last_restart_removed = before > 0 ? static_cast<double>(before - after) / static_cast<double>(before) : 1.0;
    }
    if (!o.debug_point.empty()) {
      o.debug_point = pre.crush(o.debug_point);
      if (q.max_violation(o.debug_point) > 1e-6) {
        log.log(0, "DEBUG-SOL: the debug point is cut off by presolve (violation %.3g)\n", q.max_violation(o.debug_point));
        o.debug_point.clear();
      }
    }
    MipProblem qs;
    if (symmetry_rows(q, params, timer, log, opts, &qs)) {
      // Solved as a presolved problem of its own (presolve uses the fixings and new rows); its
      // restarts happen inside, and its columns are q's.
      o.sst_done = true;
      if (!o.debug_point.empty() && qs.max_violation(o.debug_point) > 1e-6) {
        log.log(0, "DEBUG-SOL: the debug point is cut off by the symmetry-breaking rows (violation %.3g)\n",
                qs.max_violation(o.debug_point));
        o.debug_point.clear();
      }
      if (!solve_presolved(qs, params, log, timer, &mr, o)) mr = search(qs, params, log, timer, o);
    } else {
      o.sst_done = true;
      mr = search(q, params, log, timer, o);
    }
  }
  if (mr.restart) {
    // Root restart: tightened bounds + root cuts, presolved again; the
    // incumbent found so far is the cutoff and the fallback answer.
    const double t_restart = timer.elapsed();
    MipProblem q2 = q;
    q2.col_lb = mr.restart_lb;
    q2.col_ub = mr.restart_ub;
    if (mr.has_solution) {
      // Keep the incumbent feasible (the fixings only matter for better points) so the
      // restarted search can start from it (RINS, guided dives, local branching).
      for (Index j = 0; j < q.num_cols(); ++j) {
        q2.col_lb[j] = std::min(q2.col_lb[j], mr.x[j]);
        q2.col_ub[j] = std::max(q2.col_ub[j], mr.x[j]);
      }
    }
    const Index m0 = q.num_rows(), extra = static_cast<Index>(mr.cut_lb.size());
    if (extra > 0) {
      std::vector<Index> J, I;
      std::vector<double> V;
      for (Index j = 0; j < q.num_cols(); ++j)
        for (Index k = q.A.begin(j); k < q.A.end(j); ++k) J.push_back(j), I.push_back(q.A.index[k]), V.push_back(q.A.value[k]);
      for (Index r = 0; r < extra; ++r)
        for (Index k = mr.cut_start[r]; k < mr.cut_start[r + 1]; ++k)
          J.push_back(mr.cut_index[k]), I.push_back(m0 + r), V.push_back(mr.cut_value[k]);
      q2.A = SparseMatrix::from_triplets(q.num_cols(), m0 + extra, J, I, V);
      q2.row_lb.insert(q2.row_lb.end(), mr.cut_lb.begin(), mr.cut_lb.end());
      q2.row_ub.insert(q2.row_ub.end(), mr.cut_ub.begin(), mr.cut_ub.end());
    }
    SearchOptions o2 = opts;
    o2.on_incumbent = q_hook;  // q2 has q's columns
    o2.sst_done = true;  // symmetry-breaking rows only before the first search
    // Root and in-tree restarts are counted separately (MILP_MAX_RESTARTS / MILP_MAX_TREE_RESTARTS).
    // The estimation-driven tree restart: once per solve, apart from the fixing-driven ones.
    o2.restarts_done = opts.restarts_done + (mr.tree_restart ? 0 : 1);
    o2.tree_restarts_done = opts.tree_restarts_done + (mr.tree_restart ? 1 : 0);
    o2.est_restarts_done = opts.est_restarts_done + (mr.est_restart ? 1 : 0);
    o2.allow_restart = o2.restarts_done < rule.max_restarts;
    o2.allow_tree_restart =
        opts.allow_tree_restart && o2.tree_restarts_done - o2.est_restarts_done < rule.max_tree_restarts;
    o2.allow_est_restart = opts.allow_est_restart && !mr.est_restart;
    o2.restart_frac = o2.restarts_done == 0 ? rule.frac : rule.frac2;
    o2.restart_min_cols = rule.min_cols;
    o2.restart_min_removed = rule.min_removed;
    {
      // Budget of the restarted roots (see RestartRule): the first root's cost, less the restarted roots.
      const bool first_root = opts.restarts_done == 0 && opts.tree_restarts_done == 0;
      if (first_root) {
        o2.restart_work_left = rule.work * mr.root_lp_work;
        o2.restart_time_left =
            rule.work * (is_finite(params.time_limit) ? std::max(mr.root_time, 0.05 * params.time_limit) : mr.root_time);
      } else {
        o2.restart_work_left = opts.restart_work_left - mr.root_lp_work;
        o2.restart_time_left = opts.restart_time_left - mr.root_time;
      }
    }
    o2.restart_t0 = t_restart;
    o2.root_cut_rounds = o2.restarts_done >= 2 ? rule.cheap_rounds : -1;
    o2.root_heuristics = o2.restarts_done < 2 || mr.improved;
    o2.pc_init = rule.pc ? std::move(mr.pc) : PseudocostSnapshot{};
    // The restarted problem has q's columns: the search's incumbent, else the start it was given (o2 is a copy of
    // opts, whose start is in p's columns: drayage-25-23 on x86 restarted a second time from a search that had
    // found nothing better than its start, the stale start was crushed through the wrong presolve, the
    // restarted root was pruned against it, and the incumbent 101344.27 came back as optimal; the optimum is
    // 101282.65).
    o2.start = mr.has_solution ? mr.x : q_start;
    o2.start_objective = mr.has_solution ? q.objective(mr.x) : q_start_obj;
    o2.debug_point = mr.debug_point;
    o2.feasibility_jump = false;  // it already ran
    // After a restart from a quick root (set packing/covering: a few hundred LP iterations on a few hundred rows), a second
    // zero-half loop mostly costs time (setpack/setcover medium 0.1 -> 0.2 s); MILP_RESTART_ZEROHALF=1 keeps it.
    static const bool restart_zh = std::getenv("MILP_RESTART_ZEROHALF") != nullptr;
    o2.no_zerohalf = !restart_zh && mr.root_lp_work < 1e6;
    o2.root_lns = !mr.root_lns_ran || mr.root_lns_improved;
    MipResult mr2;
    if (!solve_presolved(q2, params, log, timer, &mr2, o2)) mr2 = search(q2, params, log, timer, o2);
    mr2.nodes += mr.nodes;
    mr2.lp_iterations += mr.lp_iterations;
    mr2.root_bound = mr.root_bound;
    mr2.restarts += mr.restarts + (mr.tree_restart ? 0 : 1);
    mr2.tree_restarts += mr.tree_restarts + (mr.tree_restart ? 1 : 0);
    if (!mr2.has_solution && mr.has_solution) {
      // Nothing better than the incumbent exists (or was found in time).
      mr2.has_solution = true;
      mr2.x = std::move(mr.x);
      mr2.objective = mr.objective;
      if (mr2.status == Status::Infeasible) {
        mr2.status = Status::Optimal;
        mr2.bound = mr.objective;
      }
    }
    // A restarted search stopped at a limit (time, interrupt) may not have re-solved its root yet;
    // the bound before the restart is still valid (the restart only adds valid reductions).
    if (mr2.status != Status::Optimal && mr2.status != Status::Infeasible && is_finite(mr.bound))
      mr2.bound = std::max(mr2.bound, mr.bound);
    mr2.bound = std::min(mr2.bound, mr2.objective);
    mr = std::move(mr2);
  }
  if (!postsolve_result(p, pre, params, timer, log, &mr)) {
    out->bound = mr.bound;  // still a valid bound for the caller's re-solve
    out->presolve = summary;
    return false;
  }
  *out = std::move(mr);
  out->presolve = summary;
  return true;
}

// Reduced problem as a user model (columns c<j>, rows r<i>; minimization).
// With orig_names / col_map, columns keep their original names (diagnostics: diffs against other presolvers).
Model reduced_model(const MipProblem& q, const std::vector<std::string>* orig_names = nullptr,
                    const std::vector<Index>* col_map = nullptr) {
  Model m;
  m.name = "presolved";
  m.obj = q.cost;
  m.obj_offset = q.offset;
  m.col_lb = q.col_lb;
  m.col_ub = q.col_ub;
  m.row_lb = q.row_lb;
  m.row_ub = q.row_ub;
  m.A = q.A;
  for (Index j = 0; j < q.num_cols(); ++j) {
    m.col_type.push_back(q.is_int[j] ? VarType::Integer : VarType::Continuous);
    if (orig_names && col_map && (*col_map)[j] < static_cast<Index>(orig_names->size())) {
      m.col_names.push_back((*orig_names)[(*col_map)[j]]);
    } else {
      m.col_names.push_back("c" + std::to_string(j));
    }
  }
  for (Index i = 0; i < q.num_rows(); ++i) m.row_names.push_back("r" + std::to_string(i));
  return m;
}

// Presolve tooling (no search): MILP_PRESOLVE_ONLY=1 just presolves; MILP_PRESOLVE_DUMP=<file.mps> writes the
// reduced problem; MILP_PRESOLVE_CHECK=<file.sol> postsolves a solution of that dump and reports its violation
// and objective; MILP_PRESOLVE_ONLY=symmetry also reports symmetry detection. Returns false if no such variable is set.
bool presolve_tool(const Model& model, const MipProblem& p, const Logger& log, SolveResult* r) {
  const char* only = std::getenv("MILP_PRESOLVE_ONLY");
  const char* dump = std::getenv("MILP_PRESOLVE_DUMP");
  const char* check = std::getenv("MILP_PRESOLVE_CHECK");
  if (!only && !dump && !check) return false;
  Presolver pre;
  const PresolveStatus ps = pre.run(p, PresolveOptions::from_env(), log);
  const PresolveStats& st = pre.stats();
  log.log(1, "Presolve: %d rows, %d cols, %d nnz -> %d rows, %d cols, %d nnz (%.2fs)\n", st.rows_before,
          st.cols_before, st.nnz_before, st.rows_after, st.cols_after, st.nnz_after, st.time);
  r->status = ps == PresolveStatus::Infeasible ? Status::Infeasible
              : ps == PresolveStatus::Unbounded ? Status::InfeasibleOrUnbounded : Status::Unknown;
  if (ps != PresolveStatus::Reduced) return true;
  if (only && std::string(only) == "symmetry") {
    const Symmetry sym = detect_symmetry(pre.reduced());
    long long support = 0;
    for (const auto& g : sym.gens) support += static_cast<long long>(g.size());
    log.log(1, "Symmetry: %d generators, %d columns moved, support %lld, order ~1e%.1f, work %lld, %.3fs%s\n",
            sym.num_generators(), sym.moved_cols, support, sym.log10_order(), sym.work, sym.seconds,
            sym.aborted ? " (aborted)" : "");
  }
  const bool keep_names = std::getenv("MILP_PRESOLVE_NAMES") != nullptr;  // original column names in the dump
  const Model red = keep_names ? reduced_model(pre.reduced(), &model.col_names, &pre.col_map()) : reduced_model(pre.reduced());
  if (dump) write_mps(red, dump);
  if (check) {
    std::vector<double> xr = read_solution(check, red);
    for (double& v : xr) {
      if (v != v) v = 0.0;
    }
    const std::vector<double> x = pre.postsolve(xr);
    const double viol = p.max_violation(x);
    std::printf("Postsolve check: violation %.3g, objective %.17g (reduced objective %.17g)\n", viol,
                model.objective_value(x), red.objective_value(xr));
    r->has_solution = true;
    r->x = x;
    r->objective = model.objective_value(x);
  }
  return true;
}

// Reports new incumbents to SolverParams::on_incumbent: in the user model's columns (the first
// num_cols of the solved problem's), serialized, only strictly improving ones. A callback that asks
// to stop (or throws) sets the interrupt flag.
class IncumbentReporter {
 public:
  IncumbentReporter(const Model& model, const SolverParams& params)
      : model_(model), sense_(model.sense == ObjSense::Maximize ? -1.0 : 1.0), fn_(params.on_incumbent),
        flag_(params.interrupt) {}
  void report(const std::vector<double>& x_in, double bound) {
    std::vector<double> x(x_in);
    x.resize(static_cast<size_t>(model_.num_cols()));
    const double obj = model_.objective_value(x);
    std::lock_guard<std::mutex> lk(mu_);
    if (stopped_ || !(sense_ * obj < best_)) return;
    best_ = sense_ * obj;
    bool stop = true;
    try {
      stop = fn_(x, obj, sense_ * std::min(bound, sense_ * obj));
    } catch (...) {
    }
    if (stop) {
      stopped_ = true;  // no calls after a stop request
      if (flag_) flag_->store(true);
    }
  }

 private:
  const Model& model_;
  const double sense_;
  const std::function<bool(const std::vector<double>&, double, double)>& fn_;
  std::atomic<bool>* flag_;
  std::mutex mu_;
  double best_ = kInf;
  bool stopped_ = false;
};

// Collects SolveResult::incumbent_trace (SolverParams::trace_incumbents) from the searches'
// SearchOptions::on_trace / MiqpOptions::on_trace: only incumbents better than every one before by
// more than a relative 1e-9 (a restarted search accepts its start again, possibly with an objective
// that differs in the last bits after the new presolve; racing searches report their own).
class IncumbentTracer {
 public:
  explicit IncumbentTracer(ObjSense sense) : sense_(sense == ObjSense::Maximize ? -1.0 : 1.0) {}
  void record(double objective, const char* source, double time) {
    const double v = sense_ * objective;
    std::lock_guard<std::mutex> lk(mu_);
    const double tol = is_finite(best_) ? 1e-9 * std::max(1.0, std::abs(best_)) : 0.0;
    if (!(v < best_ - tol)) return;
    best_ = v;
    const char* src = source ? source : "";
    if (!events_.empty()) {
      IncumbentEvent& e = events_.back();
      if (e.source == src && time - e.last_time <= std::max(0.01, 0.01 * e.last_time)) {
        ++e.count;
        e.last_time = time;
        e.last_objective = objective;
        return;
      }
    }
    IncumbentEvent e;
    e.time = e.last_time = time;
    e.objective = e.last_objective = objective;
    e.source = src;
    events_.push_back(std::move(e));
  }
  std::function<void(double, const char*, double)> hook() {
    return [this](double objective, const char* source, double time) { record(objective, source, time); };
  }
  std::vector<IncumbentEvent> take() {
    std::lock_guard<std::mutex> lk(mu_);
    return std::move(events_);
  }

 private:
  const double sense_;
  std::mutex mu_;
  double best_ = kInf;
  std::vector<IncumbentEvent> events_;
};

// LP duals of a pure LP (the solution is optimal) or the fixed-MIP duals of a MILP
// (SolverParams::duals >= 2), in the model's sense, with the basis they come from (solver/duals.hpp).
void add_duals(const MipProblem& p, const SolverParams& params, const Timer& timer, const Logger& log, SolveResult& r) {
  const bool mip = std::find(p.is_int.begin(), p.is_int.end(), 1) != p.is_int.end();
  if (params.duals < (mip ? 2 : 1)) return;
  if (!mip && r.status != Status::Optimal) return;
  std::vector<double> lb, ub, x = r.x;
  if (mip) {
    lb = p.col_lb;
    ub = p.col_ub;
    for (Index j = 0; j < p.num_cols(); ++j) {
      if (!p.is_int[j]) continue;
      x[j] = std::round(x[j]);
      lb[j] = ub[j] = std::clamp(x[j], p.col_lb[j], p.col_ub[j]);
    }
  }
  const double t0 = timer.elapsed();
  // Post-processing: at least a second even when the time limit is (nearly) used up.
  const LpDualResult d = recover_lp_duals(p, lb, ub, x, std::max(1.0, params.time_limit - t0));
  log.log(1, "Duals%s: %s, %lld simplex iterations (%.2fs)\n", mip ? " (integers fixed)" : "",
          d.ok ? "recovered" : "failed", d.iterations, timer.elapsed() - t0);
  if (!d.ok) return;
  // The basis the duals come from must reproduce the solution's objective (an LP: the same optimum;
  // fixed-MIP: the incumbent is feasible for the fixed LP, so its optimum is no worse). Otherwise
  // the recovery went wrong (numerics, a point that is not optimal) and the duals would be stale.
  const double obj = p.objective(x);
  const DualObjectiveCheck check = check_dual_objective(d.objective, obj, mip);
  if (check == DualObjectiveCheck::Mismatch) {
    log.log(1, "Duals: dropped (their basis has objective %.10g, the solution %.10g)\n", p.display_sense * d.objective,
            p.display_sense * obj);
    return;
  }
  if (check == DualObjectiveCheck::BetterThanSolution) {
    log.log(1, "Duals: the fixed LP's optimum %.10g is better than the incumbent's %.10g (its continuous part is not "
            "optimal): the duals are those of the fixed LP's optimum\n", p.display_sense * d.objective,
            p.display_sense * obj);
  }
  const double sense = p.display_sense;
  r.row_dual = d.row_dual;
  r.reduced_cost = d.reduced_cost;
  for (double& v : r.row_dual) v *= sense;
  for (double& v : r.reduced_cost) v *= sense;
  r.col_basis.resize(d.col_status.size());
  r.row_basis.resize(d.row_status.size());
  for (size_t j = 0; j < d.col_status.size(); ++j) r.col_basis[j] = static_cast<int8_t>(d.col_status[j]);
  for (size_t i = 0; i < d.row_status.size(); ++i) r.row_basis[i] = static_cast<int8_t>(d.row_status[i]);
  r.has_duals = true;
  r.fixed_mip_duals = mip;
  r.stats.time = timer.elapsed();
}

// Solves a plain MILP; `implications` (logical constraints without a big-M form) go into the
// searched problem, and presolve is skipped when there are any.
SolveResult solve_plain(const Model& model, const std::vector<Implication>& implications, const SolverParams& params_in,
                        const std::vector<double>& start, const Timer& timer, const Logger& log,
                        IncumbentReporter* reporter, IncumbentTracer* tracer, bool allow_duals) {
  SolveResult r;
  MipProblem p;
  if (!MipProblem::from_model(model, &p)) {
    r.status = Status::Infeasible;
    return r;
  }
  p.implications = implications;
  SolverParams params = params_in;
  if (!implications.empty() && params.presolve) {
    log.log(1, "Presolve: skipped (%zu logical implications)\n", implications.size());
    params.presolve = false;
  }
  const double sense = model.sense == ObjSense::Maximize ? -1.0 : 1.0;
  if (params.presolve && presolve_tool(model, p, log, &r)) return r;
  MipResult mr;
  SearchOptions opts;
  opts.allow_restart = true;
  opts.allow_tree_restart = true;
  opts.allow_est_restart = true;
  if (reporter) {
    opts.on_incumbent = [reporter](const std::vector<double>& x, double bound) { reporter->report(x, bound); };
  }
  if (tracer) opts.on_trace = tracer->hook();
  if (const char* v = std::getenv("MILP_CUTOFF")) opts.cutoff = sense * std::atof(v);  // diagnostics: known bound
  if (const char* f = std::getenv("MILP_DEBUG_SOL_ORIG")) {
    // Debugging: a known optimal point of the user model (one value per column), followed through
    // presolve and restarts (MILP_DEBUG_SOL is in the columns of the searched problem instead).
    if (std::FILE* fp = std::fopen(f, "r")) {
      double v;
      while (std::fscanf(fp, "%lf", &v) == 1) opts.debug_point.push_back(v);
      std::fclose(fp);
    }
    if (static_cast<Index>(opts.debug_point.size()) != p.num_cols()) opts.debug_point.clear();
  }
  // User start: made feasible here (original columns), then followed through presolve as the first incumbent.
  std::vector<double> start_x;
  if (!start.empty()) {
    StartOutcome so = prepare_start(p, start, params, timer);
    log.log(1, "%s\n", so.message.c_str());
    start_x = std::move(so.x);
    opts.start = start_x;
  }
  if (!params.presolve || !solve_presolved(p, params, log, timer, &mr, opts)) {
    // A failed presolved solve may still leave a valid bound (postsolve failure).
    const double presolved_bound = params.presolve && is_finite(mr.bound) ? mr.bound : -kInf;
    const PresolveSummary presolved = mr.presolve;  // a presolve that gave up still ran
    SearchOptions plain;
    plain.start = start_x;
    plain.on_incumbent = opts.on_incumbent;
    plain.on_trace = opts.on_trace;
    mr = branch_and_bound(p, params, log, timer, plain);
    mr.presolve = presolved;
    if (presolved_bound > mr.bound && !(mr.has_solution && presolved_bound > mr.objective)) {
      mr.bound = presolved_bound;
    }
  }
  if (!start_x.empty()) {
    // The start is feasible; the search may have lost it (presolve reductions can cut it off, and its
    // objective then only served as a cutoff). Nothing better found means it is the answer.
    const double so = p.objective(start_x);
    if (!mr.has_solution || so < mr.objective) {
      mr.has_solution = true;
      mr.objective = so;
      mr.x = start_x;
      if (mr.status == Status::Infeasible) {
        mr.status = Status::Optimal;
        mr.bound = so;
      } else if (mr.status == Status::InfeasibleOrUnbounded) {
        mr.status = Status::Unbounded;
      }
    }
    if (mr.status != Status::Unbounded) mr.bound = std::min(mr.bound, mr.objective);
  }
  r.status = mr.status;
  r.has_solution = mr.has_solution;
  if (mr.has_solution) {
    r.x = std::move(mr.x);
    r.objective = model.objective_value(r.x);
  }
  // Internal bounds are for minimization; flip back for maximization.
  r.bound = sense * mr.bound;
  r.stats.time = timer.elapsed();
  r.stats.nodes = mr.nodes;
  r.stats.lp_iterations = mr.lp_iterations;
  r.stats.root_bound = sense * mr.root_bound;
  r.stats.restarts = mr.restarts;
  r.stats.tree_restarts = mr.tree_restarts;
  r.stats.presolve = mr.presolve;
  if (allow_duals && implications.empty() && r.has_solution && !interrupted()) add_duals(p, params, timer, log, r);
  return r;
}

// Values within the tolerance of zero that the logical constraints require to be zero (a semi-continuous
// column below its lower bound, SOS members outside the nonzero pattern) are set to exactly 0, if that
// does not make the point less feasible.
void snap_logical(const Model& model, std::vector<double>& x) {
  constexpr double kSnap = 1e-5;
  std::vector<double> y = x;
  for (Index j = 0; j < model.num_cols(); ++j) {
    if (!is_semi_type(model.col_type[j]) || std::abs(y[j]) > kSnap) continue;
    if (y[j] < model.col_lb[j] || y[j] > model.col_ub[j]) y[j] = 0.0;
  }
  for (const SosConstraint& s : model.sos) {
    // Keep the largest member (SOS1) or the pair of neighbours with the largest values (SOS2).
    const size_t k = s.cols.size();
    std::vector<size_t> ord(k);
    for (size_t t = 0; t < k; ++t) ord[t] = t;
    std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return s.weights[a] < s.weights[b]; });
    size_t keep = 0;
    double best = -1.0;
    const size_t span = s.type == 2 ? 2 : 1;
    for (size_t t = 0; t + span <= k; ++t) {
      double v = 0.0;
      for (size_t u = t; u < t + span; ++u) v = std::max(v, std::abs(y[s.cols[ord[u]]]));
      if (v > best) best = v, keep = t;
    }
    for (size_t t = 0; t < k; ++t) {
      const Index j = s.cols[ord[t]];
      if ((t < keep || t >= keep + span) && std::abs(y[j]) <= kSnap) y[j] = 0.0;
    }
  }
  if (y != x && model.violation(y).max() <= std::max(1e-9, model.violation(x).max())) x = std::move(y);
}

SolveResult solve_quadratic(const Model& model, const SolverParams& params, const std::vector<double>& start,
                            const Timer& timer, const Logger& log, IncumbentTracer* tracer);

// Convex quadratic rows (and with them, or with MILP_MIQP_OA=1, a convex quadratic objective as the epigraph row
// 1/2 x' Q x - eta <= 0): solved by outer approximation in the MILP search (MipProblem::quad_rows, gradient cuts in
// cuts/quad_oa.cpp), without presolve, symmetry handling or restarts, none of which knows the quadratic rows.
// A quadratic row whose quadratic part is not convex but which is a second-order cone over its columns' bounds
// (QuadRow::cone). In the row's <= direction, after completing the squares of the positive diagonal terms,
//   sum_k c_k (x_k + d_k)^2 + gamma <= c_t (t - s)^2   with t - s of one sign over t's bounds (a cone or, gamma > 0,
//                                                       one sheet of a hyperboloid), or
//   sum_k c_k (x_k + d_k)^2 + gamma <= c u v           with u, v >= 0 (rotated: c u v = c/4 ((u + v)^2 - (u - v)^2)),
// where c_k, c_t, c > 0, gamma >= 0, and no other linear terms. Fills the cone form and returns true, else false.
bool detect_cone(QuadRow& q, const std::vector<double>& lb, const std::vector<double>& ub) {
  const double dir = is_finite(q.hi) ? 1.0 : -1.0;
  const double side = dir * (dir > 0 ? q.hi : q.lo);
  std::map<std::pair<Index, Index>, double> mono;
  for (size_t k = 0; k < q.qc.size(); ++k) mono[{q.qi[k], q.qj[k]}] += dir * q.qc[k];
  std::map<Index, double> lin;
  for (size_t k = 0; k < q.idx.size(); ++k) lin[q.idx[k]] += dir * q.val[k];
  std::vector<std::pair<Index, double>> pos;
  Index t = -1, u = -1, v = -1;
  double ct = 0.0, cuv = 0.0;
  for (const auto& [key, c] : mono) {
    if (c == 0.0) continue;
    if (key.first == key.second) {
      if (c > 0.0) {
        pos.emplace_back(key.first, c);
      } else {
        if (t >= 0) return false;
        t = key.first, ct = -c;
      }
    } else {
      if (c > 0.0 || u >= 0) return false;
      u = key.first, v = key.second, cuv = -c;
    }
  }
  if ((t >= 0) == (u >= 0)) return false;  // exactly one negative square or one negative product
  double H = side;                           // sum c_k (x_k + d_k)^2 - (the cone's right side) <= H
  std::vector<std::vector<std::pair<Index, double>>> a;
  std::vector<double> b;
  for (const auto& [j, c] : pos) {
    if (j == u || j == v) return false;
    double bj = 0.0;
    if (auto it = lin.find(j); it != lin.end()) bj = it->second, lin.erase(it);
    H += bj * bj / (4.0 * c);
    a.push_back({{j, std::sqrt(c)}});
    b.push_back(std::sqrt(c) * bj / (2.0 * c));
  }
  std::vector<std::pair<Index, double>> h;
  double h0 = 0.0;
  if (t >= 0) {
    double bt = 0.0;
    if (auto it = lin.find(t); it != lin.end()) bt = it->second, lin.erase(it);
    const double sh = bt / (2.0 * ct), w = std::sqrt(ct);  // -c_t t^2 + b_t t = -c_t (t - sh)^2 + b_t^2 / (4 c_t)
    H -= bt * bt / (4.0 * ct);
    if (lb[t] >= sh) {
      h = {{t, w}}, h0 = -w * sh;
    } else if (ub[t] <= sh) {
      h = {{t, -w}}, h0 = w * sh;
    } else {
      return false;
    }
  } else {
    if (!(lb[u] >= 0.0 && lb[v] >= 0.0)) return false;
    const double w = 0.5 * std::sqrt(cuv);
    a.push_back({{u, w}, {v, -w}});
    b.push_back(0.0);
    h = {{u, w}, {v, w}};
  }
  for (const auto& [j, c] : lin) {
    if (c != 0.0) return false;
  }
  if (H > 1e-12 * std::max(1.0, std::abs(side))) return false;  // sum of squares <= (rhs)^2 + H, H > 0: not convex
  q.cone = true;
  q.cone_a = std::move(a);
  q.cone_b = std::move(b);
  q.cone_h = std::move(h);
  q.cone_h0 = h0;
  q.cone_gamma = std::max(0.0, -H);
  return true;
}

// Products of binaries linearized exactly: a product of two binaries becomes a column y in [0, 1] with
// y >= x_i + x_j - 1, y <= x_i, y <= x_j; a product of a binary x and a column z in [l, u] (finite; with_bounded) a
// column y with l x <= y <= u x, z - u (1 - x) <= y <= z - l (1 - x); a binary square becomes x_i; a term with a fixed
// column becomes linear (or a constant). Only the side a product needs is written: in a row with a finite upper side
// (or a minimized objective) a positive coefficient needs y >= x_i x_j only (the lower rows), a negative one
// y <= x_i x_j; a ranged row both. A quadratic row qualifies when every term is of these kinds; it then leaves
// the outer approximation as a linear row of the MILP, whose propagation and cuts are much stronger on such rows than
// tangents of the convex function (QPLIB_10010, a quadratic knapsack over 262 binaries: the OA search 2.3% gap at
// 60 s, its bound stuck at the root's; linearized OPTIMAL in 10 s), and a nonconvex one becomes solvable. The
// objective's Q likewise when `objective` and every term qualifies. At most max_products new columns (QPLIB_10007's
// row has 9.5 million terms).
struct BinaryProducts {
  const Model& m;
  bool with_bounded;
  bool fixed(Index j) const { return m.col_lb[j] == m.col_ub[j]; }
  bool binary(Index j) const {
    return (m.col_type[j] == VarType::Integer || m.col_type[j] == VarType::Binary) && m.col_lb[j] >= 0.0 &&
           m.col_ub[j] <= 1.0;
  }
  bool bounded(Index j) const { return with_bounded && is_finite(m.col_lb[j]) && is_finite(m.col_ub[j]); }
  bool linearizable(Index a, Index b) const {
    if (fixed(a) || fixed(b)) return true;
    if (a == b) return binary(a);
    return (binary(a) && (binary(b) || bounded(b))) || (binary(b) && bounded(a));
  }
  // Every term of the objective's Q qualifies.
  bool objective_linearizable() const {
    for (size_t k = 0; k < m.q_val.size(); ++k) {
      if (m.q_val[k] != 0.0 && !linearizable(m.q_row[k], m.q_col[k])) return false;
    }
    return !m.q_val.empty();
  }
};

// The linearization of `in` into m (a copy of in with the new columns and rows appended; m is left alone when nothing
// qualifies). Returns the number of quadratic rows linearized (+ 1 when the objective was), 0 if none; products lists
// the new columns' (i, j).
Index linearize_binary_products(const Model& in, Model& m, size_t max_products, bool with_bounded, bool objective,
                                std::vector<std::pair<Index, Index>>* products) {
  const BinaryProducts bp{in, with_bounded};
  objective = objective && bp.objective_linearizable();
  if (in.qr_val.empty() && !objective) return 0;
  const Index n = in.num_cols(), nr = in.num_rows();
  std::vector<char> eligible(static_cast<size_t>(nr), 0);
  for (size_t k = 0; k < in.qr_val.size(); ++k) eligible[in.qr_row[k]] = 1;
  for (size_t k = 0; k < in.qr_val.size(); ++k) {
    if (!bp.linearizable(in.qr_i[k], in.qr_j[k])) eligible[in.qr_row[k]] = 0;
  }
  std::map<std::pair<Index, Index>, Index> prod;  // (i, j), i > j -> new column
  auto key = [](Index a, Index b) { return std::make_pair(std::max(a, b), std::min(a, b)); };
  auto product = [&](Index a, Index b) { return a != b && !bp.fixed(a) && !bp.fixed(b); };
  for (size_t k = 0; k < in.qr_val.size() && prod.size() <= max_products; ++k) {
    const Index a = in.qr_i[k], b = in.qr_j[k];
    if (eligible[in.qr_row[k]] && product(a, b)) prod.emplace(key(a, b), -1);
  }
  if (objective) {
    for (size_t k = 0; k < in.q_val.size() && prod.size() <= max_products; ++k) {
      if (in.q_val[k] != 0.0 && product(in.q_row[k], in.q_col[k])) prod.emplace(key(in.q_row[k], in.q_col[k]), -1);
    }
  }
  Index done = objective ? 1 : 0;
  for (Index i = 0; i < nr; ++i) done += eligible[i];
  if (done == 0 || prod.size() > max_products) return 0;
  m = in;
  // Triplets of A plus the linearized terms, the new columns and their rows.
  std::vector<Index> J, I;
  std::vector<double> V;
  for (Index j = 0; j < n; ++j) {
    for (Index k = m.A.begin(j); k < m.A.end(j); ++k) J.push_back(j), I.push_back(m.A.index[k]), V.push_back(m.A.value[k]);
  }
  // The sides each product needs: bit 1 y >= x_i x_j (y pushed down), bit 2 y <= x_i x_j.
  std::vector<char> need(prod.size(), 0);
  Index col = n;
  for (auto& [ab, c] : prod) {
    c = col++;
    const auto [a, b] = ab;
    const Index z = bp.binary(b) ? a : b;  // the other column of a binary-bounded product
    m.col_names.push_back("_y" + std::to_string(a) + "_" + std::to_string(b));
    m.obj.push_back(0.0);
    m.col_lb.push_back(std::min(0.0, m.col_lb[z]));
    m.col_ub.push_back(std::max(0.0, m.col_ub[z]));
    m.col_type.push_back(VarType::Continuous);
  }
  std::vector<Index> qr, qi, qj;
  std::vector<double> qv;
  for (size_t k = 0; k < m.qr_val.size(); ++k) {
    const Index row = m.qr_row[k], a = m.qr_i[k], b = m.qr_j[k];
    const double c = m.qr_val[k];
    if (!eligible[row]) {
      qr.push_back(row), qi.push_back(a), qj.push_back(b), qv.push_back(c);
      continue;
    }
    if (bp.fixed(a) && bp.fixed(b)) {
      const double v = c * m.col_lb[a] * m.col_lb[b];
      if (is_finite(m.row_lb[row])) m.row_lb[row] -= v;
      if (is_finite(m.row_ub[row])) m.row_ub[row] -= v;
    } else if (bp.fixed(a) || bp.fixed(b)) {
      const Index other = bp.fixed(a) ? b : a;
      J.push_back(other), I.push_back(row), V.push_back(c * m.col_lb[bp.fixed(a) ? a : b]);
    } else if (a == b) {
      J.push_back(a), I.push_back(row), V.push_back(c);
    } else {
      const Index y = prod.at(key(a, b));
      J.push_back(y), I.push_back(row), V.push_back(c);
      if (is_finite(m.row_ub[row])) need[y - n] |= c > 0 ? 1 : 2;
      if (is_finite(m.row_lb[row])) need[y - n] |= c > 0 ? 2 : 1;
    }
  }
  if (objective) {
    // 1/2 Q_jj x_j^2 and Q_ij x_i x_j (i > j).
    for (size_t k = 0; k < m.q_val.size(); ++k) {
      const Index a = m.q_row[k], b = m.q_col[k];
      const double c = a == b ? 0.5 * m.q_val[k] : m.q_val[k];
      if (bp.fixed(a) && bp.fixed(b)) m.obj_offset += c * m.col_lb[a] * m.col_lb[b];
      else if (bp.fixed(a) || bp.fixed(b)) m.obj[bp.fixed(a) ? b : a] += c * m.col_lb[bp.fixed(a) ? a : b];
      else if (a == b) m.obj[a] += c;
      else {
        const Index y = prod.at(key(a, b));
        m.obj[y] += c;
        need[y - n] |= (m.sense == ObjSense::Maximize ? -c : c) > 0 ? 1 : 2;
      }
    }
    m.q_row.clear(), m.q_col.clear(), m.q_val.clear();
  }
  Index row = nr;
  auto add_row = [&](const std::string& name, double lo, double hi, std::initializer_list<std::pair<Index, double>> t) {
    m.row_names.push_back(name);
    m.row_lb.push_back(lo), m.row_ub.push_back(hi);
    for (const auto& [j, v] : t) {
      if (v != 0.0) J.push_back(j), I.push_back(row), V.push_back(v);
    }
    ++row;
  };
  for (const auto& [ab, c] : prod) {
    const auto [a, b] = ab;
    const std::string tag = std::to_string(a) + "_" + std::to_string(b);
    const bool lo = need[c - n] & 1, hi = need[c - n] & 2;
    if (bp.binary(a) && bp.binary(b)) {
      // y - x_a - x_b >= -1 (y >= 0 by its bound); y - x_a <= 0, y - x_b <= 0
      if (lo) add_row("_mc" + tag, -1.0, kInf, {{c, 1.0}, {a, -1.0}, {b, -1.0}});
      if (hi) add_row("_ma" + tag, -kInf, 0.0, {{c, 1.0}, {a, -1.0}});
      if (hi) add_row("_mb" + tag, -kInf, 0.0, {{c, 1.0}, {b, -1.0}});
    } else {
      // y = x z, x binary, z in [l, u]: y - l x >= 0, y - z - u x >= -u; y - u x <= 0, y - z - l x <= -l
      const Index x = bp.binary(b) ? b : a, z = bp.binary(b) ? a : b;
      const double l = m.col_lb[z], u = m.col_ub[z];
      if (lo) add_row("_ml" + tag, 0.0, kInf, {{c, 1.0}, {x, -l}});
      if (lo) add_row("_mz" + tag, -u, kInf, {{c, 1.0}, {z, -1.0}, {x, -u}});
      if (hi) add_row("_mu" + tag, -kInf, 0.0, {{c, 1.0}, {x, -u}});
      if (hi) add_row("_mw" + tag, -kInf, -l, {{c, 1.0}, {z, -1.0}, {x, -l}});
    }
  }
  m.A = SparseMatrix::from_triplets(col, row, J, I, V);
  m.qr_row.swap(qr), m.qr_i.swap(qi), m.qr_j.swap(qj), m.qr_val.swap(qv);
  if (products) {
    products->clear();
    for (const auto& kv : prod) products->push_back(kv.first);
  }
  return done;
}

// The linearization's switches: MILP_QBIN_LINEARIZE (0: off; else at most that many product columns, default
// 200000), MILP_QBIN_CONT=0 (binary-bounded products stay quadratic), MILP_QBIN_OBJ (0: objectives stay quadratic;
// 1, the default: a nonconvex objective is linearized; 2: every qualifying objective).
struct LinearizeOptions {
  size_t max_products = 200000;
  bool bounded = true;
  int objective = 1;
  LinearizeOptions() {
    if (const char* e = std::getenv("MILP_QBIN_LINEARIZE")) max_products = static_cast<size_t>(std::max(0.0, std::atof(e)));
    if (const char* e = std::getenv("MILP_QBIN_CONT")) bounded = *e != '0';
    if (const char* e = std::getenv("MILP_QBIN_OBJ")) objective = std::atoi(e);
  }
};

// Whether the objective of `model` is to be linearized (MILP_QBIN_OBJ): every term qualifies and, by default, the
// objective is not convex (convex ones keep the MIQP routes).
bool linearize_objective(const Model& model, const LinearizeOptions& o) {
  if (o.objective <= 0 || !model.has_quadratic() || !BinaryProducts{model, o.bounded}.objective_linearizable()) {
    return false;
  }
  if (o.objective >= 2) return true;
  SparseMatrix Q = model.q_lower();
  if (model.sense == ObjSense::Maximize) {
    for (double& v : Q.value) v = -v;
  }
  return !check_convexity(Q).convex;
}

// Solves `model` through its linearization when one applies (nullopt: none did): `inner` solves the linearized model
// (still quadratic or not), the solution is cut back to the model's columns.
template <class Inner>
std::optional<SolveResult> solve_linearized(const Model& model, bool objective, const std::vector<double>& start,
                                            const Logger& log, Inner&& inner) {
  static const LinearizeOptions opts;
  if (opts.max_products == 0 || model.has_logical()) return std::nullopt;
  Model lm;
  std::vector<std::pair<Index, Index>> products;
  const Index done = linearize_binary_products(model, lm, opts.max_products, opts.bounded, objective, &products);
  if (done == 0) return std::nullopt;
  const bool obj_done = model.has_quadratic() && !lm.has_quadratic();
  log.log(1, "Products of binaries: %d quadratic rows%s linearized exactly (%zu product columns)\n",
          done - (obj_done ? 1 : 0), obj_done ? " and the objective" : "", products.size());
  std::vector<double> st;
  if (!start.empty() && static_cast<Index>(start.size()) == model.num_cols()) {
    st = start;
    for (const auto& [a, b] : products) st.push_back(start[a] * start[b]);
  }
  SolveResult r = inner(lm, st);
  if (r.has_solution) {
    r.x.resize(static_cast<size_t>(model.num_cols()));
    r.objective = model.objective_value(r.x);
  }
  if (r.has_duals) {
    r.row_dual.resize(static_cast<size_t>(model.num_rows()));
    r.reduced_cost.resize(static_cast<size_t>(model.num_cols()));
  }
  return r;
}

SolveResult solve_oa_core(const Model& model, const SolverParams& params_in, const std::vector<double>& start,
                          const Timer& timer, const Logger& log, IncumbentTracer* tracer,
                          const std::vector<Bilinear>* products = nullptr,
                          const std::vector<MipProblem::LiftedRow>* lifted = nullptr);

// A model found not convex: an error, unless spatial branch-and-bound takes it from there (solve_model sets the flag
// for the calling thread; the reason then goes to the debug log only).
thread_local bool t_spatial_next = false;
void log_nonconvex(const Logger& log, const std::string& msg) {
  if (t_spatial_next) log.log(2, "Not convex: %s\n", msg.c_str());
  else log.log(1, "ERROR: %s\n", msg.c_str());
}

// The quadratic rows of `model` as the search's QuadRows (their linear parts, terms and sides; `quad` flags the
// rows), each checked for convexity or recognized as a cone (detect_cone). False, with the reason, if one is neither.
bool build_quad_rows(const Model& model, std::vector<char>& quad, std::vector<QuadRow>& rows, size_t& cones,
                     std::string& why, std::vector<char>* bad = nullptr) {
  if (bad) bad->assign(static_cast<size_t>(model.num_rows()), 0);
  const Index n = model.num_cols(), m = model.num_rows();
  quad.assign(static_cast<size_t>(m), 0);
  for (size_t k = 0; k < model.qr_val.size(); ++k) {
    if (model.qr_val[k] != 0.0) quad[model.qr_row[k]] = 1;
  }
  const SparseMatrix At = model.A.transpose();
  std::vector<Index> slot(static_cast<size_t>(m), -1);
  for (Index i = 0; i < m; ++i) {
    if (!quad[i]) continue;
    slot[i] = static_cast<Index>(rows.size());
    QuadRow q;
    q.lo = model.row_lb[i];
    q.hi = model.row_ub[i];
    for (Index k = At.begin(i); k < At.end(i); ++k) q.idx.push_back(At.index[k]), q.val.push_back(At.value[k]);
    rows.push_back(std::move(q));
  }
  for (size_t k = 0; k < model.qr_val.size(); ++k) {
    const Index s = slot[model.qr_row[k]];
    if (s < 0 || model.qr_val[k] == 0.0) continue;
    rows[s].qi.push_back(model.qr_i[k]);
    rows[s].qj.push_back(model.qr_j[k]);
    rows[s].qc.push_back(model.qr_val[k]);
  }
  // Convexity: the Hessian of each row (2c on the diagonal, c off it, as 1/2 x' H x) PSD towards a finite upper side,
  // NSD towards a finite lower side; a quadratic row with both sides finite is not convex.
  std::vector<Index> row_of;  // model row of rows[s]
  for (Index i = 0; i < m; ++i) {
    if (quad[i]) row_of.push_back(i);
  }
  bool all_convex = true;
  cones = 0;
  for (size_t s = 0; s < rows.size(); ++s) {
    // With `bad`: every row is judged and the rows that are not convex are marked; without: the first one ends it.
    auto nonconvex = [&](const std::string& msg) {
      if (all_convex) why = msg;
      all_convex = false;
      if (bad) (*bad)[row_of[s]] = 1;
      return bad != nullptr;
    };
    QuadRow& q = rows[s];
    if (is_finite(q.lo) && is_finite(q.hi)) {
      if (nonconvex("a quadratic constraint with two finite sides is not convex")) continue;
      return false;
    }
    const double dir = is_finite(q.hi) ? 1.0 : -1.0;
    std::vector<Index> I, J;
    std::vector<double> V;
    for (size_t k = 0; k < q.qc.size(); ++k) {
      J.push_back(q.qj[k]), I.push_back(q.qi[k]), V.push_back(dir * (q.qi[k] == q.qj[k] ? 2.0 : 1.0) * q.qc[k]);
    }
    const ConvexityResult cv = check_convexity(SparseMatrix::from_triplets(n, n, J, I, V));
    if (cv.convex) continue;
    if (!detect_cone(q, model.col_lb, model.col_ub)) {
      if (nonconvex("quadratic constraint not convex: " + cv.message)) continue;
      return false;
    }
    ++cones;
  }
  return all_convex;
}

SolveResult solve_oa_presolved(const Model& model, const SolverParams& params_in, const std::vector<double>& start,
                               const Timer& timer, const Logger& log, IncumbentTracer* tracer);

// MILP_QUAD_PRESOLVE (default 1; 0: off): mixed-integer models with quadratic rows go through quad_presolve
// (solver/quad_presolve.hpp) first. Its model has the same columns, so the solution needs only its alias
// columns recomputed; it is then checked against the model itself, and a point that fails (or a presolved
// model that is not convex where the model may be) sends the model through the search as it is.
SolveResult solve_oa(const Model& model, const SolverParams& params_in, const std::vector<double>& start,
                     const Timer& timer, const Logger& log, IncumbentTracer* tracer) {
  static const bool on = [] {
    const char* e = std::getenv("MILP_QUAD_PRESOLVE");
    return !(e && *e == '0');
  }();
  if (on && model.has_quadratic_rows() && model.num_integers() > 0 && !model.has_logical()) {
    Model pm;
    const QuadPresolve ps = quad_presolve(model, pm);
    if (ps.changed()) {
      log.log(1, "Quadratic presolve: %zu alias members replaced, %d fixed terms folded, %d perspective cones, %d bounds"
              " tightened\n", ps.aliases.size(), ps.fixed_terms, ps.perspective, ps.bounds);
      SolveResult r = solve_oa_presolved(pm, params_in, start, timer, log, tracer);
      bool ok = r.status != Status::NonConvex;
      if (ok && r.has_solution) {
        // The model's own rows by the search's measure (QuadRow::violation: a cone by its norm form, absolute
        // where it has no constants; a rewritten row can be a convex quadratic judged relative to its terms),
        // and its raw forms no worse than the presolved model's.
        const double pviol = pm.violation(r.x).max();
        ps.restore(r.x);
        double viol = model.violation(r.x).max();
        ok = viol <= std::max(1e-6, 2.0 * pviol + 1e-9);
        std::vector<char> oquad;
        std::vector<QuadRow> orows;
        size_t ocones = 0;
        std::string why;
        if (ok && build_quad_rows(model, oquad, orows, ocones, why)) {
          for (const QuadRow& q : orows) viol = std::max(viol, q.violation(r.x.data()));
          for (const QuadRow& q : orows) ok = ok && q.violation(r.x.data()) <= 1e-6;
        }
        if (!ok) log.log(1, "Quadratic presolve: its solution violates the model by %.3g; solving without it\n", viol);
        r.objective = model.objective_value(r.x);
      }
      if (ok) return r;
    }
  }
  return solve_oa_presolved(model, params_in, start, timer, log, tracer);
}

SolveResult solve_oa_presolved(const Model& model, const SolverParams& params_in, const std::vector<double>& start,
                               const Timer& timer, const Logger& log, IncumbentTracer* tracer) {
  static const LinearizeOptions opts;
  if (!model.has_quadratic_rows() && !model.has_quadratic()) {  // every quadratic row became linear in the presolve
    return solve_plain(model, {}, params_in, start, timer, log, nullptr, tracer, true);
  }
  const auto lr = solve_linearized(model, linearize_objective(model, opts), start, log,
                                   [&](const Model& lm, const std::vector<double>& st) {
                                     if (lm.has_quadratic_rows()) return solve_oa_core(lm, params_in, st, timer, log, tracer);
                                     if (lm.has_quadratic()) return solve_quadratic(lm, params_in, st, timer, log, tracer);
                                     return solve_plain(lm, {}, params_in, st, timer, log, nullptr, tracer, true);
                                   });
  if (lr) return *lr;
  return solve_oa_core(model, params_in, start, timer, log, tracer);
}

SolveResult solve_oa_core(const Model& model, const SolverParams& params_in, const std::vector<double>& start,
                          const Timer& timer, const Logger& log, IncumbentTracer* tracer,
                          const std::vector<Bilinear>* products, const std::vector<MipProblem::LiftedRow>* lifted) {
  SolveResult r;
  if (model.has_logical()) throw std::runtime_error("quadratic constraints with logical constraints are not supported yet");
  const double sense = model.sense == ObjSense::Maximize ? -1.0 : 1.0;
  const Index n = model.num_cols(), m = model.num_rows();
  std::vector<char> quad;
  std::vector<QuadRow> rows;
  size_t cones = 0;
  auto nonconvex = [&](const std::string& msg) {
    log_nonconvex(log, msg);
    r.status = Status::NonConvex;
    r.stats.time = timer.elapsed();
    return r;
  };
  {
    std::string why;
    if (!build_quad_rows(model, quad, rows, cones, why)) return nonconvex(why);
  }
  const SparseMatrix At = model.A.transpose();
  std::vector<Index> slot(static_cast<size_t>(m), -1);  // row -> its index in rows
  for (Index i = 0, k = 0; i < m; ++i) {
    if (quad[i]) slot[i] = k++;
  }
  Model lin = model;
  lin.q_row.clear(), lin.q_col.clear(), lin.q_val.clear();
  lin.remove_rows(quad);
  MipProblem p;
  if (!MipProblem::from_model(lin, &p)) {
    r.status = Status::Infeasible;
    r.bound = sense * kInf;
    r.stats.time = timer.elapsed();
    return r;
  }
  // The objective's epigraph: column eta with cost 1 and the row 1/2 x' Q x - eta <= 0 (Q of the minimization).
  if (model.has_quadratic()) {
    SparseMatrix Q = model.q_lower();
    for (double& v : Q.value) v *= sense;
    const ConvexityResult cv = check_convexity(Q);
    if (!cv.convex) return nonconvex(cv.message);
    const Index eta = p.num_cols();
    p.A.num_major += 1;
    p.A.start.push_back(p.A.start.back());
    p.cost.push_back(1.0);
    p.col_lb.push_back(-kInf);
    p.col_ub.push_back(kInf);
    p.is_int.push_back(0);
    QuadRow q;
    q.idx.push_back(eta);
    q.val.push_back(-1.0);
    q.hi = 0.0;
    q.epigraph = eta;
    for (Index j = 0; j < Q.num_major; ++j) {
      for (Index k = Q.begin(j); k < Q.end(j); ++k) {
        q.qi.push_back(Q.index[k]), q.qj.push_back(j), q.qc.push_back((Q.index[k] == j ? 0.5 : 1.0) * Q.value[k]);
      }
    }
    rows.push_back(std::move(q));
  }
  // A rotated cone sum_k c_k x_k^2 <= c u z with a binary z: z = 0 forces every x_k to 0. The tangents say so only
  // in the limit (at z = 0 the cone has no interior: an integral point with z = 0 and x_k != 0 took 150 rounds of
  // tangents on CBLIB achtziger_stolpe06-6.2flowc and was still violated), and the columns' ranges give no
  // big-M there (u and x_k unbounded). The search takes it as an implication instead: x_k is fixed where z is
  // fixed at 0, and an integral point that violates it is branched on z (MipProblem::implications, as for
  // indicator constraints without a big-M). Not where a row x_k <= M z with x_k >= 0 says it already.
  // MILP_OA_CONE_IMPL=0: off.
  static const bool cone_impl = [] {
    const char* e = std::getenv("MILP_OA_CONE_IMPL");
    return !(e && *e == '0');
  }();
  if (cone_impl) {
    auto binary = [&](Index j) {
      return j < n && (model.col_type[j] == VarType::Binary || model.col_type[j] == VarType::Integer) &&
             model.col_lb[j] == 0.0 && model.col_ub[j] == 1.0;
    };
    auto linked = [&](Index x, Index z) {  // a row a x + b z <= 0 with a > 0 > b (or its mirror image) and x >= 0
      if (!(model.col_lb[x] >= 0.0)) return false;
      for (Index k = model.A.begin(x); k < model.A.end(x); ++k) {
        const Index i = model.A.index[k];
        if (quad[i] || At.end(i) - At.begin(i) != 2) continue;
        const Index k0 = At.begin(i), kz = At.index[k0] == x ? k0 + 1 : k0, kx = At.index[k0] == x ? k0 : k0 + 1;
        if (At.index[kz] != z) continue;
        if (model.row_ub[i] == 0.0 && At.value[kx] > 0.0 && At.value[kz] < 0.0) return true;
        if (model.row_lb[i] == 0.0 && At.value[kx] < 0.0 && At.value[kz] > 0.0) return true;
      }
      return false;
    };
    size_t added = 0;
    for (const QuadRow& q : rows) {
      if (!q.cone || q.cone_gamma != 0.0 || !q.idx.empty()) continue;
      Index u = -1, v = -1;
      bool ok = true;
      for (size_t k = 0; k < q.qc.size() && ok; ++k) {
        if (q.qc[k] == 0.0 || q.qi[k] == q.qj[k]) continue;
        ok = u < 0;
        u = q.qi[k], v = q.qj[k];
      }
      if (!ok || u < 0) continue;
      for (const Index z : {u, v}) {
        if (!binary(z)) continue;
        for (size_t k = 0; k < q.qc.size(); ++k) {
          const Index x = q.qi[k];
          if (q.qc[k] == 0.0 || q.qi[k] != q.qj[k] || x == u || x == v || linked(x, z)) continue;
          if (model.col_lb[x] > 0.0 || model.col_ub[x] < 0.0) continue;  // 0 outside the column's range: left to the cuts
          p.implications.push_back({z, 0.0, x});
          ++added;
        }
      }
    }
    if (added > 0) log.log(1, "Cones switched by binaries: %zu implications (binary 0: member 0)\n", added);
  }
  p.quad_rows = std::move(rows);
  if (products) p.bilinear = *products;  // spatial branching (solve_spatial)
  if (lifted) {
    // The model's row indices become the search's: the quadratic rows left the linear part.
    std::vector<Index> pos(static_cast<size_t>(m), -1);
    for (Index i = 0, k = 0; i < m; ++i) {
      if (!quad[i]) pos[i] = k++;
    }
    for (const MipProblem::LiftedRow& lr : *lifted) {
      if (pos[lr.row] < 0) continue;
      p.lifted_rows.push_back(lr);
      p.lifted_rows.back().row = pos[lr.row];
    }
  }
  // The continuous relaxation by the conic interior-point method (conic/socp.hpp, MILP_CONIC=0: off): it solves a
  // continuous model outright (the outer approximation stays the fallback); with integers, its point gives tangents.
  std::vector<double> relax_x, round_start;
  {
    bool any_int = false;
    for (Index j = 0; j < p.num_cols(); ++j) any_int = any_int || p.is_int[j];
    const char* env = std::getenv("MILP_CONIC");
    ConicProblem cp;
    std::string why;
    ConicMap cmap;
    // (Not with products: the relaxation's point says nothing about them, and its verdicts are for the search.)
    const bool built = !products && !(env && *env == '0') && conic_relaxation(p, cp, &why, &cmap);
    if (!built && !why.empty()) log.log(1, "Conic relaxation skipped: %s\n", why.c_str());
    if (built) {
      ConicOptions co;
      co.log = &log;
      // With integers the relaxation only seeds the search: at most a fifth of the time left.
      const double left = std::max(0.0, params_in.time_limit - timer.elapsed());
      co.time_limit = any_int ? 0.2 * left : left;
      co.threads = params_in.threads > 0 ? params_in.threads : 1;
      const ConicResult cr = solve_conic(cp, co);
      if (cr.status == ConicStatus::Optimal && any_int && start.empty()) {
        // Rounding heuristic: the relaxation's integers rounded and fixed, the continuous rest by the conic method;
        // a feasible point starts the search (MISOCP layouts and flows have no incumbent for a long time without it).
        // (Rounding upwards as a second attempt was tried for cones switched by binaries, CBLIB achtziger *flowc:
        // infeasible there as well, so only the nearest integers.)
        for (const bool up : {false}) {
          MipProblem q = p;
          for (Index j = 0; j < q.num_cols(); ++j) {
            if (!q.is_int[j]) continue;
            const double v = std::clamp(up ? std::ceil(cr.x[j] - 1e-6) : std::round(cr.x[j]), q.col_lb[j], q.col_ub[j]);
            q.col_lb[j] = q.col_ub[j] = v;
          }
          if (!q.implications.empty() && !q.apply_implications(q.col_lb, q.col_ub)) continue;
          ConicProblem cq;
          if (!flatten_fixed_cones(q) || !conic_relaxation(q, cq)) continue;
          ConicOptions co2;
          co2.time_limit = 0.1 * std::max(0.0, params_in.time_limit - timer.elapsed());
          co2.threads = co.threads;
          const ConicResult rr = solve_conic(cq, co2);
          log.log(2, "Conic rounding%s: %s, %d iterations\n", up ? " (upwards)" : "", to_string(rr.status), rr.iterations);
          if (rr.status != ConicStatus::Optimal) continue;
          std::vector<double> x(rr.x.begin(), rr.x.begin() + n);
          for (Index j = 0; j < n; ++j) {
            if (model.col_type[j] == VarType::Integer || model.col_type[j] == VarType::Binary) x[j] = std::round(x[j]);
            else if (q.col_lb[j] == q.col_ub[j]) x[j] = q.col_lb[j];
          }
          log.log(2, "Conic rounding%s: violation %.3g\n", up ? " (upwards)" : "", model.violation(x).max());
          if (model.violation(x).max() <= 1e-6) {
            log.log(1, "Conic rounding%s: a feasible point, objective %.10g\n", up ? " (upwards)" : "", model.objective_value(x));
            round_start = std::move(x);
            break;
          }
        }
      }
      if (cr.status == ConicStatus::Optimal) {
        relax_x = cr.x;
        // The epigraph column was fixed at 0 in the conic form (its row's quadratic part in P): its value is 1/2 x'Qx.
        for (const QuadRow& q : p.quad_rows) {
          if (q.epigraph >= 0 && !q.cone) relax_x[q.epigraph] += q.activity(relax_x.data());
        }
      }
      if (!any_int) {
        if (cr.status == ConicStatus::Optimal) {
          std::vector<double> x(cr.x.begin(), cr.x.begin() + n);
          const double viol = model.violation(x).max();
          if (viol <= 1e-6) {
            r.status = Status::Optimal;
            r.has_solution = true;
            r.x = std::move(x);
            r.objective = model.objective_value(r.x);
            const double bound = sense * std::min(cr.dobj, cr.pobj);  // minimization sense inside
            r.bound = sense > 0 ? std::min(bound, r.objective) : std::max(bound, r.objective);
            // Duals from the conic ones (ConicMap): rows -sign * (y or z); a quadratic row's multiplier (z0 + z1) / 2
            // of its rotated cone, a cone row's z0 / (2 (h x + h0)); the user's sense; reduced costs from the
            // Lagrangian's gradient d = c + Q x - A'y - sum_q y_q grad q(x).
            {
              std::vector<Index> lin_row, quad_row(p.quad_rows.size(), -1);
              for (Index i = 0; i < m; ++i) {
                if (!quad[i]) lin_row.push_back(i);
                else if (slot[i] >= 0) quad_row[slot[i]] = i;
              }
              std::vector<double> y(static_cast<size_t>(m), 0.0);
              for (size_t k = 0; k < cmap.a_kind.size(); ++k) {
                if (cmap.a_kind[k] == 0) y[lin_row[cmap.a_index[k]]] -= cmap.a_sign[k] * cr.y[k];
              }
              for (Index k = 0; k < cp.l; ++k) {
                if (cmap.g_kind[k] == 0) y[lin_row[cmap.g_index[k]]] -= cmap.g_sign[k] * cr.z[k];
              }
              Index off = cp.l;
              for (size_t c = 0; c < cp.cones.size(); ++c) {
                const QuadRow& q = p.quad_rows[cmap.cone_quad[c]];
                double lambda = 0.5 * (cr.z[off] + cr.z[off + 1]);
                if (cmap.cone_norm[c]) {
                  double head = q.cone_h0;
                  for (const auto& [j, h] : q.cone_h) head += h * cr.x[j];
                  lambda = head > 1e-7 ? cr.z[off] / (2.0 * head) : 0.0;  // at the apex: no finite multiplier
                }
                if (quad_row[cmap.cone_quad[c]] >= 0) y[quad_row[cmap.cone_quad[c]]] = -cmap.cone_dir[c] * lambda;
                off += cp.cones[c];
              }
              for (double& v : y) v *= sense;
              std::vector<double> d(model.obj.begin(), model.obj.end());
              const SparseMatrix Qm = model.q_lower();
              for (Index j = 0; j < Qm.num_major; ++j) {
                for (Index k = Qm.begin(j); k < Qm.end(j); ++k) {
                  const Index i = Qm.index[k];
                  d[i] += Qm.value[k] * r.x[j];
                  if (i != j) d[j] += Qm.value[k] * r.x[i];
                }
              }
              for (Index j = 0; j < n; ++j) {
                for (Index k = model.A.begin(j); k < model.A.end(j); ++k) d[j] -= model.A.value[k] * y[model.A.index[k]];
              }
              for (size_t k = 0; k < model.qr_val.size(); ++k) {
                const double yq = y[model.qr_row[k]], c = model.qr_val[k];
                const Index a = model.qr_i[k], b = model.qr_j[k];
                if (a == b) {
                  d[a] -= yq * 2.0 * c * r.x[a];
                } else {
                  d[a] -= yq * c * r.x[b];
                  d[b] -= yq * c * r.x[a];
                }
              }
              r.row_dual = std::move(y);
              r.reduced_cost = std::move(d);
              r.has_duals = true;
            }
            r.stats.time = timer.elapsed();
            return r;
          }
          log.log(1, "  conic solution violates the model by %.2e: outer approximation from its point\n", viol);
        } else if (cr.status == ConicStatus::PrimalInfeasible) {
          r.status = Status::Infeasible;
          r.bound = sense * kInf;
          r.stats.time = timer.elapsed();
          return r;
        } else if (cr.status == ConicStatus::DualInfeasible) {
          r.status = Status::Unbounded;
          r.bound = -sense * kInf;
          r.stats.time = timer.elapsed();
          return r;
        }
      }
    }
  }
  // Initial cuts as rows (tangent planes, valid everywhere): each row's at the bounds' point nearest 0, and for a column
  // free in some direction at 1e3 along it, so that the first LP is bounded where the rows' curvature allows; at the
  // conic relaxation's point for each row nearly active there.
  {
    const Index nc = p.num_cols();
    std::vector<double> x0(static_cast<size_t>(nc), 0.0), grad(static_cast<size_t>(nc), 0.0);
    for (Index j = 0; j < nc; ++j) x0[j] = std::clamp(0.0, p.col_lb[j], p.col_ub[j]);
    std::vector<Index> J, I;
    std::vector<double> V, rlb, rub;
    std::vector<std::pair<Index, double>> g;
    Index row = p.num_rows();
    const bool tdebug = std::getenv("MILP_TANGENT_DEBUG") != nullptr;
    const char* tkind = "";
    double rhs = 0.0;  // outside the lambda: gcc 15 took the local for a dangling pointer in rub.push_back (-Werror)
    auto tangent = [&](const QuadRow& q, const std::vector<double>& pt) {
      rhs = 0.0;
      if (!q.linearize(pt.data(), grad, g, rhs)) return;
      if (tdebug) {
        log.log(0, "tangent row %d (%s, quad row %d, cone %d, gamma %g, h0 %g):", row, tkind,
                static_cast<int>(&q - p.quad_rows.data()), q.cone ? 1 : 0, q.cone_gamma, q.cone_h0);
        for (const auto& [j, v] : g) log.log(0, " %+.6g*x%d", v, j);
        log.log(0, " <= %.10g\n", rhs);
      }
      for (const auto& [j, v] : g) J.push_back(j), I.push_back(row), V.push_back(v);
      rlb.push_back(-kInf);
      rub.push_back(rhs);
      ++row;
    };
    for (const QuadRow& q : p.quad_rows) {
      if (!relax_x.empty()) {
        const double a = q.activity(relax_x.data()), side = is_finite(q.hi) ? q.hi : q.lo;
        tkind = "relax";
        if (std::abs(a - side) <= 1e-3 * std::max(1.0, q.magnitude(relax_x.data()))) tangent(q, relax_x);
      }
      tkind = "x0";
      tangent(q, x0);
      tkind = "far";
      std::vector<Index> cols(q.qi);
      cols.insert(cols.end(), q.qj.begin(), q.qj.end());
      std::sort(cols.begin(), cols.end());
      cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
      if (cols.size() > 50) continue;
      for (Index j : cols) {
        for (double d : {1e3, -1e3}) {
          if (is_finite(d > 0 ? p.col_ub[j] : p.col_lb[j])) continue;
          std::vector<double> pt = x0;
          pt[j] += d;
          tangent(q, pt);
        }
      }
    }
    if (!rlb.empty()) {
      for (Index j = 0; j < nc; ++j)
        for (Index k = p.A.begin(j); k < p.A.end(j); ++k) J.push_back(j), I.push_back(p.A.index[k]), V.push_back(p.A.value[k]);
      p.A = SparseMatrix::from_triplets(nc, row, J, I, V);
      p.row_lb.insert(p.row_lb.end(), rlb.begin(), rlb.end());
      p.row_ub.insert(p.row_ub.end(), rub.begin(), rub.end());
    }
  }
  log.log(1, "Quadratic constraints: %zu convex rows (%zu second-order cones) by outer approximation (gradient cuts)%s; "
          "presolve, symmetry handling and restarts off\n", p.quad_rows.size() - (model.has_quadratic() ? 1 : 0), cones,
          model.has_quadratic() ? ", the objective by its epigraph" : "");
  SolverParams params = params_in;
  params.symmetry = false;
  SearchOptions opts;
  opts.allow_restart = false;
  opts.allow_tree_restart = false;
  opts.allow_est_restart = false;
  if (tracer) opts.on_trace = tracer->hook();
  const std::vector<double>& st = start.empty() ? round_start : start;
  if (static_cast<Index>(st.size()) == n) {
    bool complete = true;
    for (double v : st) complete = complete && std::isfinite(v);
    if (complete) {
      opts.start = st;
      if (model.has_quadratic()) opts.start.push_back(sense * (model.objective_value(st) - model.obj_offset));
      for (Index j = 0; j < model.num_cols(); ++j) {
        if (model.obj[j] != 0.0 && model.has_quadratic()) opts.start.back() -= sense * model.obj[j] * st[j];
      }
    }
  }
  const MipResult mr = branch_and_bound(p, params, log, timer, opts);
  r.status = mr.status;
  r.has_solution = mr.has_solution;
  if (mr.has_solution) {
    r.x.assign(mr.x.begin(), mr.x.begin() + n);
    r.objective = model.objective_value(r.x);
  }
  r.bound = sense * mr.bound;
  r.stats.nodes = mr.nodes;
  r.stats.lp_iterations = mr.lp_iterations;
  r.stats.root_bound = sense * mr.root_bound;
  r.stats.time = timer.elapsed();
  return r;
}

// Bounds from the quadratic rows themselves (any convexity), for the columns with a square term: in a row
//   lo <= c x^2 + B x + R <= hi,  B = a_k + sum_j c_kj x_j,  R = the terms without x,
// with B and R in intervals from the other columns' bounds, c > 0 gives c x^2 - beta |x| <= hi - min R (beta =
// max |B|), so |x| <= (beta + sqrt(beta^2 + 4 c U)) / (2 c) with U = max(hi - min R, 0); c < 0 the same from the
// lower side. A few passes (a new bound makes other intervals finite). Returns the number of bounds tightened.
size_t quad_row_bounds(const Model& model, std::vector<double>& lo, std::vector<double>& hi) {
  const Index n = model.num_cols(), m = model.num_rows();
  const SparseMatrix At = model.A.transpose();
  std::vector<std::vector<size_t>> terms(static_cast<size_t>(m));
  for (size_t k = 0; k < model.qr_val.size(); ++k) {
    if (model.qr_val[k] != 0.0) terms[model.qr_row[k]].push_back(k);
  }
  auto mul = [](double a, double b) { return a == 0.0 || b == 0.0 ? 0.0 : a * b; };
  struct Iv {
    double l = 0.0, u = 0.0;
    void add(double a, double b) { l += a, u += b; }
  };
  // c * x_i * x_j over the bounds.
  auto term_iv = [&](double c, Index i, Index j) {
    double a, b;
    if (i == j) {
      a = lo[i] <= 0.0 && hi[i] >= 0.0 ? 0.0 : std::min(mul(lo[i], lo[i]), mul(hi[i], hi[i]));
      b = std::max(mul(lo[i], lo[i]), mul(hi[i], hi[i]));
    } else {
      const double v[4] = {mul(lo[i], lo[j]), mul(lo[i], hi[j]), mul(hi[i], lo[j]), mul(hi[i], hi[j])};
      a = *std::min_element(v, v + 4), b = *std::max_element(v, v + 4);
    }
    return c >= 0 ? Iv{mul(c, a), mul(c, b)} : Iv{mul(c, b), mul(c, a)};
  };
  size_t found = 0;
  for (int pass = 0; pass < 10; ++pass) {
    bool any = false;
    for (Index r = 0; r < m; ++r) {
      if (terms[r].empty() || terms[r].size() > 5000) continue;
      for (const size_t kk : terms[r]) {
        if (model.qr_i[kk] != model.qr_j[kk]) continue;
        const Index x = model.qr_i[kk];
        const double c = model.qr_val[kk];
        Iv B, R;
        for (Index p = At.begin(r); p < At.end(r); ++p) {
          const Index j = At.index[p];
          const double a = At.value[p];
          if (j == x) B.add(a, a);
          else R.add(a >= 0 ? mul(a, lo[j]) : mul(a, hi[j]), a >= 0 ? mul(a, hi[j]) : mul(a, lo[j]));
        }
        for (const size_t t : terms[r]) {
          if (t == kk) continue;
          const Index i = model.qr_i[t], j = model.qr_j[t];
          const double v = model.qr_val[t];
          if (i == x || j == x) {
            const Index o = i == x ? j : i;
            B.add(v >= 0 ? mul(v, lo[o]) : mul(v, hi[o]), v >= 0 ? mul(v, hi[o]) : mul(v, lo[o]));
          } else {
            const Iv iv = term_iv(v, i, j);
            R.add(iv.l, iv.u);
          }
        }
        if (std::isnan(B.l) || std::isnan(B.u) || !is_finite(B.l) || !is_finite(B.u)) continue;
        const double beta = std::max(std::abs(B.l), std::abs(B.u));
        // c x^2 + B x <= U (c > 0), or -c x^2 - B x <= U (c < 0).
        double U;
        if (c > 0) {
          if (!is_finite(model.row_ub[r]) || std::isnan(R.l) || !is_finite(R.l)) continue;
          U = model.row_ub[r] - R.l;
        } else {
          if (!is_finite(model.row_lb[r]) || std::isnan(R.u) || !is_finite(R.u)) continue;
          U = R.u - model.row_lb[r];
        }
        const double ac = std::abs(c);
        double bnd = (beta + std::sqrt(beta * beta + 4.0 * ac * std::max(U, 0.0))) / (2.0 * ac);
        bnd += 1e-9 * std::max(1.0, bnd);
        const bool integer = model.col_type[x] == VarType::Integer || model.col_type[x] == VarType::Binary;
        if (integer) bnd = std::floor(bnd + 1e-9);
        // Only clear gains: a bound that was infinite, or 0.1% of the domain.
        const double gain = is_finite(hi[x] - lo[x]) ? 1e-3 * (hi[x] - lo[x]) : 0.0;
        if (bnd < hi[x] - gain && bnd >= lo[x]) hi[x] = bnd, ++found, any = true;
        if (-bnd > lo[x] + gain && -bnd <= hi[x]) lo[x] = -bnd, ++found, any = true;
      }
    }
    if (!any) break;
  }
  (void)n;
  return found;
}

// Nonconvex quadratic models (v0.10; MILP_SPATIAL=0: off, the model is then reported NONCONVEX as before): spatial
// branch-and-bound. Every product x_i x_j of a quadratic row that is not convex, and of the objective if that is not
// convex, becomes a column w with McCormick rows over the factors' bounds (a square: its secant and three tangents);
// the rows and the objective are then linear in the w. Convex quadratic rows and a convex objective stay as they
// are (outer approximation). The search keeps w = x_i x_j by McCormick rows of each node's bounds and by branching
// on the factors, continuous ones included (MipProblem::bilinear). A factor without finite bounds (the linear rows'
// implied bounds count) has only the McCormick rows its finite bounds give, until branching bounds it. nullopt with
// the reason if the model has logical constraints or nothing in it is nonconvex.
std::optional<SolveResult> solve_spatial(const Model& model, const SolverParams& params, const std::vector<double>& start,
                                         const Timer& timer, const Logger& log, IncumbentTracer* tracer, std::string& why) {
  if (model.has_logical()) {
    why = "logical constraints";
    return std::nullopt;
  }
  const Index n = model.num_cols(), m = model.num_rows();
  const double sense = model.sense == ObjSense::Maximize ? -1.0 : 1.0;
  std::vector<char> quad, bad;
  bool obj_bad = false;
  {
    std::vector<QuadRow> rows;
    size_t cones = 0;
    std::string w;
    build_quad_rows(model, quad, rows, cones, w, &bad);
    if (model.has_quadratic()) {
      SparseMatrix Q = model.q_lower();
      for (double& v : Q.value) v *= sense;
      obj_bad = !check_convexity(Q).convex;
    }
  }
  if (!obj_bad && std::find(bad.begin(), bad.end(), 1) == bad.end()) {
    why = "nothing nonconvex";
    return std::nullopt;
  }
  // Bounds of the factors: the columns' own, tightened by the linear rows.
  std::vector<double> lo = model.col_lb, hi = model.col_ub;
  {
    Model lin = model;
    lin.q_row.clear(), lin.q_col.clear(), lin.q_val.clear();
    lin.qr_row.clear(), lin.qr_i.clear(), lin.qr_j.clear(), lin.qr_val.clear();
    lin.remove_rows(quad);
    implied_bounds(lin, lo, hi);
    for (Index j = 0; j < n; ++j) {
      lo[j] = std::max(lo[j], model.col_lb[j]), hi[j] = std::min(hi[j], model.col_ub[j]);
      if (lo[j] > hi[j]) lo[j] = hi[j] = model.col_lb[j] > hi[j] ? model.col_lb[j] : hi[j];  // crossed within rounding
    }
  }
  {
    const size_t found = quad_row_bounds(model, lo, hi);
    if (found > 0) log.log(1, "  %zu bounds from the quadratic rows' square terms\n", found);
  }
  // A factor still without a finite bound: the bound from the convex part of the model (its linear and convex
  // quadratic rows, integers relaxed), by minimizing and maximizing the column there. A circle x^2 + y^2 <= 4
  // bounds x and y, and no linear row says so. At most 200 columns and a tenth of the time limit.
  {
    std::vector<char> factor(static_cast<size_t>(n), 0);
    for (size_t k = 0; k < model.qr_val.size(); ++k) {
      if (model.qr_val[k] != 0.0 && bad[model.qr_row[k]]) factor[model.qr_i[k]] = factor[model.qr_j[k]] = 1;
    }
    if (obj_bad) {
      for (size_t k = 0; k < model.q_val.size(); ++k) {
        if (model.q_val[k] != 0.0) factor[model.q_row[k]] = factor[model.q_col[k]] = 1;
      }
    }
    std::vector<Index> open;
    for (Index j = 0; j < n; ++j) {
      if (factor[j] && (!is_finite(lo[j]) || !is_finite(hi[j]))) open.push_back(j);
    }
    if (!open.empty() && open.size() <= 200) {
      Model R = model;
      R.q_row.clear(), R.q_col.clear(), R.q_val.clear();
      R.remove_rows(bad);
      R.compact_qrows();
      for (VarType& t : R.col_type) t = VarType::Continuous;
      R.col_lb = lo, R.col_ub = hi;
      R.sense = ObjSense::Minimize;
      R.obj_offset = 0.0;
      SolverParams rp = params;
      rp.verbosity = 0;
      rp.threads = 1;
      const double budget = timer.elapsed() + 0.1 * std::max(0.0, params.time_limit - timer.elapsed());
      size_t found = 0;
      for (const Index j : open) {
        for (const double dir : {1.0, -1.0}) {
          if (dir > 0 ? is_finite(lo[j]) : is_finite(hi[j])) continue;
          if (timer.elapsed() >= budget) break;
          std::fill(R.obj.begin(), R.obj.end(), 0.0);
          R.obj[j] = dir;
          rp.time_limit = budget - timer.elapsed();
          const SolveResult rr = solve(R, rp);
          if (rr.status != Status::Optimal || !std::isfinite(rr.bound)) continue;
          // The dual bound, widened by the solve's tolerance.
          const double b = rr.bound - 1e-6 * std::max(1.0, std::abs(rr.bound));
          if (dir > 0) lo[j] = b; else hi[j] = -b;
          R.col_lb[j] = lo[j], R.col_ub[j] = hi[j];
          ++found;
        }
      }
      if (found > 0) log.log(1, "  %zu bounds of unbounded factors from the convex part of the model\n", found);
    }
  }
  quad_row_bounds(model, lo, hi);
  Model L = model;
  std::vector<Bilinear> terms;
  std::vector<MipProblem::LiftedRow> lifted;
  std::map<Index, size_t> lifted_of;  // model row -> its entry in lifted
  // With very many products their root McCormick rows are left out (the search adds a node's rows where a point
  // violates them): QPLIB_8758's 1,072,260 products would be 4.3 million rows, 6.9 GB.
  size_t product_terms = obj_bad ? model.q_val.size() : 0;
  for (size_t k = 0; k < model.qr_val.size(); ++k) product_terms += model.qr_val[k] != 0.0 && bad[model.qr_row[k]];
  const bool root_rows = product_terms <= 50000;
  std::map<std::pair<Index, Index>, Index> wcol;
  std::vector<Index> ti, tj;  // new matrix entries (row, column)
  std::vector<double> tv;
  Index rows_now = m;
  auto new_row = [&](double rlb, double rub, std::initializer_list<std::pair<Index, double>> coef) {
    for (const auto& [j, v] : coef) {
      if (v != 0.0) ti.push_back(rows_now), tj.push_back(j), tv.push_back(v);
    }
    L.row_lb.push_back(rlb);
    L.row_ub.push_back(rub);
    L.row_names.push_back("mc" + std::to_string(rows_now));
    ++rows_now;
  };
  size_t open_factors = 0;  // products with a factor that has an infinite bound
  // a * b for bounds: 0 * inf = 0.
  auto mul = [](double a, double b) { return a == 0.0 || b == 0.0 ? 0.0 : a * b; };
  auto product = [&](Index i, Index j) -> Index {
    if (i < j) std::swap(i, j);
    const auto it = wcol.find({i, j});
    if (it != wcol.end()) return it->second;
    const Index w = L.num_cols();
    const double lx = lo[i], ux = hi[i], ly = lo[j], uy = hi[j];
    const bool flx = is_finite(lx), fux = is_finite(ux), fly = is_finite(ly), fuy = is_finite(uy);
    open_factors += !(flx && fux && fly && fuy);
    double wl, wu;
    if (i == j) {
      wl = lx <= 0.0 && ux >= 0.0 ? 0.0 : std::min(mul(lx, lx), mul(ux, ux));
      wu = std::max(mul(lx, lx), mul(ux, ux));
    } else {
      wl = std::min({mul(lx, ly), mul(lx, uy), mul(ux, ly), mul(ux, uy)});
      wu = std::max({mul(lx, ly), mul(lx, uy), mul(ux, ly), mul(ux, uy)});
    }
    if (!is_finite(wl)) wl = -kInf;
    if (!is_finite(wu)) wu = kInf;
    L.obj.push_back(0.0);
    L.col_lb.push_back(wl);
    L.col_ub.push_back(wu);
    L.col_type.push_back(VarType::Continuous);
    L.col_names.push_back("w" + std::to_string(i) + "_" + std::to_string(j));
    // Each McCormick row needs two of the four bounds: only the rows whose bounds are finite exist (the search
    // adds the others once branching has made the bounds finite).
    if (!root_rows) {
    } else if (i == j) {
      // w <= secant; w >= the tangents at the ends and the middle (the search adds those at its points).
      if (flx && fux) new_row(-kInf, -lx * ux, {{w, 1.0}, {i, -(lx + ux)}});
      const double a = flx ? lx : fux ? ux - 1.0 : -1.0, b = fux ? ux : flx ? lx + 1.0 : 1.0;
      for (const double x0 : {a, 0.5 * (a + b), b}) new_row(-kInf, x0 * x0, {{w, -1.0}, {i, 2.0 * x0}});
    } else {
      if (flx && fly) new_row(-kInf, lx * ly, {{w, -1.0}, {i, ly}, {j, lx}});
      if (fux && fuy) new_row(-kInf, ux * uy, {{w, -1.0}, {i, uy}, {j, ux}});
      if (fux && fly) new_row(-kInf, -ux * ly, {{w, 1.0}, {i, -ly}, {j, -ux}});
      if (flx && fuy) new_row(-kInf, -lx * uy, {{w, 1.0}, {i, -uy}, {j, -lx}});
    }
    terms.push_back({w, i, j});
    wcol[{i, j}] = w;
    return w;
  };
  // The rows that are not convex: their products by columns.
  {
    std::vector<Index> qr, qi, qj;
    std::vector<double> qv;
    for (size_t k = 0; k < model.qr_val.size(); ++k) {
      const Index row = model.qr_row[k];
      if (model.qr_val[k] == 0.0) continue;
      if (!bad[row]) {
        qr.push_back(row), qi.push_back(model.qr_i[k]), qj.push_back(model.qr_j[k]), qv.push_back(model.qr_val[k]);
        continue;
      }
      const Index w = product(model.qr_i[k], model.qr_j[k]);
      ti.push_back(row), tj.push_back(w), tv.push_back(model.qr_val[k]);
      const auto [it, fresh] = lifted_of.try_emplace(row, lifted.size());
      if (fresh) lifted.push_back({row, {}});
      lifted[it->second].terms.push_back({w - n, model.qr_val[k]});
    }
    L.qr_row = std::move(qr), L.qr_i = std::move(qi), L.qr_j = std::move(qj), L.qr_val = std::move(qv);
  }
  if (obj_bad) {
    for (size_t k = 0; k < model.q_val.size(); ++k) {
      if (model.q_val[k] == 0.0) continue;
      const Index w = product(model.q_row[k], model.q_col[k]);
      L.obj[w] += (model.q_row[k] == model.q_col[k] ? 0.5 : 1.0) * model.q_val[k];
    }
    L.q_row.clear(), L.q_col.clear(), L.q_val.clear();
  }
  // The bounds found (valid for every feasible point), as the McCormick rows use them.
  for (Index j = 0; j < n; ++j) L.col_lb[j] = lo[j], L.col_ub[j] = hi[j];
  {
    std::vector<Index> J, I;
    std::vector<double> V;
    for (Index j = 0; j < n; ++j) {
      for (Index k = model.A.begin(j); k < model.A.end(j); ++k) J.push_back(j), I.push_back(model.A.index[k]), V.push_back(model.A.value[k]);
    }
    J.insert(J.end(), tj.begin(), tj.end());
    I.insert(I.end(), ti.begin(), ti.end());
    V.insert(V.end(), tv.begin(), tv.end());
    L.A = SparseMatrix::from_triplets(L.num_cols(), rows_now, J, I, V);
  }
  size_t squares = 0;
  for (const Bilinear& b : terms) squares += b.x == b.y;
  log.log(1, "Nonconvex quadratic model: spatial branch-and-bound on %zu products (%zu squares) of %s%zu rows\n",
          terms.size(), squares, obj_bad ? "the objective and " : "",
          static_cast<size_t>(std::count(bad.begin(), bad.end(), 1)));
  if (open_factors > 0) {
    log.log(1, "  %zu products have a factor without finite bounds: no McCormick rows there until branching bounds it\n",
            open_factors);
  }
  std::vector<double> st;
  if (static_cast<Index>(start.size()) == n) {
    st = start;
    for (const Bilinear& b : terms) st.push_back(start[b.x] * start[b.y]);
  }
  SolveResult r = solve_oa_core(L, params, st, timer, log, tracer, &terms, &lifted);
  if (r.has_solution) {
    r.x.resize(static_cast<size_t>(n));
    r.objective = model.objective_value(r.x);
    // The model as given, with its products computed from the factors: each row relative to the size of its terms.
    const std::vector<double> act = model.row_activity(r.x);
    std::vector<double> scale(static_cast<size_t>(m), 1.0);
    for (size_t k = 0; k < model.qr_val.size(); ++k) {
      double& sc = scale[model.qr_row[k]];
      sc = std::max(sc, std::abs(model.qr_val[k] * r.x[model.qr_i[k]] * r.x[model.qr_j[k]]));
    }
    double worst = 0.0;
    for (Index i = 0; i < m; ++i) {
      const double v = std::max((model.row_lb[i] - act[i]) / scale[i], (act[i] - model.row_ub[i]) / scale[i]);
      worst = v <= worst ? worst : std::isnan(v) ? kInf : v;
    }
    for (const double v : r.x) {
      if (!std::isfinite(v)) worst = kInf;
    }
    if (!(worst <= 1.5e-6) || !std::isfinite(r.objective)) {
      // The products' tolerances added up in some row: no answer rather than a wrong one.
      log.log(1, "ERROR: the solution violates a row of the model by %.2e (relative to the row's terms); discarded\n", worst);
      r.has_solution = false;
      r.x.clear();
      r.status = Status::NumericalError;
    }
  }
  return r;
}

// A separable convex objective on integer columns (square_terms.hpp) as a MILP: tangent rows of each square
// around the continuous relaxation's point, the MILP solved by the MILP pipeline, its solution scored with
// the exact objective, and tangent windows added where it falls outside them, until the solution is exact.
// Each MILP is a relaxation of the model, so its bound is the model's. MILP_SQ_WINDOW: the window (25);
// MILP_SQ_ROUNDS: at most that many MILPs (10); MILP_SQ_CENTER=0: no relaxation (centers at 0). A user's integer QP (141637 rows, 265 squared general
// integers): the MIQP search had no incumbent after 180 s (its root dive solves a barrier QP per step); as
// a MILP, OPTIMAL at the root in 4.6 s (root LP 9: the tangents give the convex hull of x^2 on the integers).
SolveResult solve_square_milp(const Model& model, SquareTerms& sq, const SolverParams& params,
                              const std::vector<double>& start, const Timer& timer, const Logger& log,
                              IncumbentTracer* tracer) {
  const double sense = model.sense == ObjSense::Maximize ? -1.0 : 1.0;
  const Index n = model.num_cols();
  log.log(1, "Quadratic objective: %zu squares of integer columns (%zu binary), solved as a MILP with tangent rows\n",
          sq.num_terms(), sq.num_terms() - sq.num_general());
  SolveResult r;
  const char* center_env = std::getenv("MILP_SQ_CENTER");  // 0: no relaxation, centers at 0 (in the bounds)
  if (sq.num_general() > 0 && !(center_env && *center_env == '0')) {
    // Centers: the continuous relaxation's point.
    Model rel = model;
    for (VarType& t : rel.col_type) {
      if (t == VarType::Integer || t == VarType::Binary) t = VarType::Continuous;
    }
    const Logger quiet(log.verbosity() >= 2 ? 1 : 0);
    const SolveResult rr = solve_quadratic(rel, params, {}, timer, quiet, nullptr);
    if (rr.has_solution) sq.center_at(rr.x);
    log.log(1, "  relaxation %s %.10g (%.2fs); %zu tangent rows\n", to_string(rr.status), rr.objective,
            timer.elapsed(), sq.num_tangents());
  }
  const char* rounds_env = std::getenv("MILP_SQ_ROUNDS");
  const int max_rounds = rounds_env ? std::max(1, std::atoi(rounds_env)) : 10;
  std::vector<double> best;
  double best_obj = kInf;      // minimization sense
  double best_bound = -kInf;   // minimization sense, valid for the model
  bool converged = false, infeasible = false;
  Status last = Status::Unknown;
  long long nodes = 0, iters = 0;
  for (int round = 0; round < max_rounds; ++round) {
    const Model milp = sq.build();
    std::vector<double> st;
    if (!best.empty()) {
      st = sq.extend(best);
    } else if (!start.empty() && static_cast<Index>(start.size()) == n) {
      bool complete = true;
      for (double v : start) complete = complete && std::isfinite(v);
      st = complete ? sq.extend(start) : start;
      st.resize(static_cast<size_t>(milp.num_cols()), std::numeric_limits<double>::quiet_NaN());
    }
    const SolveResult s = solve_plain(milp, {}, params, st, timer, log, nullptr, tracer, false);
    nodes += s.stats.nodes;
    iters += s.stats.lp_iterations;
    last = s.status;
    if (s.status == Status::Optimal || s.status == Status::TimeLimit || s.status == Status::NodeLimit ||
        s.status == Status::Interrupted || s.status == Status::IterationLimit || s.status == Status::MemoryLimit) {
      best_bound = std::max(best_bound, sense * s.bound);
    }
    if (s.status == Status::Infeasible) {
      infeasible = best.empty();
      converged = !best.empty();
      break;
    }
    if (!s.has_solution) break;
    std::vector<double> x(s.x.begin(), s.x.begin() + n);
    const double obj = sense * model.objective_value(x);
    if (obj < best_obj) best_obj = obj, best = std::move(x);
    if (s.status != Status::Optimal) break;
    const int missed = sq.refine(std::vector<double>(s.x.begin(), s.x.begin() + n));
    if (missed == 0) {
      converged = true;
      break;
    }
    log.log(1, "Tangent round %d: %d squares outside their tangents (MILP %.10g, exact %.10g); %zu tangent rows\n",
            round + 1, missed, s.objective, sense * obj, sq.num_tangents());
  }
  r.stats.nodes = nodes;
  r.stats.lp_iterations = iters;
  if (infeasible) {
    r.status = Status::Infeasible;
    r.bound = sense * kInf;
  } else {
    if (!best.empty()) {
      r.has_solution = true;
      r.x = std::move(best);
      r.objective = model.objective_value(r.x);
      best_bound = std::min(best_bound, best_obj);
    }
    const double tol = std::max(params.mip_abs_gap, params.mip_rel_gap * std::abs(best_obj));
    if (r.has_solution && (converged || best_obj - best_bound <= tol)) {
      r.status = Status::Optimal;
    } else {
      r.status = last == Status::Optimal ? Status::Unknown : last;
    }
    r.bound = sense * best_bound;
  }
  r.stats.root_bound = r.bound;
  r.stats.time = timer.elapsed();
  return r;
}

// A model with a quadratic objective: the QP presolve (qp/qp_presolve.hpp), the convexity test,
// then the barrier (continuous QP) or the MIQP branch-and-bound. A separable convex objective on integer
// columns goes to solve_square_milp (MILP_SQ_MILP=0: the MIQP search).
// The MIQP race's own thread runs solve_quadratic without racing again (the MIQP branch-and-bound route).
thread_local bool t_in_race = false;

bool conclusive(Status s) {
  return s == Status::Optimal || s == Status::Infeasible || s == Status::Unbounded ||
         s == Status::InfeasibleOrUnbounded || s == Status::NonConvex;
}

// MILP_MIQP_RACE: the MIQP branch-and-bound (one thread) and the outer approximation in the MILP search (the other
// threads) on the same model; the first conclusive answer stops the other (LocalStopScope), else the better incumbent
// with the tighter of the two (valid) bounds.
SolveResult race_miqp(const Model& model, const SolverParams& params, const std::vector<double>& start,
                      const Timer& timer, const Logger& log, IncumbentTracer* tracer, int threads) {
  std::atomic<bool> stop_miqp{false}, stop_oa{false};
  SolveResult rm, ro;
  SolverParams pm = params, po = params;
  pm.threads = 1;
  po.threads = threads - 1;
  log.log(1, "MIQP race: the MIQP branch-and-bound (1 thread) against the outer approximation (%d threads)\n",
          po.threads);
  const Logger quiet(0);
  std::thread t(inherit_interrupt([&] {
    LocalStopScope scope(&stop_miqp);
    t_in_race = true;
    rm = solve_quadratic(model, pm, start, timer, quiet, nullptr);
    if (conclusive(rm.status)) stop_oa = true;
  }));
  {
    LocalStopScope scope(&stop_oa);
    ro = solve_oa(model, po, start, timer, log, tracer);
    if (conclusive(ro.status)) stop_miqp = true;
  }
  t.join();
  const double sense = model.sense == ObjSense::Maximize ? -1.0 : 1.0;
  const bool cm = conclusive(rm.status), co = conclusive(ro.status);
  SolveResult r;
  bool won_m = false;
  if (cm != co) {
    won_m = cm;
    r = cm ? rm : ro;
  } else if (cm) {
    r = ro;
  } else {
    // Neither finished: the better incumbent, the tighter bound (both valid).
    const bool use_m = rm.has_solution && (!ro.has_solution || sense * rm.objective < sense * ro.objective);
    won_m = use_m;
    r = use_m ? rm : ro;
    const double bm = rm.bound, bo = ro.bound;
    if (std::isfinite(bm) && std::isfinite(bo)) r.bound = sense > 0 ? std::max(bm, bo) : std::min(bm, bo);
    else if (std::isfinite(bm) || std::isfinite(bo)) r.bound = std::isfinite(bm) ? bm : bo;
    if (r.status == Status::Interrupted) r.status = (use_m ? ro : rm).status;
  }
  log.log(1, "MIQP race: %s (MIQP search %s %.10g, outer approximation %s %.10g)\n",
          won_m ? "the MIQP search's result" : "the outer approximation's result", to_string(rm.status), rm.objective,
          to_string(ro.status), ro.objective);
  r.stats.time = timer.elapsed();
  return r;
}

SolveResult solve_quadratic(const Model& model, const SolverParams& params, const std::vector<double>& start,
                            const Timer& timer, const Logger& log, IncumbentTracer* tracer) {
  {
    const char* on = std::getenv("MILP_SQ_MILP");
    const char* w = std::getenv("MILP_SQ_WINDOW");
    SquareTerms sq;
    if (!(on && *on == '0') && sq.init(model, w ? std::atoi(w) : 25)) {
      return solve_square_milp(model, sq, params, start, timer, log, tracer);
    }
    // A nonconvex objective over binaries (and bounded columns times binaries): its products linearized, a MILP.
    static const LinearizeOptions opts;
    const auto lr = solve_linearized(model, linearize_objective(model, opts), start, log,
                                     [&](const Model& lm, const std::vector<double>& st) {
                                       return solve_plain(lm, {}, params, st, timer, log, nullptr, tracer, true);
                                     });
    if (lr) return *lr;
    // MILP_MIQP_OA=1 (opt-in until its A/B): other convex MIQPs by outer approximation in the MILP search (the
    // objective's epigraph row and gradient cuts, solve_oa) instead of the MIQP branch-and-bound.
    const char* oa = std::getenv("MILP_MIQP_OA");
    if (oa && *oa == '1' && model.num_integers() > 0 && !model.has_logical()) {
      return solve_oa(model, params, start, timer, log, tracer);
    }
    // Default mode, 2 threads or more: race_miqp (MILP_MIQP_RACE=0: off). Neither route wins everywhere (QPLIB_5577 /
    // 5924: the outer approximation; 3694 / 3708 and the generated families: the MIQP search). Server A/B (QPLIB MIQP +
    // generated MIQP families, 120 s): solved 19 = 19, sgm 1.003; incumbents where neither finishes: 5577 none ->
    // 7195003.6 (optimum 7193179.5), 5924 none -> 118886.3 (118609.8), 4270 105.3 -> 99.5, 3861 436.2 -> 431.8.
    const char* race = std::getenv("MILP_MIQP_RACE");
    const int threads = params.threads > 0 ? params.threads : static_cast<int>(std::thread::hardware_concurrency());
    if (!(race && *race == '0') && !t_in_race && model.num_integers() > 0 && !model.has_logical() &&
        !params.deterministic && threads >= 2) {
      return race_miqp(model, params, start, timer, log, tracer, threads);
    }
  }
  SolveResult r;
  MipProblem p;
  if (!MipProblem::from_model(model, &p)) {
    r.status = Status::Infeasible;
    r.stats.time = timer.elapsed();
    return r;
  }
  const double sense = model.sense == ObjSense::Maximize ? -1.0 : 1.0;
  log.log(1, "Quadratic objective: %d nonzeros in the lower triangle of Q\n", p.Q.nnz());
  QpPresolver pre;
  const MipProblem* sp = &p;  // the problem solved: p or the presolved one
  if (params.presolve && p.implications.empty()) {
    const double t0 = timer.elapsed();
    const QpPresolveStatus qps = pre.run(p, log);
    PresolveSummary& ps = r.stats.presolve;
    ps.ran = true;
    ps.rows_before = p.num_rows(), ps.cols_before = p.num_cols(), ps.nnz_before = p.A.nnz();
    if (qps != QpPresolveStatus::Infeasible) {
      ps.rows = pre.reduced().num_rows(), ps.cols = pre.reduced().num_cols(), ps.nnz = pre.reduced().A.nnz();
    }
    ps.time = timer.elapsed() - t0;
    if (qps == QpPresolveStatus::Infeasible) {
      r.status = Status::Infeasible;
      r.bound = sense * kInf;
      r.stats.root_bound = r.bound;
      r.stats.time = timer.elapsed();
      return r;
    }
    sp = &pre.reduced();
  }
  // Convexity of the problem solved (cross terms with fixed columns are linear terms).
  const ConvexityResult cv = check_convexity(sp->Q);
  if (!cv.convex) {
    std::string msg = cv.message;
    if (model.sense == ObjSense::Maximize) msg += " (maximization needs a concave objective)";
    log_nonconvex(log, msg);
    r.status = Status::NonConvex;
    r.stats.time = timer.elapsed();
    return r;
  }
  if (cv.perturbed) log.log(1, "WARNING: %s\n", cv.message.c_str());
  const std::vector<double>& shift = cv.diag_shift;
  auto postsolve = [&](const std::vector<double>& x) { return sp == &p ? x : pre.postsolve(x); };
  const int threads = params.threads > 0 ? params.threads : static_cast<int>(std::thread::hardware_concurrency());
  if (model.num_integers() == 0) {
    if (!start.empty()) log.log(1, "QP: the start solution is not used (the barrier has no warm start)\n");
    QpSolution qs;
    bool no_duals = false;
    if (sp->num_cols() == 0) {
      // Presolve fixed every column (and removed every row).
      qs.status = QpStatus::Optimal;
    } else {
      // The conic interior-point method (homogeneous self-dual) with Q as its P, row duals from its A / G duals
      // through the ConicMap; the QP barrier on anything but an optimum or the time limit: its infeasibility
      // certificates are not trusted (Maros-Meszaros QFORPLAN / QGFRDXPN / QSHELL, objectives of 1e9 to 1e12: "primal
      // infeasible", the barrier's optimum in 0.0 s), nor its dual ones (unbounded or infeasible?). The barrier
      // stalls where the embedding does not (QPLIB_8559 / 8567: primal infeasibility -4% per iteration; conic 5 s).
      // MILP_QP_CONIC=0: the barrier only.
      const char* qc = std::getenv("MILP_QP_CONIC");
      bool done = false;
      ConicProblem cp;
      ConicMap cmap;
      if (!(qc && *qc == '0') && conic_relaxation(*sp, cp, nullptr, &cmap)) {
        if (!shift.empty()) {
          std::vector<Index> J, I;
          std::vector<double> V;
          for (Index j = 0; j < cp.P.num_major; ++j)
            for (Index k = cp.P.begin(j); k < cp.P.end(j); ++k) J.push_back(j), I.push_back(cp.P.index[k]), V.push_back(cp.P.value[k]);
          for (Index j = 0; j < sp->num_cols(); ++j)
            if (shift[j] != 0.0) J.push_back(j), I.push_back(j), V.push_back(shift[j]);
          cp.P = SparseMatrix::from_triplets(sp->num_cols(), sp->num_cols(), J, I, V);
        }
        ConicOptions co;
        co.log = &log;
        co.time_limit = std::max(0.0, params.time_limit - timer.elapsed());
        co.threads = std::max(1, threads);
        const ConicResult cr = solve_conic(cp, co);
        done = true;
        qs.iterations = cr.iterations;
        no_duals = true;
        if (cr.status == ConicStatus::Optimal && sp->max_violation(cr.x) <= 1e-6) {
          qs.status = QpStatus::Optimal;
          qs.x = cr.x;
          // Within the bounds exactly, as the barrier's point (a fixed column came back 3.0000000000000004 for 3).
          for (Index j = 0; j < sp->num_cols(); ++j) qs.x[j] = std::clamp(qs.x[j], sp->col_lb[j], sp->col_ub[j]);
          qs.approximate = cr.reduced_accuracy;
          // Row duals as the QP barrier's (p's minimization): A rows and orthant G rows are sign times a row.
          qs.y.assign(static_cast<size_t>(sp->num_rows()), 0.0);
          for (size_t k = 0; k < cmap.a_kind.size(); ++k) {
            if (cmap.a_kind[k] == 0) qs.y[cmap.a_index[k]] -= cmap.a_sign[k] * cr.y[k];
          }
          for (Index k = 0; k < cp.l; ++k) {
            if (cmap.g_kind[k] == 0) qs.y[cmap.g_index[k]] -= cmap.g_sign[k] * cr.z[k];
          }
          no_duals = false;
        } else if (cr.status == ConicStatus::Optimal) {
          // Residuals relative to large data norms can hide an absolute violation (QPLIB_10034 at reduced accuracy:
          // the objective 1.2e-3 off): the barrier decides.
          log.log(1, "QP: the conic point violates the problem by %.2e: the barrier instead\n", sp->max_violation(cr.x));
          done = false;
          no_duals = false;
        } else if (cr.status == ConicStatus::TimeLimit) {
          qs.status = QpStatus::TimeLimit;
        } else {
          done = false;
          no_duals = false;
        }
      }
      if (!done) {
        QpOptions qo;
        qo.time_limit = params.time_limit - timer.elapsed();
        qo.threads = std::max(1, threads);
        qo.log = &log;
        qs = solve_qp(*sp, sp->col_lb, sp->col_ub, qo, shift);
      }
    }
    r.stats.barrier_iterations = qs.iterations;
    r.stats.lp_iterations = qs.iterations;
    switch (qs.status) {
      case QpStatus::Optimal: r.status = Status::Optimal; break;
      case QpStatus::Infeasible: r.status = Status::Infeasible; break;
      case QpStatus::Unbounded: r.status = Status::Unbounded; break;
      case QpStatus::InfeasibleOrUnbounded: r.status = Status::InfeasibleOrUnbounded; break;
      case QpStatus::TimeLimit: r.status = Status::TimeLimit; break;
      case QpStatus::NumericalError: r.status = Status::NumericalError; break;
    }
    if (qs.status == QpStatus::Optimal) {
      r.has_solution = true;
      r.x = postsolve(qs.x);
      r.objective = model.objective_value(r.x);
      r.bound = r.objective;
      if (qs.approximate) log.log(1, "QP: solved only to about 1e-6 (numerical difficulties)\n");
      // Duals in the model's sense (p minimizes sense * f).
      if (!no_duals) {
        if (sp == &p) {
          r.row_dual = qs.y;
          r.reduced_cost = qp_reduced_costs(p, r.x, r.row_dual);
        } else {
          pre.postsolve_duals(r.x, qs.y, r.row_dual, r.reduced_cost);
        }
        r.has_duals = true;
        for (double& v : r.row_dual) v *= sense;
        for (double& v : r.reduced_cost) v *= sense;
      }
    } else if (qs.status == QpStatus::Infeasible) {
      r.bound = sense * kInf;
    }
    r.stats.root_bound = r.bound;
    r.stats.time = timer.elapsed();
    return r;
  }
  MiqpOptions mo;
  mo.threads = std::max(1, threads);
  mo.diag_shift = shift;
  if (tracer) mo.on_trace = tracer->hook();
  if (!start.empty()) mo.start = sp == &p ? start : pre.crush(start);
  MiqpResult mr;
  if (sp->num_cols() == 0) {
    mr.status = Status::Optimal;
    mr.has_solution = true;
    mr.bound = mr.root_bound = sp->offset;
  } else {
    mr = miqp_solve(*sp, params, log, timer, mo);
  }
  if (mr.status == Status::NumericalError && !mr.has_solution && !t_in_race && !model.has_logical() &&
      timer.elapsed() < params.time_limit) {
    // The MIQP search's QP relaxations failed (a nearly singular Q: QPLIB_10040 / 10057 / 10060, convex up to the
    // 1e-6 perturbation, NUMERIC in 0.1 s; by the outer approximation OPTIMAL in 0.1-0.4 s): the outer approximation
    // in the MILP search instead, its result if it has one.
    log.log(1, "MIQP: numerical failure, the outer approximation in the MILP search instead\n");
    SolveResult ro = solve_oa(model, params, start, timer, log, tracer);
    if (ro.has_solution || ro.status == Status::Optimal || ro.status == Status::Infeasible) return ro;
  }
  r.status = mr.status;
  r.has_solution = mr.has_solution;
  if (mr.has_solution) {
    r.x = postsolve(mr.x);
    r.objective = model.objective_value(r.x);
  }
  r.bound = sense * mr.bound;
  r.stats.nodes = mr.nodes;
  r.stats.barrier_iterations = mr.barrier_iterations;
  r.stats.lp_iterations = mr.barrier_iterations + mr.lp_iterations;
  r.stats.root_bound = sense * mr.root_bound;
  r.stats.time = timer.elapsed();
  return r;
}

}  // namespace

SolveResult solve(const Model& model, const SolverParams& params) { return solve(model, params, {}); }

namespace {

SolveResult solve_model(const Model& model, const SolverParams& params, const std::vector<double>& start,
                        IncumbentReporter* reporter, IncumbentTracer* tracer) {
  Timer timer;
  Logger log(params.verbosity);
  // Quadratic constraints: outer approximation in the MILP search (solve_oa). A model these routes find not convex
  // goes on to spatial branch-and-bound (solve_spatial).
  if ((model.has_quadratic_rows() || model.has_quadratic()) && !model.has_logical()) {
    const char* sp = std::getenv("MILP_SPATIAL");
    const bool spatial = !(sp && *sp == '0');
    t_spatial_next = spatial;
    SolveResult r = model.has_quadratic_rows() ? solve_oa(model, params, start, timer, log, tracer)
                                               : solve_quadratic(model, params, start, timer, log, tracer);
    t_spatial_next = false;
    if (r.status != Status::NonConvex || !spatial) return r;
    std::string why;
    if (auto rs = solve_spatial(model, params, start, timer, log, tracer, why)) return *rs;
    log.log(1, "ERROR: the model is not convex, and spatial branch-and-bound does not apply: %s\n", why.c_str());
    return r;
  }
  if (model.has_quadratic_rows()) return solve_oa(model, params, start, timer, log, tracer);
  if (model.has_quadratic()) {
    if (!model.has_logical()) return solve_quadratic(model, params, start, timer, log, tracer);
    const LogicalReformulation R = reformulate_logical(model, params.logic_big_m);
    if (!R.implications.empty()) {
      throw std::runtime_error("logical constraints without finite big-M bounds are not supported yet with a "
                               "quadratic objective");
    }
    log.log(1, "Logical constraints: %d semi-continuous, %zu SOS, %zu indicator -> %d auxiliary columns, %d rows\n",
            model.num_semi(), model.sos.size(), model.indicators.size(), R.aux_cols, R.aux_rows);
    SolveResult r = solve_quadratic(R.model, params, start.empty() ? start : R.extend(start), timer, log, tracer);
    if (r.has_solution) {
      r.x.resize(static_cast<size_t>(model.num_cols()));
      snap_logical(model, r.x);
      r.objective = model.objective_value(r.x);
    }
    return r;
  }
  if (!model.has_logical()) return solve_plain(model, {}, params, start, timer, log, reporter, tracer, true);
  // Semi-continuous columns, SOS sets, indicator constraints: solved as a plain MILP with auxiliary
  // binaries (reformulate.hpp). MILP_LOGIC_BIGM overrides params.logic_big_m (diagnostics).
  const char* env_big_m = std::getenv("MILP_LOGIC_BIGM");
  const LogicalReformulation R = reformulate_logical(model, env_big_m ? std::atof(env_big_m) : params.logic_big_m);
  log.log(1, "Logical constraints: %d semi-continuous, %zu SOS, %zu indicator -> %d auxiliary columns, %d rows (%d with big-M), "
          "%zu implications\n", model.num_semi(), model.sos.size(), model.indicators.size(), R.aux_cols, R.aux_rows,
          R.big_m, R.implications.size());
  const std::vector<double> st = start.empty() ? start : R.extend(start);
  SolveResult r = solve_plain(R.model, R.implications, params, st, timer, log, reporter, tracer, false);
  if (r.has_solution) {
    r.x.resize(static_cast<size_t>(model.num_cols()));
    snap_logical(model, r.x);
    r.objective = model.objective_value(r.x);
    const double viol = model.violation(r.x).max();
    if (!(viol <= 1e-5)) log.log(1, "WARNING: solution violates the logical constraints by %.3g\n", viol);
  }
  r.stats.time = timer.elapsed();
  return r;
}

}  // namespace

SolveResult solve(const Model& model, const SolverParams& params_in, const std::vector<double>& start) {
  // The interrupt flags of this solve: the caller's (never written here) and a local one, set when
  // the incumbent callback asks to stop (so the caller's flag stays clear for its next solve).
  std::atomic<bool> local_flag{false};
  SolverParams params = params_in;
  params.interrupt = &local_flag;
  params.on_incumbent = nullptr;  // reported through the reporter only (not copied into sub-solves)
  InterruptScope scope(&local_flag, params_in.interrupt);
  SolverParams cb_params = params;
  cb_params.on_incumbent = params_in.on_incumbent;
  std::optional<IncumbentReporter> reporter;
  if (params_in.on_incumbent && !model.has_quadratic()) reporter.emplace(model, cb_params);
  std::optional<IncumbentTracer> tracer;
  if (params_in.trace_incumbents) tracer.emplace(model.sense);
  SolveResult r = solve_model(model, params, start, reporter ? &*reporter : nullptr, tracer ? &*tracer : nullptr);
  if (tracer) {
    if (r.has_solution) tracer->record(r.objective, "final", r.stats.time);
    r.incumbent_trace = tracer->take();
  }
  // The final solution (postsolved and polished) if it improves on the last one reported.
  if (reporter && r.has_solution) reporter->report(r.x, model.sense == ObjSense::Maximize ? -r.bound : r.bound);
  if (local_flag.load() || (params_in.interrupt && params_in.interrupt->load())) {
    switch (r.status) {
      case Status::TimeLimit:
      case Status::IterationLimit:
      case Status::Unknown:
      case Status::NumericalError:
        r.status = Status::Interrupted;
        break;
      default:
        break;
    }
  }
  return r;
}

}  // namespace milp
