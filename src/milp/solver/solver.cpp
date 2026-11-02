#include "milp/solver/solver.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include "milp/core/log.hpp"
#include "milp/core/timer.hpp"
#include "milp/io/mps_reader.hpp"
#include "milp/io/solution.hpp"
#include "milp/lp/simplex.hpp"
#include "milp/presolve/presolve.hpp"
#include "milp/search/branch_and_bound.hpp"
#include "milp/search/symmetry.hpp"
#include "milp/search/mip_problem.hpp"
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
  }
  return "UNKNOWN";
}

namespace {

// Repairs a point that violates the original rows slightly (postsolve round-off):
// integers fixed at their rounded values, continuous columns re-optimized by LP.
// Returns false if that LP fails or the result is still not feasible.
bool polish_point(const MipProblem& p, const Timer& timer, double time_limit, std::vector<double>& x) {
  std::vector<double> lb = p.col_lb, ub = p.col_ub;
  bool any_cont = false;
  for (Index j = 0; j < p.num_cols(); ++j) {
    if (p.is_int[j]) lb[j] = ub[j] = std::round(x[j]);
    else any_cont = true;
  }
  std::vector<double> y = x;
  if (any_cont) {
    DualSimplex lp;
    lp.load(p.A, p.cost, lb, ub, p.row_lb, p.row_ub);
    LpOptions opt;
    opt.perturb = false;
    opt.time_limit = std::max(1.0, time_limit - timer.elapsed());
    if (lp.solve(opt) != LpStatus::Optimal) return false;
    y.assign(lp.x(), lp.x() + p.num_cols());
  } else {
    for (Index j = 0; j < p.num_cols(); ++j) y[j] = lb[j];
  }
  if (!(p.max_violation(y) <= 1e-6)) return false;
  x = std::move(y);
  return true;
}

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
    pool.emplace_back([&, k] {
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
    });
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

// Solves the problem with presolve; returns false if the caller should solve
// the original problem instead (presolve gave up or postsolve failed).
bool solve_presolved(const MipProblem& p, const SolverParams& params, const Logger& log, const Timer& timer,
                     MipResult* out, const SearchOptions& opts = {}) {
  Presolver pre;
  const PresolveStatus ps = pre.run(p, PresolveOptions::from_env(), log);
  const PresolveStats& st = pre.stats();
  log.log(1, "Presolve: %d rows, %d cols, %d nnz -> %d rows, %d cols, %d nnz (%.2fs)\n", st.rows_before,
          st.cols_before, st.nnz_before, st.rows_after, st.cols_after, st.nnz_after, st.time);
  if (ps == PresolveStatus::Infeasible) {
    log.log(1, "Presolve: problem is infeasible\n");
    out->status = Status::Infeasible;
    out->bound = kInf;
    return true;
  }
  // Unbounded-or-infeasible: let the full search classify it precisely.
  if (ps == PresolveStatus::Unbounded) {
    log.log(1, "Presolve: problem is unbounded or infeasible\n");
    return false;
  }
  const MipProblem& q = pre.reduced();
  MipResult mr;
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
    if (!o.start.empty()) o.start = pre.crush(o.start);
    if (!o.debug_point.empty()) {
      o.debug_point = pre.crush(o.debug_point);
      if (q.max_violation(o.debug_point) > 1e-6) {
        log.log(0, "DEBUG-SOL: the debug point is cut off by presolve (violation %.3g)\n", q.max_violation(o.debug_point));
        o.debug_point.clear();
      }
    }
    mr = search(q, params, log, timer, o);
  }
  if (mr.restart) {
    // Root restart: tightened bounds + root cuts, presolved again; the
    // incumbent found so far is the cutoff and the fallback answer.
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
    // Root and in-tree restarts are counted separately (MILP_MAX_RESTARTS / MILP_MAX_TREE_RESTARTS).
    o2.restarts_done = opts.restarts_done + (mr.tree_restart ? 0 : 1);
    o2.tree_restarts_done = opts.tree_restarts_done + (mr.tree_restart ? 1 : 0);
    {
      static const int max_restarts = [] {
        const char* v = std::getenv("MILP_MAX_RESTARTS");
        return v ? std::atoi(v) : 1;
      }();
      static const int max_tree_restarts = [] {
        const char* v = std::getenv("MILP_MAX_TREE_RESTARTS");
        return v ? std::atoi(v) : 1;
      }();
      o2.allow_restart = o2.restarts_done < max_restarts;
      o2.allow_tree_restart = opts.allow_tree_restart && o2.tree_restarts_done < max_tree_restarts;
    }
    if (mr.has_solution) o2.start = mr.x;
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
    mr2.bound = std::min(mr2.bound, mr2.objective);
    mr = std::move(mr2);
  }
  if (mr.has_solution) {
    mr.x = pre.postsolve(mr.x);
    const double viol = p.max_violation(mr.x);
    std::vector<double> fixed = mr.x;
    // The polished point rounds the integers exactly, so its objective may differ from the
    // presolved one by round-off (n5-3: 8429 vs 8428.999991). Accept it within a relative 1e-6,
    // and at any objective when the search stopped at a limit (a feasible point beats none).
    if (!(viol <= 1e-6) && polish_point(p, timer, params.time_limit, fixed) &&
        (mr.status != Status::Optimal ||
         p.objective(fixed) <= mr.objective + 1e-6 * std::max(1.0, std::abs(mr.objective)))) {
      log.log(1, "Postsolved solution violated the original problem by %.3g; repaired by an LP over the continuous "
              "columns\n", viol);
      mr.x = std::move(fixed);
    } else if (!(viol <= 1e-6)) {
      log.log(1, "WARNING: postsolved solution violates the original problem by %.3g; re-solving without presolve\n",
              viol);
      out->bound = mr.bound;  // still a valid bound for the caller's re-solve
      return false;
    }
    mr.objective = p.objective(mr.x);
    mr.bound = std::min(mr.bound, mr.objective);
  }
  *out = std::move(mr);
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

}  // namespace

SolveResult solve(const Model& model, const SolverParams& params) { return solve(model, params, {}); }

SolveResult solve(const Model& model, const SolverParams& params, const std::vector<double>& start) {
  Timer timer;
  Logger log(params.verbosity);
  SolveResult r;
  MipProblem p;
  if (!MipProblem::from_model(model, &p)) {
    r.status = Status::Infeasible;
    return r;
  }
  const double sense = model.sense == ObjSense::Maximize ? -1.0 : 1.0;
  if (params.presolve && presolve_tool(model, p, log, &r)) return r;
  MipResult mr;
  SearchOptions opts;
  opts.allow_restart = true;
  opts.allow_tree_restart = true;
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
    SearchOptions plain;
    plain.start = start_x;
    mr = branch_and_bound(p, params, log, timer, plain);
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
  return r;
}

}  // namespace milp
