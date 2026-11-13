#pragma once
// Parallel LP-based branch-and-bound.

#include <atomic>
#include <functional>
#include <mutex>
#include <vector>

#include "milp/core/log.hpp"
#include "milp/core/timer.hpp"
#include "milp/search/mip_problem.hpp"
#include "milp/solver/solver.hpp"

namespace milp {

// Pseudocosts handed from a search to its restart (solver.cpp): per (column, direction) entry, key
// 2 j + (dir > 0), the sum of the objective gains per unit and their count, and the same for the
// inference history (bounds propagated after the branching). In the columns of the problem the
// search solved.
struct PseudocostSnapshot {
  std::vector<double> sum, inf_sum;
  std::vector<int> cnt, inf_cnt;
  bool empty() const { return cnt.empty(); }
  // In the columns of a presolved problem: column k is column col_map[k] here (Presolver::col_map; a
  // kept column has the same values, a merged parallel column keeps the surviving column's entry).
  PseudocostSnapshot mapped(const std::vector<Index>& col_map) const {
    PseudocostSnapshot out;
    if (empty()) return out;
    const size_t n = col_map.size();
    out.sum.assign(2 * n, 0.0), out.inf_sum.assign(2 * n, 0.0);
    out.cnt.assign(2 * n, 0), out.inf_cnt.assign(2 * n, 0);
    for (size_t k = 0; k < n; ++k) {
      const size_t j = static_cast<size_t>(col_map[k]);
      if (2 * j + 1 >= cnt.size() || 2 * j + 1 >= sum.size() || 2 * j + 1 >= inf_cnt.size() || 2 * j + 1 >= inf_sum.size())
        continue;
      for (size_t d = 0; d < 2; ++d) {
        out.sum[2 * k + d] = sum[2 * j + d], out.cnt[2 * k + d] = cnt[2 * j + d];
        out.inf_sum[2 * k + d] = inf_sum[2 * j + d], out.inf_cnt[2 * k + d] = inf_cnt[2 * j + d];
      }
    }
    return out;
  }
  // Counts capped at `cap` (the reliability threshold), the averages kept: a carried entry counts as
  // reliable at most, and the new search's own observations soon outweigh it.
  PseudocostSnapshot capped(int cap) const {
    PseudocostSnapshot out = *this;
    auto cap_entries = [cap](std::vector<double>& s, std::vector<int>& c) {
      for (size_t k = 0; k < c.size() && k < s.size(); ++k) {
        if (c[k] <= cap) continue;
        s[k] *= static_cast<double>(cap) / static_cast<double>(c[k]);
        c[k] = cap;
      }
    };
    cap_entries(out.sum, out.cnt);
    cap_entries(out.inf_sum, out.inf_cnt);
    return out;
  }
};

// Default of MILP_RESTART_PC (pseudocosts carried across restarts): 0 off, 1 as history (counts capped at
// MILP_RESTART_PC_CAP, default the reliability threshold), 2 as estimates for columns without observations
// of the restarted search.
inline constexpr int kRestartPcDefault = 0;

struct MipResult {
  Status status = Status::Unknown;
  bool has_solution = false;
  double objective = kInf;  // internal (minimization) objective incl. offset
  double bound = -kInf;     // internal lower bound incl. offset
  std::vector<double> x;
  long long nodes = 0;
  long long lp_iterations = 0;
  // LP work (DualSimplex::work() units, ~1 ns each) of the root and the tree workers' LPs; opportunistic
  // searches only (the sub-MIPs' budgets, SearchOptions::work_limit).
  long long lp_work = 0;
  // Of lp_iterations: the dives (root and tree) and the root feasibility pump, and the new incumbents
  // they found (statistics of the sub-MIPs, SubMipStats).
  long long dive_iterations = 0, fp_iterations = 0;
  int dive_wins = 0, fp_runs = 0, fp_wins = 0;
  long long mem_drains = 0;  // drain phases under the memory limit (SearchOptions::mem_limit_bytes)
  long long mem_drain_exits = 0;  // drain phases left again (the memory fell below the exit threshold)
  double mem_drain_rise = 0.0;    // lower-bound rise over the finished drain phases (internal objective)
  long long mem_purged = 0;       // queued nodes purged at an improved cutoff (memory pressure)
  double root_lp_work = 0.0;  // root LP iterations (incl. the cut loop) x (rows + columns)
  double root_time = 0.0;  // seconds to the end of the root node from the restart before it (else the search start)
  double root_bound = -kInf;
  bool root_lns_ran = false, root_lns_improved = false;  // root LNS rounds ran / found a better incumbent
  bool improved = false;  // the search found a better incumbent than its start (SearchOptions::start)
  int restarts = 0, tree_restarts = 0;  // restarts done in this solve (solver.cpp)
  // Root restart request: the problem with these column bounds and the extra
  // rows (root cuts, CSR) appended should be presolved and solved again.
  bool restart = false;
  bool tree_restart = false;  // the request comes from the tree (not the root)
  bool est_restart = false;   // ... from its size estimate (MILP_EST_RESTART)
  std::vector<double> restart_lb, restart_ub;
  std::vector<Index> cut_start{0}, cut_index;
  std::vector<double> cut_value, cut_lb, cut_ub;
  std::vector<double> debug_point;  // the debug point of this search (after canonicalization), for a restart
  PseudocostSnapshot pc;  // with a restart request: the pseudocosts (MILP_RESTART_PC=0: none)
  PresolveSummary presolve;  // solver.cpp: the first presolve of the problem solved
};

// Incumbent shared between concurrent (racing) searches.
struct SharedIncumbent {
  std::mutex mu;
  std::atomic<double> value{kInf};  // internal objective incl. offset
  std::vector<double> x;
};

struct SearchOptions {
  double cutoff = kInf;            // only solutions with internal objective (incl. offset) below this matter
  bool sub_mip_heuristics = true;  // RENS/RINS; disabled inside sub-MIPs to avoid recursion
  bool allow_restart = false;      // may return a root restart request (MipResult::restart)
  int restarts_done = 0;           // root restarts before this search (solver.cpp)
  bool allow_tree_restart = false; // may stop the tree for a restart after many global fixings
  int tree_restarts_done = 0;      // in-tree restarts before this search
  bool allow_est_restart = false;  // may stop the tree for a restart when its size estimate is huge (once per solve)
  int est_restarts_done = 0;       // of the in-tree restarts, the estimation-driven ones
  // Root restart rule (solver.cpp sets it per search): at least this share of the integer columns, and
  // at least restart_min_cols of them, newly fixed globally at the root.
  double restart_frac = 0.1;
  Index restart_min_cols = 20;
  // Restart budget, from the second root restart on (restarts_done >= 1): another one only if the
  // restarted roots so far plus this root's cost again stay within the budget: LP work (iterations x
  // (rows + columns), deterministic mode) or seconds (opportunistic mode) left of the first root's
  // (restart_work_left / restart_time_left = budget - restarted roots before this search); and only
  // if the presolve after the last restart removed at least restart_min_removed of the integer
  // columns (last_restart_removed).
  double restart_work_left = kInf, restart_time_left = kInf;
  double last_restart_removed = 1.0, restart_min_removed = 0.0;
  double restart_t0 = -1.0;  // timer when the restart before this search began (its root time counts from there)
  // Cheaper restarted roots: at most this many root cut-loop rounds (-1: no cap), and root dives /
  // RENS / root LNS only if root_heuristics (the feasibility pump and the LP sub-MIP, which run only
  // without an incumbent, stay).
  int root_cut_rounds = -1;
  bool root_heuristics = true;
  bool root_cuts = true;  // false: no root cut loop at all (sub-MIPs, MILP_SUB_CUTS)
  // Pseudocosts carried from the search before a restart (this problem's columns), counts capped.
  PseudocostSnapshot pc_init;
  std::atomic<bool>* stop_flag = nullptr;  // racing: stop when set (status Interrupted)
  SharedIncumbent* shared = nullptr;       // racing: exchange incumbents with other searches
  bool feasibility_jump = true;            // run FJ on a side thread (racing: first search only)
  // Known point (e.g. the incumbent before a restart), tried as the first incumbent. If it is
  // infeasible here (reductions may cut it off), its objective becomes the cutoff instead:
  // start_objective (with the offset) when set, the objective of `start` in this problem otherwise.
  // A caller that crushed the start through a presolve sets start_objective from the point before the
  // crush: a dual reduction may remove the point, and the crushed point's objective is then not the
  // start's (neos-3083819-nubu, deterministic, 2 threads, seed 4 on x86: the second root restart's
  // presolve crushed the incumbent 6315970 to a rejected point of objective 6307737, which became the
  // cutoff below the optimum 6307996, and 6315970 came back as optimal).
  std::vector<double> start;
  double start_objective = kInf;
  bool no_zerohalf = false;  // root cut loop without zero-half cuts
  bool root_lns = true;  // root LNS rounds allowed (restart: only if they improved before it)
  std::atomic<long long>* progress = nullptr;  // LP iterations of this search are added here as it goes
  long long iteration_limit = -1;  // stop (status IterationLimit) after this many LP iterations (-1: none)
  // Stop (status IterationLimit) once the root's and the tree's LPs have done this much work (DualSimplex::work()
  // units, ~1 ns each; -1: none). Checked after each node; deterministic in a single-threaded search (the sub-MIPs).
  long long work_limit = -1;
  int solution_limit = 0;          // stop (status Interrupted) after this many new incumbents (0: none)
  // Enforce the memory limit on the resident size of the whole process too. Off in sub-MIPs:
  // the process includes the main search, so a sub-MIP's small limit was always exceeded and
  // every sub-MIP of a large problem stopped after 2 nodes (the main search checks RSS).
  bool check_resident = true;
  long long mem_limit_bytes = -1;  // tests: memory cap in bytes instead of SolverParams::mem_limit_mb
  // Known optimal point in this problem's columns (debugging; MILP_DEBUG_SOL_ORIG crushed through presolve).
  std::vector<double> debug_point;
  // Symmetry-breaking rows were already considered for this problem (solver.cpp symmetry_rows).
  bool sst_done = false;
  // Called after each new incumbent (outside the search's locks, possibly from several threads at
  // once) with the point in this problem's columns and the current lower bound (internal, incl.
  // the offset). Deterministic mode: at the deterministic merge points. Not passed to sub-MIPs.
  std::function<void(const std::vector<double>& x, double bound)> on_incumbent;
  // Called for each incumbent the search accepts (under its incumbent lock, possibly from several
  // racing searches at once) with its objective in the user's sense (display_sense, offset
  // included), what found it and the timer's time: SolverParams::trace_incumbents. Copied with the
  // options into restarts and racing searches; sub-MIPs start from fresh options.
  std::function<void(double objective, const char* source, double time)> on_trace;
};

// Root restart rule (thresholds and budget in SearchOptions, set by solver.cpp's RestartRule): the
// outcome for `newly_fixed` of the `ints` integer columns fixed globally at the root, with the root's
// cost so far (LP work = iterations x (rows + columns) in deterministic mode, seconds otherwise; the
// budget applies from the second restart on). The search decides with it at the end of the root, and
// root strong branching asks it first (MILP_RESTART_SKIP_SB): no probes in a root that restarts anyway.
struct RestartDecision {
  enum Outcome { TooFew, Refused, Restart };
  Outcome outcome = TooFew;
  Index need = 0;                 // newly fixed columns needed
  const char* refused = nullptr;  // Refused: the reason
};
RestartDecision root_restart_decision(const SearchOptions& o, bool deterministic, Index ints, Index newly_fixed,
                                      double root_work, double root_time);

MipResult branch_and_bound(const MipProblem& p, const SolverParams& params, const Logger& log, const Timer& timer,
                           const SearchOptions& opts = {});

}  // namespace milp
