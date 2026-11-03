#pragma once
// Parallel LP-based branch-and-bound.

#include <atomic>
#include <mutex>
#include <vector>

#include "milp/core/log.hpp"
#include "milp/core/timer.hpp"
#include "milp/search/mip_problem.hpp"
#include "milp/solver/solver.hpp"

namespace milp {

struct MipResult {
  Status status = Status::Unknown;
  bool has_solution = false;
  double objective = kInf;  // internal (minimization) objective incl. offset
  double bound = -kInf;     // internal lower bound incl. offset
  std::vector<double> x;
  long long nodes = 0;
  long long lp_iterations = 0;
  long long mem_drains = 0;  // drain phases under the memory limit (SearchOptions::mem_limit_bytes)
  double root_lp_work = 0.0;  // root LP iterations (incl. the cut loop) x (rows + columns)
  double root_bound = -kInf;
  bool root_lns_ran = false, root_lns_improved = false;  // root LNS rounds ran / found a better incumbent
  // Root restart request: the problem with these column bounds and the extra
  // rows (root cuts, CSR) appended should be presolved and solved again.
  bool restart = false;
  bool tree_restart = false;  // the request comes from the tree (not the root)
  std::vector<double> restart_lb, restart_ub;
  std::vector<Index> cut_start{0}, cut_index;
  std::vector<double> cut_value, cut_lb, cut_ub;
  std::vector<double> debug_point;  // the debug point of this search (after canonicalization), for a restart
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
  std::atomic<bool>* stop_flag = nullptr;  // racing: stop when set (status Interrupted)
  SharedIncumbent* shared = nullptr;       // racing: exchange incumbents with other searches
  bool feasibility_jump = true;            // run FJ on a side thread (racing: first search only)
  // Known point (e.g. the incumbent before a restart), tried as the first incumbent. If it is
  // infeasible here (reductions may cut it off), its objective becomes the cutoff instead.
  std::vector<double> start;
  bool no_zerohalf = false;  // root cut loop without zero-half cuts
  bool root_lns = true;  // root LNS rounds allowed (restart: only if they improved before it)
  std::atomic<long long>* progress = nullptr;  // LP iterations of this search are added here as it goes
  long long iteration_limit = -1;  // stop (status IterationLimit) after this many LP iterations (-1: none)
  // Enforce the memory limit on the resident size of the whole process too. Off in sub-MIPs:
  // the process includes the main search, so a sub-MIP's small limit was always exceeded and
  // every sub-MIP of a large problem stopped after 2 nodes (the main search checks RSS).
  bool check_resident = true;
  long long mem_limit_bytes = -1;  // tests: memory cap in bytes instead of SolverParams::mem_limit_mb
  // Known optimal point in this problem's columns (debugging; MILP_DEBUG_SOL_ORIG crushed through presolve).
  std::vector<double> debug_point;
};

MipResult branch_and_bound(const MipProblem& p, const SolverParams& params, const Logger& log, const Timer& timer,
                           const SearchOptions& opts = {});

}  // namespace milp
