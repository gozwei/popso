#pragma once
// Large-neighbourhood heuristics that solve a restricted copy of the problem
// with a small branch-and-bound (no recursive sub-MIPs).

#include <atomic>
#include <cstdint>
#include <vector>

#include "milp/lp/simplex.hpp"
#include "milp/search/mip_problem.hpp"

namespace milp {

// Outcome of the sub-MIPs solved by one heuristic call (accumulated over retries).
struct SubMipStats {
  int solves = 0;              // sub-MIPs actually solved (0: neighbourhood rejected)
  bool exhausted = false;      // last sub-MIP finished (optimal or infeasible): neighbourhood too small
  long long lp_iterations = 0, nodes = 0;
  long long lp_work = 0;  // LP work of the sub-MIPs and of lp_rens's dives (DualSimplex::work() units, ~1 ns)
  long long dive_iterations = 0, fp_iterations = 0;  // of lp_iterations: the sub-MIPs' dives and pumps
  int dive_wins = 0, fp_runs = 0, fp_wins = 0;       // their new incumbents in the sub-MIPs; pumps run
  Index fixed = 0, ints = 0;  // integer variables fixed / total in the last sub-MIP
};

struct SubMipLimits {
  long long node_limit = 500;
  double time_limit = 5.0;  // seconds
  double min_fixed_fraction = 0.3;  // skip if fewer integer variables are fixed
  SubMipStats* stats = nullptr;     // optional output
  std::atomic<bool>* stop = nullptr;           // optional: abort when set (result: whatever was found)
  std::atomic<long long>* progress = nullptr;  // optional: LP iterations so far are added here
  long long iteration_limit = -1;  // LP iterations per sub-MIP solve (-1: none); deterministic, unlike time
  // LP work per call (DualSimplex::work() units, ~1 ns each; -1: none): the deterministic counterpart of
  // time_limit (lp_rens: shared by its dives and sub-MIPs).
  long long work_limit = -1;
  int mem_limit_mb = 256;          // search memory cap of the sub-MIP
  int solution_limit = 0;          // stop after this many incumbents of the sub-MIP (0: none)
  // Fixing neighbourhoods searched before: the sub-MIP's integer bounds are hashed into *fingerprint, and
  // one whose hash is in *skip is not solved (no solve counted). Only for neighbourhoods of p itself with
  // a cutoff that does not grow (an exhausted one holds no better point later either).
  std::uint64_t* fingerprint = nullptr;
  const std::vector<std::uint64_t>* skip = nullptr;
};

// Solves p restricted to column bounds [lb, ub]. Returns true if a solution
// with internal objective (incl. offset) below `cutoff` was found.
bool solve_sub_mip(const MipProblem& p, const std::vector<double>& lb, const std::vector<double>& ub, double cutoff,
                   const SubMipLimits& limits, std::vector<double>& sol);

// RENS: fix integers that are integral in the LP solution, restrict the rest to floor/ceil.
// With `redcost` given, an infeasible sub-MIP is retried fixing fewer variables
// (keeping those with the largest |reduced cost| fixed).
bool rens(const MipProblem& p, const double* lp_x, double cutoff, const SubMipLimits& limits,
          std::vector<double>& sol, const std::vector<double>* redcost = nullptr);

// Root-LP sub-MIP (RENS with LP-checked, relaxed fixing; as HiGHS' RENS): starting from the solved
// LP `root` (its columns = p's), fix integer columns at their integral LP values in batches, most
// certain first (largest |reduced cost|, then the longest columns), propagate, and re-solve the LP
// after each batch; a batch that makes the LP infeasible (or reach the cutoff) is undone and split,
// a single such column is left free. When no integral column is left, the least fractional one is
// rounded (a dive). Stops at `target` of the free integer columns fixed, then solves the sub-MIP on
// the remaining domain. Retries (up to `attempts` sub-MIPs, sharing the limits): fewer fixings when
// the sub-MIP was infeasible (exhausted), more when it hit its limits without a solution.
struct LpRensStats {
  int attempts = 0;           // sub-MIPs solved
  long long dive_iterations = 0;  // LP iterations of the fixing dives
  double last_rate = 0.0;     // fixing rate of the last sub-MIP
  int presolve_rejects = 0;   // fixing batches undone because presolve proved them infeasible
  bool stopped_blind = false;  // stop_blind ended the retries (below)
};
// `stop_blind`: no further attempt after a first sub-MIP that was exhausted (infeasible, or nothing below
// the cutoff): the dive's LP accepted fixings that are infeasible together, so it gives no signal and
// other rates are blind guesses (enlight_hard: first sub-MIP infeasible after 1-5 nodes, then 3 more
// down to rate 0.07 while the root waited 1 s; where the sub-MIP finds the first incumbent -- binkar10_1,
// cbs-cta, piperout-08, fiball -- the first attempt succeeds).
bool lp_rens(const MipProblem& p, const DualSimplex& root, double cutoff, const SubMipLimits& limits,
             std::vector<double>& sol, double target = 0.7, int attempts = 3, LpRensStats* out = nullptr,
             bool stop_blind = false);

// RINS: fix integers on which the LP solution and the incumbent agree. If that fixes less
// than `target_fixed` of the integers, more are fixed at their incumbent values, those
// whose LP value is closest to the incumbent first. If it fixes more than `max_fixed` of them
// (the agreement alone leaves too small a neighbourhood: nexp-150-20-8-5 95%, every sub-MIP solved
// in presolve), a random subset (`seed`) of the agreeing ones is left free down to that rate.
bool rins(const MipProblem& p, const double* lp_x, const std::vector<double>& incumbent, double cutoff,
          const SubMipLimits& limits, std::vector<double>& sol, double target_fixed = 0.0, double max_fixed = 1.0,
          unsigned seed = 0);

// Repair: fix the integers of an infeasible point x except those in its violated
// rows and in rows sharing variables with them (at most about half of the integers stay free).
bool repair(const MipProblem& p, const std::vector<double>& x, double cutoff, const SubMipLimits& limits,
            std::vector<double>& sol);

// Mutation: fix a random `fix_fraction` of the integer variables (that are not
// fixed already) to their incumbent values.
bool mutation(const MipProblem& p, const std::vector<double>& incumbent, double fix_fraction, unsigned seed,
              double cutoff, const SubMipLimits& limits, std::vector<double>& sol);

// Graph-induced mutation (GINS, SCIP 4.0; the generic fixing order of adaptive LNS, Hendel, MPC 14
// (2022)): leave free the 1 - `fix_fraction` of the (not fixed) integer variables closest to a random
// integer variable (`seed`) in the variable-row graph (breadth first, rows in index order; rows with
// more than `max_row` entries are not followed, a new random start is taken when the reachable part
// is used up), fix the others at their incumbent values.
bool graph_mutation(const MipProblem& p, const std::vector<double>& incumbent, double fix_fraction, unsigned seed,
                    double cutoff, const SubMipLimits& limits, std::vector<double>& sol, Index max_row = 0);

// Crossover: fix the integer variables on which all the given feasible points agree (their
// common value); the others keep their bounds. Needs at least two points.
bool crossover(const MipProblem& p, const std::vector<const std::vector<double>*>& points, double cutoff,
               const SubMipLimits& limits, std::vector<double>& sol);

}  // namespace milp
