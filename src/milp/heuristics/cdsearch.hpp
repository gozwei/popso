#pragma once
// Conflict search: a depth-first search over the integer variables without the LP that learns from its dead
// ends, for models where no other heuristic finds a point (pure feasibility structure: peg-solitaire-a3).
//
// Each decision fixes one bound of an integer variable and is propagated over the rows (DomainPropagator) and the
// learned clauses (ConflictPool, watched literals). When propagation fails, the failure is analysed on the bound
// trail (analyze_propagation_conflict: the first unique implication point of the last decision level), the clause
// is stored, and the search jumps back to the level where the clause implies a bound. Variables are picked by
// activity (bumped when they occur in a learned clause, decayed otherwise) and take the value they had last
// (phase saving); the search restarts on a Luby sequence, keeping clauses, activities and phases.
//
// Scope: linear models whose columns are all integer with finite bounds. Learned clauses are used only inside the
// search; a point is returned only after MipProblem::max_violation accepts it. Deterministic for a given seed and
// work budget.

#include <atomic>
#include <vector>

#include "milp/search/mip_problem.hpp"

namespace milp {

struct CdsLimits {
  long long effort = 1000000000;  // work units (nonzeros and literals touched)
  double time_limit = kInf;       // seconds (safety cap; makes the result timing dependent if it binds)
  unsigned seed = 0;
  const std::atomic<bool>* stop = nullptr;
  const std::vector<double>* hint = nullptr;  // initial values (e.g. the closest point of a local search)
  size_t max_clauses = 50000;                 // learned clauses kept (the least recently used go first)
  Index max_lits = 2000;                      // longer analysed clauses are replaced by the decision clause
  int restart_unit = 100;                     // conflicts per Luby unit
  Index long_lits = 50000;                    // longer clauses up to this length still direct the jump back
  bool minimize = false;                      // clause minimization (also MILP_CDS_MIN=1)
  bool check = false;                         // tests: verify the bounds after every backtrack to the root
};

struct CdsStats {
  long long effort = 0, decisions = 0, conflicts = 0, restarts = 0, learned = 0, lits = 0, units = 0;
  long long decision_clauses = 0;  // conflicts without an analysed clause (the decisions are the clause)
  long long not_asserting = 0;     // clauses with several literals at their highest level
  long long long_clauses = 0;      // clauses too long to keep, used for the jump back only
  long long dropped = 0;           // literals removed by clause minimization
  int max_level = 0;
  bool applicable = false;  // the model is in scope
  bool infeasible = false;  // the search proved that no point exists (not used as a proof by the solver)
  bool check_failed = false;
};

// Returns true with a feasible point in `sol` (max violation <= 1e-6), false if none was found within the limits
// or the model is out of scope.
bool conflict_search(const MipProblem& p, const CdsLimits& limits, std::vector<double>& sol, CdsStats* stats = nullptr);

}  // namespace milp
