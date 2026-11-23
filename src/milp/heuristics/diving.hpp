#pragma once
// LP diving heuristics: repeatedly round one fractional integer variable
// (by tightening its bound) and resolve the LP until the solution is
// integral, the LP becomes infeasible, or the budget runs out.

#include <atomic>
#include <vector>

#include "milp/heuristics/rounding.hpp"
#include "milp/lp/simplex.hpp"
#include "milp/search/mip_problem.hpp"
#include "milp/search/propagation.hpp"

namespace milp {

enum class DiveRule { Fractional, Coefficient, Guided, VectorLength };
const char* to_string(DiveRule r);

struct DiveLimits {
  long long iteration_budget = 5000;  // total simplex iterations for the dive
  double time_limit = kInf;           // seconds
  double cutoff = kInf;               // abandon once the LP objective reaches this
  const std::atomic<bool>* stop = nullptr;  // optional: the dive ends at its next step once set
};

// How a dive ended (optional output of dive()).
struct DiveOutcome {
  int steps = 0;           // roundings whose LP was solved
  bool limit_hit = false;  // ended by its iteration / time limit (not by a point, infeasibility or the cutoff)
};

// `lp` must hold an optimal LP solution of the starting node. On return the
// LP bounds are restored, but its basis is left wherever the dive ended
// (callers re-install the node basis). `incumbent` may be null (Guided then
// behaves like Fractional). With `prop`, each rounding is propagated and the
// implied integer bounds are applied to the LP too. Returns true with a
// feasible point in `sol`.
bool dive(const MipProblem& p, DualSimplex& lp, DiveRule rule, const SimpleRounding& rounding,
          const std::vector<double>* incumbent, const DiveLimits& limits, std::vector<double>& sol,
          long long* iterations, DomainPropagator* prop = nullptr, DiveOutcome* outcome = nullptr);

// Tree dives of one worker (branch-and-bound): dives, new incumbents, dives that finished (not ended by
// their limit) and their LP iterations, and the budget the next dive waits for after a limit hit.
struct TreeDiveHistory {
  long long calls = 0, wins = 0, done = 0, done_iters = 0, need = 0;
  void record(long long iters, bool win, bool limit_hit);
};

// LP iterations the next tree dive may use: up to a share of the search's LP iterations `iters` (plus
// 1000) minus those of all dives so far (`dived`, root dives included). The share is 10%, or with
// `adapt` 5% x (1 + 10 (wins + 1) / (calls + 1)) (SCIP's diving rule) capped at `adapt_max`, and with
// `adapt` a dive starts only when the budget covers a whole dive (the mean of the finished dives, and
// after a limit hit twice that dive's iterations). <= 0: no dive now: the budget is used up, or (`wait`
// set) it does not cover a whole dive yet.
long long tree_dive_budget(const TreeDiveHistory& h, long long iters, long long dived, bool adapt, double adapt_max,
                           bool* wait = nullptr);

}  // namespace milp
