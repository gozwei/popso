#pragma once
// User starting solutions (MIP starts).
//
// A start has one value per model column; NaN means "not given". A complete
// start that is feasible is used as it is. Otherwise the given integer values
// are rounded, clamped to their bounds and fixed, and the remaining columns
// are completed by a short sub-MIP (an LP if every integer is fixed). If that
// fails the start is dropped. Continuous values of a partial start are not
// fixed (they rarely fit the completion).

#include <string>
#include <vector>

#include "milp/core/timer.hpp"
#include "milp/search/mip_problem.hpp"
#include "milp/solver/solver.hpp"

namespace milp {

struct StartOutcome {
  std::vector<double> x;  // feasible point of p (empty: no usable start)
  std::string message;    // what happened, for the log
};

StartOutcome prepare_start(const MipProblem& p, const std::vector<double>& start, const SolverParams& params,
                           const Timer& timer);

}  // namespace milp
