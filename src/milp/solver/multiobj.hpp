#pragma once
// Multiple objectives (Model::objectives): lexicographic optimization by priority, objectives of one priority
// blended by their weights.
//
// The objectives are grouped by priority, highest first. Each group is one stage: an ordinary solve of the model
// with the objective sum of weight * objective over the group. After a stage, a row keeps its blended value: no
// worse than the value found by more than max(abs_tol, rel_tol * |value|), with the largest abs_tol and rel_tol
// of the group. The row is on the model as given, so every stage is presolved anew; the solution of a stage is
// the start of the next.
//
// Limits: the time limit covers the whole sequence, the node limit each stage. A stage that stops at a limit
// with a solution ends the sequence: that solution satisfies the rows of the earlier stages, the status is the
// limit's. Deterministic mode is deterministic stage by stage.

#include <vector>

#include "milp/core/model.hpp"
#include "milp/solver/solver.hpp"

namespace milp {

// One stage: the objectives (0 = the model's own, k >= 1 = Model::objectives[k - 1]) of one priority.
struct ObjectiveStage {
  int priority = 0;
  std::vector<int> members;
  double abs_tol = 0.0, rel_tol = 0.0;  // the largest of the members
};

// The stages in the order solved (descending priority; members in objective order).
std::vector<ObjectiveStage> objective_stages(const Model& model);

// model.objectives must not be empty. Called by solve().
SolveResult solve_multi_objective(const Model& model, const SolverParams& params, const std::vector<double>& start);

}  // namespace milp
