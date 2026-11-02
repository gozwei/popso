#pragma once
// Public entry point of the solver.

#include <string>
#include <vector>

#include "milp/core/model.hpp"
#include "milp/core/types.hpp"

namespace milp {

enum class Status {
  Unknown,
  Optimal,
  Infeasible,
  Unbounded,
  InfeasibleOrUnbounded,
  TimeLimit,
  NodeLimit,
  MemoryLimit,
  IterationLimit,
  Interrupted,
  NumericalError,
};

const char* to_string(Status s);

struct SolverParams {
  double time_limit = kInf;   // seconds
  int threads = 0;            // 0 = hardware concurrency
  int verbosity = 1;          // 0 silent, 1 summary, 2 progress, 3 debug
  double mip_rel_gap = 1e-4;  // relative gap termination
  double mip_abs_gap = 1e-10;
  long long node_limit = -1;  // -1 = unlimited
  long long mem_limit_mb = 4096;  // approximate cap on search-tree memory
  unsigned seed = 0;
  int racing = 1;             // concurrent differently-seeded searches sharing incumbents (opportunistic mode)
  bool deterministic = false;  // reproducible parallel search (same seed + threads -> same result); slower
  bool presolve = true;
  Tolerances tol;
};

struct SolveStats {
  double time = 0.0;
  long long nodes = 0;
  long long lp_iterations = 0;
  double root_bound = -kInf;
};

struct SolveResult {
  Status status = Status::Unknown;
  bool has_solution = false;
  double objective = kInf;   // in the model's original sense
  double bound = -kInf;      // best dual bound, original sense
  std::vector<double> x;     // primal solution (if has_solution)
  SolveStats stats;
};

SolveResult solve(const Model& model, const SolverParams& params);

// Solves with a starting solution: one value per model column, NaN = not given
// (see start.hpp). An empty vector means no start.
SolveResult solve(const Model& model, const SolverParams& params, const std::vector<double>& start);

}  // namespace milp
