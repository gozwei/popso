#pragma once
// Crossover from an interior-point solution to a basic solution, and the barrier driver that
// combines both with the simplex.
//
// crossover() works in the simplex's variable space: n structurals and m logicals
// s_i = -a_i x (bounds [-row_ub, -row_lb]), so [A I] (x, s) = 0. It
//  1. ranks all variables by how far they are from their bounds relative to their reduced cost
//     (interior variables with zero reduced costs are basic in every optimal basis near the
//     point) and factorizes the m best as a starting basis, replacing dependent columns by
//     logicals;
//  2. primal push: every nonbasic variable strictly between its bounds (a superbasic) is moved
//     to a bound, preferring the one its reduced cost points to; a basic variable that reaches
//     a bound first leaves the basis and the superbasic enters (a primal simplex pivot without
//     pricing). Primal feasibility is kept up to the interior point's own violation.
// The basis returned is then primal (nearly) feasible; the simplex cleanup in barrier_solve
// starts with primal simplex pivots on the remaining dual infeasibilities (LpOptions::
// primal_first) and finishes with the dual simplex, so the result is an optimal basis.

#include <atomic>
#include <vector>

#include "milp/core/log.hpp"
#include "milp/core/sparse.hpp"
#include "milp/lp/barrier.hpp"
#include "milp/lp/simplex.hpp"

namespace milp {

struct CrossoverStats {
  Index superbasics = 0;     // nonbasic variables strictly between their bounds at the start
  long long pushes = 0;      // primal push steps
  long long pivots = 0;      // pushes that changed the basis
  Index replaced = 0;        // dependent columns of the initial basis replaced by logicals
  Index dual_candidates = 0;  // basic variables at a bound with a nonzero reduced cost
  long long dual_pivots = 0;  // dual push steps that changed the basis
  Index primal_infeasibilities = 0;  // basic variables outside their bounds at the end
  double max_primal_infeasibility = 0.0;
  double time = 0.0;
};

// x: column values, y: row duals (d = c - A^T y). Returns a basis for DualSimplex::set_basis.
// stop: abandons the push once set (the basis returned is then valid but not pushed).
Basis crossover(const SparseMatrix& A, const std::vector<double>& cost, const std::vector<double>& col_lb,
                const std::vector<double>& col_ub, const std::vector<double>& row_lb, const std::vector<double>& row_ub,
                const std::vector<double>& x, const std::vector<double>& y, CrossoverStats* stats = nullptr,
                const std::atomic<bool>* stop = nullptr);

struct BarrierOptions {
  IpmOptions ipm;
  bool crossover = true;
  double time_limit = kInf;  // whole solve, seconds
  const Logger* log = nullptr;
  const std::atomic<bool>* stop = nullptr;  // abandons the solve (status TimeLimit) once set
};

struct BarrierStats {
  IpmResult ipm;
  CrossoverStats crossover;
  long long cleanup_iterations = 0;
  double total_time = 0.0;
  bool used = false;  // the basis of `lp` comes from the barrier (false: the caller should solve normally)
};

// Solves the LP loaded in `lp` (with the same data passed here) by barrier, crossover and
// simplex cleanup. On success lp holds an optimal basis and Optimal is returned. If the
// interior-point method does not reach an optimal point (infeasible, unbounded, numerical
// trouble, limits) `lp` is left untouched, stats->used is false and the IPM's outcome is
// returned as the closest LpStatus; the caller then solves with the simplex, which also gives
// the definite infeasibility / unboundedness verdict.
LpStatus barrier_solve(DualSimplex& lp, const SparseMatrix& A, const std::vector<double>& cost,
                       const std::vector<double>& col_lb, const std::vector<double>& col_ub,
                       const std::vector<double>& row_lb, const std::vector<double>& row_ub,
                       const BarrierOptions& opt, BarrierStats* stats = nullptr);

}  // namespace milp
