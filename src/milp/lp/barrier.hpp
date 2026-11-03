#pragma once
// Interior-point (barrier) solver for
//
//   min c^T x   s.t.  row_lb <= A x <= row_ub,  col_lb <= x <= col_ub
//
// Mehrotra's predictor-corrector primal-dual method with Gondzio's multiple centrality
// correctors. The problem is brought to the form A' x' = b, l <= x' <= u (fixed columns are
// substituted, free rows dropped, every inequality row gets a slack column) and scaled
// (geometric scaling, powers of two). Lower and upper bounds have their own slacks and duals
// (x - x_l = l, x + x_u = u, x_l, x_u, z_l, z_u >= 0); free variables have none and are
// handled by primal regularization.
//
// Linear algebra: every iteration solves the regularized Newton (KKT) system
//
//   [ -(Q + H + rho I)   A^T     ] [dx]   [r1]
//   [   A              delta I   ] [dy] = [r2]
//
// with H = X_l^-1 Z_l + X_u^-1 Z_u diagonal. For LP (Q = 0) it is reduced to the normal
// equations (A Theta A^T + delta I) dy = r2 + A Theta r1, Theta = (H + rho I)^-1, factored by the
// supernodal sparse Cholesky (cholesky.hpp, AMD ordering). Dense columns of A are left out of the
// factorization (only their diagonal enters it) and handled by preconditioned conjugate
// gradients on the full matrix, which also refines every solve. The KktSolver class in
// barrier.cpp is the only place that knows about the reduction: a convex QP needs Q in the
// (1,1) block, i.e. a quasi-definite augmented-system factorization behind the same interface
// (or normal equations when Q is diagonal).
//
// Infeasibility: the method is an infeasible-start IPM. Diverging iterates (dual or primal
// values growing without bound while the residuals stall) are reported as suspected primal or
// dual infeasibility; callers confirm such outcomes with the simplex method.

#include <atomic>
#include <vector>

#include "milp/core/log.hpp"
#include "milp/core/sparse.hpp"
#include "milp/core/types.hpp"

namespace milp {

enum class IpmStatus { Optimal, PrimalInfeasible, DualInfeasible, IterationLimit, TimeLimit, NumericalError };
const char* to_string(IpmStatus s);

struct IpmOptions {
  double tol = 1e-8;       // relative primal and dual infeasibility and duality gap
  int max_iterations = 300;
  double time_limit = kInf;  // seconds
  int max_correctors = 2;    // Gondzio centrality correctors per iteration
  bool scale = true;
  int threads = 1;  // for the Cholesky factorization of large normal equations
  int kkt = 0;      // Newton system: 0 the cheaper one, 1 normal equations, 2 augmented system
  const Logger* log = nullptr;  // level 1: summary, 2: one line per iteration
  const std::atomic<bool>* stop = nullptr;  // stops the method (status TimeLimit) once set
  // The log shows log_sense * (objective + log_offset) (the user's objective of a presolved problem).
  double log_offset = 0.0, log_sense = 1.0;
};

struct IpmResult {
  IpmStatus status = IpmStatus::NumericalError;
  int iterations = 0;
  double objective = 0.0;  // c^T x of the returned point
  // Of the returned point in the original problem: max violation of rows and bounds, max dual
  // infeasibility (reduced costs of the wrong sign), relative duality gap.
  double primal_infeas = kInf, dual_infeas = kInf, rel_gap = kInf;
  std::vector<double> x;  // per column
  std::vector<double> y;  // row duals, d = c - A^T y
  std::vector<double> d;  // reduced costs per column
  double time = 0.0;
  double analysis_time = 0.0;  // orderings and symbolic factorizations
  double factor_time = 0.0;    // numerical factorizations
  long long factor_nnz = 0;
  Index dense_columns = 0;
  bool augmented = false;  // the augmented system was factored (else the normal equations)
  // Optimal only to about 1e-6 (the method broke down before the requested accuracy; the
  // crossover and simplex cleanup still give an exact basis).
  bool approximate = false;
};

// A is CSC with cost.size() columns and row_lb.size() rows.
IpmResult ipm_solve(const SparseMatrix& A, const std::vector<double>& cost, const std::vector<double>& col_lb,
                    const std::vector<double>& col_ub, const std::vector<double>& row_lb,
                    const std::vector<double>& row_ub, const IpmOptions& opt);

}  // namespace milp
