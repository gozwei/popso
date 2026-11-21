#pragma once
// Activity-based domain propagation on the rows of a MIP: tightens integer
// variable bounds implied by the current bounds of the other variables and
// detects infeasible nodes without solving the LP.

#include <cstdint>
#include <memory>
#include <vector>

#include "milp/search/mip_problem.hpp"
#include "milp/search/trail.hpp"

namespace milp {

class DomainPropagator {
 public:
  // rows: shared_rows(p), to share one row copy among the propagators of a problem (a worker each); else an own copy.
  explicit DomainPropagator(const MipProblem& p, std::shared_ptr<const SparseMatrix> rows = nullptr);
  // The rows of p (CSR) with the objective row appended when a cost is nonzero: what a propagator holds after
  // enable_objective. Read-only, so one copy serves any number of propagators (12 bytes per nonzero each before).
  static std::shared_ptr<const SparseMatrix> shared_rows(const MipProblem& p);

  // Propagates starting from the rows of the `seeds` columns. Bounds are
  // tightened in place; every tightened column is appended to `tightened`.
  // Returns false if some row is proven infeasible.
  bool propagate(std::vector<double>& lb, std::vector<double>& ub, const std::vector<Index>& seeds,
                 std::vector<Index>& tightened);

  long long work_limit = 0;  // max nonzeros touched per call (0 = 20 * nnz + 1000)
  long long last_work = 0;   // nonzeros touched by the last call
  bool continuous = false;   // also tighten continuous columns (by at least 1e-3 of their domain width)
  // With `continuous`: rows that failed or tightened an integer column using a continuous bound
  // tightened earlier in the same call (did the continuous propagation pay off?).
  long long last_cont_useful = 0;
  // With enable_objective: whether this call uses the objective row (it is dense on many models), and
  // whether it failed or tightened a bound (set by propagate, reset by the caller).
  bool objective = true;
  bool last_obj_useful = false;
  BoundTrail* trail = nullptr;  // if set: records each tightening with its row, and the reason of a failure

  // Rows as propagated: the model rows, plus the objective row (index num_rows(), see
  // enable_objective) when enabled. Conflict analysis explains bounds with these.
  const SparseMatrix& rows() const { return *rows_; }
  const std::vector<double>& row_lb() const { return row_lb_; }
  const std::vector<double>& row_ub() const { return row_ub_; }

  // Objective cutoff propagation: appends the row  c x <= cutoff  (c = the problem's costs, no offset)
  // with cutoff +inf (inactive) until set_objective_cutoff. Improving points satisfy it, so bounds it
  // implies hold for every point the search still has to find (like LP cutoff pruning).
  void enable_objective();
  void set_objective_cutoff(double cutoff);

 private:
  const MipProblem& p_;
  std::shared_ptr<const SparseMatrix> rows_;  // CSR copy of A (+ the objective row), possibly shared
  std::vector<double> row_lb_, row_ub_;
  Index obj_row_ = -1;
  std::vector<char> queued_;
  std::vector<char> cont_mark_;     // continuous column tightened in the current call
  std::vector<Index> cont_marked_;
  std::vector<Index> queue_;
  // Rows found idle in the current call and the room they have left: see propagate().
  std::vector<uint32_t> room_stamp_;
  std::vector<double> room_hi_, room_lo_;
  uint32_t room_call_ = 0;
};

}  // namespace milp
