#pragma once
// Bounded dual simplex (with a primal simplex for cleanup) for
//
//   min c^T x   s.t.  row_lb <= A x <= row_ub,  col_lb <= x <= col_ub.
//
// Internally every row i gets a logical variable s_i = -a_i x with bounds
// [-row_ub_i, -row_lb_i], so the constraints read [A I] (x, s) = 0. Variables
// 0..n-1 are structural, n..n+m-1 logical.
//
// Designed for branch-and-bound: bounds can be changed and the solve warm
// started from the previous (dual feasible) basis.

#include <atomic>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include "milp/core/sparse.hpp"
#include "milp/core/types.hpp"
#include "milp/lp/lu.hpp"

namespace milp {

enum class LpStatus { Optimal, Infeasible, Unbounded, CutoffReached, IterationLimit, TimeLimit, NumericalError };
const char* to_string(LpStatus s);

enum class VarStatus : int8_t { Basic = 0, AtLower = 1, AtUpper = 2, AtZero = 3 };

struct Basis {
  std::vector<VarStatus> status;  // size n + m
  bool empty() const { return status.empty(); }
};

struct LpOptions {
  long long iteration_limit = -1;  // -1 = unlimited
  double time_limit = kInf;        // seconds for this solve
  double objective_cutoff = kInf;  // stop once the dual bound exceeds this
  bool perturb = true;
  Tolerances tol;
  int refactor_interval = 100;
  int verbosity = 0;
  // Crossover cleanup: if the starting basis is primal feasible but not dual feasible, run primal
  // simplex pivots first (instead of the dual phase 1, which would give up primal feasibility).
  bool primal_first = false;
  // Stops the solve (status TimeLimit) once set, e.g. when a concurrent method has finished.
  const std::atomic<bool>* stop = nullptr;
  // Called with the solve's iterations so far after every checkpoint_every iterations (points that depend on
  // the iteration count only); false stops the solve (status TimeLimit). Deterministic mode takes the
  // feasibility-jump helper's solutions there during a long root LP.
  std::function<bool(long long)> checkpoint;
  long long checkpoint_every = 0;
};

class DualSimplex {
 public:
  // A is CSC with n columns and m rows.
  void load(const SparseMatrix& A, const std::vector<double>& cost, const std::vector<double>& col_lb,
            const std::vector<double>& col_ub, const std::vector<double>& row_lb, const std::vector<double>& row_ub);

  Index num_cols() const { return n_; }
  Index num_rows() const { return m_; }

  void set_col_bounds(Index j, double lb, double ub);
  void set_row_bounds(Index i, double lb, double ub);
  double col_lower(Index j) const { return lb_[j]; }
  double col_upper(Index j) const { return ub_[j]; }

  void set_basis(const Basis& b);
  Basis get_basis() const;
  void reset_basis();  // slack basis
  void set_seed(unsigned seed);

  // Snapshot of the basis together with its factorization, so a caller can
  // return to a basis without refactorizing (strong branching, dives).
  // Bounds are not part of the state. Only valid while the rows are unchanged.
  // After an optimal solve the primal and dual values are kept too, so a solve
  // after restore_state (and bound changes) starts warm without recomputation.
  struct State {
    std::vector<VarStatus> status;
    std::vector<Index> basic;
    std::vector<double> dse;
    LuFactor lu;
    bool has_factor = false;
    std::vector<double> x, d;
    bool has_values = false;
  };
  void save_state(State& s) const;
  void restore_state(const State& s);

  // Append rows lb <= a x <= ub given in CSR form (start has count+1 entries).
  // Their logicals become basic, so a dual feasible basis stays dual feasible.
  void add_rows(Index count, const std::vector<Index>& start, const std::vector<Index>& index,
                const std::vector<double>& value, const std::vector<double>& lb, const std::vector<double>& ub);
  // Remove rows; rows whose logical is nonbasic are made basic first by
  // dropping the basis (callers normally remove only slack, i.e. basic, rows).
  void remove_rows(const std::vector<Index>& rows);
  // Exchanges the nonbasic logicals of `rows` into the basis (each against the basic variable with
  // the largest pivot element), so that the rows can be removed with the basis kept. The basis may
  // become primal and dual infeasible; the next solve repairs it. Returns how many entered.
  Index pivot_out_rows(const std::vector<Index>& rows);
  bool row_is_basic(Index i) const { return status_[n_ + i] == VarStatus::Basic; }
  VarStatus status(Index j) const { return status_[j]; }

  // Simplex tableau row for the basic variable at the position holding
  // structural/logical `basic_var`: out[j] = (B^{-1} [A I])_{r j} for all
  // n + m variables. Returns false if the variable is not basic.
  bool tableau_row(Index basic_var, std::vector<double>& out);
  // Sparse variant: `out` (size n + m) must be all zero on entry; only the positions listed in
  // `nz` (ascending) are written. The caller zeroes them again before the next call.
  bool tableau_row(Index basic_var, std::vector<double>& out, std::vector<Index>& nz);
  // Work of the last sparse tableau_row: the B^{-1} row's nonzeros, the row matrix entries passed over and the
  // entries of the index list (all n + m when the row was scanned densely).
  long long tableau_row_work() const { return tableau_work_; }
  const std::vector<Index>& basic_vars() const { return basic_; }
  // Rows are scaled internally (row i by rscale_[i], a power of two); every
  // accessor below reports unscaled data.
  const SparseMatrix& row_matrix() const { return At_orig_; }  // CSR of all rows (incl. added)
  const SparseMatrix& col_matrix() const { return A_orig_; }   // CSC
  // Strict safe bounds (default off): a dual infeasibility inside the tolerance on a column or row whose range is
  // infinite on that side makes the safe bound -inf (unless it is at rounding level), instead of being charged at
  // the variable's current size. That charge is no proof: a product column near 2e8 with a reduced cost of -2e-8
  // gained 5e-3 by moving 0.1%, and the node's bound was above a feasible point (spatial branch-and-bound).
  void set_safe_strict(bool on) { safe_strict_ = on; }
  double var_lower(Index j) const { return j < n_ ? lb_[j] : lb_[j] / rscale_[j - n_]; }
  double var_upper(Index j) const { return j < n_ ? ub_[j] : ub_[j] / rscale_[j - n_]; }
  double var_value(Index j) const { return j < n_ ? x_[j] : x_[j] / rscale_[j - n_]; }
  // Reduced cost of variable j after an optimal solve (true costs, no perturbation).
  double reduced_cost(Index j) const { return j < n_ ? d_[j] : d_[j] * rscale_[j - n_]; }

  LpStatus solve(const LpOptions& opt);

  // Results of the last solve.
  double objective() const;  // c^T x (without any offset)
  std::vector<double> col_values() const;
  std::vector<double> row_activities() const;
  std::vector<double> row_duals() const;     // y, with d = c - A^T y
  void row_duals(std::vector<double>& y) const;
  std::vector<double> reduced_costs() const;  // d for structurals
  // Safe lower bound on the LP's optimum for any row multipliers y (Neumaier and Shcherbina, "Safe
  // bounds in linear and mixed-integer linear programming", Math. Program. 99 (2004)): the Lagrangian
  //   sum_i y_i b_i + sum_j min_{l_j <= x_j <= u_j} r_j x_j,   r = c - A^T y,
  // on the unscaled data (rows incl. added ones, current column bounds), minus an a posteriori bound
  // on its rounding errors, so it needs no directed rounding. y_i > 0 takes the row's lower side,
  // y_i < 0 the upper one; a y_i whose side is infinite is dropped (y_i = 0). A column whose needed
  // bound is infinite (or whose r_j has an uncertain sign next to an infinite bound) makes the bound
  // -inf unless |r_j| <= tau; such a r_j, and a dropped-side |y_i| <= tau (kept in r), count as zero
  // with |r_j| max(1, |x_j|) (|y_i| max(1, |a_i x|)) added to the error term: the bound is safe up to
  // dual infeasibilities of tau on unbounded variables, the LP's own dual tolerance.
  double safe_bound(const std::vector<double>& y, double tau) const;
  // After an optimal solve: objective(), or safe_bound(row_duals(), tau) if that is lower by more than
  // 1e-6 relative (*below set; a dual infeasibility hidden by row scaling or the tolerances makes the
  // objective an invalid bound). MILP_SAFE_BOUND=0: always objective().
  double safe_objective(double tau, bool* below = nullptr) const;
  static bool safe_bound_enabled();
  // After a solve that returned Infeasible: row multipliers y (size num_rows)
  // proving it, i.e. y^T A x has no value in the range implied by the row
  // bounds over the column box (the sign of y is not normalized). Returns
  // false if the last solve did not end with such a proof.
  bool farkas_proof(std::vector<double>& y) const;
  // safe_bound(y, tau) with zero costs: a lower bound on 0 over the LP's feasible set, so > 0 proves the LP
  // infeasible (Neumaier and Shcherbina's certificate for a ray y: the rows' sides y_i b_i minus the largest
  // (A^T y)^T x over the column box, with the same rounding error terms).
  double farkas_bound(const std::vector<double>& y, double tau) const;
  // Infeasible is only returned with a ray that passes farkas_bound > 0 on the unscaled data (rows incl. added
  // ones, current column bounds; tau 1e-9 relative to the largest |y_i|). A ray that fails is retried once from a
  // fresh factorization, then the solve returns NumericalError ("unverified infeasibility proof"), which callers
  // treat as an unsolved LP (no pruning). MILP_FARKAS_CHECK=0: rays are not verified.
  static bool farkas_check_enabled();
  // Process-wide counts: rays checked, failed (NumericalError) and passed after the retry; solves that went on
  // because only the unscaled dual feasibility test (kUnscaledDual) found dual infeasibilities, and their
  // iterations from that point on (what the scaled test would have saved).
  struct CheckCounts {
    long long farkas_checked = 0, farkas_failed = 0, farkas_retried_ok = 0;
    long long unscaled_solves = 0, unscaled_iterations = 0;
  };
  static CheckCounts check_counts();
  // Values of all n + m variables (structurals first, logicals unscaled).
  const double* x() const;
  long long iterations() const { return total_iterations_; }
  // Why the last solve returned NumericalError ("" otherwise).
  const char* fail_reason() const { return fail_reason_; }

 private:
  LpStatus run(const LpOptions& opt);
  LpStatus dual_loop(const LpOptions& opt, bool phase1);
  LpStatus primal_loop(const LpOptions& opt);
  // Records the Farkas proof: rho_ with the leaving variable below its lower bound (sign 1) or above its upper
  // bound (-1).
  LpStatus infeasible(bool phase1, double sign);
  LpStatus verify_infeasible(const LpOptions& opt);  // farkas_bound check of an Infeasible result, one retry
  bool farkas_verified() const;
  // safe_bound / farkas_bound: the Lagrangian with the LP's costs or with zero costs.
  double lagrangian_safe(const std::vector<double>& y, double tau, bool with_cost) const;
  LpStatus fail(const char* why);    // NumericalError with fail_reason_ set
  const char* fail_reason_ = "";
  long long dbg_shifts_ = 0, dbg_degen_ = 0, dbg_flips_ = 0;  // verbosity 3 trace
  bool limits_hit(const LpOptions& opt, LpStatus* st) const;
  double lagrangian_bound();

  // renumber: let basis positions follow the factorization's pivot rows (saves a permutation in
  // every solve); callers that hold positions across the call pass false.
  bool refactor(bool renumber = true);
  void compute_row_scale(Index i, const Index* idx, const double* val, Index len);
  double tol(Index j) const { return ftol_[j]; }  // relative primal tolerance factor of variable j
  void compute_primal();
  bool compute_duals(bool check = false);  // check: basic reduced costs vanish (fresh-factor level)
  bool primal_residuals_small();
  void note_move(Index j, double old_x);
  void rebuild_infeasible(double ptol);
  void update_infeasible(Index p, double ptol);  // position p's membership in infeas_list_
  void apply_moves();
  void place_nonbasic(Index j);
  void place_by_dual(Index j);
  // unscaled: the tolerance times dftol_ (the tests at the end of a solve), else in scaled terms.
  Index count_dual_infeasibilities(bool flip_boxed, bool unscaled);
  long long unscaled_from_ = -1;  // iteration at which only the unscaled test found dual infeasibilities (-1: none)
  void perturb_costs(double mul);
  void boost_perturbation(double extra);
  void perturb_rows(double b, bool add);  // nonbasic logicals by b (1 + rand) (MILP_LP_PERTURB_ROWS)
  long long boost_at_ = -1;  // cold solves: iteration at which the perturbation grows (-1: never)
  double boost_mul_ = 0.0;
  // MILP_LP_COLD_STALL / MILP_LP_WARM_STALL: iterations without objective progress that bring the boost
  // forward (-1: off), and where the current flat stretch started.
  long long stall_window_ = -1, stall_ref_it_ = 0;
  double stall_ref_obj_ = 0.0;
  uint64_t pert_seed_ = 0;
  std::vector<double> pert_col_;  // MILP_LP_PERTURB_COLSCALE: per-column magnitude factor (empty: 1)
  void compute_pert_scale();  // from the costs and bounds at load (row and bound changes keep it)
  void column(Index j, HVector& v) const;  // a_j scattered into v (row space)
  void unit_btran(Index r);                // rho_ = row r of B^{-1}
  // arow_ = rho^T [A I] (all variables, or only nonbasic ones if nonbasic_only).
  void pivot_row(const HVector& rho, bool nonbasic_only = false);
  void replace_basic(Index r, Index q, Index leaving_status_hint);

  Index n_ = 0, m_ = 0;
  SparseMatrix A_, At_;  // CSC and CSR (as transpose)
  std::vector<double> cost_, lb_, ub_;     // true data, size n + m
  std::vector<double> wcost_, wlb_, wub_;  // working data (perturbed / phase 1)
  std::vector<VarStatus> status_;
  std::vector<Index> basic_;  // position -> variable
  std::vector<double> x_, d_, dse_;
  int dse_exact_ = 0;      // recompute all DSE weights at the next solve (set_basis, MILP_LP_DSE_KEEP=2)
  std::vector<Index> dse_new_;  // basis positions of added rows whose DSE weight is computed at the next solve
  // Warm start: after an optimal solve x_ and d_ (true costs) stay valid while
  // only bounds change; nonbasic value changes are queued in moves_ and applied
  // to x_B with one ftran at the next solve.
  bool values_ok_ = false;
  std::vector<std::pair<Index, double>> moves_;
  std::vector<double> perturb_rand_;  // per-column cost perturbation factors
  double pert_scale_ = 1.0;           // cost scale of the perturbation (see perturb_costs)
  // Dual pricing candidates: basis positions whose value violates its bounds.
  std::vector<Index> infeas_list_, infeas_pos_;  // infeas_pos_[p]: index in the list or -1
  bool infeas_dirty_ = true;
  std::vector<double> arow_;
  // arow_ entry j is part of the current row iff arow_mark_[j] == arow_gen_ (no clearing pass);
  // other entries of arow_ hold stale values and are never read.
  std::vector<uint32_t> arow_mark_;
  uint32_t arow_gen_ = 1;
  std::vector<Index> arow_nz_;  // nonzero positions of arow_: the first arow_count_
  Index arow_count_ = 0;
  LuFactor lu_;
  SparseMatrix At_orig_, A_orig_;  // unscaled CSR / CSC of all rows (as passed in)
  std::vector<double> rscale_;    // per row
  std::vector<double> ftol_;      // per variable: primal tolerance factor (min(1, rscale) for logicals)
  std::vector<double> dftol_;     // per variable: dual tolerance factor (min(1, 1 / rscale) for logicals)
  mutable std::vector<double> xout_;
  bool safe_strict_ = false;
  mutable std::vector<double> sb_y_, sb_r_, sb_a_;  // safe_bound scratch
  mutable std::vector<int> sb_cnt_;
  bool has_factor_ = false;
  long long total_iterations_ = 0;
  std::vector<double> farkas_;  // row of B^{-1} proving infeasibility of the last solve
  double farkas_sign_ = 1.0;    // see infeasible()
  bool has_farkas_ = false;
  long long iter_start_ = 0;
  mutable long long next_checkpoint_ = 0;  // LpOptions::checkpoint: iterations of the solve at the next call
  double solve_start_ = 0.0;
  std::mt19937 rng_{12345};
  // Running averages of the result densities of the solves (hyper-sparse switch in the LU).
  double dens_rho_ = 0.0, dens_col_ = 0.0, dens_tau_ = 0.0, dens_flip_ = 0.0;
  long long tableau_work_ = 0;  // tableau_row_work()
  void note_density(double& avg, const HVector& v) const {
    avg = 0.95 * avg + 0.05 * (m_ > 0 ? static_cast<double>(v.count) / m_ : 0.0);
  }
  std::vector<Index> renum_old_, renum_new_;
  // Dual ratio test: per variable the direction it may move from its bound (+1 at lower, -1 at
  // upper, 2 free, 0 basic or fixed), built when a dual loop starts iterating and kept up to date by
  // its pivots and flips. One byte per variable instead of status and both bounds.
  std::vector<int8_t> move_;
  void build_move();
  int8_t move_of(Index j) const {
    const VarStatus st = status_[j];
    if (st == VarStatus::Basic || wlb_[j] == wub_[j]) return 0;
    return st == VarStatus::AtLower ? 1 : st == VarStatus::AtUpper ? -1 : 2;
  }
  std::vector<double> renum_dse_;
  // scratch
  HVector rho_, alpha_, tau_, flipcol_;
  struct Candidate {
    Index j;
    double ratio, abs_alpha;
  };
  std::vector<Candidate> cands_;
  std::vector<Candidate> cand_heap_;  // BFRT: candidates beyond the first group
  std::vector<Index> fac_start_, fac_index_;  // refactor(): basis matrix
  std::vector<double> fac_value_;
  std::vector<std::pair<Index, Index>> fac_replaced_;
};

}  // namespace milp
