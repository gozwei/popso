#pragma once
// Sparse LU factorization of a simplex basis matrix B (m x m) with
// Markowitz pivoting and Forrest-Tomlin updates (product-form etas as fallback).
//
// Index spaces: B's columns are "basis positions" 0..m-1, its rows are the
// constraint rows 0..m-1.
//   ftran: solves B y = a   (a indexed by rows -> y indexed by positions)
//   btran: solves B^T z = e (e indexed by positions -> z indexed by rows)
//
// Factorization: a triangular phase pivots column and row singletons with
// count queues (no numerical updates needed), then the remaining kernel is
// eliminated with Markowitz pivoting (threshold relative to the row maximum)
// on compact row-wise storage with a column-wise index.
// MILP_LU_STATS=1 prints per-factorization statistics to stderr.

#include <atomic>
#include <utility>
#include <vector>

#include "milp/core/sparse.hpp"
#include "milp/core/types.hpp"

namespace milp {

class LuFactor {
 public:
  // Factorize the matrix whose column p is index/value[start[p]..start[p+1]).
  // Returns the number of (numerically) dependent columns. For each such
  // column, `replaced` receives (position, row): the column at `position`
  // was replaced by the unit column e_row so that the factorization is
  // nonsingular; the caller must update its basis accordingly.
  Index factorize(Index m, const std::vector<Index>& start, const std::vector<Index>& index,
                  const std::vector<double>& value, std::vector<std::pair<Index, Index>>* replaced = nullptr);
  // With interruptible set, factorize() gives up (aborted() true, the factors unusable) once the solve is interrupted
  // or the thread's deadline passes (core/interrupt.hpp), checked every 64 pivots: a large basis can take minutes
  // (crossover of a 49800-row root LP ran 3 minutes past the time limit). Only for callers that stop then.
  bool interruptible = false;
  // factorize() also gives up once this is set (the concurrent root LP's flag: the thread that waits for the
  // crossover has no other way to end a factorization in progress).
  const std::atomic<bool>* stop = nullptr;
  bool aborted() const { return aborted_; }

  // In-place solves on dense vectors of size m.
  void ftran(std::vector<double>& v) const;
  void btran(std::vector<double>& v) const;
  // Same on an HVector; on return v.nz lists the nonzeros of the result (|v| > 1e-14; smaller values
  // are zeroed). The input index list must be valid. `density` is the caller's prediction of the
  // result's density (fraction of m, e.g. a running average of earlier results; < 0: unknown): when
  // both the right-hand side and the prediction are sparse, the solve works hyper-sparsely (a
  // symbolic depth-first search over the factors finds the entries to touch, so its cost follows the
  // nonzeros instead of m), otherwise it runs the dense loops.
  // spike = false: the result is not kept as the Forrest-Tomlin spike (a solve between the entering
  // column's ftran and the update).
  void ftran(HVector& v, double density = -1.0, bool spike = true) const;
  void btran(HVector& v, double density = -1.0) const;
  // Several ftrans; the dense ones share one pass over the factors.
  void ftran(HVector& a, HVector& b, double da = -1.0, double db = -1.0) const;
  void ftran(HVector& a, HVector& b, HVector& c, double da = -1.0, double db = -1.0, double dc = -1.0) const;

  // Basis positions in the order of the factorization's pivot rows: after this call, position p of
  // the factorized matrix is the column pivoted in row p, which saves the permutation of every
  // ftran/btran result. old_pos[p] receives the previous position of the column now at p; the
  // caller must permute its own position-indexed data the same way.
  void renumber_positions(std::vector<Index>& old_pos);

  // 0: automatic hyper-sparse switch, 1: dense loops only, 2: hyper-sparse whenever possible (tests).
  int hyper_mode = 0;

  // Replace the column at basis position p; `alpha` must be ftran(a_new)
  // computed with the current factorization, and that ftran (of a_new as the
  // first argument) must be the last ftran call: Forrest-Tomlin uses its
  // partially transformed column.
  void update(Index p, const std::vector<double>& alpha);
  void update(Index p, const HVector& alpha);

  Index num_updates() const { return static_cast<Index>(eta_pos_.size() + ft_row_.size()); }
  Index eta_nnz() const { return static_cast<Index>(eta_index_.size() + ft_col_index_.size() + r_index_.size()); }
  Index factor_nnz() const {
    return static_cast<Index>(l_index_.size() + u_row_index_.size() + dense_lu_.size());
  }
  // Refactoring pays off once the extra solve work caused by the updates
  // (accumulated since the last factorization) exceeds the factorization work.
  bool wants_refactor() const { return eta_work_ > factor_work_ || ft_unstable_; }

  bool forrest_tomlin = true;  // false: product-form updates only

  double pivot_threshold = 0.1;   // Markowitz threshold (relative to the row maximum)
  double pivot_tolerance = 1e-11; // absolute: smaller pivots are singular

 private:
  // Internally every vector lives in row space: the value of basis position p is kept at row
  // row_of_pos_[p] (the pivot row of the step owning p; Forrest-Tomlin keeps it), so the triangular
  // solves run in place. U's row-wise copy, the etas and the Forrest-Tomlin data are row-indexed.
  void finish(Index num_replaced, Index kernel_dim, Index kernel_nnz, Index dense_dim, Index basis_nnz, double t0);
  template <int K>
  void ftran_dense(double* const* v, bool spike) const;
  void ftran_hyper(HVector& v, bool spike) const;
  // btran in row space; count < 0: no index list (dense loops). Returns whether list[0..count) is a
  // valid index list of the result (else the dense loops ran).
  bool btran_rows(double* x, Index* list, Index& count) const;
  bool use_hyper(Index count, double density) const;
  Index hyper_limit() const { return hyper_mode == 2 ? m_ : m_ / 5; }
  void ftran_multi(HVector* const* vs, const double* dens, int k, bool keep_spike = true) const;
  void to_positions(HVector& v) const;    // row space -> position space (no-op if identity)
  void from_positions(HVector& v) const;  // position space -> row space
  // Symbolic reach for the hyper-sparse solves: see lu.cpp.
  bool reach(const Index* gmap, const Index* start, const Index* index, Index* list, Index& count, Index limit) const;
  bool ft_update(Index p, double alpha_p);
  bool step_in_dense(Index s) const { return dense_n_ > 0 && s >= dense_begin_ && s < dense_begin_ + dense_n_; }
  bool row_in_dense(Index r) const { return step_in_dense(orig_step_of_row_[r]); }

  Index m_ = 0;
  // Pivot sequence (by step).
  std::vector<Index> piv_row_, piv_col_;
  std::vector<double> piv_val_;
  std::vector<Index> row_of_pos_, pos_of_row_;
  bool identity_ = false;  // row_of_pos_[p] == p
  // L etas: step l_step_[g] eliminates multiples of its pivot row from rows
  // l_index_ with multipliers l_value_ in [l_start_[g], l_start_[g+1]).
  std::vector<Index> l_step_, l_start_, l_index_;
  std::vector<double> l_value_;
  std::vector<Index> lgrp_of_row_;  // row -> its L group (pivot row of l_step_[g]) or -1
  // The same entries grouped by eliminated row (for L^T solves): row
  // lr_row_[g] contributes to the pivot rows lr_index_ in [lr_start_[g], lr_start_[g+1]).
  std::vector<Index> lr_row_, lr_start_, lr_index_;
  std::vector<double> lr_value_;
  std::vector<Index> lrgrp_of_row_;  // row -> lr group or -1
  // U (off-diagonal), row-wise by step: rows (of the steps owning the columns) + values.
  std::vector<Index> u_row_start_, u_row_index_;
  std::vector<double> u_row_value_;
  // U column-wise by step: rows (pivot rows of earlier steps) + values.
  std::vector<Index> u_col_start_, u_col_index_;
  std::vector<double> u_col_value_;
  // row -> its step in the fresh factorization while that step is live and sparse (not in the dense
  // block, not replaced by a Forrest-Tomlin step), else -1. Node map of the hyper-sparse U solves.
  std::vector<Index> ustep_of_row_;
  // Sparse steps with a nonempty U column / U row, and those without but with a pivot other than 1
  // (ascending; the dense loops skip all other steps).
  std::vector<Index> ucol_steps_, ucol_div_steps_, urow_steps_, urow_div_steps_;
  // Dense trailing block: steps [dense_begin_, dense_begin_ + dense_n_) factored
  // densely; column-major n x n with L multipliers below and U above the
  // diagonal (pivots in piv_val_). Their U rows hold no sparse entries; their U
  // columns hold only entries in rows of earlier (sparse) steps.
  Index dense_begin_ = 0, dense_n_ = 0;
  std::vector<double> dense_lu_;
  // Forrest-Tomlin updates. Replacing position p kills its step and appends a
  // new last step (same row, position p, pivot in piv_val_) whose U column
  // is kept column-wise only (rows of earlier steps); killed entries of the
  // original U are zeroed in place. Row etas R_k (ftran: v[r] -= sum mu_i v[i]
  // after L) eliminate the old U row of the killed step.
  std::vector<char> step_dead_;          // by step (original m, then appended)
  std::vector<Index> step_of_pos_;       // live step owning each position
  std::vector<Index> orig_step_of_row_;  // step of each row in the fresh factorization
  std::vector<Index> orig_step_of_pos_;
  std::vector<Index> ft_row_, ft_col_start_, ft_col_len_, ft_col_index_;
  std::vector<double> ft_col_value_;
  std::vector<Index> r_row_, r_start_{0}, r_index_;
  std::vector<double> r_value_;
  // Last ftran's first rhs after L and R (row space): its nonzeros are spike_idx_ unless
  // spike_all_ (then any row may be nonzero).
  mutable std::vector<double> spike_;
  mutable std::vector<Index> spike_idx_;
  mutable bool spike_all_ = true;
  mutable bool spike_ok_ = false;
  bool ft_off_ = false;  // a product-form eta exists: no more FT updates
  bool ft_unstable_ = false;
  // Product-form etas (applied after all FT updates), row-indexed.
  std::vector<Index> eta_pos_, eta_start_{0}, eta_index_;
  std::vector<double> eta_pivot_, eta_value_;
  double factor_work_ = 0.0, eta_work_ = 0.0;
  bool aborted_ = false;
};

}  // namespace milp
