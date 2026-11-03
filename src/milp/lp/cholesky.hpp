#pragma once
// Sparse Cholesky factorization P M P^T = L L^T of a symmetric positive (semi)definite matrix,
// or P M P^T = L S L^T with a fixed sign diagonal S for a quasi-definite matrix (the regularized
// augmented system of an interior-point method: every pivot's sign is known in advance, so any
// symmetric ordering is admissible), supernodal and left-looking (Ng and Peyton, "Block sparse
// Cholesky algorithms on advanced uniprocessor computers", SIAM J. Sci. Comput. 14, 1993).
//
// analyze() orders the matrix (approximate minimum degree, see ordering.hpp), postorders the
// elimination tree, computes the column counts and the (relaxed) supernodes and their row
// structures. factorize() then works on dense column-major supernode blocks: each supernode
// gathers the updates of the descendant supernodes that touch its columns (a dense product per
// descendant, scattered through relative indices) and factors its own trapezoid.
//
// Interior-point matrices become (numerically) singular near the optimum. A pivot that is not
// positive enough is replaced by a huge value (the matching component of the solution is then
// ~0, Wright's approach) instead of failing; num_dropped() counts them.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "milp/core/types.hpp"

namespace milp {

class WorkerPool;

class SparseCholesky {
 public:
  // Pattern of the lower triangle (row >= column, the diagonal must be present) of the n x n
  // matrix in CSC form. perm: an ordering to use (perm[k] = the column eliminated k-th); null
  // computes an approximate minimum degree ordering. Returns false if the pattern is invalid.
  // signs: per column +1 or -1, the sign each pivot must have (null: all positive).
  bool analyze(Index n, const std::vector<Index>& start, const std::vector<Index>& index,
               const std::vector<Index>* perm = nullptr, const std::vector<int8_t>* signs = nullptr);

  // Numeric factorization with the values of the pattern given to analyze() (same layout).
  // Returns false only if the values are not finite.
  bool factorize(const std::vector<double>& value);

  // Solves M x = b in place (b in the original order). Components of dropped pivots are 0.
  void solve(std::vector<double>& b) const;
  void solve(double* b) const;

  Index size() const { return n_; }
  long long factor_nnz() const { return factor_nnz_; }  // nonzeros of L (incl. diagonal)
  double flops() const { return flops_; }              // multiply-adds of one factorization
  Index num_dropped() const { return dropped_; }
  Index num_supernodes() const { return static_cast<Index>(sn_first_.size()) - 1; }
  const std::vector<Index>& perm() const { return perm_; }

  // A pivot d of column j is dropped if d <= pivot_tol * (the column's diagonal entry of M).
  double pivot_tol = 1e-14;
  // Relaxed supernode amalgamation: merge a child into its parent if the merged supernode has at
  // most relax_small columns, or at most relax_cols columns and at most relax_zeros of its
  // stored entries are explicit zeros.
  Index relax_small = 8;
  Index relax_cols = 64;
  double relax_zeros = 0.1;
  // Threads for the dense block products of large supernodes (1: sequential).
  int threads = 1;
  // analyze() gives up (returns false, too_large() true) once the factor would have more
  // nonzeros than this (<= 0: no limit).
  long long nnz_limit = 0;
  bool too_large() const { return too_large_; }
  // analyze() and factorize() give up (return false) once this is set.
  const std::atomic<bool>* stop = nullptr;

  SparseCholesky();
  ~SparseCholesky();
  SparseCholesky(SparseCholesky&&) noexcept;
  SparseCholesky& operator=(SparseCholesky&&) noexcept;

 private:
  void factor_supernode(Index J, std::vector<Index>& relmap, std::vector<double>& work,
                        std::vector<Index>& link_head, std::vector<Index>& link_next, std::vector<Index>& kpos);
  // T = lower part of X X^T for X = rows [0, nr) x columns [0, kw) (leading dimension ld), only
  // the first nc columns of the product; split over the pool when large.
  // S: signs of the kw columns (null: all +1), T = X S X^T.
  void syrk_rows(const double* X, size_t ld, Index kw, Index nc, Index nr, double* T, size_t ldT, const double* S);
  std::unique_ptr<WorkerPool> pool_;

  Index n_ = 0;
  std::vector<Index> perm_, pinv_;  // perm_[k] = original column eliminated k-th; pinv_ its inverse
  // Supernode s: columns [sn_first_[s], sn_first_[s+1]); rows sn_rows_[sn_rstart_[s] .. sn_rstart_[s+1])
  // (its own columns first, then the off-diagonal rows, ascending); values column-major in
  // lx_[sn_xstart_[s] ..) with leading dimension = its number of rows.
  std::vector<Index> sn_first_, sn_rstart_, sn_rows_, sn_of_col_;
  std::vector<size_t> sn_xstart_;
  std::vector<double> lx_;
  std::vector<size_t> entry_pos_;  // input entry k -> position in lx_
  std::vector<Index> diag_entry_;  // per permuted column: input entry of its diagonal
  std::vector<double> diag_;       // per permuted column: the diagonal of M (pivot test)
  std::vector<double> psign_;      // per permuted column: pivot sign (+1 / -1); empty: all +1
  long long factor_nnz_ = 0;
  double flops_ = 0.0;
  Index dropped_ = 0;
  Index nnz_in_ = 0;
  bool too_large_ = false;
};

}  // namespace milp
