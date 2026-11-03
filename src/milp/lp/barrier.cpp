#include "milp/lp/barrier.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "milp/core/timer.hpp"
#include "milp/lp/cholesky.hpp"

namespace milp {

const char* to_string(IpmStatus s) {
  switch (s) {
    case IpmStatus::Optimal: return "optimal";
    case IpmStatus::PrimalInfeasible: return "primal infeasible";
    case IpmStatus::DualInfeasible: return "dual infeasible";
    case IpmStatus::IterationLimit: return "iteration limit";
    case IpmStatus::TimeLimit: return "time limit";
    case IpmStatus::NumericalError: return "numerical error";
  }
  return "?";
}

namespace {

double inf_norm(const std::vector<double>& v) {
  double r = 0.0;
  for (double a : v) r = std::max(r, std::abs(a));
  return r;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

// Normal equations (A Theta A^T + delta I) y = r. The sparse columns of A form the Cholesky
// factorized matrix M_s = A_s Theta_s A_s^T + delta I. Dense columns A_d enter through the
// Sherman-Morrison-Woodbury formula with the k x k Schur complement S = Theta_d^-1 + A_d^T M_s^-1 A_d
// (dense LU). Conjugate gradients on the full matrix, preconditioned by that solve, refine every
// solution.
class NormalEquations {
 public:
  // The Woodbury work (sparse triangular solves, memory bound) counted against dense-kernel flops.
  static constexpr double kWoodburyWeight = 8.0;
  // nnz_limit: give up (return false) if the factor would be larger (<= 0: no limit).
  bool analyze(const SparseMatrix& A, const SparseMatrix& At, int threads, long long nnz_limit,
               const std::atomic<bool>* stop, const Logger* log) {
    A_ = &A;
    At_ = &At;
    threads_ = threads;
    chol_.nnz_limit = nnz_limit;
    chol_.stop = stop;
    m_ = A.num_minor;
    n_ = A.num_major;
    dense_.assign(static_cast<size_t>(n_), 0);
    dense_cols_.clear();
    // Dense column candidates: far more nonzeros than average. They are handled apart only if
    // that makes the factorization much cheaper (each fills a dense block of M, but a moderate
    // clique is cheaper inside the factorization than in the Woodbury update).
    std::vector<Index> cand;
    if (n_ > 0 && m_ >= 100) {
      const double avg = static_cast<double>(A.nnz()) / n_;
      const double thr = std::max(40.0, 10.0 * avg);
      std::vector<std::pair<Index, Index>> c;
      for (Index j = 0; j < n_; ++j) {
        const Index len = A.end(j) - A.begin(j);
        if (len > thr) c.emplace_back(-len, j);
      }
      std::sort(c.begin(), c.end());
      // The Woodbury solve keeps M_s^-1 A_d (m x k) in memory.
      const size_t max_k = static_cast<size_t>(std::max<double>(0.0, 2e7 / std::max<Index>(1, m_)));
      if (c.size() > std::min<size_t>(1000, max_k)) c.resize(std::min<size_t>(1000, max_k));
      for (const auto& e : c) cand.push_back(e.second);
    }
    bool split = false;
    if (!cand.empty()) {
      for (Index j : cand) dense_[j] = 1;
      if (!build_pattern() || !chol_.analyze(m_, start_, index_)) return false;
      const double flops_split = chol_.flops();
      double clique = 0.0;
      for (Index j : cand) clique += 0.5 * static_cast<double>(A.end(j) - A.begin(j)) * (A.end(j) - A.begin(j));
      split = true;
      if (clique < 5e7) {
        std::fill(dense_.begin(), dense_.end(), 0);
        if (!build_pattern()) return false;
        SparseCholesky full;
        full.stop = stop;
        full.nnz_limit = nnz_limit > 0 ? std::min(nnz_limit, 4 * chol_.factor_nnz() + 1000000) : 4 * chol_.factor_nnz() + 1000000;
        const bool full_ok = full.analyze(m_, start_, index_);
        if (!full_ok && !full.too_large()) return false;
        full.stop = nullptr;
        // Woodbury per iteration: k solves with the factor (W), the k x k LU and about four
        // applications of W in the solves.
        const double k = static_cast<double>(cand.size());
        const double extra = kWoodburyWeight * (4.0 * k * static_cast<double>(chol_.factor_nnz()) + k * k * k / 3.0 + 16.0 * k * m_);
        split = !full_ok || full.flops() > 2.0 * (flops_split + extra);
        cost_ = split ? flops_split + extra : full.flops();
        if (log) {
          log->log(2, "Barrier: dense columns: factor %.3g flops with, %.3g + %.3g without them\n",
                   full_ok ? full.flops() : kInf, flops_split, extra);
        }
        if (!split) chol_ = std::move(full);
      }
      if (split && clique >= 5e7) {
        const double k = static_cast<double>(cand.size());
        cost_ = flops_split + kWoodburyWeight * (4.0 * k * static_cast<double>(chol_.factor_nnz()) + k * k * k / 3.0 + 16.0 * k * m_);
      }
      if (split) {
        for (Index j : cand) dense_[j] = 1;
        dense_cols_ = cand;
        if (!build_pattern()) return false;
      }
    } else {
      if (!build_pattern() || !chol_.analyze(m_, start_, index_)) return false;
      cost_ = chol_.flops();
    }
    value_.assign(index_.size(), 0.0);
    work_.assign(static_cast<size_t>(m_), 0.0);
    chol_.threads = threads_;
    if (log) {
      log->log(2, "Barrier: normal equations %d rows, %zu nonzeros (lower), factor %lld nonzeros, %.3g flops, %zu dense "
               "columns (%zu candidates)\n", m_, index_.size(), chol_.factor_nnz(), chol_.flops(), dense_cols_.size(),
               cand.size());
    }
    return true;
  }

  // Lower pattern of the sparse part (columns not marked dense), the diagonal first in each column.
  bool build_pattern() {
    const SparseMatrix& A = *A_;
    const SparseMatrix& At = *At_;
    start_.assign(1, 0);
    index_.clear();
    std::vector<Index> mark(static_cast<size_t>(m_), -1);
    for (Index i = 0; i < m_; ++i) {
      index_.push_back(i);
      mark[i] = i;
      for (Index k = At.begin(i); k < At.end(i); ++k) {
        const Index j = At.index[k];
        if (dense_[j]) continue;
        for (Index t = A.begin(j); t < A.end(j); ++t) {
          const Index r = A.index[t];
          if (r > i && mark[r] != i) {
            mark[r] = i;
            index_.push_back(r);
          }
        }
      }
      start_.push_back(static_cast<Index>(index_.size()));
      if (index_.size() > static_cast<size_t>(200000000)) return false;  // far beyond any memory budget
    }
    return true;
  }

  bool factor(const std::vector<double>& theta, double delta) {
    theta_ = &theta;
    delta_ = delta;
    const SparseMatrix& A = *A_;
    const SparseMatrix& At = *At_;
    // Rows covered (almost) only by dense columns make M_s nearly singular and the Woodbury
    // update unstable; a tiny multiple of the dense part's diagonal is added to M_s (the
    // conjugate gradients on the true matrix remove the difference).
    dense_diag_.assign(static_cast<size_t>(m_), 0.0);
    for (Index j : dense_cols_)
      for (Index t = A.begin(j); t < A.end(j); ++t) dense_diag_[A.index[t]] += A.value[t] * A.value[t] * theta[j];
    for (Index i = 0; i < m_; ++i) {
      for (Index k = At.begin(i); k < At.end(i); ++k) {
        const Index j = At.index[k];
        const double aij = At.value[k];
        if (dense_[j]) continue;
        const double f = aij * theta[j];
        // Column j's entries in rows >= i (rows are sorted).
        const Index* b = A.index.data() + A.begin(j);
        const Index* e = A.index.data() + A.end(j);
        const Index* p = std::lower_bound(b, e, i);
        for (const Index* q = p; q < e; ++q) work_[*q] += f * A.value[static_cast<size_t>(q - A.index.data())];
      }
      work_[i] += delta + kDenseDiag * dense_diag_[i];
      for (Index k = start_[i]; k < start_[i + 1]; ++k) {
        value_[k] = work_[index_[k]];
        work_[index_[k]] = 0.0;
      }
    }
    if (!chol_.factorize(value_)) return false;
    build_woodbury();
    return true;
  }

  // W = M_s^-1 A_d and the LU factors of S = Theta_d^-1 + A_d^T W.
  void build_woodbury() {
    smw_ok_ = false;
    const size_t k = dense_cols_.size();
    if (k == 0) return;
    const SparseMatrix& A = *A_;
    const std::vector<double>& theta = *theta_;
    const size_t m = static_cast<size_t>(m_);
    W_.assign(m * k, 0.0);
    for (size_t a = 0; a < k; ++a) {
      const Index j = dense_cols_[a];
      double* w = W_.data() + a * m;
      for (Index t = A.begin(j); t < A.end(j); ++t) w[A.index[t]] = A.value[t];
      chol_.solve(w);
    }
    S_.assign(k * k, 0.0);
    for (size_t a = 0; a < k; ++a) {
      const Index j = dense_cols_[a];
      for (size_t b = 0; b < k; ++b) {
        const double* w = W_.data() + b * m;
        double s = 0.0;
        for (Index t = A.begin(j); t < A.end(j); ++t) s += A.value[t] * w[A.index[t]];
        S_[a * k + b] = s;
      }
      S_[a * k + a] += 1.0 / theta[j];
    }
    // LU with partial pivoting (row-major).
    piv_.resize(k);
    for (size_t c = 0; c < k; ++c) {
      size_t p = c;
      for (size_t r = c + 1; r < k; ++r)
        if (std::abs(S_[r * k + c]) > std::abs(S_[p * k + c])) p = r;
      if (!(std::abs(S_[p * k + c]) > 0.0) || !std::isfinite(S_[p * k + c])) return;
      piv_[c] = p;
      if (p != c)
        for (size_t t = 0; t < k; ++t) std::swap(S_[c * k + t], S_[p * k + t]);
      const double inv = 1.0 / S_[c * k + c];
      for (size_t r = c + 1; r < k; ++r) {
        const double f = S_[r * k + c] * inv;
        S_[r * k + c] = f;
        if (f == 0.0) continue;
        for (size_t t = c + 1; t < k; ++t) S_[r * k + t] -= f * S_[c * k + t];
      }
    }
    smw_ok_ = true;
  }

  // z := (M_s + A_d Theta_d A_d^T)^-1 z by Woodbury (M_s^-1 z without dense columns).
  void precondition(std::vector<double>& z) const {
    chol_.solve(z);
    if (!smw_ok_) return;
    const SparseMatrix& A = *A_;
    const size_t k = dense_cols_.size(), m = static_cast<size_t>(m_);
    std::vector<double> u(k);
    for (size_t a = 0; a < k; ++a) {
      const Index j = dense_cols_[a];
      double s = 0.0;
      for (Index t = A.begin(j); t < A.end(j); ++t) s += A.value[t] * z[A.index[t]];
      u[a] = s;
    }
    // The row swaps of the factorization were applied to whole rows (multipliers included):
    // permute first, then the unit lower triangular solve.
    for (size_t c = 0; c < k; ++c) std::swap(u[c], u[piv_[c]]);
    for (size_t c = 0; c < k; ++c) {
      const double uc = u[c];
      if (uc == 0.0) continue;
      for (size_t r = c + 1; r < k; ++r) u[r] -= S_[r * k + c] * uc;
    }
    for (size_t c = k; c-- > 0;) {
      double s = u[c];
      for (size_t t = c + 1; t < k; ++t) s -= S_[c * k + t] * u[t];
      u[c] = s / S_[c * k + c];
    }
    for (size_t a = 0; a < k; ++a) {
      const double ua = u[a];
      if (ua == 0.0) continue;
      const double* w = W_.data() + a * m;
      for (size_t i = 0; i < m; ++i) z[i] -= w[i] * ua;
    }
  }

  // y = M r (full matrix, dense columns included).
  void multiply(const std::vector<double>& v, std::vector<double>& out) const {
    const SparseMatrix& A = *A_;
    const std::vector<double>& theta = *theta_;
    out.assign(static_cast<size_t>(m_), 0.0);
    for (Index j = 0; j < n_; ++j) {
      double s = 0.0;
      for (Index k = A.begin(j); k < A.end(j); ++k) s += A.value[k] * v[A.index[k]];
      s *= theta[j];
      if (s == 0.0) continue;
      for (Index k = A.begin(j); k < A.end(j); ++k) out[A.index[k]] += A.value[k] * s;
    }
    for (Index i = 0; i < m_; ++i) out[i] += delta_ * v[i];
  }

  // Preconditioned conjugate gradients from y = P^-1 r. Returns the final relative residual.
  double solve(const std::vector<double>& r, std::vector<double>& y) {
    y = r;
    precondition(y);
    const double rnorm = inf_norm(r);
    if (rnorm == 0.0) return 0.0;
    std::vector<double>& res = cg_r_;
    multiply(y, res);
    for (Index i = 0; i < m_; ++i) res[i] = r[i] - res[i];
    double rel = inf_norm(res) / rnorm;
    const int max_it = smw_ok_ || dense_cols_.empty() ? 10 : 10 + 2 * static_cast<int>(std::min<size_t>(dense_cols_.size(), 200));
    const double target = 1e-12;
    if (rel <= target) return rel;
    std::vector<double>& z = cg_z_;
    std::vector<double>& p = cg_p_;
    std::vector<double>& q = cg_q_;
    z = res;
    precondition(z);
    p = z;
    double rz = dot(res, z);
    std::vector<double> best = y;
    double best_rel = rel;
    for (int it = 0; it < max_it && rz > 0.0; ++it) {
      multiply(p, q);
      const double pq = dot(p, q);
      if (!(pq > 0.0)) break;
      const double alpha = rz / pq;
      for (Index i = 0; i < m_; ++i) {
        y[i] += alpha * p[i];
        res[i] -= alpha * q[i];
      }
      rel = inf_norm(res) / rnorm;
      if (rel < best_rel) {
        best_rel = rel;
        best = y;
      }
      if (rel <= target) break;
      z = res;
      precondition(z);
      const double rz_new = dot(res, z);
      const double beta = rz_new / rz;
      rz = rz_new;
      for (Index i = 0; i < m_; ++i) p[i] = z[i] + beta * p[i];
    }
    if (best_rel < rel) y = best;
    return best_rel;
  }

  long long factor_nnz() const { return chol_.factor_nnz(); }
  double cost() const { return cost_; }  // estimated work per factorization (flops, Woodbury included)
  Index dense_columns() const { return static_cast<Index>(dense_cols_.size()); }
  Index dropped() const { return chol_.num_dropped(); }

 private:
  const SparseMatrix* A_ = nullptr;
  const SparseMatrix* At_ = nullptr;
  int threads_ = 1;
  double cost_ = 0.0;
  Index m_ = 0, n_ = 0;
  std::vector<char> dense_;
  std::vector<Index> dense_cols_;
  std::vector<Index> start_, index_;
  std::vector<double> value_, work_;
  SparseCholesky chol_;
  const std::vector<double>* theta_ = nullptr;
  double delta_ = 0.0;
  std::vector<double> cg_r_, cg_z_, cg_p_, cg_q_;
  std::vector<double> W_, S_, dense_diag_;
  static constexpr double kDenseDiag = 1e-8;
  std::vector<size_t> piv_;
  bool smw_ok_ = false;
};

// The augmented (KKT) system K = [-(H + rho I)  A^T; A  delta I] factored directly as L S L^T
// (quasi-definite: the signs of the pivots are known, any symmetric ordering is admissible).
// Chosen when the normal equations fill much more (many more rows than columns, dense columns),
// and the form a convex QP needs (Q joins the (1,1) block). Solves are refined iteratively with
// the regularized matrix.
class AugmentedSystem {
 public:
  bool analyze(const SparseMatrix& A, int threads, const std::atomic<bool>* stop, const Logger* log) {
    A_ = &A;
    chol_.stop = stop;
    n_ = A.num_major;
    m_ = A.num_minor;
    const Index N = n_ + m_;
    start_.assign(1, 0);
    index_.clear();
    index_.reserve(static_cast<size_t>(A.nnz()) + static_cast<size_t>(N));
    for (Index j = 0; j < n_; ++j) {
      index_.push_back(j);
      for (Index k = A.begin(j); k < A.end(j); ++k) index_.push_back(n_ + A.index[k]);
      start_.push_back(static_cast<Index>(index_.size()));
    }
    for (Index i = 0; i < m_; ++i) {
      index_.push_back(n_ + i);
      start_.push_back(static_cast<Index>(index_.size()));
    }
    value_.assign(index_.size(), 0.0);
    for (Index j = 0; j < n_; ++j)
      for (Index k = A.begin(j); k < A.end(j); ++k) value_[start_[j] + 1 + (k - A.begin(j))] = A.value[k];
    std::vector<int8_t> signs(static_cast<size_t>(N), 1);
    for (Index j = 0; j < n_; ++j) signs[j] = -1;
    chol_.threads = threads;
    if (!chol_.analyze(N, start_, index_, nullptr, &signs)) return false;
    if (log) {
      log->log(2, "Barrier: augmented system %d x %d, factor %lld nonzeros, %.3g flops\n", N, N, chol_.factor_nnz(),
               chol_.flops());
    }
    return true;
  }
  double flops() const { return chol_.flops(); }
  long long factor_nnz() const { return chol_.factor_nnz(); }
  Index dropped() const { return chol_.num_dropped(); }

  bool factor(const std::vector<double>& h, double delta) {
    h_ = &h;
    delta_ = delta;
    for (Index j = 0; j < n_; ++j) value_[start_[j]] = -h[j];
    for (Index i = 0; i < m_; ++i) value_[start_[n_ + i]] = delta;
    return chol_.factorize(value_);
  }

  // [-H A^T; A delta] [dx; dy] = [r1; r2]; returns the final relative residual.
  double solve(const std::vector<double>& r1, const std::vector<double>& r2, std::vector<double>& dx,
               std::vector<double>& dy) {
    const Index N = n_ + m_;
    b_.resize(static_cast<size_t>(N));
    std::copy(r1.begin(), r1.end(), b_.begin());
    std::copy(r2.begin(), r2.end(), b_.begin() + n_);
    x_ = b_;
    chol_.solve(x_);
    const double bnorm = inf_norm(b_);
    double rel = 0.0;
    for (int it = 0; it < 4 && bnorm > 0.0; ++it) {
      multiply(x_, r_);
      for (Index k = 0; k < N; ++k) r_[k] = b_[k] - r_[k];
      rel = inf_norm(r_) / bnorm;
      if (rel <= 1e-12) break;
      chol_.solve(r_);
      for (Index k = 0; k < N; ++k) x_[k] += r_[k];
    }
    dx.assign(x_.begin(), x_.begin() + n_);
    dy.assign(x_.begin() + n_, x_.end());
    return rel;
  }

 private:
  void multiply(const std::vector<double>& x, std::vector<double>& out) const {
    const SparseMatrix& A = *A_;
    const std::vector<double>& h = *h_;
    out.assign(static_cast<size_t>(n_ + m_), 0.0);
    for (Index i = 0; i < m_; ++i) out[n_ + i] = delta_ * x[n_ + i];
    for (Index j = 0; j < n_; ++j) {
      double s = -h[j] * x[j];
      const double xj = x[j];
      for (Index k = A.begin(j); k < A.end(j); ++k) {
        s += A.value[k] * x[n_ + A.index[k]];
        out[n_ + A.index[k]] += A.value[k] * xj;
      }
      out[j] = s;
    }
  }

  const SparseMatrix* A_ = nullptr;
  Index n_ = 0, m_ = 0;
  std::vector<Index> start_, index_;
  std::vector<double> value_;
  SparseCholesky chol_;
  const std::vector<double>* h_ = nullptr;
  double delta_ = 0.0;
  std::vector<double> b_, x_, r_;
};

// The regularized Newton system of the IPM (see barrier.hpp): normal equations or the augmented
// system, whichever factors with less work (the augmented system is what a QP will need).
class KktSolver {
 public:
  // kkt: 0 choose, 1 normal equations, 2 augmented system.
  bool analyze(const SparseMatrix& A, const SparseMatrix& At, int kkt, int threads, const std::atomic<bool>* stop,
               const Logger* log) {
    A_ = &A;
    At_ = &At;
    static const int env = [] {
      const char* v = std::getenv("MILP_BARRIER_KKT");  // testing: 1 normal equations, 2 augmented system
      return v ? std::atoi(v) : 0;
    }();
    const int force = env != 0 ? env : kkt;
    Timer t;
    const bool aug_ok = force != 1 && aug_.analyze(A, threads, stop, log);
    if (stop && stop->load()) return false;
    const double t_aug = t.elapsed();
    if (force == 2) {
      augmented_ = aug_ok;
      return aug_ok;
    }
    // The normal equations are given up once their factor is far larger than the augmented one.
    const bool ne_ok = ne_.analyze(A, At, threads, aug_ok ? 20 * aug_.factor_nnz() + 100000 : 0, stop, log);
    if (stop && stop->load()) return false;
    if (log) log->log(2, "Barrier: analysis %.2fs augmented system, %.2fs normal equations\n", t_aug, t.elapsed() - t_aug);
    if (!aug_ok) {
      augmented_ = false;
      return ne_ok;
    }
    // The augmented system is larger and indefinite; it needs a clear advantage.
    augmented_ = !ne_ok || force == 2 || aug_.flops() < 0.5 * ne_.cost();
    if (log) log->log(2, "Barrier: using the %s\n", augmented_ ? "augmented system" : "normal equations");
    if (augmented_) ne_ = NormalEquations();
    else aug_ = AugmentedSystem();
    return true;
  }
  bool augmented() const { return augmented_; }
  // h: the diagonal H + rho I (> 0); delta: dual regularization.
  bool factor(const std::vector<double>& h, double delta) {
    if (augmented_) return aug_.factor(h, delta);
    theta_.resize(h.size());
    for (size_t j = 0; j < h.size(); ++j) theta_[j] = 1.0 / h[j];
    return ne_.factor(theta_, delta);
  }
  // [-H A^T; A delta] [dx; dy] = [r1; r2].
  double solve(const std::vector<double>& r1, const std::vector<double>& r2, std::vector<double>& dx,
               std::vector<double>& dy) {
    if (augmented_) return aug_.solve(r1, r2, dx, dy);
    const SparseMatrix& A = *A_;
    const Index n = A.num_major;
    rhs_ = r2;
    for (Index j = 0; j < n; ++j) {
      const double t = theta_[j] * r1[j];
      if (t == 0.0) continue;
      for (Index k = A.begin(j); k < A.end(j); ++k) rhs_[A.index[k]] += A.value[k] * t;
    }
    const double rel = ne_.solve(rhs_, dy);
    dx.resize(static_cast<size_t>(n));
    for (Index j = 0; j < n; ++j) {
      double s = -r1[j];
      for (Index k = A.begin(j); k < A.end(j); ++k) s += A.value[k] * dy[A.index[k]];
      dx[j] = theta_[j] * s;
    }
    return rel;
  }
  long long factor_nnz() const { return augmented_ ? aug_.factor_nnz() : ne_.factor_nnz(); }
  Index dense_columns() const { return augmented_ ? 0 : ne_.dense_columns(); }
  Index dropped() const { return augmented_ ? aug_.dropped() : ne_.dropped(); }

 private:
  const SparseMatrix* A_ = nullptr;
  const SparseMatrix* At_ = nullptr;
  bool augmented_ = false;
  NormalEquations ne_;
  AugmentedSystem aug_;
  std::vector<double> theta_, rhs_;
};

class Ipm {
 public:
  Ipm(const SparseMatrix& A, const std::vector<double>& cost, const std::vector<double>& col_lb,
      const std::vector<double>& col_ub, const std::vector<double>& row_lb, const std::vector<double>& row_ub,
      const IpmOptions& opt)
      : oA_(A), oc_(cost), ocl_(col_lb), ocu_(col_ub), orl_(row_lb), oru_(row_ub), opt_(opt) {}

  IpmResult run();

 private:
  bool build();  // standard form; false if trivially infeasible
  void scale();
  void starting_point();
  void residuals();
  // Newton direction for complementarity right-hand sides cl, cu; with_res: include the
  // primal/dual/bound residuals (predictor, corrector) or not (centrality correctors).
  void direction(const std::vector<double>& cl, const std::vector<double>& cu, bool with_res, std::vector<double>& dx,
                 std::vector<double>& dy, std::vector<double>& dxl, std::vector<double>& dxu, std::vector<double>& dzl,
                 std::vector<double>& dzu);
  void step_lengths(const std::vector<double>& dxl, const std::vector<double>& dxu, const std::vector<double>& dzl,
                    const std::vector<double>& dzu, double* ap, double* ad) const;
  void finish(IpmResult& res);

  const SparseMatrix& oA_;
  const std::vector<double>&oc_, &ocl_, &ocu_, &orl_, &oru_;
  IpmOptions opt_;
  Timer timer_;

  // Standard form (scaled): min c^T x, A x = b, l <= x <= u.
  Index m_ = 0, n_ = 0;
  SparseMatrix A_, At_;
  std::vector<double> c_, b_, l_, u_;
  std::vector<char> has_l_, has_u_;
  std::vector<double> rscale_, cscale_;  // A_ = R A' C
  // Maps: original column -> standard column (-1: fixed), original row -> standard row (-1: dropped).
  std::vector<Index> col_map_, row_map_;
  std::vector<double> fixed_val_;
  bool infeasible_ = false;
  Index nbounds_ = 0;

  // Iterate.
  std::vector<double> x_, xl_, xu_, zl_, zu_, y_;
  // Residuals: rp = b - A x, rl = l - x + xl, ru = u - x - xu, rd = c - A^T y - zl + zu.
  std::vector<double> rp_, rl_, ru_, rd_;
  double mu_ = 0.0;
  double last_gap_ = kInf;
  KktSolver kkt_;
  std::vector<double> h_;
  double reg_p_ = 1e-10, reg_free_ = 1e-8, reg_d_ = 1e-10;
};

bool Ipm::build() {
  const Index on = static_cast<Index>(oc_.size()), om = static_cast<Index>(orl_.size());
  col_map_.assign(static_cast<size_t>(on), -1);
  fixed_val_.assign(static_cast<size_t>(on), 0.0);
  std::vector<double> fixed_act(static_cast<size_t>(om), 0.0);
  std::vector<Index> row_len(static_cast<size_t>(om), 0);
  Index ncols = 0;
  for (Index j = 0; j < on; ++j) {
    if (ocl_[j] == ocu_[j] && is_finite(ocl_[j])) {
      fixed_val_[j] = ocl_[j];
      for (Index k = oA_.begin(j); k < oA_.end(j); ++k) fixed_act[oA_.index[k]] += oA_.value[k] * ocl_[j];
    } else {
      col_map_[j] = ncols++;
      for (Index k = oA_.begin(j); k < oA_.end(j); ++k) ++row_len[oA_.index[k]];
    }
  }
  row_map_.assign(static_cast<size_t>(om), -1);
  m_ = 0;
  std::vector<Index> slack_rows;
  for (Index i = 0; i < om; ++i) {
    const double lo = orl_[i], hi = oru_[i];
    if (!is_finite(lo) && !is_finite(hi)) continue;
    if (row_len[i] == 0) {
      const double t = 1e-9 * (1.0 + std::abs(fixed_act[i]));
      if (fixed_act[i] < lo - t || fixed_act[i] > hi + t) infeasible_ = true;
      continue;
    }
    row_map_[i] = m_++;
    b_.push_back(lo == hi ? lo - fixed_act[i] : -fixed_act[i]);
    if (lo != hi) slack_rows.push_back(i);
  }
  n_ = ncols + static_cast<Index>(slack_rows.size());
  c_.assign(static_cast<size_t>(n_), 0.0);
  l_.assign(static_cast<size_t>(n_), -kInf);
  u_.assign(static_cast<size_t>(n_), kInf);
  std::vector<Index> J, I;
  std::vector<double> V;
  J.reserve(static_cast<size_t>(oA_.nnz()) + slack_rows.size());
  I.reserve(J.capacity());
  V.reserve(J.capacity());
  for (Index j = 0; j < on; ++j) {
    const Index cj = col_map_[j];
    if (cj < 0) continue;
    c_[cj] = oc_[j];
    l_[cj] = ocl_[j];
    u_[cj] = ocu_[j];
    for (Index k = oA_.begin(j); k < oA_.end(j); ++k) {
      const Index r = row_map_[oA_.index[k]];
      if (r < 0 || oA_.value[k] == 0.0) continue;
      J.push_back(cj);
      I.push_back(r);
      V.push_back(oA_.value[k]);
    }
  }
  for (size_t s = 0; s < slack_rows.size(); ++s) {
    const Index i = slack_rows[s];
    const Index cj = ncols + static_cast<Index>(s);
    l_[cj] = orl_[i] - fixed_act[i];
    u_[cj] = oru_[i] - fixed_act[i];
    J.push_back(cj);
    I.push_back(row_map_[i]);
    V.push_back(-1.0);
  }
  A_ = SparseMatrix::from_triplets(n_, m_, J, I, V);
  for (Index j = 0; j < n_; ++j) {
    if (l_[j] > u_[j]) infeasible_ = true;
  }
  return !infeasible_;
}

void Ipm::scale() {
  rscale_.assign(static_cast<size_t>(m_), 1.0);
  cscale_.assign(static_cast<size_t>(n_), 1.0);
  if (!opt_.scale) return;
  At_ = A_.transpose();
  // Geometric scaling: alternate row and column passes with 1/sqrt(max * min).
  for (int pass = 0; pass < 8; ++pass) {
    for (Index i = 0; i < m_; ++i) {
      double mx = 0.0, mn = kInf;
      for (Index k = At_.begin(i); k < At_.end(i); ++k) {
        const double v = std::abs(At_.value[k]) * cscale_[At_.index[k]];
        mx = std::max(mx, v);
        mn = std::min(mn, v);
      }
      if (mx > 0.0) rscale_[i] = 1.0 / std::sqrt(mx * mn);
    }
    for (Index j = 0; j < n_; ++j) {
      double mx = 0.0, mn = kInf;
      for (Index k = A_.begin(j); k < A_.end(j); ++k) {
        const double v = std::abs(A_.value[k]) * rscale_[A_.index[k]];
        mx = std::max(mx, v);
        mn = std::min(mn, v);
      }
      if (mx > 0.0) cscale_[j] = 1.0 / std::sqrt(mx * mn);
    }
  }
  auto pow2 = [](double v) { return std::exp2(std::round(std::log2(v))); };
  for (double& v : rscale_) v = pow2(v);
  for (double& v : cscale_) v = pow2(v);
  for (Index j = 0; j < n_; ++j) {
    for (Index k = A_.begin(j); k < A_.end(j); ++k) A_.value[k] *= rscale_[A_.index[k]] * cscale_[j];
    c_[j] *= cscale_[j];
    l_[j] /= cscale_[j];
    u_[j] /= cscale_[j];
  }
  for (Index i = 0; i < m_; ++i) b_[i] *= rscale_[i];
}

void Ipm::residuals() {
  rp_ = b_;
  for (Index j = 0; j < n_; ++j) {
    const double v = x_[j];
    if (v == 0.0) continue;
    for (Index k = A_.begin(j); k < A_.end(j); ++k) rp_[A_.index[k]] -= A_.value[k] * v;
  }
  rl_.assign(static_cast<size_t>(n_), 0.0);
  ru_.assign(static_cast<size_t>(n_), 0.0);
  rd_.assign(static_cast<size_t>(n_), 0.0);
  double comp = 0.0;
  for (Index j = 0; j < n_; ++j) {
    double s = c_[j];
    for (Index k = A_.begin(j); k < A_.end(j); ++k) s -= A_.value[k] * y_[A_.index[k]];
    if (has_l_[j]) {
      rl_[j] = l_[j] - x_[j] + xl_[j];
      s -= zl_[j];
      comp += xl_[j] * zl_[j];
    }
    if (has_u_[j]) {
      ru_[j] = u_[j] - x_[j] - xu_[j];
      s += zu_[j];
      comp += xu_[j] * zu_[j];
    }
    rd_[j] = s;
  }
  mu_ = nbounds_ > 0 ? comp / nbounds_ : 0.0;
}

void Ipm::direction(const std::vector<double>& cl, const std::vector<double>& cu, bool with_res, std::vector<double>& dx,
                    std::vector<double>& dy, std::vector<double>& dxl, std::vector<double>& dxu,
                    std::vector<double>& dzl, std::vector<double>& dzu) {
  std::vector<double> r1(static_cast<size_t>(n_)), r2;
  for (Index j = 0; j < n_; ++j) {
    double s = with_res ? rd_[j] : 0.0;
    if (has_l_[j]) s -= (cl[j] + zl_[j] * (with_res ? rl_[j] : 0.0)) / xl_[j];
    if (has_u_[j]) s += (cu[j] - zu_[j] * (with_res ? ru_[j] : 0.0)) / xu_[j];
    r1[j] = s;
  }
  if (with_res) r2 = rp_;
  else r2.assign(static_cast<size_t>(m_), 0.0);
  kkt_.solve(r1, r2, dx, dy);
  dxl.assign(static_cast<size_t>(n_), 0.0);
  dxu.assign(static_cast<size_t>(n_), 0.0);
  dzl.assign(static_cast<size_t>(n_), 0.0);
  dzu.assign(static_cast<size_t>(n_), 0.0);
  for (Index j = 0; j < n_; ++j) {
    if (has_l_[j]) {
      dxl[j] = dx[j] - (with_res ? rl_[j] : 0.0);
      dzl[j] = (cl[j] - zl_[j] * dxl[j]) / xl_[j];
    }
    if (has_u_[j]) {
      dxu[j] = (with_res ? ru_[j] : 0.0) - dx[j];
      dzu[j] = (cu[j] - zu_[j] * dxu[j]) / xu_[j];
    }
  }
}

void Ipm::step_lengths(const std::vector<double>& dxl, const std::vector<double>& dxu, const std::vector<double>& dzl,
                       const std::vector<double>& dzu, double* ap, double* ad) const {
  double a = 1.0, b = 1.0;
  for (Index j = 0; j < n_; ++j) {
    if (has_l_[j]) {
      if (dxl[j] < 0.0) a = std::min(a, -xl_[j] / dxl[j]);
      if (dzl[j] < 0.0) b = std::min(b, -zl_[j] / dzl[j]);
    }
    if (has_u_[j]) {
      if (dxu[j] < 0.0) a = std::min(a, -xu_[j] / dxu[j]);
      if (dzu[j] < 0.0) b = std::min(b, -zu_[j] / dzu[j]);
    }
  }
  *ap = a;
  *ad = b;
}

void Ipm::starting_point() {
  x_.assign(static_cast<size_t>(n_), 0.0);
  xl_.assign(static_cast<size_t>(n_), 0.0);
  xu_.assign(static_cast<size_t>(n_), 0.0);
  zl_.assign(static_cast<size_t>(n_), 0.0);
  zu_.assign(static_cast<size_t>(n_), 0.0);
  y_.assign(static_cast<size_t>(m_), 0.0);
  // x = A^T (A A^T)^-1 b (least norm), y = (A A^T)^-1 A c, z = c - A^T y.
  std::vector<double> h(static_cast<size_t>(n_), 1.0);
  kkt_.factor(h, reg_d_);
  std::vector<double> zero_n(static_cast<size_t>(n_), 0.0), dy, dx;
  kkt_.solve(zero_n, b_, dx, dy);  // dx = A^T (A A^T)^-1 b
  x_ = dx;
  // [-I A^T; A 0] [dx; y] = [c; 0]: dx = A^T y - c, A dx = 0, so y = (A A^T)^-1 A c.
  std::vector<double> zero_m(static_cast<size_t>(m_), 0.0);
  kkt_.solve(c_, zero_m, dx, y_);
  std::vector<double> z(static_cast<size_t>(n_));
  for (Index j = 0; j < n_; ++j) {
    double s = c_[j];
    for (Index k = A_.begin(j); k < A_.end(j); ++k) s -= A_.value[k] * y_[A_.index[k]];
    z[j] = s;
  }
  double minx = kInf, minz = kInf;
  for (Index j = 0; j < n_; ++j) {
    if (has_l_[j] && has_u_[j]) {
      xl_[j] = x_[j] - l_[j];
      xu_[j] = u_[j] - x_[j];
      zl_[j] = std::max(z[j], 0.0);
      zu_[j] = std::max(-z[j], 0.0);
    } else if (has_l_[j]) {
      xl_[j] = x_[j] - l_[j];
      zl_[j] = z[j];
    } else if (has_u_[j]) {
      xu_[j] = u_[j] - x_[j];
      zu_[j] = -z[j];
    }
    if (has_l_[j]) minx = std::min(minx, xl_[j]), minz = std::min(minz, zl_[j]);
    if (has_u_[j]) minx = std::min(minx, xu_[j]), minz = std::min(minz, zu_[j]);
  }
  if (nbounds_ == 0) return;
  const double dxs = std::max(-1.5 * minx, 0.0), dzs = std::max(-1.5 * minz, 0.0);
  double sx = 0.0, sz = 0.0, xz = 0.0;
  for (Index j = 0; j < n_; ++j) {
    if (has_l_[j]) {
      xl_[j] += dxs;
      zl_[j] += dzs;
      sx += xl_[j], sz += zl_[j], xz += xl_[j] * zl_[j];
    }
    if (has_u_[j]) {
      xu_[j] += dxs;
      zu_[j] += dzs;
      sx += xu_[j], sz += zu_[j], xz += xu_[j] * zu_[j];
    }
  }
  double ex = sz > 0.0 ? 0.5 * xz / sz : 0.0, ez = sx > 0.0 ? 0.5 * xz / sx : 0.0;
  // Degenerate guesses (all zero): start from a unit point instead.
  if (!(ex > 0.0) || !std::isfinite(ex)) ex = 1.0;
  if (!(ez > 0.0) || !std::isfinite(ez)) ez = 1.0;
  for (Index j = 0; j < n_; ++j) {
    if (has_l_[j]) xl_[j] = std::max(xl_[j] + ex, 1e-8), zl_[j] = std::max(zl_[j] + ez, 1e-8);
    if (has_u_[j]) xu_[j] = std::max(xu_[j] + ex, 1e-8), zu_[j] = std::max(zu_[j] + ez, 1e-8);
  }
}

IpmResult Ipm::run() {
  IpmResult res;
  if (!build()) {
    res.status = IpmStatus::PrimalInfeasible;
    finish(res);
    return res;
  }
  scale();
  At_ = A_.transpose();
  has_l_.assign(static_cast<size_t>(n_), 0);
  has_u_.assign(static_cast<size_t>(n_), 0);
  nbounds_ = 0;
  for (Index j = 0; j < n_; ++j) {
    has_l_[j] = is_finite(l_[j]);
    has_u_[j] = is_finite(u_[j]);
    nbounds_ += has_l_[j] + has_u_[j];
  }
  const double t_an = timer_.elapsed();
  if (!kkt_.analyze(A_, At_, opt_.kkt, opt_.threads, opt_.stop, opt_.log)) {
    res.status = opt_.stop && opt_.stop->load() ? IpmStatus::TimeLimit : IpmStatus::NumericalError;
    finish(res);
    return res;
  }
  res.analysis_time = timer_.elapsed() - t_an;
  double factor_time = 0.0;
  res.factor_nnz = kkt_.factor_nnz();
  res.dense_columns = kkt_.dense_columns();
  res.augmented = kkt_.augmented();
  if (kkt_.augmented()) {
    // The augmented system's pivots are the regularizations themselves until the elimination
    // reaches the coupled entries: it needs more than the normal equations.
    constexpr double kAugmentedReg = 1e-7;
    reg_p_ = std::max(reg_p_, kAugmentedReg);
    reg_d_ = std::max(reg_d_, kAugmentedReg);
  }
  starting_point();

  const double bnorm = std::max(inf_norm(b_), [&] {
    double v = 0.0;
    for (Index j = 0; j < n_; ++j) {
      if (has_l_[j]) v = std::max(v, std::abs(l_[j]));
      if (has_u_[j]) v = std::max(v, std::abs(u_[j]));
    }
    return v;
  }());
  const double cnorm = inf_norm(c_);
  if (opt_.log) {
    opt_.log->log(2, "Barrier: %d rows, %d columns (standard form), %d nonzeros\n", m_, n_, A_.nnz());
    opt_.log->log(2, " iter     primal obj        dual obj    p.inf    d.inf       mu\n");
  }

  std::vector<double> dx, dy, dxl, dxu, dzl, dzu;
  std::vector<double> ax, ay, axl, axu, azl, azu;  // affine direction
  std::vector<double> cx, cy, cxl, cxu, czl, czu;  // corrector
  std::vector<double> cl(static_cast<size_t>(n_)), cu(static_cast<size_t>(n_));
  IpmStatus status = IpmStatus::IterationLimit;
  int small_steps = 0;
  double best_merit = kInf;
  int no_progress = 0;
  int iter = 0;
  double pinf0 = 0.0, dinf0 = 0.0, mu0 = 0.0;
  // Best iterate so far (largest of the relative measures): when the method breaks down near the
  // end (round-off), a point within kNearOptimal is still good enough for the crossover.
  constexpr double kNearOptimal = 1e-6;
  struct Snapshot {
    std::vector<double> x, xl, xu, zl, zu, y;
    double merit = kInf;
  } best;
  for (;; ++iter) {
    residuals();
    double pobj = dot(c_, x_), dobj = dot(b_, y_);
    for (Index j = 0; j < n_; ++j) {
      if (has_l_[j]) dobj += l_[j] * zl_[j];
      if (has_u_[j]) dobj -= u_[j] * zu_[j];
    }
    const double pinf = std::max({inf_norm(rp_), inf_norm(rl_), inf_norm(ru_)}) / (1.0 + bnorm);
    const double dinf = inf_norm(rd_) / (1.0 + cnorm);
    const double gap = std::abs(pobj - dobj) / (1.0 + std::abs(pobj));
    last_gap_ = gap;
    if (opt_.log && opt_.log->verbosity() >= 2) {
      opt_.log->log(2, "%5d %15.8e %15.8e %8.2e %8.2e %8.2e %6.2fs\n", iter, pobj, dobj, pinf, dinf, mu_, timer_.elapsed());
    }
    if (pinf <= opt_.tol && dinf <= opt_.tol && gap <= opt_.tol) {
      status = IpmStatus::Optimal;
      break;
    }
    if (iter >= opt_.max_iterations) {
      status = IpmStatus::IterationLimit;
      break;
    }
    if (timer_.elapsed() > opt_.time_limit || (opt_.stop && opt_.stop->load(std::memory_order_relaxed))) {
      status = IpmStatus::TimeLimit;
      break;
    }
    // Divergence: unbounded duals with a primal residual that does not go away mean primal
    // infeasibility, unbounded primal values with a stalled dual residual dual infeasibility.
    double xmax = inf_norm(x_), zmax = std::max(inf_norm(zl_), inf_norm(zu_));
    zmax = std::max(zmax, inf_norm(y_));
    const double scale_x = 1e10 * (1.0 + bnorm), scale_z = 1e10 * (1.0 + cnorm);
    if (zmax > scale_z && pinf > opt_.tol && iter > 5) {
      status = IpmStatus::PrimalInfeasible;
      break;
    }
    if (xmax > scale_x && dinf > opt_.tol && iter > 5) {
      status = IpmStatus::DualInfeasible;
      break;
    }
    const double merit = std::max({pinf, dinf, gap});
    if (iter == 0) pinf0 = pinf, dinf0 = dinf, mu0 = mu_;
    if (!std::isfinite(merit)) {
      status = IpmStatus::NumericalError;
      break;
    }
    if (merit < best.merit && merit <= kNearOptimal) {
      best.merit = merit;
      best.x = x_, best.xl = xl_, best.xu = xu_, best.zl = zl_, best.zu = zu_, best.y = y_;
    }
    if (merit < 0.9 * best_merit) {
      best_merit = merit;
      no_progress = 0;
    } else if (++no_progress >= 30) {
      status = IpmStatus::NumericalError;
      break;
    }

    // Newton matrix.
    h_.assign(static_cast<size_t>(n_), 0.0);
    for (Index j = 0; j < n_; ++j) {
      double h = 0.0;
      if (has_l_[j]) h += zl_[j] / xl_[j];
      if (has_u_[j]) h += zu_[j] / xu_[j];
      h_[j] = (has_l_[j] || has_u_[j]) ? h + reg_p_ : reg_free_;
    }
    const double t_f = timer_.elapsed();
    if (!kkt_.factor(h_, reg_d_)) {
      status = opt_.stop && opt_.stop->load() ? IpmStatus::TimeLimit : IpmStatus::NumericalError;
      break;
    }
    factor_time += timer_.elapsed() - t_f;

    // Predictor (affine scaling).
    for (Index j = 0; j < n_; ++j) {
      cl[j] = has_l_[j] ? -xl_[j] * zl_[j] : 0.0;
      cu[j] = has_u_[j] ? -xu_[j] * zu_[j] : 0.0;
    }
    direction(cl, cu, true, ax, ay, axl, axu, azl, azu);
    double ap = 0.0, ad = 0.0;
    step_lengths(axl, axu, azl, azu, &ap, &ad);
    double sigma = 0.0;
    if (nbounds_ > 0 && mu_ > 0.0) {
      double comp = 0.0;
      for (Index j = 0; j < n_; ++j) {
        if (has_l_[j]) comp += (xl_[j] + ap * axl[j]) * (zl_[j] + ad * azl[j]);
        if (has_u_[j]) comp += (xu_[j] + ap * axu[j]) * (zu_[j] + ad * azu[j]);
      }
      const double mu_aff = comp / nbounds_;
      sigma = std::clamp(std::pow(mu_aff / mu_, 3.0), 0.0, 1.0);
    }
    // Mehrotra corrector. Infeasible-IPM neighbourhood: complementarity may not fall much faster
    // than the infeasibilities (||r|| / ||r0|| <= mu / mu0 up to a factor); otherwise the iterates
    // reach the boundary of a wrong face with the residuals stuck (the regularization absorbs them).
    const double infeas0 = std::max(pinf0, dinf0);
    const double mu_floor = infeas0 > 0.0 ? 1e-3 * mu0 * std::max(pinf, dinf) / infeas0 : 0.0;
    const double smu = std::max(sigma * mu_, std::min(mu_floor, mu_));
    for (Index j = 0; j < n_; ++j) {
      cl[j] = has_l_[j] ? smu - xl_[j] * zl_[j] - axl[j] * azl[j] : 0.0;
      cu[j] = has_u_[j] ? smu - xu_[j] * zu_[j] - axu[j] * azu[j] : 0.0;
    }
    direction(cl, cu, true, dx, dy, dxl, dxu, dzl, dzu);
    step_lengths(dxl, dxu, dzl, dzu, &ap, &ad);

    // Gondzio's centrality correctors: aim at a longer step, move outlying complementarity
    // products into [0.1, 10] * sigma * mu at the enlarged trial point.
    for (int k = 0; k < opt_.max_correctors && nbounds_ > 0 && smu > 0.0; ++k) {
      if (ap >= 1.0 && ad >= 1.0) break;
      const double tp = std::min(1.0, 1.5 * ap + 0.1), td = std::min(1.0, 1.5 * ad + 0.1);
      const double lo = 0.1 * smu, hi = 10.0 * smu;
      auto target = [&](double v) {
        double t = 0.0;
        if (v < lo) t = lo - v;
        else if (v > hi) t = std::max(hi - v, -hi);
        return t;
      };
      for (Index j = 0; j < n_; ++j) {
        cl[j] = has_l_[j] ? target((xl_[j] + tp * dxl[j]) * (zl_[j] + td * dzl[j])) : 0.0;
        cu[j] = has_u_[j] ? target((xu_[j] + tp * dxu[j]) * (zu_[j] + td * dzu[j])) : 0.0;
      }
      direction(cl, cu, false, cx, cy, cxl, cxu, czl, czu);
      for (Index j = 0; j < n_; ++j) {
        cx[j] += dx[j], cxl[j] += dxl[j], cxu[j] += dxu[j], czl[j] += dzl[j], czu[j] += dzu[j];
      }
      for (Index i = 0; i < m_; ++i) cy[i] += dy[i];
      double ap2 = 0.0, ad2 = 0.0;
      step_lengths(cxl, cxu, czl, czu, &ap2, &ad2);
      if (ap2 + ad2 < 1.01 * (ap + ad)) break;
      dx.swap(cx), dy.swap(cy), dxl.swap(cxl), dxu.swap(cxu), dzl.swap(czl), dzu.swap(czu);
      ap = ap2, ad = ad2;
    }

    const double eta = 0.9995;
    ap = std::min(1.0, eta * ap);
    ad = std::min(1.0, eta * ad);
    if (ap < 1e-8 && ad < 1e-8) {
      if (++small_steps >= 3) {
        status = IpmStatus::NumericalError;
        break;
      }
    } else {
      small_steps = 0;
    }
    for (Index j = 0; j < n_; ++j) {
      x_[j] += ap * dx[j];
      if (has_l_[j]) xl_[j] += ap * dxl[j], zl_[j] += ad * dzl[j];
      if (has_u_[j]) xu_[j] += ap * dxu[j], zu_[j] += ad * dzu[j];
    }
    for (Index i = 0; i < m_; ++i) y_[i] += ad * dy[i];
  }
  if ((status == IpmStatus::NumericalError || status == IpmStatus::IterationLimit) && best.merit <= kNearOptimal) {
    x_ = std::move(best.x), xl_ = std::move(best.xl), xu_ = std::move(best.xu);
    zl_ = std::move(best.zl), zu_ = std::move(best.zu), y_ = std::move(best.y);
    last_gap_ = best.merit;
    if (opt_.log) {
      opt_.log->log(2, "Barrier: %s; continuing from the best iterate (relative error %.2g)\n", to_string(status),
                    best.merit);
    }
    status = IpmStatus::Optimal;
    res.approximate = true;
  }
  res.status = status;
  res.iterations = iter;
  res.factor_time = factor_time;
  finish(res);
  if (opt_.log) {
    opt_.log->log(1, "Barrier: %s after %d iterations, objective %.10g, %.2fs (%s, %lld nonzeros in the factor; "
                  "analysis %.2fs, factorizations %.2fs)\n",
                  to_string(res.status), res.iterations, opt_.log_sense * (res.objective + opt_.log_offset), res.time,
                  res.augmented ? "augmented system" : "normal equations", res.factor_nnz, res.analysis_time,
                  res.factor_time);
  }
  return res;
}

void Ipm::finish(IpmResult& res) {
  const Index on = static_cast<Index>(oc_.size()), om = static_cast<Index>(orl_.size());
  res.x.assign(static_cast<size_t>(on), 0.0);
  res.y.assign(static_cast<size_t>(om), 0.0);
  res.d.assign(static_cast<size_t>(on), 0.0);
  const bool have = !x_.empty() || n_ == 0;
  for (Index j = 0; j < on; ++j) {
    const Index cj = col_map_.empty() ? -1 : col_map_[j];
    if (cj < 0) res.x[j] = fixed_val_.empty() ? 0.0 : fixed_val_[j];
    else if (have && cj < static_cast<Index>(x_.size())) res.x[j] = x_[cj] * cscale_[cj];
  }
  for (Index i = 0; i < om; ++i) {
    const Index r = row_map_.empty() ? -1 : row_map_[i];
    if (r >= 0 && r < static_cast<Index>(y_.size())) res.y[i] = y_[r] * rscale_[r];
  }
  // Measures in the original problem.
  std::vector<double> act(static_cast<size_t>(om), 0.0);
  double pviol = 0.0, dviol = 0.0, obj = 0.0;
  for (Index j = 0; j < on; ++j) {
    double s = oc_[j];
    for (Index k = oA_.begin(j); k < oA_.end(j); ++k) {
      act[oA_.index[k]] += oA_.value[k] * res.x[j];
      s -= oA_.value[k] * res.y[oA_.index[k]];
    }
    res.d[j] = s;
    obj += oc_[j] * res.x[j];
    pviol = std::max({pviol, ocl_[j] - res.x[j], res.x[j] - ocu_[j]});
    // Reduced cost of the wrong sign for a variable away from the bound it would need.
    if (s > 0.0 && !is_finite(ocl_[j])) dviol = std::max(dviol, s);
    if (s < 0.0 && !is_finite(ocu_[j])) dviol = std::max(dviol, -s);
  }
  for (Index i = 0; i < om; ++i) {
    pviol = std::max({pviol, orl_[i] - act[i], act[i] - oru_[i]});
    if (res.y[i] > 0.0 && !is_finite(orl_[i])) dviol = std::max(dviol, res.y[i]);
    if (res.y[i] < 0.0 && !is_finite(oru_[i])) dviol = std::max(dviol, -res.y[i]);
  }
  res.objective = obj;
  res.rel_gap = last_gap_;
  res.primal_infeas = pviol;
  res.dual_infeas = dviol;
  res.time = timer_.elapsed();
}

}  // namespace

IpmResult ipm_solve(const SparseMatrix& A, const std::vector<double>& cost, const std::vector<double>& col_lb,
                    const std::vector<double>& col_ub, const std::vector<double>& row_lb,
                    const std::vector<double>& row_ub, const IpmOptions& opt) {
  Ipm ipm(A, cost, col_lb, col_ub, row_lb, row_ub, opt);
  return ipm.run();
}

}  // namespace milp
