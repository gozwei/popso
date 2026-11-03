#include "milp/lp/cholesky.hpp"

#include <algorithm>
#include <cmath>

#include "milp/core/worker_pool.hpp"
#include "milp/lp/ordering.hpp"

namespace milp {

namespace {

// Strictly lower pattern of P M P^T row by row: for row i the columns j < i.
void permuted_rows(Index n, const std::vector<Index>& start, const std::vector<Index>& index,
                   const std::vector<Index>& pinv, std::vector<Index>& rstart, std::vector<Index>& rcols) {
  rstart.assign(static_cast<size_t>(n) + 1, 0);
  for (Index c = 0; c < n; ++c) {
    for (Index k = start[c]; k < start[c + 1]; ++k) {
      const Index pr = pinv[index[k]], pc = pinv[c];
      if (pr != pc) ++rstart[std::max(pr, pc) + 1];
    }
  }
  for (Index i = 0; i < n; ++i) rstart[i + 1] += rstart[i];
  rcols.resize(static_cast<size_t>(rstart[n]));
  std::vector<Index> fill(rstart.begin(), rstart.end() - 1);
  for (Index c = 0; c < n; ++c) {
    for (Index k = start[c]; k < start[c + 1]; ++k) {
      const Index pr = pinv[index[k]], pc = pinv[c];
      if (pr != pc) rcols[fill[std::max(pr, pc)]++] = std::min(pr, pc);
    }
  }
}

// T[r + c * ldT] = sum_{k < kw} X[k * ld + r] * X[k * ld + c] for 0 <= c < nc and the rows r in
// [r0, r1) with r >= (c rounded down to a multiple of 4): the lower part of X X^T (and a few entries
// above the diagonal, which callers ignore). Blocks of 8 rows x 4 columns are accumulated in
// registers.
void syrk_lower(const double* X, size_t ld, Index kw, Index nc, Index r0, Index r1, double* T, size_t ldT,
                const double* S) {
  for (Index c0 = 0; c0 < nc; c0 += 4) {
    const Index cw = std::min<Index>(4, nc - c0);
    Index rb = std::max(r0, c0);
    if (cw == 4) {
      for (; rb + 8 <= r1; rb += 8) {
        double acc[4][8] = {};
        for (Index k = 0; k < kw; ++k) {
          const double* xk = X + static_cast<size_t>(k) * ld;
          double a[8], b[4];
          for (int i = 0; i < 8; ++i) a[i] = xk[rb + i];
          const double sk = S ? S[k] : 1.0;
          for (int j = 0; j < 4; ++j) b[j] = xk[c0 + j] * sk;
          for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 8; ++i) acc[j][i] += a[i] * b[j];
        }
        for (int j = 0; j < 4; ++j)
          for (int i = 0; i < 8; ++i) T[static_cast<size_t>(c0 + j) * ldT + rb + i] = acc[j][i];
      }
    }
    for (; rb < r1; ++rb) {
      for (Index c = c0; c < c0 + cw; ++c) {
        double s = 0.0;
        for (Index k = 0; k < kw; ++k) {
          const double* xk = X + static_cast<size_t>(k) * ld;
          s += xk[rb] * xk[c] * (S ? S[k] : 1.0);
        }
        T[static_cast<size_t>(c) * ldT + rb] = s;
      }
    }
  }
}

// Elimination tree from the row lists (Liu's algorithm with path compression).
std::vector<Index> etree(Index n, const std::vector<Index>& rstart, const std::vector<Index>& rcols) {
  std::vector<Index> parent(static_cast<size_t>(n), -1), anc(static_cast<size_t>(n), -1);
  for (Index i = 0; i < n; ++i) {
    for (Index k = rstart[i]; k < rstart[i + 1]; ++k) {
      Index r = rcols[k];
      while (anc[r] != -1 && anc[r] != i) {
        const Index t = anc[r];
        anc[r] = i;
        r = t;
      }
      if (anc[r] == -1) {
        anc[r] = i;
        parent[r] = i;
      }
    }
  }
  return parent;
}

}  // namespace

SparseCholesky::SparseCholesky() = default;
SparseCholesky::~SparseCholesky() = default;
SparseCholesky::SparseCholesky(SparseCholesky&&) noexcept = default;
SparseCholesky& SparseCholesky::operator=(SparseCholesky&&) noexcept = default;

void SparseCholesky::syrk_rows(const double* X, size_t ld, Index kw, Index nc, Index nr, double* T, size_t ldT,
                               const double* S) {
  const double work = static_cast<double>(kw) * nc * nr;
  if (!pool_ || work < 4e5 || nr < 64) {
    syrk_lower(X, ld, kw, nc, 0, nr, T, ldT, S);
    return;
  }
  // Row chunks (multiples of 8 rows); the rows above the first columns cost less, so use more
  // chunks than threads.
  const Index chunks = std::min<Index>(4 * pool_->size(), nr / 32);
  const Index step = ((nr + chunks - 1) / chunks + 7) / 8 * 8;
  const int tasks = static_cast<int>((nr + step - 1) / step);
  pool_->run(tasks, [&](int t) {
    const Index r0 = static_cast<Index>(t) * step, r1 = std::min(nr, r0 + step);
    syrk_lower(X, ld, kw, nc, r0, r1, T, ldT, S);
  });
}

bool SparseCholesky::analyze(Index n, const std::vector<Index>& start, const std::vector<Index>& index,
                             const std::vector<Index>* perm, const std::vector<int8_t>* signs) {
  n_ = n;
  if (n < 0 || static_cast<Index>(start.size()) != n + 1) return false;
  nnz_in_ = start[n];
  if (static_cast<Index>(index.size()) < nnz_in_) return false;
  std::vector<Index> diag_entry(static_cast<size_t>(n), -1);
  for (Index c = 0; c < n; ++c) {
    for (Index k = start[c]; k < start[c + 1]; ++k) {
      if (index[k] < c || index[k] >= n) return false;
      if (index[k] == c && diag_entry[c] < 0) diag_entry[c] = k;
    }
  }
  for (Index c = 0; c < n; ++c) {
    if (diag_entry[c] < 0) return false;
  }
  if (perm) {
    perm_ = *perm;
    if (static_cast<Index>(perm_.size()) != n) return false;
  } else {
    perm_ = amd_order(n, start, index, stop);
    if (static_cast<Index>(perm_.size()) != n) return false;
  }
  pinv_.assign(static_cast<size_t>(n), -1);
  for (Index k = 0; k < n; ++k) {
    if (perm_[k] < 0 || perm_[k] >= n || pinv_[perm_[k]] >= 0) return false;
    pinv_[perm_[k]] = k;
  }

  // Postorder the elimination tree (supernodes become contiguous), then recompute it.
  std::vector<Index> rstart, rcols;
  permuted_rows(n, start, index, pinv_, rstart, rcols);
  std::vector<Index> parent = etree(n, rstart, rcols);
  {
    std::vector<Index> head(static_cast<size_t>(n), -1), nxt(static_cast<size_t>(n), -1);
    for (Index j = n - 1; j >= 0; --j) {
      if (parent[j] >= 0) {
        nxt[j] = head[parent[j]];
        head[parent[j]] = j;
      }
    }
    std::vector<Index> post;
    post.reserve(static_cast<size_t>(n));
    std::vector<Index> stack;
    for (Index r = 0; r < n; ++r) {
      if (parent[r] >= 0) continue;
      stack.push_back(r);
      while (!stack.empty()) {
        const Index v = stack.back();
        if (head[v] >= 0) {
          const Index c = head[v];
          head[v] = nxt[c];
          stack.push_back(c);
        } else {
          stack.pop_back();
          post.push_back(v);
        }
      }
    }
    std::vector<Index> p2(static_cast<size_t>(n));
    for (Index k = 0; k < n; ++k) p2[k] = perm_[post[k]];
    perm_ = std::move(p2);
    for (Index k = 0; k < n; ++k) pinv_[perm_[k]] = k;
  }
  permuted_rows(n, start, index, pinv_, rstart, rcols);
  parent = etree(n, rstart, rcols);

  // Column counts of L from the row subtrees.
  std::vector<long long> cnt(static_cast<size_t>(n), 1);
  too_large_ = false;
  {
    std::vector<Index> visited(static_cast<size_t>(n), -1);
    long long total = n;
    for (Index i = 0; i < n; ++i) {
      visited[i] = i;
      for (Index k = rstart[i]; k < rstart[i + 1]; ++k) {
        for (Index j = rcols[k]; visited[j] != i; j = parent[j]) {
          visited[j] = i;
          ++cnt[j];
          ++total;
        }
      }
      if (nnz_limit > 0 && total > nnz_limit) {
        too_large_ = true;
        return false;
      }
    }
  }
  flops_ = 0.0;
  for (Index j = 0; j < n; ++j) flops_ += static_cast<double>(cnt[j]) * static_cast<double>(cnt[j]);

  // Fundamental supernodes, then relaxed amalgamation of a child into its adjacent parent.
  struct Group {
    Index first, last;  // columns [first, last]
    long long rows, nnz;
  };
  std::vector<Group> groups;
  {
    std::vector<Group> fund;
    for (Index j = 0; j < n; ++j) {
      if (j > 0 && parent[j - 1] == j && cnt[j - 1] == cnt[j] + 1) {
        fund.back().last = j;
        fund.back().nnz += cnt[j];
      } else {
        fund.push_back({j, j, cnt[j], cnt[j]});
      }
    }
    auto stored = [](long long w, long long rows) { return w * rows - w * (w - 1) / 2; };
    for (const Group& g : fund) {
      if (!groups.empty()) {
        Group& cur = groups.back();
        if (parent[cur.last] == g.first) {
          const long long w = g.last - cur.first + 1;
          const long long rows = (cur.last - cur.first + 1) + g.rows;
          const long long st = stored(w, rows), nz = cur.nnz + g.nnz;
          if (w <= relax_small || (w <= relax_cols && static_cast<double>(st - nz) <= relax_zeros * static_cast<double>(st))) {
            cur.last = g.last;
            cur.rows = rows;
            cur.nnz = nz;
            continue;
          }
        }
      }
      groups.push_back(g);
    }
  }
  const Index ns = static_cast<Index>(groups.size());
  sn_first_.assign(static_cast<size_t>(ns) + 1, n);
  sn_of_col_.assign(static_cast<size_t>(n), 0);
  for (Index s = 0; s < ns; ++s) {
    sn_first_[s] = groups[s].first;
    for (Index j = groups[s].first; j <= groups[s].last; ++j) sn_of_col_[j] = s;
  }

  // Row structures: own columns, rows of P M P^T below the supernode, and the rows of the child
  // supernodes below it.
  std::vector<Index> cstart(static_cast<size_t>(n) + 1, 0), crows;  // strictly lower, column-wise
  for (Index i = 0; i < n; ++i)
    for (Index k = rstart[i]; k < rstart[i + 1]; ++k) ++cstart[rcols[k] + 1];
  for (Index j = 0; j < n; ++j) cstart[j + 1] += cstart[j];
  crows.resize(static_cast<size_t>(cstart[n]));
  {
    std::vector<Index> fill(cstart.begin(), cstart.end() - 1);
    for (Index i = 0; i < n; ++i)
      for (Index k = rstart[i]; k < rstart[i + 1]; ++k) crows[fill[rcols[k]]++] = i;
  }
  std::vector<Index> child_head(static_cast<size_t>(ns), -1), child_next(static_cast<size_t>(ns), -1);
  std::vector<Index> mark(static_cast<size_t>(n), -1);
  sn_rstart_.assign(1, 0);
  sn_rows_.clear();
  std::vector<Index> off;
  for (Index s = 0; s < ns; ++s) {
    const Index f = sn_first_[s], e = sn_first_[s + 1];
    off.clear();
    for (Index j = f; j < e; ++j) {
      for (Index k = cstart[j]; k < cstart[j + 1]; ++k) {
        const Index r = crows[k];
        if (r >= e && mark[r] != s) {
          mark[r] = s;
          off.push_back(r);
        }
      }
    }
    for (Index c = child_head[s]; c >= 0; c = child_next[c]) {
      for (Index k = sn_rstart_[c] + (sn_first_[c + 1] - sn_first_[c]); k < sn_rstart_[c + 1]; ++k) {
        const Index r = sn_rows_[k];
        if (r >= e && mark[r] != s) {
          mark[r] = s;
          off.push_back(r);
        }
      }
    }
    std::sort(off.begin(), off.end());
    for (Index j = f; j < e; ++j) sn_rows_.push_back(j);
    sn_rows_.insert(sn_rows_.end(), off.begin(), off.end());
    sn_rstart_.push_back(static_cast<Index>(sn_rows_.size()));
    if (!off.empty()) {
      const Index p = sn_of_col_[off[0]];
      child_next[s] = child_head[p];
      child_head[p] = s;
    }
  }
  sn_xstart_.assign(static_cast<size_t>(ns) + 1, 0);
  factor_nnz_ = 0;
  for (Index s = 0; s < ns; ++s) {
    const size_t w = static_cast<size_t>(sn_first_[s + 1] - sn_first_[s]);
    const size_t rows = static_cast<size_t>(sn_rstart_[s + 1] - sn_rstart_[s]);
    sn_xstart_[s + 1] = sn_xstart_[s] + w * rows;
    factor_nnz_ += static_cast<long long>(w * rows - w * (w - 1) / 2);
  }
  lx_.assign(sn_xstart_[ns], 0.0);

  // Where each input entry goes.
  entry_pos_.assign(static_cast<size_t>(nnz_in_), 0);
  diag_entry_.assign(static_cast<size_t>(n), -1);
  for (Index c = 0; c < n; ++c) {
    for (Index k = start[c]; k < start[c + 1]; ++k) {
      const Index pr = pinv_[index[k]], pc = pinv_[c];
      const Index i = std::max(pr, pc), j = std::min(pr, pc);
      const Index s = sn_of_col_[j];
      const Index* rb = sn_rows_.data() + sn_rstart_[s];
      const Index* re = sn_rows_.data() + sn_rstart_[s + 1];
      const Index lr = static_cast<Index>(std::lower_bound(rb, re, i) - rb);
      const size_t rows = static_cast<size_t>(re - rb);
      entry_pos_[k] = sn_xstart_[s] + static_cast<size_t>(j - sn_first_[s]) * rows + static_cast<size_t>(lr);
    }
  }
  diag_.assign(static_cast<size_t>(n), 0.0);
  psign_.clear();
  if (signs) {
    psign_.assign(static_cast<size_t>(n), 1.0);
    for (Index k = 0; k < n; ++k) psign_[k] = (*signs)[perm_[k]] < 0 ? -1.0 : 1.0;
  }
  return true;
}

bool SparseCholesky::factorize(const std::vector<double>& value) {
  if (threads > 1 && (!pool_ || pool_->size() != threads) && flops_ > 1e7) pool_ = std::make_unique<WorkerPool>(threads);
  if (threads <= 1) pool_.reset();
  std::fill(lx_.begin(), lx_.end(), 0.0);
  for (Index k = 0; k < nnz_in_; ++k) lx_[entry_pos_[k]] += value[k];
  const Index ns = num_supernodes();
  for (Index s = 0; s < ns; ++s) {
    const size_t rows = static_cast<size_t>(sn_rstart_[s + 1] - sn_rstart_[s]);
    for (Index j = sn_first_[s]; j < sn_first_[s + 1]; ++j) {
      const size_t c = static_cast<size_t>(j - sn_first_[s]);
      diag_[j] = lx_[sn_xstart_[s] + c * rows + c];
    }
  }
  dropped_ = 0;
  std::vector<Index> relmap(static_cast<size_t>(n_), 0), link_head(static_cast<size_t>(ns), -1),
      link_next(static_cast<size_t>(ns), -1), kpos(static_cast<size_t>(ns), 0);
  std::vector<double> work;
  for (Index s = 0; s < ns; ++s) {
    if (stop && (s & 63) == 0 && stop->load(std::memory_order_relaxed)) return false;
    factor_supernode(s, relmap, work, link_head, link_next, kpos);
  }
  for (double v : lx_) {
    if (!std::isfinite(v)) return false;
  }
  return true;
}

void SparseCholesky::factor_supernode(Index J, std::vector<Index>& relmap, std::vector<double>& work,
                                      std::vector<Index>& link_head, std::vector<Index>& link_next,
                                      std::vector<Index>& kpos) {
  const Index fJ = sn_first_[J], eJ = sn_first_[J + 1];
  const Index wJ = eJ - fJ;
  const Index* RJ = sn_rows_.data() + sn_rstart_[J];
  const Index nrJ = sn_rstart_[J + 1] - sn_rstart_[J];
  double* LJ = lx_.data() + sn_xstart_[J];
  for (Index r = 0; r < nrJ; ++r) relmap[RJ[r]] = r;

  // Updates from the descendant supernodes K with rows in [fJ, eJ).
  Index K = link_head[J];
  link_head[J] = -1;
  while (K >= 0) {
    const Index nextK = link_next[K];
    const Index* RK = sn_rows_.data() + sn_rstart_[K];
    const Index nrK = sn_rstart_[K + 1] - sn_rstart_[K];
    const Index wK = sn_first_[K + 1] - sn_first_[K];
    const double* LK = lx_.data() + sn_xstart_[K];
    const Index p = kpos[K];
    Index q = p;
    while (q < nrK && RK[q] < eJ) ++q;
    const Index nc = q - p, nr = nrK - p;
    // T = LK[p:nrK, :] * LK[p:q, :]^T (nr x nc, lower part).
    work.resize(static_cast<size_t>(nr) * static_cast<size_t>(nc));
    syrk_rows(LK + p, static_cast<size_t>(nrK), wK, nc, nr, work.data(), static_cast<size_t>(nr),
              psign_.empty() ? nullptr : psign_.data() + sn_first_[K]);
    for (Index c = 0; c < nc; ++c) {
      double* dst = LJ + static_cast<size_t>(RK[p + c] - fJ) * nrJ;
      const double* T0 = work.data() + static_cast<size_t>(c) * nr;
      for (Index r = c; r < nr; ++r) dst[relmap[RK[p + r]]] -= T0[r];
    }
    kpos[K] = q;
    if (q < nrK) {
      const Index S2 = sn_of_col_[RK[q]];
      link_next[K] = link_head[S2];
      link_head[S2] = K;
    }
    K = nextK;
  }

  // Dense factorization of the supernode's trapezoid: panels of kPanel columns; each panel is
  // first updated by all columns left of it (one blocked product), then factored column by column.
  constexpr Index kPanel = 32;
  for (Index j0 = 0; j0 < wJ; j0 += kPanel) {
    const Index j1 = std::min(wJ, j0 + kPanel), pw = j1 - j0, nr = nrJ - j0;
    if (j0 > 0) {
      work.resize(static_cast<size_t>(nr) * static_cast<size_t>(pw));
      syrk_rows(LJ + j0, static_cast<size_t>(nrJ), j0, pw, nr, work.data(), static_cast<size_t>(nr),
                psign_.empty() ? nullptr : psign_.data() + fJ);
      for (Index c = 0; c < pw; ++c) {
        double* dst = LJ + static_cast<size_t>(j0 + c) * nrJ + j0;
        const double* T0 = work.data() + static_cast<size_t>(c) * nr;
        for (Index r = c; r < nr; ++r) dst[r] -= T0[r];
      }
    }
    for (Index j = j0; j < j1; ++j) {
      double* cj = LJ + static_cast<size_t>(j) * nrJ;
      for (Index k = j0; k < j; ++k) {
        const double* ck = LJ + static_cast<size_t>(k) * nrJ;
        const double l = psign_.empty() ? ck[j] : ck[j] * psign_[fJ + k];
        if (l == 0.0) continue;
        for (Index r = j; r < nrJ; ++r) cj[r] -= ck[r] * l;
      }
      const double sgn = psign_.empty() ? 1.0 : psign_[fJ + j];
      const double d = sgn * cj[j];
      if (!(d > pivot_tol * std::abs(diag_[fJ + j])) || !(d > 1e-300)) {
        // Dropped pivot: a huge diagonal makes this component of the solution ~0.
        ++dropped_;
        cj[j] = 1e64;
        for (Index r = j + 1; r < nrJ; ++r) cj[r] *= 1e-64;
        continue;
      }
      const double sd = std::sqrt(d), inv = sgn / sd;  // L_rj = K_rj / (s_j L_jj)
      cj[j] = sd;
      for (Index r = j + 1; r < nrJ; ++r) cj[r] *= inv;
    }
  }
  if (nrJ > wJ) {
    kpos[J] = wJ;
    const Index S2 = sn_of_col_[RJ[wJ]];
    link_next[J] = link_head[S2];
    link_head[S2] = J;
  }
}

void SparseCholesky::solve(std::vector<double>& b) const { solve(b.data()); }

void SparseCholesky::solve(double* b) const {
  std::vector<double> x(static_cast<size_t>(n_));
  for (Index k = 0; k < n_; ++k) x[k] = b[perm_[k]];
  const Index ns = num_supernodes();
  for (Index J = 0; J < ns; ++J) {
    const Index fJ = sn_first_[J], wJ = sn_first_[J + 1] - fJ;
    const Index* RJ = sn_rows_.data() + sn_rstart_[J];
    const Index nrJ = sn_rstart_[J + 1] - sn_rstart_[J];
    const double* LJ = lx_.data() + sn_xstart_[J];
    for (Index j = 0; j < wJ; ++j) {
      const double* cj = LJ + static_cast<size_t>(j) * nrJ;
      const double xj = x[fJ + j] / cj[j];
      x[fJ + j] = xj;
      if (xj == 0.0) continue;
      for (Index r = j + 1; r < nrJ; ++r) x[RJ[r]] -= cj[r] * xj;
    }
  }
  if (!psign_.empty()) {
    for (Index k = 0; k < n_; ++k) x[k] *= psign_[k];
  }
  for (Index J = ns - 1; J >= 0; --J) {
    const Index fJ = sn_first_[J], wJ = sn_first_[J + 1] - fJ;
    const Index* RJ = sn_rows_.data() + sn_rstart_[J];
    const Index nrJ = sn_rstart_[J + 1] - sn_rstart_[J];
    const double* LJ = lx_.data() + sn_xstart_[J];
    for (Index j = wJ - 1; j >= 0; --j) {
      const double* cj = LJ + static_cast<size_t>(j) * nrJ;
      double s = x[fJ + j];
      for (Index r = j + 1; r < nrJ; ++r) s -= cj[r] * x[RJ[r]];
      x[fJ + j] = s / cj[j];
    }
  }
  for (Index k = 0; k < n_; ++k) b[perm_[k]] = x[k];
}

}  // namespace milp
