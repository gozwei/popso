#include "milp/lp/lu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace milp {

namespace {

// Doubly linked lists of items bucketed by their (nonzero) count.
class CountBuckets {
 public:
  void init(Index n) {
    head_.assign(n + 2, -1);
    next_.assign(n, -1);
    prev_.assign(n, -1);
    count_.assign(n, -1);
  }
  void insert(Index i, Index c) {
    count_[i] = c;
    prev_[i] = -1;
    next_[i] = head_[c];
    if (head_[c] >= 0) prev_[head_[c]] = i;
    head_[c] = i;
  }
  void remove(Index i) {
    Index c = count_[i];
    if (c < 0) return;
    if (prev_[i] >= 0) next_[prev_[i]] = next_[i]; else head_[c] = next_[i];
    if (next_[i] >= 0) prev_[next_[i]] = prev_[i];
    count_[i] = -1;
  }
  void move(Index i, Index c) {
    if (count_[i] == c) return;
    remove(i);
    insert(i, c);
  }
  Index head(Index c) const { return head_[c]; }
  Index next(Index i) const { return next_[i]; }

 private:
  std::vector<Index> head_, next_, prev_, count_;
};

constexpr int kSearch = 4;  // Markowitz search: candidate rows+columns examined per pivot

bool lu_stats_enabled() {
  static const bool on = [] {
    const char* s = std::getenv("MILP_LU_STATS");
    return s && *s && *s != '0';
  }();
  return on;
}

// Dot products with independent partial sums (FP reductions are not vectorized
// or reordered by the compiler otherwise, leaving a latency-bound chain).
double dot(const double* a, const double* b, Index n) {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  Index i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; ++i) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

double dot_gather(const double* val, const Index* idx, const double* v, Index n) {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  Index i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += val[i] * v[idx[i]];
    s1 += val[i + 1] * v[idx[i + 1]];
    s2 += val[i + 2] * v[idx[i + 2]];
    s3 += val[i + 3] * v[idx[i + 3]];
  }
  for (; i < n; ++i) s0 += val[i] * v[idx[i]];
  return (s0 + s1) + (s2 + s3);
}

double now_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Process-wide totals, printed at exit with MILP_LU_STATS=1.
struct LuTotals {
  std::atomic<long long> factors{0}, updates{0}, factor_us{0};
  ~LuTotals() {
    if (!lu_stats_enabled() || factors == 0) return;
    std::fprintf(stderr, "[lu] total: %lld factorizations, %.3f s (avg %.2f ms), %lld updates\n", factors.load(),
                 factor_us * 1e-6, factor_us * 1e-3 / factors, updates.load());
  }
};
LuTotals g_totals;

// Active kernel submatrix: rows hold (column, value) pairs, columns hold row
// indices only. Each row/column owns a slot range [beg, beg+cap) in a shared
// array; a list that outgrows its slot is moved to the end of the array, and
// the array is compacted when too much space is wasted.
struct Kernel {
  std::vector<Index> rbeg, rlen, rcap, ridx;
  std::vector<double> rval, rmax;  // rmax < 0: row maximum unknown
  std::vector<Index> cbeg, clen, ccap, cidx;
  Index rwaste = 0, cwaste = 0;

  void reserve_row(Index i, Index need) {
    if (rcap[i] >= need) return;
    if (rwaste > static_cast<Index>(ridx.size()) / 2) compact_rows();
    const Index cap = need + need / 2 + 4, nb = static_cast<Index>(ridx.size());
    ridx.resize(nb + cap);
    rval.resize(nb + cap);
    std::copy_n(ridx.begin() + rbeg[i], rlen[i], ridx.begin() + nb);
    std::copy_n(rval.begin() + rbeg[i], rlen[i], rval.begin() + nb);
    rwaste += rcap[i];
    rbeg[i] = nb;
    rcap[i] = cap;
  }
  void reserve_col(Index j, Index need) {
    if (ccap[j] >= need) return;
    if (cwaste > static_cast<Index>(cidx.size()) / 2) compact_cols();
    const Index cap = need + need / 2 + 4, nb = static_cast<Index>(cidx.size());
    cidx.resize(nb + cap);
    std::copy_n(cidx.begin() + cbeg[j], clen[j], cidx.begin() + nb);
    cwaste += ccap[j];
    cbeg[j] = nb;
    ccap[j] = cap;
  }
  void compact_rows() {
    std::vector<Index> ni;
    std::vector<double> nv;
    ni.reserve(ridx.size() - rwaste);
    nv.reserve(ridx.size() - rwaste);
    for (Index i = 0; i < static_cast<Index>(rbeg.size()); ++i) {
      if (rcap[i] == 0) continue;
      const Index nb = static_cast<Index>(ni.size());
      ni.insert(ni.end(), ridx.begin() + rbeg[i], ridx.begin() + rbeg[i] + rcap[i]);
      nv.insert(nv.end(), rval.begin() + rbeg[i], rval.begin() + rbeg[i] + rcap[i]);
      rbeg[i] = nb;
    }
    ridx.swap(ni);
    rval.swap(nv);
    rwaste = 0;
  }
  void compact_cols() {
    std::vector<Index> ni;
    ni.reserve(cidx.size() - cwaste);
    for (Index j = 0; j < static_cast<Index>(cbeg.size()); ++j) {
      if (ccap[j] == 0) continue;
      const Index nb = static_cast<Index>(ni.size());
      ni.insert(ni.end(), cidx.begin() + cbeg[j], cidx.begin() + cbeg[j] + ccap[j]);
      cbeg[j] = nb;
    }
    cidx.swap(ni);
    cwaste = 0;
  }
  Index find_in_row(Index i, Index c) const {
    for (Index k = rbeg[i], e = rbeg[i] + rlen[i]; k < e; ++k)
      if (ridx[k] == c) return k;
    return -1;
  }
  void remove_from_col(Index j, Index i) {
    Index* p = cidx.data() + cbeg[j];
    const Index n = clen[j];
    for (Index k = 0; k < n; ++k) {
      if (p[k] == i) {
        p[k] = p[n - 1];
        --clen[j];
        return;
      }
    }
  }
  double row_max(Index i) {
    if (rmax[i] < 0.0) {
      double mx = 0.0;
      for (Index k = rbeg[i], e = rbeg[i] + rlen[i]; k < e; ++k) mx = std::max(mx, std::abs(rval[k]));
      rmax[i] = mx;
    }
    return rmax[i];
  }
};

// Per-thread scratch of factorize()/finish(): node LPs refactor small bases very
// often, so the buffers are reused instead of reallocated.
struct FactorWork {
  std::vector<Index> cs, ci, rs, ri, fill, rcnt, ccnt, queue, pidx, seen, prj, colc;
  std::vector<double> cv, rv, prv;
  std::vector<char> row_done, col_done;
  std::vector<Index> pos, cnt, grp_pos;
  Kernel K;
  CountBuckets rbuck, cbuck;
};
FactorWork& factor_work() {
  thread_local FactorWork w;
  return w;
}

}  // namespace

Index LuFactor::factorize(Index m, const std::vector<Index>& start, const std::vector<Index>& index,
                          const std::vector<double>& value, std::vector<std::pair<Index, Index>>* replaced) {
  const double t0 = lu_stats_enabled() ? now_seconds() : 0.0;
  m_ = m;
  piv_row_.clear();
  piv_col_.clear();
  piv_val_.clear();
  l_step_.clear();
  dense_begin_ = dense_n_ = 0;
  dense_lu_.clear();
  l_start_.assign(1, 0);
  l_index_.clear();
  l_value_.clear();
  u_row_start_.assign(1, 0);
  u_row_index_.clear();
  u_row_value_.clear();
  eta_pos_.clear();
  eta_start_.assign(1, 0);
  eta_index_.clear();
  eta_value_.clear();
  eta_pivot_.clear();
  ft_row_.clear();
  ft_col_start_.clear();
  ft_col_len_.clear();
  ft_col_index_.clear();
  ft_col_value_.clear();
  r_row_.clear();
  r_start_.assign(1, 0);
  r_index_.clear();
  r_value_.clear();
  spike_ok_ = ft_off_ = ft_unstable_ = false;
  spike_all_ = false;  // spike_ is zeroed in finish()
  spike_idx_.clear();
  if (replaced) replaced->clear();

  // Column-wise and row-wise copies of B without explicit zeros.
  FactorWork& W = factor_work();
  std::vector<Index>&cs = W.cs, &ci = W.ci, &rs = W.rs, &ri = W.ri;
  std::vector<double>&cv = W.cv, &rv = W.rv;
  cs.assign(m + 1, 0);
  rs.assign(m + 1, 0);
  ci.clear();
  cv.clear();
  for (Index c = 0; c < m; ++c) {
    for (Index p = start[c]; p < start[c + 1]; ++p) {
      if (value[p] == 0.0) continue;
      ci.push_back(index[p]);
      cv.push_back(value[p]);
      ++rs[index[p] + 1];
    }
    cs[c + 1] = static_cast<Index>(ci.size());
  }
  const Index basis_nnz = cs[m];
  double work = 4.0 * basis_nnz + 8.0 * m;  // factorization work estimate (refactor heuristic)
  for (Index i = 0; i < m; ++i) rs[i + 1] += rs[i];
  ri.resize(basis_nnz);
  rv.resize(basis_nnz);
  {
    std::vector<Index>& fill = W.fill;
    fill.assign(rs.begin(), rs.end() - 1);
    for (Index c = 0; c < m; ++c) {
      for (Index p = cs[c]; p < cs[c + 1]; ++p) {
        const Index q = fill[ci[p]]++;
        ri[q] = c;
        rv[q] = cv[p];
      }
    }
  }
  std::vector<Index>&rcnt = W.rcnt, &ccnt = W.ccnt;
  rcnt.resize(m);
  ccnt.resize(m);
  for (Index i = 0; i < m; ++i) rcnt[i] = rs[i + 1] - rs[i];
  for (Index c = 0; c < m; ++c) ccnt[c] = cs[c + 1] - cs[c];
  std::vector<char>&row_done = W.row_done, &col_done = W.col_done;
  row_done.assign(m, 0);
  col_done.assign(m, 0);

  auto push_pivot = [&](Index r, Index c, double pv) {
    piv_row_.push_back(r);
    piv_col_.push_back(c);
    piv_val_.push_back(pv);
    row_done[r] = 1;
    col_done[c] = 1;
  };
  auto end_l = [&]() {
    if (static_cast<Index>(l_index_.size()) > l_start_.back()) {
      l_step_.push_back(static_cast<Index>(piv_row_.size()) - 1);
      l_start_.push_back(static_cast<Index>(l_index_.size()));
    }
  };

  // ---- Triangular phase ----
  // Column singleton (r, c): no other active row has an entry in column c, so
  // the step needs no elimination; row r's active entries become the U row.
  // Row singleton (r, c): row r has no other active entry, so eliminating
  // column c changes nothing else; the column entries become L multipliers.
  // Active entries therefore keep their original values throughout.
  std::vector<Index>& queue = W.queue;
  queue.clear();
  for (Index c = 0; c < m; ++c)
    if (ccnt[c] == 1) queue.push_back(c);
  while (!queue.empty()) {
    const Index c = queue.back();
    queue.pop_back();
    if (col_done[c] || ccnt[c] != 1) continue;
    Index r = -1;
    double pv = 0.0;
    for (Index p = cs[c]; p < cs[c + 1]; ++p) {
      if (!row_done[ci[p]]) {
        r = ci[p], pv = cv[p];
        break;
      }
    }
    if (std::abs(pv) <= pivot_tolerance) continue;  // left to the kernel
    push_pivot(r, c, pv);
    for (Index p = rs[r]; p < rs[r + 1]; ++p) {
      const Index j = ri[p];
      if (col_done[j]) continue;
      u_row_index_.push_back(j);
      u_row_value_.push_back(rv[p]);
      if (--ccnt[j] == 1) queue.push_back(j);
    }
    u_row_start_.push_back(static_cast<Index>(u_row_index_.size()));
  }
  for (Index i = 0; i < m; ++i) {
    // Recount rows: column singleton steps removed columns.
    if (row_done[i]) continue;
    Index cnt = 0;
    for (Index p = rs[i]; p < rs[i + 1]; ++p) cnt += !col_done[ri[p]];
    rcnt[i] = cnt;
    if (cnt == 1) queue.push_back(i);
  }
  while (!queue.empty()) {
    const Index r = queue.back();
    queue.pop_back();
    if (row_done[r] || rcnt[r] != 1) continue;
    Index c = -1;
    double pv = 0.0;
    for (Index p = rs[r]; p < rs[r + 1]; ++p) {
      if (!col_done[ri[p]]) {
        c = ri[p], pv = rv[p];
        break;
      }
    }
    if (std::abs(pv) <= pivot_tolerance) continue;
    push_pivot(r, c, pv);
    for (Index p = cs[c]; p < cs[c + 1]; ++p) {
      const Index i = ci[p];
      if (row_done[i]) continue;
      l_index_.push_back(i);
      l_value_.push_back(cv[p] / pv);
      if (--rcnt[i] == 1) queue.push_back(i);
    }
    end_l();
    u_row_start_.push_back(static_cast<Index>(u_row_index_.size()));
  }
  const Index tri_steps = static_cast<Index>(piv_row_.size());

  // ---- Markowitz kernel ----
  Kernel& K = W.K;
  Index kernel_dim = 0, kernel_nnz = 0, dense_kernel = 0;
  if (tri_steps < m) {
    K.ridx.clear();
    K.rval.clear();
    K.cidx.clear();
    K.rwaste = K.cwaste = 0;
    K.rbeg.assign(m, 0);
    K.rlen.assign(m, 0);
    K.rcap.assign(m, 0);
    K.rmax.assign(m, -1.0);
    K.cbeg.assign(m, 0);
    K.clen.assign(m, 0);
    K.ccap.assign(m, 0);
    for (Index i = 0; i < m; ++i) {
      if (row_done[i]) continue;
      ++kernel_dim;
      for (Index p = rs[i]; p < rs[i + 1]; ++p) {
        const Index j = ri[p];
        if (col_done[j]) continue;
        ++K.rlen[i];
        ++K.clen[j];
      }
    }
    for (Index i = 0; i < m; ++i) {
      if (row_done[i]) continue;
      K.rbeg[i] = static_cast<Index>(K.ridx.size());
      K.rcap[i] = K.rlen[i] + 4;
      K.ridx.resize(K.ridx.size() + K.rcap[i]);
      K.rval.resize(K.ridx.size());
      Index w = K.rbeg[i];
      for (Index p = rs[i]; p < rs[i + 1]; ++p) {
        if (col_done[ri[p]]) continue;
        K.ridx[w] = ri[p];
        K.rval[w++] = rv[p];
      }
      kernel_nnz += K.rlen[i];
    }
    for (Index j = 0; j < m; ++j) {
      if (col_done[j]) continue;
      K.cbeg[j] = static_cast<Index>(K.cidx.size());
      K.ccap[j] = K.clen[j] + 4;
      K.cidx.resize(K.cidx.size() + K.ccap[j]);
      K.clen[j] = 0;
    }
    for (Index i = 0; i < m; ++i) {
      if (row_done[i]) continue;
      for (Index k = K.rbeg[i]; k < K.rbeg[i] + K.rlen[i]; ++k) {
        const Index j = K.ridx[k];
        K.cidx[K.cbeg[j] + K.clen[j]++] = i;
      }
    }

    CountBuckets &rbuck = W.rbuck, &cbuck = W.cbuck;
    rbuck.init(m);
    cbuck.init(m);
    for (Index i = 0; i < m; ++i) {
      if (!row_done[i]) rbuck.insert(i, K.rlen[i]);
      if (!col_done[i]) cbuck.insert(i, K.clen[i]);
    }
    std::vector<Index>&pidx = W.pidx, &seen = W.seen, &prj = W.prj, &colc = W.colc;
    std::vector<double>& prv = W.prv;
    pidx.assign(m, -1);
    seen.assign(m, -1);
    Index stamp = 0;

    // Dense LU with partial pivoting (in place) for a remaining kernel that has
    // filled in. Empty rows/columns are left out (they end up replaced). A
    // nonsingular block is kept as a dense matrix for the solves; otherwise its
    // factors are emitted into the sparse structures.
    auto dense_phase = [&]() {
      std::vector<Index> drow, dcol, cmap(m, -1);
      for (Index i = 0; i < m; ++i) {
        if (!row_done[i] && K.rlen[i] > 0) drow.push_back(i);
        if (!col_done[i] && K.clen[i] > 0) cmap[i] = static_cast<Index>(dcol.size()), dcol.push_back(i);
      }
      const size_t nr = drow.size(), nc = dcol.size();
      std::vector<double> D(nr * nc, 0.0);
      for (size_t a = 0; a < nr; ++a) {
        const Index i = drow[a];
        for (Index k = K.rbeg[i], e = K.rbeg[i] + K.rlen[i]; k < e; ++k) D[a * nc + cmap[K.ridx[k]]] = K.rval[k];
      }
      // Blocked right-looking elimination: panels of kPanel columns are factored
      // with partial pivoting, then the pivot rows' remaining parts (U12) and the
      // trailing rows are updated panel-wise (each row stays in cache).
      constexpr size_t kPanel = 32;
      std::vector<size_t> prow_of(nc, nr), rank(nr, nc), ppiv, pcol;
      size_t npiv = 0;
      auto axpy = [](double* dst, const double* src, double l, size_t from, size_t to) {
        for (size_t j = from; j < to; ++j) dst[j] -= l * src[j];
      };
      for (size_t b0 = 0; b0 < nc; b0 += kPanel) {
        const size_t b1 = std::min(nc, b0 + kPanel);
        ppiv.clear();
        pcol.clear();
        for (size_t b = b0; b < b1; ++b) {
          size_t pa = nr;
          double mx = pivot_tolerance;
          for (size_t a = 0; a < nr; ++a) {
            if (rank[a] == nc && std::abs(D[a * nc + b]) > mx) mx = std::abs(D[a * nc + b]), pa = a;
          }
          work += static_cast<double>(nr);
          if (pa == nr) continue;  // dependent column: replaced below
          rank[pa] = b;
          prow_of[b] = pa;
          ++npiv;
          ppiv.push_back(pa);
          pcol.push_back(b);
          const double* prow = &D[pa * nc];
          const double pv = prow[b];
          for (size_t a = 0; a < nr; ++a) {
            if (rank[a] != nc || D[a * nc + b] == 0.0) continue;
            double* row = &D[a * nc];
            const double l = row[b] / pv;
            row[b] = l;  // multiplier stored in place
            axpy(row, prow, l, b + 1, b1);
          }
        }
        if (b1 == nc || ppiv.empty()) continue;
        // U12: pivot rows of this panel, in pivot order.
        for (size_t k = 0; k < ppiv.size(); ++k) {
          double* row = &D[ppiv[k] * nc];
          for (size_t j = 0; j < k; ++j) {
            const double l = row[pcol[j]];
            if (l != 0.0) axpy(row, &D[ppiv[j] * nc], l, b1, nc);
          }
        }
        // Trailing update of the rows not pivoted yet.
        for (size_t a = 0; a < nr; ++a) {
          if (rank[a] != nc) continue;
          double* row = &D[a * nc];
          for (size_t j = 0; j < ppiv.size(); ++j) {
            const double l = row[pcol[j]];
            if (l == 0.0) continue;
            axpy(row, &D[ppiv[j] * nc], l, b1, nc);
            work += static_cast<double>(nc - b1);
          }
        }
      }
      if (npiv == nr && npiv == nc) {
        const size_t n = nc;
        dense_begin_ = static_cast<Index>(piv_row_.size());
        dense_n_ = static_cast<Index>(n);
        dense_lu_.assign(n * n, 0.0);
        for (size_t b = 0; b < n; ++b) {
          push_pivot(drow[prow_of[b]], dcol[b], D[prow_of[b] * nc + b]);
          u_row_start_.push_back(static_cast<Index>(u_row_index_.size()));
          double* col = &dense_lu_[b * n];
          for (size_t a = 0; a < n; ++a)
            if (a != b) col[a] = D[prow_of[a] * nc + b];
        }
        return;
      }
      for (size_t b = 0; b < nc; ++b) {
        const size_t pa = prow_of[b];
        if (pa == nr) continue;
        push_pivot(drow[pa], dcol[b], D[pa * nc + b]);
        for (size_t b2 = b + 1; b2 < nc; ++b2) {
          const double u = D[pa * nc + b2];
          if (u == 0.0 || prow_of[b2] == nr) continue;
          u_row_index_.push_back(dcol[b2]);
          u_row_value_.push_back(u);
        }
        u_row_start_.push_back(static_cast<Index>(u_row_index_.size()));
        for (size_t a = 0; a < nr; ++a) {
          const double l = D[a * nc + b];
          if (rank[a] <= b || l == 0.0) continue;
          l_index_.push_back(drow[a]);
          l_value_.push_back(l);
        }
        end_l();
      }
    };

    // Live (nonempty) kernel rows/columns: the dense switch looks at their density.
    Index active_nnz = kernel_nnz, live_rows = 0, live_cols = 0;
    for (Index i = 0; i < m; ++i) {
      live_rows += !row_done[i] && K.rlen[i] > 0;
      live_cols += !col_done[i] && K.clen[i] > 0;
    }
    for (Index step = tri_steps; step < m; ++step) {
      const double lr = live_rows, lc = live_cols;
      if (std::min(lr, lc) >= 50 && std::max(lr, lc) <= 1500 && active_nnz >= 0.5 * lr * lc) {
        dense_kernel = live_cols;
        dense_phase();
        break;
      }
      // ---- Markowitz pivot search over the sparsest columns and rows ----
      Index best_r = -1, best_c = -1;
      double best_cost = 1e300, best_abs = 0.0;
      int searched = 0;
      const Index remaining = m - step;
      for (Index cnt = 1; cnt <= remaining; ++cnt) {
        if (best_r >= 0 && (searched >= kSearch || best_cost <= static_cast<double>(cnt - 1) * (cnt - 1))) break;
        for (Index c = cbuck.head(cnt); c >= 0; c = cbuck.next(c)) {
          for (Index k = K.cbeg[c], e = K.cbeg[c] + cnt; k < e; ++k) {
            const Index r = K.cidx[k];
            const Index kr = K.find_in_row(r, c);
            const double v = std::abs(K.rval[kr]);
            if (v <= pivot_tolerance || v < pivot_threshold * K.row_max(r)) continue;
            const double cost = static_cast<double>(K.rlen[r] - 1) * (cnt - 1);
            if (cost < best_cost || (cost == best_cost && v > best_abs)) {
              best_cost = cost, best_abs = v, best_r = r, best_c = c;
            }
          }
          if (++searched >= kSearch && best_r >= 0) break;
        }
        if (best_r >= 0 && (searched >= kSearch || best_cost <= static_cast<double>(cnt - 1) * cnt)) break;
        for (Index r = rbuck.head(cnt); r >= 0; r = rbuck.next(r)) {
          const double thr = std::max(pivot_threshold * K.row_max(r), pivot_tolerance);
          for (Index k = K.rbeg[r], e = K.rbeg[r] + cnt; k < e; ++k) {
            const double v = std::abs(K.rval[k]);
            if (v <= pivot_tolerance || v < thr) continue;
            const Index c = K.ridx[k];
            const double cost = static_cast<double>(cnt - 1) * (K.clen[c] - 1);
            if (cost < best_cost || (cost == best_cost && v > best_abs)) {
              best_cost = cost, best_abs = v, best_r = r, best_c = c;
            }
          }
          if (++searched >= kSearch && best_r >= 0) break;
        }
      }
      if (best_r < 0) break;  // singular remainder

      const Index r = best_r, c = best_c;
      const double pv = K.rval[K.find_in_row(r, c)];
      rbuck.remove(r);
      cbuck.remove(c);
      push_pivot(r, c, pv);

      // Pivot row -> U; detach it from the column lists.
      prj.clear();
      prv.clear();
      for (Index k = K.rbeg[r], e = K.rbeg[r] + K.rlen[r]; k < e; ++k) {
        const Index j = K.ridx[k];
        if (j == c) continue;
        pidx[j] = static_cast<Index>(prj.size());
        prj.push_back(j);
        prv.push_back(K.rval[k]);
        u_row_index_.push_back(j);
        u_row_value_.push_back(K.rval[k]);
        K.remove_from_col(j, r);
      }
      u_row_start_.push_back(static_cast<Index>(u_row_index_.size()));
      active_nnz -= K.rlen[r];
      --live_rows;
      --live_cols;
      K.rlen[r] = 0;
      K.rwaste += K.rcap[r];
      K.rcap[r] = 0;
      const Index plen = static_cast<Index>(prj.size());

      // Eliminate column c from the other rows.
      colc.assign(K.cidx.begin() + K.cbeg[c], K.cidx.begin() + K.cbeg[c] + K.clen[c]);
      K.clen[c] = 0;
      K.cwaste += K.ccap[c];
      K.ccap[c] = 0;
      for (Index i : colc) {
        if (i == r) continue;
        Index ki = K.find_in_row(i, c);
        const double l = K.rval[ki] / pv;
        const Index last = K.rbeg[i] + K.rlen[i] - 1;
        K.ridx[ki] = K.ridx[last];
        K.rval[ki] = K.rval[last];
        --K.rlen[i];
        --active_nnz;
        l_index_.push_back(i);
        l_value_.push_back(l);
        K.rmax[i] = -1.0;
        if (plen == 0) {
          if (K.rlen[i] == 0) --live_rows;
          rbuck.move(i, K.rlen[i]);
          continue;
        }
        K.reserve_row(i, K.rlen[i] + plen);
        work += K.rlen[i] + 2 * plen;
        ++stamp;
        for (Index k = K.rbeg[i], e = K.rbeg[i] + K.rlen[i]; k < e; ++k) {
          const Index j = K.ridx[k];
          const Index t = pidx[j];
          if (t < 0) continue;
          K.rval[k] -= l * prv[t];
          seen[j] = stamp;
        }
        for (Index t = 0; t < plen; ++t) {
          const Index j = prj[t];
          if (seen[j] == stamp) continue;
          const Index w = K.rbeg[i] + K.rlen[i]++;
          ++active_nnz;
          K.ridx[w] = j;
          K.rval[w] = -l * prv[t];
          K.reserve_col(j, K.clen[j] + 1);
          K.cidx[K.cbeg[j] + K.clen[j]++] = i;
        }
        rbuck.move(i, K.rlen[i]);
      }
      end_l();
      for (Index t = 0; t < plen; ++t) {
        pidx[prj[t]] = -1;
        cbuck.move(prj[t], K.clen[prj[t]]);
        if (K.clen[prj[t]] == 0) --live_cols;
      }
    }
  }

  // Handle a singular remainder by substituting unit columns.
  Index num_replaced = 0;
  const Index done = static_cast<Index>(piv_row_.size());
  if (done < m) {
    std::vector<Index> rem_rows, rem_cols;
    for (Index i = 0; i < m; ++i) {
      if (!row_done[i]) rem_rows.push_back(i);
      if (!col_done[i]) rem_cols.push_back(i);
    }
    // Drop U entries that referenced the replaced columns.
    Index w = 0;
    std::vector<Index> new_start(1, 0);
    for (Index k = 0; k < done; ++k) {
      for (Index p = u_row_start_[k]; p < u_row_start_[k + 1]; ++p) {
        if (!col_done[u_row_index_[p]]) continue;
        u_row_index_[w] = u_row_index_[p];
        u_row_value_[w] = u_row_value_[p];
        ++w;
      }
      new_start.push_back(w);
    }
    u_row_index_.resize(w);
    u_row_value_.resize(w);
    u_row_start_ = std::move(new_start);
    for (size_t k = 0; k < rem_cols.size(); ++k) {
      piv_row_.push_back(rem_rows[k]);
      piv_col_.push_back(rem_cols[k]);
      piv_val_.push_back(1.0);
      u_row_start_.push_back(static_cast<Index>(u_row_index_.size()));
      if (replaced) replaced->emplace_back(rem_cols[k], rem_rows[k]);
      ++num_replaced;
    }
  }
  // Calibrated (x3) so the operation count matches measured factorization time
  // relative to solve work on randlp/timeidx instances.
  factor_work_ = 3.0 * (work + 3.0 * factor_nnz() + 20.0 * m);
  eta_work_ = 0.0;
  finish(num_replaced, kernel_dim, kernel_nnz, dense_kernel, basis_nnz, t0);
  step_of_pos_ = orig_step_of_pos_;
  return num_replaced;
}

void LuFactor::finish(Index num_replaced, Index kernel_dim, Index kernel_nnz, Index dense_dim, Index basis_nnz,
                      double t0) {
  const Index m = m_;
  FactorWork& W = factor_work();
  std::vector<Index>&step_of_row = orig_step_of_row_, &step_of_col = orig_step_of_pos_;
  step_of_row.resize(m);
  step_of_col.resize(m);
  step_dead_.assign(m, 0);
  spike_.assign(m, 0.0);
  for (Index k = 0; k < m; ++k) {
    step_of_row[piv_row_[k]] = k;
    step_of_col[piv_col_[k]] = k;
  }
  // Column-wise copy of U: entry (step k, column c') goes to the step that
  // pivoted c', keyed by the pivot row of step k.
  u_col_start_.assign(m + 1, 0);
  for (Index cc : u_row_index_) ++u_col_start_[step_of_col[cc] + 1];
  for (Index k = 0; k < m; ++k) u_col_start_[k + 1] += u_col_start_[k];
  u_col_index_.resize(u_row_index_.size());
  u_col_value_.resize(u_row_index_.size());
  std::vector<Index>& pos = W.pos;
  pos.assign(u_col_start_.begin(), u_col_start_.end() - 1);
  for (Index k = 0; k < m; ++k) {
    for (Index p = u_row_start_[k]; p < u_row_start_[k + 1]; ++p) {
      const Index q = pos[step_of_col[u_row_index_[p]]]++;
      u_col_index_[q] = piv_row_[k];
      u_col_value_[q] = u_row_value_[p];
    }
  }
  // Row-wise copy of L, grouped by the step of the eliminated row.
  std::vector<Index>& cnt = W.cnt;
  cnt.assign(m + 1, 0);
  for (Index i : l_index_) ++cnt[step_of_row[i] + 1];
  lr_row_.clear();
  lr_start_.assign(1, 0);
  std::vector<Index>& grp_pos = W.grp_pos;
  grp_pos.assign(m, -1);
  for (Index k = 0; k < m; ++k) {
    if (cnt[k + 1] == 0) continue;
    grp_pos[k] = lr_start_.back();
    lr_row_.push_back(piv_row_[k]);
    lr_start_.push_back(lr_start_.back() + cnt[k + 1]);
  }
  lr_index_.resize(l_index_.size());
  lr_value_.resize(l_index_.size());
  for (size_t g = 0; g < l_step_.size(); ++g) {
    const Index target = piv_row_[l_step_[g]];
    for (Index p = l_start_[g]; p < l_start_[g + 1]; ++p) {
      const Index q = grp_pos[step_of_row[l_index_[p]]]++;
      lr_index_[q] = target;
      lr_value_[q] = l_value_[p];
    }
  }
  // Position <-> row maps; U's row-wise copy switches to rows (the value of a position lives at its
  // row in the solves).
  row_of_pos_.resize(m);
  pos_of_row_.resize(m);
  identity_ = true;
  for (Index p = 0; p < m; ++p) {
    const Index r = piv_row_[step_of_col[p]];
    row_of_pos_[p] = r;
    pos_of_row_[r] = p;
    identity_ = identity_ && r == p;
  }
  for (Index& c : u_row_index_) c = row_of_pos_[c];
  ustep_of_row_.resize(m);
  for (Index r = 0; r < m; ++r) ustep_of_row_[r] = step_in_dense(step_of_row[r]) ? -1 : step_of_row[r];
  lgrp_of_row_.assign(m, -1);
  for (size_t g = 0; g < l_step_.size(); ++g) lgrp_of_row_[piv_row_[l_step_[g]]] = static_cast<Index>(g);
  lrgrp_of_row_.assign(m, -1);
  for (size_t g = 0; g < lr_row_.size(); ++g) lrgrp_of_row_[lr_row_[g]] = static_cast<Index>(g);
  // Sparse steps that need work in the dense U / U^T loops (ascending): a nonempty U column / row,
  // or else a pivot other than 1 (division only; most slack steps need nothing).
  ucol_steps_.clear();
  ucol_div_steps_.clear();
  urow_steps_.clear();
  urow_div_steps_.clear();
  for (Index k = 0; k < m; ++k) {
    if (step_in_dense(k)) continue;
    if (u_col_start_[k + 1] > u_col_start_[k]) ucol_steps_.push_back(k);
    else if (piv_val_[k] != 1.0) ucol_div_steps_.push_back(k);
    if (u_row_start_[k + 1] > u_row_start_[k]) urow_steps_.push_back(k);
    else if (piv_val_[k] != 1.0) urow_div_steps_.push_back(k);
  }
  if (lu_stats_enabled()) {
    g_totals.factors++;
    g_totals.factor_us += static_cast<long long>((now_seconds() - t0) * 1e6);
    std::fprintf(stderr,
                 "[lu] m %d basis_nnz %d kernel %d (nnz %d, dense %d) L %zu U %zu replaced %d time %.2f ms\n",
                 m, basis_nnz, kernel_dim, kernel_nnz, dense_dim, l_index_.size(), u_row_index_.size(), num_replaced,
                 (now_seconds() - t0) * 1e3);
  }
}


// ---------------------------------------------------------------------------
// Solves.
//
// Every solve works in row space and in place: the pivot of step s sits in row piv_row_[s], and the
// value of the basis position p is held at row row_of_pos_[p] (identity after renumber_positions).
//
// Hyper-sparse solves (HVector API): each triangular stage first finds the rows it will touch by a
// depth-first search from the current nonzeros along the stage's dependency graph (node = row,
// edges = the row's L group / U column / U row / L^T group), then processes them in topological
// order (reverse postorder). The cost then follows the touched entries instead of m. The index list
// of the vector is kept valid through all stages (every row that may become nonzero is appended once,
// tracked by a generation-stamped mark array). If a search reaches more than a fifth of the rows the
// solve continues with the dense loops and rebuilds the index list at the end.
// ---------------------------------------------------------------------------

namespace {

struct SolveScratch {
  std::vector<uint32_t> inlist, visit;
  uint32_t in_gen = 0, vis_gen = 0;
  std::vector<Index> stk_node, stk_pos, order, idx;
  Index norder = 0;  // rows written to `order` by the last reach()
  std::vector<double> dense_x, tmp;  // tmp: all zero between uses
  Index size = 0;
  void ensure(Index m, Index dense_n) {
    if (size < m) {
      size = m;
      inlist.assign(m, 0);
      visit.assign(m, 0);
      in_gen = vis_gen = 0;
      stk_node.resize(m);
      stk_pos.resize(m);
      order.resize(m);
      idx.resize(m);
      tmp.assign(m, 0.0);
    }
    if (dense_x.size() < 3 * static_cast<size_t>(dense_n)) dense_x.assign(3 * static_cast<size_t>(dense_n), 0.0);
  }
  uint32_t new_in() {
    if (++in_gen == 0) {
      std::fill(inlist.begin(), inlist.end(), 0u);
      in_gen = 1;
    }
    return in_gen;
  }
  uint32_t new_vis() {
    if (++vis_gen == 0) {
      std::fill(visit.begin(), visit.end(), 0u);
      vis_gen = 1;
    }
    return vis_gen;
  }
};
SolveScratch& scratch() {
  thread_local SolveScratch s;
  return s;
}

constexpr double kDrop = 1e-14;
// Largest Forrest-Tomlin row eta entry accepted without asking for a refactorization (relative error
// of the later solves ~ 2.2e-16 * 1e7 = 2e-9).
constexpr double kMaxRowEta = 1e7;

// Index list of a vector being solved hyper-sparsely; `valid` false once a stage ran dense.
struct NzList {
  Index* list;
  Index count;
  bool valid;
  uint32_t* inl;
  uint32_t gen;
  void add(Index r) {
    if (valid && inl[r] != gen) inl[r] = gen, list[count++] = r;
  }
};

}  // namespace

bool LuFactor::use_hyper(Index count, double density) const {
  static const int env_mode = [] {
    const char* e = std::getenv("MILP_LU_HYPER");  // 1: dense loops only, 2: always hyper-sparse
    return e ? std::atoi(e) : 0;
  }();
  const int mode = hyper_mode != 0 ? hyper_mode : env_mode;
  if (mode == 1) return false;
  if (mode == 2) return true;
  if (count > m_ / 10) return false;
  if (density >= 0.0) return density < 0.1;
  return count < m_ / 20;
}

// Depth-first search from the rows list[0..count) along r -> index[start[g]..start[g+1]) with
// g = gmap[r] (rows with g < 0 have no edges and need no processing). Rows reached for the first
// time are appended to the list (inlist marks of the current generation). The rows with g >= 0 are
// written to scratch().order in postorder (so the reverse is a topological order), their number to
// scratch().norder. Returns false (values untouched) if the list grows beyond `limit`.
bool LuFactor::reach(const Index* gmap, const Index* start, const Index* index, Index* list, Index& count,
                     Index limit) const {
  SolveScratch& S = scratch();
  const uint32_t vg = S.new_vis(), ig = S.in_gen;
  uint32_t* vis = S.visit.data();
  uint32_t* inl = S.inlist.data();
  Index* sn = S.stk_node.data();
  Index* sp = S.stk_pos.data();
  Index* order = S.order.data();
  Index no = 0;
  const Index nroots = count;
  for (Index k = 0; k < nroots; ++k) {
    const Index r0 = list[k];
    if (vis[r0] == vg) continue;
    vis[r0] = vg;
    const Index g0 = gmap[r0];
    if (g0 < 0) continue;
    Index top = 0;
    sn[0] = r0;
    sp[0] = start[g0];
    while (top >= 0) {
      const Index r = sn[top];
      Index p = sp[top];
      const Index e = start[gmap[r] + 1];
      bool pushed = false;
      while (p < e) {
        const Index c = index[p++];
        if (vis[c] == vg) continue;
        vis[c] = vg;
        if (inl[c] != ig) {
          inl[c] = ig;
          list[count++] = c;
          if (count > limit) return false;
        }
        const Index gc = gmap[c];
        if (gc < 0) continue;
        sp[top] = p;
        ++top;
        sn[top] = c;
        sp[top] = start[gc];
        pushed = true;
        break;
      }
      if (pushed) continue;
      order[no++] = r;
      --top;
    }
  }
  S.norder = no;
  return true;
}

// ---- Dense joint ftran of K right-hand sides (row space, in place) ----
template <int K>
void LuFactor::ftran_dense(double* const* v, bool spike) const {
  SolveScratch& S = scratch();
  double a[K];
  // L etas.
  const Index ng = static_cast<Index>(l_step_.size());
  for (Index g = 0; g < ng; ++g) {
    const Index pr = piv_row_[l_step_[g]];
    bool any = false;
    for (int k = 0; k < K; ++k) any |= (a[k] = v[k][pr]) != 0.0;
    if (!any) continue;
    for (Index p = l_start_[g]; p < l_start_[g + 1]; ++p) {
      const Index i = l_index_[p];
      const double l = l_value_[p];
      for (int k = 0; k < K; ++k) v[k][i] -= l * a[k];
    }
  }
  const Index n = dense_n_, db = dense_begin_;
  double* x[K];
  for (int k = 0; k < K; ++k) x[k] = S.dense_x.data() + static_cast<size_t>(k) * n;
  if (n > 0) {
    // Dense L block (its rows are only touched by earlier steps).
    const Index* rows = piv_row_.data() + db;
    bool any = false;
    for (int k = 0; k < K; ++k)
      for (Index i = 0; i < n; ++i) any |= (x[k][i] = v[k][rows[i]]) != 0.0;
    if (any) {
      for (Index b = 0; b < n; ++b) {
        bool nz = false;
        for (int k = 0; k < K; ++k) nz |= (a[k] = x[k][b]) != 0.0;
        if (!nz) continue;
        const double* col = &dense_lu_[static_cast<size_t>(b) * n];
        for (int k = 0; k < K; ++k) {
          if (a[k] == 0.0) continue;
          double* xk = x[k];
          const double ak = a[k];
          for (Index i = b + 1; i < n; ++i) xk[i] -= col[i] * ak;
        }
      }
      for (int k = 0; k < K; ++k)
        for (Index i = 0; i < n; ++i) v[k][rows[i]] = x[k][i];
    }
  }
  // Forrest-Tomlin row etas.
  const Index nr = static_cast<Index>(r_row_.size());
  for (Index e = 0; e < nr; ++e) {
    const Index q0 = r_start_[e], len = r_start_[e + 1] - q0;
    for (int k = 0; k < K; ++k) v[k][r_row_[e]] -= dot_gather(r_value_.data() + q0, r_index_.data() + q0, v[k], len);
  }
  if (spike && forrest_tomlin && !ft_off_) {
    std::copy_n(v[0], m_, spike_.data());
    spike_all_ = true;
    spike_ok_ = true;
  }
  // U back substitution in place. Forrest-Tomlin steps come last (column-wise),
  // then the live steps of the fresh factorization.
  const char* dead = step_dead_.data();
  for (Index t = static_cast<Index>(ft_row_.size()) - 1; t >= 0; --t) {
    const Index s = m_ + t;
    if (dead[s]) continue;
    const Index pr = piv_row_[s];
    bool any = false;
    for (int k = 0; k < K; ++k) {
      a[k] = v[k][pr];
      if (a[k] != 0.0) a[k] /= piv_val_[s], v[k][pr] = a[k], any = true;
    }
    if (!any) continue;
    for (Index p = ft_col_start_[t], e = p + ft_col_len_[t]; p < e; ++p) {
      const Index r = ft_col_index_[p];
      const double u = ft_col_value_[p];
      for (int k = 0; k < K; ++k) v[k][r] -= u * a[k];
    }
  }
  // The dense block (its steps come after all sparse ones), then the sparse steps with a nonempty U
  // column backward, then the divisions of the steps with an empty column (nothing reads them).
  if (n > 0) {
    {
      for (int k = 0; k < K; ++k)
        for (Index i = 0; i < n; ++i) x[k][i] = v[k][piv_row_[db + i]];
      for (Index i = n - 1; i >= 0; --i) {
        bool nz = false;
        for (int k = 0; k < K; ++k) {
          a[k] = x[k][i] / piv_val_[db + i];
          v[k][piv_row_[db + i]] = a[k];
          nz |= a[k] != 0.0;
        }
        if (!nz) continue;
        const double* col = &dense_lu_[static_cast<size_t>(i) * n];
        for (int k = 0; k < K; ++k) {
          if (a[k] == 0.0) continue;
          double* xk = x[k];
          const double ak = a[k];
          for (Index b = 0; b < i; ++b) xk[b] -= col[b] * ak;
        }
        for (Index p = u_col_start_[db + i]; p < u_col_start_[db + i + 1]; ++p) {
          const Index r = u_col_index_[p];
          const double u = u_col_value_[p];
          for (int k = 0; k < K; ++k) v[k][r] -= u * a[k];
        }
      }
    }
  }
  for (Index t = static_cast<Index>(ucol_steps_.size()) - 1; t >= 0; --t) {
    const Index s = ucol_steps_[t];
    if (dead[s]) continue;
    const Index pr = piv_row_[s];
    bool any = false;
    for (int k = 0; k < K; ++k) {
      a[k] = v[k][pr];
      if (a[k] != 0.0) a[k] /= piv_val_[s], v[k][pr] = a[k], any = true;
    }
    if (!any) continue;
    for (Index p = u_col_start_[s]; p < u_col_start_[s + 1]; ++p) {
      const Index r = u_col_index_[p];
      const double u = u_col_value_[p];
      for (int k = 0; k < K; ++k) v[k][r] -= u * a[k];
    }
  }
  for (const Index s : ucol_div_steps_) {
    if (dead[s]) continue;
    const Index pr = piv_row_[s];
    for (int k = 0; k < K; ++k) v[k][pr] /= piv_val_[s];
  }
  // Product-form etas.
  const Index ne = static_cast<Index>(eta_pos_.size());
  for (Index e = 0; e < ne; ++e) {
    const Index p = eta_pos_[e];
    bool any = false;
    for (int k = 0; k < K; ++k) {
      a[k] = v[k][p];
      if (a[k] != 0.0) a[k] /= eta_pivot_[e], v[k][p] = a[k], any = true;
    }
    if (!any) continue;
    for (Index q = eta_start_[e]; q < eta_start_[e + 1]; ++q) {
      const Index i = eta_index_[q];
      const double ev = eta_value_[q];
      for (int k = 0; k < K; ++k) v[k][i] -= ev * a[k];
    }
  }
}

// ---- Single right-hand side ftran, hyper-sparse where possible (row space, in place) ----
void LuFactor::ftran_hyper(HVector& v, bool spike) const {
  SolveScratch& S = scratch();
  double* x = v.dense.data();
  NzList L{v.nz.data(), 0, true, S.inlist.data(), S.new_in()};
  for (Index k = 0; k < v.count; ++k) L.add(v.nz[k]);  // also drops duplicates
  const Index limit = hyper_limit();
  // L etas.
  if (reach(lgrp_of_row_.data(), l_start_.data(), l_index_.data(), L.list, L.count, limit)) {
    const Index* order = S.order.data();
    for (Index k = S.norder - 1; k >= 0; --k) {
      const Index pr = order[k];
      const double a = x[pr];
      if (a == 0.0) continue;
      const Index g = lgrp_of_row_[pr];
      for (Index p = l_start_[g]; p < l_start_[g + 1]; ++p) x[l_index_[p]] -= l_value_[p] * a;
    }
  } else {
    L.valid = false;
    const Index ng = static_cast<Index>(l_step_.size());
    for (Index g = 0; g < ng; ++g) {
      const double a = x[piv_row_[l_step_[g]]];
      if (a == 0.0) continue;
      for (Index p = l_start_[g]; p < l_start_[g + 1]; ++p) x[l_index_[p]] -= l_value_[p] * a;
    }
  }
  const Index n = dense_n_, db = dense_begin_;
  if (n > 0) {
    const Index* rows = piv_row_.data() + db;
    double* d = S.dense_x.data();
    bool any = false;
    for (Index i = 0; i < n; ++i) any |= (d[i] = x[rows[i]]) != 0.0;
    if (any) {
      for (Index b = 0; b < n; ++b) {
        const double a = d[b];
        if (a == 0.0) continue;
        const double* col = &dense_lu_[static_cast<size_t>(b) * n];
        for (Index i = b + 1; i < n; ++i) d[i] -= col[i] * a;
      }
      for (Index i = 0; i < n; ++i) {
        x[rows[i]] = d[i];
        L.add(rows[i]);
      }
    }
  }
  const Index nr = static_cast<Index>(r_row_.size());
  for (Index e = 0; e < nr; ++e) {
    const Index q0 = r_start_[e], len = r_start_[e + 1] - q0;
    const double dv = dot_gather(r_value_.data() + q0, r_index_.data() + q0, x, len);
    if (dv == 0.0) continue;
    x[r_row_[e]] -= dv;
    L.add(r_row_[e]);
  }
  if (spike && forrest_tomlin && !ft_off_) {
    if (spike_all_) {
      std::fill(spike_.begin(), spike_.end(), 0.0);
    } else {
      for (Index i : spike_idx_) spike_[i] = 0.0;
    }
    if (L.valid) {
      spike_idx_.assign(L.list, L.list + L.count);
      for (Index i : spike_idx_) spike_[i] = x[i];
      spike_all_ = false;
    } else {
      std::copy_n(x, m_, spike_.data());
      spike_all_ = true;
    }
    spike_ok_ = true;
  }
  // U: Forrest-Tomlin steps (last), the dense block, then the sparse steps of the fresh factorization.
  const char* dead = step_dead_.data();
  for (Index t = static_cast<Index>(ft_row_.size()) - 1; t >= 0; --t) {
    const Index s = m_ + t;
    if (dead[s]) continue;
    const Index pr = piv_row_[s];
    double a = x[pr];
    if (a == 0.0) continue;
    a /= piv_val_[s];
    x[pr] = a;
    for (Index p = ft_col_start_[t], e = p + ft_col_len_[t]; p < e; ++p) {
      const Index r = ft_col_index_[p];
      x[r] -= ft_col_value_[p] * a;
      L.add(r);
    }
  }
  if (n > 0) {
    double* d = S.dense_x.data();
    bool any = false;
    for (Index i = 0; i < n; ++i) any |= (d[i] = x[piv_row_[db + i]]) != 0.0;
    if (any) {
      for (Index i = n - 1; i >= 0; --i) {
        const double a = d[i] / piv_val_[db + i];
        x[piv_row_[db + i]] = a;
        L.add(piv_row_[db + i]);
        if (a == 0.0) continue;
        const double* col = &dense_lu_[static_cast<size_t>(i) * n];
        for (Index b = 0; b < i; ++b) d[b] -= col[b] * a;
        for (Index p = u_col_start_[db + i]; p < u_col_start_[db + i + 1]; ++p) {
          x[u_col_index_[p]] -= u_col_value_[p] * a;
          L.add(u_col_index_[p]);
        }
      }
    }
  }
  if (L.valid && reach(ustep_of_row_.data(), u_col_start_.data(), u_col_index_.data(), L.list, L.count, limit)) {
    const Index* order = S.order.data();
    for (Index k = S.norder - 1; k >= 0; --k) {
      const Index pr = order[k];
      double a = x[pr];
      if (a == 0.0) continue;
      const Index s = ustep_of_row_[pr];
      a /= piv_val_[s];
      x[pr] = a;
      for (Index p = u_col_start_[s]; p < u_col_start_[s + 1]; ++p) x[u_col_index_[p]] -= u_col_value_[p] * a;
    }
  } else {
    L.valid = false;
    for (Index t = static_cast<Index>(ucol_steps_.size()) - 1; t >= 0; --t) {
      const Index s = ucol_steps_[t];
      if (dead[s]) continue;
      const Index pr = piv_row_[s];
      double a = x[pr];
      if (a == 0.0) continue;
      a /= piv_val_[s];
      x[pr] = a;
      for (Index p = u_col_start_[s]; p < u_col_start_[s + 1]; ++p) x[u_col_index_[p]] -= u_col_value_[p] * a;
    }
    for (const Index s : ucol_div_steps_)
      if (!dead[s]) x[piv_row_[s]] /= piv_val_[s];
  }
  const Index ne = static_cast<Index>(eta_pos_.size());
  for (Index e = 0; e < ne; ++e) {
    const Index p = eta_pos_[e];
    double a = x[p];
    if (a == 0.0) continue;
    a /= eta_pivot_[e];
    x[p] = a;
    for (Index q = eta_start_[e]; q < eta_start_[e + 1]; ++q) {
      x[eta_index_[q]] -= eta_value_[q] * a;
      L.add(eta_index_[q]);
    }
  }
  if (L.valid) {
    Index c = 0;
    for (Index k = 0; k < L.count; ++k) {
      const Index i = L.list[k];
      if (std::abs(x[i]) > kDrop) {
        L.list[c++] = i;
      } else {
        x[i] = 0.0;
      }
    }
    v.count = c;
  } else {
    v.rebuild_index();
  }
}

// ---- btran (row space, in place): PF etas, U^T, Forrest-Tomlin steps, R^T, L^T ----
bool LuFactor::btran_rows(double* x, Index* list, Index& count) const {
  SolveScratch& S = scratch();
  NzList L{list, 0, count >= 0, S.inlist.data(), S.new_in()};
  if (L.valid) {
    const Index c0 = count;
    for (Index k = 0; k < c0; ++k) L.add(list[k]);
  }
  const Index limit = hyper_limit();
  for (Index e = static_cast<Index>(eta_pos_.size()) - 1; e >= 0; --e) {
    const Index p = eta_pos_[e];
    const Index q0 = eta_start_[e];
    const double a = (x[p] - dot_gather(eta_value_.data() + q0, eta_index_.data() + q0, x, eta_start_[e + 1] - q0)) /
                     eta_pivot_[e];
    x[p] = a;
    if (a != 0.0) L.add(p);
  }
  // U^T: sparse steps of the fresh factorization in order (each scatters its U row into later steps).
  const Index n = dense_n_, db = dense_begin_;
  if (L.valid && reach(ustep_of_row_.data(), u_row_start_.data(), u_row_index_.data(), L.list, L.count, limit)) {
    const Index* order = S.order.data();
    for (Index k = S.norder - 1; k >= 0; --k) {
      const Index pr = order[k];
      double a = x[pr];
      if (a == 0.0) continue;
      const Index s = ustep_of_row_[pr];
      a /= piv_val_[s];
      x[pr] = a;
      for (Index p = u_row_start_[s]; p < u_row_start_[s + 1]; ++p) x[u_row_index_[p]] -= u_row_value_[p] * a;
    }
  } else {
    L.valid = false;
    const char* dead = step_dead_.data();
    for (const Index k : urow_steps_) {
      if (dead[k]) continue;
      const Index pr = piv_row_[k];
      double a = x[pr];
      if (a == 0.0) continue;
      a /= piv_val_[k];
      x[pr] = a;
      for (Index p = u_row_start_[k]; p < u_row_start_[k + 1]; ++p) x[u_row_index_[p]] -= u_row_value_[p] * a;
    }
    for (const Index k : urow_div_steps_)
      if (!dead[k]) x[piv_row_[k]] /= piv_val_[k];
  }
  // Dense block (no sparse U row entries: nothing to scatter).
  if (n > 0) {
    double* d = S.dense_x.data();
    bool any = false;
    for (Index a = 0; a < n; ++a) any |= (d[a] = x[piv_row_[db + a]]) != 0.0;
    if (any) {
      for (Index a = 0; a < n; ++a) {
        const double* col = &dense_lu_[static_cast<size_t>(a) * n];
        d[a] = (d[a] - dot(col, d, a)) / piv_val_[db + a];
      }
      for (Index a = 0; a < n; ++a) {
        x[piv_row_[db + a]] = d[a];
        L.add(piv_row_[db + a]);
      }
    }
  }
  // Forrest-Tomlin steps: their U columns hold rows of earlier steps.
  const Index na = static_cast<Index>(ft_row_.size());
  for (Index t = 0; t < na; ++t) {
    const Index k = m_ + t;
    if (step_dead_[k]) continue;
    const Index q0 = ft_col_start_[t], pr = piv_row_[k];
    const double a =
        (x[pr] - dot_gather(ft_col_value_.data() + q0, ft_col_index_.data() + q0, x, ft_col_len_[t])) / piv_val_[k];
    x[pr] = a;
    if (a != 0.0) L.add(pr);
  }
  // Forrest-Tomlin row etas transposed, in reverse order.
  for (Index e = static_cast<Index>(r_row_.size()) - 1; e >= 0; --e) {
    const double a = x[r_row_[e]];
    if (a == 0.0) continue;
    for (Index q = r_start_[e]; q < r_start_[e + 1]; ++q) {
      x[r_index_[q]] -= r_value_[q] * a;
      L.add(r_index_[q]);
    }
  }
  // L^T: the dense block first (its rows receive contributions only from each other), then the groups.
  if (n > 0) {
    const Index* rows = piv_row_.data() + db;
    double* d = S.dense_x.data();
    bool any = false;
    for (Index a = 0; a < n; ++a) any |= (d[a] = x[rows[a]]) != 0.0;
    if (any) {
      for (Index b = n - 1; b >= 0; --b) {
        const double* col = &dense_lu_[static_cast<size_t>(b) * n];
        d[b] -= dot(col + b + 1, d + b + 1, n - b - 1);
      }
      for (Index a = 0; a < n; ++a) {
        x[rows[a]] = d[a];
        L.add(rows[a]);
      }
    }
  }
  if (L.valid && reach(lrgrp_of_row_.data(), lr_start_.data(), lr_index_.data(), L.list, L.count, limit)) {
    const Index* order = S.order.data();
    for (Index k = S.norder - 1; k >= 0; --k) {
      const Index r = order[k];
      const double a = x[r];
      if (a == 0.0) continue;
      const Index g = lrgrp_of_row_[r];
      for (Index p = lr_start_[g]; p < lr_start_[g + 1]; ++p) x[lr_index_[p]] -= lr_value_[p] * a;
    }
  } else {
    L.valid = false;
    for (Index g = static_cast<Index>(lr_row_.size()) - 1; g >= 0; --g) {
      const double a = x[lr_row_[g]];
      if (a == 0.0) continue;
      for (Index p = lr_start_[g]; p < lr_start_[g + 1]; ++p) x[lr_index_[p]] -= lr_value_[p] * a;
    }
  }
  count = L.count;
  return L.valid;
}

// Row space -> position space for a vector with a valid index list.
void LuFactor::to_positions(HVector& v) const {
  if (identity_) return;
  double* t = scratch().tmp.data();
  double* x = v.dense.data();
  for (Index k = 0; k < v.count; ++k) {
    const Index r = v.nz[k], p = pos_of_row_[r];
    t[p] = x[r];
    x[r] = 0.0;
    v.nz[k] = p;
  }
  for (Index k = 0; k < v.count; ++k) {
    const Index p = v.nz[k];
    x[p] = t[p];
    t[p] = 0.0;
  }
}

void LuFactor::from_positions(HVector& v) const {
  if (identity_) return;
  double* t = scratch().tmp.data();
  double* x = v.dense.data();
  for (Index k = 0; k < v.count; ++k) {
    const Index p = v.nz[k], r = row_of_pos_[p];
    t[r] = x[p];
    x[p] = 0.0;
    v.nz[k] = r;
  }
  for (Index k = 0; k < v.count; ++k) {
    const Index r = v.nz[k];
    x[r] = t[r];
    t[r] = 0.0;
  }
}

void LuFactor::ftran(std::vector<double>& v) const {
  scratch().ensure(m_, dense_n_);
  double* vs[1] = {v.data()};
  ftran_dense<1>(vs, true);
  if (!identity_) {
    std::vector<double>& t = scratch().tmp;
    for (Index p = 0; p < m_; ++p) t[p] = v[row_of_pos_[p]];
    std::copy_n(t.data(), m_, v.data());
    std::fill_n(t.data(), m_, 0.0);
  }
}

void LuFactor::btran(std::vector<double>& v) const {
  SolveScratch& S = scratch();
  S.ensure(m_, dense_n_);
  if (!identity_) {
    std::vector<double>& t = S.tmp;
    for (Index p = 0; p < m_; ++p) t[row_of_pos_[p]] = v[p];
    std::copy_n(t.data(), m_, v.data());
    std::fill_n(t.data(), m_, 0.0);
  }
  Index count = -1;  // no index list: dense loops
  btran_rows(v.data(), nullptr, count);
}

void LuFactor::ftran_multi(HVector* const* vs, const double* dens, int k) const {
  scratch().ensure(m_, dense_n_);
  double* dv[3];
  HVector* dh[3];
  int nd = 0;
  for (int i = 0; i < k; ++i) {
    if (use_hyper(vs[i]->count, dens[i])) {
      ftran_hyper(*vs[i], i == 0);
      to_positions(*vs[i]);
    } else {
      dh[nd] = vs[i];
      dv[nd++] = vs[i]->dense.data();
    }
  }
  if (nd == 0) return;
  const bool spike = dh[0] == vs[0];
  if (nd == 1) ftran_dense<1>(dv, spike);
  else if (nd == 2) ftran_dense<2>(dv, spike);
  else ftran_dense<3>(dv, spike);
  for (int i = 0; i < nd; ++i) {
    dh[i]->rebuild_index();
    to_positions(*dh[i]);
  }
}

void LuFactor::ftran(HVector& v, double density) const {
  HVector* vs[1] = {&v};
  ftran_multi(vs, &density, 1);
}

void LuFactor::ftran(HVector& a, HVector& b, double da, double db) const {
  HVector* vs[2] = {&a, &b};
  const double d[2] = {da, db};
  ftran_multi(vs, d, 2);
}

void LuFactor::ftran(HVector& a, HVector& b, HVector& c, double da, double db, double dc) const {
  HVector* vs[3] = {&a, &b, &c};
  const double d[3] = {da, db, dc};
  ftran_multi(vs, d, 3);
}

void LuFactor::btran(HVector& v, double density) const {
  scratch().ensure(m_, dense_n_);
  from_positions(v);
  Index count = use_hyper(v.count, density) ? v.count : -1;  // -1: dense loops
  if (btran_rows(v.dense.data(), v.nz.data(), count)) {
    Index c = 0;
    double* x = v.dense.data();
    for (Index k = 0; k < count; ++k) {
      const Index i = v.nz[k];
      if (std::abs(x[i]) > kDrop) {
        v.nz[c++] = i;
      } else {
        x[i] = 0.0;
      }
    }
    v.count = c;
  } else {
    v.rebuild_index();
  }
}

void LuFactor::renumber_positions(std::vector<Index>& old_pos) {
  old_pos = pos_of_row_;
  if (identity_) return;
  const Index m = m_;
  for (Index k = 0; k < static_cast<Index>(piv_row_.size()); ++k) piv_col_[k] = piv_row_[k];
  orig_step_of_pos_ = orig_step_of_row_;
  step_of_pos_.resize(m);
  for (Index r = 0; r < m; ++r) step_of_pos_[r] = orig_step_of_row_[r];
  // Forrest-Tomlin steps keep their row: the position of a replaced column is its row too.
  for (Index t = 0; t < static_cast<Index>(ft_row_.size()); ++t)
    if (!step_dead_[m + t]) step_of_pos_[ft_row_[t]] = m + t;
  for (Index p = 0; p < m; ++p) row_of_pos_[p] = pos_of_row_[p] = p;
  identity_ = true;
}

bool LuFactor::ft_update(Index p, double alpha_p) {
  if (!forrest_tomlin || ft_off_ || !spike_ok_) return false;
  spike_ok_ = false;
  const Index s = step_of_pos_[p];
  if (step_in_dense(s)) return false;
  const Index r = piv_row_[s], na = static_cast<Index>(ft_row_.size());
  SolveScratch& S = scratch();
  S.ensure(m_, dense_n_);
  // Row r of U right of the diagonal (row space), removed from the column-wise copies; it is the
  // right-hand side of U^T mu = u_r (solved in place in the zeroed scratch vector).
  HVector h;
  h.dense.swap(S.tmp);
  h.nz.swap(S.idx);
  h.count = 0;
  double* u = h.dense.data();
  if (s < m_) {
    for (Index q = u_row_start_[s]; q < u_row_start_[s + 1]; ++q) {
      const double val = u_row_value_[q];
      if (val == 0.0) continue;
      const Index c = u_row_index_[q], t = orig_step_of_row_[c];
      u[c] = val;
      h.nz[h.count++] = c;
      for (Index k = u_col_start_[t]; k < u_col_start_[t + 1]; ++k) {
        if (u_col_index_[k] == r) {
          u_col_value_[k] = 0.0;
          break;
        }
      }
    }
  }
  for (Index t = 0; t < na; ++t) {
    if (step_dead_[m_ + t]) continue;
    const Index q0 = ft_col_start_[t];
    for (Index q = q0; q < q0 + ft_col_len_[t]; ++q) {
      if (ft_col_index_[q] != r) continue;
      const Index pr = piv_row_[m_ + t];
      if (u[pr] == 0.0) h.nz[h.count++] = pr;
      u[pr] = ft_col_value_[q];
      const Index last = q0 + --ft_col_len_[t];
      ft_col_index_[q] = ft_col_index_[last];
      ft_col_value_[q] = ft_col_value_[last];
      break;
    }
  }
  // mu = U^-T u: the steps after s only (u is zero in the rows of s and earlier steps). The PF etas
  // and L parts of btran do not apply; run the U^T part on a factor view without them.
  {
    // U^T of the original sparse steps.
    NzList L{h.nz.data(), 0, true, S.inlist.data(), S.new_in()};
    const Index cnt0 = h.count;
    for (Index k = 0; k < cnt0; ++k) L.add(h.nz[k]);
    const Index limit = m_ / 5;
    if (reach(ustep_of_row_.data(), u_row_start_.data(), u_row_index_.data(), L.list, L.count, limit)) {
      const Index* order = S.order.data();
      for (Index k = S.norder - 1; k >= 0; --k) {
        const Index pr = order[k];
        double a = u[pr];
        if (a == 0.0) continue;
        const Index st = ustep_of_row_[pr];
        a /= piv_val_[st];
        u[pr] = a;
        for (Index q = u_row_start_[st]; q < u_row_start_[st + 1]; ++q) u[u_row_index_[q]] -= u_row_value_[q] * a;
      }
    } else {
      L.valid = false;
      for (Index k = std::min(s, m_); k < m_; ++k) {
        if (step_in_dense(k) || step_dead_[k]) continue;
        const Index pr = piv_row_[k];
        double a = u[pr];
        if (a == 0.0) continue;
        a /= piv_val_[k];
        u[pr] = a;
        for (Index q = u_row_start_[k]; q < u_row_start_[k + 1]; ++q) u[u_row_index_[q]] -= u_row_value_[q] * a;
      }
    }
    if (dense_n_ > 0) {
      const Index n = dense_n_, db = dense_begin_;
      double* d = S.dense_x.data();
      bool any = false;
      for (Index a = 0; a < n; ++a) any |= (d[a] = u[piv_row_[db + a]]) != 0.0;
      if (any) {
        for (Index a = 0; a < n; ++a) {
          const double* col = &dense_lu_[static_cast<size_t>(a) * n];
          d[a] = (d[a] - dot(col, d, a)) / piv_val_[db + a];
        }
        for (Index a = 0; a < n; ++a) {
          u[piv_row_[db + a]] = d[a];
          L.add(piv_row_[db + a]);
        }
      }
    }
    for (Index t = std::max<Index>(s - m_ + 1, 0); t < na; ++t) {
      const Index k = m_ + t;
      if (step_dead_[k]) continue;
      const Index q0 = ft_col_start_[t], pr = piv_row_[k];
      const double a =
          (u[pr] - dot_gather(ft_col_value_.data() + q0, ft_col_index_.data() + q0, u, ft_col_len_[t])) / piv_val_[k];
      u[pr] = a;
      if (a != 0.0) L.add(pr);
    }
    if (L.valid) {
      h.count = L.count;
    } else {
      // Dense fallback: list every row (zeros are skipped below).
      for (Index i = 0; i < m_; ++i) h.nz[i] = i;
      h.count = m_;
    }
  }
  // Row eta R = I - e_r mu^T, and the new pivot (row r of R times the spike).
  double piv = spike_[r], mu_max = 0.0;
  for (Index k = 0; k < h.count; ++k) {
    const Index i = h.nz[k];
    const double v = u[i];
    if (v == 0.0) continue;
    u[i] = 0.0;
    if (i == r || std::abs(v) < 1e-14) continue;
    piv -= v * spike_[i];
    mu_max = std::max(mu_max, std::abs(v));
    r_index_.push_back(i);
    r_value_.push_back(v);
  }
  h.dense.swap(S.tmp);  // all zero again
  h.nz.swap(S.idx);
  r_row_.push_back(r);
  r_start_.push_back(static_cast<Index>(r_index_.size()));
  const double expected = piv_val_[s] * alpha_p;
  // Besides a new pivot that disagrees with the ftran's, a large row eta makes the factorization
  // inaccurate: every later solve applies it, with rounding errors of about eps * max|mu| relative to
  // the solution (on ill-conditioned bases while the pivots still agree).
  if (!(std::abs(piv - expected) <= 1e-8 * std::max(1.0, std::abs(expected))) || std::abs(piv) < pivot_tolerance ||
      mu_max > kMaxRowEta)
    ft_unstable_ = true;
  // Column p of U leaves the rows of earlier (original) steps.
  if (s < m_) {
    for (Index q = u_col_start_[s]; q < u_col_start_[s + 1]; ++q) {
      if (u_col_value_[q] == 0.0) continue;
      const Index t = orig_step_of_row_[u_col_index_[q]];
      for (Index k = u_row_start_[t]; k < u_row_start_[t + 1]; ++k) {
        if (u_row_index_[k] == r) {
          u_row_value_[k] = 0.0;
          break;
        }
      }
    }
    ustep_of_row_[r] = -1;
  }
  // The spike becomes the U column of a new last step.
  step_dead_[s] = 1;
  const Index ns = static_cast<Index>(piv_row_.size());
  piv_row_.push_back(r);
  piv_col_.push_back(p);
  piv_val_.push_back(piv);
  step_dead_.push_back(0);
  step_of_pos_[p] = ns;
  ft_row_.push_back(r);
  ft_col_start_.push_back(static_cast<Index>(ft_col_index_.size()));
  auto take = [&](Index i) {
    const double v = spike_[i];
    if (i == r || std::abs(v) < 1e-14) return;
    ft_col_index_.push_back(i);
    ft_col_value_.push_back(v);
  };
  if (spike_all_) {
    for (Index i = 0; i < m_; ++i) take(i);
  } else {
    for (Index i : spike_idx_) take(i);
  }
  ft_col_len_.push_back(static_cast<Index>(ft_col_index_.size()) - ft_col_start_.back());
  eta_work_ += 4.0 * eta_nnz();
  if (lu_stats_enabled()) g_totals.updates++;
  return true;
}

void LuFactor::update(Index p, const std::vector<double>& alpha) {
  if (ft_update(p, alpha[p])) return;
  ft_off_ = true;
  eta_pos_.push_back(row_of_pos_[p]);
  eta_pivot_.push_back(alpha[p]);
  for (Index i = 0; i < m_; ++i) {
    if (i == p || alpha[i] == 0.0 || std::abs(alpha[i]) < 1e-14) continue;
    eta_index_.push_back(row_of_pos_[i]);
    eta_value_.push_back(alpha[i]);
  }
  eta_start_.push_back(static_cast<Index>(eta_index_.size()));
  eta_work_ += 4.0 * eta_nnz();
  if (lu_stats_enabled()) g_totals.updates++;
}

void LuFactor::update(Index p, const HVector& alpha) {
  if (ft_update(p, alpha.dense[p])) return;
  ft_off_ = true;
  eta_pos_.push_back(row_of_pos_[p]);
  eta_pivot_.push_back(alpha.dense[p]);
  for (Index k = 0; k < alpha.count; ++k) {
    const Index i = alpha.nz[k];
    const double a = alpha.dense[i];
    if (i == p || std::abs(a) < 1e-14) continue;
    eta_index_.push_back(row_of_pos_[i]);
    eta_value_.push_back(a);
  }
  eta_start_.push_back(static_cast<Index>(eta_index_.size()));
  eta_work_ += 4.0 * eta_nnz();
  if (lu_stats_enabled()) g_totals.updates++;
}

}  // namespace milp
