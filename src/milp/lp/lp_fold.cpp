#include "milp/lp/lp_fold.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <tuple>

namespace milp {
namespace {

uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

uint64_t bits(double v) {
  if (v == 0.0) v = 0.0;  // -0.0
  uint64_t b;
  std::memcpy(&b, &v, sizeof b);
  return b;
}

}  // namespace

LpFold fold_lp(const SparseMatrix& A, const std::vector<double>& cost, const std::vector<double>& col_lb,
               const std::vector<double>& col_ub, const std::vector<double>& row_lb, const std::vector<double>& row_ub,
               double max_share) {
  LpFold f;
  const Index n = A.num_major, m = A.num_minor;
  if (n == 0 || m == 0) return f;
  const SparseMatrix R = A.transpose();  // CSR
  // Initial colours: columns by (cost, lb, ub), rows by (lb, ub); the two sides never share a colour. Nodes: columns
  // 0..n-1, rows n..n+m-1.
  const Index N = n + m;
  std::vector<Index> color(N);
  Index nclass = 0;
  {
    std::map<std::tuple<double, double, double>, Index> ids;
    for (Index j = 0; j < n; ++j) {
      auto [it, fresh] = ids.try_emplace({cost[j], col_lb[j], col_ub[j]}, nclass);
      nclass += fresh;
      color[j] = it->second;
    }
    std::map<std::pair<double, double>, Index> rids;
    for (Index i = 0; i < m; ++i) {
      auto [it, fresh] = rids.try_emplace({row_lb[i], row_ub[i]}, nclass);
      nclass += fresh;
      color[n + i] = it->second;
    }
  }
  const long long limit = static_cast<long long>(max_share * static_cast<double>(N));
  if (nclass > limit) return f;
  // Partition refinement with a queue of splitter classes (Hopcroft's smaller-half rule): a class is split by the
  // multiset of coefficients its nodes have into the splitter, hashed additively (so the multiset into the last
  // part of a split class follows from the others); the exact check below catches a hash collision. Each class
  // is a contiguous range of `elems`.
  std::vector<Index> elems(N), pos(N), pbeg, psize;
  {
    pbeg.assign(static_cast<size_t>(nclass) + 1, 0);
    for (Index v = 0; v < N; ++v) ++pbeg[static_cast<size_t>(color[v]) + 1];
    for (Index c = 0; c < nclass; ++c) pbeg[static_cast<size_t>(c) + 1] += pbeg[c];
    std::vector<Index> at(pbeg.begin(), pbeg.end() - 1);
    for (Index v = 0; v < N; ++v) pos[v] = at[color[v]]++, elems[pos[v]] = v;
    psize.resize(nclass);
    for (Index c = 0; c < nclass; ++c) psize[c] = pbeg[static_cast<size_t>(c) + 1] - pbeg[c];
    pbeg.pop_back();
  }
  std::vector<Index> queue;
  std::vector<char> inq(static_cast<size_t>(nclass), 1);
  for (Index c = nclass; c-- > 0;) queue.push_back(c);
  std::vector<uint64_t> key(N, 0);
  std::vector<char> mark(N, 0);
  std::vector<Index> hit_nodes, members;
  std::vector<std::pair<std::pair<Index, uint64_t>, Index>> order;  // ((class, key), node)
  long long work = 0;
  const long long max_work = 200LL * static_cast<long long>(A.nnz()) + 100LL * N + 1000000;
  while (!queue.empty()) {
    const Index sc = queue.back();
    queue.pop_back();
    inq[sc] = 0;
    ++f.rounds;
    members.assign(elems.begin() + pbeg[sc], elems.begin() + pbeg[sc] + psize[sc]);
    hit_nodes.clear();
    for (Index u : members) {
      const SparseMatrix& M = u < n ? A : R;
      const Index uu = u < n ? u : u - n, off = u < n ? n : 0;
      for (Index k = M.begin(uu); k < M.end(uu); ++k) {
        const Index v = off + M.index[k];
        if (!mark[v]) mark[v] = 1, hit_nodes.push_back(v);
        key[v] += mix(bits(M.value[k]));
      }
      work += M.end(uu) - M.begin(uu);
    }
    order.clear();
    for (Index v : hit_nodes) order.push_back({{color[v], key[v]}, v});
    std::sort(order.begin(), order.end());
    work += static_cast<long long>(order.size()) * 4;
    for (size_t a = 0; a < order.size();) {
      const Index c = order[a].first.first;
      size_t e = a;
      while (e < order.size() && order[e].first.first == c) ++e;
      const Index nt = static_cast<Index>(e - a), rest = psize[c] - nt;
      const bool one_key = order[a].first.second == order[e - 1].first.second;
      if (!(one_key && rest == 0)) {
        // The hit_nodes nodes to the front of the class's range, in key order.
        const Index base = pbeg[c];
        for (size_t t = a; t < e; ++t) {
          const Index v = order[t].second, to = base + static_cast<Index>(t - a), w = elems[to];
          elems[pos[v]] = w, pos[w] = pos[v];
          elems[to] = v, pos[v] = to;
        }
        // Parts: the key groups, then the untouched rest. The largest keeps the class id.
        struct Part { Index beg, size; };
        std::vector<Part> parts;
        for (size_t t = a; t < e;) {
          size_t t2 = t;
          while (t2 < e && order[t2].first.second == order[t].first.second) ++t2;
          parts.push_back({base + static_cast<Index>(t - a), static_cast<Index>(t2 - t)});
          t = t2;
        }
        if (rest > 0) parts.push_back({base + nt, rest});
        size_t big = 0;
        for (size_t q = 1; q < parts.size(); ++q) {
          if (parts[q].size > parts[big].size) big = q;
        }
        const bool was_queued = inq[c] != 0;
        for (size_t q = 0; q < parts.size(); ++q) {
          Index id = c;
          if (q != big) {
            id = nclass++;
            pbeg.push_back(parts[q].beg), psize.push_back(parts[q].size), inq.push_back(0);
            for (Index t = parts[q].beg; t < parts[q].beg + parts[q].size; ++t) color[elems[t]] = id;
            work += parts[q].size;
          } else {
            pbeg[c] = parts[q].beg, psize[c] = parts[q].size;
          }
          if ((q != big || was_queued) && !inq[id]) inq[id] = 1, queue.push_back(id);
        }
      }
      a = e;
    }
    for (Index v : hit_nodes) key[v] = 0, mark[v] = 0;
    if (nclass > limit || work > max_work) return f;
  }
  // Dense class ids per side.
  std::vector<Index> cc(n), rc(m), remap(static_cast<size_t>(nclass), -1);
  Index ncc = 0, nrc = 0;
  for (Index j = 0; j < n; ++j) {
    if (remap[color[j]] < 0) remap[color[j]] = ncc++;
    cc[j] = remap[color[j]];
  }
  for (Index i = 0; i < m; ++i) {
    if (remap[color[n + i]] < 0) remap[color[n + i]] = nrc++;
    rc[i] = remap[color[n + i]];
  }
  // Class sizes and representatives (the first member).
  std::vector<Index> crep(ncc, -1), rrep(nrc, -1);
  f.col_size.assign(ncc, 0), f.row_size.assign(nrc, 0);
  for (Index j = 0; j < n; ++j) {
    if (crep[cc[j]] < 0) crep[cc[j]] = j;
    ++f.col_size[cc[j]];
  }
  for (Index i = 0; i < m; ++i) {
    if (rrep[rc[i]] < 0) rrep[rc[i]] = i;
    ++f.row_size[rc[i]];
  }
  // Exact check of the partition: every row's sums per column class equal its representative's (and its sides),
  // every column's sums per row class equal its representative's (and its cost and bounds). Sums of the same
  // multiset of coefficients in another order may differ by rounding: compared within 1e-12 of their magnitude.
  auto same = [](double a, double b, double mag) { return std::abs(a - b) <= 1e-12 * std::max(1.0, mag); };
  // Members of each class, grouped (counting sort by class).
  auto grouped = [](const std::vector<Index>& cls, Index nc, std::vector<Index>& beg, std::vector<Index>& mem) {
    beg.assign(static_cast<size_t>(nc) + 1, 0);
    for (Index c : cls) ++beg[static_cast<size_t>(c) + 1];
    for (Index c = 0; c < nc; ++c) beg[static_cast<size_t>(c) + 1] += beg[c];
    mem.resize(cls.size());
    std::vector<Index> next(beg.begin(), beg.end() - 1);
    for (Index v = 0; v < static_cast<Index>(cls.size()); ++v) mem[next[cls[v]]++] = v;
  };
  std::vector<Index> rbeg, rmem, cbeg, cmem;
  grouped(rc, nrc, rbeg, rmem);
  grouped(cc, ncc, cbeg, cmem);
  {
    std::vector<double> acc(ncc, 0.0), mag(ncc, 0.0), ref(ncc, 0.0);
    std::vector<Index> touched;
    for (Index r = 0; r < nrc; ++r) {
      const Index i0 = rrep[r];
      for (Index k = R.begin(i0); k < R.end(i0); ++k) ref[cc[R.index[k]]] += R.value[k];
      for (Index t = rbeg[r]; t < rbeg[r + 1]; ++t) {
        const Index i = rmem[t];
        if (i == i0) continue;
        if (row_lb[i] != row_lb[i0] || row_ub[i] != row_ub[i0]) return f;
        touched.clear();
        for (Index k = R.begin(i); k < R.end(i); ++k) {
          const Index c = cc[R.index[k]];
          if (acc[c] == 0.0 && mag[c] == 0.0) touched.push_back(c);
          acc[c] += R.value[k], mag[c] += std::abs(R.value[k]);
        }
        bool okr = true;
        for (Index c : touched) okr = okr && same(acc[c], ref[c], mag[c]);
        // Classes the representative has and this row lacks.
        Index hit = 0;
        for (Index k = R.begin(i0); k < R.end(i0); ++k) hit += mag[cc[R.index[k]]] == 0.0 && ref[cc[R.index[k]]] != 0.0;
        for (Index c : touched) acc[c] = 0.0, mag[c] = 0.0;
        if (!okr || hit > 0) return f;
      }
      for (Index k = R.begin(i0); k < R.end(i0); ++k) ref[cc[R.index[k]]] = 0.0;
    }
  }
  {
    std::vector<double> acc(nrc, 0.0), mag(nrc, 0.0), ref(nrc, 0.0);
    std::vector<Index> touched;
    for (Index c = 0; c < ncc; ++c) {
      const Index j0 = crep[c];
      for (Index k = A.begin(j0); k < A.end(j0); ++k) ref[rc[A.index[k]]] += A.value[k];
      for (Index t = cbeg[c]; t < cbeg[c + 1]; ++t) {
        const Index j = cmem[t];
        if (j == j0) continue;
        if (cost[j] != cost[j0] || col_lb[j] != col_lb[j0] || col_ub[j] != col_ub[j0]) return f;
        touched.clear();
        for (Index k = A.begin(j); k < A.end(j); ++k) {
          const Index r = rc[A.index[k]];
          if (acc[r] == 0.0 && mag[r] == 0.0) touched.push_back(r);
          acc[r] += A.value[k], mag[r] += std::abs(A.value[k]);
        }
        bool okc = true;
        for (Index r : touched) okc = okc && same(acc[r], ref[r], mag[r]);
        Index hit = 0;
        for (Index k = A.begin(j0); k < A.end(j0); ++k) hit += mag[rc[A.index[k]]] == 0.0 && ref[rc[A.index[k]]] != 0.0;
        for (Index r : touched) acc[r] = 0.0, mag[r] = 0.0;
        if (!okc || hit > 0) return f;
      }
      for (Index k = A.begin(j0); k < A.end(j0); ++k) ref[rc[A.index[k]]] = 0.0;
    }
  }
  // The folded LP.
  std::vector<Index> major, minor;
  std::vector<double> val;
  {
    std::vector<double> acc(ncc, 0.0);
    std::vector<char> seen(ncc, 0);
    std::vector<Index> touched;
    for (Index r = 0; r < nrc; ++r) {
      const Index i0 = rrep[r];
      touched.clear();
      for (Index k = R.begin(i0); k < R.end(i0); ++k) {
        const Index c = cc[R.index[k]];
        if (!seen[c]) seen[c] = 1, touched.push_back(c);
        acc[c] += R.value[k];
      }
      for (Index c : touched) {
        if (acc[c] != 0.0) major.push_back(c), minor.push_back(r), val.push_back(acc[c]);
        acc[c] = 0.0, seen[c] = 0;
      }
    }
  }
  f.A = SparseMatrix::from_triplets(ncc, nrc, major, minor, val);
  f.cost.resize(ncc), f.col_lb.resize(ncc), f.col_ub.resize(ncc);
  for (Index c = 0; c < ncc; ++c) {
    f.cost[c] = static_cast<double>(f.col_size[c]) * cost[crep[c]];
    f.col_lb[c] = col_lb[crep[c]], f.col_ub[c] = col_ub[crep[c]];
  }
  f.row_lb.resize(nrc), f.row_ub.resize(nrc);
  for (Index r = 0; r < nrc; ++r) f.row_lb[r] = row_lb[rrep[r]], f.row_ub[r] = row_ub[rrep[r]];
  f.num_col_classes = ncc, f.num_row_classes = nrc;
  f.col_class = std::move(cc), f.row_class = std::move(rc);
  f.ok = true;
  return f;
}

}  // namespace milp
