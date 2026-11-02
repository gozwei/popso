#include "milp/presolve/clique_merge.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <unordered_map>
#include <utility>

namespace milp {

namespace {

bool binary(const MipProblem& p, Index j) { return p.is_int[j] && p.col_lb[j] == 0.0 && p.col_ub[j] == 1.0; }

uint64_t edge_key(Index a, Index b) {
  if (a > b) std::swap(a, b);
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}

}  // namespace

Index merge_cliques(MipProblem& p, Index max_new_rows) {
  const Index n = p.num_cols(), m = p.num_rows();
  const SparseMatrix R = p.A.transpose();
  // Set-packing rows: all binary, all coefficients 1, row_ub == 1, no lower side.
  std::vector<char> packing(m, 0);
  std::vector<std::vector<Index>> adj(n);
  std::unordered_map<uint64_t, Index> pair_row;  // edge -> pairwise row
  constexpr Index kMaxEdgeRow = 64;
  for (Index i = 0; i < m; ++i) {
    if (p.row_ub[i] != 1.0 || (is_finite(p.row_lb[i]) && p.row_lb[i] > 0.0)) continue;
    const Index len = R.end(i) - R.begin(i);
    if (len < 2 || len > kMaxEdgeRow) continue;
    bool ok = true;
    for (Index k = R.begin(i); k < R.end(i) && ok; ++k) ok = binary(p, R.index[k]) && R.value[k] == 1.0;
    if (!ok) continue;
    packing[i] = 1;
    for (Index a = R.begin(i); a < R.end(i); ++a)
      for (Index b = a + 1; b < R.end(i); ++b) {
        adj[R.index[a]].push_back(R.index[b]);
        adj[R.index[b]].push_back(R.index[a]);
      }
    if (len == 2) pair_row.emplace(edge_key(R.index[R.begin(i)], R.index[R.begin(i) + 1]), i);
  }
  if (pair_row.size() < 3) return 0;
  for (auto& a : adj) {
    std::sort(a.begin(), a.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
  }
  auto adjacent = [&](Index u, Index v) { return std::binary_search(adj[u].begin(), adj[u].end(), v); };

  // Grow a maximal clique from every uncovered pairwise row.
  std::vector<char> remove(m, 0);
  std::vector<std::vector<Index>> new_rows;
  for (const auto& [key, row] : pair_row) {
    if (remove[row] || static_cast<Index>(new_rows.size()) >= max_new_rows) continue;
    const Index u = static_cast<Index>(key >> 32), v = static_cast<Index>(key & 0xffffffffu);
    std::vector<Index> K{u, v};
    // Candidates: common neighbours of u and v, highest degree first.
    std::vector<Index> cand;
    std::set_intersection(adj[u].begin(), adj[u].end(), adj[v].begin(), adj[v].end(), std::back_inserter(cand));
    std::sort(cand.begin(), cand.end(), [&](Index a, Index b) { return adj[a].size() > adj[b].size(); });
    for (Index w : cand) {
      bool all = true;
      for (Index k : K) all = all && adjacent(k, w);
      if (all) K.push_back(w);
    }
    if (K.size() < 3) continue;
    // Pairwise rows inside K are dominated by the clique row.
    Index covered = 0;
    for (size_t a = 0; a < K.size(); ++a)
      for (size_t b = a + 1; b < K.size(); ++b) {
        auto it = pair_row.find(edge_key(K[a], K[b]));
        if (it != pair_row.end() && !remove[it->second]) remove[it->second] = 1, ++covered;
      }
    if (covered >= 2) {
      std::sort(K.begin(), K.end());
      new_rows.push_back(std::move(K));
    }
  }
  if (new_rows.empty()) return 0;

  // Rebuild the matrix row-wise: kept rows, then clique rows.
  std::vector<Index> J, I;
  std::vector<double> V, lb, ub;
  Index out = 0, removed = 0;
  for (Index i = 0; i < m; ++i) {
    if (remove[i]) {
      ++removed;
      continue;
    }
    for (Index k = R.begin(i); k < R.end(i); ++k) J.push_back(R.index[k]), I.push_back(out), V.push_back(R.value[k]);
    lb.push_back(p.row_lb[i]);
    ub.push_back(p.row_ub[i]);
    ++out;
  }
  for (const auto& K : new_rows) {
    for (Index j : K) J.push_back(j), I.push_back(out), V.push_back(1.0);
    lb.push_back(-kInf);
    ub.push_back(1.0);
    ++out;
  }
  p.A = SparseMatrix::from_triplets(n, out, J, I, V);
  p.row_lb = std::move(lb);
  p.row_ub = std::move(ub);
  return removed;
}

}  // namespace milp

namespace milp {

CliqueExtension find_clique_extensions(const MipProblem& p, long long work_limit) {
  CliqueExtension res;
  CliqueExtendStats& st = res.stats;
  const Index n = p.num_cols(), m = p.num_rows();
  const SparseMatrix R = p.A.transpose();
  long long work = 0;
  // Cliques from the rows (sorted column lists); kind: 1 packing row (<= 1), 2 partitioning row (= 1).
  std::vector<Index> cstart{0}, cidx;
  std::vector<char> kind(m, 0);
  std::vector<std::pair<double, Index>> pos;
  for (Index i = 0; i < m; ++i) {
    const Index rb = R.begin(i), re = R.end(i);
    if (re - rb < 2) continue;
    work += re - rb;
    bool unit = p.row_ub[i] == 1.0;
    for (Index k = rb; k < re && unit; ++k) unit = R.value[k] == 1.0 && binary(p, R.index[k]);
    if (unit) kind[i] = p.row_lb[i] >= 1.0 ? 2 : (p.row_lb[i] <= 0.0 ? 1 : 0);
    for (int side = 0; side < 2; ++side) {
      const double s = side == 0 ? 1.0 : -1.0;
      const double b = side == 0 ? p.row_ub[i] : -p.row_lb[i];
      if (!is_finite(b) || std::abs(b) > 1e9) continue;
      double minact = 0.0;
      bool ok = true;
      pos.clear();
      for (Index k = rb; k < re && ok; ++k) {
        const Index j = R.index[k];
        const double a = s * R.value[k];
        const double lo = a > 0 ? p.col_lb[j] : p.col_ub[j];
        if (!is_finite(lo) || std::abs(lo) > 1e9) ok = false;
        minact += a * lo;
        if (a > 0 && binary(p, j)) pos.push_back({a, j});
      }
      if (!ok || pos.size() < 2) continue;
      const double slack = b - minact;
      // Pairs must violate the side by more than the feasibility tolerance.
      const double tol = 1e-6 * (1.0 + std::abs(b)) + 1e-9 * std::abs(minact);
      std::sort(pos.begin(), pos.end(), [](const auto& x, const auto& y) { return x.first > y.first || (x.first == y.first && x.second < y.second); });
      if (!(pos[0].first + pos[1].first > slack + tol)) continue;
      size_t t = 1;
      while (t + 1 < pos.size() && pos[t].first + pos[t + 1].first > slack + tol) ++t;
      const size_t c0 = cidx.size();
      for (size_t q = 0; q <= t; ++q) cidx.push_back(pos[q].second);
      std::sort(cidx.begin() + static_cast<std::ptrdiff_t>(c0), cidx.end());
      cstart.push_back(static_cast<Index>(cidx.size()));
      work += static_cast<long long>(pos.size()) * 4;
    }
  }
  const Index nq = static_cast<Index>(cstart.size()) - 1;
  st.cliques = nq;
  if (nq == 0) return res;
  std::vector<double> ub = p.col_ub;  // binaries fixed to 0 here
  // Column -> cliques incidence (CSR).
  std::vector<Index> qstart(n + 1, 0), qidx(cidx.size());
  for (Index j : cidx) ++qstart[j + 1];
  for (Index j = 0; j < n; ++j) qstart[j + 1] += qstart[j];
  {
    std::vector<Index> fill(qstart.begin(), qstart.end() - 1);
    for (Index q = 0; q < nq; ++q)
      for (Index k = cstart[q]; k < cstart[q + 1]; ++k) qidx[fill[cidx[k]]++] = q;
  }
  auto in_clique = [&](Index q, Index v) {
    return std::binary_search(cidx.begin() + cstart[q], cidx.begin() + cstart[q + 1], v);
  };
  auto adjacent = [&](Index u, Index v) {
    if (qstart[u + 1] - qstart[u] > qstart[v + 1] - qstart[v]) std::swap(u, v);
    for (Index k = qstart[u]; k < qstart[u + 1]; ++k) {
      ++work;
      if (in_clique(qidx[k], v)) return true;
    }
    return false;
  };
  // Neighbourhood size of j (entries visited when enumerating its neighbours).
  auto nbsize = [&](Index j) {
    long long s = 0;
    for (Index k = qstart[j]; k < qstart[j + 1]; ++k) s += cstart[qidx[k] + 1] - cstart[qidx[k]];
    return s;
  };

  std::vector<Index> stamp(n, -1), last(n, -1), cnt(n, 0);
  std::vector<std::vector<Index>> ext(m);  // extension of packing row i
  std::vector<Index> cand, members;
  for (Index i = 0; i < m && work < work_limit; ++i) {
    if (!kind[i]) continue;
    members.clear();
    for (Index k = R.begin(i); k < R.end(i); ++k) members.push_back(R.index[k]);
    long long cost = 0;
    Index k0 = members[0];
    long long best = -1;
    for (Index j : members) {
      const long long s = nbsize(j);
      cost += s;
      if (best < 0 || s < best) best = s, k0 = j;
    }
    if (work + cost > work_limit) continue;
    work += cost;
    // Candidates: neighbours of k0 outside the row; count how many members each one is adjacent to.
    for (Index j : members) stamp[j] = -2 - i;  // members are excluded
    cand.clear();
    for (Index k = qstart[k0]; k < qstart[k0 + 1]; ++k) {
      const Index q = qidx[k];
      for (Index t = cstart[q]; t < cstart[q + 1]; ++t) {
        const Index w = cidx[t];
        if (stamp[w] == i || stamp[w] == -2 - i) continue;
        stamp[w] = i;
        cnt[w] = 0;
        last[w] = -1;
        cand.push_back(w);
      }
    }
    if (cand.empty()) continue;
    for (Index j : members) {
      for (Index k = qstart[j]; k < qstart[j + 1]; ++k) {
        const Index q = qidx[k];
        for (Index t = cstart[q]; t < cstart[q + 1]; ++t) {
          const Index w = cidx[t];
          if (stamp[w] != i || last[w] == j) continue;
          last[w] = j;
          ++cnt[w];
        }
      }
    }
    const Index need = static_cast<Index>(members.size());
    cand.erase(std::remove_if(cand.begin(), cand.end(), [&](Index w) { return cnt[w] != need; }), cand.end());
    if (cand.empty()) continue;
    if (kind[i] == 2) {
      // x_w = 1 would force every member of the partitioning row to 0.
      for (Index w : cand) {
        if (ub[w] > 0.0) ub[w] = 0.0, ++st.fixed, res.fixed.push_back(w);
      }
      continue;
    }
    // Greedy: most connected candidates first; each must be adjacent to the ones already added.
    std::sort(cand.begin(), cand.end(), [&](Index a, Index b) {
      const Index da = qstart[a + 1] - qstart[a], db = qstart[b + 1] - qstart[b];
      return da > db || (da == db && a < b);
    });
    auto& e = ext[i];
    for (Index w : cand) {
      if (ub[w] == 0.0) continue;  // fixed above
      bool ok = true;
      for (Index u : e) {
        if (!adjacent(u, w)) {
          ok = false;
          break;
        }
      }
      if (ok) e.push_back(w);
      if (work > work_limit) break;
    }
    if (!e.empty()) ++st.extended, st.added += static_cast<long long>(e.size());
  }

  // Dominated packing rows: contained in another packing or partitioning row (after extension).
  std::vector<std::vector<Index>> sets(m);
  std::vector<Index> order;
  for (Index i = 0; i < m; ++i) {
    if (!kind[i]) continue;
    auto& s = sets[i];
    for (Index k = R.begin(i); k < R.end(i); ++k) {
      if (ub[R.index[k]] > 0.0 || kind[i] == 2) s.push_back(R.index[k]);
    }
    s.insert(s.end(), ext[i].begin(), ext[i].end());
    std::sort(s.begin(), s.end());
    order.push_back(i);
  }
  std::sort(order.begin(), order.end(), [&](Index a, Index b) {
    if (sets[a].size() != sets[b].size()) return sets[a].size() > sets[b].size();
    if (kind[a] != kind[b]) return kind[a] > kind[b];
    return a < b;
  });
  std::vector<char> removed(m, 0);
  std::vector<std::vector<Index>> kept_of(n);  // kept rows containing a column
  std::vector<Index> mark(n, -1);
  for (Index i : order) {
    const auto& s = sets[i];
    if (kind[i] == 1 && work < work_limit) {
      if (s.size() <= 1) {
        removed[i] = 1;  // at most one free binary left: the row is redundant
        ++st.dominated;
        continue;
      }
      Index base = s[0];
      for (Index j : s) {
        if (kept_of[j].size() < kept_of[base].size()) base = j;
      }
      for (Index j : s) mark[j] = i;
      bool dom = false;
      for (Index t : kept_of[base]) {
        const auto& T = sets[t];
        if (T.size() < s.size()) continue;
        Index hit = 0;
        for (Index j : T) hit += mark[j] == i;
        work += static_cast<long long>(T.size());
        if (hit == static_cast<Index>(s.size())) {
          dom = true;
          break;
        }
      }
      if (dom) {
        removed[i] = 1;
        ++st.dominated;
        continue;
      }
    }
    for (Index j : s) kept_of[j].push_back(i);
  }
  st.work = work;
  for (Index i = 0; i < m; ++i) {
    if (removed[i]) {
      res.removed.push_back(i);
    } else if (kind[i] == 1 && !ext[i].empty()) {
      res.ext.push_back({i, std::move(ext[i])});
    }
  }
  return res;
}

bool extend_cliques(MipProblem& p, long long work_limit, CliqueExtendStats* stats) {
  CliqueExtension ce = find_clique_extensions(p, work_limit);
  if (stats) *stats = ce.stats;
  for (Index j : ce.fixed) p.col_ub[j] = 0.0;
  if (ce.ext.empty() && ce.removed.empty()) return !ce.fixed.empty();
  const Index n = p.num_cols(), m = p.num_rows();
  const SparseMatrix R = p.A.transpose();
  std::vector<char> removed(m, 0);
  for (Index i : ce.removed) removed[i] = 1;
  std::vector<const std::vector<Index>*> ext(m, nullptr);
  for (const auto& e : ce.ext) ext[e.first] = &e.second;
  std::vector<Index> J, I;
  std::vector<double> V, lb, ub;
  Index out = 0;
  for (Index i = 0; i < m; ++i) {
    if (removed[i]) continue;
    for (Index k = R.begin(i); k < R.end(i); ++k) J.push_back(R.index[k]), I.push_back(out), V.push_back(R.value[k]);
    if (ext[i]) {
      for (Index j : *ext[i]) J.push_back(j), I.push_back(out), V.push_back(1.0);
    }
    lb.push_back(ext[i] ? -kInf : p.row_lb[i]);
    ub.push_back(p.row_ub[i]);
    ++out;
  }
  p.A = SparseMatrix::from_triplets(n, out, J, I, V);
  p.row_lb = std::move(lb);
  p.row_ub = std::move(ub);
  return true;
}

}  // namespace milp
