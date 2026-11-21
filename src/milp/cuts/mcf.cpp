#include "milp/cuts/mcf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>

namespace milp {

namespace {

double sgn(double v) { return v > 0 ? 1.0 : -1.0; }

Index find_root(std::vector<Index>& parent, Index a) {
  while (parent[a] != a) a = parent[a] = parent[parent[a]];
  return a;
}

}  // namespace

int g_mcf_drop_long = -1;

namespace {

McfNetwork detect(const MipProblem& p, bool drop_long) {
  McfNetwork net;
  const Index n = p.num_cols(), m = p.num_rows();
  if (n == 0 || m == 0) return net;
  const SparseMatrix& C = p.A;
  const SparseMatrix R = p.A.transpose();
  auto active = [&](Index j) { return p.col_lb[j] < p.col_ub[j]; };

  // 1. Candidate flow rows: at least two active entries, all on continuous columns, equal magnitudes.
  std::vector<double> scale(m, 0.0);  // > 0: candidate (the common |coefficient|)
  for (Index i = 0; i < m; ++i) {
    if (!is_finite(p.row_lb[i]) && !is_finite(p.row_ub[i])) continue;
    Index len = 0;
    double s = 0.0;
    bool ok = true;
    for (Index k = R.begin(i); k < R.end(i) && ok; ++k) {
      const Index j = R.index[k];
      if (!active(j)) continue;
      const double a = std::abs(R.value[k]);
      ok = !p.is_int[j] && a > 0.0 && (s == 0.0 || std::abs(a - s) <= 1e-9 * s);
      if (s == 0.0) s = a;
      ++len;
    }
    if (ok && len >= 2) scale[i] = s;
  }
  // 2. A flow column lies in at most two flow rows: columns in more drop their inequality rows first,
  // then all of them.
  std::vector<Index> cnt(n);
  // drop_long (MILP_MCF_LONG=1, the second attempt of detect_mcf_network): a column in more than two
  // candidate equality rows drops only the longest of them (per
  // pass), not all: models that carry, next to the commodities' conservation rows, equations with the same
  // unit coefficients over many flow columns (an arc's total flow T_a = sum_k f_ak, or the conservation of
  // the totals, n5-3) kept no flow row at all, although those rows are much longer than a conservation row.
  std::vector<Index> row_len;
  if (drop_long) {
    row_len.assign(m, 0);
    for (Index i = 0; i < m; ++i) {
      for (Index k = R.begin(i); k < R.end(i); ++k) row_len[i] += active(R.index[k]);
    }
  }
  for (int pass = 0; pass < (drop_long ? 12 : 4); ++pass) {
    bool changed = false;
    for (Index j = 0; j < n; ++j) {
      cnt[j] = 0;
      for (Index k = C.begin(j); k < C.end(j); ++k) cnt[j] += scale[C.index[k]] > 0.0;
    }
    for (Index j = 0; j < n; ++j) {
      if (cnt[j] <= 2 || p.is_int[j] || !active(j)) continue;
      bool any_ineq = false;
      for (Index k = C.begin(j); k < C.end(j); ++k) {
        const Index i = C.index[k];
        any_ineq = any_ineq || (scale[i] > 0.0 && p.row_lb[i] != p.row_ub[i]);
      }
      if (drop_long && !any_ineq) {
        // The longest candidate row of the column, if it is longer than the column's shortest: dropped alone
        // (the counts are those of the pass's start, so a row dropped for another column is seen next pass).
        Index longest = -1, shortest = -1;
        for (Index k = C.begin(j); k < C.end(j); ++k) {
          const Index i = C.index[k];
          if (scale[i] <= 0.0) continue;
          if (longest < 0 || row_len[i] > row_len[longest]) longest = i;
          if (shortest < 0 || row_len[i] < row_len[shortest]) shortest = i;
        }
        Index left = 0;
        for (Index k = C.begin(j); k < C.end(j); ++k) left += scale[C.index[k]] > 0.0;
        if (left > 2 && longest >= 0 && row_len[longest] > row_len[shortest]) {
          scale[longest] = 0.0, changed = true;
          continue;
        }
        if (left <= 2) continue;
      }
      for (Index k = C.begin(j); k < C.end(j); ++k) {
        const Index i = C.index[k];
        if (scale[i] > 0.0 && (!any_ineq || p.row_lb[i] != p.row_ub[i])) scale[i] = 0.0, changed = true;
      }
    }
    if (!changed) break;
  }
  for (Index j = 0; j < n; ++j) {
    if (!active(j)) continue;
    cnt[j] = 0;
    for (Index k = C.begin(j); k < C.end(j); ++k) cnt[j] += scale[C.index[k]] > 0.0;
    if (cnt[j] > 2) return net;  // not resolved: no network
  }
  // 3. Commodities: connected components of the flow rows; orientations o_r with o_r a_rj = -o_s a_sj for a
  // column j in rows r and s (outflow +1 at its tail, -1 at its head).
  std::vector<Index> comp(m, -1);
  std::vector<double> orient(m, 0.0);
  std::vector<char> comp_bad;
  std::vector<Index> comp_size, queue;
  Index ncomp = 0;
  for (Index r0 = 0; r0 < m; ++r0) {
    if (scale[r0] <= 0.0 || comp[r0] >= 0) continue;
    comp[r0] = ncomp, orient[r0] = 1.0;
    comp_bad.push_back(0), comp_size.push_back(0);
    queue.assign(1, r0);
    for (size_t q = 0; q < queue.size(); ++q) {
      const Index r = queue[q];
      ++comp_size[ncomp];
      for (Index k = R.begin(r); k < R.end(r); ++k) {
        const Index j = R.index[k];
        if (!active(j)) continue;
        const double sr = orient[r] * sgn(R.value[k]);
        for (Index t = C.begin(j); t < C.end(j); ++t) {
          const Index r2 = C.index[t];
          if (r2 == r || scale[r2] <= 0.0) continue;
          const double o2 = -sr * sgn(C.value[t]);
          if (comp[r2] < 0) {
            comp[r2] = ncomp, orient[r2] = o2;
            queue.push_back(r2);
          } else if (orient[r2] != o2) {
            comp_bad[ncomp] = 1;
          }
        }
      }
    }
    ++ncomp;
  }
  std::vector<Index> comm_of(ncomp, -1);
  Index ncomm = 0;
  for (Index c = 0; c < ncomp; ++c) {
    if (!comp_bad[c] && comp_size[c] >= 2) comm_of[c] = ncomm++;
  }
  std::vector<char> flow_row(m, 0);
  for (Index i = 0; i < m; ++i) flow_row[i] = scale[i] > 0.0 && comm_of[comp[i]] >= 0;
  // 4. Flow columns: tail row (outflow) and head row (inflow), -1 outside.
  std::vector<Index> tail(n, -1), head(n, -1);
  net.col_comm.assign(n, -1);
  for (Index j = 0; j < n; ++j) {
    if (p.is_int[j] || !active(j)) continue;
    for (Index k = C.begin(j); k < C.end(j); ++k) {
      const Index i = C.index[k];
      if (!flow_row[i]) continue;
      net.col_comm[j] = comm_of[comp[i]];
      (orient[i] * sgn(C.value[k]) > 0 ? tail[j] : head[j]) = i;
    }
  }
  // 5. Capacity rows: flow columns with one sign, integer columns with the other (at least one), nothing
  // else; the row side bounding the flows must be finite. A commodity has one flow column in the row, or two
  // in opposite directions (an undirected edge: both directions share the capacity, as in cost266-UUE).
  // Each flow column takes the capacity row with the most flow columns.
  std::vector<Index> best_cap(n, -1), best_len(n, 0);
  std::vector<double> best_coef(n, 0.0);
  std::vector<Index> seen_comm(ncomm, -1), seen_col(ncomm, -1), seen_cnt(ncomm, 0);
  std::vector<char> undirected(m, 0);
  for (Index i = 0; i < m; ++i) {
    if (flow_row[i]) continue;
    Index nflow = 0, nint = 0;
    double fsign = 0.0;
    bool ok = true;
    for (Index k = R.begin(i); k < R.end(i) && ok; ++k) {
      const Index j = R.index[k];
      if (!active(j)) continue;
      const double a = R.value[k];
      if (p.is_int[j]) {
        ++nint;
      } else if (net.col_comm[j] >= 0) {
        const Index q = net.col_comm[j];
        ok = fsign == 0.0 || sgn(a) == fsign;
        if (seen_comm[q] != i) {
          seen_comm[q] = i, seen_col[q] = j, seen_cnt[q] = 1;
        } else {
          const Index f = seen_col[q];
          ok = ok && ++seen_cnt[q] == 2 && tail[j] == head[f] && head[j] == tail[f];
          undirected[i] = 1;
        }
        fsign = sgn(a);
        ++nflow;
      } else {
        ok = false;
      }
    }
    if (!ok || nflow == 0 || nint == 0) {
      undirected[i] = 0;
      continue;
    }
    if (!is_finite(fsign > 0 ? p.row_ub[i] : p.row_lb[i])) continue;
    bool opposite = false;
    for (Index k = R.begin(i); k < R.end(i); ++k) {
      const Index j = R.index[k];
      opposite = opposite || (active(j) && p.is_int[j] && sgn(R.value[k]) != fsign);
    }
    if (!opposite) continue;
    for (Index k = R.begin(i); k < R.end(i); ++k) {
      const Index j = R.index[k];
      if (!active(j) || p.is_int[j] || nflow <= best_len[j]) continue;
      best_cap[j] = i, best_len[j] = nflow, best_coef[j] = R.value[k];
    }
  }
  // 6. Arcs (one per capacity row, one per uncapacitated flow column) and nodes. Flow rows of different
  // commodities with the same set of incident capacity rows are one node (independent of the orientation;
  // the only identification for undirected edges). Then the commodities' orientations are aligned (a
  // commodity whose rows read inflow - outflow is flipped: at a directed capacity row its columns' tails
  // are the other commodities' heads), and the tail (head) rows of the columns of a directed capacitated
  // arc become one node.
  std::vector<Index> parent(m);
  std::iota(parent.begin(), parent.end(), 0);
  {
    // Signatures: the sorted capacity rows of a flow row's columns.
    std::vector<std::pair<uint64_t, Index>> sig;  // (hash, row)
    std::vector<std::vector<Index>> caps(m);
    for (Index i = 0; i < m; ++i) {
      if (!flow_row[i]) continue;
      for (Index k = R.begin(i); k < R.end(i); ++k) {
        const Index j = R.index[k];
        if (active(j) && best_cap[j] >= 0) caps[i].push_back(best_cap[j]);
      }
      std::sort(caps[i].begin(), caps[i].end());
      caps[i].erase(std::unique(caps[i].begin(), caps[i].end()), caps[i].end());
      if (caps[i].empty()) continue;
      uint64_t h = 1469598103934665603ULL;
      for (Index c : caps[i]) h = (h ^ static_cast<uint64_t>(c)) * 1099511628211ULL;
      sig.emplace_back(h, i);
    }
    std::sort(sig.begin(), sig.end());
    std::vector<Index> group, gcomm;
    for (size_t a = 0; a < sig.size();) {
      size_t b = a + 1;
      while (b < sig.size() && sig[b].first == sig[a].first) ++b;
      group.clear(), gcomm.clear();
      for (size_t t = a; t < b; ++t) {
        if (caps[sig[t].second] == caps[sig[a].second]) group.push_back(sig[t].second), gcomm.push_back(comm_of[comp[sig[t].second]]);
      }
      std::sort(gcomm.begin(), gcomm.end());
      if (group.size() >= 2 && std::adjacent_find(gcomm.begin(), gcomm.end()) == gcomm.end()) {
        for (Index r : group) parent[find_root(parent, r)] = find_root(parent, group[0]);
      }
      a = b;
    }
  }
  std::vector<Index> cap_first(m, -1);  // first flow column of each directed capacity row
  {
    // Orientation parity of the commodities relative to each other (union-find with parity).
    std::vector<Index> cpar(ncomm);
    std::vector<int> cflip(ncomm, 0);
    std::iota(cpar.begin(), cpar.end(), 0);
    auto croot = [&](Index k, int& par) {
      par = 0;
      while (cpar[k] != k) par ^= cflip[k], k = cpar[k];
      return k;
    };
    auto same = [&](Index r1, Index r2) { return r1 >= 0 && r2 >= 0 && find_root(parent, r1) == find_root(parent, r2); };
    for (Index j = 0; j < n; ++j) {
      if (net.col_comm[j] < 0 || best_cap[j] < 0 || undirected[best_cap[j]]) continue;
      Index& f = cap_first[best_cap[j]];
      if (f < 0) {
        f = j;
        continue;
      }
      int rel;
      if (same(tail[j], tail[f]) || same(head[j], head[f])) {
        rel = 0;
      } else if (same(tail[j], head[f]) || same(head[j], tail[f])) {
        rel = 1;
      } else {
        continue;
      }
      int p1, p2;
      const Index k1 = croot(net.col_comm[j], p1), k2 = croot(net.col_comm[f], p2);
      if (k1 != k2) cpar[k1] = k2, cflip[k1] = p1 ^ p2 ^ rel;
    }
    std::vector<int> flip(ncomm);
    for (Index k = 0; k < ncomm; ++k) {
      int par;
      croot(k, par);
      flip[k] = par;
    }
    for (Index i = 0; i < m; ++i) {
      if (flow_row[i] && flip[comm_of[comp[i]]]) orient[i] = -orient[i];
    }
    for (Index j = 0; j < n; ++j) {
      if (net.col_comm[j] >= 0 && flip[net.col_comm[j]]) std::swap(tail[j], head[j]);
    }
  }
  for (Index j = 0; j < n; ++j) {
    if (net.col_comm[j] < 0 || best_cap[j] < 0 || undirected[best_cap[j]]) continue;
    const Index f = cap_first[best_cap[j]];
    if (f == j) continue;
    if (tail[j] >= 0 && tail[f] >= 0) parent[find_root(parent, tail[j])] = find_root(parent, tail[f]);
    if (head[j] >= 0 && head[f] >= 0) parent[find_root(parent, head[j])] = find_root(parent, head[f]);
    if ((tail[j] < 0) != (tail[f] < 0) || (head[j] < 0) != (head[f] < 0)) return McfNetwork{};
  }
  std::vector<Index> node_of(m, -1);
  Index nnodes = 0;
  for (Index i = 0; i < m; ++i) {
    if (!flow_row[i]) continue;
    const Index r = find_root(parent, i);
    if (node_of[r] < 0) node_of[r] = nnodes++;
    node_of[i] = node_of[r];
  }
  // A node has at most one flow row per commodity.
  {
    std::vector<std::pair<Index, Index>> nc;
    for (Index i = 0; i < m; ++i) {
      if (flow_row[i]) nc.emplace_back(node_of[i], comm_of[comp[i]]);
    }
    std::sort(nc.begin(), nc.end());
    if (std::adjacent_find(nc.begin(), nc.end()) != nc.end()) return McfNetwork{};
  }
  net.col_arc.assign(n, -1);
  net.col_cap.assign(n, 0.0);
  std::vector<Index> cap_arc(m, -1);
  bool any_cap = false;
  for (Index j = 0; j < n; ++j) {
    if (net.col_comm[j] < 0) continue;
    const Index t = tail[j] >= 0 ? node_of[tail[j]] : -1, h = head[j] >= 0 ? node_of[head[j]] : -1;
    const Index c = best_cap[j];
    if (c >= 0 && cap_arc[c] >= 0) {
      net.col_arc[j] = cap_arc[c];
    } else {
      net.col_arc[j] = static_cast<Index>(net.arcs.size());
      McfNetwork::Arc a;
      a.tail = t, a.head = h, a.cap_row = c;
      net.arcs.push_back(a);
      if (c >= 0) cap_arc[c] = net.col_arc[j], any_cap = true;
    }
    if (c >= 0) net.col_cap[j] = best_coef[j];
  }
  if (!any_cap || nnodes < 2) return McfNetwork{};
  net.num_nodes = nnodes, net.num_comms = ncomm;
  for (Index i = 0; i < m; ++i) {
    if (flow_row[i]) net.flow_rows.push_back({i, node_of[i], comm_of[comp[i]], orient[i] / scale[i]});
  }
  std::stable_sort(net.flow_rows.begin(), net.flow_rows.end(),
                   [](const McfNetwork::FlowRow& a, const McfNetwork::FlowRow& b) { return a.node < b.node; });
  net.node_start.assign(nnodes + 1, 0);
  for (const auto& fr : net.flow_rows) ++net.node_start[fr.node + 1];
  for (Index v = 0; v < nnodes; ++v) net.node_start[v + 1] += net.node_start[v];
  return net;
}

}  // namespace

McfNetwork detect_mcf_network(const MipProblem& p) {
  static const bool env_long = [] { const char* v = std::getenv("MILP_MCF_LONG"); return v && *v && *v != '0'; }();
  McfNetwork net = detect(p, false);
  // Only where the first rule finds nothing: the second rule can lose a network the first one finds
  // (neos-3627168-kasai), so it never replaces a found one.
  if (net.empty() && (g_mcf_drop_long >= 0 ? g_mcf_drop_long != 0 : env_long)) net = detect(p, true);
  return net;
}

}  // namespace milp