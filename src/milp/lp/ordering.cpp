#include "milp/lp/ordering.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace milp {

namespace {

enum : int8_t { kVar = 0, kElement = 1, kMerged = 2, kDense = 3, kDead = 4 };

}  // namespace

std::vector<Index> amd_order(Index n, const std::vector<Index>& start, const std::vector<Index>& index,
                             const std::atomic<bool>* stop) {
  std::vector<Index> order;
  order.reserve(static_cast<size_t>(n));
  if (n <= 0) return order;

  // Symmetric adjacency without the diagonal and duplicates.
  std::vector<std::vector<Index>> adj(static_cast<size_t>(n));
  for (Index j = 0; j < n; ++j) {
    for (Index k = start[j]; k < start[j + 1]; ++k) {
      const Index i = index[k];
      if (i == j || i < 0 || i >= n) continue;
      adj[j].push_back(i);
      adj[i].push_back(j);
    }
  }
  for (auto& a : adj) {
    std::sort(a.begin(), a.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
  }

  std::vector<int8_t> status(static_cast<size_t>(n), kVar);
  // Dense nodes (degree far above average) would make every element they touch large; they are
  // ordered last.
  const double dense_limit = std::max(16.0, 10.0 * std::sqrt(static_cast<double>(n)));
  std::vector<Index> dense;
  for (Index j = 0; j < n; ++j) {
    if (static_cast<double>(adj[j].size()) > dense_limit) {
      status[j] = kDense;
      dense.push_back(j);
    }
  }
  if (!dense.empty()) {
    for (Index j = 0; j < n; ++j) {
      if (status[j] == kDense) {
        std::vector<Index>().swap(adj[j]);
        continue;
      }
      auto& a = adj[j];
      a.erase(std::remove_if(a.begin(), a.end(), [&](Index v) { return status[v] == kDense; }), a.end());
    }
  }

  std::vector<Index> nv(static_cast<size_t>(n), 1);      // supervariable weights
  std::vector<long long> deg(static_cast<size_t>(n), 0);  // approximate external degrees
  std::vector<std::vector<Index>> elem(static_cast<size_t>(n));   // adjacent elements of a variable
  std::vector<std::vector<Index>> evars(static_cast<size_t>(n));  // variables of an element
  std::vector<long long> esize(static_cast<size_t>(n), 0);        // weighted size of an element
  // Variables merged into a supervariable: child lists for the output order.
  std::vector<Index> child_head(static_cast<size_t>(n), -1), child_next(static_cast<size_t>(n), -1);
  // Degree buckets (doubly linked lists).
  std::vector<Index> head(static_cast<size_t>(n) + 1, -1), next(static_cast<size_t>(n), -1),
      prev(static_cast<size_t>(n), -1);
  auto bucket_insert = [&](Index i) {
    const Index d = static_cast<Index>(deg[i]);
    prev[i] = -1;
    next[i] = head[d];
    if (head[d] >= 0) prev[head[d]] = i;
    head[d] = i;
  };
  auto bucket_remove = [&](Index i) {
    const Index d = static_cast<Index>(deg[i]);
    if (prev[i] >= 0) next[prev[i]] = next[i];
    else head[d] = next[i];
    if (next[i] >= 0) prev[next[i]] = prev[i];
    prev[i] = next[i] = -1;
  };

  long long live = 0;
  for (Index j = 0; j < n; ++j) {
    if (status[j] != kVar) continue;
    deg[j] = static_cast<long long>(adj[j].size());
    ++live;
    bucket_insert(j);
  }

  std::vector<int> mark(static_cast<size_t>(n), 0);
  std::vector<int> wflag(static_cast<size_t>(n), 0);
  std::vector<long long> w(static_cast<size_t>(n), 0);
  int stamp = 0;
  auto new_stamp = [&]() {
    if (stamp == INT32_MAX - 2) {
      std::fill(mark.begin(), mark.end(), 0);
      std::fill(wflag.begin(), wflag.end(), 0);
      stamp = 0;
    }
    return ++stamp;
  };

  std::vector<Index> stack;
  auto emit = [&](Index p) {
    // p and, depth first, every variable merged into it.
    stack.assign(1, p);
    while (!stack.empty()) {
      const Index v = stack.back();
      stack.pop_back();
      order.push_back(v);
      for (Index c = child_head[v]; c >= 0; c = child_next[c]) stack.push_back(c);
    }
  };

  std::vector<Index> Lp;
  std::vector<std::pair<uint64_t, Index>> hashes;
  long long eliminated = 0;
  Index mindeg = 0;
  long long steps = 0;
  while (eliminated < live) {
    if (stop && (++steps & 255) == 0 && stop->load(std::memory_order_relaxed)) return {};
    while (mindeg <= n && head[mindeg] < 0) ++mindeg;
    if (mindeg > n) break;  // cannot happen while variables remain
    const Index p = head[mindeg];
    bucket_remove(p);

    // Lp: the variables adjacent to p through its elements and its variable neighbours.
    const int s = new_stamp();
    mark[p] = s;
    Lp.clear();
    long long degLp = 0;
    for (Index e : elem[p]) {
      if (status[e] != kElement) continue;
      for (Index v : evars[e]) {
        if (status[v] == kVar && mark[v] != s) {
          mark[v] = s;
          Lp.push_back(v);
          degLp += nv[v];
        }
      }
      status[e] = kDead;  // absorbed into p
      std::vector<Index>().swap(evars[e]);
    }
    for (Index v : adj[p]) {
      if (status[v] == kVar && mark[v] != s) {
        mark[v] = s;
        Lp.push_back(v);
        degLp += nv[v];
      }
    }
    status[p] = kElement;
    evars[p] = Lp;
    esize[p] = degLp;
    std::vector<Index>().swap(adj[p]);
    std::vector<Index>().swap(elem[p]);
    eliminated += nv[p];
    emit(p);

    // Clean the lists of the variables in Lp: dead elements go, p comes in; variable neighbours
    // inside Lp are now reachable through p and are pruned.
    for (Index i : Lp) {
      bucket_remove(i);
      auto& el = elem[i];
      size_t c = 0;
      for (Index e : el) {
        if (status[e] == kElement) el[c++] = e;
      }
      el.resize(c);
      el.push_back(p);
      auto& a = adj[i];
      c = 0;
      for (Index v : a) {
        if (status[v] == kVar && mark[v] != s) a[c++] = v;
      }
      a.resize(c);
    }
    // w(e) = |Le \ Lp| for the other elements of Lp's variables.
    for (Index i : Lp) {
      for (Index e : elem[i]) {
        if (e == p) continue;
        if (wflag[e] != s) {
          wflag[e] = s;
          w[e] = esize[e];
        }
        w[e] -= nv[i];
      }
    }
    // Approximate degrees; elements inside Lp are absorbed (aggressive absorption).
    const long long remaining = live - eliminated;
    hashes.clear();
    for (Index i : Lp) {
      long long d = degLp - nv[i];
      uint64_t h = 0;
      auto& el = elem[i];
      size_t c = 0;
      for (Index e : el) {
        if (e != p) {
          if (w[e] <= 0 || status[e] != kElement) {
            if (status[e] == kElement) {
              status[e] = kDead;
              std::vector<Index>().swap(evars[e]);
            }
            continue;
          }
          d += w[e];
        }
        el[c++] = e;
        h += static_cast<uint64_t>(e) * 0x9E3779B97F4A7C15ULL + 1;
      }
      el.resize(c);
      for (Index v : adj[i]) {
        d += nv[v];
        h += static_cast<uint64_t>(v) * 0xC2B2AE3D27D4EB4FULL + 7;
      }
      d = std::min(d, deg[i] + degLp - nv[i]);
      d = std::min(d, remaining - nv[i]);
      deg[i] = std::max<long long>(d, 0);
      h ^= (static_cast<uint64_t>(el.size()) << 32) ^ static_cast<uint64_t>(adj[i].size());
      hashes.emplace_back(h, i);
    }
    // Supervariables: variables of Lp with identical element and variable lists merge.
    std::sort(hashes.begin(), hashes.end());
    for (size_t a = 0; a < hashes.size();) {
      size_t b = a + 1;
      while (b < hashes.size() && hashes[b].first == hashes[a].first) ++b;
      for (size_t x = a; x < b; ++x) {
        const Index i = hashes[x].second;
        if (status[i] != kVar) continue;
        bool marked = false;
        int si = 0;
        for (size_t y = x + 1; y < b; ++y) {
          const Index j = hashes[y].second;
          if (status[j] != kVar || elem[j].size() != elem[i].size() || adj[j].size() != adj[i].size()) continue;
          if (!marked) {
            si = new_stamp();
            for (Index e : elem[i]) mark[e] = si;
            for (Index v : adj[i]) mark[v] = si;
            marked = true;
          }
          bool same = true;
          for (Index e : elem[j]) same = same && mark[e] == si;
          for (Index v : adj[j]) same = same && mark[v] == si;
          if (!same) continue;
          // Merge j into i.
          nv[i] += nv[j];
          deg[i] = std::max<long long>(0, deg[i] - nv[j]);
          nv[j] = 0;
          status[j] = kMerged;
          child_next[j] = child_head[i];
          child_head[i] = j;
          std::vector<Index>().swap(elem[j]);
          std::vector<Index>().swap(adj[j]);
        }
      }
      a = b;
    }
    for (Index i : Lp) {
      if (status[i] != kVar) continue;
      deg[i] = std::min<long long>(deg[i], n);
      bucket_insert(i);
      mindeg = std::min(mindeg, static_cast<Index>(deg[i]));
    }
  }
  for (Index j : dense) order.push_back(j);
  return order;
}

}  // namespace milp
