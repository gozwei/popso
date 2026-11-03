// Clique cuts from a conflict graph on binary literals (x_j = 1 or x_j = 0).
//
// Each problem row side is relaxed to a knapsack over literals with positive
// weights (as for cover cuts). With weights sorted a_1 >= a_2 >= ..., the
// longest prefix whose two smallest weights already exceed the capacity is a
// clique: any two of its literals cannot both be 1. Two literals conflict if
// they share such a clique; greedy maximal cliques over the fractional
// literals give cuts sum_{literals in K} l <= 1. Literal pairs that conflict by
// probing (CutData::probe) are further edges of the graph.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "milp/cuts/cut.hpp"

namespace milp {

namespace {

class CliqueSeparator : public Separator {
 public:
  const char* name() const override { return "clique"; }
  void separate(const SepaContext& ctx, std::vector<Cut>& out) override;

 private:
  void build(const MipProblem& p, const SparseMatrix& R, Index m0);
  bool built_ = false;
  std::vector<std::vector<Index>> cliques_;    // literal lists (2j: x_j, 2j+1: 1 - x_j)
  std::vector<std::vector<Index>> lit_cliques_;  // literal -> clique ids
};

bool binary(const MipProblem& p, Index j) { return p.is_int[j] && p.col_lb[j] == 0.0 && p.col_ub[j] == 1.0; }

void CliqueSeparator::build(const MipProblem& p, const SparseMatrix& R, Index m0) {
  built_ = true;
  lit_cliques_.assign(2 * static_cast<size_t>(p.num_cols()), {});
  std::vector<std::pair<double, Index>> items;
  for (Index i = 0; i < m0; ++i) {
    for (double sign : {1.0, -1.0}) {
      const double side = sign > 0 ? p.row_ub[i] : -p.row_lb[i];
      if (!is_finite(side)) continue;
      double b = side;
      items.clear();
      bool ok = true;
      for (Index k = R.begin(i); k < R.end(i) && ok; ++k) {
        const Index j = R.index[k];
        const double a = sign * R.value[k];
        if (binary(p, j)) {
          if (a > 0) {
            items.emplace_back(a, 2 * j);
          } else {
            items.emplace_back(-a, 2 * j + 1);
            b -= a;
          }
        } else {
          const double bound = a > 0 ? p.col_lb[j] : p.col_ub[j];
          ok = is_finite(bound);
          b -= a * bound;
        }
      }
      if (!ok || items.size() < 2) continue;
      std::sort(items.begin(), items.end(), [](const auto& u, const auto& v) { return u.first > v.first; });
      const double eps = 1e-9 * std::max(1.0, std::abs(b));
      size_t len = 1;
      while (len < items.size() && items[len - 1].first + items[len].first > b + eps) ++len;
      if (len < 2) continue;
      std::vector<Index> cl;
      for (size_t k = 0; k < len; ++k) cl.push_back(items[k].second);
      const Index id = static_cast<Index>(cliques_.size());
      for (Index l : cl) lit_cliques_[l].push_back(id);
      cliques_.push_back(std::move(cl));
    }
  }
}

void CliqueSeparator::separate(const SepaContext& ctx, std::vector<Cut>& out) {
  const MipProblem& p = ctx.p;
  if (!built_) build(p, ctx.lp.row_matrix(), ctx.num_problem_rows);
  const std::vector<Index>& cs = ctx.data.conflict_start;
  const std::vector<Index>& cl = ctx.data.conflict_lit;
  const bool probed = !cs.empty() && !cl.empty();
  if (cliques_.empty() && !probed) return;
  const Index n = p.num_cols();
  auto has_edges = [&](Index l) { return !lit_cliques_[l].empty() || (probed && cs[l + 1] > cs[l]); };
  auto value = [&](Index l) { return (l & 1) ? 1.0 - ctx.x[l / 2] : ctx.x[l / 2]; };
  // Literals with positive LP value (a violated clique may contain literals
  // at 1, e.g. one pigeon fully in a hole plus fractional others).
  std::vector<Index> frac;
  std::vector<char> is_frac(2 * static_cast<size_t>(n), 0);
  bool any_fractional = false;
  for (Index j = 0; j < n; ++j) {
    if (!binary(p, j)) continue;
    any_fractional = any_fractional || (ctx.x[j] > 1e-6 && ctx.x[j] < 1 - 1e-6);
    for (Index l : {2 * j, 2 * j + 1}) {
      if (has_edges(l) && value(l) > 1e-6) frac.push_back(l), is_frac[l] = 1;
    }
  }
  if (std::getenv("MILP_CLIQUE_DEBUG")) std::fprintf(stderr, "clique: %zu cliques, %zu candidate literals, frac %d\n", cliques_.size(), frac.size(), (int)any_fractional);
  if (!any_fractional) return;
  if (frac.size() < 2) return;
  std::sort(frac.begin(), frac.end(), [&](Index u, Index v) { return value(u) > value(v); });
  std::vector<Index> mark(2 * static_cast<size_t>(n), -1);  // neighbour stamp
  std::vector<std::vector<Index>> nbrs(2 * static_cast<size_t>(n));
  auto neighbours = [&](Index u) -> const std::vector<Index>& {
    std::vector<Index>& nb = nbrs[u];
    if (!nb.empty()) return nb;
    for (Index c : lit_cliques_[u]) {
      for (Index v : cliques_[c]) {
        if (v != u && is_frac[v] && mark[v] != u) mark[v] = u, nb.push_back(v);
      }
    }
    if (probed) {
      for (Index k = cs[u]; k < cs[u + 1]; ++k) {
        const Index v = cl[k];
        if (is_frac[v] && mark[v] != u) mark[v] = u, nb.push_back(v);
      }
    }
    std::sort(nb.begin(), nb.end(), [&](Index a, Index b) { return value(a) > value(b); });
    return nb;
  };
  std::vector<Index> adj_stamp(2 * static_cast<size_t>(n), -1);
  std::vector<std::vector<Index>> seen;  // sorted cliques already emitted
  // Maximal cliques: a violated clique over the positive literals is extended by literals at
  // value 0 adjacent to all its members (same violation, a stronger cut: SCIP's clique separator
  // also returns maximal cliques). Full adjacency over the row cliques and probing conflicts,
  // under a work budget per call.
  static const bool extend = [] { const char* v = std::getenv("MILP_CLIQUE_EXTEND"); return !(v && *v == '0'); }();
  const long long nnz = static_cast<long long>(p.A.index.size());
  long long ext_work = 0;
  const long long ext_budget = std::clamp<long long>(10 * nnz, 200000LL, 20000000LL);
  std::vector<int> ext_stamp(2 * static_cast<size_t>(n), 0);
  int ext_gen = 0;
  auto degree = [&](Index u) {
    long long d = probed ? cs[u + 1] - cs[u] : 0;
    for (Index c : lit_cliques_[u]) d += static_cast<long long>(cliques_[c].size());
    return d;
  };
  auto for_each_nbr = [&](Index u, auto&& fn) {
    for (Index c : lit_cliques_[u]) {
      for (Index v : cliques_[c]) {
        if (v != u) fn(v);
      }
    }
    if (probed) {
      for (Index k = cs[u]; k < cs[u + 1]; ++k) fn(cl[k]);
    }
  };
  constexpr long long kMaxExtDegree = 20000;
  std::vector<Index> ext_cand, in_k;
  auto extend_clique = [&](std::vector<Index>& K) {
    if (ext_work >= ext_budget) return;
    Index k0 = K[0];
    long long d0 = degree(k0);
    for (Index k : K) {
      const long long d = degree(k);
      if (d < d0) k0 = k, d0 = d;
    }
    if (d0 > kMaxExtDegree) return;
    ext_work += d0;
    // Candidates: neighbours of k0 not in K and not positive in x (those were tried by the greedy).
    ++ext_gen;
    for (Index k : K) ext_stamp[k] = ext_gen;
    ext_cand.clear();
    for_each_nbr(k0, [&](Index v) {
      if (ext_stamp[v] == ext_gen || is_frac[v]) return;
      ext_stamp[v] = ext_gen;
      ext_cand.push_back(v);
    });
    std::sort(ext_cand.begin(), ext_cand.end());
    for (Index w : ext_cand) {
      if (ext_work >= ext_budget) break;
      const long long dw = degree(w);
      if (dw > kMaxExtDegree) continue;
      ext_work += dw + static_cast<long long>(K.size());
      ++ext_gen;
      for_each_nbr(w, [&](Index v) { ext_stamp[v] = ext_gen; });
      bool all = true;
      for (Index k : K) all = all && (k == k0 || ext_stamp[k] == ext_gen);
      if (all) K.push_back(w);
    }
  };
  int made = 0;
  // Greedy growth depends on the start, so seed it with every pair {u, v}
  // (u's strongest neighbours first) and keep the distinct violated cliques.
  for (Index u : frac) {
    if (made >= 200) break;
    const std::vector<Index>& nu = neighbours(u);
    const size_t seeds = std::min<size_t>(nu.size(), 16);
    for (size_t s0 = 0; s0 < seeds && made < 200; ++s0) {
      std::vector<Index> K{u, nu[s0]};
      double sum = value(u) + value(nu[s0]);
      for (Index v : nu) {
        if (v == nu[s0]) continue;
        // v must be adjacent to every member of K.
        const std::vector<Index>& nv = neighbours(v);
        for (Index w : nv) adj_stamp[w] = v;
        bool all = true;
        for (Index k : K) all = all && adj_stamp[k] == v;
        if (!all) continue;
        K.push_back(v);
        sum += value(v);
      }
      if (sum <= 1.0 + 1e-6) continue;
      std::sort(K.begin(), K.end());
      if (std::find(seen.begin(), seen.end(), K) != seen.end()) continue;
      seen.push_back(K);
      if (extend) extend_clique(K);
      Cut c;
      c.origin = "clique";
      c.rhs = 1.0;
      for (Index l : K) {
        c.idx.push_back(l / 2);
        if (l & 1) {
          c.val.push_back(-1.0);
          c.rhs -= 1.0;
        } else {
          c.val.push_back(1.0);
        }
      }
      out.push_back(std::move(c));
      ++made;
    }
  }
}

}  // namespace

std::unique_ptr<Separator> make_clique_separator() { return std::make_unique<CliqueSeparator>(); }

}  // namespace milp
