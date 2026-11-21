// Open-node storage (search/node_store.hpp): the queue against a reference ordered set under
// random pushes, pops, purges, compactions and drain toggles; bound-path segments against explicit
// per-node bounds; inline branching changes against one segment per child; stored bases (dense,
// sparse, pool rows) and their release accounting.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <tuple>
#include <vector>

#include "milp/search/node_store.hpp"
#include "test_framework.hpp"

using namespace milp;
using namespace milp::bnb;

namespace {

// Counts bases released through the owner (and their bytes).
struct CountingOwner : BasisOwner {
  long long released = 0, bytes = 0;
  void basis_released(const PackedBasis& b) override {
    ++released;
    bytes += b.bytes();
  }
};

Basis random_basis(std::mt19937& rng, int n, const std::vector<VarStatus>* ref, int changes) {
  Basis b;
  if (ref) {
    b.status = *ref;
    b.status.resize(static_cast<size_t>(n), VarStatus::Basic);
    for (int k = 0; k < changes; ++k) b.status[rng() % static_cast<unsigned>(n)] = static_cast<VarStatus>(rng() % 4);
  } else {
    b.status.resize(static_cast<size_t>(n));
    for (auto& s : b.status) s = static_cast<VarStatus>(rng() % 4);
  }
  return b;
}

using Changes = std::vector<std::tuple<Index, double, double>>;

// The bound changes of a node in application order: path segments oldest first, then the inline
// change.
Changes changes_of(const Node& n) {
  std::vector<const PathSeg*> segs;
  for (const PathSeg* s = n.path.get(); s; s = s->parent()) segs.push_back(s);
  Changes out;
  for (auto it = segs.rbegin(); it != segs.rend(); ++it) {
    (*it)->for_each([&](const BoundChange& c) { out.emplace_back(c.col, c.lb, c.ub); });
  }
  if (n.change.col >= 0) out.emplace_back(n.change.col, n.change.lb, n.change.ub);
  return out;
}

// What a reference node remembers (to compare a popped Node with what was pushed).
struct Ref {
  double bound, estimate, parent_obj, branch_dist;
  int depth, dir;
  long long parent, mem;
  Index branch_var;
  Changes changes;
  const PackedBasis* basis;
  Index anc_var;
  float anc_dist;
};

}  // namespace

// Stored bases: dense and sparse forms (the sparse one against a reference basis, with statuses
// beyond the reference's length), pool rows, the reference count and the release accounting.
TEST(packed_basis_round_trip) {
  std::mt19937 rng(7);
  CountingOwner owner;
  for (int trial = 0; trial < 300; ++trial) {
    const int n = 1 + static_cast<int>(rng() % 200);
    std::vector<VarStatus> refv(static_cast<size_t>(std::max(1, n - static_cast<int>(rng() % 5))));
    for (auto& s : refv) s = static_cast<VarStatus>(rng() % 4);
    const bool use_ref = trial % 3 != 0;
    const int changes = trial % 2 ? 2 : n;
    Basis b = random_basis(rng, n + 7, use_ref ? &refv : nullptr, changes);
    const Index size = trial % 4 == 0 ? -1 : n;
    std::vector<std::pair<int, VarStatus>> nb;
    int id = -1;
    const int np = static_cast<int>(rng() % 6);
    for (int k = 0; k < np; ++k) {
      id += 1 + static_cast<int>(rng() % 300);
      nb.emplace_back(id, static_cast<VarStatus>(1 + rng() % 3));
    }
    // Every third trial with steepest-edge weights behind the pool stream.
    std::vector<float> wts;
    if (trial % 3 == 1) {
      wts.resize(1 + rng() % 50);
      for (float& x : wts) x = 1.0f + static_cast<float>(rng() % 1000) / 7.0f;
    }
    PackedBasis* p = PackedBasis::make(b, size, use_ref ? &refv : nullptr, trial % 5 ? &nb : nullptr,
                                       wts.empty() ? nullptr : &wts);
    {
      std::vector<float> back_w{-1.0f};
      CHECK_EQ(p->has_weights(), !wts.empty());
      CHECK_EQ(p->weights(back_w), !wts.empty());
      if (!wts.empty()) {
        CHECK(back_w == wts);
        CHECK(p->bytes() >= static_cast<long long>(16 + 4 + 4 * wts.size()));
      }
    }
    const Index want = size >= 0 ? size : static_cast<Index>(b.status.size());
    CHECK_EQ(p->size(), want);
    if (!use_ref) CHECK(!p->sparse());
    if (use_ref && changes == 2 && n > 40) CHECK(p->sparse());
    Basis u;
    p->unpack(u, use_ref ? &refv : nullptr);
    CHECK_EQ(static_cast<Index>(u.status.size()), want);
    bool same = true;
    for (Index j = 0; j < want; ++j) same = same && u.status[j] == b.status[j];
    CHECK(same);
    std::vector<std::pair<int, VarStatus>> back;
    p->for_each_pool([&](int i, VarStatus st) { back.emplace_back(i, st); });
    if (trial % 5) {
      CHECK_EQ(p->npool(), np);
      CHECK(back == nb);
    } else {
      CHECK_EQ(p->npool(), 0);
    }
    CHECK(p->bytes() >= 16 && p->bytes() % 16 == 0);
    // Reference counting: two handles, the owner hears of the release once.
    const long long rel0 = owner.released;
    {
      BasisRef a(p, &owner);
      BasisRef c = a;
      a.reset();
      CHECK_EQ(owner.released, rel0);
      CHECK(static_cast<bool>(c));
    }
    CHECK_EQ(owner.released, rel0 + 1);
  }
}

// Bound paths on random trees: applying the segments oldest first (as the search does, by
// intersection) gives each node's explicit bounds; integral, fractional and infinite values, and
// sides equal to the original bounds (left out by the encoding).
TEST(path_segments_match_explicit_bounds) {
  std::mt19937 rng(11);
  const Index n = 40;
  std::vector<double> olb(n), oub(n);
  for (Index j = 0; j < n; ++j) {
    olb[j] = j % 7 == 0 ? -kInf : static_cast<double>(static_cast<int>(rng() % 5)) - 2.0;
    oub[j] = j % 5 == 0 ? kInf : olb[j] + 50.0 + (j % 7 == 0 ? 10.0 : 0.0);
    if (j % 7 == 0 && j % 5 != 0) oub[j] = 30.0;
  }
  std::atomic<long long> acct{0};
  struct TNode {
    PathRef path;
    std::vector<double> lb, ub;
  };
  std::vector<TNode> nodes(1);
  nodes[0].lb = olb;
  nodes[0].ub = oub;
  std::vector<uint8_t> seg;
  for (int k = 0; k < 3000; ++k) {
    const size_t par = rng() % nodes.size();
    TNode child;
    child.lb = nodes[par].lb;
    child.ub = nodes[par].ub;
    seg.clear();
    const int nch = 1 + static_cast<int>(rng() % 3);
    for (int c = 0; c < nch; ++c) {
      const Index j = static_cast<Index>(rng() % static_cast<unsigned>(n));
      double l = child.lb[j], u = child.ub[j];
      const double lo = is_finite(l) ? l : -60.0, hi = is_finite(u) ? u : 60.0;
      const double v = lo + (hi - lo) * static_cast<double>(rng() % 1000) / 1000.0;
      const int kind = static_cast<int>(rng() % 4);
      if (kind == 0) u = std::floor(v);
      else if (kind == 1) l = std::ceil(v);
      else if (kind == 2) l = v, u = std::max(v, std::min(u, v + 1.5));  // fractional values
      else l = olb[j];                                                    // a side at its original bound
      if (l > u) continue;
      PathSeg::encode({j, l, u}, olb[j], oub[j], seg);
      child.lb[j] = std::max(child.lb[j], l);
      child.ub[j] = std::min(child.ub[j], u);
    }
    child.path = PathRef(PathSeg::make(nodes[par].path.get(), seg, &acct));
    nodes.push_back(std::move(child));
  }
  bool ok = true;
  std::vector<const PathSeg*> segs;
  for (const TNode& t : nodes) {
    std::vector<double> lb = olb, ub = oub;
    segs.clear();
    for (const PathSeg* s = t.path.get(); s; s = s->parent()) segs.push_back(s);
    for (auto it = segs.rbegin(); it != segs.rend(); ++it) {
      (*it)->for_each([&](const BoundChange& c) {
        lb[c.col] = std::max(lb[c.col], c.lb);
        ub[c.col] = std::min(ub[c.col], c.ub);
      });
    }
    ok = ok && lb == t.lb && ub == t.ub;
  }
  CHECK(ok);
  CHECK(acct.load() > 0);
  nodes.clear();
  CHECK_EQ(acct.load(), 0LL);
}

// Inline branching changes (Node::change): a node gets a segment only when it branches (its own
// change, then its reduced-cost fixings), the children keep their branching bound inline (a child
// with extra changes gets a segment of its own). On random trees this gives every node the same
// changes in the same order as one segment per child (and a reduced-cost segment per branching)
// did, also after a round trip through the queue.
TEST(inline_changes_match_segment_per_child) {
  std::mt19937 rng(23);
  const Index n = 30;
  std::vector<double> olb(n, 0.0), oub(n, 10.0);
  for (Index j = 1; j < n; j += 2) olb[j] = -5.0;
  std::atomic<long long> acct{0};
  CountingOwner owner;
  struct TNode {
    PathRef old_path;  // one segment per child
    Node node;         // inline form
  };
  std::vector<TNode> tree(1);
  std::vector<uint8_t> seg;
  auto decoded = [&](const BoundChange& c) {
    return BoundChange{c.col, c.lb != olb[c.col] && c.lb > -kInf ? PathSeg::canonical(c.lb) : -kInf,
                       c.ub != oub[c.col] && c.ub < kInf ? PathSeg::canonical(c.ub) : kInf};
  };
  long long mismatches = 0;
  {
    NodePool pool(1, &owner, &acct);
    for (int k = 0; k < 4000; ++k) {
      const size_t xi = rng() % tree.size();
      // Reduced-cost fixings of the branching node (sometimes), the branching column and value.
      std::vector<BoundChange> rcfix;
      if (rng() % 3 == 0) {
        for (int r = 0; r < 1 + static_cast<int>(rng() % 3); ++r) rcfix.push_back({static_cast<Index>(rng() % n), 0.0, 3.0});
      }
      const Index j = static_cast<Index>(rng() % n);
      const double v = std::max(olb[j] + 0.05, static_cast<double>(rng() % 120) / 10.0 - 2.95);  // ceil(-0.95) = -0.0
      const double cur_lb = rng() % 2 ? olb[j] : std::max(olb[j], std::floor(v) - 1.0);
      const bool orbit = rng() % 5 == 0;
      // Old scheme.
      PathRef old_base = tree[xi].old_path;
      if (!rcfix.empty()) {
        seg.clear();
        for (const BoundChange& f : rcfix) PathSeg::encode(f, olb[f.col], oub[f.col], seg);
        old_base = PathRef(PathSeg::make(old_base.get(), seg, &acct));
      }
      // New scheme: the branching node's segment.
      const Node& x = tree[xi].node;
      PathRef base = x.path;
      if (x.change.col >= 0 || !rcfix.empty()) {
        seg.clear();
        if (x.change.col >= 0) PathSeg::encode(x.change, olb[x.change.col], oub[x.change.col], seg);
        for (const BoundChange& f : rcfix) PathSeg::encode(f, olb[f.col], oub[f.col], seg);
        base = PathRef(PathSeg::make(x.path.get(), seg, &acct));
      }
      for (int side = 0; side < 2; ++side) {
        const BoundChange bc = side == 0 ? BoundChange{j, cur_lb, std::floor(v)} : BoundChange{j, std::ceil(v), oub[j]};
        seg.clear();
        PathSeg::encode(bc, olb[j], oub[j], seg);
        const bool extra = side == 0 && orbit;
        if (extra) PathSeg::encode({static_cast<Index>((j + 1) % n), olb[0], std::floor(v)}, olb[0], oub[0], seg);
        TNode c;
        c.old_path = PathRef(PathSeg::make(old_base.get(), seg, &acct));
        c.node.id = static_cast<long long>(tree.size()) + 1;
        c.node.branch_var = j;
        c.node.branch_dir = side == 0 ? -1 : 1;
        if (extra) {
          c.node.path = PathRef(PathSeg::make(base.get(), seg, &acct));
        } else {
          c.node.path = base;
          c.node.change = decoded(bc);
        }
        // Half of the nodes go through the queue and back.
        if (rng() % 2 == 0) {
          pool.push(std::move(c.node));
          Node out;
          if (!pool.try_pop(kInf, &out, false)) ++mismatches;
          c.node = std::move(out);
        }
        Node old_view;
        old_view.path = c.old_path;
        if (changes_of(old_view) != changes_of(c.node)) ++mismatches;
        tree.push_back(std::move(c));
      }
    }
  }
  CHECK_EQ(mismatches, 0LL);
  tree.clear();
  CHECK_EQ(acct.load(), 0LL);
}

// The queue against a reference: outside drain mode every pop is the reference's first node in the
// best-bound or best-estimate order (key, deeper first, smaller id) and the popped node is the
// pushed one field by field; in drain mode a pop is either the deepest node or the best-bound one.
// Purges, compactions (also under memory pressure) and drain toggles in between; the lower bound
// and the size follow the reference; the queue's bytes fall after it shrinks; every basis and path
// is released at the end.
TEST(node_pool_matches_reference) {
  for (unsigned seed = 1; seed <= 4; ++seed) {
    std::mt19937 rng(seed);
    CountingOwner owner;
    std::atomic<long long> acct{0};
    long long bases_made = 0;
    {
      NodePool pool(1, &owner, &acct);
      using K = std::tuple<double, int, long long>;  // (key, -depth, id)
      std::set<K> by_bound, by_est;
      std::map<long long, Ref> refs;
      long long next_id = 1;
      const double est_w = 0.5;
      auto ekey = [&](double b, double e) { return is_finite(e) ? b + est_w * (e - b) : b; };
      std::vector<VarStatus> refv(30, VarStatus::Basic);
      bool drain = false;
      long long pops = 0, mismatches = 0, drain_pops = 0, drain_best = 0;
      long long peak_alloc = 0;
      size_t peak_count = 0;
      auto erase_ref = [&](long long id) {
        const Ref& r = refs.at(id);
        by_bound.erase(K{r.bound, -r.depth, id});
        by_est.erase(K{ekey(r.bound, r.estimate), -r.depth, id});
        refs.erase(id);
      };
      for (int step = 0; step < 60000; ++step) {
        // Phases: grow, shrink (compaction), grow again.
        const int phase = step < 25000 ? 0 : step < 45000 ? 1 : 2;
        const unsigned r = rng() % 100;
        const bool do_push = phase == 1 ? r < 20 : r < 65;
        if (do_push || refs.empty()) {
          Node nd;
          nd.bound = static_cast<double>(rng() % 50) * 0.5;  // many exact ties
          nd.estimate = rng() % 10 == 0 ? -kInf : nd.bound + static_cast<double>(rng() % 20);
          nd.depth = static_cast<int>(rng() % 25);
          nd.id = next_id;
          next_id += 1 + static_cast<long long>(rng() % 3);
          nd.parent = rng() % 7 == 0 ? -1 : nd.id - 1 - static_cast<long long>(rng() % 1000);
          nd.branch_dir = static_cast<int>(rng() % 3) - 1;
          nd.branch_var = static_cast<Index>(rng() % 100) - 1;
          nd.branch_dist = static_cast<double>(rng() % 100) / 100.0;
          nd.parent_obj = rng() % 3 == 0 ? -kInf : nd.bound - 0.25;
          if (rng() % 4 == 0) nd.mem = 100 + static_cast<long long>(rng() % 50);
          if (rng() % 3 != 0) nd.anc_var = static_cast<Index>(rng() % 100), nd.anc_dist = static_cast<float>(rng() % 200) / 100.0f - 1.0f;
          // A path of 0-2 segments (encoded changes) and an inline change: packable (integral
          // sides, infinite sides, the column = branch_var or no pseudocost column) or not
          // (fractional value, another column), which the pool turns into a path segment.
          std::vector<uint8_t> seg;
          const int nseg = static_cast<int>(rng() % 3);
          for (int k = 0; k < nseg; ++k) {
            seg.clear();
            const Index col = static_cast<Index>(rng() % 50);
            PathSeg::encode({col, static_cast<double>(rng() % 5), 5.0 + static_cast<double>(rng() % 5)}, 0.0, 9.0, seg);
            nd.path = PathRef(PathSeg::make(nd.path.get(), seg, &acct));
          }
          const unsigned kind = rng() % 8;
          if (kind < 6) {
            const Index col = kind == 5 ? static_cast<Index>(rng() % 50) : nd.branch_var;
            if (kind == 4) nd.branch_var = -1, nd.branch_dir = 0;
            const double lb = rng() % 3 == 0 ? -kInf : static_cast<double>(static_cast<int>(rng() % 9)) - 4.0;
            double ub = rng() % 3 == 0 ? kInf : (std::isinf(lb) ? 0.0 : lb) + static_cast<double>(rng() % 4);
            if (kind == 3) ub = 2.5;  // fractional: not packable
            if (col >= 0) nd.change = BoundChange{col, lb, ub};
          }
          if (rng() % 2 == 0) {
            const Basis b = random_basis(rng, 30, &refv, 3);
            nd.basis = BasisRef(PackedBasis::make(b, -1, &refv, nullptr), &owner);
            ++bases_made;
          }
          refs[nd.id] = Ref{nd.bound, nd.estimate, nd.parent_obj, nd.branch_dist, nd.depth, nd.branch_dir,
                            nd.parent, nd.mem, nd.branch_var, changes_of(nd), nd.basis.get(), nd.anc_var, nd.anc_dist};
          by_bound.insert(K{nd.bound, -nd.depth, nd.id});
          by_est.insert(K{ekey(nd.bound, nd.estimate), -nd.depth, nd.id});
          pool.push(std::move(nd));
        } else if (r < 97) {
          const bool by_estimate = rng() % 2 == 0;
          Node out;
          if (!pool.try_pop(kInf, &out, by_estimate)) {
            ++mismatches;
            continue;
          }
          ++pops;
          if (drain) {
            ++drain_pops;
            int maxd = -1;
            for (const auto& kv : refs) maxd = std::max(maxd, kv.second.depth);
            const bool best = std::get<2>(*by_bound.begin()) == out.id;
            const bool deepest = out.depth == maxd;
            drain_best += best && !deepest;
            if (!best && !deepest) ++mismatches;
          } else {
            const K& want = by_estimate ? *by_est.begin() : *by_bound.begin();
            if (std::get<2>(want) != out.id) ++mismatches;
          }
          auto it = refs.find(out.id);
          if (it == refs.end()) {
            ++mismatches;
            continue;
          }
          const Ref& rf = it->second;
          const bool fields = out.bound == rf.bound && (out.estimate == rf.estimate || (std::isinf(out.estimate) && std::isinf(rf.estimate))) &&
                              out.parent_obj == rf.parent_obj && out.branch_dist == rf.branch_dist &&
                              out.depth == rf.depth && out.branch_dir == rf.dir && out.parent == rf.parent &&
                              out.mem == rf.mem && out.branch_var == rf.branch_var && changes_of(out) == rf.changes &&
                              out.anc_var == rf.anc_var && out.anc_dist == rf.anc_dist &&
                              out.basis.get() == rf.basis;
          if (!fields) ++mismatches;
          erase_ref(out.id);
        } else if (r < 98) {
          // Purge at a random cutoff (rarely: it empties much of the queue).
          if (rng() % 8 != 0) continue;
          const double cutoff = 18.0 + static_cast<double>(rng() % 16) * 0.5;
          const long long removed = pool.purge(cutoff);
          long long want = 0;
          std::vector<long long> ids;
          for (const auto& kv : refs) {
            if (kv.second.bound >= cutoff) ids.push_back(kv.first);
          }
          want = static_cast<long long>(ids.size());
          for (long long id : ids) erase_ref(id);
          if (removed != want) ++mismatches;
        } else if (r < 99) {
          drain = !drain;
          pool.set_drain(drain);
        } else {
          pool.set_pressure(rng() % 2 == 0);
        }
        if (pool.size() != refs.size()) ++mismatches;
        const double lb = pool.lower_bound();
        const double want_lb = refs.empty() ? kInf : std::get<0>(*by_bound.begin());
        if (lb != want_lb) ++mismatches;
        if (refs.size() > peak_count) peak_count = refs.size();
        peak_alloc = std::max(peak_alloc, pool.alloc_bytes());
      }
      CHECK_EQ(mismatches, 0LL);
      CHECK(pops > 10000);
      CHECK(drain_pops > 100);
      CHECK(drain_best > 0);  // best-bound pops happen in drain mode too
      CHECK(pool.compactions() > 0);
      CHECK(pool.alloc_bytes() < peak_alloc);
      CHECK(pool.bytes() <= pool.alloc_bytes());
      CHECK(peak_count > 1000);
      // Empty it: every node comes out once.
      Node out;
      while (pool.try_pop(kInf, &out, false)) erase_ref(out.id);
      CHECK(refs.empty());
    }
    CHECK_EQ(owner.released, bases_made);
    CHECK_EQ(acct.load(), 0LL);
  }
}

// Drain mode: the k deepest : 1 best bound alternation, and a parent id too far away to encode
// comes back as no parent at all (-2), never as a wrong one.
TEST(node_pool_drain_alternation) {
  CountingOwner owner;
  NodePool pool(1, &owner);
  long long id = 1;
  for (int k = 0; k < 2000; ++k) {
    Node nd;
    nd.bound = static_cast<double>(k % 97);
    nd.depth = k % 13;
    nd.id = id++;
    nd.parent = k == 0 ? nd.id - (1LL << 40) : nd.id - 1;
    pool.push(std::move(nd));
  }
  pool.set_drain(true);
  int bad = 0;
  for (int t = 0; t < 90; ++t) {
    const double lb = pool.lower_bound();
    Node out;
    CHECK(pool.try_pop(kInf, &out, true));
    if (t % 9 == 8) bad += out.bound != lb;  // every 9th pop by best bound
    else bad += out.depth != 12;             // the others deepest first (153 nodes at depth 12)
    if (out.id == 1) CHECK_EQ(out.parent, -2LL);
    else CHECK_EQ(out.parent, out.id - 1);
  }
  CHECK_EQ(bad, 0);
  CHECK_EQ(pool.drain_ratio(), 8);  // starts at 8, adapts after 1024 pops
}
