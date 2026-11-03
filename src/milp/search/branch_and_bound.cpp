#include "milp/search/branch_and_bound.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <tuple>

#include "milp/cuts/cut.hpp"
#include "milp/core/memory.hpp"
#include "milp/cuts/cut_loop.hpp"
#include "milp/heuristics/diving.hpp"
#include "milp/heuristics/feaspump.hpp"
#include "milp/heuristics/fj.hpp"
#include "milp/heuristics/localsearch.hpp"
#include "milp/heuristics/rounding.hpp"
#include "milp/heuristics/submip.hpp"
#include "milp/lp/crossover.hpp"
#include "milp/lp/simplex.hpp"
#include "milp/search/conflict.hpp"
#include "milp/search/orbitope.hpp"
#include "milp/search/propagation.hpp"
#include "milp/search/symmetry.hpp"

namespace milp {

namespace {

// Search parameter override from the environment (tuning experiments; defaults unchanged).
double knob(const char* name, double def) {
  const char* v = std::getenv(name);
  return v ? std::atof(v) : def;
}

struct BoundChange {
  Index col;
  double lb, ub;
};

// Bound changes from the root are a persistent list of segments shared between a node and its
// descendants, so each node costs O(1) memory, not O(depth). A segment holds the changes one
// branching added (the branching bound, the parent's reduced-cost fixings, orbital fixings),
// compactly: per change a 32-bit word (column, flags) and the bounds it sets, each as int32 when
// integral and small, else as double. A side that equals the column's original bound is left out
// (applying it changed nothing: bounds are intersected with the global bounds, which lie within).
// Memory accounting: every segment adds its allocation size to *acct while it lives.
class PathSeg {
 public:
  // New segment with one reference (the caller's); it holds a reference on parent.
  static PathSeg* make(const PathSeg* parent, const std::vector<uint8_t>& data, std::atomic<long long>* acct) {
    const size_t bytes = sizeof(PathSeg) + data.size();
    void* mem = ::operator new(bytes);
    PathSeg* s = new (mem) PathSeg();
    s->refs_.store(1, std::memory_order_relaxed);
    s->nbytes_ = static_cast<uint32_t>(data.size());
    s->parent_ = parent;
    s->acct_ = acct;
    if (parent) parent->retain();
    if (!data.empty()) std::memcpy(s->data(), data.data(), data.size());
    if (acct) acct->fetch_add(static_cast<long long>(alloc_size(bytes)), std::memory_order_relaxed);
    return s;
  }
  void retain() const { refs_.fetch_add(1, std::memory_order_relaxed); }
  // Drops a reference; frees the segment (and, iteratively, ancestors losing their last reference).
  static void release(const PathSeg* s) {
    while (s && s->refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      const PathSeg* parent = s->parent_;
      if (s->acct_) s->acct_->fetch_sub(static_cast<long long>(alloc_size(sizeof(PathSeg) + s->nbytes_)), std::memory_order_relaxed);
      s->~PathSeg();
      ::operator delete(const_cast<PathSeg*>(s));
      s = parent;
    }
  }
  const PathSeg* parent() const { return parent_; }
  size_t bytes() const { return alloc_size(sizeof(PathSeg) + nbytes_); }
  // Calls f(const BoundChange&) for the changes of this segment in the order they were added.
  template <class F>
  void for_each(F&& f) const {
    const uint8_t* q = data();
    const uint8_t* end = q + nbytes_;
    while (q < end) {
      uint32_t w;
      std::memcpy(&w, q, 4);
      q += 4;
      BoundChange c{static_cast<Index>(w & kColMask), -kInf, kInf};
      if (w & kHasLb) c.lb = read_value(q, (w & kLbInt) != 0);
      if (w & kHasUb) c.ub = read_value(q, (w & kUbInt) != 0);
      f(c);
    }
  }
  // Appends the encoding of c to out (orig_lb / orig_ub: the column's original bounds).
  static void encode(const BoundChange& c, double orig_lb, double orig_ub, std::vector<uint8_t>& out) {
    uint32_t w = static_cast<uint32_t>(c.col);
    const bool has_lb = c.lb != orig_lb && c.lb > -kInf, has_ub = c.ub != orig_ub && c.ub < kInf;
    const bool lb_int = has_lb && fits_int(c.lb), ub_int = has_ub && fits_int(c.ub);
    if (has_lb) w |= kHasLb | (lb_int ? kLbInt : 0u);
    if (has_ub) w |= kHasUb | (ub_int ? kUbInt : 0u);
    const size_t pos = out.size();
    out.resize(pos + 4);
    std::memcpy(out.data() + pos, &w, 4);
    if (has_lb) write_value(c.lb, lb_int, out);
    if (has_ub) write_value(c.ub, ub_int, out);
  }

 private:
  static constexpr uint32_t kColMask = (1u << 28) - 1, kHasLb = 1u << 28, kLbInt = 1u << 29, kHasUb = 1u << 30,
                            kUbInt = 1u << 31;
  static size_t alloc_size(size_t b) { return (b + 15) & ~size_t(15); }
  static bool fits_int(double v) { return v == std::floor(v) && std::abs(v) < 2e9; }
  static void write_value(double v, bool as_int, std::vector<uint8_t>& out) {
    const size_t pos = out.size();
    if (as_int) {
      const int32_t i = static_cast<int32_t>(v);
      out.resize(pos + 4);
      std::memcpy(out.data() + pos, &i, 4);
    } else {
      out.resize(pos + 8);
      std::memcpy(out.data() + pos, &v, 8);
    }
  }
  static double read_value(const uint8_t*& q, bool as_int) {
    if (as_int) {
      int32_t i;
      std::memcpy(&i, q, 4);
      q += 4;
      return static_cast<double>(i);
    }
    double v;
    std::memcpy(&v, q, 8);
    q += 8;
    return v;
  }
  uint8_t* data() { return reinterpret_cast<uint8_t*>(this + 1); }
  const uint8_t* data() const { return reinterpret_cast<const uint8_t*>(this + 1); }
  mutable std::atomic<int32_t> refs_{0};
  uint32_t nbytes_ = 0;
  const PathSeg* parent_ = nullptr;
  std::atomic<long long>* acct_ = nullptr;
};
static_assert(sizeof(PathSeg) % 8 == 0, "segment data must stay aligned");

// Owning handle of a path segment (reference counted).
class PathRef {
 public:
  PathRef() = default;
  explicit PathRef(PathSeg* adopt) : p_(adopt) {}
  PathRef(const PathRef& o) : p_(o.p_) {
    if (p_) p_->retain();
  }
  PathRef(PathRef&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
  PathRef& operator=(const PathRef& o) {
    if (o.p_) o.p_->retain();
    PathSeg::release(p_);
    p_ = o.p_;
    return *this;
  }
  PathRef& operator=(PathRef&& o) noexcept {
    if (this != &o) {
      PathSeg::release(p_);
      p_ = o.p_;
      o.p_ = nullptr;
    }
    return *this;
  }
  ~PathRef() { PathSeg::release(p_); }
  const PathSeg* get() const { return p_; }
  explicit operator bool() const { return p_ != nullptr; }

 private:
  const PathSeg* p_ = nullptr;
};

// Little-endian base-128 varints (basis encodings).
inline void put_varint(uint64_t v, std::vector<uint8_t>& out) {
  while (v >= 0x80) {
    out.push_back(static_cast<uint8_t>(v | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<uint8_t>(v));
}
inline uint64_t get_varint(const uint8_t*& q) {
  uint64_t v = 0;
  for (int sh = 0;; sh += 7) {
    const uint8_t b = *q++;
    v |= static_cast<uint64_t>(b & 0x7f) << sh;
    if (!(b & 0x80)) return v;
  }
}

// Warm-start basis stored with queued nodes. The first `size` statuses (columns and base rows) are
// kept either packed at 2 bits each or, when shorter, as the positions where they differ from a
// reference basis (the first basis stored in the search; beyond its length the reference is Basic),
// varint-encoded as (gap << 2 | status). With the dynamic cut pool, pool cut rows are identified by
// pool id and only the nonbasic ones are kept (rows with a basic logical can be dropped or added
// without losing dual feasibility), as ascending ids, varint (id gap << 2 | status). A cut deleted
// from the pool meanwhile stays alive while a stored basis refers to it (Search::pool_hold), so the
// node can still be warm started.
struct PackedBasis {
  using Ref = std::shared_ptr<const std::vector<VarStatus>>;
  Index size = 0;
  bool sparse = false;
  int npool = 0;              // nonbasic pool rows
  std::vector<uint8_t> bits;  // dense statuses or the sparse difference stream
  std::vector<uint8_t> pool;  // nonbasic pool rows
  Ref ref;                    // reference basis of the sparse form

  PackedBasis(const Basis& b, Index size_in, Ref ref_in) : size(size_in >= 0 ? size_in : static_cast<Index>(b.status.size())) {
    // Sparse form first (abandoned once it reaches the dense size).
    const size_t dense = (static_cast<size_t>(size) + 3) / 4;
    if (ref_in) {
      const std::vector<VarStatus>& r = *ref_in;
      Index last = -1;
      for (Index j = 0; j < size && bits.size() < dense; ++j) {
        const VarStatus rj = j < static_cast<Index>(r.size()) ? r[j] : VarStatus::Basic;
        if (b.status[j] == rj) continue;
        put_varint((static_cast<uint64_t>(j - last - 1) << 2) | static_cast<uint8_t>(b.status[j]), bits);
        last = j;
      }
      if (bits.size() < dense) {
        sparse = true;
        ref = std::move(ref_in);
        bits.shrink_to_fit();
        return;
      }
      bits.clear();
    }
    bits.assign(dense, 0);
    bits.shrink_to_fit();
    for (Index j = 0; j < size; ++j) bits[j >> 2] |= static_cast<uint8_t>(static_cast<uint8_t>(b.status[j]) << ((j & 3) * 2));
  }
  // nb: (pool id, status), ids ascending.
  void set_pool(const std::vector<std::pair<int, VarStatus>>& nb) {
    int last = -1;
    for (const auto& [id, st] : nb) {
      put_varint((static_cast<uint64_t>(id - last - 1) << 2) | static_cast<uint8_t>(st), pool);
      last = id;
    }
    npool = static_cast<int>(nb.size());
    pool.shrink_to_fit();
  }
  template <class F>
  void for_each_pool(F&& f) const {
    const uint8_t* q = pool.data();
    int id = -1;
    for (int k = 0; k < npool; ++k) {
      const uint64_t v = get_varint(q);
      id += static_cast<int>(v >> 2) + 1;
      f(id, static_cast<VarStatus>(v & 3));
    }
  }
  void unpack(Basis& b) const {
    b.status.resize(size);
    if (!sparse) {
      for (Index j = 0; j < size; ++j) b.status[j] = static_cast<VarStatus>((bits[j >> 2] >> ((j & 3) * 2)) & 3);
      return;
    }
    const std::vector<VarStatus>& r = *ref;
    const Index nr = std::min<Index>(size, static_cast<Index>(r.size()));
    std::copy(r.begin(), r.begin() + nr, b.status.begin());
    std::fill(b.status.begin() + nr, b.status.end(), VarStatus::Basic);
    const uint8_t* q = bits.data();
    const uint8_t* end = q + bits.size();
    Index j = -1;
    while (q < end) {
      const uint64_t v = get_varint(q);
      j += static_cast<Index>(v >> 2) + 1;
      b.status[j] = static_cast<VarStatus>(v & 3);
    }
  }
  // Heap bytes (object, shared_ptr control block with deleter, the two arrays).
  long long bytes() const {
    return static_cast<long long>(sizeof(PackedBasis) + 48 + ((bits.capacity() + 15) & ~size_t(15)) +
                                  ((pool.capacity() + 15) & ~size_t(15)));
  }
};

struct Node {
  double bound = -kInf;  // lower bound on the LP objective (without offset)
  double estimate = -kInf;  // estimated best solution value in the subtree (pseudocosts)
  int depth = 0;
  PathRef path;  // bound changes from the root (newest segment first)
  std::shared_ptr<const PackedBasis> basis;  // warm start; dropped under memory pressure
  long long mem = 0;                       // estimated bytes owned while queued
  long long id = 0, parent = -1;
  Index branch_var = -1;
  int branch_dir = 0;        // -1 down, +1 up
  double branch_dist = 0.0;  // how far the branching moved the variable
  double parent_obj = -kInf;  // bound before MILP_SB_BOUNDS raised it (pseudocost updates)
  // Symmetry: a group stabilizing this node's branching constraints; null if trivial.
  std::shared_ptr<const PermGroup> sym;
};

// Open nodes, selectable by best bound, by best estimate or (drain mode, under memory pressure)
// deepest first. Nodes live in slots; the heaps hold keys into them and skip stale keys lazily.
class NodePool {
 public:
  explicit NodePool(int workers) : wbound_(workers, kInf) {}
  // Deterministic mode: one bound slot per virtual worker (nodes they keep between rounds).
  void resize_workers(int workers) {
    std::lock_guard<std::mutex> lk(mu_);
    wbound_.assign(std::max<size_t>(wbound_.size(), static_cast<size_t>(workers)), kInf);
  }

  void push(Node&& n) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      bytes_ += n.mem;
      int s;
      if (!free_.empty()) {
        s = free_.back();
        free_.pop_back();
      } else {
        s = static_cast<int>(slots_.size());
        slots_.emplace_back();
      }
      // Drain mode keeps a depth heap instead of the estimate heap (rebuilt when it ends).
      const Key kb{n.bound, n.id, n.depth, s};
      if (drain_) push_key(dheap_, depth_key(n, s));
      else push_key(eheap_, est_key(n, s));
      slots_[s] = std::move(n);
      ++count_;
      push_key(bheap_, kb);
      update_heap_bytes();
    }
    cv_.notify_one();
  }

  // Blocks until a node is available (returns true, worker marked active) or
  // the search is over (returns false). by_estimate: take the best-estimate node.
  bool pop(int w, const std::atomic<double>& cutoff, const std::atomic<bool>& stop, Node* out, bool by_estimate) {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
      if (stop.load()) return false;
      if (count_ > 0 && min_bound() >= cutoff.load()) clear();
      if (count_ > 0) {
        take(by_estimate, out);
        ++active_;
        wbound_[w] = out->bound;
        return true;
      }
      if (active_ == 0) {
        cv_.notify_all();
        return false;
      }
      cv_.wait_for(lk, std::chrono::milliseconds(20));
    }
  }

  // Deterministic mode: non-blocking pop without active-worker accounting.
  bool try_pop(double cutoff, Node* out, bool by_estimate) {
    std::lock_guard<std::mutex> lk(mu_);
    if (count_ > 0 && min_bound() >= cutoff) clear();
    if (count_ == 0) return false;
    take(by_estimate, out);
    return true;
  }

  void set_worker_bound(int w, double b) {
    std::lock_guard<std::mutex> lk(mu_);
    wbound_[w] = b;
  }

  void done(int w) {
    std::lock_guard<std::mutex> lk(mu_);
    --active_;
    wbound_[w] = kInf;
    if (active_ == 0 && count_ == 0) cv_.notify_all();
  }

  double lower_bound() {
    std::lock_guard<std::mutex> lk(mu_);
    double lb = count_ == 0 ? kInf : min_bound();
    for (double b : wbound_) lb = std::min(lb, b);
    return lb;
  }

  size_t size() {
    std::lock_guard<std::mutex> lk(mu_);
    return count_;
  }

  // Drain mode: pops take the deepest node (ties: lowest bound), so the tree is worked off depth
  // first and the number of open nodes stops growing.
  void set_drain(bool on) {
    std::lock_guard<std::mutex> lk(mu_);
    if (on == drain_) return;
    drain_ = on;
    std::vector<Key>().swap(on ? eheap_ : dheap_);
    std::vector<Key>& h = on ? dheap_ : eheap_;
    h.clear();
    for (size_t s = 0; s < slots_.size(); ++s) {
      if (slots_[s].depth >= 0) h.push_back(on ? depth_key(slots_[s], static_cast<int>(s)) : est_key(slots_[s], static_cast<int>(s)));
    }
    heapify(h);
    update_heap_bytes();
  }
  bool drain() const { return drain_; }

  // Drops warm-start bases of queued nodes, shallowest (in drain mode: processed last) and then
  // worst-bound first, until done() says enough memory was freed. Returns the number dropped.
  template <class Done>
  long long strip_bases(Done&& done) {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::tuple<int, double, int>> order;  // (depth, -bound, slot)
    for (size_t s = 0; s < slots_.size(); ++s) {
      if (slots_[s].depth >= 0 && slots_[s].basis) order.emplace_back(slots_[s].depth, -slots_[s].bound, static_cast<int>(s));
    }
    std::sort(order.begin(), order.end());
    long long dropped = 0;
    for (const auto& [d, nb, s] : order) {
      if (done()) break;
      slots_[s].basis.reset();
      ++dropped;
    }
    return dropped;
  }

  // Diagnostics (verbosity 2 memory report): what the queued nodes hold, in one pass over the
  // slots (bases shared by queued siblings and the bound paths are accounted by the search).
  struct Breakdown {
    size_t nodes = 0;
    long long with_basis = 0, sparse_bases = 0, pool_rows = 0;  // queued nodes with a basis, sparse ones, their pool rows
    long long groups = 0, group_bytes = 0;  // symmetry groups (Node::mem)
    double depth = 0.0;                     // mean depth
  };
  Breakdown breakdown() {
    Breakdown b;
    std::lock_guard<std::mutex> lk(mu_);
    for (const Node& n : slots_) {
      if (n.depth < 0) continue;
      ++b.nodes;
      b.depth += n.depth;
      if (n.basis) {
        ++b.with_basis;
        b.sparse_bases += n.basis->sparse ? 1 : 0;
        b.pool_rows += n.basis->npool;
      }
      if (n.sym) ++b.groups;
    }
    b.group_bytes = bytes_.load();
    if (b.nodes > 0) b.depth /= static_cast<double>(b.nodes);
    return b;
  }

 private:
  struct Key {
    double key;
    long long id;
    int depth;
    int slot;
  };
  // Min-heap on the key; among equal keys prefer deeper nodes.
  struct KeyWorse {
    bool operator()(const Key& a, const Key& b) const {
      if (a.key != b.key) return a.key > b.key;
      if (a.depth != b.depth) return a.depth < b.depth;
      return a.id > b.id;
    }
  };
  // Drain order: deepest first, then lowest bound.
  struct DepthWorse {
    bool operator()(const Key& a, const Key& b) const {
      if (a.depth != b.depth) return a.depth < b.depth;
      if (a.key != b.key) return a.key > b.key;
      return a.id > b.id;
    }
  };
  static Key depth_key(const Node& n, int s) { return Key{n.bound, n.id, n.depth, s}; }
  // The "estimate" order is a hybrid: halfway between the node's bound and its pseudocost estimate
  // (MILP_NODE_EST_W, 1 = pure estimate). Pure estimates sent eil33-2's alternate pops into subtrees
  // whose bound was far from the global one (optimum found at 51-95 s instead of 34-67 s).
  static Key est_key(const Node& n, int s) {
    static const double est_w = knob("MILP_NODE_EST_W", 0.5);
    const double ek = is_finite(n.estimate) ? n.bound + est_w * (n.estimate - n.bound) : n.bound;
    return Key{ek, n.id, n.depth, s};
  }
  void heapify(std::vector<Key>& h) {
    if (&h == &dheap_) std::make_heap(h.begin(), h.end(), DepthWorse{});
    else std::make_heap(h.begin(), h.end(), KeyWorse{});
  }
  void pop_key(std::vector<Key>& h) {
    if (&h == &dheap_) std::pop_heap(h.begin(), h.end(), DepthWorse{});
    else std::pop_heap(h.begin(), h.end(), KeyWorse{});
    h.pop_back();
  }
  void push_key(std::vector<Key>& h, const Key& k) {
    h.push_back(k);
    if (&h == &dheap_) std::push_heap(h.begin(), h.end(), DepthWorse{});
    else std::push_heap(h.begin(), h.end(), KeyWorse{});
  }
  void update_heap_bytes() {
    heap_bytes_ = static_cast<long long>(slots_.size() * sizeof(Node) +
                                         (bheap_.capacity() + eheap_.capacity() + dheap_.capacity()) * sizeof(Key) +
                                         free_.capacity() * sizeof(int));
  }
  bool live(const Key& k) const { return slots_[k.slot].id == k.id && slots_[k.slot].depth >= 0; }
  void drop_stale(std::vector<Key>& h) {
    while (!h.empty() && !live(h.front())) pop_key(h);
  }
  double min_bound() {
    drop_stale(bheap_);
    return bheap_.front().key;
  }
  void take(bool by_estimate, Node* out) {
    std::vector<Key>& h = drain_ ? dheap_ : by_estimate ? eheap_ : bheap_;
    drop_stale(h);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"  // false positive through the inlined heap code
#endif
    const int s = h.front().slot;
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
    pop_key(h);
    *out = std::move(slots_[s]);
    slots_[s] = Node{};
    slots_[s].depth = -1;  // free
    free_.push_back(s);
    --count_;
    bytes_ -= out->mem;
    // Rebuild a heap that is mostly stale keys.
    for (std::vector<Key>* o : {&bheap_, &eheap_, &dheap_}) {
      if (o == (drain_ ? &eheap_ : &dheap_)) continue;
      if (o->size() > count_ + count_ / 4 + 64) {
        o->erase(std::remove_if(o->begin(), o->end(), [&](const Key& k) { return !live(k); }), o->end());
        heapify(*o);
      }
    }
    // Give memory back once the queue has shrunk to a small part of its arrays (slots in use
    // stay where they are: keys refer to them).
    if (slots_.size() > 4096 && count_ < slots_.size() / 8) compact();
  }
  // Moves the live nodes to the front of the slot array and rebuilds the heaps.
  void compact() {
    std::deque<Node> live_nodes;
    for (Node& n : slots_) {
      if (n.depth >= 0) live_nodes.push_back(std::move(n));
    }
    slots_.swap(live_nodes);
    std::deque<Node>().swap(live_nodes);
    std::vector<int>().swap(free_);
    for (std::vector<Key>* o : {&bheap_, &eheap_, &dheap_}) {
      std::vector<Key> fresh;
      o->swap(fresh);
    }
    for (size_t s = 0; s < slots_.size(); ++s) {
      const Node& n = slots_[s];
      const int si = static_cast<int>(s);
      bheap_.push_back(Key{n.bound, n.id, n.depth, si});
      if (drain_) dheap_.push_back(depth_key(n, si));
      else eheap_.push_back(est_key(n, si));
    }
    for (std::vector<Key>* o : {&bheap_, &eheap_, &dheap_}) heapify(*o);
    update_heap_bytes();
  }
  void clear() {
    std::deque<Node>().swap(slots_);
    std::vector<int>().swap(free_);
    std::vector<Key>().swap(bheap_);
    std::vector<Key>().swap(eheap_);
    std::vector<Key>().swap(dheap_);
    count_ = 0;
    bytes_ = 0;
    update_heap_bytes();
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Node> slots_;  // a deque: growing never copies (or transiently doubles) the slots
  std::vector<int> free_;
  std::vector<Key> bheap_, eheap_, dheap_;
  bool drain_ = false;
  size_t count_ = 0;
  std::vector<double> wbound_;
  std::atomic<long long> bytes_{0}, heap_bytes_{0};

 public:
  // Memory of the queue itself: slot and heap arrays plus what nodes declared in Node::mem
  // (symmetry groups). Bases and bound paths are accounted where they are allocated.
  long long bytes() const { return bytes_.load() + heap_bytes_.load(); }

 private:
  int active_ = 0;
};

class Pseudocosts {
 public:
  explicit Pseudocosts(Index n)
      : sum_(2 * static_cast<size_t>(n), 0.0), cnt_(2 * static_cast<size_t>(n), 0),
        inf_sum_(2 * static_cast<size_t>(n), 0.0), inf_cnt_(2 * static_cast<size_t>(n), 0) {}

  // Inference history: bounds propagated after branching on j in direction dir.
  void update_inference(Index j, int dir, double inferences) {
    std::lock_guard<std::mutex> lk(mu_);
    const size_t k = 2 * static_cast<size_t>(j) + (dir > 0);
    inf_sum_[k] += inferences;
    ++inf_cnt_[k];
    inf_tot_sum_ += inferences;
    ++inf_tot_cnt_;
  }

  void update(Index j, int dir, double gain_per_unit) {
    std::lock_guard<std::mutex> lk(mu_);
    const size_t k = 2 * static_cast<size_t>(j) + (dir > 0);
    sum_[k] += gain_per_unit;
    ++cnt_[k];
    tot_sum_[dir > 0] += gain_per_unit;
    ++tot_cnt_[dir > 0];
  }

  void unit_costs(Index j, double* pd, double* pu) {
    std::lock_guard<std::mutex> lk(mu_);
    const size_t k = 2 * static_cast<size_t>(j);
    *pd = cnt_[k] ? sum_[k] / cnt_[k] : (tot_cnt_[0] ? tot_sum_[0] / tot_cnt_[0] : 1.0);
    *pu = cnt_[k + 1] ? sum_[k + 1] / cnt_[k + 1] : (tot_cnt_[1] ? tot_sum_[1] / tot_cnt_[1] : 1.0);
  }
  void inference(Index j, double* id, double* iu) {
    std::lock_guard<std::mutex> lk(mu_);
    const size_t k = 2 * static_cast<size_t>(j);
    *id = inf_cnt_[k] ? inf_sum_[k] / inf_cnt_[k] : 0.0;
    *iu = inf_cnt_[k + 1] ? inf_sum_[k + 1] / inf_cnt_[k + 1] : 0.0;
  }

  struct Scored {
    Index j;
    double f, score;
    bool reliable;
    double dn, up;  // pseudocost estimates of the children's objective increase
    double bonus;   // hybrid score part beyond the pseudocost product
  };
  // Conflict score (VSIDS-like, per branching direction as SCIP's): key 2 j + (dir > 0) for each
  // literal of a learned conflict -- an upper bound x_j <= v counts for the down direction, a lower
  // bound for the up direction; recent conflicts weigh more (increment / 0.98 per conflict).
  void bump_conflict(const std::vector<Index>& keys) {
    std::lock_guard<std::mutex> lk(mu_);
    if (conf_.empty()) conf_.assign(cnt_.size(), 0.0);
    for (Index k : keys) conf_[static_cast<size_t>(k)] += conf_inc_;
    conf_inc_ /= 0.98;
    if (conf_inc_ > 1e100) {
      for (double& v : conf_) v *= 1e-100;
      conf_inc_ *= 1e-100;
    }
  }
  // Cutoff history: whether a child created by branching on j in direction dir was pruned.
  void update_cutoff(Index j, int dir, bool cut) {
    std::lock_guard<std::mutex> lk(mu_);
    if (cut_sum_.empty()) cut_sum_.assign(cnt_.size(), 0.0), cut_cnt_.assign(cnt_.size(), 0);
    const size_t k = 2 * static_cast<size_t>(j) + (dir > 0);
    cut_sum_[k] += cut ? 1.0 : 0.0;
    ++cut_cnt_[k];
    cut_tot_sum_ += cut ? 1.0 : 0.0;
    ++cut_tot_cnt_;
  }
  // Pseudocost product scores for all candidates (index, fractional part).
  // est_sum: sum over the candidates of the smaller child increase (for best-estimate search).
  // ps_scale: average product score when a hybrid score is used (score = ps_part(product, ps_scale) + bonus),
  // 0 otherwise (raw products).
  // degen: SCIP's degeneracy factor (>= 1): with a hybrid score the pseudocost part is divided by it and
  // the other parts are multiplied by it.
  void scores(const std::vector<std::pair<Index, double>>& cands, int reliability, std::vector<Scored>& out,
              double* est_sum, double* ps_scale, double degen = 1.0) {
    std::lock_guard<std::mutex> lk(mu_);
    const double avg_dn = tot_cnt_[0] ? tot_sum_[0] / tot_cnt_[0] : 1.0;
    const double avg_up = tot_cnt_[1] ? tot_sum_[1] / tot_cnt_[1] : 1.0;
    const double avg_inf = inf_tot_cnt_ ? inf_tot_sum_ / inf_tot_cnt_ : 1.0;
    out.clear();
    double ps_total = 0.0, inf_total = 0.0, conf_total = 0.0, cut_total = 0.0;
    const double avg_cut = cut_tot_cnt_ ? cut_tot_sum_ / static_cast<double>(cut_tot_cnt_) : 0.0;
    conf_scratch_.clear();
    cut_scratch_.clear();
    *est_sum = 0.0;
    *ps_scale = 0.0;  // 0: no hybrid score (raw products)
    inf_scratch_.clear();
    for (auto [j, f] : cands) {
      const size_t k = 2 * static_cast<size_t>(j);
      const double pd = cnt_[k] ? sum_[k] / cnt_[k] : avg_dn;
      const double pu = cnt_[k + 1] ? sum_[k + 1] / cnt_[k + 1] : avg_up;
      const double score = std::max(pd * f, 1e-6) * std::max(pu * (1.0 - f), 1e-6);
      const double id = inf_cnt_[k] ? inf_sum_[k] / inf_cnt_[k] : avg_inf;
      const double iu = inf_cnt_[k + 1] ? inf_sum_[k + 1] / inf_cnt_[k + 1] : avg_inf;
      const double inf = std::max(id, 0.1) * std::max(iu, 0.1);
      out.push_back({j, f, score, std::min(cnt_[k], cnt_[k + 1]) >= reliability, pd * f, pu * (1.0 - f), 0.0});
      if (!conf_.empty()) {
        // Product of the directional scores (normalized by the current increment), as SCIP.
        const double cs = std::max(conf_[k] / conf_inc_, 1e-6) * std::max(conf_[k + 1] / conf_inc_, 1e-6);
        conf_scratch_.push_back(cs);
        conf_total += cs;
      }
      if (!cut_sum_.empty()) {
        const double cd = cut_cnt_[k] ? cut_sum_[k] / cut_cnt_[k] : avg_cut;
        const double cu = cut_cnt_[k + 1] ? cut_sum_[k + 1] / cut_cnt_[k + 1] : avg_cut;
        const double ks = std::max(cd, 1e-6) * std::max(cu, 1e-6);
        cut_scratch_.push_back(ks);
        cut_total += ks;
      }
      *est_sum += std::min(pd * f, pu * (1.0 - f));
      inf_scratch_.push_back(inf);
      ps_total += score;
      inf_total += inf;
    }
    // Hybrid score: normalized pseudocost score plus weighted inference / conflict scores. Opt-in
    // (MILP_INF_WEIGHT, MILP_CONF_WEIGHT); probed candidates are scored on the same scale in select_branch.
    static const double w_inf = knob("MILP_INF_WEIGHT", 0.0), w_conf = knob("MILP_CONF_WEIGHT", 0.0);
    static const double w_cut = knob("MILP_CUT_WEIGHT", 0.0);
    const bool use_inf = w_inf > 0.0 && inf_total > 0.0, use_conf = w_conf > 0.0 && conf_total > 0.0;
    const bool use_cut = w_cut > 0.0 && cut_total > 0.0;
    if ((use_inf || use_conf || use_cut) && ps_total > 0.0) {
      const double n = static_cast<double>(out.size());
      *ps_scale = ps_total / n;
      // MILP_HYBRID=1: SCIP's bounded form, each part v / (v + average) in [0, 1) times its weight.
      const bool bounded = hybrid_mode() == 1;
      auto part = [&](double v, double avg) { return bounded ? v / (v + avg) : v / avg; };
      for (size_t i = 0; i < out.size(); ++i) {
        Scored& o = out[i];
        o.bonus = (use_inf ? w_inf * part(inf_scratch_[i], inf_total / n) : 0.0) +
                  (use_conf ? w_conf * part(conf_scratch_[i], std::max(conf_total / n, 1e-300)) : 0.0) +
                  (use_cut ? w_cut * part(cut_scratch_[i], std::max(cut_total / n, 1e-300)) : 0.0);
        o.bonus *= degen;
        o.score = ps_part(o.score, *ps_scale) / degen + o.bonus;
      }
    }
  }
  static int hybrid_mode() {
    static const int mode = static_cast<int>(knob("MILP_HYBRID", 0));
    return mode;
  }
  // Pseudocost part of the hybrid score (product score sc, average product avg).
  static double ps_part(double sc, double avg) {
    if (avg <= 0.0) return sc;
    return hybrid_mode() == 1 ? sc / (sc + avg) : sc / avg;
  }

  // Deterministic mode: workers branch on a snapshot of the global pseudocosts.
  void copy_from(Pseudocosts& o) {
    std::lock_guard<std::mutex> lk(o.mu_);
    sum_ = o.sum_;
    cnt_ = o.cnt_;
    tot_sum_[0] = o.tot_sum_[0], tot_sum_[1] = o.tot_sum_[1];
    tot_cnt_[0] = o.tot_cnt_[0], tot_cnt_[1] = o.tot_cnt_[1];
    inf_sum_ = o.inf_sum_;
    inf_cnt_ = o.inf_cnt_;
    inf_tot_sum_ = o.inf_tot_sum_;
    inf_tot_cnt_ = o.inf_tot_cnt_;
    conf_ = o.conf_;
    conf_inc_ = o.conf_inc_;
    cut_sum_ = o.cut_sum_;
    cut_cnt_ = o.cut_cnt_;
    cut_tot_sum_ = o.cut_tot_sum_;
    cut_tot_cnt_ = o.cut_tot_cnt_;
  }

  // Picks the candidate (index, fractional part) with the best product score.
  Index select(const std::vector<std::pair<Index, double>>& cands) {
    std::lock_guard<std::mutex> lk(mu_);
    const double avg_dn = tot_cnt_[0] ? tot_sum_[0] / tot_cnt_[0] : 1.0;
    const double avg_up = tot_cnt_[1] ? tot_sum_[1] / tot_cnt_[1] : 1.0;
    Index best = -1;
    double best_score = -1.0;
    for (auto [j, f] : cands) {
      const size_t k = 2 * static_cast<size_t>(j);
      const double pd = cnt_[k] ? sum_[k] / cnt_[k] : avg_dn;
      const double pu = cnt_[k + 1] ? sum_[k + 1] / cnt_[k + 1] : avg_up;
      const double score = std::max(pd * f, 1e-6) * std::max(pu * (1.0 - f), 1e-6);
      if (score > best_score) best_score = score, best = j;
    }
    return best;
  }

 private:
  std::mutex mu_;
  std::vector<double> sum_;
  std::vector<int> cnt_;
  double tot_sum_[2] = {0.0, 0.0};
  long long tot_cnt_[2] = {0, 0};
  std::vector<double> inf_sum_;
  std::vector<int> inf_cnt_;
  double inf_tot_sum_ = 0.0;
  long long inf_tot_cnt_ = 0;
  std::vector<double> inf_scratch_;
  std::vector<double> conf_;  // per (column, direction)
  double conf_inc_ = 1.0;
  std::vector<double> conf_scratch_, cut_scratch_;
  std::vector<double> cut_sum_;
  std::vector<int> cut_cnt_;
  double cut_tot_sum_ = 0.0;
  long long cut_tot_cnt_ = 0;
};

// Sub-MIP (large neighbourhood) heuristics chosen adaptively by success.
enum { kLnsRins, kLnsMutation, kLnsLocalBranching, kLnsRens, kLnsCrossover, kLns };
struct LnsState {
  long long last_node = 0;  // node count at the last call
  long long iters = 0;      // LP iterations used by sub-MIPs
  double calls[kLns] = {}, wins[kLns] = {};
  double recent_calls = 0.0, recent_wins = 0.0;  // exponentially decayed (factor 0.9 per call)
  double mut_fix = 0.7;  // mutation: fraction of the integers fixed
  double rins_fix = 0.0;  // RINS: fixes at least this fraction (more after limit hits)
  int lb_k = 10;  // local branching neighbourhood size
  double lb_inc = kInf;          // incumbent at the last local branching run
  unsigned seq = 0;
  long long rounds = 0;  // calls with an incumbent available (RINS on every other one)
  int cx_parents = 3;    // crossover: solutions combined (incumbent + others from the pool)
  long long cx_version = -1;  // solution pool version at the last crossover
  int cx_parents_used = 0;
  // Primal-limited searches (the bound stalls while a gap remains) get a larger share: the
  // global bound is compared each time the node count doubles.
  long long ref_nodes = 0;
  double ref_bound = -kInf;
  int boost = 0;
  bool pending = false;  // deterministic mode: a sub-MIP job of this worker has not entered yet
  bool may_start = true;  // deterministic mode: this worker may decide a sub-MIP job this round
};

// One LNS call (run_lns): inputs fixed when it is decided. Deterministic mode runs it as a job beside
// the tree search (see run_deterministic) on a copy of the worker's LnsState.
struct SubMipJob {
  int pick = 0, origin = 0;
  bool det = false, has_inc = false;
  double inc_now = kInf;
  long long iteration_limit = -1;
  std::vector<double> inc, xnode;
  std::vector<std::vector<double>> pool;  // crossover parents
  LnsState st;
  // Execution state and results.
  std::atomic<long long> progress{0};  // LP iterations so far (incl. root LPs and cut loops)
  std::atomic<bool> stop{false}, done{false};
  bool started = false;
  std::vector<std::pair<std::vector<double>, const char*>> sols;  // job mode: improving solutions
  long long id = 0;     // publication order
  double credit = 0.0;  // tree work done beside it (per thread doing tree work) since its publication
  double seconds = 0.0;  // statistics only
};

class Search {
 public:
  Search(const MipProblem& p, const SolverParams& params, const Logger& log, const Timer& timer,
         const SearchOptions& opts)
      : p_(p), params_(params), log_(log), timer_(timer), opts_(opts), rounding_(p), pc_(p.num_cols()) {
    threads_ = params.threads > 0 ? params.threads : static_cast<int>(std::thread::hardware_concurrency());
    threads_ = std::max(1, threads_);
    // Objective is integral if only integer (or implied integer) variables with integer costs contribute.
    obj_integral_ = true;
    std::vector<Index> cont_costs;
    for (Index j = 0; j < p.num_cols() && obj_integral_; ++j) {
      if (p.cost[j] == 0.0) continue;
      obj_integral_ = p.cost[j] == std::round(p.cost[j]);
      if (!p.is_int[j]) cont_costs.push_back(j);
    }
    if (obj_integral_ && !cont_costs.empty()) {
      const SparseMatrix rows = p.A.transpose();
      for (Index j : cont_costs) obj_integral_ = obj_integral_ && implied_integer(p, rows, j);
    }
    // Implications (logical constraints kept out of the LP) fix bounds outside the propagation trail.
    if (!p.implications.empty()) conflicts_ = false;
  }
  // Continuous column j is integral in every feasible point if an equality row has integer
  // columns only besides j, with coefficients and side integral multiples of a_j (fiball's
  // objective variable: x = sum of integers).
  static bool implied_integer(const MipProblem& p, const SparseMatrix& rows, Index j) {
    auto integral = [](double v) { return std::abs(v - std::round(v)) <= 1e-9 * std::max(1.0, std::abs(v)); };
    for (Index k = p.A.begin(j); k < p.A.end(j); ++k) {
      const Index i = p.A.index[k];
      const double aj = p.A.value[k];
      if (p.row_lb[i] != p.row_ub[i] || !integral(p.row_lb[i] / aj)) continue;
      bool ok = true;
      for (Index q = rows.begin(i); q < rows.end(i) && ok; ++q) {
        const Index c = rows.index[q];
        if (c == j) continue;
        ok = (p.is_int[c] || p.col_lb[c] == p.col_ub[c]) && integral(rows.value[q] / aj) &&
             (p.is_int[c] || integral(p.col_lb[c]));
      }
      if (ok) return true;
    }
    return false;
  }

  MipResult run();
  void load_debug_solution() {
    if (!opts_.debug_point.empty()) {
      debug_sol_ = opts_.debug_point;
    } else {
      const char* f = std::getenv("MILP_DEBUG_SOL");
      if (!f) return;
      std::FILE* fp = std::fopen(f, "r");
      if (!fp) return;
      double v;
      while (std::fscanf(fp, "%lf", &v) == 1) debug_sol_.push_back(v);
      std::fclose(fp);
    }
    if (static_cast<Index>(debug_sol_.size()) != p_.num_cols()) debug_sol_.clear();
    for (Index j = 0; j < static_cast<Index>(debug_sol_.size()); ++j) {
      // Sub-MIPs restrict bounds: a point outside them carries no information.
      if (debug_sol_[j] < p_.col_lb[j] - 1e-6 || debug_sol_[j] > p_.col_ub[j] + 1e-6) {
        if (!opts_.debug_point.empty()) log_.log(0, "DEBUG-SOL: the debug point is outside the column bounds (col %d)\n", j);
        debug_sol_.clear();
        return;
      }
    }
    if (!debug_sol_.empty() && p_.max_violation(debug_sol_) > 1e-6) debug_sol_.clear();  // sub-MIP rows
    log_.log(0, "DEBUG-SOL: loaded %zu values\n", debug_sol_.size());
  }

 private:
  struct Worker {
    DualSimplex lp;
    std::vector<Index> changed;
    long long last_node = -1;
    std::vector<double> sol;
    std::vector<std::pair<Index, double>> fracs;
    std::vector<Pseudocosts::Scored> scored;
    double est_sum = 0.0;
    std::vector<double> xnode;
    std::vector<const PathSeg*> segs;  // node path segments, newest first
    std::vector<uint8_t> seg;          // segment encoding scratch
    std::vector<std::pair<int, VarStatus>> nb;  // nonbasic pool rows of a stored basis
    Basis unpacked;
    std::unique_ptr<DomainPropagator> prop;
    std::vector<double> lb, ub;  // local bounds (root bounds + path + propagation)
    std::vector<Index> seeds, tightened;
    std::vector<Index> orbit;
    std::vector<char> orbit_mark;
    GroupScratch gscratch;
    DualSimplex::State state;  // node LP state for cheap returns after probes/dives
    int nodes_since_dive = 0;
    int next_rule = 0;
    // Shared-state views. In deterministic mode a worker sees a snapshot taken
    // at the start of each round plus its own findings; results are merged at
    // the round barrier in worker order.
    bool det = false;
    int tid = 0;
    Pseudocosts* pc = nullptr;
    std::unique_ptr<Pseudocosts> own_pc;
    double cutoff = kInf, incumbent = kInf;
    std::vector<double> inc_x;
    long long inc_version = -1;   // deterministic mode: global incumbent version inc_x is a copy of
    long long nodes = 0;         // nodes processed by this worker
    LnsState lns;  // deterministic mode: this worker's sub-MIP heuristic state
    long long iters_base = 0, dive_base = 0, dive_iters = 0;  // round snapshot of global counters
    long long next_seq = 0;
    struct Found {
      double obj;
      std::vector<double> x;
      const char* source;
    };
    std::vector<Found> found;
    std::vector<std::tuple<Index, int, double>> pc_log;
    std::vector<std::tuple<Index, int, double>> inf_log;
    std::vector<std::tuple<Index, int, bool>> cut_log;
    int node_depth = 0;  // depth of the node being branched on (select_branch)
    std::vector<std::vector<Index>> conf_bumps;
    std::vector<Node> pushed;
    std::vector<double> glb, gub;  // private copy of the global bounds
    size_t gfix_seen = 0;
    std::vector<BoundChange> rcfix;  // node reduced-cost fixings (inherited by children)
    size_t gcuts_applied = 0;
    // Dynamic cut pool: LP rows m_base.. are pool cuts (local rows, added when violated, removed
    // after aging). pool is this worker's view of the global pool (append-only, by id).
    Index m_base = -1;
    std::vector<int> prow, prow_age;  // per pool row of the LP: pool id, rounds inactive
    std::vector<int> pool_pos;        // per pool id: its pool row index, or -1
    std::vector<std::shared_ptr<const Cut>> pool;
    std::vector<double> pool_norm;
    std::vector<char> pool_need;
    std::vector<std::pair<double, int>> pool_cand;
    long long pool_iters = 0, node_iters = 0, pool_scan = 0;  // budget: LP iterations, pool nonzeros scanned
    std::vector<int> pool_used;  // ids added to the LP since the last flush
    size_t pool_del_seen = 0;    // position in the global deletion log
    std::vector<char> pool_dead;
    // Deterministic mode: per-worker adaptive frequencies and prune evidence (see sep_interval_).
    int sep_interval = 1, pool_interval = 1, ev_bad = 0;
    long long sep_tries = 0, sep_wins = 0, ev_tries = 0, ev_needed = 0, ev_share = 0;
    long long tmir_turns = 0;  // fresh separation rounds of this worker (see tmir_interval_)
    // Pool evidence: the node LP with the base rows only (problem and root cuts), see pool_evidence.
    std::unique_ptr<DualSimplex> base_lp;
    long long base_iters = 0;
    std::vector<std::unique_ptr<Separator>> seps;  // globally valid separators (no Gomory)
    std::vector<Cut> new_cuts;  // deterministic mode: cuts found this round
    std::vector<std::shared_ptr<SubMipJob>> new_jobs;  // deterministic mode: sub-MIP jobs decided this round
    std::unique_ptr<ConflictPool> conf;
    size_t conf_seen = 0;  // absolute position in the global conflict log
    std::vector<Index> ctight;
    std::vector<double> mult;
    std::vector<std::shared_ptr<const Conflict>> new_conflicts;  // deterministic mode: found this round
    BoundTrail trail;  // bound changes of the current node with reasons (propagation conflict analysis)
    // Conflict propagation may use work proportional to the LP work of the worker.
    long long conf_work = 0, lp_work = 0, conf_calls = 0, conf_prunes = 0;
    long long clause_work = 0, clause_calls = 0, clause_hits = 0, prop_work = 0;  // clause propagation budget
    double cont_calls = 0.0, cont_useful = 0.0;  // node propagations with continuous bounds / useful ones (decayed)
    double obj_calls = 0.0, obj_useful = 0.0;    // the same for the objective cutoff row
    double t_dive = 0.0, t_sub = 0.0;  // wall time in tree dives / sub-MIPs (statistics only)
    long long sub_runs = 0;            // sub-MIP heuristic calls that ran
    // Strong branching budget (per worker, so deterministic mode stays reproducible): LP iterations
    // of the node solves (incl. cut rounds) and of the strong branching probes; probe_limit is the
    // iteration limit of the current node's probes.
    long long tree_iters = 0, tree_lps = 0, sb_iters = 0, probe_limit = 100;
    // Strong branching result of the selected candidate: objective gains of LPs solved to optimality
    // (-inf when the probe stopped early), valid as the children's bounds.
    Index sb_var = -1;
    double sb_dn = -kInf, sb_up = -kInf;
  };

  double cutoff(const Worker& w) const { return w.det ? w.cutoff : cutoff_.load(); }
  double incumbent(const Worker& w) const { return w.det ? w.incumbent : incumbent_.load(); }
  void incumbent_x_global(std::vector<double>& out) {
    std::lock_guard<std::mutex> lk(inc_mu_);
    out = inc_x_;
  }
  void incumbent_x(const Worker& w, std::vector<double>& out) {
    if (w.det) out = w.inc_x;
    else incumbent_x_global(out);
  }
  void pc_update(Worker& w, Index j, int dir, double gain) {
    w.pc->update(j, dir, gain);
    if (w.det) w.pc_log.emplace_back(j, dir, gain);
  }
  long long next_node_id(Worker& w) { return w.det ? (++w.next_seq) * det_workers_ + w.tid + 1 : next_id_++; }
  int det_workers_ = 1;  // deterministic mode: number of (virtual) workers
  void run_deterministic();

  void run_dives(Worker& w);
  // Root start heuristics (dives, then feasibility pump and RENS) on up to
  // threads_ threads; results enter in task order (deterministic).
  void root_heuristics(Worker& w);
  bool root_lns_ran_ = false, root_lns_improved_ = false;  // root LNS rounds (reported for restarts)
  double t_start_ = 0.0;                                   // timer at the start of run()
  void run_sub_mips(Worker& w);
  void run_lns(SubMipJob& job, LnsState& st, Worker* w, std::unique_lock<std::mutex>* lk);
  bool async_jobs_ = false;  // deterministic mode: sub-MIPs run as jobs beside the tree search
  // Feasibility Jump on a side thread from the start of the solve. Opportunistic
  // mode publishes its solutions directly. Deterministic mode lets it pause
  // once it has used the effort released so far; fj_sync (called at
  // deterministic points) waits for that pause, takes its solutions and
  // releases more effort in proportion to the search's LP work.
  void fj_start();
  // lag: take only the solutions FJ found before the effort released at the previous sync (FJ keeps
  // running during the round, so the search rarely waits for it); otherwise wait for all released effort.
  void fj_sync(long long release, bool lag = false);
  void fj_finish();
  std::thread fj_thread_;
  std::atomic<bool> fj_stop_{false};
  bool fj_sync_ = false;
  std::mutex fj_mu_;
  std::condition_variable fj_cv_;
  long long fj_allowed_ = 0, fj_effort_ = 0, fj_iters_mark_ = 0, fj_prev_allowed_ = 0;
  long long fj_post_stamp_ = -1;  // effort stamp of solutions found after the FJ runs (repair, polish)
  bool fj_done_ = false;
  struct FjFound {
    long long stamp;  // FJ effort at the last checkpoint before the solution (deterministic)
    std::vector<double> x;
    const char* source;
  };
  std::vector<FjFound> fj_found_;
  std::mutex lns_mu_;   // opportunistic mode: guards lns_ and sub_active_
  int sub_active_ = 0;  // sub-MIPs running now (2 at a time when the search is primal-limited)
  std::atomic<bool> root_done_{false};  // the root node has been processed (ends the early LNS)
  std::atomic<long long> lns_last_node_{0}, sub_iters_{0};
  LnsState lns_;  // opportunistic mode; guarded by lns_mu_

  // Reliability branching. Returns the branching variable, or -1 if strong
  // branching proved the node infeasible.
  Index select_branch(Worker& w, double obj, const Basis& basis);
  // Strong-branching probe: LP gain of bounds [lb, ub] on j (kInf if infeasible).
  double probe(Worker& w, Index j, double lb, double ub, double obj, const Basis& basis, bool* exact);

  int process(Worker& w, Node& node, Node* children);
  int process_impl(Worker& w, Node& node, Node* children);
  // Recovery ladder for a node LP that failed (numerical error, iteration limit, unbounded below the
  // root): the same basis with a fresh factorization, then a slack basis, without and with cost
  // perturbation. Returns the last status.
  LpStatus recover_lp(Worker& w, const LpOptions& opt, long long node_id);
  // Testing: every k-th node LP (by id, not the root) fails and cannot be recovered (MILP_DEBUG_LP_FAIL=k).
  const long long debug_lp_fail_ = static_cast<long long>(knob("MILP_DEBUG_LP_FAIL", 0));
  bool debug_fail(long long id) const { return debug_lp_fail_ > 0 && id != 0 && id % debug_lp_fail_ == 0; }
  // Testing: points to verify have their first continuous column at a bound moved outside it by this much
  // (MILP_DEBUG_POINT_SLACK=v, v <= kPolishTol), as LP points within the LP's own tolerance can be.
  const double debug_point_slack_ = knob("MILP_DEBUG_POINT_SLACK", 0.0);
  // Last resort when the node LP cannot be solved: solve the node's bounds over the model rows alone
  // in a fresh LP and branch on its point, or split the domain of an unfixed integer column, so the
  // subtree stays in the search (bound: the node's). Returns the number of children (0: pruned or,
  // if nothing can be branched on, dropped with its bound recorded as lost).
  int unsolved_branch(Worker& w, Node& node, Node* children);
  // Implications: an unfixed z whose implication the point x violates (-1 if none), and the
  // branching on it (children z = 0 / z = 1 with the given bound).
  Index violated_implication(const Worker& w, const double* x) const;
  // An implication with z unfixed and its column not fixed at 0 (-1 if none).
  Index open_implication(const Worker& w) const;
  // Node with an unbounded LP and all implications decided: unbounded if feasible (sub-MIP with zero objective).
  int unbounded_leaf(Worker& w, Node& node);
  int implication_branch(Worker& w, Node& node, Node* children, Index z, double bound);
  bool debug_root_ = true;  // false inside sub-MIPs (debug dumps only from the top-level search)
  std::vector<double> debug_sol_;  // MILP_DEBUG_SOL: known optimal point, one value per line
  void worker_loop(int tid);
  bool try_incumbent(const std::vector<double>& x, const char* source);
  bool try_incumbent(Worker& w, const std::vector<double>& x, const char* source);
  // Verifies x (rounding integers if that keeps it feasible); returns false if infeasible.
  // Rounds the integers of x_in; if that violates the model, re-solves the continuous part with the
  // rounded integers fixed (solve_fixed_ints) when x_in itself passes the check and its objective is
  // below `repair_below` (else keeps x_in), or when x_in misses the check by at most kPolishTol (LP
  // points within the LP's tolerance) and its objective is below `polish_below` (else rejects it).
  bool verified_point(const std::vector<double>& x_in, std::vector<double>& x, double* obj,
                      double repair_below = -kInf, double polish_below = kInf) const;
  // The LP over the continuous columns with the integer columns fixed at round(x_int) (others within
  // [lb, ub]), cold, unperturbed, at primal tolerance primal_tol; on Optimal, out is its point with
  // exact integers.
  LpStatus solve_fixed_ints(const std::vector<double>& x_int, const std::vector<double>& lb,
                            const std::vector<double>& ub, double primal_tol, double cutoff,
                            std::vector<double>& out) const;
  static constexpr double kFeasTol = 1e-6;   // model check of accepted points (absolute, presolved model)
  static constexpr double kPolishTol = 1e-5;  // points up to this violation get the fixed-integer re-solve
  // Integral objective: the incumbent's value rounded to the integer it represents. An incumbent
  // within the feasibility tolerances can be off by more than 1e-6 (cvrp_large_1: 437.99997 for
  // 438); inc - 1 + 1e-6 then pruned nodes whose LP bound was exactly the optimum 437.
  static double integral_value(double inc) {
    const double r = std::round(inc);
    return std::abs(inc - r) <= 1e-4 ? r : std::ceil(inc);
  }
  double prune_threshold(double inc) const {
    if (!is_finite(inc)) return kInf;
    if (obj_integral_) return integral_value(inc) - 1.0 + 1e-6;
    return inc - std::max(1e-9, 1e-9 * std::abs(inc));
  }
  bool gap_closed(double lb) const {
    const double inc = incumbent_.load();
    if (!is_finite(inc)) return false;
    if (obj_integral_ && lb > integral_value(inc) - 1.0 + 1e-6) return true;
    const double gap = inc - lb;
    return gap <= params_.mip_abs_gap || gap <= params_.mip_rel_gap * std::max(std::abs(inc + p_.offset), 1e-10);
  }
  void maybe_log(bool force);
  // Symmetry handling (orbital branching): when a node branches on x_j, the
  // generators active at the node (a subgroup of the node's stabilizer) map
  // every solution with x_k >= ceil(v) for some k in the orbit of j onto one
  // with x_j >= ceil(v), so the down child may require x_k <= floor(v) on the
  // whole orbit. The down child keeps the node's group (the orbit constraints
  // are invariant), the up child gets (a subgroup of) the stabilizer of j:
  // the generators fixing j plus random Schreier generators. Other reductions
  // (cutoff, reduced-cost fixing, conflicts, propagation) only remove points
  // that are infeasible or not improving, a set invariant under the group.
  Symmetry sym_;
  std::atomic<long long> sym_fixed_{0}, sym_branches_{0}, sym_stabs_{0};
  void detect_symmetry_root();
  // Root LP by barrier + crossover (SolverParams::lp_method): leaves an optimal basis in lp.
  bool use_barrier_root() const;
  void barrier_root(DualSimplex& lp);
  // Auto for large pure LPs: dual simplex and barrier run at the same time, the first to finish
  // wins (opportunistic mode only: the winner depends on timing).
  bool use_concurrent_root() const;
  void concurrent_root(DualSimplex& lp);
  // Adds the LP iterations since the last report to opts_.progress (sub-MIP jobs, see SubMipJob).
  std::atomic<long long> progress_reported_{0};
  std::atomic<long long> prop_work_total_{0}, lp_work_total_{0};  // statistics (opportunistic workers)
  std::atomic<long long> obj_prop_calls_{0}, obj_prop_useful_{0}, obj_prop_skipped_{0};  // statistics
  void report_progress() {
    if (!opts_.progress) return;
    const long long now = lp_iters_.load();
    opts_.progress->fetch_add(now - progress_reported_.exchange(now));
  }
  // Racing: stop on the external flag and import better shared incumbents.
  void poll_race() {
    if (opts_.stop_flag && opts_.stop_flag->load()) {
      std::lock_guard<std::mutex> lk(stop_mu_);
      if (stop_reason_ == Status::Unknown) stop_reason_ = Status::Interrupted;
      stop_ = true;
    }
    if (opts_.shared && opts_.shared->value.load() < incumbent_.load() + p_.offset - 1e-9) {
      std::vector<double> x;
      {
        std::lock_guard<std::mutex> lk(opts_.shared->mu);
        x = opts_.shared->x;
      }
      if (!x.empty()) try_incumbent(x, "racing");
    }
  }

  const MipProblem& p_;
  const SolverParams& params_;
  const Logger& log_;
  const Timer& timer_;
  SearchOptions opts_;
  double external_cutoff_ = kInf;  // opts_.cutoff without offset
  SimpleRounding rounding_;
  Pseudocosts pc_;
  int threads_ = 1;
  bool obj_integral_ = false;
  DualSimplex root_lp_;

  // Global bounds: root bounds tightened by root reduced-cost fixing. Workers
  // keep private copies and replay gfix_log_ entries they have not seen yet.
  std::vector<double> glb_, gub_;
  std::mutex gfix_mu_;
  std::vector<BoundChange> gfix_log_;
  std::vector<double> root_d_;
  std::vector<int8_t> root_at_;  // 1 at lower bound, 2 at upper bound in the root LP
  double root_z_ = kInf;
  // Reduced-cost fixing from every root cut-loop LP optimum (HiGHS' "lurking" bounds): an integer
  // column can be fixed at its lower (upper) bound once the cutoff is below lurk_lo_ (lurk_up_).
  // Degenerate roots (drayage) have many alternative optima whose reduced costs fix different columns.
  std::vector<double> lurk_lo_, lurk_up_;
  std::vector<double> root_x_;  // root LP solution (child selection)
  // Memory limit (--mem-limit L, a cap on the search's memory that it tries to keep searching
  // under). Tree memory is accounted exactly where it is allocated: bound-path segments
  // (path_bytes_), stored warm-start bases (basis_bytes_) and the queue arrays (NodePool::bytes).
  // used = tree memory + the resident size when the tree search started (opportunistic mode).
  // States (mem_state_): 0 normal; 1 from 60% of L: new nodes get no warm-start basis; 2 (drain)
  // from 75% of L or a resident size of 85%: queued nodes lose their bases (shallowest first) down
  // to 65%, nodes are selected deepest first and workers always plunge, so the open-node count
  // stops growing; it ends below 60% (and a resident size below 75%). The search stops only at 95%
  // of L (used or resident). Deterministic mode decides on the tree memory alone, at round
  // barriers (det_tree_bytes_), so the outcome does not depend on timing; rounds are sized to stay
  // below these thresholds (run_deterministic, project).
  std::atomic<double> rss_checked_{-1.0};
  std::atomic<long long> rss_bytes_{0};
  std::atomic<long long> path_bytes_{0}, basis_bytes_{0};
  long long base_rss_ = 0;
  long long det_tree_bytes_ = 0;
  std::atomic<int> mem_state_{0};
  std::atomic<long long> mem_stripped_{0}, mem_drains_{0}, mem_peak_{0};
  std::atomic<double> mem_relief_t_{-1e9}, mem_strip_t_{-1e9};
  std::mutex mem_mu_;
  long long mem_limit_bytes() const {
    return opts_.mem_limit_bytes > 0 ? opts_.mem_limit_bytes : static_cast<long long>(params_.mem_limit_mb) << 20;
  }
  long long tree_bytes() const { return pool_->bytes() + path_bytes_.load() + basis_bytes_.load() + 12 * ncp_held_nnz_.load(); }
  long long tree_bytes(const Worker& w) const { return w.det ? det_tree_bytes_ : tree_bytes(); }
  long long checked_rss() {
    if (!opts_.check_resident) return 0;
    const double now = timer_.elapsed(), last = rss_checked_.load();
    if (now - last >= 0.05) {
      rss_checked_ = now;
      rss_bytes_ = resident_bytes();
    }
    return rss_bytes_.load();
  }
  void memory_control(bool det);
  bool memory_exceeded(bool det) {
    const long long limit = mem_limit_bytes();
    if (tree_bytes() + (det ? 0 : base_rss_) > limit / 20 * 19) return true;
    return checked_rss() > limit / 20 * 19;
  }
  // Memory budget of a sub-MIP started now: a quarter of what is left (16..256 MB).
  int sub_mip_mem_mb(const Worker* w) const {
    const long long left = mem_limit_bytes() - (w ? tree_bytes(*w) : tree_bytes()) - (w && w->det ? 0 : base_rss_);
    return static_cast<int>(std::clamp<long long>((left / 4) >> 20, 16, 256));
  }
  // Warm-start bases: the reference of the sparse form (set before the workers start) and the
  // shared_ptr deleter that releases the accounting and the pool cuts the basis held.
  PackedBasis::Ref ref_basis_;
  std::shared_ptr<const PackedBasis> store_basis(PackedBasis* pb);
  std::atomic<bool> root_rc_ready_{false};  // set once root_d_/root_at_ are final (FJ thread reads it)
  void root_rc_fixing(double cutoff);
  void record_lurking(const DualSimplex& lp);
  void dbg_global(const char* what);  // debug: report global bounds excluding the debug point
  // In-tree restart: once enough integer columns are fixed globally during the tree (root
  // reduced-cost fixing with a better incumbent, conflict propagation), stop the tree and hand the
  // problem with these fixings, the root cut rows and the most recently used pool cuts back for a
  // fresh presolve and root (SCIP's restarts). Checked by worker 0 every 128 nodes, or at the round
  // barriers in deterministic mode (from deterministic data only).
  void check_tree_restart(bool det);
  void fill_restart(MipResult& res, bool with_pool);
  Index count_global_fixed();
  std::atomic<bool> tree_restart_{false};
  Index tree_fixed0_ = 0;  // integer columns fixed globally (not by the problem) when the tree started

  // Global cut pool: globally valid cuts separated at tree nodes. Workers
  // append the entries they have not applied yet to their LPs.
  std::mutex gcut_mu_;
  std::vector<Cut> gcuts_;
  std::atomic<bool> node_cuts_enabled_{true};
  // Pool mode, adaptive frequencies. Fresh separation happens at every sep_interval-th node of a
  // worker: the interval doubles (up to 64) while fewer than 20% of the separation rounds pay off
  // (prune the node or close >= 5% of its gap to the cutoff) and halves at >= 40%. Pool cut rounds
  // (re-adding pool cuts, fresh separation) below kNodeCutDepth happen at every pool_interval-th node,
  // driven by prune evidence (pool_evidence). Opportunistic mode shares the statistics;
  // deterministic mode keeps them per worker (shared counters would depend on thread timing).
  std::atomic<long long> sep_tries_{0}, sep_wins_{0};
  std::atomic<int> sep_interval_{1};
  std::atomic<long long> ev_tries_{0}, ev_needed_{0}, ev_share_{0};
  std::atomic<int> pool_interval_{1}, ev_bad_{0};
  int sep_interval(const Worker& w) const { return w.det ? w.sep_interval : sep_interval_.load(); }
  // Tableau c-MIR cuts in fresh node separation (MILP_NO_TMIR=1: never): at every tmir_interval-th
  // fresh round of a worker. They win a round when it prunes the node or closes >= 20% of its gap to
  // the cutoff (a stricter test than sep_result's 5%: they are dense, and on cflp_large they mostly
  // gave small gains for more expensive node LPs, while on kasai two thirds of the rounds pass); per
  // window of 20 rounds the interval doubles below 20% wins (at most 64) and halves at >= 40%; from
  // kTmirOff on, pool rounds stop re-adding tableau cuts as well. Opportunistic mode only (shared
  // statistics); deterministic mode keeps them at every fresh round.
  std::atomic<long long> tmir_tries_{0}, tmir_wins_{0}, tmir_rounds_{0}, tmir_won_{0};
  std::atomic<int> tmir_interval_{1};
  static constexpr int kTmirOff = 8;  // from this interval on, pool rounds skip tableau cuts too
  bool tmir_turn(Worker& w);
  void tmir_result(bool win);
  int pool_interval(const Worker& w) const { return w.det ? w.pool_interval : pool_interval_.load(); }
  void sep_result(Worker& w, bool win);
  void pool_evidence(Worker& w, int depth);
  void pool_evidence_result(Worker& w, bool needed, long long share);
  LpStatus pool_strip(Worker& w, const LpOptions& opt);
  CutData cut_data_;
  void apply_global_cuts(Worker& w);
  int separate_node(Worker& w, bool tmir = false);

  // Dynamic node cut pool (HiGHS-like): root cuts dropped for slack plus cuts separated at nodes.
  // Ids are positions and never reused; a full pool deletes its least recently used quarter. Every
  // node re-adds pool cuts violated by its LP point as local rows (Worker::prow) that are removed
  // again after staying inactive. Deterministic mode publishes cuts and use times at round barriers.
  bool use_pool_ = std::getenv("MILP_NO_CUTPOOL") == nullptr;
  std::mutex ncp_mu_;
  std::vector<std::shared_ptr<const Cut>> ncp_;  // null: deleted
  std::vector<long long> ncp_used_;              // per id: node count when last added to an LP
  std::vector<int> ncp_deleted_;                 // log of deleted ids (workers replay it)
  std::vector<int> ncp_refs_;                    // per id: stored bases holding the cut (pool_hold)
  std::vector<std::shared_ptr<const Cut>> ncp_held_;  // deleted cuts still held by stored bases
  std::atomic<long long> ncp_held_nnz_{0};
  void pool_hold(const PackedBasis& b, int delta);
  std::shared_ptr<const Cut> pool_held_cut(int id);
  long long ncp_nnz_ = 0, ncp_max_nnz_ = 0;
  size_t ncp_max_ = 0, ncp_live_ = 0;
  std::atomic<long long> ncp_added_{0}, ncp_rounds_{0}, ncp_removed_{0}, ncp_purged_{0}, ncp_stripped_{0};
  void pool_publish(std::vector<Cut>& cuts, std::vector<int>* ids);
  void pool_purge();  // ncp_mu_ held
  void pool_pull(Worker& w);
  void pool_flush_used(Worker& w);  // ncp_mu_ held
  void pool_add_rows(Worker& w, const std::vector<int>& ids);
  void pool_remove_rows(Worker& w, const std::vector<char>& del);
  void pool_prepare(Worker& w, const Node& node);
  int pool_separate(Worker& w);
  void pool_age(Worker& w);

  // Conflict analysis: dual proofs of pruned nodes, shared through an
  // append-only log (bounded; workers that fall behind skip old entries).
  std::mutex conf_mu_;
  std::vector<std::shared_ptr<const Conflict>> conf_log_;
  size_t conf_base_ = 0;  // absolute index of conf_log_[0]
  bool conflicts_ = std::getenv("MILP_NO_CONFLICT") == nullptr;
  size_t conf_capacity_ = 0;
  Index conf_max_nnz_ = 0;
  std::atomic<long long> conf_learned_{0}, conf_pruned_{0}, prop_conf_learned_{0}, prop_conf_lits_{0};
  bool prop_conflicts_ = std::getenv("MILP_NO_PROPCONF") == nullptr;
  std::atomic<long long> trail_mismatch_{0};
  std::atomic<long long> sb_probes_{0}, sb_iters_{0};  // strong branching statistics
  std::atomic<long long> sb_calls_{0};  // branching decisions (select_branch calls)
  std::atomic<long long> sb_fixed_{0};  // strong branching domain reductions
  std::atomic<long long> sb_degen_skips_{0};  // branchings without strong branching (dual degenerate LP)
  void learn_conflict(Worker& w, LpStatus st, bool use_trail = true);
  void learn_prop_conflict(Worker& w, const char* what);
  void add_conflict(Worker& w, Conflict&& c, const char* what);
  void publish_conflict(const std::shared_ptr<const Conflict>& c);
  void pull_conflicts(Worker& w);
  bool propagate_node(Worker& w);
  // Orbitopal fixing (packing orbitopes, see orbitope.hpp) alternating with row propagation.
  bool orbitope_pass(Worker& w, bool trail_on);
  std::vector<Orbitope> orbs_;
  // Node propagation problem: the model rows plus, opt-in (MILP_CUTPROP=1), the root cut rows
  // (globally valid), built after the root. Neutral on kasai (60 s, 1 thread), so off by default.
  std::unique_ptr<MipProblem> prop_p_;
  const MipProblem& prop_problem() const { return prop_p_ ? *prop_p_ : p_; }
  std::atomic<long long> orb_fixed_{0}, orb_pruned_{0};
  // Statistics (verbosity 2): node LP iterations at plunge starts / within plunges, pool rows in the
  // node LPs, and how nodes end (see kOutcomeNames).
  std::atomic<long long> prow_sum_{0}, prow_nnz_{0};
  std::atomic<long long> node_it_total_{0}, pool_it_total_{0}, popped_it_{0}, popped_n_{0}, plunged_n_{0};
  static constexpr int kOutcomes = 11;
  std::atomic<long long> outcome_[kOutcomes] = {};
  std::atomic<long long> outcome_depth_[kOutcomes] = {};
  static thread_local int last_outcome_;
  std::atomic<long long> dh_br_[64] = {}, dh_cut_[64] = {};  // per depth: nodes branched / closed (verbosity 3)
  void count_outcome(int k, int depth) {
    if (k == 8) ++dh_br_[std::min(depth, 63)];
    else if (k != 7) ++dh_cut_[std::min(depth, 63)];
    last_outcome_ = k;
    outcome_[k].fetch_add(1, std::memory_order_relaxed);
    outcome_depth_[k].fetch_add(depth, std::memory_order_relaxed);
  }
  bool propagate_node_impl(Worker& w, BoundTrail* trail);

  std::unique_ptr<NodePool> pool_;
  std::atomic<double> incumbent_{kInf};  // without offset
  long long inc_version_ = 0;            // incremented with every new incumbent (under inc_mu_)
  std::atomic<double> cutoff_{kInf};
  std::mutex inc_mu_;
  std::vector<double> inc_x_;
  // The last incumbents (best first; guarded by inc_mu_) for crossover sub-MIPs. In
  // deterministic mode it only changes between rounds (merges run in worker order).
  static constexpr size_t kSolPool = 10;
  std::vector<std::pair<double, std::vector<double>>> sol_pool_;
  long long sol_pool_version_ = 0;
  std::atomic<long long> nodes_{0}, lp_iters_{0}, dive_iters_{0}, next_id_{1};
  std::atomic<bool> stop_{false};
  Status stop_reason_ = Status::Unknown;
  std::mutex stop_mu_;
  std::atomic<bool> root_unbounded_{false}, numerical_{false};
  std::atomic<long long> lp_failed_{0}, lp_recovered_{0}, lp_unsolved_{0}, fixed_recovered_{0};
  std::atomic<long long> impl_branches_{0};  // branchings on violated implications
  // verified_point's fixed-integer re-solves: tried, succeeded, total LP iterations (statistics)
  mutable std::atomic<long long> polish_tried_{0}, polish_ok_{0}, polish_iters_{0};
  double lost_bound_ = kInf;  // min bound of subtrees dropped for numerical reasons
  double root_bound_ = -kInf;
  double root_cs_before_ = -kInf;  // root LP bound before cuts (ROOTSTAT line)
  int root_cs_rounds_ = 0, root_cs_cuts_ = 0;
  double det_lb_ = -kInf;  // deterministic mode: global bound after the last round's merge
  std::mutex log_mu_;
  double last_log_ = 0.0, last_mem_log_ = 0.0;
  void log_memory();
};

LpStatus Search::solve_fixed_ints(const std::vector<double>& x_int, const std::vector<double>& lb_in,
                                  const std::vector<double>& ub_in, double primal_tol, double cutoff,
                                  std::vector<double>& out) const {
  std::vector<double> lb = lb_in, ub = ub_in;
  for (Index j = 0; j < p_.num_cols(); ++j) {
    if (p_.is_int[j]) lb[j] = ub[j] = std::round(x_int[j]);
  }
  if (!p_.implications.empty() && !p_.apply_implications(lb, ub)) return LpStatus::Infeasible;
  DualSimplex lp;
  lp.load(p_.A, p_.cost, lb, ub, p_.row_lb, p_.row_ub);
  LpOptions o;
  o.tol = params_.tol;
  o.tol.primal_feas = primal_tol;
  o.perturb = false;
  o.objective_cutoff = cutoff;
  o.iteration_limit = 20000 + 2 * static_cast<long long>(p_.num_rows() + p_.num_cols());
  if (!params_.deterministic) o.time_limit = std::max(0.0, std::min(5.0, params_.time_limit - timer_.elapsed()));
  const LpStatus st = lp.solve(o);
  polish_iters_ += lp.iterations();
  if (st == LpStatus::Optimal) {
    out = lp.col_values();
    for (Index j = 0; j < p_.num_cols(); ++j) {
      if (p_.is_int[j]) out[j] = lb[j];  // exact integers
    }
  }
  return st;
}

bool Search::verified_point(const std::vector<double>& x_arg, std::vector<double>& x, double* obj,
                            double repair_below, double polish_below) const {
  std::vector<double> noisy;
  if (debug_point_slack_ > 0.0) {
    noisy = x_arg;
    for (Index j = 0; j < p_.num_cols(); ++j) {
      if (p_.is_int[j]) continue;
      if (std::abs(noisy[j] - p_.col_lb[j]) <= 1e-9) noisy[j] = p_.col_lb[j] - debug_point_slack_;
      else if (std::abs(noisy[j] - p_.col_ub[j]) <= 1e-9) noisy[j] = p_.col_ub[j] + debug_point_slack_;
      else continue;
      break;
    }
  }
  const std::vector<double>& x_in = debug_point_slack_ > 0.0 ? noisy : x_arg;
  x = x_in;
  for (Index j = 0; j < p_.num_cols(); ++j) {
    if (p_.is_int[j]) x[j] = std::round(x[j]);
  }
  if (p_.max_violation(x) > kFeasTol) {
    // Rounding hurt, or the point itself is only feasible within the LP's tolerance. The LP guarantees
    // row and bound violations up to its primal tolerance (1e-6, the same as this check), so a node LP
    // point can miss the check by rounding noise (timtab1: basic row activities exactly 1e-6 beyond
    // their side; the node was then branched on without an LP down to all integers fixed and dropped),
    // and with integers fractional within the tolerance, big-M rows let the continuous part profit
    // from it (jobshop_medium_1: node LP points 610.99833 for the optimum 611). Re-solve the continuous
    // part with the rounded integers fixed at a tightened primal tolerance.
    std::vector<double> fixed;
    const double v_in = p_.max_violation(x_in), obj_in = p_.objective(x_in) - p_.offset;
    const bool in_ok = v_in <= kFeasTol;
    if (in_ok ? obj_in < repair_below : v_in <= kPolishTol && obj_in < polish_below) {
      ++polish_tried_;
      for (const double tol : {1e-9, 1e-7}) {
        const LpStatus fs = solve_fixed_ints(x, p_.col_lb, p_.col_ub, tol, kInf, fixed);
        if (fs == LpStatus::Optimal && p_.max_violation(fixed) <= kFeasTol) break;
        fixed.clear();
      }
    }
    if (!fixed.empty()) {
      ++polish_ok_;
      x = std::move(fixed);
    } else if (in_ok) {
      x = x_in;
    } else {
      return false;
    }
  } else if (!p_.implications.empty()) {
    // Implications held only within the tolerance (a column a hair from 0): re-solve the continuous
    // part with them enforced exactly, so that the tolerance does not buy objective.
    bool loose = false;
    for (const Implication& im : p_.implications) {
      loose = loose || (std::abs(x[im.z] - im.zval) < 0.5 && x[im.col] != 0.0);
    }
    std::vector<double> fixed;
    if (loose && solve_fixed_ints(x, p_.col_lb, p_.col_ub, 1e-9, kInf, fixed) == LpStatus::Optimal &&
        p_.max_violation(fixed) <= kFeasTol) {
      x = std::move(fixed);
    }
  }
  static const bool no_oneopt = std::getenv("MILP_NO_ONEOPT") || std::getenv("MILP_NO_LOCALSEARCH");
  if (!no_oneopt) {
    std::vector<double> y = x;
    if (one_opt(p_, y) && p_.max_violation(y) <= 1e-6) x = std::move(y);
  }
  *obj = p_.objective(x) - p_.offset;
  return true;
}

bool Search::try_incumbent(Worker& w, const std::vector<double>& x_in, const char* source) {
  if (!w.det) return try_incumbent(x_in, source);
  std::vector<double> x;
  double obj;
  const double below = std::min(w.incumbent, external_cutoff_);
  if (!verified_point(x_in, x, &obj, below, below)) return false;
  if (obj >= w.incumbent || obj >= external_cutoff_) return false;
  w.incumbent = obj;
  w.cutoff = std::min(w.cutoff, prune_threshold(obj));
  w.inc_x = x;
  w.inc_version = -1;  // differs from the global incumbent until the next round
  w.found.push_back({obj, std::move(x), source});
  return true;
}

bool Search::try_incumbent(const std::vector<double>& x_in, const char* source) {
  std::vector<double> x;
  double obj;
  const double below = std::min(incumbent_.load(), external_cutoff_);
  if (!verified_point(x_in, x, &obj, below, below)) return false;
  std::lock_guard<std::mutex> lk(inc_mu_);
  if (obj >= incumbent_.load() || obj >= external_cutoff_) return false;
  incumbent_.store(obj);
  ++inc_version_;
  cutoff_.store(prune_threshold(obj));
  if (root_rc_ready_) root_rc_fixing(cutoff_.load());
  if (opts_.shared) {
    std::lock_guard<std::mutex> slk(opts_.shared->mu);
    if (obj + p_.offset < opts_.shared->value.load()) {
      opts_.shared->value.store(obj + p_.offset);
      opts_.shared->x = x;  // the new point (inc_x_ still holds the previous one here)
    }
  }
  sol_pool_.emplace(sol_pool_.begin(), obj, x);
  if (sol_pool_.size() > kSolPool) sol_pool_.pop_back();
  ++sol_pool_version_;
  inc_x_ = std::move(x);
  log_.log(2, "  new incumbent %.10g (%s) at %.2fs\n", p_.display_sense * (obj + p_.offset), source, timer_.elapsed());
  return true;
}

void Search::maybe_log(bool force) {
  if (log_.verbosity() < 1) return;
  std::lock_guard<std::mutex> lk(log_mu_);
  const double t = timer_.elapsed();
  if (!force && t - last_log_ < 2.0) return;
  last_log_ = t;
  double lb = std::min(pool_->lower_bound(), incumbent_.load());
  if (obj_integral_ && is_finite(lb)) lb = std::ceil(lb - 1e-6);
  const double inc = incumbent_.load();
  const double gap = is_finite(inc) ? std::abs(inc - lb) / std::max(std::abs(inc + p_.offset), 1e-10) * 100.0 : kInf;
  log_.log(1, "%10lld nodes %8zu open  bound %14.8g  best %14.8g  gap %7.3f%%  %7.1fs\n", nodes_.load(),
           pool_->size(), p_.display_sense * (lb + p_.offset), p_.display_sense * (inc + p_.offset), gap, t);
  if (log_.verbosity() >= 2 && (force || t - last_mem_log_ >= 10.0)) {
    last_mem_log_ = t;
    log_memory();
  }
}

// Verbosity 2: where the search memory goes (resident size vs. the open-node estimate, what the
// queued nodes hold, the node cut pool and the conflict log).
void Search::log_memory() {
  const NodePool::Breakdown b = pool_->breakdown();
  long long cut_live = 0, cut_nnz = 0;
  {
    std::lock_guard<std::mutex> lk(ncp_mu_);
    cut_live = static_cast<long long>(ncp_live_);
    cut_nnz = ncp_nnz_;
  }
  size_t conf = 0;
  {
    std::lock_guard<std::mutex> lk(conf_mu_);
    conf = conf_log_.size();
  }
  const double mb = 1.0 / 1048576.0;
  const double per = b.nodes ? 1.0 / static_cast<double>(b.nodes) : 0.0;
  log_.log(2, "  memory: resident %.0f MB (at tree start %.0f), tree %.0f MB (state %d) = queue %.0f + bases %.0f + paths %.0f"
              " + held cuts %.0f; open %zu (depth %.0f, %.0f B per node): %lld with a basis (%lld sparse, %.1f pool rows),"
              " %lld with groups (%.0f MB); pool %lld cuts / %lld nnz; conflict log %zu; %d integers fixed globally\n",
           resident_bytes() * mb, base_rss_ * mb, tree_bytes() * mb, mem_state_.load(), pool_->bytes() * mb,
           basis_bytes_.load() * mb, path_bytes_.load() * mb, 12.0 * ncp_held_nnz_.load() * mb, b.nodes, b.depth,
           tree_bytes() * per, b.with_basis, b.sparse_bases, b.with_basis ? static_cast<double>(b.pool_rows) / b.with_basis : 0.0,
           b.groups, b.group_bytes * mb, cut_live, cut_nnz, conf, static_cast<int>(count_global_fixed()));
}

void Search::memory_control(bool det) {
  const long long limit = mem_limit_bytes();
  const long long base = det ? 0 : base_rss_;
  long long rss = det ? 0 : checked_rss();
  const double now = det ? 0.0 : timer_.elapsed();
  // The allocator keeps freed memory (old slot / heap arrays, freed bases) resident for reuse; hand
  // it back to the OS before it counts against the limit (at most every 2 s).
  if (!det && rss >= limit / 20 * 16 && now - mem_relief_t_.load() >= 2.0) {
    mem_relief_t_ = now;
    release_free_memory();
    rss_checked_ = -1.0;
    rss = checked_rss();
  }
  long long used = tree_bytes() + base;
  if (used - base > mem_peak_.load()) mem_peak_ = used - base;  // statistics (racy maximum)
  const int state = mem_state_.load();
  int want = used >= limit / 20 * 15 || rss >= limit / 20 * 17 ? 2 : used >= limit / 20 * 12 ? 1 : 0;
  if (state == 2 && want < 2 && (used >= limit / 20 * 12 || rss >= limit / 20 * 15)) want = 2;
  // Bytes to free: down to 65% of the limit used, 75% resident.
  const long long excess = std::max(used >= limit / 20 * 15 ? used - limit / 20 * 13 : 0,
                                    rss >= limit / 20 * 17 ? rss - limit / 20 * 15 : 0);
  const bool strip = want == 2 && excess > 0 && basis_bytes_.load() > limit / 100 && (det || now - mem_strip_t_.load() >= 1.0);
  if (want == state && !strip) {
    if (det) det_tree_bytes_ = tree_bytes();
    return;
  }
  std::lock_guard<std::mutex> lk(mem_mu_);
  if (strip) {
    // Free warm-start bases of queued nodes (shallowest first).
    mem_strip_t_ = now;
    const long long target = tree_bytes() - excess;
    const long long dropped = pool_->strip_bases([&] { return tree_bytes() <= target; });
    mem_stripped_ += dropped;
    const long long after = tree_bytes() + base;
    log_.log(2, "  memory: %.0f MB used (resident %.0f MB) of %.0f MB at %.1fs: dropped %lld stored bases -> %.0f MB\n",
             used / 1048576.0, rss / 1048576.0, limit / 1048576.0, timer_.elapsed(), dropped, after / 1048576.0);
    used = after;
  }
  if (want != mem_state_.load()) {
    if (want == 2) ++mem_drains_;
    pool_->set_drain(want == 2);
    log_.log(2, "  memory: %.0f MB used (resident %.0f MB) of %.0f MB at %.1fs: %s\n", used / 1048576.0, rss / 1048576.0,
             limit / 1048576.0, timer_.elapsed(),
             want == 2 ? "drain mode (deepest node first, no stored bases)" : want == 1 ? "no new stored bases" : "normal");
    mem_state_ = want;
  }
  if (det) det_tree_bytes_ = tree_bytes();
}

void Search::apply_global_cuts(Worker& w) {
  std::vector<Index> start{0}, index;
  std::vector<double> value, lb, ub;
  {
    std::lock_guard<std::mutex> lk(gcut_mu_);
    for (; w.gcuts_applied < gcuts_.size(); ++w.gcuts_applied) {
      const Cut& c = gcuts_[w.gcuts_applied];
      index.insert(index.end(), c.idx.begin(), c.idx.end());
      value.insert(value.end(), c.val.begin(), c.val.end());
      start.push_back(static_cast<Index>(index.size()));
      lb.push_back(-kInf);
      ub.push_back(c.rhs);
    }
  }
  if (!lb.empty()) w.lp.add_rows(static_cast<Index>(lb.size()), start, index, value, lb, ub);
}

int Search::separate_node(Worker& w, bool tmir) {
  static const bool disabled = std::getenv("MILP_NO_NODECUTS") != nullptr;
  if (disabled) return 0;
  // The pool may at most double the row count (cut rows slow every node LP). MILP_GCUT_MAX (default
  // 500) and MILP_NODECUT_DEPTH (6) are opt-in knobs: more node cuts helped kasai in some runs only.
  static const Index max_gcuts = static_cast<Index>(knob("MILP_GCUT_MAX", 500));
  if (!use_pool_) {
    std::lock_guard<std::mutex> lk(gcut_mu_);
    if (gcuts_.size() >= static_cast<size_t>(std::clamp<Index>(2 * p_.num_rows(), 20, max_gcuts))) return 0;
  }
  if (w.seps.empty()) {
    w.seps.push_back(make_cover_separator());
    // Flow covers at nodes: cflp_medium_1 1.4x slower over 6 seeds (more, weaker node cuts); root only
    // unless MILP_NODE_FLOWCOVER=1.
    static const bool node_fc = knob("MILP_NODE_FLOWCOVER", 0.0) > 0.0;
    w.seps.push_back(make_mir_separator(node_fc));
    w.seps.push_back(make_clique_separator());
    w.seps.push_back(make_implied_bound_separator());
    if (use_pool_) w.seps.push_back(make_tableau_mir_separator());  // last; only when tmir is set
  }
  const double* x = w.lp.x();
  bool fractional = false;
  for (Index j = 0; j < p_.num_cols() && !fractional; ++j) {
    fractional = p_.is_int[j] && std::abs(x[j] - std::round(x[j])) > params_.tol.integrality;
  }
  if (!fractional) return 0;
  SepaContext ctx{p_, w.lp, x, p_.num_rows(), cut_data_, 0};
  std::vector<Cut> cand;
  for (size_t k = 0; k < w.seps.size(); ++k) {
    if (use_pool_ && k + 1 == w.seps.size() && !tmir) continue;  // the tableau c-MIR separator
    w.seps[k]->separate(ctx, cand);
  }
  if (cand.empty()) return 0;
  CutPoolOptions popt;
  popt.max_cuts = 20;
  popt.max_nnz = std::max<Index>(50, p_.num_cols() / 4 + 20);
  CutPool pool(p_, popt);
  for (Cut& c : cand) pool.add(std::move(c), x);
  std::vector<Cut> sel = pool.select();
  if (sel.empty()) return 0;
  for (const Cut& c : sel) {
    if (debug_sol_.empty()) break;
    const double viol = c.activity(debug_sol_.data()) - c.rhs;
    if (viol > 1e-6 * std::max(1.0, std::abs(c.rhs))) {
      log_.log(0, "DEBUG-SOL: node %s cut violated by the debug solution (%g)\n", c.origin, viol);
    }
  }
  const int added = static_cast<int>(sel.size());
  if (w.det) {
    for (Cut& c : sel) w.new_cuts.push_back(std::move(c));
    return 0;  // applied by everyone after the round barrier
  }
  if (use_pool_ && w.m_base >= 0) {
    std::vector<int> ids;
    pool_publish(sel, &ids);
    pool_pull(w);
    ids.erase(std::remove(ids.begin(), ids.end(), -1), ids.end());
    pool_add_rows(w, ids);
    return static_cast<int>(ids.size());
  }
  {
    std::lock_guard<std::mutex> lk(gcut_mu_);
    for (Cut& c : sel) gcuts_.push_back(std::move(c));
  }
  apply_global_cuts(w);
  return added;
}

void Search::pool_publish(std::vector<Cut>& cuts, std::vector<int>* ids) {
  std::lock_guard<std::mutex> lk(ncp_mu_);
  const long long epoch = nodes_.load();
  for (Cut& c : cuts) {
    const long long nnz = static_cast<long long>(c.idx.size());
    if (c.idx.empty()) {
      if (ids) ids->push_back(-1);
      continue;
    }
    if (ncp_live_ >= ncp_max_ || ncp_nnz_ + nnz > ncp_max_nnz_) pool_purge();
    if (ncp_live_ >= ncp_max_ || ncp_nnz_ + nnz > ncp_max_nnz_) {
      if (ids) ids->push_back(-1);
      continue;
    }
    if (ids) ids->push_back(static_cast<int>(ncp_.size()));
    ncp_nnz_ += nnz;
    ++ncp_live_;
    c.pool_id = -1;
    ncp_.push_back(std::make_shared<const Cut>(std::move(c)));
    ncp_used_.push_back(epoch);
    ncp_refs_.push_back(0);
    ncp_held_.emplace_back();
  }
}

// A full pool deletes its least recently used quarter (by the node count at the last use; ties by
// id). Deterministic mode calls this only at round barriers, with use times merged there.
void Search::pool_purge() {
  std::vector<std::pair<long long, int>> order;
  for (size_t id = 0; id < ncp_.size(); ++id) {
    if (ncp_[id]) order.emplace_back(ncp_used_[id], static_cast<int>(id));
  }
  if (order.size() < 8) return;
  const size_t del = order.size() / 4;
  std::nth_element(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(del), order.end());
  for (size_t q = 0; q < del; ++q) {
    const int id = order[q].second;
    ncp_nnz_ -= static_cast<long long>(ncp_[id]->idx.size());
    if (ncp_refs_[id] > 0) {  // stored bases still need it
      ncp_held_nnz_ += static_cast<long long>(ncp_[id]->idx.size());
      ncp_held_[id] = std::move(ncp_[id]);
    }
    ncp_[id].reset();
    --ncp_live_;
    ncp_deleted_.push_back(id);
  }
  ncp_purged_ += static_cast<long long>(del);
}

// Stored bases hold their nonbasic pool cuts: a cut deleted from the pool while held moves to
// ncp_held_ and is freed when the last basis referring to it goes.
void Search::pool_hold(const PackedBasis& b, int delta) {
  std::lock_guard<std::mutex> lk(ncp_mu_);
  b.for_each_pool([&](int id, VarStatus) {
    ncp_refs_[id] += delta;
    if (ncp_refs_[id] == 0 && ncp_held_[id]) {
      ncp_held_nnz_ -= static_cast<long long>(ncp_held_[id]->idx.size());
      ncp_held_[id].reset();
    }
  });
}

std::shared_ptr<const Cut> Search::pool_held_cut(int id) {
  std::lock_guard<std::mutex> lk(ncp_mu_);
  return ncp_[id] ? ncp_[id] : ncp_held_[id];
}

std::shared_ptr<const PackedBasis> Search::store_basis(PackedBasis* pb) {
  const long long bytes = pb->bytes();
  basis_bytes_ += bytes;
  if (pb->npool > 0) pool_hold(*pb, 1);
  return std::shared_ptr<const PackedBasis>(pb, [this, bytes](const PackedBasis* q) {
    if (q->npool > 0) pool_hold(*q, -1);
    basis_bytes_ -= bytes;
    delete q;
  });
}

void Search::pool_flush_used(Worker& w) {
  const long long epoch = nodes_.load();
  for (int id : w.pool_used) ncp_used_[id] = epoch;
  w.pool_used.clear();
}

void Search::pool_pull(Worker& w) {
  std::lock_guard<std::mutex> lk(ncp_mu_);
  if (!w.det) pool_flush_used(w);
  for (size_t k = w.pool.size(); k < ncp_.size(); ++k) {
    double nrm = 0.0;
    if (ncp_[k]) {
      for (double v : ncp_[k]->val) nrm += v * v;
    }
    w.pool.push_back(ncp_[k]);
    w.pool_norm.push_back(std::sqrt(nrm));
    w.pool_pos.push_back(-1);
    w.pool_dead.push_back(ncp_[k] ? 0 : 1);
  }
  for (; w.pool_del_seen < ncp_deleted_.size(); ++w.pool_del_seen) {
    const int id = ncp_deleted_[w.pool_del_seen];
    w.pool_dead[id] = 1;
    if (w.pool_pos[id] < 0) w.pool[id].reset();  // rows in the LP keep their cut until removed
  }
}

void Search::pool_add_rows(Worker& w, const std::vector<int>& ids) {
  if (ids.empty()) return;
  std::vector<Index> start{0}, index;
  std::vector<double> value, lb, ub;
  for (int id : ids) {
    const Cut& c = *w.pool[id];
    index.insert(index.end(), c.idx.begin(), c.idx.end());
    value.insert(value.end(), c.val.begin(), c.val.end());
    start.push_back(static_cast<Index>(index.size()));
    lb.push_back(-kInf);
    ub.push_back(c.rhs);
    w.pool_pos[id] = static_cast<int>(w.prow.size());
    w.prow.push_back(id);
    w.prow_age.push_back(0);
  }
  w.lp.add_rows(static_cast<Index>(ids.size()), start, index, value, lb, ub);
}

// Removes the pool rows k with del[k] (their logicals must be basic to keep the basis).
void Search::pool_remove_rows(Worker& w, const std::vector<char>& del) {
  std::vector<Index> rows;
  size_t keep = 0;
  for (size_t k = 0; k < w.prow.size(); ++k) {
    if (del[k]) {
      rows.push_back(w.m_base + static_cast<Index>(k));
      w.pool_pos[w.prow[k]] = -1;
      if (w.pool_dead[w.prow[k]]) w.pool[w.prow[k]].reset();
      continue;
    }
    w.prow[keep] = w.prow[k];
    w.prow_age[keep] = w.prow_age[k];
    w.pool_pos[w.prow[keep]] = static_cast<int>(keep);
    ++keep;
  }
  if (rows.empty()) return;
  w.prow.resize(keep);
  w.prow_age.resize(keep);
  w.lp.remove_rows(rows);
  ncp_removed_ += static_cast<long long>(rows.size());
}

// Before a node's LP solve: drop aged pool rows and install the node's warm-start basis, adding
// the pool rows that were nonbasic in the parent's final LP.
void Search::pool_prepare(Worker& w, const Node& node) {
  DualSimplex& lp = w.lp;
  const Index n = p_.num_cols();
  const bool warm = node.basis && node.parent != w.last_node;
  if (warm && node.basis->size != n + w.m_base) {  // cannot happen: base rows are fixed in pool mode
    lp.reset_basis();
    return;
  }
  const size_t np = w.prow.size();
  if (warm) {
    w.nb.clear();
    node.basis->for_each_pool([&](int id, VarStatus st) { w.nb.emplace_back(id, st); });
    for (const auto& [id, st] : w.nb) {
      if (id < 0 || static_cast<size_t>(id) >= w.pool.size()) {  // cannot happen: ids come from the pool
        lp.reset_basis();
        return;
      }
    }
    // A cut deleted from the pool meanwhile comes back from the pool's hold list (pool_hold: kept
    // while stored bases refer to it); it stays dead: never separated, released when its row leaves.
    for (const auto& [id, st] : w.nb) {
      if (w.pool_pos[id] >= 0 || w.pool[id]) continue;
      w.pool[id] = pool_held_cut(id);
      if (!w.pool[id]) {  // cannot happen
        lp.reset_basis();
        return;
      }
    }
  }
  w.pool_need.assign(w.pool.size(), 0);
  if (warm) {
    for (const auto& [id, st] : w.nb) w.pool_need[id] = 1;
  } else {
    for (size_t k = 0; k < np; ++k) {
      if (!lp.row_is_basic(w.m_base + static_cast<Index>(k))) w.pool_need[w.prow[k]] = 1;
    }
  }
  // Aging: rows inactive for kAge node LPs leave; beyond the row cap the oldest leave first.
  static const int kAge = static_cast<int>(knob("MILP_POOL_AGE", 10));
  const size_t cap = static_cast<size_t>(std::clamp<Index>(p_.num_rows(), 100, 1000));
  std::vector<char> del(np, 0);
  size_t ndel = 0;
  for (size_t k = 0; k < np; ++k) {
    if (!w.pool_need[w.prow[k]] && w.prow_age[k] >= kAge) del[k] = 1, ++ndel;
  }
  if (np - ndel > cap) {
    std::vector<std::pair<int, size_t>> order;
    for (size_t k = 0; k < np; ++k) {
      if (!del[k] && !w.pool_need[w.prow[k]]) order.emplace_back(-w.prow_age[k], k);
    }
    std::sort(order.begin(), order.end());
    for (size_t q = 0; q < order.size() && np - ndel > cap; ++q) del[order[q].second] = 1, ++ndel;
  }
  // Removing rows invalidates the factorization: do it only once enough rows have aged (or with a
  // warm start, which refactors anyway).
  if (ndel > 0 && (warm || ndel >= 10 || np - ndel > cap || 4 * ndel >= np)) pool_remove_rows(w, del);
  if (!warm) return;
  std::vector<int> missing;
  for (const auto& [id, st] : w.nb) {
    if (w.pool_pos[id] < 0) missing.push_back(id);
  }
  pool_add_rows(w, missing);
  node.basis->unpack(w.unpacked);
  w.unpacked.status.resize(static_cast<size_t>(n + lp.num_rows()), VarStatus::Basic);
  for (const auto& [id, st] : w.nb) {
    w.unpacked.status[static_cast<size_t>(n + w.m_base + w.pool_pos[id])] = st;
  }
  lp.set_basis(w.unpacked);
}

// Adds pool cuts violated by the current LP point as rows; returns how many.
int Search::pool_separate(Worker& w) {
  const double* x = w.lp.x();
  const bool tmir_off = !w.det && tmir_interval_.load() >= kTmirOff;
  w.pool_cand.clear();
  for (size_t id = 0; id < w.pool.size(); ++id) {
    if (w.pool_pos[id] >= 0 || w.pool_dead[id]) continue;
    const Cut& c = *w.pool[id];
    if (tmir_off && c.origin[0] == 't') continue;  // tableau cuts ("tmir", "tcover") judged unprofitable
    w.pool_scan += static_cast<long long>(c.idx.size());
    const double viol = c.activity(x) - c.rhs;
    if (viol <= 1e-6 * std::max(1.0, std::abs(c.rhs))) continue;
    const double eff = viol / std::max(w.pool_norm[id], 1e-12);
    if (eff < 1e-4) continue;
    w.pool_cand.emplace_back(-eff, static_cast<int>(id));
  }
  if (w.pool_cand.empty()) return 0;
  static const int kMaxAdd = static_cast<int>(knob("MILP_POOL_ADD", 50));
  const size_t take = std::min<size_t>(w.pool_cand.size(), static_cast<size_t>(kMaxAdd));
  std::partial_sort(w.pool_cand.begin(), w.pool_cand.begin() + static_cast<std::ptrdiff_t>(take), w.pool_cand.end());
  std::vector<int> ids;
  for (size_t q = 0; q < take; ++q) ids.push_back(w.pool_cand[q].second);
  std::sort(ids.begin(), ids.end());  // deterministic row order
  pool_add_rows(w, ids);
  w.pool_used.insert(w.pool_used.end(), ids.begin(), ids.end());
  ncp_added_ += static_cast<long long>(ids.size());
  return static_cast<int>(ids.size());
}

// After a node's final LP solve: rows with a basic, slack logical age; binding rows are reset.
void Search::pool_age(Worker& w) {
  const Index n = p_.num_cols();
  for (size_t k = 0; k < w.prow.size(); ++k) {
    const Index i = w.m_base + static_cast<Index>(k);
    bool slack = w.lp.row_is_basic(i);
    if (slack) {
      const double act = -w.lp.var_value(n + i), ub = -w.lp.var_lower(n + i);
      slack = ub - act > 1e-6 * std::max(1.0, std::abs(ub));
    }
    w.prow_age[k] = slack ? w.prow_age[k] + 1 : 0;
  }
}

void Search::sep_result(Worker& w, bool win) {
  auto next = [](int k, long long tries, long long wins) {
    return wins * 5 < tries ? std::min(64, 2 * k) : wins * 5 >= 2 * tries ? std::max(1, k / 2) : k;
  };
  int k, nk;
  long long tries, wins;
  if (w.det) {
    tries = ++w.sep_tries;
    wins = w.sep_wins += win;
    if (tries < 50) return;
    k = w.sep_interval;
    nk = w.sep_interval = next(k, tries, wins);
    w.sep_tries = w.sep_wins = 0;
  } else {
    tries = ++sep_tries_;
    wins = win ? ++sep_wins_ : sep_wins_.load();
    if (tries != 50) return;
    k = sep_interval_.load();
    nk = next(k, tries, wins);
    sep_tries_ = 0;
    sep_wins_ = 0;
    sep_interval_ = nk;
  }
  if (nk != k) log_.log(2, "  node separation every %d nodes (%lld/%lld successful)\n", nk, wins, tries);
}

bool Search::tmir_turn(Worker& w) {
  static const bool disabled = std::getenv("MILP_NO_TMIR") != nullptr;
  if (disabled || !use_pool_) return false;
  return w.det || w.tmir_turns++ % tmir_interval_.load() == 0;
}

void Search::tmir_result(bool win) {
  ++tmir_rounds_;
  if (win) ++tmir_won_;
  auto next = [](int k, long long tries, long long wins) {
    return wins * 5 < tries ? std::min(64, 2 * k) : wins * 5 >= 2 * tries ? std::max(1, k / 2) : k;
  };
  constexpr long long kWindow = 20;
  const long long tries = ++tmir_tries_;
  const long long wins = win ? ++tmir_wins_ : tmir_wins_.load();
  if (tries != kWindow) return;
  const int k = tmir_interval_.load();
  const int nk = next(k, tries, wins);
  tmir_tries_ = 0;
  tmir_wins_ = 0;
  tmir_interval_ = nk;
  if (nk != k) log_.log(2, "  tableau cuts in every %d-th separation round (%lld/%lld successful)\n", nk, wins, tries);
}

// Prune evidence for the node cut pool. Pool rows cost LP work at every node they are in (more rows,
// more simplex iterations, and binding rows ride along in the warm-start bases of whole subtrees),
// so they have to pay for themselves by pruning. At a bound prune (LP value >= cutoff) of a node
// below kNodeCutDepth whose LP holds pool rows, the node is re-solved with the base rows only
// (problem and root cuts): the prune needed the pool if that LP stays below the cutoff. Over a
// window of kWindow such prunes, the share that needed the pool must reach kCost times the pool
// rows' average share of the LP rows (at most kCap of the prunes). Below half of it, pool cut rounds
// below kNodeCutDepth stop (every 1024th node, which keeps collecting evidence) and, from the
// second such window in a row on, pool_strip removes the remaining pool rows; below it they become kStep times rarer; at >= 1.5 times it
// kStep times more frequent, and at >= 2 times it they are back at every node. Infeasible LPs are
// not counted (mostly branching, not cuts). The base LP solves may use 5% of the node LP iterations.
void Search::pool_evidence(Worker& w, int depth) {
  static const bool disabled = std::getenv("MILP_NO_POOL_EVIDENCE") != nullptr;
  static const int kNodeCutDepth = static_cast<int>(knob("MILP_NODECUT_DEPTH", 6));
  static const bool no_infeasible_ev = std::getenv("MILP_NO_POOL_EV_INFEAS") != nullptr;
  if (disabled || depth <= kNodeCutDepth || !use_pool_ || w.m_base < 0 || w.prow.empty() ||
      (no_infeasible_ev && !is_finite(cutoff(w)))) {
    return;
  }
  if (static_cast<double>(w.base_iters) > 0.05 * static_cast<double>(w.node_iters) + 5000) return;
  if (!w.base_lp) w.base_lp = std::make_unique<DualSimplex>(root_lp_);
  DualSimplex& lp0 = *w.base_lp;
  for (Index j = 0; j < p_.num_cols(); ++j) lp0.set_col_bounds(j, w.lp.col_lower(j), w.lp.col_upper(j));
  LpOptions opt;
  opt.tol = params_.tol;
  opt.objective_cutoff = cutoff(w);
  opt.time_limit = params_.time_limit - timer_.elapsed();
  const long long it0 = lp0.iterations();
  const LpStatus st = lp0.solve(opt);
  w.base_iters += lp0.iterations() - it0;
  if (st != LpStatus::Optimal && st != LpStatus::Infeasible && st != LpStatus::CutoffReached) return;
  const bool needed = st == LpStatus::Optimal && lp0.objective() < cutoff(w);
  const long long share =
      static_cast<long long>(1000.0 * static_cast<double>(w.prow.size()) / static_cast<double>(w.lp.num_rows()));
  pool_evidence_result(w, needed, share);
}

void Search::pool_evidence_result(Worker& w, bool needed, long long share) {
  static const int kWindow = static_cast<int>(knob("MILP_POOL_EV_WINDOW", 10));
  static const int kStep = static_cast<int>(knob("MILP_POOL_EV_STEP", 4));
  static const double kCost = knob("MILP_POOL_EV_COST", 2.0);
  static const double kCap = knob("MILP_POOL_EV_CAP", 0.5);
  auto next = [](int k, long long tries, long long nd, long long sh, int& bad) {
    // Required share of prunes (per mille, summed over the window). bad counts the windows in a row
    // below half of it; pool_strip runs from the second one on (one bad window may be early noise).
    const double need = std::min(kCost * static_cast<double>(sh), 1000.0 * kCap * static_cast<double>(tries));
    const double got = 1000.0 * static_cast<double>(nd);
    bad = got < 0.5 * need ? bad + 1 : 0;
    if (bad > 0) return 1024;
    if (got >= 2.0 * need) return 1;
    return got < need ? std::min(1024, kStep * k) : got >= 1.5 * need ? std::max(1, k / kStep) : k;
  };
  int k, nk;
  long long tries, nd, sh;
  if (w.det) {
    tries = ++w.ev_tries;
    nd = w.ev_needed += needed;
    sh = w.ev_share += share;
    if (tries < kWindow) return;
    k = w.pool_interval;
    nk = w.pool_interval = next(k, tries, nd, sh, w.ev_bad);
    w.ev_tries = w.ev_needed = w.ev_share = 0;
  } else {
    tries = ++ev_tries_;
    nd = needed ? ++ev_needed_ : ev_needed_.load();
    sh = ev_share_ += share;
    if (tries != kWindow) return;
    k = pool_interval_.load();
    int bad = ev_bad_.load();
    nk = next(k, tries, nd, sh, bad);
    ev_bad_ = bad;
    ev_tries_ = 0;
    ev_needed_ = 0;
    ev_share_ = 0;
    pool_interval_ = nk;
  }
  if (nk != k) {
    log_.log(2, "  pool cut rounds every %d nodes (%lld/%lld prunes needed pool rows, %.0f%% of the LP rows)\n", nk,
             nd, tries, 0.1 * static_cast<double>(sh) / static_cast<double>(tries));
  }
}

// With pool cut rounds (nearly) off, the pool rows still in the LP are pivoted out of the basis and
// removed, and the LP is re-solved: otherwise binding rows from earlier rounds would ride along in
// the warm-start bases of whole subtrees.
LpStatus Search::pool_strip(Worker& w, const LpOptions& opt) {
  std::vector<Index> rows;
  for (size_t k = 0; k < w.prow.size(); ++k) rows.push_back(w.m_base + static_cast<Index>(k));
  w.lp.pivot_out_rows(rows);
  std::vector<char> del(w.prow.size(), 0);
  for (size_t k = 0; k < w.prow.size(); ++k) del[k] = w.lp.row_is_basic(w.m_base + static_cast<Index>(k));
  pool_remove_rows(w, del);
  const long long it0 = w.lp.iterations();
  const LpStatus st = w.lp.solve(opt);
  lp_iters_ += w.lp.iterations() - it0;
  ncp_stripped_ += 1;
  return st;
}

void Search::record_lurking(const DualSimplex& lp) {
  const double z = lp.objective();
  if (!is_finite(z)) return;
  for (Index j = 0; j < p_.num_cols(); ++j) {
    if (!p_.is_int[j]) continue;
    const VarStatus vs = lp.status(j);
    const double d = lp.reduced_cost(j);
    // Fixing at the bound holds when slack / |d| < 1 (cf. root_rc_fixing: floor(slack / d + 1e-9) == 0).
    if (vs == VarStatus::AtLower && d > 1e-7 && lp.col_lower(j) == p_.col_lb[j]) {
      lurk_lo_[j] = std::max(lurk_lo_[j], z + d * (1.0 - 1e-9));
    } else if (vs == VarStatus::AtUpper && d < -1e-7 && lp.col_upper(j) == p_.col_ub[j]) {
      lurk_up_[j] = std::max(lurk_up_[j], z - d * (1.0 - 1e-9));
    }
  }
}

void Search::root_rc_fixing(double cutoff) {
  static const bool disabled = std::getenv("MILP_NO_RCFIX") != nullptr;
  const double slack = cutoff - root_z_;
  if (disabled || !(slack >= 0.0) || !is_finite(slack)) return;
  std::lock_guard<std::mutex> lk(gfix_mu_);
  Index fixed = 0;
  const bool lurk = !lurk_lo_.empty();
  for (Index j = 0; j < p_.num_cols(); ++j) {
    if (!p_.is_int[j]) continue;
    if (lurk) {
      if (cutoff < lurk_lo_[j] && gub_[j] > p_.col_lb[j] && glb_[j] <= p_.col_lb[j]) {
        gub_[j] = p_.col_lb[j];
        gfix_log_.push_back({j, glb_[j], gub_[j]});
        ++fixed;
        continue;
      }
      if (cutoff < lurk_up_[j] && glb_[j] < p_.col_ub[j] && gub_[j] >= p_.col_ub[j]) {
        glb_[j] = p_.col_ub[j];
        gfix_log_.push_back({j, glb_[j], gub_[j]});
        ++fixed;
        continue;
      }
    }
    const double d = root_d_[j];
    if (root_at_[j] == 1 && d > 1e-7) {
      const double nu = p_.col_lb[j] + std::floor(slack / d + 1e-9);
      if (nu < gub_[j] - 0.5) {
        gub_[j] = std::max(nu, glb_[j]);
        gfix_log_.push_back({j, glb_[j], gub_[j]});
        ++fixed;
      }
    } else if (root_at_[j] == 2 && d < -1e-7) {
      const double nl = p_.col_ub[j] - std::floor(slack / -d + 1e-9);
      if (nl > glb_[j] + 0.5) {
        glb_[j] = std::min(nl, gub_[j]);
        gfix_log_.push_back({j, glb_[j], gub_[j]});
        ++fixed;
      }
    }
  }
  if (fixed > 0) log_.log(2, "  root reduced-cost fixing: %d bounds tightened\n", fixed);
  dbg_global("root reduced-cost fixing");
}

void Search::learn_conflict(Worker& w, LpStatus st, bool use_trail) {
  if (!conflicts_ || !w.conf) return;
  ConflictOptions o;
  o.max_nnz = conf_max_nnz_;
  Conflict c;
  const double co = cutoff(w);
  if (st == LpStatus::Infeasible) {
    if (!w.lp.farkas_proof(w.mult)) return;
    if (!infeasibility_conflict(w.lp, w.mult, p_.cost, p_.is_int, w.glb, w.gub, o, c)) return;
    c.cutoff = co;  // global bounds may stem from reduced-cost fixing (valid below the cutoff only)
  } else {
    // c x <= cutoff plus the rows weighted by -y: (c - A^T y) x <= cutoff - y^T b.
    if (!is_finite(co)) return;
    w.mult = w.lp.row_duals();
    for (double& v : w.mult) v = -v;
    if (!derive_conflict(w.lp, w.mult, 1.0, p_.cost, co, p_.is_int, w.glb, w.gub, o, c)) return;
  }
  if (use_trail && prop_conflicts_ && !w.trail.last.empty() && !w.trail.overflow) {
    // The proof's infeasibility, explained through the node's propagation
    // back to the branching decisions, gives a (usually short) clause.
    bool consistent = true;  // the trail must describe the LP's bounds
    for (Index j : c.idx) {
      const int kl = w.trail.last[2 * static_cast<size_t>(j)], ku = w.trail.last[2 * static_cast<size_t>(j) + 1];
      const double l = kl >= 0 ? w.trail.e[kl].val : w.glb[j], u = ku >= 0 ? w.trail.e[ku].val : w.gub[j];
      if (l != w.lp.col_lower(j) || u != w.lp.col_upper(j)) consistent = false;
    }
    if (consistent) {
      w.trail.fail_conflict(&c);
      learn_prop_conflict(w, "LP clause");  // the proof itself is kept too (it prunes more)
      w.trail.fail = BoundTrail::kNone;
    } else {
      ++trail_mismatch_;
    }
  }
  add_conflict(w, std::move(c), st == LpStatus::Infeasible ? "infeasible" : "cutoff");
}

void Search::learn_prop_conflict(Worker& w, const char* what) {
  if (!conflicts_ || !prop_conflicts_ || !w.conf) return;
  static const int max_clauses = static_cast<int>(knob("MILP_PROPCONF_CLAUSES", 2));
  std::vector<Conflict> out;
  analyze_propagation_conflict(w.trail, w.prop->rows(), w.prop->row_lb(), w.prop->row_ub(), p_.is_int, w.glb, w.gub, conf_max_nnz_,
                               max_clauses, out);
  for (Conflict& c : out) {
    c.cutoff = std::min(c.cutoff, cutoff(w));  // global bounds may stem from reduced-cost fixing
    ++prop_conf_learned_, prop_conf_lits_ += static_cast<long long>(c.idx.size());
    add_conflict(w, std::move(c), what);
  }
}

void Search::add_conflict(Worker& w, Conflict&& c, const char* what) {
  if (!debug_sol_.empty()) {
    double obj = 0.0;
    for (Index j = 0; j < p_.num_cols(); ++j) obj += p_.cost[j] * debug_sol_[j];
    if (obj < c.cutoff && c.violated_by(debug_sol_)) {
      log_.log(0, "DEBUG-SOL: conflict (%s, %zu nnz) violated by the optimum (cutoff %.10g)\n", what, c.idx.size(),
               c.cutoff);
    }
  }
  c.owner = w.tid;
  ++conf_learned_;
  static const bool conf_score = knob("MILP_CONF_WEIGHT", 0.0) > 0.0;
  if (conf_score) {
    // Literals of a clause; of a linear proof the local bounds it relies on (tightened below the
    // root): a positive coefficient uses the lower bound (up direction), a negative one the upper.
    std::vector<Index> cols;
    for (size_t k = 0; k < c.idx.size(); ++k) {
      const Index j = c.idx[k];
      if (!p_.is_int[j]) continue;
      if (c.is_clause()) {
        cols.push_back(2 * j + (c.upper[k] ? 0 : 1));
      } else if (c.val[k] > 0 && w.lb[j] > w.glb[j]) {
        cols.push_back(2 * j + 1);
      } else if (c.val[k] < 0 && w.ub[j] < w.gub[j]) {
        cols.push_back(2 * j);
      }
    }
    w.pc->bump_conflict(cols);
    if (w.det) w.conf_bumps.push_back(std::move(cols));
  }
  auto sp = std::make_shared<const Conflict>(std::move(c));
  w.conf->add(sp, w.nodes);
  if (w.det) w.new_conflicts.push_back(std::move(sp));
  else publish_conflict(sp);
}

void Search::publish_conflict(const std::shared_ptr<const Conflict>& c) {
  {
    std::lock_guard<std::mutex> lk(conf_mu_);
    conf_log_.push_back(c);
    if (conf_log_.size() > 2 * conf_capacity_) {
      const size_t drop = conf_log_.size() / 2;
      conf_log_.erase(conf_log_.begin(), conf_log_.begin() + static_cast<std::ptrdiff_t>(drop));
      conf_base_ += drop;
    }
  }
  // Implications under the global bounds become global bound changes.
  std::lock_guard<std::mutex> lk(gfix_mu_);
  if (glb_.empty()) return;
  std::vector<Index> t;
  if (!propagate_conflict(*c, p_.is_int, glb_, gub_, &t, nullptr)) return;  // no improving solution left
  for (Index j : t) gfix_log_.push_back({j, glb_[j], gub_[j]});
  dbg_global("global conflict propagation");
}

void Search::dbg_global(const char* what) {
  if (debug_sol_.empty() || glb_.empty() || !(p_.objective(debug_sol_) - p_.offset < cutoff_.load() - 1e-9)) return;
  for (Index j = 0; j < p_.num_cols(); ++j) {
    if (debug_sol_[j] < glb_[j] - 1e-6 || debug_sol_[j] > gub_[j] + 1e-6) {
      log_.log(0, "DEBUG-SOL: %s excludes the optimum globally (col %d = %g, [%g, %g])\n", what, j, debug_sol_[j],
               glb_[j], gub_[j]);
      return;
    }
  }
}

void Search::pull_conflicts(Worker& w) {
  std::lock_guard<std::mutex> lk(conf_mu_);
  w.conf_seen = std::max(w.conf_seen, conf_base_);
  for (; w.conf_seen < conf_base_ + conf_log_.size(); ++w.conf_seen) {
    const auto& c = conf_log_[w.conf_seen - conf_base_];
    if (c->owner != w.tid) w.conf->add(c, w.nodes);
  }
}

bool Search::propagate_node(Worker& w) {
  BoundTrail* trail = w.trail.last.empty() ? nullptr : &w.trail;
  w.prop->trail = trail;
  const bool ok = propagate_node_impl(w, trail);
  w.prop->trail = nullptr;
  return ok;
}

bool Search::propagate_node_impl(Worker& w, BoundTrail* trail) {
  // Rows and conflicts alternately, each seeded with what the other tightened.
  w.tightened.clear();
  const bool rows_ok0 = w.prop->propagate(w.lb, w.ub, w.seeds, w.tightened);
  w.prop_work += w.prop->last_work;
  if (!rows_ok0) {
    for (Index j : w.tightened) w.changed.push_back(j);
    return false;
  }
  for (Index j : w.tightened) w.changed.push_back(j);
  if (!conflicts_ || w.conf->size() == 0) return true;
  // Budget: a share of the worker's LP work (iterations x LP size) so far,
  // from 20% up to 200% as the conflicts prune more of the nodes they are tried on.
  const double hit_rate = static_cast<double>(w.conf_prunes + 1) / static_cast<double>(w.conf_calls + 50);
  const double share = std::min(2.0, 0.2 + 20.0 * hit_rate);
  const long long budget = static_cast<long long>(share * static_cast<double>(w.lp_work)) + 1000 - w.conf_work;
  w.conf->work_limit = budget <= 0 ? -1 : std::min<long long>(budget, 200000);
  // Clauses (watched literals) are cheaper per useful implication: a separate
  // budget relative to LP and row propagation work, from 10% up to 200% as
  // they imply bounds or prune at more of the nodes.
  const double clause_rate = static_cast<double>(w.clause_hits + 1) / static_cast<double>(w.clause_calls + 50);
  const double clause_share = std::min(2.0, 0.1 + 20.0 * clause_rate);
  const long long cbudget =
      static_cast<long long>(clause_share * static_cast<double>(w.lp_work + w.prop_work)) + 1000 - w.clause_work;
  w.conf->clause_work_limit = cbudget <= 0 ? -1 : std::min<long long>(cbudget, 200000);
  if (budget > 0) ++w.conf_calls;
  if (cbudget > 0) ++w.clause_calls;
  bool clause_hit = false;
  std::vector<Index> seeds = w.seeds;
  seeds.insert(seeds.end(), w.tightened.begin(), w.tightened.end());
  for (int pass = 0; pass < 4 && !seeds.empty(); ++pass) {
    w.ctight.clear();
    const bool ok = w.conf->propagate(p_.is_int, w.glb, w.gub, w.lb, w.ub, seeds, w.ctight, w.nodes, trail);
    w.conf_work += w.conf->last_work();
    w.clause_work += w.conf->last_clause_work();
    if (w.conf->last_clause_hits() > 0 && !clause_hit) clause_hit = true, ++w.clause_hits;
    for (Index j : w.ctight) w.changed.push_back(j);
    if (!ok) {
      ++conf_pruned_;
      ++w.conf_prunes;
      return false;
    }
    if (w.ctight.empty()) break;
    w.tightened.clear();
    const bool rows_ok = w.prop->propagate(w.lb, w.ub, w.ctight, w.tightened);
    w.prop_work += w.prop->last_work;
    for (Index j : w.tightened) w.changed.push_back(j);
    if (!rows_ok) {
      ++conf_pruned_;
      ++w.conf_prunes;
      return false;
    }
    seeds = w.tightened;
  }
  return true;
}

double Search::probe(Worker& w, Index j, double lb, double ub, double obj, const Basis& basis, bool* exact) {
  DualSimplex& lp = w.lp;
  const double old_lb = lp.col_lower(j), old_ub = lp.col_upper(j);
  lp.set_col_bounds(j, lb, ub);
  lp.restore_state(w.state);
  LpOptions opt;
  opt.tol = params_.tol;
  opt.iteration_limit = w.probe_limit;
  opt.objective_cutoff = cutoff(w);
  opt.time_limit = params_.time_limit - timer_.elapsed();
  const long long it0 = lp.iterations();
  const LpStatus st = lp.solve(opt);
  lp_iters_ += lp.iterations() - it0;
  w.sb_iters += lp.iterations() - it0;
  sb_iters_ += lp.iterations() - it0;
  ++sb_probes_;
  double gain;
  if (st == LpStatus::Infeasible || st == LpStatus::CutoffReached) gain = kInf;
  else if (st == LpStatus::Optimal || st == LpStatus::IterationLimit) gain = std::max(lp.objective() - obj, 0.0);
  else gain = 0.0;
  if (st == LpStatus::Optimal && lp.objective() >= cutoff(w)) gain = kInf;
  *exact = st == LpStatus::Optimal;
  // Opt-in (MILP_SB_CONFLICT=1): dual proof of the pruned child (its bound change is not on the
  // trail: no clause). Like MILP_SB_BOUNDS, neutral on miplib_dev / dev set and no help on the targets.
  static const bool sb_conf = std::getenv("MILP_SB_CONFLICT") != nullptr;
  if (sb_conf && (st == LpStatus::Infeasible || st == LpStatus::CutoffReached)) learn_conflict(w, st, false);
  static const bool check_lp = std::getenv("MILP_CHECK_LP") != nullptr;
  if (check_lp && !is_finite(gain)) {
    // Debug: verify the probe's infeasibility/cutoff claim with a fresh LP (all rows incl. cuts).
    std::vector<double> clb(p_.num_cols()), cub(p_.num_cols());
    for (Index k = 0; k < p_.num_cols(); ++k) clb[k] = lp.col_lower(k), cub[k] = lp.col_upper(k);
    const SparseMatrix& R = lp.row_matrix();
    std::vector<double> rlb(lp.num_rows()), rub(lp.num_rows());
    for (Index i = 0; i < lp.num_rows(); ++i) rlb[i] = -lp.var_upper(lp.num_cols() + i), rub[i] = -lp.var_lower(lp.num_cols() + i);
    DualSimplex fresh;
    fresh.load(R.transpose(), p_.cost, clb, cub, rlb, rub);
    LpOptions o2;
    o2.perturb = false;
    const LpStatus st2 = fresh.solve(o2);
    if (st2 == LpStatus::Optimal && fresh.objective() < cutoff(w) - 1e-6) {
      log_.log(0, "PROBE MISMATCH var %d [%g,%g]: probe %s, fresh optimal %.10g < cutoff %.10g\n", j, lb, ub,
               to_string(st), fresh.objective(), cutoff(w));
      static std::atomic<bool> dumped{false};
      const char* dump = std::getenv("MILP_DUMP_LP");
      if (dump && !dumped.exchange(true)) {
        // n m / cost / col bounds / row bounds / CSR rows / start basis / options
        std::FILE* f = std::fopen(dump, "w");
        const Index n = p_.num_cols(), m = lp.num_rows();
        std::fprintf(f, "%d %d\n", n, m);
        for (Index k = 0; k < n; ++k) std::fprintf(f, "%.17g %.17g %.17g\n", p_.cost[k], clb[k], cub[k]);
        for (Index i = 0; i < m; ++i) std::fprintf(f, "%.17g %.17g\n", rlb[i], rub[i]);
        std::fprintf(f, "%d\n", R.nnz());
        for (Index i = 0; i < m; ++i)
          for (Index k = R.begin(i); k < R.end(i); ++k) std::fprintf(f, "%d %d %.17g\n", i, R.index[k], R.value[k]);
        const Basis& b0 = w.state.status.empty() ? basis : Basis{w.state.status};
        for (VarStatus v : b0.status) std::fprintf(f, "%d ", static_cast<int>(v));
        std::fprintf(f, "\n%.17g %lld\n", cutoff(w), opt.iteration_limit);
        std::fclose(f);
        log_.log(0, "  dumped LP to %s\n", dump);
      }
    }
  }
  lp.set_col_bounds(j, old_lb, old_ub);
  return gain;
}

Index Search::select_branch(Worker& w, double obj, const Basis& basis) {
  static const bool no_sb = std::getenv("MILP_NO_SB") != nullptr;
  static const bool sb_dyn = knob("MILP_SB_DYN", 0) > 0;
  static const int kReliability = static_cast<int>(knob("MILP_REL", 4));
  static const int kMaxProbes = no_sb ? 0 : static_cast<int>(knob("MILP_MAXPROBES", 8));
  static const int kLookahead = static_cast<int>(knob("MILP_LOOKAHEAD", 4));
  // Dynamic strong branching effort (SCIP-like), opt-in MILP_SB_DYN=1: probe iteration limit twice
  // the worker's average node LP (at least MILP_PROBE_ITERS, at most 500), reliability between
  // kMinRel and kReliability by the share of strong branching iterations in the worker's LP work
  // (full while below half of the node LP iterations + 10000, kMinRel above all of them). Not the
  // default: piperout-08 / rmatr100-p10 / mc11 1.2-1.8x slower at 60 s / 2 threads (longer probes on
  // expensive LPs); with MILP_REL=8 drayage-25-23 0/3 solved (degenerate probes), kasai's bound at
  // 60 s better (gap 0.03-0.06% vs 0.07%) but still unsolved.
  static const long long kProbeIters = static_cast<long long>(knob("MILP_PROBE_ITERS", 100));
  static const int kMinRel = static_cast<int>(knob("MILP_REL_MIN", 1));
  int reliability = kReliability;
  w.probe_limit = kProbeIters;
  if (sb_dyn) {
    const double avg = w.tree_lps > 0 ? static_cast<double>(w.tree_iters) / static_cast<double>(w.tree_lps) : 0.0;
    w.probe_limit = std::clamp<long long>(static_cast<long long>(2.0 * avg), kProbeIters, std::max<long long>(kProbeIters, 500));
    const double sb = static_cast<double>(w.sb_iters), tree = static_cast<double>(w.tree_iters) + 10000.0;
    if (sb > tree) reliability = kMinRel;
    else if (sb > 0.5 * tree) reliability = kMinRel + static_cast<int>(std::lround((kReliability - kMinRel) * (tree - sb) / (0.5 * tree)));
  }
  ++sb_calls_;
  // Dual degeneracy (SCIP's degeneracy-aware hybrid branching, below the root): degeneracy d = share of
  // the nonbasic variables (structurals and logicals, fixed ones excluded) with a zero reduced cost,
  // ratio = (basic + zero-reduced-cost nonbasic variables) / rows (size of the optimal face). Factor
  // 10^(10 (d - 0.7)) for d >= 0.8, times 10 ratio for ratio >= 2; from factor 10 on the node's probes
  // would mostly measure other vertices of the same optimal face (zero gains): no strong branching.
  // Opt-in (MILP_DEGEN_SB=1): miplib_dev 0.91 but timtab1 1.7-1.9x the nodes on 2 of 3 seeds.
  static const bool degen_sb = knob("MILP_DEGEN_SB", 0) > 0;
  int max_probes = kMaxProbes;
  double degen_factor = 1.0;
  if (degen_sb && w.node_depth > 0 && max_probes > 0) {
    const Index n = w.lp.num_cols(), m = w.lp.num_rows();
    long long nonbasic = 0, zero = 0;
    for (Index k = 0; k < n + m; ++k) {
      const VarStatus vs = w.lp.status(k);
      if (vs == VarStatus::Basic || w.lp.var_lower(k) == w.lp.var_upper(k)) continue;
      ++nonbasic;
      if (std::abs(w.lp.reduced_cost(k)) <= 1e-9) ++zero;
    }
    const double degeneracy = nonbasic > 0 ? static_cast<double>(zero) / static_cast<double>(nonbasic) : 0.0;
    const double ratio = m > 0 ? static_cast<double>(m + zero) / static_cast<double>(m) : 1.0;
    double factor = 1.0;
    if (degeneracy >= 0.8) factor = std::pow(10.0, 10.0 * (degeneracy - 0.7));
    if (ratio >= 2.0) factor *= 10.0 * ratio;
    if (factor >= 10.0) {
      max_probes = 0;
      ++sb_degen_skips_;
    }
    degen_factor = factor;
  }
  double ps_scale = 1.0;
  w.pc->scores(w.fracs, reliability, w.scored, &w.est_sum, &ps_scale, degen_factor);
  static const int tie_rule = static_cast<int>(knob("MILP_TIE", 0));
  if (tie_rule == 0) {
    std::sort(w.scored.begin(), w.scored.end(),
              [](const Pseudocosts::Scored& a, const Pseudocosts::Scored& b) { return a.score > b.score; });
  } else if (tie_rule == 1) {
    std::stable_sort(w.scored.begin(), w.scored.end(),
                     [](const Pseudocosts::Scored& a, const Pseudocosts::Scored& b) { return a.score > b.score; });
  } else {
    // Exact ties (e.g. all products at the 1e-12 floor) broken by fractionality, then column order.
    std::stable_sort(w.scored.begin(), w.scored.end(), [](const Pseudocosts::Scored& a, const Pseudocosts::Scored& b) {
      return std::min(a.f, 1.0 - a.f) > std::min(b.f, 1.0 - b.f);
    });
    std::stable_sort(w.scored.begin(), w.scored.end(),
                     [](const Pseudocosts::Scored& a, const Pseudocosts::Scored& b) { return a.score > b.score; });
  }
  Index best = w.scored.front().j;
  double best_score = -1.0;
  int probes = 0, no_improve = 0;
  bool probed = false;
  static const bool sb_bounds = std::getenv("MILP_SB_BOUNDS") != nullptr;  // opt-in, see below
  // Opt-in MILP_SB_FIX=1 (strong branching domain reductions, below): A/B vs 6504b88 at 60 s / 2 threads
  // miplib_dev 1.049 [0.96, 1.15], dev medium 1.011 [0.97, 1.08] (gap 0.82, lotsize 1.56): no gain.
  static const bool sb_fix = knob("MILP_SB_FIX", 0) > 0;
  Index first_fixed = -1;
  w.sb_var = -1;
  double cand_dn = -kInf, cand_up = -kInf;  // exact gains of the current best candidate
  for (const auto& c : w.scored) {
    cand_dn = cand_up = -kInf;
    double score = c.score;
    if (!c.reliable && probes < max_probes && no_improve < kLookahead) {
      const double v = w.xnode[c.j];
      bool ex_dn = false, ex_up = false;
      const double dn = probe(w, c.j, w.lp.col_lower(c.j), std::floor(v), obj, basis, &ex_dn);
      const double up = probe(w, c.j, std::ceil(v), w.lp.col_upper(c.j), obj, basis, &ex_up);
      probed = true;
      ++probes;
      if (!is_finite(dn) && !is_finite(up)) {
        w.lp.restore_state(w.state);
        return -1;
      }
      // Infeasible / cutoff probes are proofs; finite gains count only from LPs solved to optimality.
      cand_dn = !is_finite(dn) || ex_dn ? dn : -kInf;
      cand_up = !is_finite(up) || ex_up ? up : -kInf;
      if (!is_finite(dn) || !is_finite(up)) {
        if (!sb_fix || sb_bounds) {
          best = c.j;  // one child is infeasible: the other is all that remains
          if (sb_bounds) w.sb_var = c.j, w.sb_dn = cand_dn, w.sb_up = cand_up;
          break;
        }
        // Strong branching domain reduction (as SCIP): the infeasible side is cut off at this node --
        // the remaining side's bound is applied to the LP (later probes see it) and enters both
        // children's paths with the reduced-cost fixings; probing continues with the other candidates.
        const double nl = is_finite(dn) ? w.lp.col_lower(c.j) : std::ceil(v);
        const double nu = is_finite(dn) ? std::floor(v) : w.lp.col_upper(c.j);
        w.lp.set_col_bounds(c.j, nl, nu);
        w.lb[c.j] = nl;
        w.ub[c.j] = nu;
        w.changed.push_back(c.j);
        w.rcfix.push_back({c.j, nl, nu});
        ++sb_fixed_;
        if (first_fixed < 0) first_fixed = c.j;
        if (is_finite(dn)) pc_update(w, c.j, -1, dn / c.f);
        else pc_update(w, c.j, 1, up / (1.0 - c.f));
        continue;
      }
      pc_update(w, c.j, -1, dn / c.f);
      pc_update(w, c.j, 1, up / (1.0 - c.f));
      score = Pseudocosts::ps_part(std::max(dn, 1e-6) * std::max(up, 1e-6), ps_scale) /
                  (ps_scale > 0.0 ? degen_factor : 1.0) + c.bonus;
      if (score > best_score) {
        no_improve = 0;
      } else {
        ++no_improve;
      }
    }
    if (score > best_score) {
      best_score = score, best = c.j;
      if (sb_bounds) w.sb_var = c.j, w.sb_dn = cand_dn, w.sb_up = cand_up;
    }
  }
  if (probed) w.lp.restore_state(w.state);
  // Every scored candidate was fixed by strong branching (or the best one was): branch on the first
  // fixed one -- its cut-off child contradicts the fixing in its path and closes without an LP.
  if (first_fixed >= 0 && (best_score < 0.0 || w.lp.col_lower(best) == w.lp.col_upper(best) ||
                           w.xnode[best] < w.lp.col_lower(best) || w.xnode[best] > w.lp.col_upper(best))) {
    best = first_fixed;
  }
  return best;
}

void Search::run_dives(Worker& w) {
  static const bool disabled = std::getenv("MILP_NO_DIVE") != nullptr;
  // Propagating dives only at the root (MILP_TREE_DIVE_PROP=1 for tree dives too): in the
  // tree they cost timtab1 its late RINS successes (0/3 vs 2/3 seeds at the optimum).
  static const bool prop = std::getenv("MILP_TREE_DIVE_PROP") != nullptr;
  if (disabled) return;
  constexpr DiveRule kRules[] = {DiveRule::Fractional, DiveRule::Coefficient, DiveRule::Guided,
                                 DiveRule::VectorLength};
  if (++w.nodes_since_dive < 100) return;
  w.nodes_since_dive = 0;
  // Diving may use up to ~10% of the search's LP iterations (plus a base budget).
  const long long iters = w.det ? w.iters_base : lp_iters_.load();
  const long long dived = w.det ? w.dive_base + w.dive_iters : dive_iters_.load();
  const long long budget = 1000 + iters / 10 - dived;
  if (budget <= 0) return;
  const DiveRule rule = kRules[w.next_rule++ % 4];
  std::vector<double> inc;
  incumbent_x(w, inc);
  DiveLimits lim;
  lim.cutoff = cutoff(w);
  lim.iteration_budget = budget;
  lim.time_limit = params_.time_limit - timer_.elapsed();
  long long its = 0;
  if (dive(p_, w.lp, rule, rounding_, inc.empty() ? nullptr : &inc, lim, w.sol, &its, prop ? w.prop.get() : nullptr)) {
    try_incumbent(w, w.sol, to_string(rule));
  }
  lp_iters_ += its;
  dive_iters_ += its;
  w.dive_iters += its;
  w.lp.restore_state(w.state);
}

void Search::root_heuristics(Worker& w) {
  // Time this search spent until the end of the root LP with cuts (after a restart: of the
  // restarted root only).
  const double t_root = timer_.elapsed() - t_start_;
  struct Result {
    std::vector<double> x;
    const char* source = nullptr;
    long long iters = 0, dive_iters = 0;
  };
  using Task = std::function<void(Result&)>;
  auto run_tasks = [&](const std::vector<Task>& tasks) {
    std::vector<Result> res(tasks.size());
    std::atomic<size_t> next{0};
    auto loop = [&] {
      for (size_t k; (k = next++) < tasks.size();) tasks[k](res[k]);
    };
    std::vector<std::thread> pool;
    const int nt = std::min(threads_, static_cast<int>(tasks.size()));
    for (int t = 1; t < nt; ++t) pool.emplace_back(loop);
    loop();
    for (auto& th : pool) th.join();
    int wins = 0;
    for (Result& r : res) {
      lp_iters_ += r.iters;
      dive_iters_ += r.dive_iters;
      if (!r.x.empty()) wins += try_incumbent(w, r.x, r.source);
    }
    return wins;
  };
  auto remaining = [&] { return std::max(0.0, params_.time_limit - timer_.elapsed()); };
  // Wall-clock caps of the heuristics below. None in deterministic mode, where a binding cap would
  // make the result timing dependent (their iteration and node limits bound them there).
  auto capped = [&](double cap) { return params_.deterministic ? remaining() : std::min(cap, remaining()); };
  // Deterministic mode: LP-work caps of the root sub-MIPs instead, like opportunistic mode's ~3x the
  // root's time (>= 2 s): 3x the root LP work, at least 2e8 work units (iterations x (rows +
  // columns), ~2 s on rmatr100-p10 and setpack_large), as LP iterations of this problem.
  const double dims = static_cast<double>(p_.num_rows() + p_.num_cols());
  const long long det_sub_iters =
      params_.deterministic
          ? static_cast<long long>(std::max(2e8, 3.0 * static_cast<double>(w.lp.iterations()) * dims) / std::max(1.0, dims))
          : -1;

  // Phase 1: one dive per rule, each on its own copy of the root LP.
  static const bool no_dive = std::getenv("MILP_NO_DIVE") != nullptr;
  static const bool no_dive_prop = std::getenv("MILP_NO_DIVE_PROP") != nullptr;
  if (!no_dive) {
    constexpr DiveRule kRules[] = {DiveRule::Fractional, DiveRule::Coefficient, DiveRule::Guided,
                                   DiveRule::VectorLength};
    std::vector<double> inc;
    incumbent_x(w, inc);
    // Each dive may use about twice the root LP effort (the dives run in parallel).
    const long long budget = std::max<long long>(5000, 2 * w.lp.iterations());
    const double tl = capped(std::max(2.0, 0.05 * remaining()));
    const double cut = cutoff(w);
    std::vector<Task> tasks;
    for (DiveRule rule : kRules) {
      tasks.push_back([&, rule](Result& r) {
        DualSimplex lp = w.lp;
        std::unique_ptr<DomainPropagator> prop;
        if (!no_dive_prop) prop = std::make_unique<DomainPropagator>(p_);
        DiveLimits lim;
        lim.cutoff = cut;
        lim.iteration_budget = budget;
        lim.time_limit = tl;
        std::vector<double> sol;
        const double t0 = timer_.elapsed();
        const bool ok = dive(p_, lp, rule, rounding_, inc.empty() ? nullptr : &inc, lim, sol, &r.iters, prop.get());
        r.dive_iters = r.iters;
        if (ok) r.x = std::move(sol), r.source = to_string(rule);
        log_.log(3, "  root %s: %s, %lld iters, %.2fs\n", to_string(rule), ok ? "found" : "failed", r.iters,
                 timer_.elapsed() - t0);
      });
    }
    run_tasks(tasks);
  }

  // Phase 2: feasibility pump (+ RENS around its closest point) if nothing was
  // found yet, and RENS around the root LP point.
  std::vector<Task> tasks;
  if (!is_finite(incumbent(w)) && std::getenv("MILP_NO_FP") == nullptr) {
    tasks.push_back([&](Result& r) {
      FeasPumpLimits lim;
      lim.time_limit = capped(std::min(10.0, 0.1 * remaining()));
      lim.iteration_budget = std::max<long long>(20000, 20 * lp_iters_.load());
      lim.seed = params_.seed;
      static const int fp_stall = static_cast<int>(knob("MILP_FP_STALL", 20));
      lim.stall_rounds = fp_stall;
      std::vector<double> closest, sol;
      const double t0 = timer_.elapsed();
      const bool ok = feasibility_pump(p_, w.lp, lim, sol, &r.iters, &closest);
      log_.log(3, "  root feasibility pump: %s, %lld iters, %.2fs\n", ok ? "found" : "failed", r.iters,
               timer_.elapsed() - t0);
      if (ok) {
        r.x = std::move(sol), r.source = "feasibility pump";
      } else if (!closest.empty() && opts_.sub_mip_heuristics) {
        // Stage 3: RENS around the LP point closest to integrality.
        SubMipLimits sl;
        sl.node_limit = 2000;
        sl.iteration_limit = det_sub_iters;
        sl.time_limit = capped(std::min(10.0, 0.1 * remaining()));
        const double t1 = timer_.elapsed();
        const bool ok3 = rens(p_, closest.data(), kInf, sl, sol);
        if (ok3) r.x = std::move(sol), r.source = "feasibility pump + RENS";
        log_.log(3, "  root FP + RENS: %s, %.2fs\n", ok3 ? "found" : "failed", timer_.elapsed() - t1);
      }
    });
  }
  static const bool no_submip = std::getenv("MILP_NO_SUBMIP") != nullptr;
  if (!no_submip && opts_.sub_mip_heuristics) {
    const double cut = incumbent(w) + p_.offset;
    tasks.push_back([&, cut, d = w.lp.reduced_costs()](Result& r) {  // only FP reads w.lp concurrently
      SubMipLimits lim;
      lim.time_limit = capped(std::min(10.0, 0.1 * remaining()));
      // At most ~3x the root's time (>= 2 s): the 1000-node limit is out of reach when nodes are
      // slow, and the 10 s cap then went to a sub-MIP the tree search would put to better use
      // (rmatr100-p10: 10 s RENS after a 0.1 s root; neos-1445765: 10 s twice, failing after the restart).
      // Not in deterministic mode, where a binding time limit would make the result timing dependent.
      // After a restart (the first root ran RENS already, and its incumbent is the start point) at
      // most about the restarted root's time (h80x6320d: 10 s failing RENS after a 3.4 s root, then
      // a 3 s tree).
      if (!params_.deterministic) {
        lim.time_limit = std::min(lim.time_limit, opts_.start.empty() ? std::max(2.0, 3.0 * t_root) : std::max(1.0, t_root));
      }
      lim.node_limit = 1000;
      lim.iteration_limit = det_sub_iters;
      SubMipStats ss;
      lim.stats = &ss;
      std::vector<double> sol;
      const double t0 = timer_.elapsed();
      const bool ok = rens(p_, w.xnode.data(), cut, lim, sol, &d);
      if (ok) r.x = std::move(sol), r.source = "RENS";
      log_.log(3, "  root RENS: %s, %lld nodes, %lld iters, %.2fs\n", ok ? "found" : "failed", ss.nodes, ss.lp_iterations,
               timer_.elapsed() - t0);
    });
  }
  run_tasks(tasks);

  // Phase 3: improve the incumbent around the root LP point after the cuts (HiGHS finds most
  // of its early incumbents this way): rounds of RINS plus mutation sub-MIPs (one per
  // thread), repeated while a round improves.
  static const bool no_root_lns = std::getenv("MILP_NO_ROOT_LNS") != nullptr;
  // Only after a root of some size (>= 2000 LP iterations incl. the cut loop): on quick roots
  // the sub-MIP setup costs more than the tree's LNS would.
  if (no_submip || no_root_lns || !opts_.sub_mip_heuristics || w.lp.iterations() < 2000) return;
  // After a restart only if the rounds before it improved the incumbent (neos-1445765: ~2.5 s
  // for a round that never succeeds).
  if (!opts_.root_lns) return;
  root_lns_ran_ = true;
  // Budget: about the time spent until the end of the root LP with cuts (at most 10% of the
  // time limit, 20 s), so easy instances pay little. Not the time so far: the dives and RENS
  // above may take 10 s on their own (neos-1445765), which doubled the LNS time for nothing.
  // Deterministic mode: an LP-iteration budget instead of the clock.
  const double lns_end = timer_.elapsed() + std::min({20.0, 0.1 * remaining(), t_root});
  const long long lns_iters = std::max<long long>(50000, 5 * w.lp.iterations());
  long long lns_used = 0;
  double fix = 0.7;
  bool rins_ok = true;
  for (int round = 0; round < 12 && is_finite(incumbent(w)); ++round) {
    // Skip when the incumbent is already within 0.5% of the root bound.
    if (incumbent(w) - root_bound_ <= 5e-3 * std::max(1.0, std::abs(incumbent(w) + p_.offset))) break;
    if (params_.deterministic ? lns_used >= lns_iters : timer_.elapsed() >= lns_end) break;
    std::vector<double> inc;
    incumbent_x(w, inc);
    const double before = incumbent(w), cut = before + p_.offset;
    std::vector<SubMipStats> stats(static_cast<size_t>(std::max(threads_, 2)));
    std::vector<Task> lns;
    auto limits = [&](SubMipStats* st) {
      SubMipLimits lim;
      lim.time_limit = params_.deterministic ? remaining() : std::max(0.1, std::min(5.0, lns_end - timer_.elapsed()));
      lim.node_limit = 500;
      lim.iteration_limit = params_.deterministic ? std::max<long long>(10000, lns_iters / 2) : -1;
      lim.stats = st;
      return lim;
    };
    const bool with_rins = rins_ok;
    if (with_rins) {
      lns.push_back([&, cut](Result& r) {
        std::vector<double> sol;
        if (rins(p_, w.xnode.data(), inc, cut, limits(&stats[0]), sol)) r.x = std::move(sol), r.source = "root RINS";
      });
    }
    for (int t = static_cast<int>(lns.size()), k = 0; t < std::max(threads_, 2); ++t, ++k) {
      const double f = std::min(0.95, fix + 0.1 * k);
      const unsigned seed = params_.seed * 7919u + 15485863u + static_cast<unsigned>(round * 64 + t);
      lns.push_back([&, cut, f, seed, t](Result& r) {
        std::vector<double> sol;
        if (mutation(p_, inc, f, seed, cut, limits(&stats[t]), sol)) r.x = std::move(sol), r.source = "root mutation";
      });
    }
    // Own successes only (the FJ thread's mutation sub-MIPs may improve the incumbent meanwhile).
    root_lns_improved_ = run_tasks(lns) > 0 || root_lns_improved_;
    const bool improved = incumbent(w) < before;
    log_.log(3, "  root LNS round %d: %s, incumbent %.10g, %.2fs\n", round, improved ? "improved" : "no",
             p_.display_sense * (incumbent(w) + p_.offset), timer_.elapsed());
    for (const SubMipStats& ss : stats) lns_used += ss.lp_iterations;
    bool any_exhausted = false, any_limit = false;
    for (size_t t = with_rins ? 1 : 0; t < lns.size(); ++t) {
      if (stats[t].solves == 0) continue;
      (stats[t].exhausted ? any_exhausted : any_limit) = true;
    }
    if (improved) {
      rins_ok = true;  // a new incumbent gives RINS a new neighbourhood
    } else {
      rins_ok = false;
      if (any_exhausted && !any_limit) fix = std::max(0.2, fix - 0.1);
      else if (any_limit) fix = std::min(0.9, fix + 0.05);
      // One retry (with adapted fixing) when the first round fails, unless this is the root
      // after a restart (the first search ran these rounds already).
      if (round >= 1 || !opts_.start.empty()) break;
    }
  }
}

bool Search::orbitope_pass(Worker& w, bool trail_on) {
  std::vector<Index> tight;
  std::vector<char> ones;
  for (int round = 0; round < 8; ++round) {
    tight.clear();
    ones.clear();
    for (const Orbitope& o : orbs_) {
      if (!orbitope_propagate(o, w.lb, w.ub, tight, ones)) {
        // Bounds tightened before the failure are reset with the node's other changes.
        for (Index j : tight) w.changed.push_back(j);
        return false;
      }
    }
    if (tight.empty()) return true;
    orb_fixed_ += static_cast<long long>(tight.size());
    for (size_t k = 0; k < tight.size(); ++k) {
      const Index j = tight[k];
      // Recorded as decisions: valid literals of learned clauses on the representatives.
      if (trail_on) w.trail.push(j, ones[k] == 0, ones[k] ? 1.0 : 0.0, BoundTrail::kDecision);
      w.changed.push_back(j);
    }
    w.prop->trail = trail_on ? &w.trail : nullptr;
    w.tightened.clear();
    const bool ok = w.prop->propagate(w.lb, w.ub, tight, w.tightened);
    w.prop->trail = nullptr;
    w.prop_work += w.prop->last_work;
    for (Index j : w.tightened) w.changed.push_back(j);
    if (!ok) {
      if (trail_on) learn_prop_conflict(w, "propagation");
      return false;
    }
    if (w.tightened.empty()) return true;
  }
  return true;
}

int Search::process_impl(Worker& w, Node& node, Node* children) {
  DualSimplex& lp = w.lp;
  if (w.lb.empty()) {
    w.glb = p_.col_lb;
    w.gub = p_.col_ub;
    w.lb = p_.col_lb;
    w.ub = p_.col_ub;
    w.prop = std::make_unique<DomainPropagator>(prop_problem());
    static const bool no_objprop = std::getenv("MILP_NO_OBJPROP") != nullptr;
    if (!no_objprop) w.prop->enable_objective();
    // Continuous bounds too (big-M models: jobshop_large 42 -> 18 s, cvrp_large 4.0 -> 1.6 s; rocI-4-11
    // prunes by infeasibility through continuous chains). MILP_PROP_CONT=0: integers only as before.
    w.conf = std::make_unique<ConflictPool>(p_.num_cols(), conf_capacity_);
    if (conflicts_ && prop_conflicts_) {
      // At most ~20 MB of trail per worker; a node with a longer trail is not analysed.
      w.trail.init(p_.num_cols(), std::min<size_t>(2 * static_cast<size_t>(p_.num_cols()) + 20000, 500000));
    }
  }
  pull_conflicts(w);
  {
    // Pick up global bound tightenings found since the last node.
    std::lock_guard<std::mutex> lk(gfix_mu_);
    for (; w.gfix_seen < gfix_log_.size(); ++w.gfix_seen) {
      const BoundChange& g = gfix_log_[w.gfix_seen];
      w.glb[g.col] = std::max(w.glb[g.col], g.lb);
      w.gub[g.col] = std::min(w.gub[g.col], g.ub);
      w.changed.push_back(g.col);
    }
  }
  // Reset bounds changed by the previous node, then apply this node's.
  for (Index j : w.changed) {
    lp.set_col_bounds(j, w.glb[j], w.gub[j]);
    w.lb[j] = w.glb[j];
    w.ub[j] = w.gub[j];
  }
  w.changed.clear();
  // The path is newest-first; apply oldest-first so later changes win.
  w.segs.clear();
  for (const PathSeg* sg = node.path.get(); sg; sg = sg->parent()) w.segs.push_back(sg);
  w.seeds.clear();
  static const bool no_prop = std::getenv("MILP_NO_PROP") != nullptr;
  // With a trail (conflict analysis), each path change is a decision whose
  // level is its depth on the path; implied bounds get the highest level among
  // the bounds they were derived from.
  const bool use_trail = !w.trail.last.empty() && !no_prop;
  if (use_trail) w.trail.clear();
  bool path_infeasible = false;
  auto apply = [&](const BoundChange& bc) {
    const Index j = bc.col;
    const double l = std::max(bc.lb, w.glb[j]), u = std::min(bc.ub, w.gub[j]);
    if (use_trail) {
      ++w.trail.level;
      if (l > w.lb[j]) w.trail.push(j, false, l, BoundTrail::kDecision);
      if (u < w.ub[j]) w.trail.push(j, true, u, BoundTrail::kDecision);
    }
    w.lb[j] = std::max(w.lb[j], l);  // later changes on the path are tighter
    w.ub[j] = std::min(w.ub[j], u);
    if (w.lb[j] > w.ub[j]) path_infeasible = true;
    w.changed.push_back(j);
    w.seeds.push_back(j);
  };
  for (auto it = w.segs.rbegin(); it != w.segs.rend(); ++it) (*it)->for_each(apply);
  if (path_infeasible) {  // global fixings contradict this node's branching
    for (Index j : w.changed) lp.set_col_bounds(j, w.lb[j], std::max(w.lb[j], w.ub[j]));
    ++nodes_;
    ++w.nodes;
    count_outcome(0, node.depth);
    return 0;
  }
  // Debug: which reduction excludes the known optimum from a node that contains it.
  auto dbg_out = [&](const char* step) {
    if (debug_sol_.empty()) return false;
    for (Index j = 0; j < p_.num_cols(); ++j) {
      if (debug_sol_[j] < w.lb[j] - 1e-6 || debug_sol_[j] > w.ub[j] + 1e-6) {
        if (step) log_.log(0, "DEBUG-SOL: node %lld: %s excludes the optimum (col %d = %g, bounds [%g, %g])\n", node.id, step, j, debug_sol_[j], w.lb[j], w.ub[j]);
        return true;
      }
    }
    return false;
  };
  // Only while the debug point is better than the cutoff (reductions may drop points that are not).
  const bool dbg_in = !debug_sol_.empty() && p_.objective(debug_sol_) - p_.offset < cutoff(w) - 1e-9 && !dbg_out(nullptr);
  // Domain propagation: implied bounds or infeasibility without an LP solve.
  if (!no_prop && !w.seeds.empty()) {
    w.prop->set_objective_cutoff(cutoff(w));
    // Continuous bounds too, while that pays off (big-M models: jobshop_large 53 -> 14 s, 194k -> 19k
    // nodes; cvrp_large 4.0 -> 1.6 s; rocI-4-11 prunes through continuous chains). On other models it
    // only costs (neos-2978193-inde: propagation became the top hotspot, nodes 2x slower): a node's
    // continuous propagation is useful if a row failed or tightened an integer with a continuous bound
    // tightened in the same call. Below one useful node in 20 (decayed counts), only every 16th node
    // propagates continuous bounds (to keep measuring). MILP_PROP_CONT=0: integers only as before;
    // 2: always.
    static const int prop_cont = static_cast<int>(knob("MILP_PROP_CONT", 1));
    const bool cont_on = prop_cont == 2 || (prop_cont == 1 && (w.cont_calls < 50.0 || w.cont_useful >= 0.05 * w.cont_calls ||
                                                             w.nodes % 16 == 0));
    w.prop->continuous = cont_on;
    // The objective cutoff row likewise (it is dense on many models: eil33-2, 4.5k columns, spent more
    // time in propagation than in node LPs, 20% of the run): useful when it failed or tightened a bound;
    // below one useful node in 20 only every 16th node uses it. Needs a finite cutoff to do anything.
    const bool obj_cut = is_finite(cutoff(w));
    const bool obj_on = !obj_cut || w.obj_calls < 50.0 || w.obj_useful >= 0.05 * w.obj_calls || w.nodes % 16 == 0;
    w.prop->objective = obj_on;
    w.prop->last_obj_useful = false;
    const size_t changed_before = w.changed.size();
    const bool ok = propagate_node(w);
    if (cont_on) {
      w.cont_calls = 0.999 * w.cont_calls + 1.0;
      w.cont_useful = 0.999 * w.cont_useful + (w.prop->last_cont_useful > 0 ? 1.0 : 0.0);
    }
    if (obj_cut) ++(obj_on ? obj_prop_calls_ : obj_prop_skipped_);
    if (obj_on && obj_cut && w.prop->last_obj_useful) ++obj_prop_useful_;
    if (obj_on && obj_cut) {
      w.obj_calls = 0.999 * w.obj_calls + 1.0;
      w.obj_useful = 0.999 * w.obj_useful + (w.prop->last_obj_useful ? 1.0 : 0.0);
    }
    if (node.branch_var >= 0) {
      // A child proven infeasible by propagation counts as many inferences.
      const double inferences = ok ? static_cast<double>(w.changed.size() - changed_before) : 10.0;
      w.pc->update_inference(node.branch_var, node.branch_dir, inferences);
      if (w.det) w.inf_log.emplace_back(node.branch_var, node.branch_dir, inferences);
    }
    if (!ok) {
      if (use_trail) learn_prop_conflict(w, "propagation");
      for (Index j : w.changed) lp.set_col_bounds(j, w.lb[j], std::max(w.lb[j], w.ub[j]));
      ++nodes_;
      ++w.nodes;
      count_outcome(1, node.depth);
      return 0;
    }
  } else if (!w.trail.last.empty()) {
    w.trail.clear();  // no propagation: the trail is not maintained
    w.trail.overflow = true;
  }
  if (dbg_in) dbg_out("propagation");
  const bool dbg_in2 = dbg_in && !dbg_out(nullptr);
  if (!orbs_.empty() && !no_prop && !orbitope_pass(w, use_trail && !w.trail.overflow)) {
    if (dbg_in2) log_.log(0, "DEBUG-SOL: node %lld: orbitope pass failed on a node containing the optimum\n", node.id);
    for (Index j : w.changed) lp.set_col_bounds(j, w.lb[j], std::max(w.lb[j], w.ub[j]));
    ++orb_pruned_;
    ++nodes_;
    ++w.nodes;
    count_outcome(2, node.depth);
    return 0;
  }
  if (dbg_in2) dbg_out("orbitope pass");
  if (!p_.implications.empty() && !p_.apply_implications(w.lb, w.ub, &w.changed)) {
    for (Index j : w.changed) lp.set_col_bounds(j, w.lb[j], std::max(w.lb[j], w.ub[j]));
    ++nodes_;
    ++w.nodes;
    count_outcome(1, node.depth);
    return 0;
  }
  for (Index j : w.changed) lp.set_col_bounds(j, w.lb[j], w.ub[j]);
  apply_global_cuts(w);
  const bool pool_on = use_pool_ && node.id != 0 && w.m_base >= 0;
  if (pool_on) {
    pool_pull(w);
    pool_prepare(w, node);
  } else if (node.basis && node.parent != w.last_node) {
    node.basis->unpack(w.unpacked);
    // Cut rows added after the basis was stored enter with basic logicals.
    const size_t need = static_cast<size_t>(lp.num_cols() + lp.num_rows());
    if (w.unpacked.status.size() < need) w.unpacked.status.resize(need, VarStatus::Basic);
    lp.set_basis(w.unpacked);
  }

  LpOptions opt;
  opt.tol = params_.tol;
  opt.objective_cutoff = cutoff(w);
  opt.time_limit = params_.time_limit - timer_.elapsed();
  const long long it0 = lp.iterations();
  static const char* dump_numeric = std::getenv("MILP_DUMP_NUMERIC");
  Basis start_basis;
  if (dump_numeric) start_basis = lp.get_basis();
  // Debug: dump a node LP that failed numerically, with the basis it started from.
  auto dump_failed = [&](const char* where, LpStatus status) {
    if (!dump_numeric || status != LpStatus::NumericalError || !debug_root_) return;
    static std::atomic<bool> dumped{false};
    if (dumped.exchange(true)) return;
    const Index n = p_.num_cols(), m = lp.num_rows();
    const SparseMatrix& R = lp.row_matrix();
    std::FILE* f = std::fopen(dump_numeric, "w");
    std::fprintf(f, "%d %d\n", n, m);
    for (Index k = 0; k < n; ++k) std::fprintf(f, "%.17g %.17g %.17g\n", p_.cost[k], lp.col_lower(k), lp.col_upper(k));
    for (Index i = 0; i < m; ++i) std::fprintf(f, "%.17g %.17g\n", -lp.var_upper(n + i), -lp.var_lower(n + i));
    std::fprintf(f, "%d\n", R.nnz());
    for (Index i = 0; i < m; ++i)
      for (Index k = R.begin(i); k < R.end(i); ++k) std::fprintf(f, "%d %d %.17g\n", i, R.index[k], R.value[k]);
    start_basis.status.resize(static_cast<size_t>(n + m), VarStatus::Basic);
    for (VarStatus v : start_basis.status) std::fprintf(f, "%d ", static_cast<int>(v));
    std::fprintf(f, "\n%.17g %lld\n", opt.objective_cutoff, opt.iteration_limit);
    std::fclose(f);
    log_.log(0, "  dumped numerically failing node LP (%s) to %s\n", where, dump_numeric);
  };
  LpStatus st = lp.solve(opt);
  if (debug_fail(node.id)) st = LpStatus::NumericalError;
  bool recovery_tried = false;
  dump_failed("node solve", st);
  lp_iters_ += lp.iterations() - it0;
  w.lp_work += (lp.iterations() - it0 + 1) * (lp.num_rows() + lp.num_cols());
  // A node that does not continue this worker's plunge (its parent was not the previous node).
  const bool plunge_start = node.parent != w.last_node;
  w.last_node = node.id;
  ++nodes_;
  ++w.nodes;

  static const bool check_lp = std::getenv("MILP_CHECK_LP") != nullptr;
  if (check_lp && (st == LpStatus::Infeasible || st == LpStatus::CutoffReached || st == LpStatus::Optimal)) {
    // Debug: re-solve this node LP from scratch and compare.
    std::vector<double> lb(p_.num_cols()), ub(p_.num_cols());
    for (Index j = 0; j < p_.num_cols(); ++j) lb[j] = lp.col_lower(j), ub[j] = lp.col_upper(j);
    DualSimplex fresh;
    fresh.load(p_.A, p_.cost, lb, ub, p_.row_lb, p_.row_ub);
    LpOptions o2;
    o2.perturb = false;
    const LpStatus st2 = fresh.solve(o2);
    bool bad;
    if (st == LpStatus::Infeasible) bad = st2 != LpStatus::Infeasible;
    else if (st == LpStatus::CutoffReached) bad = st2 == LpStatus::Optimal && fresh.objective() < opt.objective_cutoff - 1e-6;
    else bad = st2 != LpStatus::Optimal || std::abs(fresh.objective() - lp.objective()) > 1e-6 * (1 + std::abs(lp.objective()));
    if (bad && st == LpStatus::Optimal) {
      log_.log(0, "LP MISMATCH optimal: warm obj %.10g fresh %s %.10g\n", lp.objective(), to_string(st2), fresh.objective());
    }
    if (bad) {
      log_.log(0, "LP MISMATCH node %lld depth %d: warm %s, fresh %s obj %.10g cutoff %.10g\n", node.id, node.depth,
               to_string(st), to_string(st2), fresh.objective(), opt.objective_cutoff);
    }
  }
  if (st == LpStatus::Infeasible || st == LpStatus::CutoffReached) {
    if (node.id != 0) {
      if (st == LpStatus::CutoffReached || !is_finite(cutoff(w))) pool_evidence(w, node.depth);
      learn_conflict(w, st);
    }
    count_outcome(st == LpStatus::Infeasible ? 3 : 4, node.depth);
    return 0;
  }
  if (st == LpStatus::TimeLimit) {
    std::lock_guard<std::mutex> lk(stop_mu_);
    stop_reason_ = Status::TimeLimit;
    stop_ = true;
    lost_bound_ = std::min(lost_bound_, node.bound);  // the unsolved node still limits the bound
    return 0;
  }
  if (st == LpStatus::Unbounded && !p_.implications.empty()) {
    // The LP ignores the implications (their columns are free there), so an unbounded LP says nothing
    // about the problem: branch on an open implication; with all of them decided, the subproblem is a
    // plain MILP, unbounded if it has a feasible point.
    const Index z = open_implication(w);
    if (z >= 0) return implication_branch(w, node, children, z, node.bound);
    return unbounded_leaf(w, node);
  }
  if (st == LpStatus::Unbounded && node.id == 0) {
    root_unbounded_ = true;
    stop_ = true;
    return 0;
  }
  if (st != LpStatus::Optimal && st != LpStatus::Infeasible && st != LpStatus::CutoffReached) {
    // Unexpected result (a bounded LP cannot become unbounded below the root): recovery ladder.
    st = recover_lp(w, opt, node.id);
    recovery_tried = true;
    if (st == LpStatus::Infeasible || st == LpStatus::CutoffReached) {
      if (node.id != 0) learn_conflict(w, st);
      count_outcome(st == LpStatus::Infeasible ? 3 : 4, node.depth);
      return 0;
    }
  }
  static const int kNodeCutDepth = static_cast<int>(knob("MILP_NODECUT_DEPTH", 6));
  if (pool_on && st == LpStatus::Optimal) {
    w.node_iters += lp.iterations() - it0;
    node_it_total_ += lp.iterations() - it0;
    if (plunge_start) popped_it_ += lp.iterations() - it0, ++popped_n_; else ++plunged_n_;
    {
      long long nnz = 0;
      for (int id : w.prow) nnz += static_cast<long long>(w.pool[id]->idx.size());
      prow_sum_ += static_cast<long long>(w.prow.size()), prow_nnz_ += nnz;
    }
    // Cut pool rounds: re-add violated pool cuts (and, near the root, separate new ones into the
    // pool) while the LP bound moves. Budget: resolves may cost about as much as the node LPs.
    static const int kPoolRounds = static_cast<int>(knob("MILP_POOL_ROUNDS", 3));
    static const double kPoolWork = knob("MILP_POOL_WORK", 1.0);
    const bool rounds = node.depth <= kNodeCutDepth || w.nodes % pool_interval(w) == 0;
    for (int round = 0; rounds && round < kPoolRounds; ++round) {
      if (w.pool_iters > kPoolWork * static_cast<double>(w.node_iters) + 2000) break;
      // Scanning the pool costs O(pool nonzeros): keep it below the node LP work.
      if (static_cast<double>(w.pool_scan) > kPoolWork * static_cast<double>(w.lp_work) + 1e6) break;
      const double before = lp.objective();
      int added = pool_separate(w);
      // Fresh separation (into the pool) down to depth kNodeCutDepth at every sep_interval-th node
      // of a worker; deeper only at every kDeepEvery-th of those (0: never). Deep separation at
      // every node helped kasai a lot but cost 15-90% on cflp/lotsize/gap (LP work of the cuts).
      static const int kDeepEvery = static_cast<int>(knob("MILP_POOL_DEEP_EVERY", 16));
      const long long every = static_cast<long long>(sep_interval(w)) *
                              (node.depth <= kNodeCutDepth ? 1 : std::max(kDeepEvery, 1));
      const bool fresh = round == 0 && (node.depth <= kNodeCutDepth || kDeepEvery > 0) &&
                         node_cuts_enabled_.load() && w.nodes % every == 0;
      const bool tmir = fresh && tmir_turn(w);
      if (fresh) added += separate_node(w, tmir);
      if (added == 0) break;
      ncp_rounds_ += 1;
      const long long it2 = lp.iterations();
      if (dump_numeric) start_basis = lp.get_basis();
      st = lp.solve(opt);
      dump_failed("after pool cuts", st);
      lp_iters_ += lp.iterations() - it2;
      w.pool_iters += lp.iterations() - it2;
      pool_it_total_ += lp.iterations() - it2;
      const bool pruned = st == LpStatus::Infeasible || st == LpStatus::CutoffReached;
      const double gap = cutoff(w) - before;
      const bool gained = st == LpStatus::Optimal && is_finite(gap) && lp.objective() - before >= 0.05 * gap;
      if (fresh) sep_result(w, pruned || gained);  // adaptive frequency instead of switching off
      // (Deterministic mode applies fresh cuts only at the round barrier: no signal, tableau cuts stay on.)
      if (tmir && !w.det && is_finite(gap)) {
        tmir_result(pruned || (st == LpStatus::Optimal && lp.objective() - before >= 0.2 * gap));
      }
      if (pruned) {
        if (st == LpStatus::CutoffReached || !is_finite(cutoff(w))) pool_evidence(w, node.depth);
        learn_conflict(w, st);
        count_outcome(5, node.depth);
        return 0;
      }
      if (st != LpStatus::Optimal) break;
      if (lp.objective() - before <= 1e-6 * std::max(1.0, std::abs(before))) break;
    }
    if (st == LpStatus::Optimal) pool_age(w);
    const int bad = w.det ? w.ev_bad : ev_bad_.load();
    if (st == LpStatus::Optimal && !rounds && !w.prow.empty() && bad >= 2) {
      st = pool_strip(w, opt);
      if (st == LpStatus::Infeasible || st == LpStatus::CutoffReached) {
        learn_conflict(w, st);
        count_outcome(5, node.depth);
        return 0;
      }
    }
  } else if (st == LpStatus::Optimal && node.id != 0 && node.depth <= kNodeCutDepth && node_cuts_enabled_.load()) {
    const double before = lp.objective();
    if (separate_node(w) > 0) {
      const long long it2 = lp.iterations();
      if (dump_numeric) start_basis = lp.get_basis();
      st = lp.solve(opt);
      dump_failed("after node cuts", st);
      lp_iters_ += lp.iterations() - it2;
      // Adaptive: keep separating only while it pays off (prunes the node or
      // closes >= 5% of its gap to the cutoff in at least 20% of the attempts).
      const double gap = cutoff(w) - before;
      const bool pruned = st == LpStatus::Infeasible || st == LpStatus::CutoffReached;
      const bool gained = st == LpStatus::Optimal && is_finite(gap) && lp.objective() - before >= 0.05 * gap;
      const long long tries = ++sep_tries_, wins = (pruned || gained) ? ++sep_wins_ : sep_wins_.load();
      if (tries >= 50 && wins * 5 < tries) {
        if (node_cuts_enabled_.exchange(false)) log_.log(2, "  node cuts disabled (%lld/%lld successful)\n", wins, tries);
      }
      if (pruned) {
        learn_conflict(w, st);
        count_outcome(5, node.depth);
        return 0;
      }
    }
  }
  if (st != LpStatus::Optimal && st != LpStatus::TimeLimit && !recovery_tried) {
    // A cut re-solve failed (the first solve's failures went through the ladder above already).
    st = recover_lp(w, opt, node.id);
    if (st == LpStatus::Infeasible || st == LpStatus::CutoffReached) {
      if (node.id != 0) learn_conflict(w, st);
      count_outcome(5, node.depth);
      return 0;
    }
  }
  if (st != LpStatus::Optimal) {
    if (node.id == 0 || st == LpStatus::TimeLimit) {
      // The root (no node to fall back on) or out of time: the subtree's bound still limits what we can prove.
      std::lock_guard<std::mutex> lk(stop_mu_);
      lost_bound_ = std::min(lost_bound_, node.bound);
      if (st == LpStatus::TimeLimit) {
        stop_reason_ = Status::TimeLimit;
        stop_ = true;
      } else {
        numerical_ = true;
        log_.log(1, "  node %lld: LP %s (%s), dropped\n", node.id, to_string(st), lp.fail_reason());
      }
      return 0;
    }
    log_.log(2, "  node %lld: LP %s (%s) after recovery, branching without it\n", node.id, to_string(st), lp.fail_reason());
    return unsolved_branch(w, node, children);
  }

  const double obj = lp.objective();
  if (node.id == 0) {
    root_bound_ = obj;
    log_.log(1, "Root LP: objective %.10g, %lld iterations, %.2fs\n", p_.display_sense * (obj + p_.offset), lp.iterations(),
             timer_.elapsed());
  }
  if (node.branch_var >= 0 && node.branch_dist > 0) {
    const double pobj = is_finite(node.parent_obj) ? node.parent_obj : node.bound;
    pc_update(w, node.branch_var, node.branch_dir, std::max(obj - pobj, 0.0) / node.branch_dist);
  }
  if (obj >= cutoff(w)) {
    if (node.id != 0) {
      pool_evidence(w, node.depth);
      learn_conflict(w, LpStatus::CutoffReached);
    }
    count_outcome(6, node.depth);
    return 0;
  }
  if (node.id == 0) {
    root_z_ = obj;
    root_d_.assign(p_.num_cols(), 0.0);
    root_at_.assign(p_.num_cols(), 0);
    glb_ = w.glb;
    gub_ = w.gub;
    for (Index j = 0; j < p_.num_cols(); ++j) {
      const VarStatus vs = lp.status(j);
      root_d_[j] = vs == VarStatus::Basic ? 0.0 : lp.reduced_cost(j);
      // Root reduced-cost fixing counts from the global bounds: only columns nonbasic at a global bound
      // (root-local fixings, e.g. orbitopal, leave others at a tighter bound).
      root_at_[j] = vs == VarStatus::AtLower && lp.col_lower(j) == p_.col_lb[j]   ? 1
                    : vs == VarStatus::AtUpper && lp.col_upper(j) == p_.col_ub[j] ? 2
                                                                                  : 0;
    }
    root_rc_ready_ = true;
    // An incumbent known before the root (restart start, external cutoff) fixes globally too
    // (counted for a further restart).
    if (is_finite(cutoff_.load())) root_rc_fixing(cutoff_.load());
  }
  // Node reduced-cost fixing: bounds beyond which the LP bound would exceed the cutoff.
  static const bool no_rcfix = std::getenv("MILP_NO_RCFIX") != nullptr;
  w.rcfix.clear();
  if (!no_rcfix && is_finite(cutoff(w))) {
    const double slack = cutoff(w) - obj;
    for (Index j = 0; j < p_.num_cols() && w.rcfix.size() < 256; ++j) {
      if (!p_.is_int[j]) continue;
      const VarStatus vs = lp.status(j);
      const double d = lp.reduced_cost(j), l = lp.col_lower(j), u = lp.col_upper(j);
      if (vs == VarStatus::AtLower && d > 1e-7) {
        const double nu = l + std::floor(slack / d + 1e-9);
        if (nu < u - 0.5) w.rcfix.push_back({j, l, nu});
      } else if (vs == VarStatus::AtUpper && d < -1e-7) {
        const double nl = u - std::floor(slack / -d + 1e-9);
        if (nl > l + 0.5) w.rcfix.push_back({j, nl, u});
      }
    }
    if (!debug_sol_.empty() && p_.objective(debug_sol_) - p_.offset < cutoff(w) - 1e-9) {
      for (const BoundChange& f : w.rcfix) {
        const double xs = debug_sol_[f.col];
        if (xs < f.lb - 1e-6 || xs > f.ub + 1e-6) {
          double dist = 0.0;  // how far the optimum is outside the node LP's column bounds
          for (Index k = 0; k < p_.num_cols(); ++k)
            dist = std::max({dist, lp.col_lower(k) - debug_sol_[k], debug_sol_[k] - lp.col_upper(k)});
          if (dist <= 1e-6) log_.log(0, "DEBUG-SOL: node %lld rcfix col %d to [%g, %g] excludes x* %g (d %g, LP [%g, %g], node [%g, %g], obj %.10g cutoff %.10g, x* outside LP bounds by %g)\n",
                   node.id, f.col, f.lb, f.ub, xs, lp.reduced_cost(f.col), lp.col_lower(f.col), lp.col_upper(f.col),
                   w.lb[f.col], w.ub[f.col], obj + p_.offset, cutoff(w) + p_.offset, dist);
        }
      }
    }
    for (const BoundChange& f : w.rcfix) {
      // Nonbasic at the kept bound: the LP solution stays optimal.
      lp.set_col_bounds(f.col, f.lb, f.ub);
      w.lb[f.col] = f.lb;
      w.ub[f.col] = f.ub;
      w.changed.push_back(f.col);
    }
  }

  const double* x = lp.x();
  const double itol = params_.tol.integrality;
  w.fracs.clear();
  for (Index j = 0; j < p_.num_cols(); ++j) {
    if (!p_.is_int[j]) continue;
    const double f = x[j] - std::floor(x[j]);
    if (f > itol && f < 1.0 - itol) w.fracs.emplace_back(j, f);
  }
  if (w.fracs.empty() && !p_.implications.empty()) {
    // Integral, but a logical implication may still be violated: branch on its binary.
    const Index z = violated_implication(w, x);
    if (z >= 0) return implication_branch(w, node, children, z, std::max(obj, node.bound));
  }
  if (w.fracs.empty()) {
    w.sol.assign(x, x + p_.num_cols());
    std::vector<double> chk;
    double chk_obj;
    // Rejected only because it is not better than the incumbent (e.g. found
    // earlier by a heuristic, LP objective a hair below the cutoff): fine.
    if (!try_incumbent(w, w.sol, "LP") && obj < cutoff(w) && !verified_point(w.sol, chk, &chk_obj)) {
      double iv = 0.0;
      const double viol = p_.max_violation(w.sol, &iv);
      // The LP point is integral within tolerance but violates the model: we cannot branch on its
      // fractionalities, so branch as for an unsolvable node LP (fresh LP, else a domain split).
      log_.log(2, "  node %lld: integral LP point violates model (viol %.3g, int %.3g), branching without it\n",
               node.id, viol, iv);
      return unsolved_branch(w, node, children);
    }
    count_outcome(7, node.depth);
    return 0;
  }
  if (node.id == 0 || w.nodes % 8 == 0) {
    if (rounding_.run(x, w.sol)) try_incumbent(w, w.sol, "rounding");
  }

  w.tree_iters += lp.iterations() - it0;  // before dives and probes move the LP
  ++w.tree_lps;
  w.xnode.assign(x, x + p_.num_cols());  // strong branching moves the LP
  if (node.id == 0) root_x_ = w.xnode;
  auto basis = std::make_shared<const Basis>(lp.get_basis());
  lp.save_state(w.state);
  if (node.id == 0) {
    root_heuristics(w);
  } else {
    const auto c0 = std::chrono::steady_clock::now();
    run_dives(w);
    w.t_dive += std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();  // statistics only
    run_sub_mips(w);
  }
  if (obj >= cutoff(w)) {
    count_outcome(9, node.depth);
    return 0;
  }
  w.node_depth = node.depth;
  const Index j = select_branch(w, obj, *basis);
  if (j < 0) {
    count_outcome(10, node.depth);
    return 0;
  }
  const double v = w.xnode[j];
  const double f = v - std::floor(v);
  double cur_lb = lp.col_lower(j), cur_ub = lp.col_upper(j);
  Node down, up;
  // Under memory pressure (memory_control) stop storing warm-start bases with queued nodes.
  const long long limit = mem_limit_bytes();
  const bool keep_basis = mem_state_.load() == 0;
  std::shared_ptr<const PackedBasis> packed;
  if (keep_basis) {
    if (use_pool_ && w.m_base >= 0) {
      auto* pb = new PackedBasis(*basis, p_.num_cols() + w.m_base, ref_basis_);
      w.nb.clear();
      for (size_t k = 0; k < w.prow.size(); ++k) {
        const VarStatus vs = basis->status[static_cast<size_t>(p_.num_cols() + w.m_base) + k];
        if (vs != VarStatus::Basic) w.nb.emplace_back(w.prow[k], vs);
      }
      std::sort(w.nb.begin(), w.nb.end());
      pb->set_pool(w.nb);
      packed = store_basis(pb);
    } else {
      packed = store_basis(new PackedBasis(*basis, -1, ref_basis_));
    }
  }
  for (Node* c : {&down, &up}) {
    // A child's region lies within its parent's: never report a bound below the parent's (a node LP
    // can come out lower numerically; roll3000 once ended OPTIMAL with a reported bound 0.7% low).
    c->bound = std::max(obj, node.bound);
    c->parent_obj = c->bound;  // before a strong branching bound raises it (pseudocost reference)
    c->depth = node.depth + 1;
    if (keep_basis) c->basis = packed;
    c->parent = node.id;
    c->id = next_node_id(w);
    c->branch_var = j;
  }
  // Bound paths: one segment with this node's reduced-cost fixings (shared by both children; not
  // kept under memory pressure: the children's LPs only start weaker), then one per child with its
  // branching bound (the down child also with the orbital fixings).
  PathRef base = node.path;
  if (!w.rcfix.empty() && keep_basis) {
    w.seg.clear();
    for (const BoundChange& rc : w.rcfix) PathSeg::encode(rc, p_.col_lb[rc.col], p_.col_ub[rc.col], w.seg);
    base = PathRef(PathSeg::make(node.path.get(), w.seg, &path_bytes_));
  }
  w.seg.clear();
  PathSeg::encode({j, cur_lb, std::floor(v)}, p_.col_lb[j], p_.col_ub[j], w.seg);
  down.sym = up.sym = node.sym;
  if (node.sym) {
    // Orbital branching: the down child also bounds the orbit of j.
    if (w.orbit_mark.empty()) w.orbit_mark.assign(p_.num_cols(), 0);
    node.sym->orbit(j, w.orbit, w.orbit_mark);
    long long added = 0;
    for (Index k : w.orbit) {
      if (k == j || w.ub[k] <= std::floor(v)) continue;
      PathSeg::encode({k, w.lb[k], std::floor(v)}, p_.col_lb[k], p_.col_ub[k], w.seg);
      ++added;
    }
    if (added > 0) {
      sym_fixed_ += added;
      ++sym_branches_;
    }
    if (w.orbit.size() > 1) {
      // The trivial group is always valid: drop symmetry under memory pressure.
      up.sym = nullptr;
      if (tree_bytes(w) < limit / 4) {
        const uint64_t seed = static_cast<uint64_t>(j) * 0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(node.depth) * 1315423911ULL +
                              node.sym->gens.size();
        auto g = std::make_shared<PermGroup>(node.sym->stabilizer(j, p_.num_cols(), seed, w.gscratch));
        ++sym_stabs_;
        if (!g->gens.empty()) {
          up.mem += g->bytes();
          up.sym = std::move(g);
        }
      }
    }
  }
  down.path = PathRef(PathSeg::make(base.get(), w.seg, &path_bytes_));
  for (const auto& c : w.scored) {
    if (c.j != j) continue;
    const double base_est = obj + w.est_sum - std::min(c.dn, c.up);
    down.estimate = base_est + c.dn;
    up.estimate = base_est + c.up;
    break;
  }
  if (w.sb_var == j) {
    // Opt-in (MILP_SB_BOUNDS=1): strong branching solved the children's LPs to optimality, so their
    // objectives are valid bounds (an infeasible / cutoff probe prunes the child without an LP).
    if (is_finite(w.sb_dn)) down.bound = std::max(down.bound, obj + w.sb_dn);
    if (is_finite(w.sb_up)) up.bound = std::max(up.bound, obj + w.sb_up);
  }
  down.branch_dir = -1;
  down.branch_dist = f;
  w.seg.clear();
  PathSeg::encode({j, std::ceil(v), cur_ub}, p_.col_lb[j], p_.col_ub[j], w.seg);
  up.path = PathRef(PathSeg::make(base.get(), w.seg, &path_bytes_));
  up.branch_dir = 1;
  up.branch_dist = 1.0 - f;
  if (!debug_sol_.empty()) {
    // Debug: a node containing the optimum must have a child containing it.
    auto in_path = [&](const Node& c, const PathSeg* stop) {
      Index bad = -1;
      for (const PathSeg* b = c.path.get(); b && b != stop; b = b->parent()) {
        b->for_each([&](const BoundChange& bc) {
          const double xd = debug_sol_[bc.col];
          if (bad < 0 && (xd < bc.lb - 1e-6 || xd > bc.ub + 1e-6)) bad = bc.col;
        });
      }
      return bad;
    };
    bool node_in = true;
    for (Index k = 0; k < p_.num_cols() && node_in; ++k) node_in = debug_sol_[k] >= w.lb[k] - 1e-6 && debug_sol_[k] <= w.ub[k] + 1e-6;
    const Index cd = in_path(down, node.path.get()), cu = in_path(up, node.path.get());
    const double dbg_obj = p_.objective(debug_sol_) - p_.offset;
    if (node_in && dbg_obj < cutoff(w) - 1e-9) {
      if (cd < 0 && down.bound > dbg_obj + 1e-9)
        log_.log(0, "DEBUG-SOL: node %lld down child bound %.12g > optimum %.12g (sb %d)\n", node.id, down.bound, dbg_obj, w.sb_var == j);
      if (cu < 0 && up.bound > dbg_obj + 1e-9)
        log_.log(0, "DEBUG-SOL: node %lld up child bound %.12g > optimum %.12g (sb %d)\n", node.id, up.bound, dbg_obj, w.sb_var == j);
    }
    if (node_in && cd >= 0 && cu >= 0) {
      log_.log(0, "DEBUG-SOL: node %lld branches on %d (x* = %g, v = %g): down excludes via col %d (x* = %g), up via col %d (x* = %g); rcfix %zu, orbit %d\n",
               node.id, j, debug_sol_[j], v, cd, debug_sol_[cd], cu, debug_sol_[cu], w.rcfix.size(), node.sym ? 1 : 0);
    }
  }
  // Continue with the child in the rounding direction; before any incumbent
  // exists, dive upwards (commits variables, finds feasible points faster).
  static const int child_rule = static_cast<int>(knob("MILP_CHILDSEL", 0));
  bool go_up = f >= 0.5;
  if (child_rule == 1 && !root_x_.empty()) {
    // Move away from the root LP value (the direction the LP is already drifting).
    if (v > root_x_[j] + 0.4) go_up = true;
    else if (v < root_x_[j] - 0.4) go_up = false;
  } else if (child_rule == 2) {
    // Smaller estimated objective degradation.
    double pd, pu;
    w.pc->unit_costs(j, &pd, &pu);
    go_up = pu * (1.0 - f) < pd * f;
  } else if (child_rule == 3) {
    // More propagation (inference history).
    double id, iu;
    w.pc->inference(j, &id, &iu);
    if (id != iu) go_up = iu > id;
  }
  if (go_up || !is_finite(incumbent(w))) {
    children[0] = std::move(up);
    children[1] = std::move(down);
  } else {
    children[0] = std::move(down);
    children[1] = std::move(up);
  }
  count_outcome(8, node.depth);
  return 2;
}

LpStatus Search::recover_lp(Worker& w, const LpOptions& opt, long long node_id) {
  DualSimplex& lp = w.lp;
  ++lp_failed_;
  w.last_node = -1;  // the LP state no longer belongs to this node's parent
  LpStatus st = LpStatus::NumericalError;
  if (debug_fail(node_id)) return st;
  for (int step = 0; step < 3; ++step) {
    if (step == 0) lp.set_basis(lp.get_basis());  // fresh factorization and DSE weights
    else lp.reset_basis();
    LpOptions retry = opt;
    retry.perturb = step == 2;
    retry.time_limit = params_.time_limit - timer_.elapsed();
    const long long it1 = lp.iterations();
    st = lp.solve(retry);
    lp_iters_ += lp.iterations() - it1;
    if (st == LpStatus::Optimal || st == LpStatus::Infeasible || st == LpStatus::CutoffReached) {
      ++lp_recovered_;
      return st;
    }
    if (st == LpStatus::TimeLimit) return st;
  }
  return st;
}

int Search::unsolved_branch(Worker& w, Node& node, Node* children) {
  ++lp_unsolved_;
  const Index n = p_.num_cols();
  // A fresh LP over the model rows only (cuts are redundant for validity), cold, no perturbation.
  DualSimplex fresh;
  fresh.load(p_.A, p_.cost, w.lb, w.ub, p_.row_lb, p_.row_ub);
  LpOptions o;
  o.tol = params_.tol;
  o.perturb = false;
  o.objective_cutoff = cutoff(w);
  o.time_limit = params_.time_limit - timer_.elapsed();
  const long long it0 = fresh.iterations();
  const LpStatus st = fresh.solve(o);
  lp_iters_ += fresh.iterations() - it0;
  if (st == LpStatus::Infeasible || st == LpStatus::CutoffReached) return 0;
  double bound = node.bound;
  Index j = -1;
  double v = 0.0;
  if (st == LpStatus::Optimal) {
    bound = std::max(bound, fresh.objective());
    if (bound >= cutoff(w)) return 0;
    const std::vector<double> x = fresh.col_values();
    double best = 0.0;
    for (Index k = 0; k < n; ++k) {
      if (!p_.is_int[k]) continue;
      const double f = x[k] - std::floor(x[k]);
      const double score = std::min(f, 1.0 - f);
      if (score > params_.tol.integrality && score > best) best = score, j = k, v = x[k];
    }
    if (j < 0) {
      std::vector<double> sol(x.begin(), x.end()), chk;
      double chk_obj;
      // Integral: the subtree's optimum is this point if it is feasible.
      if (try_incumbent(w, sol, "LP") || verified_point(sol, chk, &chk_obj)) return 0;
      if (!p_.implications.empty()) {
        const Index z = violated_implication(w, x.data());
        if (z >= 0) return implication_branch(w, node, children, z, bound);
      }
    }
  }
  if (j < 0) {
    // Split the largest domain of an unfixed integer column.
    double width = 0.0;
    for (Index k = 0; k < n; ++k) {
      if (!p_.is_int[k] || w.ub[k] - w.lb[k] < 0.5) continue;
      const double d = std::min(w.ub[k] - w.lb[k], 1e9);
      if (d > width) width = d, j = k;
    }
    if (j < 0) {
      // All integers fixed: the subtree is the LP over the continuous columns. Re-solve it from scratch
      // at a tightened primal tolerance (its point then passes the model check with margin), then at
      // the default one (whose infeasibility / cutoff verdict prunes like any node LP's).
      std::vector<double> pt;
      for (const double tol : {1e-9, params_.tol.primal_feas}) {
        const LpStatus s2 = solve_fixed_ints(w.lb, w.lb, w.ub, tol, cutoff(w), pt);
        std::vector<double> chk;
        double chk_obj;
        if (s2 == LpStatus::Optimal && p_.max_violation(pt) <= kFeasTol &&
            (try_incumbent(w, pt, "LP") || verified_point(pt, chk, &chk_obj))) {
          ++fixed_recovered_;
          return 0;
        }
        if (tol == params_.tol.primal_feas && (s2 == LpStatus::Infeasible || s2 == LpStatus::CutoffReached)) {
          ++fixed_recovered_;
          return 0;
        }
      }
      // The LP over the continuous columns is unsolvable: give the subtree up.
      numerical_ = true;
      std::lock_guard<std::mutex> lk(stop_mu_);
      lost_bound_ = std::min(lost_bound_, bound);
      log_.log(1, "  node %lld: LP failed with all integers fixed, dropped\n", node.id);
      return 0;
    }
    const double lo = std::max(w.lb[j], -1e9), hi = std::min(w.ub[j], lo + 2e9);
    v = std::floor(0.5 * (lo + hi)) + 0.5;
  }
  Node down, up;
  for (Node* c : {&down, &up}) {
    c->bound = bound;
    c->estimate = bound;
    c->depth = node.depth + 1;
    c->parent = node.id;
    c->id = next_node_id(w);
    c->branch_var = -1;  // no pseudocost update
    c->sym = nullptr;    // the trivial group is always valid
  }
  w.seg.clear();
  PathSeg::encode({j, w.lb[j], std::floor(v)}, p_.col_lb[j], p_.col_ub[j], w.seg);
  down.path = PathRef(PathSeg::make(node.path.get(), w.seg, &path_bytes_));
  w.seg.clear();
  PathSeg::encode({j, std::ceil(v), w.ub[j]}, p_.col_lb[j], p_.col_ub[j], w.seg);
  up.path = PathRef(PathSeg::make(node.path.get(), w.seg, &path_bytes_));
  children[0] = std::move(down);
  children[1] = std::move(up);
  return 2;
}

Index Search::violated_implication(const Worker& w, const double* x) const {
  for (const Implication& im : p_.implications) {
    if (w.lb[im.z] < w.ub[im.z] && std::abs(x[im.z] - im.zval) < 0.5 && std::abs(x[im.col]) > 1e-9) return im.z;
  }
  return -1;
}

Index Search::open_implication(const Worker& w) const {
  for (const Implication& im : p_.implications) {
    if (w.lb[im.z] < w.ub[im.z] && (w.lb[im.col] < 0.0 || w.ub[im.col] > 0.0)) return im.z;
  }
  return -1;
}

int Search::unbounded_leaf(Worker& w, Node& node) {
  MipProblem q = p_;
  std::fill(q.cost.begin(), q.cost.end(), 0.0);
  SubMipLimits lim;
  SubMipStats stats;
  lim.stats = &stats;
  lim.node_limit = 100000;
  lim.time_limit = std::max(0.0, params_.time_limit - timer_.elapsed());
  lim.min_fixed_fraction = 0.0;
  lim.stop = opts_.stop_flag;
  if (params_.deterministic) lim.iteration_limit = 10000000;
  std::vector<double> sol;
  if (solve_sub_mip(q, w.lb, w.ub, kInf, lim, sol) && p_.max_violation(sol) <= kFeasTol) {
    log_.log(2, "  node %lld: unbounded LP with a feasible point: the problem is unbounded\n", node.id);
    try_incumbent(w, sol, "unbounded");
    root_unbounded_ = true;
    stop_ = true;
    return 0;
  }
  if (stats.exhausted) return 0;  // infeasible subproblem
  // Undecided (limits): the subtree is lost.
  std::lock_guard<std::mutex> lk(stop_mu_);
  lost_bound_ = -kInf;
  numerical_ = true;
  log_.log(1, "  node %lld: unbounded LP, feasibility not decided; subtree dropped\n", node.id);
  return 0;
}

int Search::implication_branch(Worker& w, Node& node, Node* children, Index z, double bound) {
  ++impl_branches_;
  Node down, up;
  for (Node* c : {&down, &up}) {
    c->bound = bound;
    c->estimate = bound;
    c->parent_obj = bound;
    c->depth = node.depth + 1;
    c->parent = node.id;
    c->id = next_node_id(w);
    c->branch_var = -1;  // no pseudocost update
    c->sym = nullptr;
  }
  w.seg.clear();
  PathSeg::encode({z, w.lb[z], std::floor(w.lb[z])}, p_.col_lb[z], p_.col_ub[z], w.seg);
  down.path = PathRef(PathSeg::make(node.path.get(), w.seg, &path_bytes_));
  w.seg.clear();
  PathSeg::encode({z, std::floor(w.lb[z]) + 1.0, w.ub[z]}, p_.col_lb[z], p_.col_ub[z], w.seg);
  up.path = PathRef(PathSeg::make(node.path.get(), w.seg, &path_bytes_));
  children[0] = std::move(down);
  children[1] = std::move(up);
  count_outcome(8, node.depth);
  return 2;
}

void Search::run_sub_mips(Worker& w) {
  static const bool disabled = std::getenv("MILP_NO_SUBMIP") != nullptr;
  // Local branching is opt-in (MILP_LB=1): it never ran in the tree before the selection fix,
  // and once it did, its full-size sub-MIPs cost easy instances 10-40% (gap/cflp medium) while
  // it improved nothing on the MIPLIB targets.
  static const bool no_lb = std::getenv("MILP_LB") == nullptr || std::getenv("MILP_NO_LB") ||
                            std::getenv("MILP_NO_LOCALSEARCH");
  static const bool no_mutation = std::getenv("MILP_NO_MUTATION") != nullptr;
  static const bool no_crossover = std::getenv("MILP_NO_CROSSOVER") != nullptr;
  if (disabled || !opts_.sub_mip_heuristics) return;
  // Sub-MIPs may use a share of the search's LP iterations (plus a start
  // allowance), at least kGap nodes apart: up to 15% while they keep improving
  // the incumbent, down to 6% when recent calls fail. Opportunistic mode: one
  // sub-MIP at a time across workers (two when boosted) with shared statistics.
  // Deterministic mode: each worker on its own node count, iteration share and statistics, and
  // (with async_jobs_) the sub-MIP call becomes a job that runs beside the tree search
  // (run_deterministic).
  constexpr long long kAllowance = 20000, kGap = 100;
  LnsState& st = w.det ? w.lns : lns_;
  // Deterministic mode with jobs: its previous job has not entered yet, or no free job slot was
  // given to this worker for the round (deciding and then dropping skewed the neighbourhood choice).
  if (w.det && (w.lns.pending || (async_jobs_ && !w.lns.may_start))) return;
  const long long nodes = w.det ? w.nodes : nodes_.load();
  if (nodes - (w.det ? w.lns.last_node : lns_last_node_.load()) < kGap) return;
  const double searched = static_cast<double>(w.det ? w.iters_base / det_workers_ : lp_iters_.load());
  const double used = static_cast<double>(w.det ? st.iters : sub_iters_.load());
  std::unique_lock<std::mutex> lk(lns_mu_, std::defer_lock);
  if (!w.det) {
    lk.lock();
    // A second concurrent sub-MIP once the search is clearly primal-limited (more tries per second).
    static const bool no_concurrent = std::getenv("MILP_NO_LNS_CONCURRENT") != nullptr;
    const int max_active = !no_concurrent && lns_.boost >= 3 && threads_ >= 3 ? 2 : 1;
    if (sub_active_ >= max_active) return;
  }
  static const double share_mult = knob("MILP_LNS_SHARE", 1.0);
  static const bool no_boost = std::getenv("MILP_NO_LNS_BOOST") != nullptr;
  if (!no_boost) {
    // Bound progress over the last doubling of the node count, relative to the gap at its start:
    // below 2% the share grows (up to 6x, at most half the search's LP iterations), above 10%
    // it returns to normal.
    const double lb = w.det ? det_lb_ : pool_->lower_bound();
    const double inc = incumbent(w);
    if (st.ref_nodes == 0) {
      st.ref_nodes = std::max<long long>(nodes, 1);
      st.ref_bound = lb;
    } else if (nodes >= 2 * st.ref_nodes && nodes - st.ref_nodes >= 1000) {
      if (is_finite(inc) && is_finite(lb) && is_finite(st.ref_bound) && inc > st.ref_bound) {
        const double progress = (lb - st.ref_bound) / (inc - st.ref_bound);
        if (progress < 0.02) st.boost = std::min(st.boost + 1, 5);
        else if (progress > 0.1) st.boost = 0;
      }
      st.ref_nodes = nodes;
      st.ref_bound = lb;
    }
  }
  const double share = std::min(0.5, share_mult * (1.0 + st.boost) *
                       (0.06 + 0.09 * std::min(1.0, 2.0 * (st.recent_wins + 0.5) / (st.recent_calls + 1.0))));
  if (used > share * searched + kAllowance) return;
  if (w.det) w.lns.last_node = nodes; else lns_last_node_ = nodes;

  // Neighbourhoods available now: RINS, mutation, local branching and crossover need an
  // incumbent, RENS only the node LP point.
  const double inc_now = incumbent(w);
  const bool has_inc = is_finite(inc_now);
  bool avail[kLns] = {};
  avail[kLnsRins] = has_inc;
  avail[kLnsMutation] = has_inc && !no_mutation;
  // Local branching (a full-size sub-MIP) once the gap to the root bound is noticeable,
  // once per incumbent (it is a full-size sub-MIP: since the LNS selection fix it ran again and
  // again on an optimal incumbent with growing k, cflp_medium 6.2 -> 7.6 s).
  avail[kLnsLocalBranching] = has_inc && !no_lb && inc_now != st.lb_inc &&
                              inc_now - root_bound_ > 5e-3 * std::max(1.0, std::abs(inc_now + p_.offset));
  avail[kLnsRens] = true;
  // Crossover: the incumbent and cx_parents - 1 other pool solutions, when the pool changed
  // since the last call or offers other combinations.
  std::vector<std::vector<double>> pool;
  if (has_inc && !no_crossover) {
    long long version;
    {
      std::lock_guard<std::mutex> ilk(inc_mu_);
      version = sol_pool_version_;
      if (sol_pool_.size() >= 2) {
        for (const auto& e : sol_pool_) pool.push_back(e.second);
      }
    }
    avail[kLnsCrossover] = pool.size() >= 2 && (version != st.cx_version || st.cx_parents != st.cx_parents_used ||
                                                static_cast<int>(pool.size()) > st.cx_parents);
    st.cx_version = version;
  }
  double total = 0.0;
  for (int k = 0; k < kLns; ++k) total += st.calls[k];
  // Every other call is RINS while it succeeds about as often as the best other neighbourhood
  // (or has had few calls); otherwise all available neighbourhoods compete by UCB.
  static const bool rins_alternate = std::getenv("MILP_LNS_RINS_ALWAYS") != nullptr;
  double best_other = 0.0;
  for (int k = 0; k < kLns; ++k) {
    if (k != kLnsRins && st.calls[k] > 0.0) best_other = std::max(best_other, st.wins[k] / st.calls[k]);
  }
  const double rins_rate = st.calls[kLnsRins] > 0.0 ? st.wins[kLnsRins] / st.calls[kLnsRins] : 0.0;
  const bool rins_turn = avail[kLnsRins] && st.rounds++ % 2 == 0 &&
                         (rins_alternate || st.calls[kLnsRins] < 3.0 || (rins_rate > 0.0 && rins_rate >= 0.5 * best_other));
  int pick = rins_turn ? kLnsRins : -1;
  double best = -kInf;
  // The other calls pick by UCB (before, the loop stopped as soon as RINS, the first entry,
  // was the provisional best: mutation, local branching and RENS never ran).
  for (int k = 0; k < kLns && !rins_turn; ++k) {
    if (!avail[k] || (k == kLnsRins && avail[kLnsMutation] && (rins_alternate || st.calls[kLnsRins] < 3.0))) continue;
    const double score =
        (st.wins[k] + 0.5) / (st.calls[k] + 1.0) + 0.5 * std::sqrt(std::log(total + 1.0) / (st.calls[k] + 1.0));
    if (score > best) best = score, pick = k;
  }
  if (pick < 0) return;
  auto job = std::make_shared<SubMipJob>();
  job->pick = pick;
  job->origin = w.tid;
  job->det = w.det;
  job->inc_now = inc_now;
  job->has_inc = has_inc;
  if (has_inc) incumbent_x(w, job->inc);
  if (pick == kLnsRins || pick == kLnsRens) job->xnode = w.xnode;
  job->pool = std::move(pool);
  // Deterministic mode: no wall-clock cap on the sub-MIPs (a binding cap would make the result timing
  // dependent); at most the worker's whole sub-MIP budget in LP iterations instead.
  if (w.det) job->iteration_limit = kAllowance + static_cast<long long>(share * searched);
  if (w.det && async_jobs_) {
    job->st = st;  // the job adapts a copy; it replaces the worker's state when the job enters
    w.lns.pending = true;
    w.new_jobs.push_back(std::move(job));
    return;
  }
  if (!w.det) ++sub_active_;
  const auto t0 = std::chrono::steady_clock::now();
  run_lns(*job, st, &w, w.det ? nullptr : &lk);
  w.t_sub += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();  // statistics only
  ++w.sub_runs;
  if (!w.det) --sub_active_;
}

// The sub-MIP attempts of one LNS call (st: the neighbourhood statistics to adapt). With a worker,
// solutions go to it; without (a deterministic-mode job), they are kept in job.sols and count as
// improvements (they beat the incumbent the job started from). lk: opportunistic mode's lns_mu_.
void Search::run_lns(SubMipJob& job, LnsState& st, Worker* w, std::unique_lock<std::mutex>* lk) {
  static const bool rins_static = std::getenv("MILP_RINS_STATIC") != nullptr;
  // A neighbourhood that turns out too small (sub-MIP exhausted with almost no LP work, e.g.
  // infeasible after presolve) is retried at once with its adapted size, up to 4 times.
  static const bool no_retry = std::getenv("MILP_NO_LNS_RETRY") != nullptr;
  const int pick = job.pick;
  const std::vector<std::vector<double>>& pool = job.pool;
  for (int attempt = 0; attempt < 4; ++attempt) {
    // Neighbourhood parameters now; the sub-MIP itself runs without the lock.
    const double rins_fix_used = st.rins_fix, mut_fix = st.mut_fix;
    const int lb_k = st.lb_k, cx_parents = st.cx_parents;
    const unsigned seq = st.seq++;
    if (pick == kLnsLocalBranching) st.lb_inc = job.inc_now;
    if (pick == kLnsCrossover) st.cx_parents_used = cx_parents;
    if (lk) lk->unlock();
    SubMipStats ss;
    SubMipLimits lim;
    const double rem = std::max(0.0, params_.time_limit - timer_.elapsed());
    lim.time_limit = job.det ? rem : std::min(10.0, 0.1 * rem);
    lim.node_limit = 500;
    lim.iteration_limit = job.iteration_limit;
    if (!job.det) lim.mem_limit_mb = sub_mip_mem_mb(nullptr);  // deterministic jobs keep the fixed cap
    lim.stats = &ss;
    lim.stop = &job.stop;
    lim.progress = &job.progress;
    std::vector<double> sol;
    std::vector<double> inc = job.inc;
    if (w && !w->det && job.has_inc) incumbent_x(*w, inc);  // the incumbent may have improved meanwhile
    const double cut = job.inc_now + p_.offset;
    bool found = false;
    const char* source = nullptr;
    switch (pick) {
      case kLnsRins:
        found = rins(p_, job.xnode.data(), inc, cut, lim, sol, rins_fix_used);
        source = "RINS";
        break;
      case kLnsMutation:
        found = mutation(p_, inc, mut_fix, params_.seed * 7919u + seq, cut, lim, sol);
        source = "mutation";
        break;
      case kLnsLocalBranching:
        if (!job.det) lim.time_limit = std::min(5.0, 0.05 * rem);
        lim.node_limit = 300;
        found = local_branching(p_, inc, lb_k, cut, lim, sol);
        source = "local branching";
        break;
      case kLnsCrossover: {
        // Parents: the incumbent plus others drawn from the pool (all of it when it is small).
        std::vector<const std::vector<double>*> parents{&inc};
        std::vector<size_t> others;
        for (size_t k = 0; k < pool.size(); ++k) {
          if (pool[k] != inc) others.push_back(k);
        }
        std::mt19937 rng(params_.seed * 7919u + 31337u + seq);
        std::shuffle(others.begin(), others.end(), rng);
        for (size_t k = 0; k < others.size() && static_cast<int>(parents.size()) < cx_parents; ++k) {
          parents.push_back(&pool[others[k]]);
        }
        lim.min_fixed_fraction = 0.3;
        found = crossover(p_, parents, cut, lim, sol);
        source = "crossover";
        break;
      }
      default:
        found = rens(p_, job.xnode.data(), cut, lim, sol);
        source = "RENS";
        break;
    }
    bool win = found;
    if (found && w) {
      win = try_incumbent(*w, sol, source);
    } else if (found) {
      job.sols.emplace_back(std::move(sol), source);
    }
    if (lk) lk->lock();
    if (ss.solves > 0) {
      for (int k = 0; k < kLns; ++k) st.calls[k] *= 0.97, st.wins[k] *= 0.97;  // forget old successes slowly
      st.calls[pick] += 1.0;
      st.wins[pick] += win;
      st.recent_calls = 0.9 * st.recent_calls + 1.0;
      st.recent_wins = 0.9 * st.recent_wins + win;
    } else {
      st.calls[pick] += 0.25;  // neighbourhood rejected (too few fixings): cheap, but do not retry at once
    }
    // Adapt the neighbourhood sizes: exhausted without improvement -> larger, limit hit -> smaller.
    if (!win && ss.solves > 0) {
      if (pick == kLnsMutation) {
        // Finer steps down: the random fixings decide as much as the rate whether the
        // neighbourhood is empty (fiball: 0.6 infeasible or improving depending on the seed).
        st.mut_fix = ss.exhausted ? std::max(0.2, mut_fix - 0.05) : std::min(0.95, mut_fix + 0.05);
      }
      if (pick == kLnsRins && !rins_static) {
        // Too large (limit hit): fix more next time, at least 10% above this call's rate.
        const double rate = ss.ints > 0 ? static_cast<double>(ss.fixed) / ss.ints : 0.0;
        st.rins_fix = ss.exhausted ? std::max(0.0, st.rins_fix - 0.1) : std::min(0.9, std::max(st.rins_fix, rate) + 0.1);
      }
      if (pick == kLnsLocalBranching) st.lb_k = ss.exhausted ? std::min(lb_k * 2, 80) : std::max(lb_k / 2, 5);
      // More parents agree on fewer variables: a larger neighbourhood.
      if (pick == kLnsCrossover) st.cx_parents = ss.exhausted ? std::min(cx_parents + 1, 6) : std::max(cx_parents - 1, 2);
    }
    if (pick == kLnsCrossover && ss.solves == 0) st.cx_parents = std::max(cx_parents - 1, 2);  // too few fixed
    if (std::getenv("MILP_LNS_DEBUG")) {
      log_.log(1, "  LNS %s: %s, %d solves, fixed %d/%d, %lld iters, %s, boost %d, %.2fs\n", source,
               win ? "improved" : "no", ss.solves, ss.fixed, ss.ints, ss.lp_iterations,
               ss.exhausted ? "exhausted" : "limit", st.boost, timer_.elapsed());
    }
    st.iters += ss.lp_iterations;
    if (w && !w->det) sub_iters_ += ss.lp_iterations;
    const bool cheap_fail = !win && ss.solves > 0 && ss.exhausted && ss.lp_iterations < 1000;
    if (no_retry || !cheap_fail || pick == kLnsRens || pick == kLnsLocalBranching ||
        (pick == kLnsCrossover && st.cx_parents == cx_parents) || (pick == kLnsRins && st.rins_fix == rins_fix_used) ||
        job.stop.load()) {
      break;
    }
  }
}

thread_local int Search::last_outcome_ = -1;

int Search::process(Worker& w, Node& node, Node* children) {
  last_outcome_ = -1;
  const int nc = process_impl(w, node, children);
  static const bool cut_hist = knob("MILP_CUT_WEIGHT", 0.0) > 0.0;
  if (cut_hist && node.branch_var >= 0 && !stop_) {
    // Cutoff history (SCIP's cutoff score): was this child of a branching pruned? (A feasible LP
    // point closes it too; counted as not cut off.)
    const bool cut = nc == 0 && last_outcome_ != 7;
    w.pc->update_cutoff(node.branch_var, node.branch_dir, cut);
    if (w.det) w.cut_log.emplace_back(node.branch_var, node.branch_dir, cut);
  }
  if (debug_sol_.empty()) return nc;
  // Debug: report when a node whose bounds contain the known optimum is closed.
  if (nc == 0) {
    bool contains = true;
    for (Index j = 0; j < p_.num_cols() && contains; ++j) {
      contains = debug_sol_[j] >= w.lp.col_lower(j) - 1e-6 && debug_sol_[j] <= w.lp.col_upper(j) + 1e-6;
    }
    if (contains) {
      log_.log(0, "DEBUG-SOL: node %lld depth %d containing the optimum closed; LP obj %.10g cutoff %.10g inc %.10g\n",
               node.id, node.depth, w.lp.objective(), cutoff_.load(), incumbent_.load());
    }
  }
  return nc;
}

void Search::worker_loop(int tid) {
  Worker w;
  w.lp = root_lp_;
  w.m_base = root_lp_.num_rows();
  w.lp.set_seed(params_.seed * 131u + static_cast<unsigned>(tid) + 1u);
  w.pc = &pc_;
  w.tid = tid;
  Node cur;
  Node children[2];
  bool have = false;
  int plunge_depth = 0;
  long long pops = 0, restart_poll = 0;
  // Node selection: 0 best bound; k > 0 best estimate with every k-th selection by best bound.
  // Default 2: the (hybrid) estimate order and best bound alternate. Pure best bound found the
  // improving solutions only by luck on primal-limited instances (timtab1: optimum at 9-116 s by
  // seed, 2/6 seeds unsolved at 120 s; alternating: optimum at 5-10 s, 6/6 solved in 9-38 s); mostly
  // pure estimate (k = 10) slowed eil33-2 (56-120 s vs 53-80 s).
  static const int best_freq = static_cast<int>(knob("MILP_NODESEL", 2));
  while (!stop_.load()) {
    if (!have) {
      const bool by_est = best_freq > 0 && ++pops % best_freq != 0;
      if (!pool_->pop(tid, cutoff_, stop_, &cur, by_est)) break;
      have = true;
    }
    if (cur.bound >= cutoff_.load()) {
      have = false;
      pool_->done(tid);
      continue;
    }
    pool_->set_worker_bound(tid, cur.bound);
    const int nc = process(w, cur, children);
    // Plunge control (as SCIP): keep diving into the preferred child only while
    // its bound stays within 25% of the gap above the global bound; otherwise
    // return to best-first selection. Always plunge a couple of levels.
    // Opt-in (MILP_PLUNGECTL=1): neutral on the dev set (helps tsp, costs lotsize/cflp).
    static const bool no_plunge_ctl = std::getenv("MILP_PLUNGECTL") == nullptr;
    bool plunge = nc == 2;
    // Drain mode (memory): always plunge.
    if (plunge && !no_plunge_ctl && plunge_depth >= 2 && mem_state_.load() < 2) {
      const double inc = incumbent_.load();
      if (is_finite(inc)) {
        const double lb = pool_->lower_bound();
        static const double q = knob("MILP_PLUNGE_Q", 0.25);
        if (is_finite(lb) && children[0].bound > lb + q * (inc - lb) + 1e-9) plunge = false;
      }
    }
    if (plunge) {
      pool_->push(std::move(children[1]));
      cur = std::move(children[0]);
      pool_->set_worker_bound(tid, cur.bound);
      ++plunge_depth;
    } else if (nc == 2) {
      pool_->push(std::move(children[0]));
      pool_->push(std::move(children[1]));
      have = false;
      plunge_depth = 0;
      pool_->done(tid);
    } else {
      plunge_depth = 0;
      have = false;
      pool_->done(tid);
    }
    if (params_.node_limit >= 0 && nodes_.load() >= params_.node_limit) {
      std::lock_guard<std::mutex> lk(stop_mu_);
      if (stop_reason_ == Status::Unknown) stop_reason_ = Status::NodeLimit;
      stop_ = true;
    }
    if (opts_.iteration_limit >= 0 && lp_iters_.load() >= opts_.iteration_limit) {
      std::lock_guard<std::mutex> lk(stop_mu_);
      if (stop_reason_ == Status::Unknown) stop_reason_ = Status::IterationLimit;
      stop_ = true;
    }
    memory_control(false);
    if (memory_exceeded(false)) {
      std::lock_guard<std::mutex> lk(stop_mu_);
      if (stop_reason_ == Status::Unknown) {
        stop_reason_ = Status::MemoryLimit;
        log_.log(1, "Memory limit reached (%.0f MB of tree memory, %.0f MB resident)\n", tree_bytes() / 1048576.0,
                 rss_bytes_.load() / 1048576.0);
      }
      stop_ = true;
    }
    if (timer_.elapsed() > params_.time_limit) {
      std::lock_guard<std::mutex> lk(stop_mu_);
      if (stop_reason_ == Status::Unknown) stop_reason_ = Status::TimeLimit;
      stop_ = true;
    }
    report_progress();
    poll_race();
    if (tid == 0 && (++restart_poll & 127) == 0) check_tree_restart(false);
    if ((nodes_.load() & 15) == 0) {
      const double lb = pool_->lower_bound();
      if (gap_closed(lb)) {
        if (!debug_sol_.empty()) log_.log(0, "DEBUG-SOL: gap closed lb %.10g inc %.10g\n", lb, incumbent_.load());
        stop_ = true;
      }
    }
    maybe_log(false);
  }
  if (have) {
    // Keep the bound of an unfinished node visible for the final report.
    pool_->push(std::move(cur));
    pool_->done(tid);
  }
  prop_work_total_ += w.prop_work;
  lp_work_total_ += w.lp_work;
}

void Search::run_deterministic() {
  constexpr int kMaxNodes = 256;  // nodes a virtual worker may process per round
  const int T = threads_;
  // Virtual workers: the round's work is split into V = vw * T independent tasks (each with its own
  // LP, node batch, budget and snapshot of the shared state) that the T threads take in any order;
  // results are merged in task order. The outcome depends on V but not on which thread ran which
  // task or when, and a task that runs long (a slow node) no longer stalls the round: the other
  // threads take the remaining tasks meanwhile.
  static const int vw = std::max(1, static_cast<int>(knob("MILP_DET_VW", 2)));
  const int V = T * vw;
  det_workers_ = V;
  // Nodes handed to each virtual worker per round (best-bound order, dealt round-robin). Within a
  // round a worker plunges like the opportunistic workers and, when a plunge ends, continues with the
  // best node of its own batch (the open children it created included). MILP_DET_BATCH=0: old scheme
  // (one node per worker, depth-first over its subtree).
  static const int batch = static_cast<int>(knob("MILP_DET_BATCH", 8));
  // Round length: about this many average nodes of work per thread (split over its virtual workers).
  static const double round_nodes = knob("MILP_DET_ROUND_NODES", 32);
  // Sub-MIP jobs. A sub-MIP decided by a worker during a round is published at the round barrier and
  // run by a helper thread beside the following rounds (at most T - 1 jobs at a time; the main
  // thread never runs one). Its result enters at the first barrier at which the tree search has
  // done job_factor times the job's LP iterations per thread left for it (each round adds the
  // round's LP iterations / (T - pending jobs) to the job's credit): a deterministic point at
  // about the time the job finishes. The barrier waits for a job only while it may still finish
  // below its credit (it has used fewer iterations so far).
  static const bool no_jobs = std::getenv("MILP_DET_NO_JOBS") != nullptr;
  static const double job_factor = knob("MILP_DET_JOB_FACTOR", 1.0);
  async_jobs_ = !no_jobs && T > 1;
  const int max_jobs = T - 1;
  long long round_budget = 2000 * T / V;  // work units (LP iterations) per virtual worker per round
  std::vector<std::unique_ptr<Worker>> ws;
  for (int t = 0; t < V; ++t) {
    auto w = std::make_unique<Worker>();
    w->lp = root_lp_;
    w->m_base = root_lp_.num_rows();
    w->lp.set_seed(params_.seed * 131u + static_cast<unsigned>(t) + 1u);
    w->det = true;
    w->tid = t;
    w->own_pc = std::make_unique<Pseudocosts>(p_.num_cols());
    w->pc = w->own_pc.get();
    ws.push_back(std::move(w));
  }
  std::vector<std::vector<Node>> assigned(V);
  std::vector<char> has(V, 0);
  // A plunge interrupted by the end of a round continues in the same worker in the next round (a
  // new subtree per round cost ~20% more LP iterations per node on cflp_large).
  static const bool carry_plunge = std::getenv("MILP_DET_NO_CARRY") == nullptr;
  std::vector<Node> carry(V);
  std::vector<char> carried(V, 0);
  // Memory limit: open nodes a worker may add this round (net: children queued minus nodes it
  // closed). Memory decisions are taken only at the barriers, so without a cap one round of V
  // workers (up to kMaxNodes new nodes each) could carry the tree past the drain threshold and on
  // past the stop threshold before drain mode took effect. The caps are projected at the round
  // start (project / set_caps below); far from the limit they do not bind.
  constexpr long long kNoCap = std::numeric_limits<long long>::max();
  std::vector<long long> grow_cap(V, kNoCap);
  bool drain_round = false;  // the pool was in drain mode at the round start
  pool_->resize_workers(V);
  // Instrumentation (wall clock, statistics only: never used for decisions).
  using Clock = std::chrono::steady_clock;
  auto secs = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
  std::vector<double> busy(V, 0.0), tbusy(T, 0.0);
  std::vector<long long> wnodes(V, 0);
  double par_wall = 0.0, serial = 0.0, fj_wait = 0.0, job_wait = 0.0, job_secs = 0.0;
  static const bool trace = std::getenv("MILP_DET_TRACE") != nullptr;
  std::vector<double> tr_busy(V), tr_sub(V);
  std::vector<long long> tr_nodes(V), tr_iters(V);
  long long rounds = 0, idle_slots = 0, jobs_run = 0, jobs_dropped = 0, jobs_discarded = 0;

  // Heap order of a worker's batch: best bound first, deeper first on ties, then by id; in drain
  // mode deepest first (as the pool), then best bound.
  auto worse = [&drain_round](const Node& a, const Node& b) {
    if (drain_round && a.depth != b.depth) return a.depth < b.depth;
    if (a.bound != b.bound) return a.bound > b.bound;
    if (a.depth != b.depth) return a.depth < b.depth;
    return a.id > b.id;
  };
  // Deterministic work clock of a worker: its LP iterations (nodes, strong branching, dives) plus
  // those of its synchronous sub-MIPs (MILP_DET_NO_JOBS).
  auto clock_of = [](const Worker& w) { return w.lp.iterations() + w.lns.iters; };
  // Memory projection of a round (between rounds, from deterministic data only: the tree memory and
  // the open nodes at the round start). The round may add about as many open nodes (grow) as fit
  // between the tree memory now and a target, at 1.5x the average bytes per open node (bases, bound
  // paths, queue arrays: an array that doubles counts its capacity). The target is 78% of the limit
  // outside drain mode (just past the drain threshold of 75%: drain mode starts about where it does
  // in opportunistic mode, not after a round overshot it) and 90% in drain mode (the search stops
  // at 95%). Where that binds, the growth goes to few workers, each with room for a plunge of
  // kMinGrow nodes (a thin share per worker cut every plunge short: the tree then grew by one node
  // per worker and round and never drained), and the others sit the round out. Sets grow (-1: not
  // binding) and returns the number of workers to hand nodes to.
  static const long long kMinGrow = static_cast<long long>(knob("MILP_DET_MEM_GROW", 64));
  long long capped_rounds = 0, grow = -1;
  auto project = [&]() -> int {
    drain_round = pool_->drain();
    grow = -1;
    const long long limit = mem_limit_bytes();
    if (limit <= 0) return V;
    long long open = static_cast<long long>(pool_->size());
    for (int t = 0; t < V; ++t) open += carried[t];
    const long long tb = tree_bytes();
    const long long per_node =
        std::max<long long>(static_cast<long long>(sizeof(Node)) + 64, tb / std::max<long long>(1, open)) * 3 / 2;
    const long long target = drain_round ? limit / 20 * 18 : limit / 100 * 78;
    const long long g = std::max<long long>(0, (target - tb) / per_node);
    if (g >= static_cast<long long>(V) * kMaxNodes) return V;  // cannot bind this round
    ++capped_rounds;
    grow = g;
    return static_cast<int>(std::clamp<long long>(g / kMinGrow, 1, V));
  };
  // Splits grow over the workers that have nodes this round.
  auto set_caps = [&](int active) {
    std::fill(grow_cap.begin(), grow_cap.end(), kNoCap);
    if (grow < 0) return;
    int k = 0;
    for (int t = 0; t < V; ++t) {
      if (has[t]) grow_cap[t] = grow / active + (k++ < grow % active ? 1 : 0);
    }
  };
  auto work = [&](int t) {
    const Clock::time_point t0 = Clock::now();
    Worker& w = *ws[t];
    // Rest of the snapshot of the shared state (not written during the round), taken in parallel.
    {
      std::lock_guard<std::mutex> lk(inc_mu_);
      if (w.inc_version != inc_version_) {  // copy the incumbent only when it changed
        w.inc_x = inc_x_;
        w.inc_version = inc_version_;
      }
    }
    w.pc->copy_from(pc_);
    Node children[2];
    std::vector<Node>& local = assigned[t];
    const long long it0 = clock_of(w);
    const double sub0 = w.t_sub;
    int processed = 0;
    // Net open nodes this worker added (its batch, carried plunge and children against the start).
    const long long open0 = static_cast<long long>(local.size()) + carried[t];
    auto more = [&](long long open_now) {
      return processed == 0 ||
             (processed < kMaxNodes && clock_of(w) - it0 < round_budget && open_now - open0 < grow_cap[t]);
    };
    if (batch <= 0) {
      // Old scheme: depth-first over the subtree of the single assigned node.
      while (!local.empty() && !stop_.load() && more(static_cast<long long>(local.size()))) {
        Node cur = std::move(local.back());
        local.pop_back();
        if (cur.bound >= w.cutoff) continue;
        ++processed;
        if (process(w, cur, children) == 2) {
          local.push_back(std::move(children[1]));
          local.push_back(std::move(children[0]));
        }
      }
    } else {
      std::make_heap(local.begin(), local.end(), worse);
      // Continue the plunge of the previous round first (its LP is still at the parent).
      Node cur;
      bool have = false;
      if (carried[t]) {
        cur = std::move(carry[t]);
        carried[t] = 0;
        have = true;
      }
      while (!stop_.load() && more(static_cast<long long>(local.size()) + have)) {
        if (!have) {
          if (local.empty()) break;
          std::pop_heap(local.begin(), local.end(), worse);
          cur = std::move(local.back());
          local.pop_back();
          have = true;
        }
        if (cur.bound >= w.cutoff) {
          have = false;
          continue;
        }
        ++processed;
        if (process(w, cur, children) == 2) {
          local.push_back(std::move(children[1]));
          std::push_heap(local.begin(), local.end(), worse);
          cur = std::move(children[0]);
        } else {
          have = false;
        }
      }
      if (have && carry_plunge && cur.bound < w.cutoff) {
        carry[t] = std::move(cur);  // kept by this worker; its bound stays visible in the pool
        carried[t] = 1;
      } else if (have) {
        local.push_back(std::move(cur));
      }
    }
    for (Node& n : local) {
      if (n.bound < w.cutoff) w.pushed.push_back(std::move(n));
    }
    local.clear();
    wnodes[t] += processed;
    const double el = secs(t0, Clock::now());
    busy[t] += el;
    if (trace) {
      tr_busy[t] = el;
      tr_nodes[t] = processed;
      tr_iters[t] = clock_of(w) - it0;
      tr_sub[t] = w.t_sub - sub0;
    }
    return el;
  };

  // Thread pool. Each round the threads take the virtual workers in index order; helper threads
  // take queued sub-MIP jobs first. Everything below is guarded by mu.
  std::mutex mu;
  std::condition_variable cv;
  int next_task = V, tasks_done = V;
  bool quit = false;
  std::deque<std::shared_ptr<SubMipJob>> job_queue;  // published, not started
  std::vector<std::shared_ptr<SubMipJob>> jobs;      // published, not entered (publication order)
  auto run_job = [&](SubMipJob& job) {
    const Clock::time_point t0 = Clock::now();
    run_lns(job, job.st, nullptr, nullptr);
    job.seconds = secs(t0, Clock::now());
  };
  std::vector<std::thread> threads;
  for (int th = 1; th < T; ++th) {
    threads.emplace_back([&, th] {
      std::unique_lock<std::mutex> lk(mu);
      for (;;) {
        cv.wait(lk, [&] { return quit || !job_queue.empty() || next_task < V; });
        if (quit) return;
        if (!job_queue.empty()) {
          std::shared_ptr<SubMipJob> job = std::move(job_queue.front());
          job_queue.pop_front();
          job->started = true;
          lk.unlock();
          run_job(*job);
          lk.lock();
          tbusy[th] += job->seconds;
          job->done = true;
          cv.notify_all();
          continue;
        }
        const int v = next_task++;
        lk.unlock();
        const double el = has[v] ? work(v) : 0.0;
        lk.lock();
        tbusy[th] += el;
        if (++tasks_done == V) cv.notify_all();
      }
    });
  }
  // Tree clock: LP iterations of all workers (deterministic at a barrier).
  const long long iters0 = root_lp_.iterations();  // the workers' LPs start with the root's count
  auto tree_clock = [&] {
    long long c = 0;
    for (const auto& w : ws) c += w->lp.iterations() - iters0;
    return c;
  };
  // Average LP iterations per tree node so far (the root's and its heuristics' work excluded: it
  // made the first rounds far too long when the root was expensive).
  auto tree_node_cost = [&] {
    long long n = 0;
    for (const auto& w : ws) n += w->nodes;
    return n < 2 * V ? 100.0 : std::max(1.0, static_cast<double>(tree_clock()) / static_cast<double>(n));
  };
  long long job_ids = 0, last_clock = tree_clock();
  // FJ effort is released for the tree's LP work only: releasing the root heuristics' work in the
  // first round made the search wait ~0.5 s for FJ there (cflp_medium).
  fj_iters_mark_ = lp_iters_.load();

  while (!stop_.load()) {
    Clock::time_point ts = Clock::now();
    // Hand out nodes in deterministic (best-bound) order with state snapshots.
    // Batch: at least `batch` nodes, and about twice the nodes a task's budget pays for (a task that
    // runs out of nodes ends early and leaves its thread idle at the barrier).
    // Under memory pressure only the first `use` workers get nodes; the others give their plunge back.
    int assigned_count = 0;
    const int use = project();
    for (int t = use; t < V; ++t) {
      if (carried[t]) pool_->push(std::move(carry[t]));
      carried[t] = 0;
    }
    const double node_cost = tree_node_cost();
    const int want = std::clamp(static_cast<int>(2.0 * static_cast<double>(round_budget) / node_cost), batch, 256);
    const int per = batch <= 0 ? 1 : std::max<int>(1, std::min<int>(want, static_cast<int>(pool_->size()) / use));
    for (int k = 0; k < per; ++k) {
      for (int t = 0; t < use; ++t) {
        Node n;
        if (!pool_->try_pop(cutoff_.load(), &n, false)) break;
        assigned[t].push_back(std::move(n));
      }
    }
    for (int t = 0; t < V; ++t) {
      if (carried[t] && carry[t].bound >= cutoff_.load()) carried[t] = 0;  // pruned meanwhile
      pool_->set_worker_bound(t, kInf);
      has[t] = !assigned[t].empty() || carried[t];
      assigned_count += has[t];
      idle_slots += !has[t];
    }
    if (assigned_count == 0) break;
    set_caps(assigned_count);
    const int free_slots = std::max(0, max_jobs - static_cast<int>(jobs.size()));
    for (int t = 0; t < V; ++t) {
      if (!has[t]) continue;
      Worker& w = *ws[t];
      w.cutoff = cutoff_.load();
      w.incumbent = incumbent_.load();
      w.iters_base = lp_iters_.load();
      w.dive_base = dive_iters_.load();
      w.dive_iters = 0;
      // Free job slots go to the workers in rotating order.
      w.lns.may_start = (t - rounds % V + V) % V < free_slots;
      w.found.clear();
      w.pc_log.clear();
      w.pushed.clear();
    }
    ++rounds;
    const Clock::time_point tp = Clock::now();
    serial += secs(ts, tp);
    {
      std::lock_guard<std::mutex> lk(mu);
      next_task = 0;
      tasks_done = 0;
    }
    cv.notify_all();
    {
      // The main thread takes node tasks only (never a job: it runs the barrier).
      std::unique_lock<std::mutex> lk(mu);
      while (next_task < V) {
        const int v = next_task++;
        lk.unlock();
        const double el = has[v] ? work(v) : 0.0;
        lk.lock();
        tbusy[0] += el;
        ++tasks_done;
      }
      cv.wait(lk, [&] { return tasks_done == V; });
    }
    ts = Clock::now();
    par_wall += secs(tp, ts);
    if (trace) {
      std::fprintf(stderr, "R %lld budget %lld wall %.4f", rounds, round_budget, secs(tp, ts));
      for (int t = 0; t < V; ++t) {
        std::fprintf(stderr, " | %.4f n%lld it%lld sub%.4f", tr_busy[t], tr_nodes[t], tr_iters[t], tr_sub[t]);
        tr_busy[t] = 0.0, tr_nodes[t] = 0, tr_iters[t] = 0, tr_sub[t] = 0.0;
      }
      std::fprintf(stderr, " | jobs %zu\n", jobs.size());
    }
    // Merge results in worker order.
    for (int t = 0; t < V; ++t) {
      Worker& w = *ws[t];
      if (!has[t]) continue;
      for (auto& f : w.found) try_incumbent(f.x, f.source);
      for (auto& [j, dir, g] : w.pc_log) pc_.update(j, dir, g);
      for (auto& [j, dir, g] : w.inf_log) pc_.update_inference(j, dir, g);
      w.inf_log.clear();
      for (auto& [j, dir, c] : w.cut_log) pc_.update_cutoff(j, dir, c);
      w.cut_log.clear();
      for (const auto& cols : w.conf_bumps) pc_.bump_conflict(cols);
      w.conf_bumps.clear();
      for (Node& n : w.pushed) pool_->push(std::move(n));
      if (use_pool_ && !w.pool_used.empty()) {
        std::lock_guard<std::mutex> lk(ncp_mu_);
        pool_flush_used(w);
      }
      if (!w.new_cuts.empty() && use_pool_) {
        pool_publish(w.new_cuts, nullptr);
        w.new_cuts.clear();
      } else if (!w.new_cuts.empty()) {
        std::lock_guard<std::mutex> lk(gcut_mu_);
        for (Cut& c : w.new_cuts) gcuts_.push_back(std::move(c));
        w.new_cuts.clear();
      }
      for (const auto& c : w.new_conflicts) publish_conflict(c);
      w.new_conflicts.clear();
      w.found.clear();
      w.pushed.clear();
      if (carried[t]) pool_->set_worker_bound(t, carry[t].bound);
    }
    // Sub-MIP jobs: results that are due enter (in publication order), then the new jobs are
    // published (in worker order; beyond max_jobs pending ones they are dropped).
    if (async_jobs_) {
      const long long clock = tree_clock();
      const double share = static_cast<double>(clock - last_clock) / std::max<int>(1, T - static_cast<int>(jobs.size()));
      last_clock = clock;
      const Clock::time_point tw = Clock::now();
      std::vector<std::shared_ptr<SubMipJob>> keep;
      for (auto& job : jobs) {
        job->credit += share;
        const long long due = static_cast<long long>(job->credit / job_factor);
        // The job enters now iff its final LP iterations are <= due. Its progress only grows, so
        // progress > due settles "not now" without waiting for the end.
        bool enter;
        {
          std::unique_lock<std::mutex> lk(mu);
          while (!job->done.load() && job->progress.load() <= due) cv.wait_for(lk, std::chrono::milliseconds(1));
          enter = job->done.load() && job->progress.load() <= due;
        }
        if (!enter) {
          keep.push_back(std::move(job));
          continue;
        }
        Worker& w = *ws[job->origin];
        for (auto& [x, src] : job->sols) try_incumbent(x, src);
        const bool may_start = w.lns.may_start;
        w.lns = job->st;  // the statistics and neighbourhood sizes the job adapted
        w.lns.may_start = may_start;
        w.lns.pending = false;
        ++w.sub_runs;
        w.t_sub += job->seconds;
        job_secs += job->seconds;
        ++jobs_run;
      }
      jobs.swap(keep);
      job_wait += secs(tw, Clock::now());
      for (int t = 0; t < V; ++t) {
        Worker& w = *ws[t];
        for (auto& job : w.new_jobs) {
          if (static_cast<int>(jobs.size()) >= max_jobs) {
            w.lns.pending = false;  // dropped: the worker decides again after its next kGap nodes
            ++jobs_dropped;
            continue;
          }
          job->id = job_ids++;
          jobs.push_back(job);
          std::lock_guard<std::mutex> lk(mu);
          job_queue.push_back(std::move(job));
        }
        w.new_jobs.clear();
      }
      cv.notify_all();
    }
    // FJ: take its solutions found up to the previous sync (it keeps running meanwhile); release
    // effort in proportion to this round's LP work.
    {
      const Clock::time_point tf = Clock::now();
      static const double fj_rate = knob("MILP_FJ_RATE", 1.0);
      fj_sync(static_cast<long long>(fj_rate * static_cast<double>(lp_iters_.load() - fj_iters_mark_) * p_.num_rows()),
              true);
      fj_wait += secs(tf, Clock::now());
    }
    det_lb_ = pool_->lower_bound();
    fj_iters_mark_ = lp_iters_.load();
    if (params_.node_limit >= 0 && nodes_.load() >= params_.node_limit) {
      if (stop_reason_ == Status::Unknown) stop_reason_ = Status::NodeLimit;
      stop_ = true;
    }
    memory_control(true);
    if (memory_exceeded(true)) {
      if (stop_reason_ == Status::Unknown) {
        stop_reason_ = Status::MemoryLimit;
        log_.log(1, "Memory limit reached (%.0f MB of tree memory, %.0f MB resident)\n", tree_bytes() / 1048576.0,
                 rss_bytes_.load() / 1048576.0);
      }
      stop_ = true;
    }
    if (timer_.elapsed() > params_.time_limit) {
      if (stop_reason_ == Status::Unknown) stop_reason_ = Status::TimeLimit;
      stop_ = true;
    }
    if (gap_closed(pool_->lower_bound())) stop_ = true;
    check_tree_restart(true);
    poll_race();
    // ~round_nodes average nodes of work per thread per round, split over its virtual workers.
    round_budget = std::clamp<long long>(static_cast<long long>(round_nodes * T / V * tree_node_cost()), 200 * T / V,
                                         50000);
    maybe_log(false);
    serial += secs(ts, Clock::now());
  }
  for (int t = 0; t < V; ++t) {
    pool_->set_worker_bound(t, kInf);
    if (carried[t]) pool_->push(std::move(carry[t]));  // unfinished: keeps the final bound right
  }
  // Jobs that have not entered are discarded (stopped; their results would be timing dependent).
  {
    std::unique_lock<std::mutex> lk(mu);
    for (auto& job : jobs) job->stop = true;
    jobs_discarded = static_cast<long long>(jobs.size());
    job_queue.clear();
    cv.wait(lk, [&] {
      for (auto& job : jobs) {
        if (job->started && !job->done.load()) return false;
      }
      return true;
    });
    quit = true;
  }
  cv.notify_all();
  for (auto& th : threads) th.join();
  async_jobs_ = false;
  // Round statistics: busy = time in node processing, idle = threads waiting at the round barrier
  // (for the last task of the round), serial = hand-out, merge, FJ sync and checks.
  static const bool det_stats = std::getenv("MILP_DET_STATS") != nullptr;
  if (det_stats || log_.verbosity() >= 2) {
    double tot_busy = 0.0;
    for (double b : tbusy) tot_busy += b;
    const double cap = par_wall * T;
    log_.log(det_stats ? 0 : 2,
             "DET %d tasks, rounds %lld nodes/round %.1f  parallel %.2fs busy %.0f%% idle %.0f%% (empty tasks %lld)"
             "  serial %.2fs (FJ wait %.2fs, job wait %.2fs)  jobs %lld (%.2fs) dropped %lld discarded %lld"
             "  memory-capped rounds %lld\n",
             V, rounds, rounds ? static_cast<double>(nodes_.load()) / static_cast<double>(rounds) : 0.0, par_wall,
             cap > 0 ? 100.0 * tot_busy / cap : 0.0, cap > 0 ? 100.0 * (cap - tot_busy) / cap : 0.0, idle_slots, serial,
             fj_wait, job_wait, jobs_run, job_secs, jobs_dropped, jobs_discarded, capped_rounds);
    for (int t = 0; t < V; ++t) {
      log_.log(det_stats ? 0 : 2, "DET   worker %d: %lld nodes, busy %.2fs (dives %.2fs, sub-MIPs %.2fs), %lld LP iters\n",
               t, wnodes[t], busy[t], ws[t]->t_dive, ws[t]->t_sub, ws[t]->lp.iterations());
    }
  }
}

void Search::fj_start() {
  static const bool disabled = std::getenv("MILP_NO_FJ") != nullptr;
  if (disabled || !opts_.feasibility_jump || !opts_.sub_mip_heuristics || is_finite(opts_.cutoff)) return;
  if (std::find(p_.is_int.begin(), p_.is_int.end(), 1) == p_.is_int.end()) return;
  fj_sync_ = params_.deterministic;
  fj_allowed_ = 1000000;
  fj_thread_ = std::thread([this] {
    FjLimits lim;
    // ~3e8 effort units per second; stops early once it stops making progress.
    lim.effort = 10000000000LL;
    lim.stall_effort = 300000000LL;
    const double deadline =
        fj_sync_ ? kInf
                 : timer_.elapsed() + std::min(30.0, 0.25 * std::max(0.0, params_.time_limit - timer_.elapsed()));
    lim.stop = &fj_stop_;
    long long base = 0;  // effort of the earlier attempts
    if (fj_sync_) {
      lim.checkpoint = [this, &base](long long e) {
        e += base;
        std::unique_lock<std::mutex> lk(fj_mu_);
        fj_effort_ = e;
        if (e >= fj_allowed_) {
          fj_cv_.notify_all();
          fj_cv_.wait(lk, [&] { return e < fj_allowed_ || fj_stop_.load(); });
        }
        return !fj_stop_.load();
      };
    }
    auto publish = [this](const std::vector<double>& x, const char* src) {
      if (!fj_sync_) {
        try_incumbent(x, src);
        return;
      }
      std::lock_guard<std::mutex> lk(fj_mu_);
      fj_found_.push_back({fj_post_stamp_ >= 0 ? fj_post_stamp_ : fj_effort_, x, src});
    };
    std::vector<double> best, closest;
    // Without a solution FJ stalls after ~1 s; fresh seeds often succeed where one failed.
    constexpr int kAttempts = 8;
    long long min_viol = -1;
    for (int a = 0; a < kAttempts && best.empty() && !fj_stop_.load(); ++a) {
      if (a > 0 && !fj_sync_ && is_finite(incumbent_.load())) break;  // another heuristic succeeded
      lim.seed = params_.seed + 1000003u * static_cast<unsigned>(a);
      lim.time_limit = deadline - timer_.elapsed();
      if (lim.time_limit <= 0.0) break;
      FjStats st;
      feasibility_jump(p_, lim, [&](const std::vector<double>& x, double obj) {
        log_.log(3, "  FJ: solution %.10g at %.2fs\n", p_.display_sense * (obj + p_.offset), timer_.elapsed());
        best = x;
        publish(x, "feasibility jump");
      }, &st);
      base += st.effort;
      log_.log(2, "  FJ: %d solutions, %lld moves, min violated %lld, effort %lld, %.2fs\n", st.solutions, st.moves,
               st.min_violated, st.effort, timer_.elapsed());
      if (best.empty() && !st.closest.empty() && (min_viol < 0 || st.min_violated < min_viol)) {
        min_viol = st.min_violated;
        closest = std::move(st.closest);
      }
    }
    {
      // Later solutions are stamped with the total effort (>= the last checkpoint); the progress the
      // syncs wait for stays at the last checkpoint until done.
      std::lock_guard<std::mutex> lk(fj_mu_);
      fj_post_stamp_ = std::max(base, fj_effort_);
    }
    // Deterministic mode: no wall-clock caps on the sub-MIPs (a binding cap would be timing dependent).
    auto sub_time = [this] {
      const double rem = std::max(0.0, params_.time_limit - timer_.elapsed());
      return fj_sync_ ? rem : std::min(10.0, 0.1 * rem);
    };
    // No solution: repair the point with the fewest violated rows in a sub-MIP.
    if (best.empty() && !closest.empty() && !fj_stop_.load() && std::getenv("MILP_NO_FJ_REPAIR") == nullptr) {
      SubMipLimits sl;
      sl.node_limit = 1000;
      sl.time_limit = sub_time();
      if (fj_sync_) sl.iteration_limit = 100000;
      else sl.stop = &fj_stop_;  // the search may end (restart) while this sub-MIP runs
      SubMipStats ss;
      sl.stats = &ss;
      std::vector<double> sol;
      const bool ok = repair(p_, closest, kInf, sl, sol);
      log_.log(2, "  FJ repair: %s, fixed %d/%d, %.2fs\n", ok ? "found" : "failed", ss.fixed, ss.ints,
               timer_.elapsed());
      if (ok) publish(sol, "feasibility jump + repair");
    }
    // Polish: re-solve the continuous part with the integers fixed.
    if (!best.empty() && !fj_stop_.load() && std::find(p_.is_int.begin(), p_.is_int.end(), 0) != p_.is_int.end()) {
      std::vector<double> lb = p_.col_lb, ub = p_.col_ub, sol;
      for (Index j = 0; j < p_.num_cols(); ++j) {
        if (p_.is_int[j]) lb[j] = ub[j] = best[j];
      }
      SubMipLimits sl;
      sl.node_limit = 10;
      sl.time_limit = sub_time();
      if (fj_sync_) sl.iteration_limit = 100000;
      if (solve_sub_mip(p_, lb, ub, kInf, sl, sol)) publish(sol, "feasibility jump + LP");
    }
    // Opportunistic mode: while the root node is still being processed (long root LPs), improve
    // the incumbent with mutation sub-MIPs on this otherwise idle thread.
    if (!fj_sync_ && std::getenv("MILP_NO_EARLY_LNS") == nullptr) {
      double fix = 0.7;
      for (unsigned seq = 0; !fj_stop_.load() && !root_done_.load(); ++seq) {
        const double rem = params_.time_limit - timer_.elapsed();
        std::vector<double> inc, sol;
        incumbent_x_global(inc);
        if (inc.empty() || rem < 1.0) break;
        SubMipLimits sl;
        sl.node_limit = 500;
        sl.time_limit = std::min(5.0, 0.05 * rem);
        sl.stop = &fj_stop_;  // the search may end (restart) while this sub-MIP runs
        SubMipStats ss;
        sl.stats = &ss;
        const unsigned seed = params_.seed * 7919u + 104729u + seq;
        const bool ok = mutation(p_, inc, fix, seed, incumbent_.load() + p_.offset, sl, sol);
        if (ss.solves == 0) break;
        if (ok && try_incumbent(sol, "mutation")) continue;
        fix = ss.exhausted ? std::max(0.2, fix - 0.1) : std::min(0.95, fix + 0.05);
      }
    }
    std::lock_guard<std::mutex> lk(fj_mu_);
    fj_done_ = true;
    fj_cv_.notify_all();
  });
}

void Search::fj_sync(long long release, bool lag) {
  if (!fj_sync_ || !fj_thread_.joinable()) return;
  std::vector<FjFound> found;
  {
    std::unique_lock<std::mutex> lk(fj_mu_);
    // Every solution stamped below `target` has been published once FJ passed a checkpoint at or
    // beyond it (or finished): the solutions taken do not depend on thread timing.
    const long long target = lag ? fj_prev_allowed_ : fj_allowed_;
    fj_cv_.wait(lk, [&] { return fj_done_ || fj_effort_ >= target; });
    auto keep = std::stable_partition(fj_found_.begin(), fj_found_.end(),
                                      [&](const FjFound& f) { return f.stamp >= target; });
    for (auto it = keep; it != fj_found_.end(); ++it) found.push_back(std::move(*it));
    fj_found_.erase(keep, fj_found_.end());
    fj_prev_allowed_ = fj_allowed_;
    static const long long min_release = static_cast<long long>(knob("MILP_FJ_MIN_RELEASE", 100000));
    fj_allowed_ += std::clamp<long long>(release, lag ? min_release : 1000000, 2000000000);
    fj_cv_.notify_all();
  }
  for (auto& f : found) try_incumbent(f.x, f.source);
}

void Search::fj_finish() {
  if (!fj_thread_.joinable()) return;
  {
    std::lock_guard<std::mutex> lk(fj_mu_);
    fj_stop_ = true;
    fj_cv_.notify_all();
  }
  fj_thread_.join();
  fj_found_.clear();  // deterministic mode: found after the last sync point (timing dependent)
}

bool Search::use_barrier_root() const {
  // Main search only (not sub-MIPs), first root only (a restarted root has cuts and a known basis).
  if (!opts_.sub_mip_heuristics || opts_.restarts_done > 0 || opts_.tree_restarts_done > 0) return false;
  if (p_.num_rows() == 0 || p_.num_cols() == 0) return false;
  switch (params_.lp_method) {
    case LpMethod::Simplex: return false;
    case LpMethod::Barrier: return true;
    case LpMethod::Auto: break;
  }
  return false;
}

bool Search::use_concurrent_root() const {
  if (params_.lp_method != LpMethod::Auto || params_.deterministic || threads_ < 2) return false;
  if (!opts_.sub_mip_heuristics || opts_.restarts_done > 0 || opts_.tree_restarts_done > 0) return false;
  if (std::find(p_.is_int.begin(), p_.is_int.end(), 1) != p_.is_int.end()) return false;  // MIP roots: simplex
  // Small LPs take milliseconds with the simplex.
  static const long long min_nnz = static_cast<long long>(knob("MILP_CONCURRENT_NNZ", 10000));
  return p_.num_rows() > 0 && static_cast<long long>(p_.A.nnz()) >= min_nnz;
}

void Search::concurrent_root(DualSimplex& lp) {
  const double t0 = timer_.elapsed();
  DualSimplex blp = lp;
  std::atomic<bool> stop{false};
  BarrierStats bs;
  LpStatus bst = LpStatus::NumericalError;
  double t_barrier = 0.0;
  Logger quiet(0);
  std::thread tb([&] {
    BarrierOptions bo;
    bo.log = &quiet;
    bo.stop = &stop;
    bo.time_limit = params_.time_limit - timer_.elapsed();
    bo.ipm.threads = std::max(1, threads_ - 1);
    bst = barrier_solve(blp, p_.A, p_.cost, p_.col_lb, p_.col_ub, p_.row_lb, p_.row_ub, bo, &bs);
    t_barrier = timer_.elapsed() - t0;
    if (bst == LpStatus::Optimal && bs.used) stop.store(true);
  });
  LpOptions so;
  so.tol = params_.tol;
  so.time_limit = params_.time_limit - timer_.elapsed();
  so.stop = &stop;
  const LpStatus sst = lp.solve(so);
  const double t_simplex = timer_.elapsed() - t0;
  const bool simplex_done = sst == LpStatus::Optimal || sst == LpStatus::Infeasible || sst == LpStatus::Unbounded;
  if (simplex_done) stop.store(true);
  tb.join();
  if (!simplex_done && bst == LpStatus::Optimal && bs.used) {
    lp = blp;
    log_.log(1, "Concurrent LP: barrier finished first (%.2fs: %d iterations, crossover and %lld simplex iterations)\n",
             t_barrier, bs.ipm.iterations, bs.cleanup_iterations);
  } else if (simplex_done) {
    log_.log(1, "Concurrent LP: dual simplex finished first (%.2fs)\n", t_simplex);
  }
}

void Search::barrier_root(DualSimplex& lp) {
  BarrierOptions bo;
  bo.log = &log_;
  bo.time_limit = params_.time_limit - timer_.elapsed();
  bo.ipm.threads = threads_;
  bo.ipm.log_offset = p_.offset;
  bo.ipm.log_sense = p_.display_sense;
  BarrierStats bs;
  const LpStatus st = barrier_solve(lp, p_.A, p_.cost, p_.col_lb, p_.col_ub, p_.row_lb, p_.row_ub, bo, &bs);
  if (!bs.used) {
    log_.log(1, "Barrier: %s (%s); solving the root LP with the dual simplex\n", to_string(bs.ipm.status),
             to_string(st));
  } else if (st != LpStatus::Optimal) {
    log_.log(1, "Barrier: crossover cleanup ended %s; the dual simplex continues from its basis\n", to_string(st));
  }
}

void Search::detect_symmetry_root() {
  if (std::getenv("MILP_NO_SYMMETRY") != nullptr || !opts_.sub_mip_heuristics) return;
  if (!p_.implications.empty()) return;  // the implications are not part of the symmetry detection
  if (std::find(p_.is_int.begin(), p_.is_int.end(), 1) == p_.is_int.end()) return;
  SymmetryOptions so;
  // Deterministic mode: the work limit only (a binding time cap would be timing dependent).
  const double rem = std::max(0.0, params_.time_limit - timer_.elapsed());
  so.time_limit = params_.deterministic ? rem : std::max(1.0, 0.05 * rem);
  sym_ = detect_symmetry(p_, so);
  if (!sym_.empty() || sym_.seconds > 0.1) {
    log_.log(1, "Symmetry: %d generators moving %d columns, group order ~1e%.1f, %.2fs%s\n", sym_.num_generators(),
             sym_.moved_cols, sym_.log10_order(), sym_.seconds, sym_.aborted ? " (budget exhausted)" : "");
  }
}

Index Search::count_global_fixed() {
  std::lock_guard<std::mutex> lk(gfix_mu_);
  Index fixed = 0;
  if (glb_.empty()) return 0;
  for (Index j = 0; j < p_.num_cols(); ++j) {
    fixed += p_.is_int[j] && glb_[j] == gub_[j] && p_.col_lb[j] != p_.col_ub[j];
  }
  return fixed;
}

void Search::check_tree_restart(bool det) {
  static const double frac = knob("MILP_TREE_RESTART", 0.1);  // <= 0: off
  if (!opts_.allow_tree_restart || frac <= 0.0 || tree_restart_.load() || stop_.load()) return;
  // Not once the tree has done most of the work: at least half of the root gap still open, and in
  // opportunistic mode at most 60% of the time limit used.
  const double inc = incumbent_.load(), lb = pool_->lower_bound();
  if (!is_finite(inc) || !is_finite(lb) || !is_finite(root_bound_)) return;
  if (inc - lb < 0.5 * (inc - root_bound_)) return;
  if (!det && is_finite(params_.time_limit) && timer_.elapsed() > t_start_ + 0.6 * (params_.time_limit - t_start_)) return;
  Index ints = 0, free_ints = 0;
  for (Index j = 0; j < p_.num_cols(); ++j) {
    if (!p_.is_int[j]) continue;
    ++ints;
    free_ints += p_.col_lb[j] != p_.col_ub[j];
  }
  const Index fixed = count_global_fixed();
  // The later in the tree, the more fixings it takes (the restart throws the tree away: mkp_hard_1
  // restarted after 20-35k nodes with 21% fixed took 4 s instead of 3).
  static const double early = knob("MILP_TREE_RESTART_NODES", 5000);
  const double late = std::max(1.0, static_cast<double>(nodes_.load()) / early);
  const Index need = static_cast<Index>(
      late * static_cast<double>(std::max<Index>(20, static_cast<Index>(frac * static_cast<double>(free_ints - tree_fixed0_)))));
  if (fixed - tree_fixed0_ < need) return;
  log_.log(1, "Restart: %d of %d integer variables fixed globally in the tree (%d at its start), %lld nodes\n", fixed,
           ints, tree_fixed0_, nodes_.load());
  tree_restart_ = true;
  stop_ = true;
}

void Search::fill_restart(MipResult& res, bool with_pool) {
  res.restart = true;
  {
    std::lock_guard<std::mutex> lk(gfix_mu_);
    res.restart_lb = glb_;
    res.restart_ub = gub_;
  }
  const SparseMatrix& R = root_lp_.row_matrix();
  for (Index i = p_.num_rows(); i < root_lp_.num_rows(); ++i) {
    for (Index k = R.begin(i); k < R.end(i); ++k) {
      res.cut_index.push_back(R.index[k]);
      res.cut_value.push_back(R.value[k]);
    }
    res.cut_start.push_back(static_cast<Index>(res.cut_index.size()));
    res.cut_lb.push_back(-root_lp_.var_upper(root_lp_.num_cols() + i));
    res.cut_ub.push_back(-root_lp_.var_lower(root_lp_.num_cols() + i));
  }
  if (with_pool && use_pool_) {
    // The most recently used live pool cuts (globally valid), at most a quarter of the problem rows (>= 20)
    // and half of nnz(A) (mkp_hard_1: 384 dense pool rows on a 10-row model made the restarted node LPs
    // 10x slower). Ties by id (deterministic mode merges use times at barriers).
    std::lock_guard<std::mutex> lk(ncp_mu_);
    std::vector<std::pair<long long, int>> order;
    for (size_t id = 0; id < ncp_.size(); ++id) {
      if (ncp_[id]) order.emplace_back(-ncp_used_[id], static_cast<int>(id));
    }
    std::sort(order.begin(), order.end());
    const size_t max_cuts = static_cast<size_t>(std::max<Index>(20, p_.num_rows() / 4));
    long long nnz = 0;
    const long long max_nnz = static_cast<long long>(p_.A.index.size()) / 2;
    size_t taken = 0;
    for (const auto& [neg_used, id] : order) {
      if (taken >= max_cuts) break;
      const Cut& c = *ncp_[id];
      if (nnz + static_cast<long long>(c.idx.size()) > max_nnz) continue;
      nnz += static_cast<long long>(c.idx.size());
      for (size_t k = 0; k < c.idx.size(); ++k) res.cut_index.push_back(c.idx[k]), res.cut_value.push_back(c.val[k]);
      res.cut_start.push_back(static_cast<Index>(res.cut_index.size()));
      res.cut_lb.push_back(-kInf);
      res.cut_ub.push_back(c.rhs);
      ++taken;
    }
    log_.log(2, "  restart keeps %zu pool cuts (%lld nonzeros) and %d root cut rows\n", taken, nnz,
             static_cast<int>(root_lp_.num_rows() - p_.num_rows()));
  }
  res.debug_point = debug_sol_;
}

MipResult Search::run() {
  MipResult res;
  t_start_ = timer_.elapsed();
  debug_root_ = opts_.sub_mip_heuristics;
  // Sub-MIP heuristics keep the old permanent node cut rows (short searches; pool separation at
  // every node would mostly add LP work there).
  if (!opts_.sub_mip_heuristics) use_pool_ = false;
  load_debug_solution();
  if (is_finite(opts_.cutoff)) {
    external_cutoff_ = opts_.cutoff - p_.offset;
    cutoff_.store(prune_threshold(external_cutoff_));
  }
  if (!opts_.start.empty() && !try_incumbent(opts_.start, "restart")) {
    external_cutoff_ = std::min(external_cutoff_, p_.objective(opts_.start) - p_.offset);
    cutoff_.store(prune_threshold(external_cutoff_));
  }
  root_lp_.load(p_.A, p_.cost, p_.col_lb, p_.col_ub, p_.row_lb, p_.row_ub);
  root_lp_.set_seed(params_.seed);
  cut_data_.build(p_);
  pool_ = std::make_unique<NodePool>(threads_);
  {
    Index ints = 0;
    for (char c : p_.is_int) ints += c != 0;
    conf_max_nnz_ = std::max<Index>(20, ints / 5 + 20);
    // Each worker keeps up to this many conflicts (~500k nonzeros at most).
    conf_capacity_ = static_cast<size_t>(std::clamp<long long>(500000LL / conf_max_nnz_, 100, 2000));
  }
  fj_start();
  struct FjGuard {
    Search* s;
    ~FjGuard() { s->fj_finish(); }
  } fj_guard{this};

  // Root node is processed on the calling thread to keep its log ordered.
  {
    Worker w;
    w.lp = root_lp_;
    w.pc = &pc_;
    Node root;
    root.id = 0;
    detect_symmetry_root();
    if (!sym_.empty()) {
      // Generators of orbitope components are handled by orbitopal fixing; orbital branching
      // uses the others only (their supports are disjoint, so both reductions combine).
      std::vector<char> used;
      const double t_orb = timer_.elapsed();
      if (std::getenv("MILP_NO_ORBITOPE") == nullptr) orbs_ = detect_packing_orbitopes(p_, sym_, &used);
      const double dt_orb = timer_.elapsed() - t_orb;
      auto g = std::make_shared<PermGroup>();
      for (size_t k = 0; k < sym_.gens.size(); ++k) {
        if (used.empty() || !used[k]) g->gens.push_back(sym_.gens[k]);
      }
      if (!orbs_.empty() && !debug_sol_.empty()) {
        for (const Orbitope& o : orbs_) orbitope_canonicalize(o, debug_sol_);
        const double v = p_.max_violation(debug_sol_);
        log_.log(0, "DEBUG-SOL: canonicalized for the orbitopes (violation %.3g)\n", v);
        if (v > 1e-6) debug_sol_.clear();
      }
      if (!orbs_.empty()) {
        Index cols = 0;
        for (const Orbitope& o : orbs_) cols += o.rows * o.cols;
        log_.log(1, "Symmetry: %zu packing orbitope(s) on %d columns, %.2fs\n", orbs_.size(), cols, dt_orb);
      } else if (dt_orb > 0.1) {
        log_.log(1, "Symmetry: no packing orbitope, %.2fs\n", dt_orb);
      }
      if (!g->gens.empty()) root.sym = std::move(g);
    }
    Node children[2];
    pool_->set_worker_bound(0, -kInf);
    if (use_barrier_root()) barrier_root(w.lp);
    else if (use_concurrent_root()) concurrent_root(w.lp);
    if (p_.is_int.end() != std::find(p_.is_int.begin(), p_.is_int.end(), 1)) {
      std::vector<Cut> dropped;
      static const bool no_lurk = std::getenv("MILP_NO_LURKRC") != nullptr;
      std::function<void(const DualSimplex&)> on_lp;
      if (!no_lurk && std::getenv("MILP_NO_RCFIX") == nullptr) {
        lurk_lo_.assign(p_.num_cols(), -kInf);
        lurk_up_.assign(p_.num_cols(), -kInf);
        on_lp = [this](const DualSimplex& lp) { record_lurking(lp); };
      }
      // Reduced-cost fixing between cut rounds against the current cutoff (deterministic: the one at the start).
      const double cutoff0 = cutoff_.load();
      std::function<double()> cutoff_fn = [this, cutoff0, det = params_.deterministic]() {
        return det ? cutoff0 : cutoff_.load();
      };
      const CutLoopStats cs = root_cut_loop(p_, w.lp, params_, log_, timer_, incumbent_.load(), &cut_data_,
                                            !opts_.sub_mip_heuristics || opts_.no_zerohalf,
                                            use_pool_ ? &dropped : nullptr, on_lp, opts_.stop_flag, cutoff_fn);
      if (!debug_sol_.empty()) {
        const SparseMatrix& RR = w.lp.row_matrix();
        for (Index i = p_.num_rows(); i < w.lp.num_rows(); ++i) {
          double act = 0.0;
          for (Index k = RR.begin(i); k < RR.end(i); ++k) act += RR.value[k] * debug_sol_[RR.index[k]];
          const double ub = -w.lp.var_lower(w.lp.num_cols() + i);
          if (act > ub + 1e-6 * (1.0 + std::abs(ub))) log_.log(0, "DEBUG-SOL: root cut row %d violated (%g > %g)\n", i, act, ub);
        }
        for (const Cut& c : dropped) {
          if (c.activity(debug_sol_.data()) > c.rhs + 1e-6 * (1.0 + std::abs(c.rhs))) log_.log(0, "DEBUG-SOL: dropped root cut (%s) violated\n", c.origin);
        }
      }
      // Root LP and cut loop work (not in lp_iters_) for sub-MIP job progress.
      if (opts_.progress) opts_.progress->fetch_add(w.lp.iterations());
      if (use_pool_) {
        const long long nnz = static_cast<long long>(p_.A.index.size());
        ncp_max_ = static_cast<size_t>(knob("MILP_POOL_MAX", 20000));
        ncp_max_nnz_ = static_cast<long long>(knob("MILP_POOL_NNZ_MULT", 1.0) * static_cast<double>(std::clamp<long long>(10 * nnz, 200000, 4000000)));
        pool_publish(dropped, nullptr);
      }
      root_cs_before_ = cs.rounds > 0 ? cs.bound_before : w.lp.objective();
      root_cs_rounds_ = cs.rounds;
      root_cs_cuts_ = cs.cuts_added;
      if (cs.rounds > 0) {
        log_.log(1, "Cuts: %d rounds, %d cuts, bound %.10g -> %.10g\n", cs.rounds, cs.cuts_added,
                 p_.display_sense * (cs.bound_before + p_.offset), p_.display_sense * (cs.bound_after + p_.offset));
      }
    }
    fj_sync(3 * w.lp.iterations() * p_.num_rows());  // deterministic point: FJ results enter before the root node
    fj_iters_mark_ = lp_iters_.load();
    const int nc = process(w, root, children);
    root_done_ = true;
    {
      // Machine-readable root summary (benchmarks/tools/rootgap.py): bound after the root node (the
      // incumbent if the root was pruned), the root LP bound before cuts and the root cut loop.
      double rb = incumbent_.load();
      if (nc > 0) {
        rb = kInf;
        for (int k = 0; k < nc; ++k) rb = std::min(rb, std::max(children[k].bound, root_bound_));
      }
      log_.log(1, "ROOTSTAT bound=%.12g lp=%.12g nodelp=%.12g rounds=%d cuts=%d rows=%d time=%.3f\n",
               p_.display_sense * (rb + p_.offset), p_.display_sense * (root_cs_before_ + p_.offset),
               p_.display_sense * (root_bound_ + p_.offset), root_cs_rounds_, root_cs_cuts_,
               static_cast<int>(w.lp.num_rows() - p_.num_rows()), timer_.elapsed());
    }
    report_progress();
    pool_->set_worker_bound(0, kInf);
    root_lp_ = w.lp;  // workers start from the solved root basis
    static const bool cutprop = std::getenv("MILP_CUTPROP") != nullptr;
    if (cutprop && root_lp_.num_rows() > p_.num_rows()) {
      auto pp = std::make_unique<MipProblem>(p_);
      SparseMatrix R = p_.A.transpose();
      const SparseMatrix& LR = root_lp_.row_matrix();
      const Index n = p_.num_cols();
      for (Index i = p_.num_rows(); i < root_lp_.num_rows(); ++i) {
        R.append(LR.index.data() + LR.begin(i), LR.value.data() + LR.begin(i), LR.end(i) - LR.begin(i));
        pp->row_lb.push_back(-root_lp_.var_upper(n + i));
        pp->row_ub.push_back(-root_lp_.var_lower(n + i));
      }
      R.num_minor = n;
      pp->A = R.transpose();
      prop_p_ = std::move(pp);
    }
    res.root_lp_work = static_cast<double>(w.lp.iterations()) * (w.lp.num_rows() + w.lp.num_cols());
    res.root_bound = root_bound_ + p_.offset;
    res.root_lns_ran = root_lns_ran_;
    res.root_lns_improved = root_lns_improved_;
    for (int k = 0; k < nc; ++k) pool_->push(std::move(children[k]));
  }

  // Root restart: if root reduced-cost fixing fixed many integer variables,
  // hand the tightened problem plus root cuts back for a fresh presolve.
  static const bool no_restart = std::getenv("MILP_NO_RESTART") != nullptr;
  if (opts_.allow_restart && !no_restart && !stop_.load() && pool_->size() > 0 && root_rc_ready_) {
    Index ints = 0, newly_fixed = 0;
    {
      std::lock_guard<std::mutex> lk(gfix_mu_);
      for (Index j = 0; j < p_.num_cols(); ++j) {
        if (!p_.is_int[j]) continue;
        ++ints;
        newly_fixed += glb_[j] == gub_[j] && p_.col_lb[j] != p_.col_ub[j];
      }
    }
    if (ints > 0 && newly_fixed >= std::max<Index>(20, ints / 10)) {
      log_.log(1, "Restart: %d of %d integer variables fixed at the root\n", newly_fixed, ints);
      fill_restart(res, false);
      stop_ = true;  // skip the tree; the caller restarts
    }
  }
  if (!stop_.load()) tree_fixed0_ = count_global_fixed();

  if (!stop_.load() && pool_->size() > 0) {
    const bool det = params_.deterministic && threads_ > 1;
    // Reference of the sparse basis form: the root basis (columns and base rows).
    if (root_lp_.num_cols() > 0) ref_basis_ = std::make_shared<const std::vector<VarStatus>>(root_lp_.get_basis().status);
    if (!det && opts_.check_resident) base_rss_ = std::max(0LL, resident_bytes() - tree_bytes());
    log_.log(1, "Branch-and-bound with %d threads%s\n", threads_, det ? " (deterministic)" : "");
    if (det) {
      run_deterministic();
    } else {
      std::vector<std::thread> threads;
      for (int t = 0; t < threads_; ++t) threads.emplace_back(&Search::worker_loop, this, t);
      for (auto& th : threads) th.join();
    }
    maybe_log(true);
    if (conf_learned_.load() > 0) {
      log_.log(1, "Conflicts: %lld learned (%lld from propagation), %lld nodes pruned by conflict propagation\n",
               conf_learned_.load(), prop_conf_learned_.load(), conf_pruned_.load());
      if (prop_conf_learned_.load() > 0) {
        log_.log(2, "  clauses: %.1f literals on average, %lld LP proofs not analysed (trail mismatch)\n",
                 static_cast<double>(prop_conf_lits_.load()) / static_cast<double>(prop_conf_learned_.load()),
                 trail_mismatch_.load());
      }
    }
    if (use_pool_ && ncp_rounds_.load() > 0) {
      log_.log(1, "Cut pool: %zu cuts (%lld deleted), %lld node rounds re-added %lld cuts, %lld rows aged out; rounds"
               " at every %d-th node, %lld strips\n",
               ncp_.size(), ncp_purged_.load(), ncp_rounds_.load(), ncp_added_.load(), ncp_removed_.load(),
               pool_interval_.load(), ncp_stripped_.load());
    }
    if (mem_peak_.load() > 0) {
      log_.log(mem_drains_.load() > 0 || mem_stripped_.load() > 0 ? 1 : 2,
               "Memory: tree peak %.1f MB (of %.1f MB), %lld drain phases, %lld stored bases dropped\n",
               mem_peak_.load() / 1048576.0, mem_limit_bytes() / 1048576.0, mem_drains_.load(),
               mem_stripped_.load());
    }
    if (lp_work_total_.load() > 0) {
      log_.log(2, "  propagation work %.3g, %.1f%% of the node LP work\n", static_cast<double>(prop_work_total_.load()),
               100.0 * static_cast<double>(prop_work_total_.load()) / static_cast<double>(lp_work_total_.load()));
    }
    if (obj_prop_calls_.load() + obj_prop_skipped_.load() > 0) {
      log_.log(2, "  objective propagation: %lld nodes, %lld useful, %lld skipped\n", obj_prop_calls_.load(),
               obj_prop_useful_.load(), obj_prop_skipped_.load());
    }
    if (sb_probes_.load() > 0) {
      log_.log(1, "Strong branching: %lld probes, %lld LP iterations (%.1f%% of all)\n", sb_probes_.load(), sb_iters_.load(),
               100.0 * static_cast<double>(sb_iters_.load()) / static_cast<double>(std::max(1LL, lp_iters_.load())));
    }
    if (sb_fixed_.load() + sb_degen_skips_.load() > 0) {
      log_.log(2, "  strong branching: %lld domain reductions (one child cut off), %lld of %lld branchings without "
               "probes (dual degenerate LP)\n", sb_fixed_.load(), sb_degen_skips_.load(), sb_calls_.load());
    }
    {
      // path: global fixings contradict the branching; late cutoff: LP optimal at or above the cutoff
      // (also after pool rounds); after heuristics: the incumbent improved during dives / sub-MIPs;
      // strong branching: both children of the chosen candidate infeasible or cut off.
      static const char* const kOutcomeNames[kOutcomes] = {
          "path",        "propagation", "orbitope", "LP infeasible",    "LP cutoff",       "cut round",
          "late cutoff", "integral",    "branched", "after heuristics", "strong branching"};
      std::string line;
      long long total = 0;
      for (int k = 0; k < kOutcomes; ++k) total += outcome_[k].load();
      for (int k = 0; k < kOutcomes; ++k) {
        const long long c = outcome_[k].load();
        if (c == 0) continue;
        char buf[96];
        std::snprintf(buf, sizeof(buf), " %s %lld (depth %.1f)", kOutcomeNames[k], c,
                      static_cast<double>(outcome_depth_[k].load()) / static_cast<double>(c));
        line += buf;
      }
      if (total > 0) log_.log(2, "Node outcomes:%s\n", line.c_str());
      {
        std::string h;
        for (int d = 0; d < 64; ++d) {
          if (dh_br_[d] + dh_cut_[d] == 0) continue;
          char b[48];
          std::snprintf(b, sizeof b, " %d:%lld/%lld", d, dh_br_[d].load(), dh_cut_[d].load());
          h += b;
        }
        log_.log(3, "Node outcomes by depth (branched/closed):%s\n", h.c_str());
      }
      if (popped_n_.load() + plunged_n_.load() > 0) {
        const double k = static_cast<double>(popped_n_.load() + plunged_n_.load());
        log_.log(2, "Pool rows in node LPs: %.1f on average, %.1f nonzeros each\n", prow_sum_.load() / k,
                 prow_nnz_.load() / std::max(1.0, static_cast<double>(prow_sum_.load())));
      }
      log_.log(2, "Node LP iterations: %lld first solves (%lld of them at %lld plunge starts, %lld in plunges), %lld cut rounds\n",
               node_it_total_.load(), popped_it_.load(), popped_n_.load(), plunged_n_.load(), pool_it_total_.load());
      if (tmir_rounds_.load() > 0) {
        log_.log(2, "Tableau cuts: %lld separation rounds, %lld successful\n", tmir_rounds_.load(), tmir_won_.load());
      }
    }
    if (orb_fixed_.load() > 0 || orb_pruned_.load() > 0) {
      log_.log(1, "Orbitopes: %lld fixings, %lld nodes pruned\n", orb_fixed_.load(), orb_pruned_.load());
    }
    if (lp_failed_.load() > 0 || lp_unsolved_.load() > 0) {
      log_.log(1, "Node LP failures: %lld, recovered %lld, branched without LP %lld, all-integers-fixed re-solves %lld\n",
               lp_failed_.load(), lp_recovered_.load(), lp_unsolved_.load(), fixed_recovered_.load());
    }
    if (impl_branches_.load() > 0) {
      log_.log(1, "Logical implications: %lld branchings\n", impl_branches_.load());
    }
    if (polish_tried_.load() > 0) {
      log_.log(2, "Fixed-integer re-solves of near-feasible points: %lld, %lld feasible, %lld LP iterations\n",
               polish_tried_.load(), polish_ok_.load(), polish_iters_.load());
    }
    if (sym_branches_.load() > 0) {
      log_.log(1, "Symmetry: %lld orbital branchings fixed %lld variables, %lld stabilizers\n", sym_branches_.load(),
               sym_fixed_.load(), sym_stabs_.load());
    }
  }

  if (tree_restart_.load()) {
    fill_restart(res, true);
    res.tree_restart = true;
  }
  fj_finish();
  report_progress();
  res.nodes = nodes_.load();
  res.lp_iterations = lp_iters_.load();
  res.mem_drains = mem_drains_.load();
  const double inc = incumbent_.load();
  res.has_solution = is_finite(inc);
  if (res.has_solution) {
    res.x = inc_x_;
    res.objective = inc + p_.offset;
  }
  double lb = std::min(pool_->lower_bound(), inc);
  if (root_unbounded_) {
    res.status = res.has_solution ? Status::Unbounded : Status::InfeasibleOrUnbounded;
    res.bound = -kInf;
    return res;
  }
  if (stop_reason_ == Status::TimeLimit || stop_reason_ == Status::NodeLimit || stop_reason_ == Status::MemoryLimit ||
      stop_reason_ == Status::Interrupted || stop_reason_ == Status::IterationLimit) {
    res.status = stop_reason_;
  } else {
    res.status = res.has_solution ? Status::Optimal : Status::Infeasible;
    if (!res.has_solution) lb = kInf;
  }
  if (lost_bound_ < lb) {
    lb = lost_bound_;
    if (res.status == Status::Optimal && !gap_closed(lb)) res.status = Status::NumericalError;
    if (res.status == Status::Infeasible) res.status = Status::NumericalError;
  }
  if (obj_integral_ && is_finite(lb)) lb = std::ceil(lb - 1e-6);
  if (res.has_solution) lb = std::min(lb, inc);
  res.bound = lb + p_.offset;
  if (numerical_) log_.log(1, "Warning: some node LPs failed numerically; result may be inexact\n");
  return res;
}

}  // namespace

MipResult branch_and_bound(const MipProblem& p, const SolverParams& params, const Logger& log, const Timer& timer,
                           const SearchOptions& opts) {
  Search s(p, params, log, timer, opts);
  return s.run();
}

}  // namespace milp
