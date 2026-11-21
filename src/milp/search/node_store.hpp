#pragma once
// Open-node storage of the branch-and-bound search (search/branch_and_bound.cpp): bound paths
// (PathSeg), stored warm-start bases (PackedBasis), the worker-side Node and the queue of open
// nodes (NodePool).
//
// Memory per open node is what limits long searches under --mem-limit, so the queue stores nodes
// compactly: an 88-byte QNode per open node in chunked slots and two 20-byte heap keys (best bound,
// best estimate; in drain mode an 8-byte entry in a per-depth bucket replaces the estimate key).
// Bases are single allocations with an intrusive reference count. The pop order is a total order
// on immutable key data (key, deeper first, smaller id), so it does not depend on the heap layout,
// slot positions, compaction or purges. (16-byte keys with the depth read from the slot on ties
// would not be immutable: a stale key's slot may hold another node by then, and a heap whose
// comparisons change can hide its minimum.)

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "milp/core/types.hpp"
#include "milp/lp/simplex.hpp"

namespace milp {

struct PermGroup;

namespace bnb {

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
  // The value for_each decodes from the encoding of v (integral values pass through int32: -0.0
  // comes back as 0.0).
  static double canonical(double v) { return fits_int(v) ? static_cast<double>(static_cast<int32_t>(v)) : v; }
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
  // Raw ownership transfer (NodePool slots hold the plain pointer).
  static PathRef adopt(const PathSeg* p) {
    PathRef r;
    r.p_ = p;
    return r;
  }
  const PathSeg* detach() {
    const PathSeg* p = p_;
    p_ = nullptr;
    return p;
  }

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

// Warm-start basis stored with queued nodes: one allocation (16-byte header, then the status
// stream, then the pool stream) with an intrusive reference count. The first `size` statuses
// (columns and base rows) are kept either packed at 2 bits each or, when shorter, as the positions
// where they differ from a reference basis (the search's first stored basis, Search::ref_basis_,
// passed to make() and unpack(); beyond its length the reference is Basic), varint-encoded as
// (gap << 2 | status). With the dynamic cut pool, pool cut rows are identified by pool id and only
// the nonbasic ones are kept (rows with a basic logical can be dropped or added without losing
// dual feasibility), as ascending ids, varint (id gap << 2 | status). A cut deleted from the pool
// meanwhile stays alive while a stored basis refers to it (Search::pool_hold), so the node can
// still be warm started. Release (the last reference gone) goes through a BasisOwner, which keeps
// the accounting and the pool holds.
class PackedBasis {
 public:
  // size < 0: all of b. ref: reference of the sparse form (null: dense). pool_nb: nonbasic pool
  // rows (pool id, status), ids ascending (null: none). One reference (the caller's).
  // weights: steepest-edge weights of the basic variables among the first `size` (DualSimplex::basic_weights),
  // kept after the pool stream as a count and single-precision values (null or empty: none).
  static PackedBasis* make(const Basis& b, Index size, const std::vector<VarStatus>* ref,
                           const std::vector<std::pair<int, VarStatus>>* pool_nb,
                           const std::vector<float>* weights = nullptr);
  bool has_weights() const { return (nbits_ & kWeights) != 0; }
  // Copies the stored weights; false if there are none.
  bool weights(std::vector<float>& out) const;
  void retain() const { refs_.fetch_add(1, std::memory_order_relaxed); }
  // Drops a reference; true if it was the last one (the caller then calls destroy()).
  bool unref() const { return refs_.fetch_sub(1, std::memory_order_acq_rel) == 1; }
  static void destroy(const PackedBasis* b) {
    b->~PackedBasis();
    ::operator delete(const_cast<PackedBasis*>(b));
  }
  Index size() const { return size_; }
  bool sparse() const { return (nbits_ & kSparse) != 0; }
  uint32_t status_bytes() const { return nbits_ & ~(kSparse | kWeights); }
  int npool() const { return npool_; }
  template <class F>
  void for_each_pool(F&& f) const {
    const uint8_t* q = data() + status_bytes();
    int id = -1;
    for (int k = 0; k < npool_; ++k) {
      const uint64_t v = get_varint(q);
      id += static_cast<int>(v >> 2) + 1;
      f(id, static_cast<VarStatus>(v & 3));
    }
  }
  // ref: the reference make() was given (needed by the sparse form).
  void unpack(Basis& b, const std::vector<VarStatus>* ref) const;
  // Heap bytes (the allocation, rounded up to the allocator's 16-byte quantum).
  long long bytes() const;

 private:
  PackedBasis() = default;
  static constexpr uint32_t kSparse = 1u << 31, kWeights = 1u << 30;
  const uint8_t* pool_end() const;
  const uint8_t* data() const { return reinterpret_cast<const uint8_t*>(this + 1); }
  uint8_t* data() { return reinterpret_cast<uint8_t*>(this + 1); }
  mutable std::atomic<int32_t> refs_{0};
  int32_t size_ = 0;
  uint32_t nbits_ = 0;  // bytes of the status stream; kSparse: difference form; kWeights: weights follow the pool stream
  int32_t npool_ = 0;   // nonbasic pool rows
};
static_assert(sizeof(PackedBasis) == 16, "basis header");

// Receives stored bases whose last reference is gone (accounting, pool holds); the basis is
// freed right after the call.
class BasisOwner {
 public:
  virtual void basis_released(const PackedBasis& b) = 0;

 protected:
  ~BasisOwner() = default;
};

// Owning handle of a stored basis on a worker-side Node (reference counted; its owner is told
// when the last reference goes).
class BasisRef {
 public:
  BasisRef() = default;
  BasisRef(PackedBasis* adopt, BasisOwner* owner) : p_(adopt), owner_(owner) {}
  BasisRef(const BasisRef& o) : p_(o.p_), owner_(o.owner_) {
    if (p_) p_->retain();
  }
  BasisRef(BasisRef&& o) noexcept : p_(o.p_), owner_(o.owner_) { o.p_ = nullptr; }
  BasisRef& operator=(const BasisRef& o) {
    if (o.p_) o.p_->retain();
    drop(p_, owner_);
    p_ = o.p_;
    owner_ = o.owner_;
    return *this;
  }
  BasisRef& operator=(BasisRef&& o) noexcept {
    if (this != &o) {
      drop(p_, owner_);
      p_ = o.p_;
      owner_ = o.owner_;
      o.p_ = nullptr;
    }
    return *this;
  }
  ~BasisRef() { drop(p_, owner_); }
  const PackedBasis* operator->() const { return p_; }
  const PackedBasis* get() const { return p_; }
  BasisOwner* owner() const { return owner_; }
  explicit operator bool() const { return p_ != nullptr; }
  void reset() {
    drop(p_, owner_);
    p_ = nullptr;
  }
  // Raw ownership transfer (NodePool slots hold the plain pointer; the pool knows the owner).
  PackedBasis* detach() {
    PackedBasis* p = p_;
    p_ = nullptr;
    return p;
  }
  static void drop(const PackedBasis* p, BasisOwner* owner) {
    if (p && p->unref()) {
      if (owner) owner->basis_released(*p);
      PackedBasis::destroy(p);
    }
  }

 private:
  PackedBasis* p_ = nullptr;
  BasisOwner* owner_ = nullptr;
};

// A node as the workers handle it (the queue keeps a packed copy, NodePool::QNode).
//
// Bound changes from the root: the segments of `path` (newest first), then `change`, the node's own
// branching bound kept inline (col < 0: none; a side equal to the column's original bound is
// infinite, as PathSeg::for_each decodes it). A node gets a segment only when it branches: its own
// change and its reduced-cost fixings, shared by both children (branch_segment), so an open node
// costs no segment of its own; a child gets one only for more than one change (orbital fixings).
// Applying the segments oldest first and then the inline change gives the same changes in the same
// order as one segment per child did.
struct Node {
  double bound = -kInf;  // lower bound on the LP objective (without offset)
  double estimate = -kInf;  // estimated best solution value in the subtree (pseudocosts)
  int depth = 0;
  Index anc_var = -1;  // the parent's branching variable (ancestral pseudocosts, MILP_APS)
  PathRef path;  // bound changes from the root (newest segment first), without `change`
  BoundChange change{-1, -kInf, kInf};  // own branching bound (applied after the path)
  BasisRef basis;  // warm start; dropped under memory pressure
  long long mem = 0;                       // estimated bytes owned while queued
  long long id = 0, parent = -1;
  Index branch_var = -1;
  int branch_dir = 0;        // -1 down, +1 up
  float anc_dist = 0.0f;     // the parent's branch_dist, negative if it branched down (MILP_APS)
  double branch_dist = 0.0;  // how far the branching moved the variable
  double parent_obj = -kInf;  // bound before MILP_SB_BOUNDS raised it (pseudocost updates)
  // Symmetry: a group stabilizing this node's branching constraints; null if trivial.
  std::shared_ptr<const PermGroup> sym;
};

// Open nodes, selectable by best bound, by best estimate or (drain mode, under memory pressure)
// deepest first with every (k+1)-th pop by best bound.
//
// Slots: QNodes in chunks of 4096 with a free list; a heap key refers to its node by slot and the
// low 32 bits of the node id (tag), and is stale once the slot no longer holds that node (keys are
// removed lazily: at the front, or by a rebuild once a heap holds 25% stale keys). Keys carry the
// whole order (key, depth, tag), so comparisons never read slots and the pop sequence is the total
// order (key, deeper first, smaller id) of the live nodes, whatever the heap layout (ids stay below
// 2^32). Compaction moves the live nodes to the front, frees the tail chunks and rebuilds the keys
// at exact size: below 1/8 occupancy, under memory pressure below 1/2.
//
// Drain mode (memory pressure): per-depth buckets of (slot, tag), LIFO within a depth, instead of
// the estimate heap (rebuilt when drain mode ends); pops alternate k deepest : 1 best bound. k
// adapts every 1024 pops (in a queue of fewer than 16384 nodes at drain entry every 1/16 of it, at
// least 64): when the open count fell by at least 1/16 of the window's pops, k halves (min 1), else
// it doubles (max 64); it starts at 8 and carries over between drain phases. Close to the hard
// stop (set_deep_only) every pop is deepest first. The pool's state evolves only through its
// calls, so in deterministic mode (all calls from the barrier thread) every decision is
// deterministic.
// Tree weight (estimation-driven restarts): every branching is binary with depth + 1, so the weights
// 2^-depth of the leaves sum to 1 when the tree is done. Leaves are counted per depth in atomic
// integers, so the weight does not depend on the order in which they close (deterministic mode reads
// the same weight at a barrier); weight() adds the counts deepest first. Depths from kMaxDepth - 1 on
// share the last count (their weight is below the double range anyway).
class TreeWeight {
 public:
  static constexpr int kMaxDepth = 1100;
  void close(int depth) {
    cnt_[static_cast<size_t>(std::clamp(depth, 0, kMaxDepth - 1))].fetch_add(1, std::memory_order_relaxed);
    leaves_.fetch_add(1, std::memory_order_relaxed);
  }
  double weight() const;
  long long leaves() const { return leaves_.load(std::memory_order_relaxed); }

 private:
  std::array<std::atomic<long long>, kMaxDepth> cnt_{};
  std::atomic<long long> leaves_{0};
};

// Process-wide: a pool destroyed after this point in time (steady clock, nanoseconds since its epoch) leaves its
// open nodes' memory to the operating system. Set only by the command-line program, to the end of its time
// limit: the process exits right after the solve, and freeing millions of nodes one by one first only delays
// the result (markshare2: 0.3 s per minute of a 4-thread search). Never set for library use.
inline std::atomic<long long> g_leak_pools_after_ns{std::numeric_limits<long long>::max()};

class NodePool {
 public:
  // path_acct: accounting of path segments the pool creates (an inline change it cannot pack).
  NodePool(int workers, BasisOwner* owner, std::atomic<long long>* path_acct = nullptr);
  ~NodePool();
  NodePool(const NodePool&) = delete;
  NodePool& operator=(const NodePool&) = delete;

  // Deterministic mode: one bound slot per virtual worker (nodes they keep between rounds).
  void resize_workers(int workers);
  void push(Node&& n);
  // Blocks until a node is available (returns true, worker marked active) or
  // the search is over (returns false). by_estimate: take the best-estimate node.
  bool pop(int w, const std::atomic<double>& cutoff, const std::atomic<bool>& stop, Node* out, bool by_estimate);
  // Deterministic mode: non-blocking pop without active-worker accounting.
  bool try_pop(double cutoff, Node* out, bool by_estimate);
  void set_worker_bound(int w, double b);
  void done(int w);
  double lower_bound();
  size_t size();
  size_t approx_size() const { return acount_.load(std::memory_order_relaxed); }  // without the lock

  // Drain mode: pops take the deepest node (with every (k+1)-th pop by best bound), so the tree is
  // worked off depth first and the number of open nodes stops growing.
  void set_drain(bool on);
  bool drain() const { return drain_; }
  // Bytes the queue grows by when drain mode ends (the estimate heap comes back, the buckets go).
  long long exit_growth() const;
  int drain_ratio() const { return drain_k_.load(std::memory_order_relaxed); }  // k of the k deepest : 1 best bound
  // Memory pressure (state >= 1): compaction already below 1/2 occupancy.
  void set_pressure(bool on);
  // Close to the hard stop: drain pops all deepest (no best-bound turns).
  void set_deep_only(bool on);
  bool deep_only() const { return deep_only_.load(std::memory_order_relaxed); }
  // True once after a compaction under memory pressure (the caller hands freed memory back to the OS).
  bool take_compacted() { return compacted_.exchange(false); }

  // Drops warm-start bases of queued nodes, shallowest (in drain mode: processed last) and then
  // worst-bound first, until done() says enough memory was freed. Returns the number dropped.
  template <class Done>
  long long strip_bases(Done&& done) {
    std::lock_guard<std::mutex> lk(mu_);
    // Counting sort of the slots holding a basis by depth (4 bytes per such node), then each
    // depth in (worst bound, slot) order as far as needed.
    std::vector<uint32_t> start;
    for (uint32_t s = 0; s < nslots_; ++s) {
      const QNode& q = slot(s);
      if (q.depth < 0 || !q.basis) continue;
      if (static_cast<size_t>(q.depth) + 2 > start.size()) start.resize(static_cast<size_t>(q.depth) + 2, 0);
      ++start[static_cast<size_t>(q.depth) + 1];
    }
    if (start.empty()) return 0;
    for (size_t d = 1; d < start.size(); ++d) start[d] += start[d - 1];
    std::vector<uint32_t> order(start.back());
    {
      std::vector<uint32_t> pos(start.begin(), start.end() - 1);
      for (uint32_t s = 0; s < nslots_; ++s) {
        const QNode& q = slot(s);
        if (q.depth >= 0 && q.basis) order[pos[static_cast<size_t>(q.depth)]++] = s;
      }
    }
    long long dropped = 0;
    for (size_t d = 0; d + 1 < start.size(); ++d) {
      if (start[d] == start[d + 1]) continue;
      if (done()) break;
      const auto b = order.begin() + start[d], e = order.begin() + start[d + 1];
      std::sort(b, e, [&](uint32_t x, uint32_t y) {
        const double bx = slot(x).bound, by = slot(y).bound;
        return bx != by ? bx > by : x < y;
      });
      bool enough = false;
      for (auto it = b; it != e; ++it) {
        if (done()) {
          enough = true;
          break;
        }
        QNode& q = slot(*it);
        BasisRef::drop(q.basis, owner_);
        q.basis = nullptr;
        ++dropped;
      }
      if (enough) break;
    }
    return dropped;
  }

  // Removes the queued nodes with bound >= cutoff (best-first selection never takes them; they
  // would be discarded when popped). Returns the number removed.
  long long purge(double cutoff);
  // Queued nodes the pool drops (purge, or all of them once the best bound reaches the cutoff) close
  // as leaves of this tree weight.
  void set_tree_weight(TreeWeight* tw) { tw_ = tw; }
  // Sum of 2^-depth over the queued nodes (diagnostics: with the closed weight it makes 1).
  double open_weight();

  // Diagnostics (verbosity 2 memory report): what the queued nodes hold, in one pass over the
  // slots (bases shared by queued siblings and the bound paths are accounted by the search).
  struct Breakdown {
    size_t nodes = 0;
    long long with_basis = 0, sparse_bases = 0, pool_rows = 0;  // queued nodes with a basis, sparse ones, their pool rows
    long long groups = 0, group_bytes = 0;  // symmetry groups (Node::mem)
    double depth = 0.0;                     // mean depth
  };
  Breakdown breakdown();

  // Memory of the queue itself (slots, keys, buckets, side table) plus what nodes declared in
  // Node::mem (symmetry groups); bases and bound paths are accounted where they are allocated.
  // bytes(): what the open nodes use (the memory-state decisions; free slots and spare array
  // capacity are reused as the queue grows again, and compaction keeps them bounded);
  // alloc_bytes(): what the queue has allocated (the hard stop).
  long long bytes() const { return live_bytes_.load(std::memory_order_relaxed); }
  long long alloc_bytes() const { return alloc_bytes_.load(std::memory_order_relaxed); }
  // Bytes per open node at least (QNode and its two keys): the deterministic memory projection.
  static constexpr long long kNodeBytes = 88 + 2 * 20;

  // Statistics.
  long long compactions() const { return compactions_; }
  long long purged() const { return purged_; }

 private:
  // Packed open node (88 bytes; every field exact). depth < 0: free slot. sym: index + 1 into the
  // side table of symmetry groups (0: none; Node::mem lives there too). pd: the parent as a signed
  // 30-bit difference id - parent (kParentNone: -1, kParentFar: not representable) and the
  // branching direction (2 bits; 3: direction 0 and no pseudocost column although the node has an
  // inline change). Inline change (Node::change): ilb != kNoInline; its column is branch_var, its
  // sides integral int32 values (kLbInf / kUbInf: infinite). A change that does not fit becomes
  // a path segment when the node is queued.
  struct QNode {
    double bound, estimate, parent_obj, branch_dist;
    const PathSeg* path;
    PackedBasis* basis;
    long long id;
    int32_t branch_var;
    int32_t depth;
    uint32_t sym;
    uint32_t pd;
    int32_t ilb, iub;
    int32_t anc_var;
    float anc_dist;
  };
  static_assert(sizeof(QNode) == 88, "queue node size");
  static constexpr int32_t kNoInline = INT32_MAX, kLbInf = INT32_MIN, kUbInf = INT32_MAX;
#pragma pack(push, 4)
  struct Key {
    double key;
    uint32_t slot;
    uint32_t tag;  // low 32 bits of the node id
    int32_t depth;
  };
#pragma pack(pop)
  static_assert(sizeof(Key) == 20, "heap key size");
  static_assert(kNodeBytes == static_cast<long long>(sizeof(QNode) + 2 * sizeof(Key)), "kNodeBytes");
  struct DEntry {
    uint32_t slot, tag;
  };
  struct SymEntry {
    std::shared_ptr<const PermGroup> g;
    long long mem = 0;
  };
  // Min-heap on the key; among equal keys prefer deeper nodes, then smaller ids.
  struct KeyWorse {
    bool operator()(const Key& a, const Key& b) const {
      if (a.key != b.key) return a.key > b.key;
      if (a.depth != b.depth) return a.depth < b.depth;
      return a.tag > b.tag;
    }
  };
  static constexpr int kChunkBits = 12;
  static constexpr uint32_t kChunk = 1u << kChunkBits;
  static constexpr int32_t kParentNone = -(1 << 29) + 1, kParentFar = -(1 << 29);

  QNode& slot(uint32_t s) { return chunks_[s >> kChunkBits][s & (kChunk - 1)]; }
  const QNode& slot(uint32_t s) const { return chunks_[s >> kChunkBits][s & (kChunk - 1)]; }
  bool live(uint32_t s, uint32_t tag) const {
    const QNode& q = slot(s);
    return q.depth >= 0 && static_cast<uint32_t>(q.id) == tag;
  }
  bool live(const Key& k) const { return live(k.slot, k.tag); }
  double est_key(const QNode& q) const;
  static uint32_t encode_pd(long long id, long long parent, int dir, bool no_pc);
  static void decode_pd(uint32_t pd, long long id, long long* parent, int* dir, bool* no_pc);

  uint32_t alloc_slot();
  void free_slot(uint32_t s);   // releases the slot's contents
  void pack(Node&& n, QNode& q);
  void unpack(uint32_t s, Node* out);
  void push_key(std::vector<Key>& h, const Key& k);
  void pop_key(std::vector<Key>& h);
  void drop_stale(std::vector<Key>& h);
  void rebuild(std::vector<Key>& h);
  void bucket_push(int depth, uint32_t s, uint32_t tag);
  void build_buckets();
  void clear_buckets();
  void rebuild_buckets();
  void build_eheap();
  bool pop_deepest(uint32_t* s);
  void pop_best(uint32_t* s);
  double min_bound();
  void take(bool by_estimate, Node* out);
  void maybe_compact();
  void compact();
  void clear();
  void close_all();  // clear(), the dropped nodes closed as leaves of tw_
  void update_bytes();

  BasisOwner* owner_;
  std::atomic<long long>* path_acct_;
  int threads_ = 1;  // the search's thread count: the destructor frees the open nodes on that many threads
  TreeWeight* tw_ = nullptr;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::unique_ptr<QNode[]>> chunks_;
  uint32_t nslots_ = 0;  // slots in use or on the free list (the rest of the last chunk is untouched)
  std::vector<uint32_t> free_;
  std::vector<Key> bheap_, eheap_;
  std::vector<std::vector<DEntry>> buckets_;  // drain mode: per depth
  size_t dentries_ = 0, dcap_ = 0;            // bucket entries (live or stale), their capacity
  int dmax_ = -1;                             // deepest bucket that may be non-empty
  std::vector<SymEntry> syms_;
  std::vector<uint32_t> sym_free_;
  bool drain_ = false, pressure_ = false;
  std::atomic<int> drain_k_{8};
  unsigned long long drain_turn_ = 0;
  std::atomic<bool> deep_only_{false};
  int win_pops_ = 0, win_len_ = 1024;
  size_t win_count0_ = 0;
  size_t count_ = 0;
  std::vector<double> wbound_;
  long long group_bytes_ = 0;  // Node::mem of the queued nodes
  std::atomic<long long> live_bytes_{0}, alloc_bytes_{0};
  std::atomic<size_t> acount_{0};
  std::atomic<bool> compacted_{false};
  long long compactions_ = 0, purged_ = 0;
  int active_ = 0;
};

}  // namespace bnb
}  // namespace milp
