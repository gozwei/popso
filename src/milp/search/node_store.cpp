#include "milp/search/node_store.hpp"

#include <cassert>
#include <cmath>
#include <thread>

namespace milp::bnb {

namespace {
size_t alloc16(size_t b) { return (b + 15) & ~size_t(15); }
}  // namespace

PackedBasis* PackedBasis::make(const Basis& b, Index size_in, const std::vector<VarStatus>* ref,
                               const std::vector<std::pair<int, VarStatus>>* pool_nb,
                               const std::vector<float>* weights) {
  thread_local std::vector<uint8_t> bits, pool;
  bits.clear();
  pool.clear();
  const Index size = size_in >= 0 ? size_in : static_cast<Index>(b.status.size());
  // Sparse form first (abandoned once it reaches the dense size).
  const size_t dense = (static_cast<size_t>(size) + 3) / 4;
  bool sparse = false;
  if (ref) {
    const std::vector<VarStatus>& r = *ref;
    Index last = -1;
    for (Index j = 0; j < size && bits.size() < dense; ++j) {
      const VarStatus rj = j < static_cast<Index>(r.size()) ? r[j] : VarStatus::Basic;
      if (b.status[j] == rj) continue;
      put_varint((static_cast<uint64_t>(j - last - 1) << 2) | static_cast<uint8_t>(b.status[j]), bits);
      last = j;
    }
    if (bits.size() < dense) sparse = true;
    else bits.clear();
  }
  if (!sparse) {
    bits.assign(dense, 0);
    for (Index j = 0; j < size; ++j) bits[j >> 2] |= static_cast<uint8_t>(static_cast<uint8_t>(b.status[j]) << ((j & 3) * 2));
  }
  int np = 0;
  if (pool_nb) {
    int last = -1;
    for (const auto& [id, st] : *pool_nb) {
      put_varint((static_cast<uint64_t>(id - last - 1) << 2) | static_cast<uint8_t>(st), pool);
      last = id;
    }
    np = static_cast<int>(pool_nb->size());
  }
  const bool with_w = weights && !weights->empty();
  const size_t wbytes = with_w ? sizeof(uint32_t) + weights->size() * sizeof(float) : 0;
  void* mem = ::operator new(sizeof(PackedBasis) + bits.size() + pool.size() + wbytes);
  PackedBasis* p = new (mem) PackedBasis();
  p->refs_.store(1, std::memory_order_relaxed);
  p->size_ = size;
  p->nbits_ = static_cast<uint32_t>(bits.size()) | (sparse ? kSparse : 0u) | (with_w ? kWeights : 0u);
  p->npool_ = np;
  if (!bits.empty()) std::memcpy(p->data(), bits.data(), bits.size());
  if (!pool.empty()) std::memcpy(p->data() + bits.size(), pool.data(), pool.size());
  if (with_w) {
    uint8_t* q = p->data() + bits.size() + pool.size();
    const uint32_t cnt = static_cast<uint32_t>(weights->size());
    std::memcpy(q, &cnt, sizeof cnt);
    std::memcpy(q + sizeof cnt, weights->data(), weights->size() * sizeof(float));
  }
  return p;
}

void PackedBasis::unpack(Basis& b, const std::vector<VarStatus>* ref) const {
  b.status.resize(size_);
  const uint8_t* q = data();
  const uint32_t nbits = status_bytes();
  if (!sparse()) {
    for (Index j = 0; j < size_; ++j) b.status[j] = static_cast<VarStatus>((q[j >> 2] >> ((j & 3) * 2)) & 3);
    return;
  }
  assert(ref != nullptr);
  const std::vector<VarStatus>& r = *ref;
  const Index nr = std::min<Index>(size_, static_cast<Index>(r.size()));
  std::copy(r.begin(), r.begin() + nr, b.status.begin());
  std::fill(b.status.begin() + nr, b.status.end(), VarStatus::Basic);
  const uint8_t* end = q + nbits;
  Index j = -1;
  while (q < end) {
    const uint64_t v = get_varint(q);
    j += static_cast<Index>(v >> 2) + 1;
    b.status[j] = static_cast<VarStatus>(v & 3);
  }
}

const uint8_t* PackedBasis::pool_end() const {
  const uint8_t* q = data() + status_bytes();
  for (int k = 0; k < npool_; ++k) get_varint(q);
  return q;
}

bool PackedBasis::weights(std::vector<float>& out) const {
  if (!has_weights()) return false;
  const uint8_t* q = pool_end();
  uint32_t cnt = 0;
  std::memcpy(&cnt, q, sizeof cnt);
  out.resize(cnt);
  std::memcpy(out.data(), q + sizeof cnt, static_cast<size_t>(cnt) * sizeof(float));
  return true;
}

long long PackedBasis::bytes() const {
  const uint8_t* q = pool_end();
  size_t len = static_cast<size_t>(q - data());
  if (has_weights()) {
    uint32_t cnt = 0;
    std::memcpy(&cnt, q, sizeof cnt);
    len += sizeof cnt + static_cast<size_t>(cnt) * sizeof(float);
  }
  return static_cast<long long>(alloc16(sizeof(PackedBasis) + len));
}

// ---------------------------------------------------------------------------------------------

NodePool::NodePool(int workers, BasisOwner* owner, std::atomic<long long>* path_acct)
    : owner_(owner), path_acct_(path_acct), threads_(std::max(1, workers)), wbound_(workers, kInf) {
  update_bytes();
}

// The open nodes' paths and bases: an allocation or two per node, scattered over the heap. Freed one by one on
// the calling thread this was 0.3 s per 7 million nodes of a 4-thread minute (markshare2), and 7-8 s past a
// 600 s limit on 10 threads. Chunks of slots are dealt out to the search's thread count (the reference counts
// are atomic; a shared ancestor segment is freed by whoever drops its last reference), and the owner's
// accounting of the bases is skipped: the pool is destroyed with its search.
NodePool::~NodePool() {
  if (count_ >= 50000 && std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch()).count() >=
                             g_leak_pools_after_ns.load(std::memory_order_relaxed)) {
    return;  // the command-line program past its time limit: see g_leak_pools_after_ns
  }
  const uint32_t nchunks = (nslots_ + kChunk - 1) >> kChunkBits;
  const uint32_t nt = count_ < 50000 ? 1u : std::min<uint32_t>({static_cast<uint32_t>(threads_), 16u, nchunks});
  std::atomic<uint32_t> next{0};
  auto run = [this, nchunks, &next] {
    for (uint32_t c = next.fetch_add(1); c < nchunks; c = next.fetch_add(1)) {
      const uint32_t end = std::min(nslots_, (c + 1) << kChunkBits);
      for (uint32_t s = c << kChunkBits; s < end; ++s) {
        QNode& q = slot(s);
        if (q.depth < 0) continue;
        PathSeg::release(q.path);
        BasisRef::drop(q.basis, nullptr);
      }
    }
  };
  std::vector<std::thread> helpers;
  try {
    for (uint32_t t = 1; t < nt; ++t) helpers.emplace_back(run);
  } catch (...) {  // no thread to be had: this one frees what the others do not take
  }
  run();
  for (auto& th : helpers) th.join();
}

void NodePool::resize_workers(int workers) {
  std::lock_guard<std::mutex> lk(mu_);
  wbound_.assign(std::max<size_t>(wbound_.size(), static_cast<size_t>(workers)), kInf);
}

// The "estimate" order is a hybrid: halfway between the node's bound and its pseudocost estimate
// (MILP_NODE_EST_W, 1 = pure estimate). Pure estimates sent eil33-2's alternate pops into subtrees
// whose bound was far from the global one (optimum found at 51-95 s instead of 34-67 s).
double NodePool::est_key(const QNode& q) const {
  static const double est_w = [] {
    const char* v = std::getenv("MILP_NODE_EST_W");
    return v ? std::atof(v) : 0.5;
  }();
  return is_finite(q.estimate) ? q.bound + est_w * (q.estimate - q.bound) : q.bound;
}

uint32_t NodePool::encode_pd(long long id, long long parent, int dir, bool no_pc) {
  int32_t f;
  if (parent == -1) {
    f = kParentNone;
  } else {
    const long long d = id - parent;
    f = d > kParentNone && d < (1 << 29) ? static_cast<int32_t>(d) : kParentFar;
  }
  const uint32_t dc = no_pc ? 3u : dir < 0 ? 1u : dir > 0 ? 2u : 0u;
  return (static_cast<uint32_t>(f) << 2) | dc;
}

void NodePool::decode_pd(uint32_t pd, long long id, long long* parent, int* dir, bool* no_pc) {
  const int32_t f = static_cast<int32_t>(pd) >> 2;  // arithmetic shift: sign-extends the 30-bit field
  // A parent too far away to encode (more than 2^29 ids apart) comes back as -2, which matches no
  // node: the node is then not taken for the continuation of its parent's LP (a cold start).
  *parent = f == kParentNone ? -1 : f == kParentFar ? -2 : id - f;
  const uint32_t dc = pd & 3u;
  *dir = dc == 1 ? -1 : dc == 2 ? 1 : 0;
  *no_pc = dc == 3;
}

uint32_t NodePool::alloc_slot() {
  if (!free_.empty()) {
    const uint32_t s = free_.back();
    free_.pop_back();
    return s;
  }
  const uint32_t s = nslots_++;
  if ((s >> kChunkBits) >= chunks_.size()) {
    chunks_.push_back(std::unique_ptr<QNode[]>(new QNode[kChunk]));  // not zeroed: pages are touched as slots are used
  }
  return s;
}

void NodePool::free_slot(uint32_t s) {
  QNode& q = slot(s);
  PathSeg::release(q.path);
  BasisRef::drop(q.basis, owner_);
  if (q.sym) {
    SymEntry& e = syms_[q.sym - 1];
    group_bytes_ -= e.mem;
    e.g.reset();
    e.mem = 0;
    sym_free_.push_back(q.sym - 1);
  }
  q.path = nullptr;
  q.basis = nullptr;
  q.sym = 0;
  q.depth = -1;
}

void NodePool::pack(Node&& n, QNode& q) {
  assert(n.depth >= 0);
  assert(!n.basis || n.basis.owner() == owner_);
  q.bound = n.bound;
  q.estimate = n.estimate;
  q.parent_obj = n.parent_obj;
  q.branch_dist = n.branch_dist;
  q.path = n.path.detach();
  q.basis = n.basis.detach();
  q.id = n.id;
  q.branch_var = n.branch_var;
  q.depth = n.depth;
  q.anc_var = n.anc_var;
  q.anc_dist = n.anc_dist;
  q.ilb = q.iub = kNoInline;
  bool no_pc = false;
  if (n.change.col >= 0) {
    const BoundChange& c = n.change;
    auto fits = [](double v) { return v == std::floor(v) && std::abs(v) < 2e9; };
    const bool vals = (c.lb == -kInf || fits(c.lb)) && (c.ub == kInf || fits(c.ub));
    const bool col_ok = n.branch_var == c.col || (n.branch_var == -1 && n.branch_dir == 0);
    if (vals && col_ok) {
      q.branch_var = c.col;
      no_pc = n.branch_var == -1;
      q.ilb = c.lb == -kInf ? kLbInf : static_cast<int32_t>(c.lb);
      q.iub = c.ub == kInf ? kUbInf : static_cast<int32_t>(c.ub);
    } else {
      // Not packable: the change becomes the newest path segment (same changes, same order).
      // Infinite sides stay out of the encoding, finite ones are kept (they differ from the original
      // bounds), so it decodes to exactly this change.
      std::vector<uint8_t> seg;
      PathSeg::encode(c, -kInf, kInf, seg);
      const PathSeg* p = PathSeg::make(q.path, seg, path_acct_);
      PathSeg::release(q.path);  // the new segment holds the parent reference
      q.path = p;
    }
  }
  q.pd = encode_pd(n.id, n.parent, n.branch_dir, no_pc);
  q.sym = 0;
  if (n.sym || n.mem != 0) {
    uint32_t e;
    if (!sym_free_.empty()) {
      e = sym_free_.back();
      sym_free_.pop_back();
    } else {
      e = static_cast<uint32_t>(syms_.size());
      syms_.emplace_back();
    }
    syms_[e].g = std::move(n.sym);
    syms_[e].mem = n.mem;
    group_bytes_ += n.mem;
    q.sym = e + 1;
  }
}

void NodePool::unpack(uint32_t s, Node* out) {
  QNode& q = slot(s);
  out->bound = q.bound;
  out->estimate = q.estimate;
  out->parent_obj = q.parent_obj;
  out->branch_dist = q.branch_dist;
  out->path = PathRef::adopt(q.path);
  out->basis = BasisRef(q.basis, owner_);
  out->id = q.id;
  out->branch_var = q.branch_var;
  out->depth = q.depth;
  out->anc_var = q.anc_var;
  out->anc_dist = q.anc_dist;
  bool no_pc = false;
  decode_pd(q.pd, q.id, &out->parent, &out->branch_dir, &no_pc);
  if (q.ilb != kNoInline) {
    out->change = BoundChange{q.branch_var, q.ilb == kLbInf ? -kInf : static_cast<double>(q.ilb),
                              q.iub == kUbInf ? kInf : static_cast<double>(q.iub)};
    if (no_pc) out->branch_var = -1;
  } else {
    out->change = BoundChange{-1, -kInf, kInf};
  }
  if (q.sym) {
    SymEntry& e = syms_[q.sym - 1];
    out->sym = std::move(e.g);
    out->mem = e.mem;
    group_bytes_ -= e.mem;
    e.g.reset();
    e.mem = 0;
    sym_free_.push_back(q.sym - 1);
  } else {
    out->sym.reset();
    out->mem = 0;
  }
  q.path = nullptr;
  q.basis = nullptr;
  q.sym = 0;
  q.depth = -1;
}

void NodePool::push_key(std::vector<Key>& h, const Key& k) {
  // Grow by 1.25x (not 2x: the arrays are a large part of the queue's memory).
  if (h.size() == h.capacity()) h.reserve(h.capacity() + std::max<size_t>(16, h.capacity() / 4));
  h.push_back(k);
  std::push_heap(h.begin(), h.end(), KeyWorse{});
}

void NodePool::pop_key(std::vector<Key>& h) {
  std::pop_heap(h.begin(), h.end(), KeyWorse{});
  h.pop_back();
}

void NodePool::drop_stale(std::vector<Key>& h) {
  while (!h.empty() && !live(h.front())) pop_key(h);
}

void NodePool::rebuild(std::vector<Key>& h) {
  h.erase(std::remove_if(h.begin(), h.end(), [&](const Key& k) { return !live(k); }), h.end());
  std::make_heap(h.begin(), h.end(), KeyWorse{});
}

void NodePool::bucket_push(int depth, uint32_t s, uint32_t tag) {
  if (static_cast<size_t>(depth) >= buckets_.size()) buckets_.resize(static_cast<size_t>(depth) + 1);
  std::vector<DEntry>& b = buckets_[static_cast<size_t>(depth)];
  const size_t cap0 = b.capacity();
  b.push_back({s, tag});
  dcap_ += b.capacity() - cap0;
  ++dentries_;
  dmax_ = std::max(dmax_, depth);
}

void NodePool::build_buckets() {
  clear_buckets();
  for (uint32_t s = 0; s < nslots_; ++s) {
    const QNode& q = slot(s);
    if (q.depth >= 0) bucket_push(q.depth, s, static_cast<uint32_t>(q.id));
  }
}

void NodePool::clear_buckets() {
  std::vector<std::vector<DEntry>>().swap(buckets_);
  dentries_ = dcap_ = 0;
  dmax_ = -1;
}

void NodePool::rebuild_buckets() {
  dentries_ = dcap_ = 0;
  dmax_ = -1;
  for (size_t d = 0; d < buckets_.size(); ++d) {
    std::vector<DEntry>& b = buckets_[d];
    b.erase(std::remove_if(b.begin(), b.end(), [&](const DEntry& e) { return !live(e.slot, e.tag); }), b.end());
    if (b.capacity() > 2 * b.size() + 16) b.shrink_to_fit();
    dentries_ += b.size();
    dcap_ += b.capacity();
    if (!b.empty()) dmax_ = static_cast<int>(d);
  }
}

bool NodePool::pop_deepest(uint32_t* s) {
  for (; dmax_ >= 0; --dmax_) {
    std::vector<DEntry>& b = buckets_[static_cast<size_t>(dmax_)];
    while (!b.empty()) {
      const DEntry e = b.back();
      b.pop_back();
      --dentries_;
      if (live(e.slot, e.tag)) {
        *s = e.slot;
        return true;
      }
    }
  }
  return false;
}

void NodePool::pop_best(uint32_t* s) {
  drop_stale(bheap_);
  *s = bheap_.front().slot;
  pop_key(bheap_);
}

double NodePool::min_bound() {
  drop_stale(bheap_);
  return bheap_.front().key;
}

bool NodePool::pop(int w, const std::atomic<double>& cutoff, const std::atomic<bool>& stop, Node* out, bool by_estimate) {
  std::unique_lock<std::mutex> lk(mu_);
  for (;;) {
    if (stop.load()) return false;
    if (count_ > 0 && min_bound() >= cutoff.load()) close_all();
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

bool NodePool::try_pop(double cutoff, Node* out, bool by_estimate) {
  std::lock_guard<std::mutex> lk(mu_);
  if (count_ > 0 && min_bound() >= cutoff) close_all();
  if (count_ == 0) return false;
  take(by_estimate, out);
  return true;
}

void NodePool::push(Node&& n) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    const uint32_t s = alloc_slot();
    QNode& q = slot(s);
    pack(std::move(n), q);
    ++count_;
    const uint32_t tag = static_cast<uint32_t>(q.id);
    push_key(bheap_, Key{q.bound, s, tag, q.depth});
    if (drain_) bucket_push(q.depth, s, tag);
    else push_key(eheap_, Key{est_key(q), s, tag, q.depth});
    update_bytes();
  }
  cv_.notify_one();
}

void NodePool::set_worker_bound(int w, double b) {
  std::lock_guard<std::mutex> lk(mu_);
  wbound_[w] = b;
}

void NodePool::done(int w) {
  std::lock_guard<std::mutex> lk(mu_);
  --active_;
  wbound_[w] = kInf;
  if (active_ == 0 && count_ == 0) cv_.notify_all();
}

double NodePool::lower_bound() {
  std::lock_guard<std::mutex> lk(mu_);
  double lb = count_ == 0 ? kInf : min_bound();
  for (double b : wbound_) lb = std::min(lb, b);
  return lb;
}

size_t NodePool::size() {
  std::lock_guard<std::mutex> lk(mu_);
  return count_;
}

void NodePool::set_drain(bool on) {
  std::lock_guard<std::mutex> lk(mu_);
  if (on == drain_) return;
  drain_ = on;
  // Drain mode keeps the depth buckets instead of the estimate heap (no estimate pops there).
  if (on) {
    std::vector<Key>().swap(eheap_);
    build_buckets();
    drain_turn_ = 0;
    win_pops_ = 0;
    win_count0_ = count_;
    // Adaptation window: 1024 pops, fewer in a small queue (a tiny cap needs the ratio at once).
    win_len_ = static_cast<int>(std::clamp<size_t>(count_ / 16, 64, 1024));
  } else {
    clear_buckets();
    build_eheap();
  }
  update_bytes();
}

void NodePool::build_eheap() {
  std::vector<Key>().swap(eheap_);
  eheap_.reserve(count_);
  for (uint32_t s = 0; s < nslots_; ++s) {
    const QNode& q = slot(s);
    if (q.depth >= 0) eheap_.push_back(Key{est_key(q), s, static_cast<uint32_t>(q.id), q.depth});
  }
  std::make_heap(eheap_.begin(), eheap_.end(), KeyWorse{});
}

long long NodePool::exit_growth() const {
  std::lock_guard<std::mutex> lk(mu_);
  return static_cast<long long>(count_ * sizeof(Key)) - static_cast<long long>(dentries_ * sizeof(DEntry));
}

void NodePool::set_pressure(bool on) {
  std::lock_guard<std::mutex> lk(mu_);
  pressure_ = on;
}

void NodePool::set_deep_only(bool on) {
  std::lock_guard<std::mutex> lk(mu_);
  deep_only_.store(on, std::memory_order_relaxed);
}

void NodePool::take(bool by_estimate, Node* out) {
  uint32_t s = 0;
  if (drain_) {
    // k deepest : 1 best bound.
    const unsigned long long k = static_cast<unsigned long long>(drain_k_.load(std::memory_order_relaxed));
    const bool deep = deep_only_.load(std::memory_order_relaxed) || drain_turn_++ % (k + 1) != k;
    if (!deep || !pop_deepest(&s)) pop_best(&s);
  } else {
    std::vector<Key>& h = by_estimate ? eheap_ : bheap_;
    drop_stale(h);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"  // false positive through the inlined heap code
#endif
    s = h.front().slot;
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
    pop_key(h);
  }
  unpack(s, out);
  free_.push_back(s);
  --count_;
  if (drain_ && ++win_pops_ >= win_len_) {
    // Draining fast enough (the open count fell by >= 1/16 of the pops): more best-bound pops.
    const long long fell = static_cast<long long>(win_count0_) - static_cast<long long>(count_);
    const int k = drain_k_.load(std::memory_order_relaxed);
    drain_k_.store(fell >= win_pops_ / 16 ? std::max(1, k / 2) : std::min(64, k * 2), std::memory_order_relaxed);
    win_pops_ = 0;
    win_count0_ = count_;
  }
  // Rebuild a heap (or the buckets) that is mostly stale keys.
  const size_t lim = count_ + count_ / 4 + 64;
  if (bheap_.size() > lim) rebuild(bheap_);
  if (!drain_ && eheap_.size() > lim) rebuild(eheap_);
  if (drain_ && dentries_ > lim) rebuild_buckets();
  maybe_compact();
  update_bytes();
}

void NodePool::maybe_compact() {
  // Give memory back once the queue has shrunk to a small part of its slots: below 1/8, under
  // memory pressure below 1/2 (at most once per halving: compaction leaves no free slot).
  if (nslots_ > kChunk && count_ < nslots_ / (pressure_ ? 2 : 8)) compact();
}

// Moves the live nodes to the front of the slots (free slots from the bottom take nodes from the
// top), frees the tail chunks and rebuilds the keys at exact size.
void NodePool::compact() {
  uint32_t lo = 0, hi = nslots_;
  for (;;) {
    while (lo < hi && slot(lo).depth >= 0) ++lo;
    while (hi > lo && slot(hi - 1).depth < 0) --hi;
    if (hi - lo < 2) break;  // [lo, hi) starts with a free slot and ends with a live one
    QNode& from = slot(hi - 1);
    slot(lo) = from;
    from.path = nullptr;
    from.basis = nullptr;
    from.sym = 0;
    from.depth = -1;
    ++lo;
    --hi;
  }
  nslots_ = static_cast<uint32_t>(count_);
  chunks_.resize((nslots_ + kChunk - 1) >> kChunkBits);
  chunks_.shrink_to_fit();
  std::vector<uint32_t>().swap(free_);
  std::vector<Key>().swap(bheap_);
  bheap_.reserve(count_);
  for (uint32_t s = 0; s < nslots_; ++s) {
    const QNode& q = slot(s);
    bheap_.push_back(Key{q.bound, s, static_cast<uint32_t>(q.id), q.depth});
  }
  std::make_heap(bheap_.begin(), bheap_.end(), KeyWorse{});
  if (drain_) build_buckets();
  else build_eheap();
  // The side table of symmetry groups, renumbered in slot order.
  if (!syms_.empty()) {
    std::vector<SymEntry> fresh;
    fresh.reserve(syms_.size() - sym_free_.size());
    for (uint32_t s = 0; s < nslots_; ++s) {
      QNode& q = slot(s);
      if (!q.sym) continue;
      fresh.push_back(std::move(syms_[q.sym - 1]));
      q.sym = static_cast<uint32_t>(fresh.size());
    }
    syms_.swap(fresh);
    std::vector<uint32_t>().swap(sym_free_);
  }
  ++compactions_;
  if (pressure_) compacted_ = true;
}

long long NodePool::purge(double cutoff) {
  std::lock_guard<std::mutex> lk(mu_);
  long long removed = 0;
  for (uint32_t s = 0; s < nslots_; ++s) {
    QNode& q = slot(s);
    if (q.depth < 0 || q.bound < cutoff) continue;
    if (tw_) tw_->close(q.depth);
    free_slot(s);
    free_.push_back(s);
    --count_;
    ++removed;
  }
  if (removed > 0) {
    purged_ += removed;
    const size_t lim = count_ + count_ / 4 + 64;
    if (bheap_.size() > lim) rebuild(bheap_);
    if (!drain_ && eheap_.size() > lim) rebuild(eheap_);
    if (drain_ && dentries_ > lim) rebuild_buckets();
    maybe_compact();
    update_bytes();
  }
  return removed;
}

void NodePool::close_all() {
  if (tw_) {
    for (uint32_t s = 0; s < nslots_; ++s) {
      if (slot(s).depth >= 0) tw_->close(slot(s).depth);
    }
  }
  clear();
}

double NodePool::open_weight() {
  std::lock_guard<std::mutex> lk(mu_);
  double w = 0.0;
  for (uint32_t s = 0; s < nslots_; ++s) {
    if (slot(s).depth >= 0) w += std::ldexp(1.0, -slot(s).depth);
  }
  return w;
}

double TreeWeight::weight() const {
  double w = 0.0;
  for (int d = kMaxDepth - 1; d >= 0; --d) {
    const long long c = cnt_[static_cast<size_t>(d)].load(std::memory_order_relaxed);
    if (c != 0) w += std::ldexp(static_cast<double>(c), -d);
  }
  return w;
}

void NodePool::clear() {
  for (uint32_t s = 0; s < nslots_; ++s) {
    if (slot(s).depth >= 0) free_slot(s);
  }
  std::vector<std::unique_ptr<QNode[]>>().swap(chunks_);
  nslots_ = 0;
  std::vector<uint32_t>().swap(free_);
  std::vector<Key>().swap(bheap_);
  std::vector<Key>().swap(eheap_);
  if (drain_) clear_buckets();
  std::vector<SymEntry>().swap(syms_);
  std::vector<uint32_t>().swap(sym_free_);
  count_ = 0;
  group_bytes_ = 0;
  update_bytes();
}

NodePool::Breakdown NodePool::breakdown() {
  Breakdown b;
  std::lock_guard<std::mutex> lk(mu_);
  for (uint32_t s = 0; s < nslots_; ++s) {
    const QNode& q = slot(s);
    if (q.depth < 0) continue;
    ++b.nodes;
    b.depth += q.depth;
    if (q.basis) {
      ++b.with_basis;
      b.sparse_bases += q.basis->sparse() ? 1 : 0;
      b.pool_rows += q.basis->npool();
    }
    if (q.sym && syms_[q.sym - 1].g) ++b.groups;
  }
  b.group_bytes = group_bytes_;
  if (b.nodes > 0) b.depth /= static_cast<double>(b.nodes);
  return b;
}

void NodePool::update_bytes() {
  acount_.store(count_, std::memory_order_relaxed);
  const long long key = static_cast<long long>(sizeof(Key)), ent = static_cast<long long>(sizeof(DEntry));
  const long long live_syms = static_cast<long long>(syms_.size() - sym_free_.size());
  live_bytes_.store(static_cast<long long>(count_) * static_cast<long long>(sizeof(QNode)) +
                        static_cast<long long>(bheap_.size() + eheap_.size()) * key +
                        static_cast<long long>(dentries_) * ent + live_syms * static_cast<long long>(sizeof(SymEntry)) +
                        group_bytes_,
                    std::memory_order_relaxed);
  // Slots handed out so far (the untouched rest of the last chunk is not resident: not zeroed).
  alloc_bytes_.store(static_cast<long long>(nslots_) * static_cast<long long>(sizeof(QNode)) +
                         static_cast<long long>(bheap_.capacity() + eheap_.capacity()) * key +
                         static_cast<long long>(dcap_) * ent +
                         static_cast<long long>(buckets_.capacity() * sizeof(std::vector<DEntry>)) +
                         static_cast<long long>(free_.capacity() * sizeof(uint32_t)) +
                         static_cast<long long>(syms_.capacity() * sizeof(SymEntry) + sym_free_.capacity() * sizeof(uint32_t)) +
                         group_bytes_,
                     std::memory_order_relaxed);
}

}  // namespace milp::bnb
