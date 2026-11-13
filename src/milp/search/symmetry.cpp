#include "milp/search/symmetry.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <tuple>

#include "milp/core/interrupt.hpp"
#include "milp/core/timer.hpp"

namespace milp {

namespace {

uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

uint64_t hash_value(double v) {
  if (v == 0.0) v = 0.0;  // -0 == +0
  uint64_t b;
  std::memcpy(&b, &v, sizeof b);
  return mix64(b);
}

// Vertices 0..n-1 are columns, n..n+m-1 rows. The ordered partition keeps
// column vertices at positions [0, n) (their initial cells never mix with rows).
class Detector {
 public:
  Detector(const MipProblem& p, const SymmetryOptions& o) : p_(p), o_(o) {}
  void run(Symmetry& out);

 private:
  void search();
  struct Level {
    Index target;  // start of the individualized cell (before individualization)
    Index size;    // its size
    Index v;       // the vertex individualized on the first path
    size_t mark;   // trail size before individualization
    uint64_t trace;
    Index cells;
  };
  struct Split {
    Index start, end, pieces;
  };

  void build();
  uint64_t refine();
  void split_cell(Index c, size_t k0, size_t k1, uint64_t& tr);
  void individualize(Index v);
  void undo_to(size_t mark);
  // First non-singleton cell of integer columns, else of continuous ones.
  // Called on the first path only, where cells only split: a singleton stays one and a cell keeps its column
  // type, so both scans resume where the last call stopped (from 0 each time it was a pass over all cells per
  // level: supportcase12 without presolve, 800k columns, 26 s that no budget counted).
  Index target() {
    while (tgt_int_ < n_ && (cend_[tgt_int_] - tgt_int_ <= 1 || !p_.is_int[lab_[tgt_int_]])) tgt_int_ = cend_[tgt_int_];
    if (tgt_int_ < n_) return tgt_int_;
    while (tgt_cont_ < n_ && cend_[tgt_cont_] - tgt_cont_ <= 1) tgt_cont_ = cend_[tgt_cont_];
    return tgt_cont_ < n_ ? tgt_cont_ : -1;
  }
  Index tgt_int_ = 0, tgt_cont_ = 0;
  void enqueue(Index s) {
    if (!inq_[s]) {
      inq_[s] = 1;
      queue_.push_back(s);
    }
  }
  bool key_less(Index a, Index b) const { return cnt_[a] != cnt_[b] ? cnt_[a] < cnt_[b] : sig_[a] < sig_[b]; }
  bool key_eq(Index a, Index b) const { return cnt_[a] == cnt_[b] && sig_[a] == sig_[b]; }
  bool descend(size_t t);
  bool check_leaf();
  bool verify();
  bool check_col(Index j);
  Index find(Index j) {
    while (uf_[j] != j) j = uf_[j] = uf_[uf_[j]];
    return j;
  }

  const MipProblem& p_;
  SymmetryOptions o_;
  Timer timer_;
  Index n_ = 0, m_ = 0, nv_ = 0;
  std::vector<Index> astart_, adj_;
  std::vector<uint64_t> aw_;
  // Ordered partition: lab_[position] = vertex, cstart_[position] = start of
  // its cell, cend_[cell start] = end of the cell.
  std::vector<Index> lab_, pos_, cstart_, cend_;
  Index cells_ = 0;
  std::vector<Split> trail_;
  std::vector<uint64_t> sig_;
  std::vector<Index> cnt_, tv_, tc_, slots_, pieces_, queue_;
  std::vector<char> ctouch_, inq_;
  long long work_ = 0, work_limit_ = 0, try_nodes_ = 0;
  bool aborted_ = false;
  long long next_clock_ = 0;  // work_ at the next look at the clock
  std::vector<Level> levels_;
  std::vector<Index> leaf_, gc_, gr_, uf_, failed_;
  std::vector<double> rowv_;
  std::vector<int> rowst_, colst_;
  int rstamp_ = 0, cstamp_ = 0;
  std::vector<std::vector<std::pair<Index, Index>>> gens_;
  long long support_ = 0;
  int cur_level_ = 0;
  std::vector<int> gen_level_;
};

void Detector::build() {
  n_ = p_.num_cols();
  m_ = p_.num_rows();
  nv_ = n_ + m_;
  const SparseMatrix& A = p_.A;
  astart_.assign(nv_ + 1, 0);
  for (Index j = 0; j < n_; ++j) {
    astart_[j + 1] = A.end(j) - A.begin(j);
    for (Index k = A.begin(j); k < A.end(j); ++k) ++astart_[n_ + A.index[k] + 1];
  }
  for (Index v = 0; v < nv_; ++v) astart_[v + 1] += astart_[v];
  adj_.resize(astart_[nv_]);
  aw_.resize(astart_[nv_]);
  std::vector<Index> fill(astart_.begin(), astart_.end() - 1);
  for (Index j = 0; j < n_; ++j) {
    for (Index k = A.begin(j); k < A.end(j); ++k) {
      const Index r = n_ + A.index[k];
      const uint64_t h = hash_value(A.value[k]);
      adj_[fill[j]] = r;
      aw_[fill[j]++] = h;
      adj_[fill[r]] = j;
      aw_[fill[r]++] = h;
    }
  }
  // Initial cells: columns by (type, cost, bounds), then rows by sides.
  std::vector<Index> cols(n_), rows(m_);
  for (Index j = 0; j < n_; ++j) cols[j] = j;
  for (Index r = 0; r < m_; ++r) rows[r] = r;
  auto ckey = [&](Index j) { return std::make_tuple(p_.is_int[j], p_.cost[j], p_.col_lb[j], p_.col_ub[j]); };
  auto rkey = [&](Index r) { return std::make_tuple(p_.row_lb[r], p_.row_ub[r]); };
  std::sort(cols.begin(), cols.end(), [&](Index a, Index b) { return ckey(a) < ckey(b); });
  std::sort(rows.begin(), rows.end(), [&](Index a, Index b) { return rkey(a) < rkey(b); });
  lab_.resize(nv_);
  pos_.resize(nv_);
  cstart_.resize(nv_);
  cend_.assign(nv_ + 1, 0);
  for (Index j = 0; j < n_; ++j) lab_[j] = cols[j];
  for (Index r = 0; r < m_; ++r) lab_[n_ + r] = n_ + rows[r];
  inq_.assign(nv_ + 1, 0);
  Index s = 0;
  for (Index q = 0; q <= nv_; ++q) {
    bool boundary = q == nv_ || q == n_;
    if (!boundary && q > s) {
      boundary = q < n_ ? ckey(lab_[q]) != ckey(lab_[q - 1]) : rkey(lab_[q] - n_) != rkey(lab_[q - 1] - n_);
    }
    if (boundary && q > s) {
      cend_[s] = q;
      ++cells_;
      enqueue(s);
      s = q;
    }
    if (q < nv_) {
      pos_[lab_[q]] = q;
      cstart_[q] = s;
    }
  }
  sig_.assign(nv_, 0);
  cnt_.assign(nv_, 0);
  ctouch_.assign(nv_, 0);
  work_ += 4LL * (nv_ + astart_[nv_]);
}

void Detector::split_cell(Index c, size_t k0, size_t k1, uint64_t& tr) {
  const Index e = cend_[c], size = e - c, t = static_cast<Index>(k1 - k0);
  if (size == 1 || (t == size && key_eq(tv_[k0], tv_[k1 - 1]))) return;
  // Touched vertices move to the tail [e - t, e) in key order; the untouched
  // ones found there take the slots they leave in front.
  const Index tail = e - t;
  slots_.clear();
  for (size_t k = k0; k < k1; ++k) {
    if (pos_[tv_[k]] < tail) slots_.push_back(pos_[tv_[k]]);
  }
  size_t si = 0;
  for (Index q = tail; q < e; ++q) {
    const Index x = lab_[q];
    if (cnt_[x] == 0) {
      const Index d = slots_[si++];
      lab_[d] = x;
      pos_[x] = d;
    }
  }
  pieces_.clear();
  if (t < size) pieces_.push_back(c);
  for (size_t k = k0; k < k1; ++k) {
    const Index q = tail + static_cast<Index>(k - k0);
    lab_[q] = tv_[k];
    pos_[tv_[k]] = q;
    if (k == k0 || !key_eq(tv_[k], tv_[k - 1])) pieces_.push_back(q);
  }
  const Index np = static_cast<Index>(pieces_.size());
  trail_.push_back({c, e, np});
  cells_ += np - 1;
  Index largest = 0;
  for (Index i = 0; i < np; ++i) {
    const Index ps = pieces_[i], pe = i + 1 < np ? pieces_[i + 1] : e;
    cend_[ps] = pe;
    if (i > 0) {
      for (Index q = ps; q < pe; ++q) cstart_[q] = ps;
    }
    if (pe - ps > cend_[pieces_[largest]] - pieces_[largest]) largest = i;
    const bool untouched = i == 0 && t < size;
    const uint64_t key = untouched ? 0 : static_cast<uint64_t>(cnt_[lab_[ps]]) * 0x100000001b3ULL + sig_[lab_[ps]];
    tr = mix64(tr + static_cast<uint64_t>(ps) * 0x9e3779b1ULL + static_cast<uint64_t>(pe - ps) * 0x85ebca6bULL + key);
  }
  work_ += t + 2;
  // Hopcroft: a cell already queued gets all new pieces queued; otherwise
  // all but the largest suffice.
  const bool queued = inq_[c] != 0;
  for (Index i = 0; i < np; ++i) {
    if (queued ? i > 0 : i != largest) enqueue(pieces_[i]);
  }
}

uint64_t Detector::refine() {
  uint64_t tr = 0x2545f4914f6cdd1dULL;
  size_t qh = 0;
  while (qh < queue_.size()) {
    const Index s = queue_[qh++];
    inq_[s] = 0;
    if (aborted_) continue;
    const Index e = cend_[s];
    for (Index q = s; q < e; ++q) {
      const Index v = lab_[q];
      for (Index k = astart_[v]; k < astart_[v + 1]; ++k) {
        const Index u = adj_[k];
        if (cnt_[u]++ == 0) {
          tv_.push_back(u);
          const Index c = cstart_[pos_[u]];
          if (!ctouch_[c]) {
            ctouch_[c] = 1;
            tc_.push_back(c);
          }
        }
        sig_[u] += aw_[k];
      }
      work_ += astart_[v + 1] - astart_[v] + 1;
    }
    std::sort(tc_.begin(), tc_.end());
    std::sort(tv_.begin(), tv_.end(), [&](Index a, Index b) {
      const Index ca = cstart_[pos_[a]], cb = cstart_[pos_[b]];
      return ca != cb ? ca < cb : key_less(a, b);
    });
    work_ += 3 * static_cast<long long>(tv_.size()) + static_cast<long long>(tc_.size());
    size_t k0 = 0;
    for (Index c : tc_) {
      size_t k1 = k0;
      while (k1 < tv_.size() && cstart_[pos_[tv_[k1]]] == c) ++k1;
      split_cell(c, k0, k1, tr);
      k0 = k1;
    }
    for (Index u : tv_) {
      cnt_[u] = 0;
      sig_[u] = 0;
    }
    for (Index c : tc_) ctouch_[c] = 0;
    tv_.clear();
    tc_.clear();
    // The clock by work, not by cells: a few huge cells are few queue entries (supportcase12 without presolve,
    // 800k columns: 27 s at a 1 s cap, past the solve's time limit).
    if (work_ > work_limit_) aborted_ = true;
    if (work_ >= next_clock_) {
      next_clock_ = work_ + 1000000;
      if (timer_.elapsed() > o_.time_limit || interrupted()) aborted_ = true;
    }
  }
  queue_.clear();
  return mix64(tr + static_cast<uint64_t>(cells_));
}

void Detector::individualize(Index v) {
  const Index c = cstart_[pos_[v]], e = cend_[c];
  const Index q = pos_[v], x = lab_[c];
  lab_[c] = v;
  pos_[v] = c;
  lab_[q] = x;
  pos_[x] = q;
  trail_.push_back({c, e, 2});
  ++cells_;
  cend_[c] = c + 1;
  cend_[c + 1] = e;
  for (Index p = c + 1; p < e; ++p) cstart_[p] = c + 1;
  work_ += e - c;
  enqueue(c);
}

void Detector::undo_to(size_t mark) {
  while (trail_.size() > mark) {
    const Split s = trail_.back();
    trail_.pop_back();
    const Index ne = cend_[s.start];
    for (Index q = ne; q < s.end; ++q) cstart_[q] = s.start;
    cend_[s.start] = s.end;
    cells_ -= s.pieces - 1;
    work_ += s.end - ne;
  }
}

bool Detector::check_col(Index j) {
  if (colst_[j] == cstamp_) return true;
  colst_[j] = cstamp_;
  const Index k = gc_[j];
  const SparseMatrix& A = p_.A;
  if (p_.cost[j] != p_.cost[k] || p_.col_lb[j] != p_.col_lb[k] || p_.col_ub[j] != p_.col_ub[k] ||
      p_.is_int[j] != p_.is_int[k] || A.end(j) - A.begin(j) != A.end(k) - A.begin(k)) {
    return false;
  }
  ++rstamp_;
  for (Index e = A.begin(k); e < A.end(k); ++e) {
    rowv_[A.index[e]] = A.value[e];
    rowst_[A.index[e]] = rstamp_;
  }
  work_ += 2 * (A.end(j) - A.begin(j)) + 1;
  for (Index e = A.begin(j); e < A.end(j); ++e) {
    const Index r = gr_[A.index[e]];
    if (rowst_[r] != rstamp_ || rowv_[r] != A.value[e]) return false;
  }
  return true;
}

// Exact check that (gc_, gr_) maps the problem onto itself. Only columns that
// move or meet a moved row can be affected.
bool Detector::verify() {
  ++cstamp_;
  work_ += nv_;
  for (Index j = 0; j < n_; ++j) {
    if (gc_[j] != j && !check_col(j)) return false;
  }
  for (Index r = 0; r < m_; ++r) {
    if (gr_[r] == r) continue;
    if (p_.row_lb[r] != p_.row_lb[gr_[r]] || p_.row_ub[r] != p_.row_ub[gr_[r]]) return false;
    for (Index k = astart_[n_ + r]; k < astart_[n_ + r + 1]; ++k) {
      if (!check_col(adj_[k])) return false;
    }
  }
  return true;
}

bool Detector::check_leaf() {
  for (Index q = 0; q < nv_; ++q) {
    const Index a = leaf_[q], b = lab_[q];
    if (a < n_) gc_[a] = b;
    else gr_[a - n_] = b - n_;
  }
  if (!verify()) return false;
  std::vector<std::pair<Index, Index>> g;
  bool has_int = false;
  for (Index j = 0; j < n_; ++j) {
    if (gc_[j] == j) continue;
    g.emplace_back(j, gc_[j]);
    has_int |= p_.is_int[j] != 0;
    const Index a = find(j), b = find(gc_[j]);
    if (a != b) uf_[a] = b;
  }
  // Generators moving only continuous columns still prune the search but are
  // not kept (branching happens on integer columns only; integer base points
  // come first, so such generators only occur at the deepest levels).
  if (has_int) {
    support_ += static_cast<long long>(g.size());
    gens_.push_back(std::move(g));
    gen_level_.push_back(cur_level_);
  }
  return true;
}

bool Detector::descend(size_t t) {
  if (t == levels_.size()) return check_leaf();
  const Level& L = levels_[t];
  const Index s = L.target;
  if (cend_[s] - s != L.size) return false;
  const std::vector<Index> cell(lab_.begin() + s, lab_.begin() + cend_[s]);
  const size_t mark = trail_.size();
  for (Index w : cell) {
    if (aborted_ || try_nodes_ <= 0) return false;
    --try_nodes_;
    individualize(w);
    const uint64_t tr = refine();
    if (!aborted_ && tr == L.trace && cells_ == L.cells && descend(t + 1)) return true;
    undo_to(mark);
  }
  return false;
}

void Detector::run(Symmetry& out) {
  search();
  out.gens = std::move(gens_);
  out.gen_level = std::move(gen_level_);
  for (const Level& L : levels_) out.base.push_back(L.v);
  out.work = work_;
  out.aborted = aborted_;
  out.mov_start.assign(n_ + 1, 0);
  for (const auto& g : out.gens) {
    for (const auto& [j, k] : g) ++out.mov_start[j + 1];
  }
  for (Index j = 0; j < n_; ++j) {
    out.moved_cols += out.mov_start[j + 1] > 0;
    out.mov_start[j + 1] += out.mov_start[j];
  }
  out.movers.resize(out.mov_start[n_]);
  std::vector<Index> fill(out.mov_start.begin(), out.mov_start.end() - 1);
  for (int g = 0; g < out.num_generators(); ++g) {
    for (const auto& [j, k] : out.gens[g]) out.movers[fill[j]++] = {g, k};
  }
  out.build_chain(n_);
  out.seconds = timer_.elapsed();
}

void Detector::search() {
  build();
  const long long size = static_cast<long long>(nv_) + astart_[nv_];
  work_limit_ = o_.work_limit > 0 ? o_.work_limit : std::min<long long>(60 * size + 30000000LL, 400000000LL);
  refine();
  if (aborted_ || target() < 0) return;
  // First path: individualize the first vertex of the first non-singleton
  // column cell until all columns are singletons.
  for (Index t; !aborted_ && (t = target()) >= 0;) {
    Level L{t, cend_[t] - t, lab_[t], trail_.size(), 0, 0};
    individualize(L.v);
    L.trace = refine();
    L.cells = cells_;
    levels_.push_back(L);
  }
  if (aborted_) return;
  leaf_ = lab_;
  gc_.resize(n_);
  gr_.resize(m_);
  rowv_.assign(m_, 0.0);
  rowst_.assign(m_, 0);
  colst_.assign(n_, 0);
  uf_.resize(n_);
  for (Index j = 0; j < n_; ++j) uf_[j] = j;
  failed_.assign(n_, 0);
  // Generators bottom-up: at level l, try to map v_l to each vertex u of its
  // cell that is not yet known to be in v_l's orbit. All generators found so
  // far fix v_0..v_{l-1}, so their orbits are orbits of the level-l stabilizer.
  for (size_t l = levels_.size(); l-- > 0 && !aborted_;) {
    undo_to(levels_[l].mark);
    const Level L = levels_[l];
    const std::vector<Index> cell(lab_.begin() + L.target, lab_.begin() + L.target + L.size);
    for (Index u : cell) {
      if (aborted_ || static_cast<int>(gens_.size()) >= o_.max_generators || support_ >= o_.max_support) break;
      if (u == L.v || find(u) == find(L.v) || failed_[find(u)] == static_cast<Index>(l + 1)) continue;
      try_nodes_ = 256;
      cur_level_ = static_cast<int>(l);
      individualize(u);
      const uint64_t tr = refine();
      const bool ok = !aborted_ && tr == L.trace && cells_ == L.cells && descend(l + 1);
      if (!ok) failed_[find(u)] = static_cast<Index>(l + 1);
      undo_to(L.mark);
    }
  }
}

}  // namespace

void Symmetry::build_chain(Index n) {
  const int G = num_generators();
  if (G == 0) return;
  // Schreier tree of base[l] under the generators of levels >= l, for each
  // level that has generators of its own.
  const int L = static_cast<int>(base.size());
  tree.assign(L, {});
  std::vector<char> at(L, 0);
  for (int g = 0; g < G; ++g) at[gen_level[g]] = 1;
  std::vector<Index> parent(n, -1), queue;
  std::vector<int> via(n, -1);
  for (int l = 0; l < L; ++l) {
    if (!at[l]) continue;
    const Index b = base[l];
    queue.assign(1, b);
    parent[b] = b;
    for (size_t h = 0; h < queue.size(); ++h) {
      const Index x = queue[h];
      for (Index k = mov_start[x]; k < mov_start[x + 1]; ++k) {
        const auto [g, z] = movers[k];
        if (gen_level[g] < l || parent[z] >= 0) continue;
        parent[z] = x;
        via[z] = g;
        queue.push_back(z);
      }
    }
    for (Index x : queue) tree[l].push_back({x, parent[x], x == b ? -1 : via[x]});
    std::sort(tree[l].begin(), tree[l].end(), [](const TreeEntry& a, const TreeEntry& c) { return a.point < c.point; });
    for (Index x : queue) parent[x] = -1, via[x] = -1;
  }
}

double Symmetry::log10_order() const {
  double o = 0.0;
  for (const auto& t : tree) o += t.empty() ? 0.0 : std::log10(static_cast<double>(t.size()));
  return o;
}

bool Symmetry::transversal(int l, Index y, std::vector<Index>& t) const {
  if (l >= static_cast<int>(tree.size()) || tree[l].empty()) return false;
  auto find = [&](Index x) {
    auto it = std::lower_bound(tree[l].begin(), tree[l].end(), x,
                               [](const TreeEntry& e, Index v) { return e.point < v; });
    return it != tree[l].end() && it->point == x ? &*it : nullptr;
  };
  const TreeEntry* e = find(y);
  if (!e) return false;
  // y = s_k(...s_1(base)): collect s_k..s_1, then compose t = s_k o ... o s_1.
  std::vector<int> path;
  while (e->gen >= 0) {
    path.push_back(e->gen);
    e = find(e->parent);
  }
  const Index n = static_cast<Index>(mov_start.size()) - 1;
  t.resize(n);
  for (Index j = 0; j < n; ++j) t[j] = j;
  std::vector<Index> s(n);
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    for (Index j = 0; j < n; ++j) s[j] = j;
    for (const auto& [a, b] : gens[*it]) s[a] = b;
    for (Index j = 0; j < n; ++j) t[j] = s[t[j]];
  }
  return true;
}

Index PermGroup::image(int g, Index j) const {
  const auto& v = gens[g];
  auto it = std::lower_bound(v.begin(), v.end(), j, [](const std::pair<Index, Index>& e, Index c) { return e.first < c; });
  return it != v.end() && it->first == j ? it->second : j;
}

void PermGroup::orbit(Index j, std::vector<Index>& out, std::vector<char>& mark) const {
  out.assign(1, j);
  mark[j] = 1;
  for (size_t h = 0; h < out.size(); ++h) {
    for (int g = 0; g < static_cast<int>(gens.size()); ++g) {
      const Index z = image(g, out[h]);
      if (!mark[z]) {
        mark[z] = 1;
        out.push_back(z);
      }
    }
  }
  for (Index c : out) mark[c] = 0;
}

long long PermGroup::bytes() const {
  long long b = 64;
  for (const auto& g : gens) b += 32 + static_cast<long long>(g.size() * sizeof(g[0]));
  return b;
}

PermGroup PermGroup::stabilizer(Index j, Index n, uint64_t seed, GroupScratch& s, int max_kept, int max_new) const {
  PermGroup out;
  std::vector<int> tg;  // generators moving j: they span the Schreier tree
  for (int g = 0; g < static_cast<int>(gens.size()); ++g) {
    if (image(g, j) == j) {
      if (static_cast<int>(out.gens.size()) < max_kept) out.gens.push_back(gens[g]);
    } else if (tg.size() < 32) {
      tg.push_back(g);
    }
  }
  if (tg.empty()) return out;
  const int T = static_cast<int>(tg.size());
  if (static_cast<int>(s.dense.size()) < T) s.dense.resize(T);
  for (int k = 0; k < T; ++k) {
    s.dense[k].resize(n);
    for (Index c = 0; c < n; ++c) s.dense[k][c] = c;
    for (const auto& [a, b] : gens[tg[k]]) s.dense[k][a] = b;
  }
  s.parent.assign(n, -1);
  s.via.assign(n, -1);
  s.orbit.assign(1, j);
  s.parent[j] = j;
  for (size_t h = 0; h < s.orbit.size(); ++h) {
    const Index x = s.orbit[h];
    for (int k = 0; k < T; ++k) {
      const Index z = s.dense[k][x];
      if (s.parent[z] >= 0) continue;
      s.parent[z] = x;
      s.via[z] = k;
      s.orbit.push_back(z);
    }
  }
  // u_x maps j to x: the product of the tree generators on the path.
  auto word = [&](Index x, std::vector<Index>& u) {
    u.resize(n);
    for (Index c = 0; c < n; ++c) u[c] = c;
    std::vector<int> path;
    for (; x != j; x = s.parent[x]) path.push_back(s.via[x]);
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      const std::vector<Index>& d = s.dense[*it];
      for (Index c = 0; c < n; ++c) u[c] = d[u[c]];
    }
  };
  uint64_t rng = mix64(seed);
  auto next = [&]() {
    rng = mix64(rng);
    return rng;
  };
  int added = 0;
  for (int a = 0; a < 3 * max_new && added < max_new; ++a) {
    const Index x = s.orbit[next() % s.orbit.size()];
    const int k = static_cast<int>(next() % static_cast<uint64_t>(T));
    const Index z = s.dense[k][x];
    word(x, s.ux);
    word(z, s.uz);
    s.uzinv.resize(n);
    for (Index c = 0; c < n; ++c) s.uzinv[s.uz[c]] = c;
    // g = u_z^-1 o s_k o u_x fixes j.
    std::vector<std::pair<Index, Index>> g;
    for (Index c = 0; c < n; ++c) {
      const Index v = s.uzinv[s.dense[k][s.ux[c]]];
      if (v != c) g.emplace_back(c, v);
    }
    if (g.empty() || std::find(out.gens.begin(), out.gens.end(), g) != out.gens.end()) continue;
    out.gens.push_back(std::move(g));
    ++added;
  }
  return out;
}

Symmetry detect_symmetry(const MipProblem& p, const SymmetryOptions& opts) {
  Symmetry sym;
  Detector d(p, opts);
  d.run(sym);
  return sym;
}

}  // namespace milp
