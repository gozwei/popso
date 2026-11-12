#include "milp/heuristics/fpr.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <utility>

#include "milp/core/interrupt.hpp"
#include "milp/core/log.hpp"
#include "milp/core/timer.hpp"
#include "milp/heuristics/fj.hpp"
#include "milp/heuristics/lmip.hpp"
#include "milp/heuristics/localsearch.hpp"
#include "milp/lp/simplex.hpp"
#include "milp/search/propagation.hpp"

namespace milp {

namespace {

constexpr double kBig = 1e12;      // larger bounds are infinite (as in DomainPropagator)
constexpr double kViolTol = 1e-6;  // row violation tolerance of a partial assignment
constexpr double kFeasTol = 1e-6;  // final check (max_violation), as the search's incumbent check
constexpr double kNoise = 0.75;    // WalkSAT noise parameter p (paper: 0.75)
constexpr int kRepairIters = 200;  // repair walk iterations per call (paper: 200)
constexpr int kLeafLps = 2;        // continuous-completion LPs per strategy run

inline double lo_val(double v) { return v < -kBig ? -kInf : v; }
inline double hi_val(double v) { return v > kBig ? kInf : v; }

bool mode_propagates(FprMode m) { return m != FprMode::Dive; }
bool mode_repairs(FprMode m) { return m != FprMode::Dfs; }
bool mode_backtracks(FprMode m) { return m == FprMode::Dfs || m == FprMode::DfsRep; }
// Runs with the same seed-independent choices repeat themselves: only these get further rounds.
bool randomized(const FprStrategy& s) {
  return mode_repairs(s.mode) || s.order == FprOrder::Random || s.value == FprValue::Random ||
         s.value == FprValue::Lp;
}

class Fpr {
 public:
  Fpr(const MipProblem& p, const FprLimits& lim) : p_(p), lim_(lim), prop_(p), R_(prop_.rows()) {
    n_ = p.num_cols();
    m_ = p.num_rows();
    prop_.continuous = true;
  }

  // Root propagation and fixing of trivially roundable integers. False: proven infeasible.
  bool init();
  // One strategy run (a limited DFS from the root state) with `budget` more work units.
  bool run(const FprStrategy& s, unsigned seed, long long budget, std::vector<double>& sol);
  long long work() const { return work_; }
  bool out_of_time() const {
    if ((lim_.stop && lim_.stop->load(std::memory_order_relaxed)) || interrupted()) return true;
    return is_finite(lim_.time_limit) &&
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count() > lim_.time_limit;
  }

  FprStats st;

 private:
  struct TrailEntry {
    Index col;
    double lb, ub;
  };
  struct Node {
    Index col;  // -1: root
    double lb, ub;
    size_t trail;
    size_t cursor;  // position in order_ to continue the scan for the next variable
  };
  struct Cand {
    Index col;
    double shift, damage;
  };

  // --- activities and violations ---
  // Infinite contributions are counted, with the sum of their column indices (the column itself
  // when there is one).
  void add_contrib(Index i, Index j, double a, double l, double u, int sgn) {
    const double lo = a > 0 ? lo_val(l) : hi_val(u);
    const double hi = a > 0 ? hi_val(u) : lo_val(l);
    if (is_finite(lo)) minact_[i] += sgn * a * lo;
    else mininf_[i] += sgn, minsum_[i] += sgn * static_cast<long long>(j);
    if (is_finite(hi)) maxact_[i] += sgn * a * hi;
    else maxinf_[i] += sgn, maxsum_[i] += sgn * static_cast<long long>(j);
  }
  void recompute_row(Index i) {
    minact_[i] = maxact_[i] = 0.0;
    mininf_[i] = maxinf_[i] = 0;
    minsum_[i] = maxsum_[i] = 0;
    for (Index k = R_.begin(i); k < R_.end(i); ++k)
      add_contrib(i, R_.index[k], R_.value[k], alb_[R_.index[k]], aub_[R_.index[k]], 1);
    work_ += R_.end(i) - R_.begin(i);
  }
  // Violation of row i if its activity bounds were mn / mx (0 within the tolerance).
  double viol_of(Index i, double mn, double mx) const {
    double v = 0.0;
    const double L = p_.row_lb[i], U = p_.row_ub[i];
    if (maxinf_[i] == 0 && is_finite(L)) v = std::max(v, L - mx);
    if (mininf_[i] == 0 && is_finite(U)) v = std::max(v, mn - U);
    return v > kViolTol ? v : 0.0;
  }
  void refresh(Index i) {
    double v = viol_of(i, minact_[i], maxact_[i]);
    // Incremental updates drift: re-evaluate rows that look slightly violated from scratch.
    if (v > 0.0 && v < 1e-5 + 1e-9 * (std::abs(minact_[i]) + std::abs(maxact_[i]))) {
      recompute_row(i);
      v = viol_of(i, minact_[i], maxact_[i]);
    }
    if (v > 0.0) {
      if (vpos_[i] < 0) {
        vpos_[i] = static_cast<Index>(vlist_.size());
        vlist_.push_back(i);
      }
      viol_[i] = v;
    } else if (vpos_[i] >= 0) {
      const Index last = vlist_.back();
      vlist_[vpos_[i]] = last;
      vpos_[last] = vpos_[i];
      vlist_.pop_back();
      vpos_[i] = -1;
      viol_[i] = 0.0;
    }
  }
  double total_violation() const {
    double s = 0.0;
    for (Index i : vlist_) s += viol_[i];
    return s;
  }
  // Moves column j to [nl, nu] (activities, violations, both bound copies); not trailed.
  void update_col(Index j, double nl, double nu) {
    const double ol = alb_[j], ou = aub_[j];
    alb_[j] = lb_[j] = nl;
    aub_[j] = ub_[j] = nu;
    const double width = hi_val(nu) - lo_val(nl);
    const bool wider = width > hi_val(ou) - lo_val(ol);  // undo: the row ranges may grow
    for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
      const Index i = p_.A.index[k];
      const double a = p_.A.value[k];
      if (a == 0.0) continue;
      add_contrib(i, j, a, ol, ou, -1);
      add_contrib(i, j, a, nl, nu, 1);
      refresh(i);
      if (wider) raise_range(i, j, a, std::abs(a) * width);
      if (collect_ && !inq_[i]) inq_[i] = 1, queue_.push_back(i);
    }
    work_ += p_.A.end(j) - p_.A.begin(j);
  }
  void set_bounds(Index j, double nl, double nu) {
    trail_.push_back({j, alb_[j], aub_[j]});
    update_col(j, nl, nu);
  }
  void undo_to(size_t ptr) {
    while (trail_.size() > ptr) {
      const TrailEntry e = trail_.back();
      trail_.pop_back();
      update_col(e.col, e.lb, e.ub);
    }
  }
  // Propagates from column j (whose bounds just changed) on the incrementally maintained
  // activities (DomainPropagator recomputes each queued row from scratch: up to 20x more work
  // per fixing on models with long rows); the bounds it tightens are trailed. False: proven
  // infeasible (a violated row or crossing bounds).
  bool propagate(Index j) {
    for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
      const Index i = p_.A.index[k];
      if (!inq_[i]) inq_[i] = 1, queue_.push_back(i);
    }
    const long long w0 = work_;
    const long long limit = work_ + 20LL * p_.A.nnz() + 1000;
    collect_ = true;
    bool ok = true;
    for (size_t h = 0; h < queue_.size(); ++h) {
      const Index i = queue_[h];
      inq_[i] = 0;
      if (!ok || work_ > limit) continue;  // drain the flags
      ok = vpos_[i] < 0 && propagate_row(i);
    }
    queue_.clear();
    collect_ = false;
    prop_work_ += work_ - w0;
    return ok;
  }
  bool propagate_row(Index i);
  bool tighten(Index i, Index j, double a, bool su, bool sl);

  // --- strategies ---
  void build_order(FprOrder o);
  void build_clique_order();
  double pick_value(Index j, FprValue rule);
  Index next_var(size_t& cursor) const {
    while (cursor < order_.size() && lb_[order_[cursor]] == ub_[order_[cursor]]) ++cursor;
    return cursor < order_.size() ? order_[cursor] : -1;
  }

  // --- repair ---
  double compute_shift(Index i, Index j, double a, bool toward) const;
  double damage(Index j, double s) {
    double d = 0.0;
    for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
      const Index r = p_.A.index[k];
      const double a = p_.A.value[k];
      const double after = viol_of(r, minact_[r] + a * s, maxact_[r] + a * s);
      if (after > viol_[r]) d += after - viol_[r];
    }
    work_ += p_.A.end(j) - p_.A.begin(j);
    return d;
  }
  void repair_walk(long long work_end);

  // --- leaf ---
  bool complete(std::vector<double>& sol);
  double rand01() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng_); }

  const MipProblem& p_;
  const FprLimits& lim_;
  DomainPropagator prop_;
  const SparseMatrix& R_;  // rows (CSR) of the propagator
  Index n_ = 0, m_ = 0;
  std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();
  long long work_ = 0, prop_work_ = 0, lp_work_ = 0;
  const bool debug_ = std::getenv("MILP_FPR_DEBUG") != nullptr;  // per-run summaries on stdout
  std::mt19937 rng_;

  std::vector<double> glb_, gub_;  // global domain (after root propagation and dual fixings)
  std::vector<double> lb_, ub_;    // current domain (the propagator writes here)
  std::vector<double> alb_, aub_;  // domain the activities and the trail reflect
  std::vector<double> minact_, maxact_;
  std::vector<int> mininf_, maxinf_;
  std::vector<long long> minsum_, maxsum_;
  std::vector<double> viol_;
  std::vector<Index> vlist_, vpos_;  // violated rows
  std::vector<TrailEntry> trail_;
  std::vector<int> up_locks_, down_locks_;
  std::vector<char> is_bin_;
  std::vector<Index> order_;
  std::vector<Index> clique_order_;  // TypeCliques order (built once)
  bool clique_order_built_ = false;
  bool has_cont_ = false;
  Index num_free_ints_ = 0;
  int leaf_lps_run_ = 0;
  long long run_budget_ = 0;
  // Leaf LP over the continuous columns (built on first use, warm started afterwards).
  bool leaf_built_ = false, leaf_zero_cost_ = false;
  DualSimplex leaf_lp_;
  SparseMatrix leaf_A_;
  std::vector<double> leaf_l_, leaf_u_, fixed_act_;
  std::vector<Index> leaf_cols_, leaf_rows_;
  long long leaf_nnz_ = 0;

 public:
  long long lp_allowance_ = 0;  // work units the leaf LPs may still use

 private:
  std::vector<Index> tight_;
  // Per row: upper bounds on the largest bound range |a_j| (ub_j - lb_j) of its columns (range_,
  // attained by column top_ with coefficient topa_) and on that of the other columns (second_);
  // exact after a full scan, raised when a domain widens (undo).
  std::vector<double> range_, second_, topa_;
  std::vector<Index> top_;
  void raise_range(Index i, Index j, double a, double r) {
    if (j == top_[i]) {
      range_[i] = std::max(range_[i], r);
    } else if (r > range_[i]) {
      second_[i] = std::max(second_[i], range_[i]);
      range_[i] = r, top_[i] = j, topa_[i] = a;
    } else {
      second_[i] = std::max(second_[i], r);
    }
  }
  std::vector<char> inq_;
  std::vector<Index> queue_;
  bool collect_ = false;  // update_col queues the rows of changed columns (during propagation)
  std::vector<Cand> cand_;
  std::vector<double> x_;
};

bool Fpr::init() {
  glb_ = p_.col_lb;
  gub_ = p_.col_ub;
  std::vector<Index> seeds(n_);
  std::iota(seeds.begin(), seeds.end(), 0);
  if (!prop_.propagate(glb_, gub_, seeds, tight_)) return false;
  work_ += prop_.last_work;
  // Static locks: rows that may become violated when the variable moves up / down.
  up_locks_.assign(n_, 0);
  down_locks_.assign(n_, 0);
  for (Index j = 0; j < n_; ++j) {
    for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
      const Index i = p_.A.index[k];
      const double a = p_.A.value[k];
      const bool lo = is_finite(p_.row_lb[i]), hi = is_finite(p_.row_ub[i]);
      if (a > 0) up_locks_[j] += hi, down_locks_[j] += lo;
      else if (a < 0) up_locks_[j] += lo, down_locks_[j] += hi;
    }
  }
  work_ += p_.A.nnz();
  // Trivially roundable integers: a direction without locks never hurts feasibility.
  seeds.clear();
  for (Index j = 0; j < n_; ++j) {
    if (!p_.is_int[j] || glb_[j] == gub_[j]) continue;
    const bool down_free = down_locks_[j] == 0 && is_finite(lo_val(glb_[j]));
    const bool up_free = up_locks_[j] == 0 && is_finite(hi_val(gub_[j]));
    if (down_free && (!up_free || p_.cost[j] >= 0.0)) gub_[j] = glb_[j], seeds.push_back(j);
    else if (up_free) glb_[j] = gub_[j], seeds.push_back(j);
  }
  if (!seeds.empty()) {
    tight_.clear();
    if (!prop_.propagate(glb_, gub_, seeds, tight_)) return false;
    work_ += prop_.last_work;
  }
  lb_ = alb_ = glb_;
  ub_ = aub_ = gub_;
  minact_.assign(m_, 0.0);
  maxact_.assign(m_, 0.0);
  mininf_.assign(m_, 0);
  maxinf_.assign(m_, 0);
  minsum_.assign(m_, 0);
  maxsum_.assign(m_, 0);
  viol_.assign(m_, 0.0);
  vpos_.assign(m_, -1);
  vlist_.clear();
  range_.assign(m_, 0.0);
  second_.assign(m_, 0.0);
  topa_.assign(m_, 0.0);
  top_.assign(m_, -1);
  inq_.assign(m_, 0);
  for (Index i = 0; i < m_; ++i) {
    recompute_row(i);
    refresh(i);
    for (Index k = R_.begin(i); k < R_.end(i); ++k) {
      const Index j = R_.index[k];
      raise_range(i, j, R_.value[k], std::abs(R_.value[k]) * (hi_val(gub_[j]) - lo_val(glb_[j])));
    }
  }
  is_bin_.assign(n_, 0);
  num_free_ints_ = 0;
  for (Index j = 0; j < n_; ++j) {
    if (!p_.is_int[j]) {
      has_cont_ = true;
      continue;
    }
    if (glb_[j] == gub_[j]) continue;
    ++num_free_ints_;
    is_bin_[j] = glb_[j] == 0.0 && gub_[j] == 1.0;
  }
  return true;
}

// Bound tightening of the variables of row i from its current activity bounds (same rules and
// tolerances as DomainPropagator), scanning the row only when needed: a side whose slack is at
// least the largest bound range |a_j| (ub_j - lb_j) of the row cannot tighten anything; if it is
// at least the second largest, only the column with the largest range can; with one infinite
// contribution only that column can.
bool Fpr::propagate_row(Index i) {
  const double L = p_.row_lb[i], U = p_.row_ub[i];
  // Per side: -1 nothing, -2 full scan, else the only column that can be tightened.
  auto side = [&](bool fin, int ninf, long long infsum, double slack) -> Index {
    if (!fin || ninf > 1) return -1;
    if (ninf == 1) return static_cast<Index>(infsum);
    if (slack >= range_[i]) return -1;
    return slack >= second_[i] && top_[i] >= 0 ? top_[i] : -2;
  };
  const Index su = side(is_finite(U), mininf_[i], minsum_[i], U - minact_[i]);
  const Index sl = side(is_finite(L), maxinf_[i], maxsum_[i], maxact_[i] - L);
  if (su == -1 && sl == -1) return true;
  if (su != -2 && sl != -2) {
    const Index* first = R_.index.data() + R_.begin(i);
    const Index* last = R_.index.data() + R_.end(i);
    auto coef = [&](Index j) {
      if (j == top_[i]) return topa_[i];
      const Index* q = std::lower_bound(first, last, j);
      return q != last && *q == j ? R_.value[q - R_.index.data()] : 0.0;
    };
    work_ += 2 + static_cast<long long>(std::log2(1.0 + static_cast<double>(last - first)));
    if (su >= 0 && !tighten(i, su, coef(su), true, su == sl)) return false;
    if (sl >= 0 && sl != su && !tighten(i, sl, coef(sl), false, true)) return false;
    return true;
  }
  work_ += R_.end(i) - R_.begin(i);
  double r1 = 0.0, r2 = 0.0, a1 = 0.0;
  Index t1 = -1;
  for (Index k = R_.begin(i); k < R_.end(i); ++k) {
    const Index j = R_.index[k];
    const double a = R_.value[k];
    if (!tighten(i, j, a, su != -1, sl != -1)) return false;
    const double r = std::abs(a) * (hi_val(aub_[j]) - lo_val(alb_[j]));
    if (r > r1) r2 = r1, r1 = r, t1 = j, a1 = a;
    else r2 = std::max(r2, r);
  }
  range_[i] = r1;
  second_[i] = r2;
  top_[i] = t1;
  topa_[i] = a1;
  return true;
}

// Tightens column j (coefficient a in row i) from the row's upper (su) / lower (sl) side.
// False: the implied bounds cross.
bool Fpr::tighten(Index i, Index j, double a, bool su, bool sl) {
  const bool is_int = p_.is_int[j];
  const double l = lo_val(alb_[j]), u = hi_val(aub_[j]);
  if (l == u || a == 0.0) return true;
  static const bool no_cont = std::getenv("MILP_FPR_NOCONT") != nullptr;
  if (!is_int && no_cont) return true;
  // Residual activities without j.
  double rmin, rmax;
  bool rmin_ok, rmax_ok;
  if (a > 0) {
    rmin_ok = mininf_[i] - (is_finite(l) ? 0 : 1) == 0;
    rmin = is_finite(l) ? minact_[i] - a * l : minact_[i];
    rmax_ok = maxinf_[i] - (is_finite(u) ? 0 : 1) == 0;
    rmax = is_finite(u) ? maxact_[i] - a * u : maxact_[i];
  } else {
    rmin_ok = mininf_[i] - (is_finite(u) ? 0 : 1) == 0;
    rmin = is_finite(u) ? minact_[i] - a * u : minact_[i];
    rmax_ok = maxinf_[i] - (is_finite(l) ? 0 : 1) == 0;
    rmax = is_finite(l) ? maxact_[i] - a * l : maxact_[i];
  }
  const double L = p_.row_lb[i], U = p_.row_ub[i];
  double new_lb = l, new_ub = u;
  const double slack = kViolTol * (1.0 + std::abs(rmin) + std::abs(rmax));
  auto down = [&](double v) { return is_int ? std::floor(v + 1e-9) : v; };
  auto up = [&](double v) { return is_int ? std::ceil(v - 1e-9) : v; };
  if (su && rmin_ok) {
    const double bound = (U - rmin) / a;
    if (a > 0) new_ub = std::min(new_ub, down(bound + slack / a));
    else new_lb = std::max(new_lb, up(bound - slack / -a));
  }
  if (sl && rmax_ok) {
    const double bound = (L - rmax) / a;
    if (a > 0) new_lb = std::max(new_lb, up(bound - slack / a));
    else new_ub = std::min(new_ub, down(bound + slack / -a));
  }
  if (new_lb > new_ub + (is_int ? 0.0 : kViolTol * (1.0 + std::abs(new_lb)))) return false;
  if (!is_int) {
    // Meaningful changes only (no long chains of tiny steps); infinite bounds always count.
    static const double ce = [] { const char* v = std::getenv("MILP_PROP_CONT_EPS"); return v ? std::atof(v) : 1e-3; }();
    const double width = (is_finite(u) && is_finite(l)) ? u - l : kInf;
    const double eps = is_finite(width) ? ce * std::max(width, 1e-3) : ce * (1.0 + std::abs(new_lb > l ? new_lb : new_ub));
    if (new_lb < l + eps) new_lb = l;
    if (new_ub > u - eps) new_ub = u;
    if (new_lb > new_ub) return true;
  }
  if (new_lb > l || new_ub < u) set_bounds(j, std::max(alb_[j], new_lb), std::min(aub_[j], new_ub));
  return true;
}

void Fpr::build_order(FprOrder o) {
  order_.clear();
  auto free_int = [&](Index j) { return p_.is_int[j] && glb_[j] < gub_[j]; };
  if (o == FprOrder::LeftRight) {
    for (Index j = 0; j < n_; ++j)
      if (free_int(j)) order_.push_back(j);
    return;
  }
  if (o == FprOrder::TypeCliques) {
    if (!clique_order_built_) build_clique_order();
    order_ = clique_order_;
    return;
  }
  std::vector<Index> bins, gens;
  for (Index j = 0; j < n_; ++j) {
    if (!free_int(j)) continue;
    (is_bin_[j] ? bins : gens).push_back(j);
  }
  if (o == FprOrder::Random) {
    std::shuffle(bins.begin(), bins.end(), rng_);
    std::shuffle(gens.begin(), gens.end(), rng_);
  } else if (o == FprOrder::Locks) {
    auto by_locks = [&](Index a, Index b) {
      return std::max(up_locks_[a], down_locks_[a]) > std::max(up_locks_[b], down_locks_[b]);
    };
    std::stable_sort(bins.begin(), bins.end(), by_locks);
    std::stable_sort(gens.begin(), gens.end(), by_locks);
  }
  order_ = std::move(bins);
  order_.insert(order_.end(), gens.begin(), gens.end());
}

// Binaries grouped by a greedy clique cover (strategy typecl): equality cliques first (disjoint
// ones, model order), then each remaining binary goes to the largest clique covering it and these
// groups follow by decreasing size; uncovered binaries and then general integers come last.
void Fpr::build_clique_order() {
  clique_order_built_ = true;
  std::vector<Index> cstart{0}, cvars;
  std::vector<char> ceq;
  for (Index i = 0; i < m_; ++i) {
    const Index b = R_.begin(i), e = R_.end(i);
    double fixed = 0.0, alpha = -1.0;
    int pos = 0, neg = 0;
    bool ok = e - b >= 2;
    for (Index k = b; k < e && ok; ++k) {
      const Index j = R_.index[k];
      const double a = R_.value[k];
      if (glb_[j] == gub_[j]) {
        fixed += a * glb_[j];
        continue;
      }
      if (!is_bin_[j]) ok = false;
      else if (alpha < 0) alpha = std::abs(a);
      else if (std::abs(std::abs(a) - alpha) > 1e-9 * alpha) ok = false;
      (a > 0 ? pos : neg)++;
    }
    work_ += e - b;
    if (!ok || pos + neg < 2) continue;
    // sum_P x - sum_N x <= U' is a clique iff U' = 1 - |N|; the >= side iff -L' = 1 - |P|.
    const double L = p_.row_lb[i], U = p_.row_ub[i];
    const bool le = is_finite(U) && std::abs((U - fixed) / alpha - (1.0 - neg)) < 1e-9;
    const bool ge = is_finite(L) && std::abs(-(L - fixed) / alpha - (1.0 - pos)) < 1e-9;
    if (!le && !ge) continue;
    for (Index k = b; k < e; ++k)
      if (glb_[R_.index[k]] != gub_[R_.index[k]]) cvars.push_back(R_.index[k]);
    cstart.push_back(static_cast<Index>(cvars.size()));
    ceq.push_back(le && ge);
  }
  const Index nc = static_cast<Index>(ceq.size());
  std::vector<char> covered(n_, 0);
  clique_order_.clear();
  for (Index c = 0; c < nc; ++c) {
    if (!ceq[c]) continue;
    bool disjoint = true;
    for (Index k = cstart[c]; k < cstart[c + 1] && disjoint; ++k) disjoint = !covered[cvars[k]];
    if (!disjoint) continue;
    for (Index k = cstart[c]; k < cstart[c + 1]; ++k) covered[cvars[k]] = 1, clique_order_.push_back(cvars[k]);
  }
  // Size of each clique among the still uncovered binaries; each such binary picks its largest.
  std::vector<Index> size(nc, 0);
  for (Index c = 0; c < nc; ++c)
    for (Index k = cstart[c]; k < cstart[c + 1]; ++k) size[c] += !covered[cvars[k]];
  std::vector<Index> best(n_, -1);
  for (Index c = 0; c < nc; ++c) {
    for (Index k = cstart[c]; k < cstart[c + 1]; ++k) {
      const Index j = cvars[k];
      if (!covered[j] && (best[j] < 0 || size[c] > size[best[j]])) best[j] = c;
    }
  }
  std::vector<Index> gsize(nc, 0);
  for (Index j = 0; j < n_; ++j)
    if (best[j] >= 0) ++gsize[best[j]];
  std::vector<Index> groups;
  for (Index c = 0; c < nc; ++c)
    if (gsize[c] > 0) groups.push_back(c);
  std::stable_sort(groups.begin(), groups.end(), [&](Index a, Index b) { return gsize[a] > gsize[b]; });
  for (Index c : groups) {
    for (Index k = cstart[c]; k < cstart[c + 1]; ++k) {
      const Index j = cvars[k];
      if (!covered[j] && best[j] == c) covered[j] = 1, clique_order_.push_back(j);
    }
  }
  work_ += 3 * static_cast<long long>(cvars.size());
  for (Index j = 0; j < n_; ++j)
    if (is_bin_[j] && !covered[j]) clique_order_.push_back(j);
  for (Index j = 0; j < n_; ++j)
    if (p_.is_int[j] && glb_[j] < gub_[j] && !is_bin_[j]) clique_order_.push_back(j);
}

double Fpr::pick_value(Index j, FprValue rule) {
  const double l = lo_val(lb_[j]), u = hi_val(ub_[j]);
  const double c = p_.cost[j];
  double v = l;
  switch (rule) {
    case FprValue::Up: v = u; break;
    case FprValue::Random: v = (rng_() & 1u) ? u : l; break;
    case FprValue::GoodObj: v = c < 0 ? u : l; break;
    case FprValue::BadObj: v = c > 0 ? u : l; break;
    case FprValue::LooseDyn: {
      // Locks of the rows that are not redundant yet on the side the move could violate.
      int up = 0, down = 0;
      for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
        const Index i = p_.A.index[k];
        const double a = p_.A.value[k];
        const bool hi_active = is_finite(p_.row_ub[i]) && (maxinf_[i] > 0 || maxact_[i] > p_.row_ub[i] + kViolTol);
        const bool lo_active = is_finite(p_.row_lb[i]) && (mininf_[i] > 0 || minact_[i] < p_.row_lb[i] - kViolTol);
        if (a > 0) up += hi_active, down += lo_active;
        else if (a < 0) up += lo_active, down += hi_active;
      }
      work_ += p_.A.end(j) - p_.A.begin(j);
      v = up < down ? u : l;
      break;
    }
    case FprValue::Lp: {
      const double x = lim_.lp_x ? (*lim_.lp_x)[j] : 0.0;
      const double fl = std::floor(x + 1e-9);
      v = rand01() < x - fl ? fl + 1.0 : fl;
      break;
    }
  }
  // Infinite bound picked: the finite one, else 0.
  if (!is_finite(v)) v = is_finite(l) ? l : is_finite(u) ? u : 0.0;
  return std::clamp(v, lb_[j], ub_[j]);
}

// Shift of column j's domain [l, u] -> [l + s, u + s] that reduces the violation of row i (with
// coefficient a) as far as possible within the global domain; binaries flip. Integers round the
// shift away from zero (enough to repair the row) or, with `toward`, toward zero. 0: no move.
double Fpr::compute_shift(Index i, Index j, double a, bool toward) const {
  const double l = lb_[j], u = ub_[j];
  if (!is_finite(lo_val(l)) || !is_finite(hi_val(u))) return 0.0;
  if (is_bin_[j]) {
    if (l != u || toward) return 0.0;
    return l < 0.5 ? 1.0 : -1.0;
  }
  const double L = p_.row_lb[i], U = p_.row_ub[i];
  const double over = mininf_[i] == 0 && is_finite(U) ? minact_[i] - U : -kInf;
  const double under = maxinf_[i] == 0 && is_finite(L) ? L - maxact_[i] : -kInf;
  double s = over >= under ? (U - minact_[i]) / a : (L - maxact_[i]) / a;
  if (p_.is_int[j]) {
    const bool pos = s > 0;
    s = (pos != toward) ? (pos ? std::ceil(s - 1e-9) : std::floor(s + 1e-9)) : (pos ? std::floor(s + 1e-9) : std::ceil(s - 1e-9));
  }
  const double gl = lo_val(glb_[j]), gu = hi_val(gub_[j]);
  if (is_finite(gl)) s = std::max(s, gl - l);
  if (is_finite(gu)) s = std::min(s, gu - u);
  return std::abs(s) < 1e-9 ? 0.0 : s;
}

// WalkSAT-like repair of the partial assignment (paper Fig. 4): pick a violated row at random,
// collect the shifts of its variables that reduce its violation, and apply one with the least
// damage (violation increase of the other rows), or a random one with probability p if every
// candidate does damage. Returns to the least violated state seen.
void Fpr::repair_walk(long long work_end) {
  ++st.repairs;
  double best = total_violation();
  size_t best_ptr = trail_.size();
  for (int it = 0; it < kRepairIters && !vlist_.empty() && work_ < work_end; ++it) {
    const Index i = vlist_[rng_() % vlist_.size()];
    cand_.clear();
    double best_damage = kInf;
    for (Index k = R_.begin(i); k < R_.end(i); ++k) {
      const Index j = R_.index[k];
      const double a = R_.value[k];
      for (const bool toward : {false, true}) {
        const double s = compute_shift(i, j, a, toward);
        if (s == 0.0) continue;
        if (viol_of(i, minact_[i] + a * s, maxact_[i] + a * s) >= viol_[i] - 1e-9) continue;
        const double d = damage(j, s);
        best_damage = std::min(best_damage, d);
        cand_.push_back({j, s, d});
        break;
      }
    }
    work_ += R_.end(i) - R_.begin(i);
    if (cand_.empty()) continue;
    const Cand* pick;
    if (best_damage > 0.0 && rand01() < kNoise) {
      pick = &cand_[rng_() % cand_.size()];
    } else {
      size_t cnt = 0;
      for (const Cand& c : cand_) cnt += c.damage <= best_damage;
      size_t r = rng_() % cnt;
      pick = nullptr;
      for (const Cand& c : cand_) {
        if (c.damage <= best_damage && r-- == 0) {
          pick = &c;
          break;
        }
      }
    }
    set_bounds(pick->col, lb_[pick->col] + pick->shift, ub_[pick->col] + pick->shift);
    const double cur = total_violation();
    if (cur < best - 1e-12 || vlist_.empty()) {
      best = cur;
      best_ptr = trail_.size();
    }
  }
  undo_to(best_ptr);
  if (vlist_.empty()) ++st.repairs_ok;
}

// All integers are fixed and no row is violated by the activity bounds: the continuous variables
// get their values from an LP over them with the integers fixed (their global bounds, the
// original costs). The LP (continuous columns, the rows they appear in) is built once; leaves
// only change its row bounds, so later solves start from the previous basis. Writes the point to
// `sol` if it is feasible.
bool Fpr::complete(std::vector<double>& sol) {
  x_.assign(n_, 0.0);
  bool need_lp = false;
  for (Index j = 0; j < n_; ++j) {
    if (p_.is_int[j] || lb_[j] == ub_[j]) x_[j] = lb_[j];
    else need_lp = true;
  }
  work_ += p_.A.nnz();
  if (!need_lp && p_.max_violation(x_) <= kFeasTol) {
    sol = x_;
    return true;
  }
  if (!has_cont_ || leaf_lps_run_ >= kLeafLps || lp_allowance_ <= 0) return false;
  ++leaf_lps_run_;
  ++st.leaf_lps;
  if (!leaf_built_) {
    leaf_built_ = true;
    std::vector<Index> rid(m_, -1);
    SparseMatrix A;
    std::vector<double> c, l, u;
    for (Index j = 0; j < n_; ++j) {
      if (p_.is_int[j]) continue;
      leaf_cols_.push_back(j);
      for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
        const Index i = p_.A.index[k];
        if (rid[i] < 0) rid[i] = static_cast<Index>(leaf_rows_.size()), leaf_rows_.push_back(i);
        A.index.push_back(rid[i]);
        A.value.push_back(p_.A.value[k]);
      }
      A.start.push_back(static_cast<Index>(A.index.size()));
      c.push_back(p_.cost[j]);
      l.push_back(glb_[j]);
      u.push_back(gub_[j]);
    }
    A.num_major = static_cast<Index>(leaf_cols_.size());
    A.num_minor = static_cast<Index>(leaf_rows_.size());
    leaf_nnz_ = A.nnz();
    work_ += p_.A.nnz();
    if (!leaf_rows_.empty()) {
      std::vector<double> rl(leaf_rows_.size()), ru(leaf_rows_.size());
      for (size_t r = 0; r < leaf_rows_.size(); ++r) rl[r] = p_.row_lb[leaf_rows_[r]], ru[r] = p_.row_ub[leaf_rows_[r]];
      leaf_lp_.load(A, c, l, u, rl, ru);
      leaf_A_ = std::move(A);
      leaf_l_ = std::move(l);
      leaf_u_ = std::move(u);
    }
  }
  const Index nc = static_cast<Index>(leaf_cols_.size()), mr = static_cast<Index>(leaf_rows_.size());
  if (mr == 0) {
    for (Index j : leaf_cols_) {
      const double lo = lo_val(glb_[j]), hi = hi_val(gub_[j]);
      double v = p_.cost[j] > 0 ? lo : p_.cost[j] < 0 ? hi : std::clamp(0.0, lo, hi);
      if (!is_finite(v)) v = is_finite(lo) ? lo : is_finite(hi) ? hi : 0.0;
      x_[j] = v;
    }
  } else {
    std::vector<double>& fa = fixed_act_;
    fa.assign(m_, 0.0);
    for (Index j = 0; j < n_; ++j) {
      if (!p_.is_int[j] || x_[j] == 0.0) continue;
      for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) fa[p_.A.index[k]] += p_.A.value[k] * x_[j];
    }
    work_ += p_.A.nnz() + mr;
    for (Index r = 0; r < mr; ++r) {
      const Index i = leaf_rows_[r];
      leaf_lp_.set_row_bounds(r, p_.row_lb[i] - fa[i], p_.row_ub[i] - fa[i]);
    }
    LpStatus status = LpStatus::NumericalError;
    // Work per simplex iteration ~ (rows + columns) / 4 (measured: 0.1-2x of that); leaf LPs share
    // an allowance of half the call's budget.
    const long long per_iter = (nc + mr) / 4 + 1;
    auto solve = [&](bool tight) {
      LpOptions o;
      if (tight) o.tol.primal_feas = 1e-9, o.perturb = false;
      o.stop = lim_.stop;
      o.iteration_limit = std::max<long long>(200, std::min<long long>(10LL * (nc + mr) + 1000, lp_allowance_ / per_iter));
      if (is_finite(lim_.time_limit)) {
        o.time_limit = std::max(0.1, lim_.time_limit - std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count());
      }
      const long long it0 = leaf_lp_.iterations();
      const auto tlp = std::chrono::steady_clock::now();
      const LpStatus ls = leaf_lp_.solve(o);
      const long long iters = leaf_lp_.iterations() - it0;
      if (debug_) {
        std::printf("FPR leaf LP%s: %s, %lld iterations, %d x %d, nnz %lld, %.4fs\n", tight ? " (tight)" : "",
                    to_string(ls), iters, mr, nc, leaf_nnz_,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tlp).count());
      }
      const long long w = iters * per_iter + mr;
      work_ += w;
      lp_work_ += w;
      lp_allowance_ -= w;
      return ls;
    };
    static const int lp_mode = [] { const char* v = std::getenv("MILP_FPR_LP_MODE"); return v ? std::atoi(v) : 3; }();
    for (int attempt = 0; attempt < 2; ++attempt) {
      status = solve(lp_mode == 1 || lp_mode == 2);
      if (status == LpStatus::Optimal) {
        const double* y = leaf_lp_.x();
        for (Index t = 0; t < nc; ++t) x_[leaf_cols_[t]] = y[t];
        // Clean-up at a tight primal tolerance from the optimal basis (as the search's polishing):
        // a point that uses the default 1e-6 slack of the rows could beat the true optimum by that.
        // If it does not finish (stopped, iteration limit), the point is not used.
        if (lp_mode == 3) {
          status = solve(true);
          if (status == LpStatus::Optimal) {
            const double* z = leaf_lp_.x();
            for (Index t = 0; t < nc; ++t) x_[leaf_cols_[t]] = z[t];
          }
        }
      }
      if (status != LpStatus::Optimal && status != LpStatus::Infeasible) leaf_lp_.reset_basis();
      if (status != LpStatus::Unbounded || leaf_zero_cost_) break;
      // Unbounded objective: any feasible point will do (from now on).
      leaf_zero_cost_ = true;
      std::vector<double> rl(mr), ru(mr), zero(nc, 0.0);
      for (Index r = 0; r < mr; ++r) {
        const Index i = leaf_rows_[r];
        rl[r] = p_.row_lb[i] - fa[i], ru[r] = p_.row_ub[i] - fa[i];
      }
      leaf_lp_.load(leaf_A_, zero, leaf_l_, leaf_u_, rl, ru);
    }
    if (status != LpStatus::Optimal) return false;
  }
  ++st.leaf_lps_ok;
  work_ += p_.A.nnz();
  if (!(p_.max_violation(x_) <= kFeasTol)) return false;
  sol = x_;
  return true;
}

bool Fpr::run(const FprStrategy& s, unsigned seed, long long budget, std::vector<double>& sol) {
  rng_.seed(seed);
  undo_to(0);
  leaf_lps_run_ = 0;
  run_budget_ = budget;
  const long long work_end = work_ + budget;
  build_order(s.order);
  const bool prop = mode_propagates(s.mode), rep = mode_repairs(s.mode), back = mode_backtracks(s.mode);
  std::vector<Node> stack;
  stack.push_back({-1, 0.0, 0.0, trail_.size(), 0});
  const long long node_limit = static_cast<long long>(num_free_ints_) + 1;
  long long nodes = 0;
  bool found = false;
  while (!stack.empty() && nodes < node_limit && work_ < work_end) {
    if ((nodes & 63) == 0 && out_of_time()) break;
    const Node nd = stack.back();
    stack.pop_back();
    undo_to(nd.trail);
    ++nodes;
    bool infeas = !vlist_.empty();
    if (nd.col >= 0) {
      set_bounds(nd.col, nd.lb, nd.ub);
      infeas = !vlist_.empty();
      if (!infeas && prop) infeas = !propagate(nd.col) || !vlist_.empty();
    }
    if (infeas && rep && !vlist_.empty()) {
      repair_walk(work_end);
      infeas = !vlist_.empty();
    }
    if (infeas && back) continue;
    size_t cursor = nd.cursor;
    const Index j = next_var(cursor);
    if (j < 0) {
      if (infeas) continue;
      if (complete(sol)) {
        found = true;
        break;
      }
      continue;
    }
    const double v = pick_value(j, s.value);
    const size_t tp = trail_.size();
    // Alternatives first (popped after the preferred child fails).
    if (is_bin_[j]) {
      const double w = 1.0 - v;
      if (w >= lb_[j] && w <= ub_[j]) stack.push_back({j, w, w, tp, cursor});
    } else {
      if (v > lb_[j]) stack.push_back({j, lb_[j], v - 1.0, tp, cursor});
      if (v < ub_[j]) stack.push_back({j, v + 1.0, ub_[j], tp, cursor});
    }
    stack.push_back({j, v, v, tp, cursor});
  }
  st.nodes += nodes;
  if (debug_) {
    std::printf("FPR run %s: %s, %lld nodes, work %lld (propagation %lld, leaf LPs %lld)\n", s.name,
                found ? "found" : "failed", nodes, work_ - (work_end - budget), prop_work_, lp_work_);
  }
  prop_work_ = lp_work_ = 0;
  undo_to(0);
  return found;
}

}  // namespace

const std::vector<FprStrategy>& fpr_lp_free_portfolio() {
  static const std::vector<FprStrategy> s = {
      {FprMode::Dfs, FprOrder::TypeCliques, FprValue::BadObj, "dfs-badobjcl"},
      {FprMode::Dfs, FprOrder::Locks, FprValue::LooseDyn, "dfs-locks2"},
      {FprMode::Dive, FprOrder::Locks, FprValue::LooseDyn, "dive-locks2"},
      {FprMode::DfsRep, FprOrder::LeftRight, FprValue::LooseDyn, "dfsrep-locks"},
      {FprMode::DfsRep, FprOrder::TypeCliques, FprValue::BadObj, "dfsrep-badobjcl"},
      {FprMode::DiveProp, FprOrder::TypeCliques, FprValue::Random, "diveprop-random"},
  };
  return s;
}

const std::vector<FprStrategy>& fpr_lp_portfolio() {
  static const std::vector<FprStrategy> s = {
      {FprMode::Dfs, FprOrder::TypeCliques, FprValue::Lp, "dfs-lp"},
      {FprMode::Dive, FprOrder::TypeCliques, FprValue::Lp, "dive-lp"},
      {FprMode::DiveProp, FprOrder::TypeCliques, FprValue::Lp, "diveprop-lp"},
  };
  return s;
}

bool fix_propagate_repair(const MipProblem& p, const FprLimits& lim, std::vector<double>& sol, FprStats* stats) {
  FprStats local;
  FprStats& out = stats ? *stats : local;
  out = FprStats{};
  if (p.quadratic() || !p.implications.empty() || p.nonlinear() || p.num_cols() == 0) return false;
  if (std::find(p.is_int.begin(), p.is_int.end(), 1) == p.is_int.end()) return false;
  static const bool dbg = std::getenv("MILP_FPR_DEBUG") != nullptr;
  const auto tc = std::chrono::steady_clock::now();
  auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - tc).count(); };
  Fpr f(p, lim);
  f.lp_allowance_ = lim.effort / 2;
  bool found = false;
  const bool init_ok = f.init();
  if (dbg) std::printf("FPR init: %.4fs, work %lld\n", secs(), f.work());
  if (!init_ok) {
    f.st.infeasible = true;
  } else {
    std::vector<FprStrategy> strategies = lim.strategies;
    if (strategies.empty()) {
      strategies = fpr_lp_free_portfolio();
      if (lim.lp_x) strategies.insert(strategies.end(), fpr_lp_portfolio().begin(), fpr_lp_portfolio().end());
    }
    static const double run_mult = [] { const char* v = std::getenv("MILP_FPR_RUN_MULT"); return v ? std::atof(v) : 100.0; }();
    static const double run_min = [] { const char* v = std::getenv("MILP_FPR_RUN_MIN"); return v ? std::atof(v) : 1e5; }();
    const long long run_effort =
        lim.run_effort > 0 ? lim.run_effort : static_cast<long long>(run_mult * p.A.nnz() + run_min);
    static const int env_rounds = [] { const char* v = std::getenv("MILP_FPR_ROUNDS"); return v ? std::atoi(v) : -1; }();
    const int want = env_rounds >= 0 ? env_rounds : lim.rounds;
    const int rounds = want > 0 ? want : 1000000;  // <= 0: until the budget is spent
    bool any_random = false;
    for (const FprStrategy& s : strategies) any_random = any_random || randomized(s);
    bool spent = false;
    for (int round = 0; round < rounds && !found && !spent && (round == 0 || any_random); ++round) {
      for (size_t k = 0; k < strategies.size() && !found && !spent; ++k) {
        const FprStrategy& s = strategies[k];
        if (round > 0 && !randomized(s)) continue;
        const long long remaining = lim.effort - f.work();
        spent = remaining <= 0 || f.out_of_time();
        if (spent) break;
        ++f.st.runs;
        const unsigned seed = lim.seed * 1000003u + static_cast<unsigned>(round) * 7919u + static_cast<unsigned>(k) * 104729u;
        if (f.run(s, seed, std::min(remaining, run_effort), sol)) {
          found = true;
          f.st.strategy = s.name;
        }
        if (dbg) std::printf("FPR after %s: %.4fs\n", s.name, secs());
      }
    }
  }
  if (found && lim.one_opt) {
    std::vector<double> y = sol;
    if (one_opt(p, y) && p.max_violation(y) <= kFeasTol) sol = std::move(y);
    if (dbg) std::printf("FPR after 1-opt: %.4fs\n", secs());
  }
  f.st.effort = f.work();
  out = f.st;
  return found;
}

void heuristic_bench(const MipProblem& p, long long effort, unsigned seed, const Logger& log) {
  Index ints = 0;
  for (Index j = 0; j < p.num_cols(); ++j) ints += p.is_int[j] != 0;
  log.log(0, "HEURBENCH problem rows=%d cols=%d ints=%d nnz=%d effort=%lld\n", p.num_rows(), p.num_cols(), ints,
          p.A.nnz(), effort);
  auto user_obj = [&](double internal_no_offset) { return p.display_sense * (internal_no_offset + p.offset); };
  std::vector<double> fj_closest;  // FJ without a solution: its point with the fewest violated rows
  double fj_time = 0.0;
  if (!std::getenv("MILP_FPR_BENCH_NOFJ")) {
    // Feasibility Jump as the search runs it: fresh seeds after a stall until a point is found.
    Timer t;
    FjLimits lim;
    lim.stall_effort = 300000000LL;
    long long used = 0;
    double t_first = -1.0, obj_first = kInf, obj_best = kInf;
    int sols = 0;
    long long min_viol = -1, effort_first = -1;
    for (int a = 0; a < 8 && used < effort && sols == 0; ++a) {
      lim.effort = effort - used;
      lim.seed = seed + 1000003u * static_cast<unsigned>(a);
      FjStats st;
      feasibility_jump(p, lim, [&](const std::vector<double>&, double obj) {
        if (sols++ == 0) t_first = t.elapsed(), obj_first = obj;
        obj_best = obj;
      }, &st);
      if (effort_first < 0 && st.first_effort >= 0) effort_first = used + st.first_effort;
      used += st.effort;
      if (sols == 0 && !st.closest.empty() && (min_viol < 0 || st.min_violated < min_viol)) {
        min_viol = st.min_violated;
        fj_closest = std::move(st.closest);
      }
    }
    fj_time = t.elapsed();
    log.log(0,
            "HEURBENCH fj found=%d time_first=%.4f obj_first=%.10g obj_best=%.10g effort_first=%lld effort=%lld "
            "time=%.4f\n",
            sols > 0, t_first, sols ? user_obj(obj_first) : kInf, sols ? user_obj(obj_best) : kInf, effort_first, used,
            t.elapsed());
  }
  // Local MIP as the root runs it (MILP_LMIP): random restarts on stalls (MILP_LMIP_STALL) before
  // the first solution, then lift / breakthrough moves until the budget is used. With `start`:
  // improvement of that point (cutoff = its objective).
  auto lmip_line = [&](const char* name, long long budget, const std::vector<double>* start, double t_before) {
    Timer t;
    LocalMipLimits ll;
    ll.effort = budget;
    const char* sv = std::getenv("MILP_LMIP_STALL");
    ll.stall_effort = sv ? std::atoll(sv) : 500000000LL;
    ll.stall_restart = true;
    ll.seed = seed;
    if (start) ll.start = start, ll.cutoff = p.objective(*start) - p.offset;
    double t_first = -1.0, obj_first = kInf, obj_best = start ? ll.cutoff : kInf;
    int sols = 0;
    LocalMipStats st;
    local_mip(p, ll, [&](const std::vector<double>& x, double obj) {
      if (p.max_violation(x) > kFeasTol) log.log(0, "HEURBENCH ERROR %s infeasible point\n", name);
      if (sols++ == 0) t_first = t.elapsed(), obj_first = obj;
      obj_best = std::min(obj_best, obj);
    }, &st);
    const bool found = sols > 0 || start;
    log.log(0,
            "HEURBENCH %s found=%d time_first=%.4f obj_first=%.10g obj_best=%.10g effort_first=%lld effort=%lld "
            "time=%.4f\n",
            name, found, sols ? t_before + t_first : (start ? t_before : -1.0), sols ? user_obj(obj_first) : kInf,
            found ? user_obj(obj_best) : kInf, st.first_effort, st.effort, t_before + t.elapsed());
  };
  if (!std::getenv("MILP_FPR_BENCH_NOLMIP")) lmip_line("lmip", effort, nullptr, 0.0);
  std::vector<double> fpr_sol;
  long long fpr_effort = 0;
  double fpr_time = 0.0;
  auto fpr_line = [&](const char* name, const FprLimits& fl, double t_before = 0.0) {
    Timer t;
    FprStats st;
    std::vector<double> sol;
    const bool ok = fix_propagate_repair(p, fl, sol, &st);
    const double obj = ok ? user_obj(p.objective(sol) - p.offset) : kInf;
    log.log(0,
            "HEURBENCH %s found=%d time_first=%.4f obj_first=%.10g obj_best=%.10g effort_first=%lld strategy=%s "
            "effort=%lld time=%.4f runs=%d nodes=%lld repairs=%lld/%lld leaflps=%lld/%lld%s\n",
            name, ok, ok ? t_before + t.elapsed() : -1.0, obj, obj, ok ? st.effort : -1LL,
            st.strategy ? st.strategy : "-", st.effort, t_before + t.elapsed(), st.runs, st.nodes, st.repairs_ok,
            st.repairs, st.leaf_lps_ok, st.leaf_lps, st.infeasible ? " infeasible" : "");
    fpr_sol = ok ? sol : std::vector<double>();
    fpr_effort = st.effort;
    fpr_time = t_before + t.elapsed();
  };
  FprLimits fl;
  fl.effort = effort;
  fl.seed = seed;
  fpr_line("fpr", fl);
  if (!fpr_sol.empty() && !std::getenv("MILP_FPR_BENCH_NOLMIP")) {
    // FPR's point improved by local MIP with the rest of the budget: fpr-lmip (objective quality).
    const std::vector<double> start = fpr_sol;
    lmip_line("fpr-lmip", std::max(1LL, effort - fpr_effort), &start, fpr_time);
  }
  if (std::getenv("MILP_FPR_BENCH_FJ") && !fj_closest.empty()) {
    // FPR as a repair of FJ's closest point (the LP-guided rule follows that point); its time
    // includes the FJ run before it: fpr-fj.
    FprLimits ff = fl;
    ff.lp_x = &fj_closest;
    ff.strategies = fpr_lp_portfolio();
    fpr_line("fpr-fj", ff, fj_time);
  }
  if (std::getenv("MILP_FPR_BENCH_LP")) {
    // LP-guided portfolio after the root LP (its time and iterations count): fpr-lp.
    Timer t;
    DualSimplex lp;
    lp.load(p.A, p.cost, p.col_lb, p.col_ub, p.row_lb, p.row_ub);
    LpOptions o;
    o.time_limit = 60.0;
    const LpStatus ls = lp.solve(o);
    const double t_lp = t.elapsed();
    if (ls == LpStatus::Optimal) {
      std::vector<double> x(lp.x(), lp.x() + p.num_cols());
      FprLimits ll = fl;
      ll.lp_x = &x;
      ll.strategies = fpr_lp_portfolio();
      log.log(0, "HEURBENCH lp time=%.4f iterations=%lld\n", t_lp, lp.iterations());
      fpr_line("fpr-lp", ll, t_lp);
    } else {
      log.log(0, "HEURBENCH lp time=%.4f status=%s\n", t_lp, to_string(ls));
    }
  }
  if (std::getenv("MILP_FPR_BENCH_EACH")) {
    // Every strategy of the LP-free portfolio alone (one run each, the paper's per-run budget).
    for (const FprStrategy& s : fpr_lp_free_portfolio()) {
      FprLimits one = fl;
      one.strategies = {s};
      one.rounds = 1;
      fpr_line(s.name, one);
    }
  }
}

}  // namespace milp
