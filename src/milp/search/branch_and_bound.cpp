#include "milp/conic/relaxation.hpp"
#include "milp/conic/socp.hpp"
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
#include <unordered_map>
#include <unordered_set>

#include "milp/cuts/cut.hpp"
#include "milp/core/memory.hpp"
#include "milp/core/interrupt.hpp"
#include "milp/cuts/cut_loop.hpp"
#include "milp/heuristics/diving.hpp"
#include "milp/heuristics/feaspump.hpp"
#include "milp/heuristics/fj.hpp"
#include "milp/heuristics/fpr.hpp"
#include "milp/heuristics/lmip.hpp"
#include "milp/heuristics/localsearch.hpp"
#include "milp/heuristics/rounding.hpp"
#include "milp/heuristics/submip.hpp"
#include "milp/lp/crossover.hpp"
#include "milp/lp/simplex.hpp"
#include "milp/presolve/implied_int.hpp"
#include "milp/search/branch_score.hpp"
#include "milp/search/conflict.hpp"
#include "milp/search/orbitope.hpp"
#include "milp/search/propagation.hpp"
#include "milp/search/symmetry.hpp"
#include "milp/search/genres.hpp"
#include "milp/search/node_store.hpp"
#include "milp/search/obj_cliques.hpp"

namespace milp {

namespace {

// Search parameter override from the environment (tuning experiments; defaults unchanged).
double knob(const char* name, double def) {
  const char* v = std::getenv(name);
  return v ? std::atof(v) : def;
}

long long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Open-node storage (bound paths, stored bases, Node, NodePool): search/node_store.hpp.
using bnb::BasisOwner;
using bnb::BasisRef;
using bnb::BoundChange;
using bnb::Node;
using bnb::NodePool;
using bnb::PackedBasis;
using bnb::PathRef;
using bnb::PathSeg;
using bnb::TreeWeight;

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
    *pd = unit(k, avg(0));
    *pu = unit(k + 1, avg(1));
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
  // Ancestral pseudocosts (SCIP 10, MILP_APS = discount factor gamma, 0 off): the objective gain of a
  // grandchild over its parent per unit of the grandparent's branching distance, credited to the
  // grandparent's branching variable and direction (PS_1). Candidates score with PS_0 + gamma PS_1
  // at nodes where every candidate is reliable in both.
  void update_anc(Index j, int dir, double gain_per_unit) {
    std::lock_guard<std::mutex> lk(mu_);
    if (anc_sum_.empty()) anc_sum_.assign(cnt_.size(), 0.0), anc_cnt_.assign(cnt_.size(), 0);
    const size_t k = 2 * static_cast<size_t>(j) + (dir > 0);
    anc_sum_[k] += gain_per_unit;
    ++anc_cnt_[k];
  }
  static double aps_gamma() {
    static const double g = knob("MILP_APS", 0.0);
    return g;
  }
  // Leaves by the direction of their last branching (Shah-Dey last-assignment asymmetry, MILP_SD bit
  // 4): pruned as infeasible or integral (inf) or in any way.
  void update_leaf(int dir, bool inf) {
    std::lock_guard<std::mutex> lk(mu_);
    leaf_[2 + (dir > 0)] += 1.0;
    if (inf) leaf_[dir > 0] += 1.0;
  }
  // Score rule of a node with primal-dual gap `gap` (cutoff - LP objective; kInf without incumbent).
  // MILP_SD bits: 1 gains capped at the gap, 2 q_min^0.3 q_max^0.7, 4 last-assignment asymmetry.
  static int sd_mode() {
    static const int mode = static_cast<int>(knob("MILP_SD", 0));
    return mode;
  }
  BranchScoreRule score_rule(double gap) {
    BranchScoreRule r;
    const int mode = sd_mode();
    if (mode == 0) return r;
    r.cap = (mode & 1) != 0;
    r.minmax = (mode & 2) != 0;
    r.gap = gap;
    if (mode & 4) {
      static const double eta = knob("MILP_SD_ETA", 0.15), min_leaves = knob("MILP_SD_LEAVES", 10);
      std::lock_guard<std::mutex> lk(mu_);
      asymmetry_exponents(leaf_[0], leaf_[1], leaf_[2], leaf_[3], eta, min_leaves, &r.a_dn, &r.a_up);
    }
    return r;
  }

  // Pseudocost product scores for all candidates (index, fractional part).
  // est_sum: sum over the candidates of the smaller child increase (for best-estimate search).
  // ps_scale: average product score when a hybrid score is used (score = ps_part(product, ps_scale) + bonus),
  // 0 otherwise (raw products).
  // degen: SCIP's degeneracy factor (>= 1): with a hybrid score the pseudocost part is divided by it and
  // the other parts are multiplied by it.
  void scores(const std::vector<std::pair<Index, double>>& cands, int reliability, std::vector<Scored>& out,
              double* est_sum, double* ps_scale, double degen = 1.0, const BranchScoreRule& rule = {},
              bool* used_aps = nullptr, bool* used_flat = nullptr) {
    std::lock_guard<std::mutex> lk(mu_);
    const double avg_dn = avg(0), avg_up = avg(1);
    const double avg_inf = inf_tot_cnt_ ? inf_tot_sum_ / inf_tot_cnt_ : 1.0;
    out.clear();
    double ps_total = 0.0, inf_total = 0.0, conf_total = 0.0, cut_total = 0.0;
    const double avg_cut = cut_tot_cnt_ ? cut_tot_sum_ / static_cast<double>(cut_tot_cnt_) : 0.0;
    conf_scratch_.clear();
    cut_scratch_.clear();
    *est_sum = 0.0;
    *ps_scale = 0.0;  // 0: no hybrid score (raw products)
    inf_scratch_.clear();
    const double gamma = aps_gamma();
    bool aps = gamma > 0.0 && !anc_sum_.empty();
    for (size_t c = 0; c < cands.size() && aps; ++c) {
      const size_t k = 2 * static_cast<size_t>(cands[c].first);
      aps = std::min({cnt_[k], cnt_[k + 1], anc_cnt_[k], anc_cnt_[k + 1]}) >= reliability;
    }
    if (used_aps) *used_aps = aps;
    for (auto [j, f] : cands) {
      const size_t k = 2 * static_cast<size_t>(j);
      const double pd = unit(k, avg_dn), pu = unit(k + 1, avg_up);
      const double score = aps ? branch_score((pd + gamma * anc_sum_[k] / anc_cnt_[k]) * f,
                                              (pu + gamma * anc_sum_[k + 1] / anc_cnt_[k + 1]) * (1.0 - f), rule)
                               : branch_score(pd * f, pu * (1.0 - f), rule);
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
    static const double w_inf0 = knob("MILP_INF_WEIGHT", 0.0), w_conf0 = knob("MILP_CONF_WEIGHT", 0.0);
    static const double w_cut0 = knob("MILP_CUT_WEIGHT", 0.0);
    double w_inf = w_inf0, w_conf = w_conf0, w_cut = w_cut0;
    // Flat objective (sct2, csched008: every candidate's pseudocost score at the floor of zero gains in
    // both directions): the order would be the column order. Then the conflict and inference scores
    // decide (SCIP's hybrid weights relative to each other: conflict 1, inference and cutoff 0.01), in
    // the bounded form. flat_hybrid (set by the search): by default on zero-objective problems, where
    // every node is flat (fhnw-binpack4-4 with SST rows, 4 threads: infeasibility proof in 3.4-5.9 s
    // over 4 seeds, 41-83k nodes at depth ~20; without it TL at 120 s, 2.8M nodes at depth ~50; with
    // inference scores only 20.6 s); MILP_FLAT_HYBRID=1 on all problems (sct2 / csched008 probes,
    // 120 s: it scores 5-10% of the branchings, both bounds unchanged at -231.0636 / 171), 0 off.
    const double floor = branch_score(0.0, 0.0, rule);
    bool flat = flat_hybrid && !out.empty();
    for (size_t i = 0; i < out.size() && flat; ++i) flat = out[i].score <= floor * (1.0 + 1e-9);
    if (flat) {
      w_conf = std::max(w_conf, 1.0), w_inf = std::max(w_inf, 0.01);
      if (w_cut0 > 0.0) w_cut = std::max(w_cut, 0.01);
    }
    if (used_flat) *used_flat = flat;
    const bool use_inf = w_inf > 0.0 && inf_total > 0.0, use_conf = w_conf > 0.0 && conf_total > 0.0;
    const bool use_cut = w_cut > 0.0 && cut_total > 0.0;
    if ((use_inf || use_conf || use_cut) && ps_total > 0.0) {
      const double n = static_cast<double>(out.size());
      *ps_scale = ps_total / n;
      // MILP_HYBRID=1: SCIP's bounded form, each part v / (v + average) in [0, 1) times its weight.
      const bool bounded = hybrid_mode() == 1 || flat;
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
  bool flat_hybrid = false;  // conflict / inference scores decide at flat nodes (set by the search)
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
    std::copy(o.leaf_, o.leaf_ + 4, leaf_);
    flat_hybrid = o.flat_hybrid;
    anc_sum_ = o.anc_sum_;
    anc_cnt_ = o.anc_cnt_;
    prior_ = o.prior_;
    prior_avg_[0] = o.prior_avg_[0], prior_avg_[1] = o.prior_avg_[1];
  }

  // Restarts: the per-column history as a snapshot, and a snapshot (same number of columns) as the
  // initial state of a new search's pseudocosts; the totals (averages for columns without history)
  // are those of the loaded entries.
  PseudocostSnapshot snapshot() {
    std::lock_guard<std::mutex> lk(mu_);
    PseudocostSnapshot s;
    s.sum = sum_, s.cnt = cnt_, s.inf_sum = inf_sum_, s.inf_cnt = inf_cnt_;
    return s;
  }
  // prior_only: the snapshot's averages serve as the estimates of columns (and the default for columns)
  // without observations of this search, and count for nothing else (reliability, totals); else its
  // entries are this search's history.
  void load(const PseudocostSnapshot& s, bool prior_only) {
    std::lock_guard<std::mutex> lk(mu_);
    const size_t n = cnt_.size();
    if (s.cnt.size() != n || s.sum.size() != n || s.inf_cnt.size() != n || s.inf_sum.size() != n) return;
    inf_sum_ = s.inf_sum, inf_cnt_ = s.inf_cnt;
    inf_tot_sum_ = 0.0;
    inf_tot_cnt_ = 0;
    for (size_t k = 0; k < n; ++k) inf_tot_sum_ += inf_sum_[k], inf_tot_cnt_ += inf_cnt_[k];
    double tsum[2] = {0.0, 0.0};
    long long tcnt[2] = {0, 0};
    for (size_t k = 0; k < n; ++k) tsum[k & 1] += s.sum[k], tcnt[k & 1] += s.cnt[k];
    if (prior_only) {
      prior_.assign(n, -1.0);
      for (size_t k = 0; k < n; ++k) {
        if (s.cnt[k] > 0) prior_[k] = s.sum[k] / s.cnt[k];
      }
      for (int d = 0; d < 2; ++d) prior_avg_[d] = tcnt[d] ? tsum[d] / static_cast<double>(tcnt[d]) : -1.0;
      return;
    }
    sum_ = s.sum, cnt_ = s.cnt;
    for (int d = 0; d < 2; ++d) tot_sum_[d] = tsum[d], tot_cnt_[d] = tcnt[d];
  }

  // Picks the candidate (index, fractional part) with the best product score.
  Index select(const std::vector<std::pair<Index, double>>& cands) {
    std::lock_guard<std::mutex> lk(mu_);
    const double avg_dn = avg(0), avg_up = avg(1);
    Index best = -1;
    double best_score = -1.0;
    for (auto [j, f] : cands) {
      const size_t k = 2 * static_cast<size_t>(j);
      const double pd = unit(k, avg_dn), pu = unit(k + 1, avg_up);
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
  double leaf_[4] = {0.0, 0.0, 0.0, 0.0};  // infeasible / integral leaves down, up; all leaves down, up
  std::vector<double> anc_sum_;  // ancestral pseudocosts (MILP_APS), per (column, direction)
  std::vector<int> anc_cnt_;
  // Restart priors (load with prior_only): average gain per (column, direction), -1 none; the
  // average over all of them per direction (-1 none).
  std::vector<double> prior_;
  double prior_avg_[2] = {-1.0, -1.0};
  // Unit gain estimate of key k (mu_ held): own history, else the restart prior, else `fallback`.
  double unit(size_t k, double fallback) const {
    if (cnt_[k]) return sum_[k] / cnt_[k];
    if (!prior_.empty() && prior_[k] >= 0.0) return prior_[k];
    return fallback;
  }
  double avg(int d) const {
    if (tot_cnt_[d]) return tot_sum_[d] / static_cast<double>(tot_cnt_[d]);
    return prior_avg_[d] >= 0.0 ? prior_avg_[d] : 1.0;
  }
};

// Sub-MIP (large neighbourhood) heuristics chosen adaptively by success. Proximity search and the
// graph-induced mutation run only with the adaptive selection (MILP_ALNS, run_sub_mips).
enum { kLnsRins, kLnsMutation, kLnsLocalBranching, kLnsRens, kLnsCrossover, kLnsProximity, kLnsGraph, kLns };
struct LnsState {
  long long last_node = 0;  // node count at the last call
  long long last_iters = 0;  // search LP iterations at the last call
  long long iters = 0;      // LP iterations used by sub-MIPs
  double calls[kLns] = {}, wins[kLns] = {};
  double recent_calls = 0.0, recent_wins = 0.0;  // exponentially decayed (factor 0.9 per call)
  double mut_fix = 0.7;  // mutation: fraction of the integers fixed
  double rins_fix = 0.0;  // RINS: fixes at least this fraction (more after limit hits)
  double rins_max = 1.0;  // RINS: fixes at most this fraction (less after cheap exhausted calls)
  int lb_k = 10;  // local branching neighbourhood size
  double lb_inc = kInf;          // incumbent at the last local branching run
  unsigned seq = 0;
  long long rounds = 0;  // calls with an incumbent available (RINS on every other one)
  int cx_parents = 3;    // crossover: solutions combined (incumbent + others from the pool)
  long long cx_version = -1;  // solution pool version at the last crossover
  int cx_parents_used = 0;
  // Adaptive selection (MILP_ALNS): discounted reward sums and counts per neighbourhood (bandit arms),
  // the LP iterations of an average call (effort scale), step sizes of the fixing rates of mutation and
  // graph mutation (shrinking with each adjustment, as SCIP's ALNS), proximity search's required
  // improvement as a fraction of the gap, and the incumbent it last proved (nearly) optimal.
  double rsum[kLns] = {}, rcost[kLns] = {}, rcnt[kLns] = {};
  double iter_ref = 0.0;
  double graph_fix = 0.7;
  double mut_step = 0.1, graph_step = 0.1;
  double prox_imp = 0.02;
  double prox_inc = kInf;
  // Fingerprints of the last 64 fixing neighbourhoods searched completely without a better point: not
  // solved again (glass4: the same crossover sub-MIP, 3793 iterations, 3-4 times in 60 s).
  std::vector<std::uint64_t> searched;
  // Primal-limited searches (the bound stalls while a gap remains) get a larger share: the
  // global bound is compared each time the node count doubles.
  long long ref_nodes = 0;
  double ref_bound = -kInf;
  int boost = 0;
  bool pending = false;  // deterministic mode: a sub-MIP job of this worker has not entered yet
  bool may_start = true;  // deterministic mode: this worker may decide a sub-MIP job this round
};

// Adaptive LNS (MILP_ALNS=1; Hendel, "Adaptive large neighborhood search for mixed integer programming",
// MPC 14 (2022) 185-221, SCIP's heur_alns). A call's value is ALNS's success reward, 0.8 for a better point
// plus 0.2 x the share of the gap (to the root bound) it closed, 0 without one; its cost is a fixed 0.25 for
// the call plus its LP iterations over those of an average call. A neighbourhood scores its discounted value
// per unit of cost (discount MILP_ALNS_DECAY per call) plus a UCB term (UCB had the best primal integral in
// the paper). Not ALNS's per-call reward 0.5 (1 - effort) for failures: here a sub-MIP that fails in presolve
// (tree RENS on glass4, 0 iterations, again and again) would score 0.5 and beat a neighbourhood that improves
// on 20% of its calls at average cost (0.2 x 0.78 + 0.8 x 0.25 = 0.36), while popso budgets sub-MIPs by LP
// iterations. Deterministic for a deterministic history: LP iterations and objective values only.
//
// Measured (benchmarks/sets/miplib_primal.txt without var-smallemery-m6j6: 79 instances, 120 s, 2 threads,
// mean primal gap knob off -> on, 90% bootstrap CI of the difference): the default (UCB term
// 0.3 sqrt(ln n / n_k), discount 0.95: scores of ~0.05-0.15 against a term of ~0.3, so the choice stays
// nearly uniform over the 7 neighbourhoods) 0.316 -> 0.305 over 4 runs per side, [-0.029, +0.005], better on
// 37, worse on 26 (first 2 seeds 0.321 -> 0.287, [-0.061, -0.008], tbfp-network solved on both; again
// 0.307 -> 0.318, [-0.008, +0.032]: runs ending without any solution, A 11 / B 14, decide the mean; where
// every run has one, -0.004). A term scaled by the pooled value per cost, c = 1 (MILP_ALNS_UCB_SCALE=1),
// discount 0.99 and the re-pick (MILP_ALNS_REPICK=1): 0.301 -> 0.314, [-0.010, +0.035], better on 23 of
// the 69 with a solution in every run, worse on 30: the cheap neighbourhoods took over (crossover 367 of 661
// calls on mad, graph mutation 141 of 211 on glass4) and proximity search, whose rare successes were the big
// steps, ran rarely. As in the paper, where Exp.3, spreading the calls over several neighbourhoods, did best
// in the full solver. miplib_wide_core (4 seeds): time ratio on / off 1.016-1.024 in three A/Bs (CIs
// include 1), solved -2 / 0 / 0 of 252.
void alns_reward(LnsState& st, int pick, double value, double iterations, double weight = 1.0) {
  static const double gamma = knob("MILP_ALNS_DECAY", 0.95);
  if (st.iter_ref <= 0.0) st.iter_ref = std::max(1000.0, iterations);
  const double cost = 0.25 + iterations / st.iter_ref;
  st.iter_ref = 0.9 * st.iter_ref + 0.1 * std::max(100.0, iterations);
  for (int k = 0; k < kLns; ++k) st.rsum[k] *= gamma, st.rcost[k] *= gamma, st.rcnt[k] *= gamma;
  st.rsum[pick] += weight * value;
  st.rcost[pick] += weight * cost;
  st.rcnt[pick] += weight;
}

// Proximity search's smallest required improvement (fraction of the gap) for a fractional objective.
constexpr double kProxMin = 1e-3;

// The available neighbourhood with the best score; untried ones first, in enum order (RINS first). UCB term
// c sqrt(ln n / n_k), c = MILP_ALNS_UCB (0.3); MILP_ALNS_UCB_SCALE=1: times the pooled value per cost of all
// neighbourhoods (at least 0.02), which exploits more (see alns_reward).
int alns_select(const LnsState& st, const bool* avail) {
  static const double c = knob("MILP_ALNS_UCB", 0.3);
  static const bool scaled = knob("MILP_ALNS_UCB_SCALE", 0) > 0;
  double n = 0.0, value = 0.0, cost = 0.0;
  for (int k = 0; k < kLns; ++k) {
    if (avail[k]) n += st.rcnt[k], value += st.rsum[k], cost += st.rcost[k];
  }
  const double scale = scaled ? std::max(0.02, value / std::max(1e-9, cost)) : 1.0;
  int pick = -1;
  double best = -kInf;
  for (int k = 0; k < kLns; ++k) {
    if (!avail[k]) continue;
    if (st.rcnt[k] <= 0.0) return k;
    const double score =
        st.rsum[k] / std::max(1e-9, st.rcost[k]) + c * scale * std::sqrt(std::log(1.0 + n) / st.rcnt[k]);
    if (score > best) best = score, pick = k;
  }
  return pick;
}

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

class Search : public BasisOwner {
 public:
  Search(const MipProblem& p, const SolverParams& params, const Logger& log, const Timer& timer,
         const SearchOptions& opts)
      : p_(p), params_(params), log_(log), timer_(timer), opts_(opts), rounding_(p), zi_rounding_(p), pc_(p.num_cols()) {
    threads_ = params.threads > 0 ? params.threads : static_cast<int>(std::thread::hardware_concurrency());
    threads_ = std::max(1, threads_);
    // Objective is integral if only integer (or implied integer) variables with integer costs contribute.
    obj_integral_ = true;
    std::vector<Index> cont_costs;
    for (Index j = 0; j < p.num_cols() && obj_integral_; ++j) {
      if (p.cost[j] == 0.0) continue;
      obj_integral_ = p.cost[j] == std::round(p.cost[j]);
      if (!p.is_int[j] && !p.is_implied_int(j)) cont_costs.push_back(j);
    }
    if (obj_integral_ && !cont_costs.empty()) {
      // Not flagged by presolve (e.g. --no-presolve): detect here (presolve/implied_int.hpp).
      const std::vector<char> impl = detect_implied_integers(p);
      for (Index j : cont_costs) obj_integral_ = obj_integral_ && impl[j];
    }
    zero_obj_ = true;
    for (Index j = 0; j < p.num_cols() && zero_obj_; ++j) zero_obj_ = p.cost[j] == 0.0;
    {
      // MILP_FLAT_HYBRID: 1 all problems, 0 none, unset: zero-objective ones (see Pseudocosts::scores).
      const double fh = knob("MILP_FLAT_HYBRID", -1);
      pc_.flat_hybrid = fh > 0 || (fh < 0 && zero_obj_);
    }
    // Proximity search needs binaries (at least 10, and 10% of the integers: as heuristics/localsearch.cpp).
    {
      Index bins = 0, ints = 0;
      for (Index j = 0; j < p.num_cols(); ++j) {
        ints += p.is_int[j];
        bins += p.is_int[j] && p.col_lb[j] == 0.0 && p.col_ub[j] == 1.0;
      }
      prox_ok_ = bins >= 10 && 10 * bins >= ints;
    }
    // Implications (logical constraints kept out of the LP) fix bounds outside the propagation trail.
    if (!p.implications.empty()) conflicts_ = false;
    static const bool trivial = knob("MILP_TRIVIAL_BOUND", 1) > 0;
    if (trivial) {
      double b = 0.0;
      for (Index j = 0; j < p.num_cols() && b > -kInf; ++j) {
        const double c = p.cost[j];
        if (c != 0.0) b += c > 0.0 ? c * p.col_lb[j] : c * p.col_ub[j];
      }
      trivial_lb_ = b;
    }
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
    if (!debug_sol_.empty() && p_.max_violation(debug_sol_) > 1e-6) {  // sub-MIP rows
      if (debug_root_) {
        double iv = 0.0, lv = 0.0;
        const double v = p_.max_violation(debug_sol_, &iv, &lv);
        log_.log(0, "DEBUG-SOL: the point violates the problem by %.3g (integrality %.3g, linear %.3g)\n", v, iv, lv);
        std::vector<double> act(static_cast<size_t>(p_.num_rows()), 0.0);
        for (Index j = 0; j < p_.num_cols(); ++j)
          for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) act[p_.A.index[k]] += p_.A.value[k] * debug_sol_[j];
        for (Index i = 0; i < p_.num_rows(); ++i) {
          const double vi = std::max(p_.row_lb[i] - act[i], act[i] - p_.row_ub[i]);
          if (vi > 1e-6) log_.log(0, "DEBUG-SOL:   row %d of %d: activity %.10g, bounds [%.10g, %.10g]\n", i, p_.num_rows(), act[i], p_.row_lb[i], p_.row_ub[i]);
        }
      }
      debug_sol_.clear();
    }
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
    TreeDiveHistory dives;  // this worker's tree dives (MILP_DIVE_ADAPT)
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
    std::vector<std::pair<int, bool>> leaf_log;  // deterministic mode: update_leaf calls of this round
    std::vector<std::tuple<Index, int, double>> anc_log;  // deterministic mode: update_anc calls
    LookaheadSample la;                          // probabilistic lookahead: gains probed at this node
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
    std::vector<char> prow_bind;      // per pool row: binding after some node LP since it was added (statistics)
    std::vector<int> pool_pos;        // per pool id: its pool row index, or -1
    std::vector<std::shared_ptr<const Cut>> pool;
    std::vector<double> pool_norm;
    std::vector<char> pool_need;
    std::vector<std::pair<double, int>> pool_cand;
    std::vector<std::vector<std::pair<int, double>>> par_occ;  // pool_separate's parallelism filter
    std::vector<Index> par_cols;
    std::vector<double> par_dot;
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
    long long zi_calls = 0, zi_found = 0;  // ZI rounding calls / feasible points (its stop rule, per worker)
    // Strong branching budget (per worker, so deterministic mode stays reproducible): LP iterations
    // of the node solves (incl. cut rounds) and of the strong branching probes; probe_limit is the
    // iteration limit of the current node's probes.
    long long tree_iters = 0, tree_lps = 0, sb_iters = 0, probe_limit = 100;
    // Strong branching result of the selected candidate: objective gains of LPs solved to optimality
    // (-inf when the probe stopped early), valid as the children's bounds.
    Index sb_var = -1;
    double sb_dn = -kInf, sb_up = -kInf;
    // The node LP's objective is not confirmed by its safe bound (lp_bound): gains relative to it are no
    // bounds (root strong branching's bound, MILP_SB_BOUNDS).
    bool lp_unsafe = false;
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
  // Root start heuristics (dives, then feasibility pump and RENS) on up to nthreads (0: threads_)
  // threads; results enter in task order (deterministic). t_root: the root's time (< 0: until now).
  void root_heuristics(Worker& w, int nthreads = 0, double t_root = -1.0);
  // MILP_ROOT_HEUR_ASYNC = k (opt-in; 0: off): opportunistic mode with at least k threads runs the root
  // heuristics beside the tree instead of before it, on async_heur_threads() of the threads (the tree starts on
  // the others, which the heuristic threads join when done). heur_w_: the root LP and point they start from.
  std::unique_ptr<Worker> heur_w_;
  double heur_t_root_ = 0.0;
  bool prompt_restart_check_ = false;  // restart_check_due_ honoured (set with heur_w_)
  std::atomic<bool> heur_stop_{false};  // the tree has ended: stops the heuristics (LocalStopScope)
  int async_heur_threads() const;
  bool root_lns_ran_ = false, root_lns_improved_ = false;  // root LNS rounds (reported for restarts)
  double t_start_ = 0.0;                                   // timer at the start of run()
  double inc_start_ = kInf;  // objective (without offset) of the start point (MipResult::improved)
  void run_sub_mips(Worker& w);
  int run_lns(SubMipJob& job, LnsState& st, Worker* w, std::unique_lock<std::mutex>* lk);
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
  // Threads for the root cut loop's separation (SepaContext::threads): the search's, less the FJ / local-MIP
  // helper's while it runs (opportunistic mode; deterministic FJ waits for its next release then).
  int root_sep_threads();
  // Fix-propagate-repair (heuristics/fpr.hpp): the LP-free portfolio on the FJ thread before FJ,
  // and the LP-guided one among the root heuristics while there is no incumbent.
  bool fpr_run_lp_free(const std::function<void(const std::vector<double>&, const char*)>& publish, double deadline,
                       std::vector<double>& point);
  bool fpr_run_lp(const std::vector<double>& lp_x, std::vector<double>& sol);
  std::thread fj_thread_;
  std::atomic<bool> fj_stop_{false};
  bool fj_sync_ = false;
  std::mutex fj_mu_;
  std::condition_variable fj_cv_;
  long long fj_allowed_ = 0, fj_effort_ = 0, fj_iters_mark_ = 0, fj_prev_allowed_ = 0;
  long long root_fj_syncs_ = 0;  // deterministic FJ syncs during the initial root LP (MILP_DET_ROOT_SYNC)
  long long root_progress_ = 0;  // root LP iterations already added to opts_.progress
  double root_fj_wait_ = 0.0;     // their wall-clock wait (statistics only)
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
  std::atomic<long long> lns_last_iters_{0};  // search LP iterations at the last tree sub-MIP
  LnsState lns_;  // opportunistic mode; guarded by lns_mu_

  // Reliability branching. Returns the branching variable, or -1 if strong
  // branching proved the node infeasible.
  Index select_branch(Worker& w, double obj, const Basis& basis);
  // Strong-branching probe: LP gain of bounds [lb, ub] on j (kInf if infeasible). bound: the gain of an
  // optimal probe is used as a bound (root strong branching, MILP_SB_BOUNDS), so it comes from the probe's
  // safe bound (lp_bound); otherwise only a cutoff verdict is checked that way.
  double probe(Worker& w, Index j, double lb, double ub, double obj, const Basis& basis, bool* exact,
               bool bound = false);
  // Root strong branching rounds (see the definition): 0 = continue with branching, 1 = global
  // fixings applied (the root is processed again), 2 = the root is infeasible / cut off.
  int root_strong_branching(Worker& w, double obj, const Basis& basis);
  void root_tighten(Worker& w, Index j, double l, double u);
  long long root_sb_budget_ = -1;  // root strong branching LP iterations left (-1: not set yet)
  long long root_sb_base_ = 0;     // root LP iterations (initial solve + cut loop) before the first pass
  int root_sb_passes_ = 0;
  long long root_sb_fixed_ = 0, root_sb_probes_ = 0, root_sb_iters_ = 0;
  int root_sb_aborted_ = 0;  // passes ended by MILP_ROOT_SB_ABORT
  double root_sb_bound_ = -kInf;  // max over probed candidates of the smaller child bound (valid globally)
  Index root_sb_var_ = -1;        // branching candidate of the last pass (and its exact child gains)
  double root_sb_dn_ = -kInf, root_sb_up_ = -kInf;
  bool root_redo_ = false, root_heur_done_ = false;
  std::vector<Index> root_seeds_;  // columns fixed by root strong branching (propagated on the next pass)
  // Root restart rule (root_restart_decision) for the current global fixings and the root's cost
  // (root_work: LP iterations x (rows + columns) of the root worker's LP; root_elapsed: seconds since the
  // restart before this search, else since its start). restart_allowed: a root restart is possible at all.
  bool restart_allowed() const;
  RestartDecision restart_check(double root_work, double root_time, Index* ints, Index* newly_fixed);
  static double root_work(const DualSimplex& lp) {
    return static_cast<double>(lp.iterations()) * (lp.num_rows() + lp.num_cols());
  }
  double root_elapsed() const { return timer_.elapsed() - (opts_.restart_t0 >= 0.0 ? opts_.restart_t0 : t_start_); }
  // Root strong branching skipped because the root restarts anyway (MILP_RESTART_SKIP_SB=1): the restart
  // decision taken then stands at the end of the root.
  bool root_sb_skipped_ = false;

  int process(Worker& w, Node& node, Node* children);
  int process_impl(Worker& w, Node& node, Node* children);
  // Recovery ladder for a node LP that failed (numerical error, iteration limit, unbounded below the
  // root): the same basis with a fresh factorization, then a slack basis, without and with cost
  // perturbation. Returns the last status.
  LpStatus recover_lp(Worker& w, const LpOptions& opt, long long node_id);
  // Bound from an LP solved to optimality: its objective, or its safe dual bound when that is lower
  // (DualSimplex::safe_objective: a dual infeasibility hidden by row scaling or the tolerances; *below
  // set). Counted for the statistics; node_id < 0: a strong branching probe.
  double lp_bound(const DualSimplex& lp, bool* below, long long node_id);
  std::atomic<long long> safe_checks_{0}, safe_below_{0}, safe_ns_{0};
  // DualSimplex::check_counts at the start of run() (Farkas proof checks, unscaled dual feasibility): process-wide
  // counts, so the summary shows the difference (it includes heuristics' LPs and sub-MIPs, and concurrent
  // searches if any).
  DualSimplex::CheckCounts checks0_;
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
  // An integral node LP point (objective lp_obj) closes its node: it is offered as an incumbent, and the node
  // closes if its LP bound then reaches the cutoff, or if the point is feasible at the LP's objective (within
  // 1e-6 relative): the subtree's optimum. A point feasible only after repairing what the LP's tolerance
  // allowed, at a clearly worse objective, does not: the LP gained by the tolerance and better points may
  // remain below (a node LP gaining by its tolerance, e.g. through big-M rows; found reviewing the path of the
  // drayage-25-23 wrong answer, which had another cause).
  // *feasible: false if the point is not feasible even after the repair.
  bool integral_point_closes(Worker& w, const std::vector<double>& sol, double lp_obj, bool* feasible = nullptr);
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
  bool accept_incumbent(std::vector<double>& x, double obj, const char* source);  // under inc_mu_
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
  // time_cap >= 0: that many seconds (opportunistic mode), also past the time limit; else min(5 s, time left).
  LpStatus solve_fixed_ints(const std::vector<double>& x_int, const std::vector<double>& lb,
                            const std::vector<double>& ub, double primal_tol, double cutoff,
                            std::vector<double>& out, double time_cap = -1.0) const;
  // The final incumbent re-solved in its continuous part (see the definition).
  void final_polish(MipResult& res) const;
  // An integral node LP point (w.sol) whose rounded point violates the model by more than 1e-9: the
  // node LP warm with the integer columns fixed at their rounded values, at primal tolerance 1e-9;
  // replaces w.sol when that point violates less. The LP's state and bounds are restored.
  void polish_integral_lp(Worker& w, const LpOptions& opt);
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
  bool gap_closed(double lb) const { return gap_closed_at(lb, incumbent_.load()); }
  bool gap_closed_at(double lb, double inc) const {
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
  // Auto for large pure LPs (and large MIP roots, MILP_CONCURRENT_MIP_NNZ): dual simplex and barrier run at
  // the same time, the first to finish wins (opportunistic mode only: the winner depends on timing).
  bool use_concurrent_root() const;
  void concurrent_root(DualSimplex& lp);
  // MILP_BARRIER_DSE (default 1; 0: off): the crossover basis gets exact dual steepest-edge weights for the
  // re-solves after it (set_basis: computed at the next solve, one unit btran per row). The weights the
  // crossover's primal cleanup leaves made the root cut rounds slow: uccase9, 23217 cleanup iterations, its
  // first cut round re-solve took 16159 iterations and ran into the cut loop's time limit (no round), 4672
  // with exact weights (3 rounds as from the simplex's own root, bound 10819.58 -> 10880.85).
  static void barrier_basis_weights(DualSimplex& lp) {
    static const bool on = knob("MILP_BARRIER_DSE", 1) > 0;
    if (on) lp.set_basis(lp.get_basis());
  }
  // MILP_BARRIER_EQUIV (default 1; 0: off): a root LP solved by the barrier counts as the dual simplex
  // iterations it stands for (root_lp_equiv_iters_) in the budgets relative to the root LP's effort (cut
  // loop, root dives).
  static bool barrier_equiv() {
    static const bool on = knob("MILP_BARRIER_EQUIV", 1) > 0;
    return on;
  }
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
    if ((opts_.stop_flag && opts_.stop_flag->load()) || interrupted()) {
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
  ZiRounding zi_rounding_;
  Pseudocosts pc_;
  int threads_ = 1;
  bool obj_integral_ = false;
  bool prox_ok_ = false;  // enough binaries for proximity search (MILP_ALNS)
  const bool alns_ = knob("MILP_ALNS", 0) > 0;  // adaptive LNS selection (alns_select); off: as before
  const bool alns_repick_ = alns_ && knob("MILP_ALNS_REPICK", 0) > 0;  // see run_sub_mips
  std::atomic<long long> arm_calls_[kLns] = {}, arm_found_[kLns] = {}, arm_iters_[kLns] = {};  // MILP_ALNS statistics
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
  // (path_bytes_), stored warm-start bases (basis_bytes_) and the queue (NodePool::bytes: what
  // the open nodes use; alloc_bytes: what the queue holds, incl. free slots and spare capacity).
  // used = tree memory + the resident size when the tree search started (opportunistic mode).
  // States (mem_state_): 0 normal; 1 from 60% of L: new nodes get no warm-start basis, cut-off
  // nodes are purged from the queue when the cutoff improves, the queue compacts below half
  // occupancy; 2 (drain) from 75% of L or 85% allocated or resident: queued nodes lose their bases
  // (shallowest first) down to 65%, pops alternate k deepest : 1 best bound (NodePool; k adapts so
  // that the open-node count falls slowly) and workers always plunge, so the bound keeps moving
  // while the tree shrinks; it ends below 70% (allocated below 80%, resident below 75%). The search stops
  // only at 95% of L (allocated or resident). Deterministic mode decides on the tree memory alone,
  // at round barriers (det_tree_bytes_), so the outcome does not depend on timing; rounds are
  // sized to stay below these thresholds (run_deterministic, project).
  std::atomic<double> rss_checked_{-1.0};
  std::atomic<long long> rss_bytes_{0};
  std::atomic<long long> path_bytes_{0}, basis_bytes_{0};
  long long base_rss_ = 0;
  long long det_tree_bytes_ = 0;
  std::atomic<int> mem_state_{0};
  std::atomic<long long> mem_stripped_{0}, mem_drains_{0}, mem_peak_{0};
  std::atomic<double> mem_relief_t_{-1e9}, mem_strip_t_{-1e9};
  std::mutex mem_mu_;
  // Drain phases (statistics, MipResult): exits, the bound at the last entry, the bound rise over
  // the finished phases. Queued nodes purged at an improved cutoff.
  long long mem_drain_exits_ = 0, mem_purged_ = 0;
  double drain_entry_lb_ = -kInf, drain_rise_ = 0.0;
  std::atomic<double> purge_cutoff_{kInf}, purge_t_{-1e9};
  long long mem_limit_bytes() const {
    return opts_.mem_limit_bytes > 0 ? opts_.mem_limit_bytes : static_cast<long long>(params_.mem_limit_mb) << 20;
  }
  long long tree_bytes() const { return pool_->bytes() + path_bytes_.load() + basis_bytes_.load() + 12 * ncp_held_nnz_.load(); }
  long long tree_bytes(const Worker& w) const { return w.det ? det_tree_bytes_ : tree_bytes(); }
  long long tree_alloc_bytes() const {
    return pool_->alloc_bytes() + path_bytes_.load() + basis_bytes_.load() + 12 * ncp_held_nnz_.load();
  }
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
    if (tree_alloc_bytes() + (det ? 0 : base_rss_) > limit / 20 * 19) return true;
    return checked_rss() > limit / 20 * 19;
  }
  // Memory budget of a sub-MIP started now: a quarter of what is left (16..256 MB).
  int sub_mip_mem_mb(const Worker* w) const {
    const long long left = mem_limit_bytes() - (w ? tree_bytes(*w) : tree_bytes()) - (w && w->det ? 0 : base_rss_);
    return static_cast<int>(std::clamp<long long>((left / 4) >> 20, 16, 256));
  }
  // Warm-start bases: the reference of the sparse form (set before the workers start, never
  // changed afterwards), storing a basis (accounting, pool holds) and its release (BasisOwner).
  std::shared_ptr<const std::vector<VarStatus>> ref_basis_;
  BasisRef store_basis(const Basis& b, Index size, const std::vector<std::pair<int, VarStatus>>* pool_nb);
  void basis_released(const PackedBasis& b) override;
  // Bound paths (Node): the segment a branching node's children share (its own change, then its
  // reduced-cost fixings if with_rcfix; its path itself when there is neither), and a child's own
  // change (inline, or a segment of its own with the extra changes in extra).
  PathRef branch_segment(Worker& w, const Node& node, bool with_rcfix);
  void set_child_change(Node& c, const PathRef& base, const BoundChange& bc, const std::vector<uint8_t>* extra);
  std::atomic<bool> root_rc_ready_{false};  // set once root_d_/root_at_ are final (FJ thread reads it)
  void root_rc_fixing(double cutoff);
  // Global objective-cutoff propagation (in root_rc_fixing, under gfix_mu_): the propagator of the
  // global bounds and the columns with a nonzero cost (its seeds).
  std::unique_ptr<DomainPropagator> gprop_;
  std::vector<Index> obj_cols_;
  // Objective propagation with clique implications (search/obj_cliques.hpp) on bounds that hold for improving
  // points: in root_rc_fixing and between the root cut rounds. Built at the first call (under gfix_mu_; null
  // when the rows give no conflicts). MILP_OBJ_CLIQUES=0: off.
  std::unique_ptr<ObjectiveCliques> ocl_;
  bool ocl_built_ = false;
  long long ocl_calls_ = 0, ocl_fixed_ = 0;
  double ocl_time_ = 0.0;
  double ocl_cut_cutoff_ = kInf;  // the cutoff of the last call between the root cut rounds
  Index obj_clique_fix(double cutoff, std::vector<double>& lb, std::vector<double>& ub, std::vector<Index>& changed);
  std::vector<Index> ocl_new_;  // columns root_rc_fixing fixed by it since the root's last check (ocl_root_redo)
  int ocl_redos_ = 0;
  bool ocl_root_redo(Worker& w);
  void record_lurking(const DualSimplex& lp);
  void dbg_global(const char* what);  // debug: report global bounds excluding the debug point
  // In-tree restart: once enough integer columns are fixed globally during the tree (root
  // reduced-cost fixing with a better incumbent, conflict propagation), stop the tree and hand the
  // problem with these fixings, the root cut rows and the most recently used pool cuts back for a
  // fresh presolve and root (SCIP's restarts). Checked by worker 0 every 128 nodes, or at the round
  // barriers in deterministic mode (from deterministic data only), and once the root heuristics beside
  // the tree are done (MILP_ROOT_HEUR_ASYNC; under tree_restart_mu_).
  // Estimation-driven restart (v0.6 R2, opt-in MILP_EST_RESTART=1): also when the tree-size estimate says
  // the tree is huge and it fixed at least a few columns (est_restart_vote).
  void check_tree_restart(bool det);
  // One vote of the tree-size estimator (HiGHS's rule, as in the 2026-11-03 design): over the window
  // since the last check that did not find the tree huge, the tree weight W (TreeWeight) went from
  // W_ref to W in N - N_ref nodes, so the secant estimate of the nodes for W = 1 is N^ = N_ref +
  // (N - N_ref)(1 - W) / (W - W_ref). Huge: N^ >= MILP_EST_FACTOR (20) x r x N while the gap shrank
  // by less than 5% / r in the window (r = (1 - share of the integer columns fixed)^2). Returns the
  // consecutive huge votes, 0 after a reset.
  int est_restart_vote(Index fixed, Index free_ints);
  void fill_restart(MipResult& res, bool with_pool);
  Index count_global_fixed();
  std::atomic<bool> tree_restart_{false};
  std::mutex tree_restart_mu_;
  // Set by root reduced-cost fixing; with MILP_ROOT_HEUR_ASYNC the next worker to finish a node checks for
  // an in-tree restart (not only worker 0 every 128 of its nodes: mc11's nodes are slow, ~4 s).
  std::atomic<bool> restart_check_due_{false};
  bool est_restart_ = false;  // the tree restart came from the estimate
  Index tree_fixed0_ = 0;  // integer columns fixed globally (not by the problem) when the tree started
  TreeWeight tw_;          // leaves closed so far (worker loops, det task loop, pool drops)
  // Estimator window (worker 0 / the barrier thread only).
  long long est_n_ref_ = -1, est_last_ = -1;
  double est_w_ref_ = 0.0, est_gap_ref_ = kInf, est_nhat_ = kInf;
  int est_votes_ = 0;

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
  std::atomic<long long> ncp_never_bind_{0}, ncp_flat_rounds_{0};  // statistics (verbosity 2)
  // MILP_POOL_DEDUP=1: live pool cuts by coefficient hash; a published cut with the coefficients of a
  // live one and no smaller right-hand side maps to that cut's id instead of a new entry.
  std::unordered_map<uint64_t, int> ncp_by_hash_;
  std::vector<uint64_t> ncp_hash_;  // per id (dedup only)
  long long ncp_dups_ = 0;
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
  // Cut-based conflict analysis (genres.hpp), MILP_GENRES: 0 off, 1 with the clauses (default), 2 clauses
  // only when resolution fails. Statistics: calls, learned (by outcome), steps, reductions, MIR, nonzeros.
  int genres_ = static_cast<int>(knob("MILP_GENRES", 1));
  std::atomic<long long> gr_calls_{0}, gr_learned_{0}, gr_uip_{0}, gr_partial_{0}, gr_steps_{0}, gr_reduced_{0},
      gr_mir_{0}, gr_nnz_{0}, gr_ns_{0}, gr_fail_[9] = {};
  std::atomic<long long> sb_probes_{0}, sb_iters_{0};  // strong branching statistics
  std::atomic<long long> sb_pl_stops_{0};               // probabilistic lookahead stops (MILP_PL)
  std::atomic<long long> sb_aps_nodes_{0};              // branchings scored with ancestral pseudocosts
  std::atomic<long long> sb_flat_nodes_{0};             // branchings scored without pseudocost gains (flat objective)
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
  // Node cut time (ns, pool mode): fresh separation, pool scans, cut-round re-solves, first node LP solves;
  // node LP rows summed over the pool-mode nodes. MILP_POOL_TRACE=1 logs them per progress line.
  std::atomic<long long> sep_ns_{0}, scan_ns_{0}, cutlp_ns_{0}, nodelp_ns_{0}, lprows_sum_{0};
  static constexpr size_t kSepStats = 5;  // separate_node: cover, c-MIR, clique, implied bound, tableau c-MIR
  std::atomic<long long> sep_k_ns_[kSepStats] = {};
  std::atomic<long long> sep_calls_{0};
  struct TraceSnap {
    long long n = 0, prow = 0, rows = 0, it = 0, cit = 0, add = 0, rem = 0, sep = 0, scan = 0, cutlp = 0, nodelp = 0;
  } trace_last_;
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
  // Primal heuristic statistics (the "Heuristics" line): tree dives (calls, points found, new
  // incumbents, LP iterations), the root feasibility pump's LP iterations, and the tree's sub-MIPs
  // (calls, new incumbents, LP iterations, of them in the sub-MIPs' dives and pumps).
  std::atomic<long long> tdive_calls_{0}, tdive_found_{0}, tdive_wins_{0}, tdive_limit_{0}, tdive_iters_{0}, fp_iters_{0};
  std::atomic<long long> hsub_calls_{0}, hsub_wins_{0}, hsub_iters_{0}, hsub_dive_iters_{0}, hsub_fp_iters_{0};
  std::atomic<long long> hsub_dive_wins_{0}, hsub_fp_runs_{0}, hsub_fp_wins_{0};
  std::atomic<long long> zi_calls_{0}, zi_found_{0}, zi_wins_{0};  // ZI rounding at nodes (statistics)
  // New incumbents of this search from dives / the feasibility pump, and pumps run (MipResult).
  std::atomic<int> dive_wins_{0}, fp_runs_{0}, fp_wins_{0};
  void count_win(const char* source) {
    if (!source) return;
    if (std::strstr(source, "diving")) ++dive_wins_;
    else if (std::strncmp(source, "feasibility pump", 16) == 0) ++fp_wins_;
  }
  std::atomic<bool> stop_{false};
  Status stop_reason_ = Status::Unknown;
  std::mutex stop_mu_;
  // SearchOptions::solution_limit: incumbents accepted so far (guarded by inc_mu_), and the limit was hit.
  int solutions_ = 0;
  std::atomic<bool> sol_limit_hit_{false};
  // Trivial objective bound: sum over the columns of min(c_j lb_j, c_j ub_j) (without the offset; -inf
  // if a column with a nonzero cost is unbounded in its improving direction; 0 for a zero objective),
  // the search's initial global dual bound. An incumbent that meets it (gap test) solves the problem:
  // trivial_solved ends the search, and its status is Optimal whatever stopped it meanwhile (an LP that
  // returned at the interrupt, a time limit). MILP_TRIVIAL_BOUND=0: off (-inf).
  double trivial_lb_ = -kInf;
  bool zero_obj_ = false;  // every cost is zero (a feasibility problem): node LP bounds never move
  std::atomic<bool> solved_{false};
  // The search's interrupt flag (core/interrupt.hpp, SearchStopScope), opportunistic mode only:
  // trivial_solved sets it to stop the LPs, cut loop and heuristics running for the search at once
  // (ns1116954: local MIP finds an optimal point of its zero objective at 3 s, the 131k-row root LP
  // then ran for 50 s). Deterministic mode accepts incumbents at deterministic points only, so stop_
  // and solved_ are set there; the flag would cut concurrent work (sub-MIP jobs, FJ) at
  // timing-dependent points.
  std::atomic<bool> solved_stop_{false};
  // Under inc_mu_. root: the incumbent meets the root LP bound instead (opportunistic mode, MILP_ROOT_GAP_STOP).
  void trivial_solved(double obj, bool root = false);
  std::atomic<bool> root_unbounded_{false}, numerical_{false};
  std::atomic<long long> lp_failed_{0}, lp_recovered_{0}, lp_unsolved_{0}, fixed_recovered_{0};
  std::atomic<long long> impl_branches_{0};  // branchings on violated implications
  // verified_point's fixed-integer re-solves: tried, succeeded, total LP iterations (statistics)
  mutable std::atomic<long long> polish_tried_{0}, polish_ok_{0}, polish_iters_{0};
  // Quadratic rows: verified_point's fixed-integer repairs remembered, so a repeated candidate costs nothing:
  // portqc_hard_1 in deterministic mode (no node outer approximation) repaired the same points node after node, each
  // with conic solves (~1 s per node; TL at 60 s, 2.2 s with the caches). Successes by integer assignment (the repaired
  // point is feasible for it whatever started the repair); failures by the whole point (the repair's tangents start
  // from its continuous part, and another start may succeed: a failure kept by assignment made an all-integers-fixed
  // node be dropped).
  struct Repair {
    std::vector<double> key, x;  // the assignment (successes) or the whole point (failures); the repaired point
  };
  mutable std::mutex repair_mu_;
  mutable std::unordered_map<uint64_t, Repair> repair_cache_, repair_failed_;
  mutable std::atomic<long long> repair_hits_{0};
  std::atomic<long long> oa_rounds_{0}, oa_cuts_{0};  // outer approximation of quadratic rows at nodes
  std::atomic<long long> lp_polished_{0};  // integral node LP points re-solved warm (polish_integral_lp)
  double lost_bound_ = kInf;  // min bound of subtrees dropped for numerical reasons
  std::atomic<double> root_bound_{-kInf};  // also read by try_incumbent (incumbent callback) on other threads
  double root_cs_before_ = -kInf;  // root LP bound before cuts (ROOTSTAT line)
  double root_lp_lb_ = -kInf;  // best objective of the root cut loop's optimal LPs (a global bound)
  long long root_lp0_iters_ = -1;  // LP iterations of the root LP's first solve (before cuts; -1: not recorded)
  // Root LP solved by the barrier: dual simplex iterations that stand for its effort (w.lp.iterations() then
  // counts the crossover cleanup and the cut loop only); 0 otherwise.
  long long root_lp_equiv_iters_ = 0;
  // Concurrent root won by the dual simplex: its iterations, for the cut loop (its warm initial solve takes none).
  long long root_lp_race_iters_ = 0;
  double root_barrier_time_ = 0.0;  // seconds of a root LP solved by the barrier (0: the simplex solved it)
  int root_cs_rounds_ = 0, root_cs_cuts_ = 0;
  double det_lb_ = -kInf;  // deterministic mode: global bound after the last round's merge
  std::atomic<bool> det_running_{false};  // run_deterministic() runs (try_incumbent is then called by its thread only)
  std::mutex log_mu_;
  double last_log_ = 0.0, last_mem_log_ = 0.0;
  void log_memory();
};

LpStatus Search::solve_fixed_ints(const std::vector<double>& x_int, const std::vector<double>& lb_in,
                                  const std::vector<double>& ub_in, double primal_tol, double cutoff,
                                  std::vector<double>& out, double time_cap) const {
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
  if (!params_.deterministic) {
    o.time_limit = time_cap >= 0.0 ? time_cap : std::max(0.0, std::min(5.0, params_.time_limit - timer_.elapsed()));
  }
  LpStatus st = lp.solve(o);
  polish_iters_ += lp.iterations();
  // Quadratic rows: the outer approximation of the continuous part (gradient cuts at the point as rows, re-solved).
  static const int kOaFixRounds = static_cast<int>(knob("MILP_OA_FIX_ROUNDS", 50));
  const double oa_end = timer_.elapsed() + o.time_limit;  // the rounds share the first solve's time limit
  for (int round = 0; st == LpStatus::Optimal && !p_.quad_rows.empty() && round < kOaFixRounds; ++round) {
    std::vector<Cut> cuts;
    if (quad_oa_cuts(p_, lp.x(), 1e-8, cuts) == 0) break;
    std::vector<Index> start{0}, index;
    std::vector<double> value, rlb, rub;
    for (const Cut& c : cuts) {
      index.insert(index.end(), c.idx.begin(), c.idx.end());
      value.insert(value.end(), c.val.begin(), c.val.end());
      start.push_back(static_cast<Index>(index.size()));
      rlb.push_back(-kInf);
      rub.push_back(c.rhs);
    }
    lp.add_rows(static_cast<Index>(rlb.size()), start, index, value, rlb, rub);
    if (is_finite(o.time_limit)) {
      o.time_limit = oa_end - timer_.elapsed();
      if (o.time_limit <= 0.0) {
        st = LpStatus::TimeLimit;
        break;
      }
    }
    const long long it0 = lp.iterations();
    st = lp.solve(o);
    polish_iters_ += lp.iterations() - it0;
  }
  if (st == LpStatus::Optimal) {
    out = lp.col_values();
    for (Index j = 0; j < p_.num_cols(); ++j) {
      if (p_.is_int[j]) out[j] = lb[j];  // exact integers
    }
  }
  // The rounds can stop short of the quadratic rows' tolerance (a cone's apex, where the norm form is judged and
  // tangents converge slowly: estein4_A's all-integers-fixed nodes were dropped as failed LPs, the result NUMERIC):
  // then the continuous part by the conic method, exactly (its infeasibility certificate settles the subproblem too).
  if (!p_.quad_rows.empty() && (st != LpStatus::Optimal || p_.max_violation(out) > 0.1 * kFeasTol) &&
      st != LpStatus::Infeasible && st != LpStatus::CutoffReached) {
    MipProblem q = p_;
    q.col_lb = lb;
    q.col_ub = ub;
    ConicProblem cp;
    if (conic_relaxation(q, cp)) {
      ConicOptions co;
      co.time_limit = is_finite(o.time_limit) ? std::max(0.1, o.time_limit) : kInf;
      const ConicResult cr = solve_conic(cp, co);
      if (cr.status == ConicStatus::Optimal) {
        std::vector<double> y(cr.x.begin(), cr.x.begin() + p_.num_cols());
        for (Index j = 0; j < p_.num_cols(); ++j) {
          if (p_.is_int[j]) y[j] = lb[j];
        }
        // Within the feasibility tolerance and better than the tangents' point (a reduced-accuracy exit can miss 1e-7:
        // portqc_hard_1's all-integers-fixed node 1927, deterministic mode, kept the tangents' 2.5e-4 and was dropped).
        if (p_.max_violation(y) <= kFeasTol && (st != LpStatus::Optimal || p_.max_violation(y) < p_.max_violation(out))) {
          out = std::move(y);
          st = LpStatus::Optimal;
        }
      } else if (cr.status == ConicStatus::PrimalInfeasible) {
        st = LpStatus::Infeasible;
      }
    }
  }
  return st;
}

void Search::final_polish(MipResult& res) const {
  // The incumbent may use the feasibility tolerance (rows up to 1e-6 beyond their sides: LP points and sub-MIP
  // solutions are accepted within it), and its objective may profit from it: map16715-04 (2 threads, seed 1)
  // ended at -111.00075 with 257 rows exactly 1e-6 beyond their sides (optimum -111; the fixed-integer LP gives
  // -111), fastxgemm-n2r6s0t2 at 229.9997 (optimum 230). The search's own result (not in sub-MIPs): its continuous
  // part re-solved with the integers fixed at a 1e-9 primal tolerance, taken if it violates less; its objective
  // may be a little worse, the honest one. Once, under a short time cap that may pass the time limit (at most
  // max(0.5 s, 1% of it), 2 s; deterministic mode: the LP's iteration limit). MILP_FINAL_POLISH=0: off.
  static const bool on = knob("MILP_FINAL_POLISH", 1) > 0;
  if (!on || !opts_.sub_mip_heuristics || res.x.empty()) return;
  bool cont = false;
  for (Index j = 0; j < p_.num_cols() && !cont; ++j) cont = !p_.is_int[j];
  const double v0 = p_.max_violation(res.x);
  if (!cont || v0 <= 1e-9) return;
  const double cap = std::min(2.0, std::max(0.5, 0.01 * (is_finite(params_.time_limit) ? params_.time_limit : 200.0)));
  std::vector<double> y;
  if (solve_fixed_ints(res.x, p_.col_lb, p_.col_ub, 1e-9, kInf, y, cap) != LpStatus::Optimal) return;
  const double v1 = p_.max_violation(y);
  if (!(v1 < v0) || v1 > kFeasTol) return;
  const double obj = p_.objective(y);
  log_.log(2, "  final polish: violation %.3g -> %.3g, objective %.12g -> %.12g\n", v0, v1,
           p_.display_sense * res.objective, p_.display_sense * obj);
  res.x = std::move(y);
  res.objective = obj;
}

void Search::polish_integral_lp(Worker& w, const LpOptions& opt) {
  // The node LP's point is integral and feasible within its primal tolerance (1e-6), which postsolve
  // can amplify past the model check (dano3_5: 2.7e-6 after undoing substitutions). The checks that
  // repair such points re-solve the fixed-integer LP cold, ~30k iterations on dano3_5's 13.9k
  // columns: verified_point's ran into its 5 s cap twice, postsolve's into the end of the time limit,
  // and the solve was repeated without presolve and ended with no solution. Warm from the node LP's
  // basis the same re-solve takes a few iterations.
  std::vector<double> r = w.sol;
  for (Index j = 0; j < p_.num_cols(); ++j) {
    if (p_.is_int[j]) r[j] = std::round(r[j]);
  }
  const double v0 = p_.max_violation(r);
  if (v0 <= 1e-9) return;
  DualSimplex& lp = w.lp;
  DualSimplex::State st;
  lp.save_state(st);
  std::vector<std::pair<Index, std::pair<double, double>>> saved;
  bool ok = true;
  for (Index j = 0; j < p_.num_cols(); ++j) {
    if (!p_.is_int[j]) continue;
    const double lo = lp.col_lower(j), hi = lp.col_upper(j);
    if (r[j] < lo - 1e-9 || r[j] > hi + 1e-9) ok = false;
    if (lo == r[j] && hi == r[j]) continue;
    saved.push_back({j, {lo, hi}});
    lp.set_col_bounds(j, r[j], r[j]);
  }
  if (ok) {
    LpOptions o = opt;
    o.tol.primal_feas = 1e-9;
    o.objective_cutoff = kInf;
    o.iteration_limit = 1000 + static_cast<long long>(p_.num_rows());
    if (lp.solve(o) == LpStatus::Optimal) {
      std::vector<double> y = lp.col_values();
      y.resize(p_.num_cols());
      for (Index j = 0; j < p_.num_cols(); ++j) {
        if (p_.is_int[j]) y[j] = r[j];
      }
      if (p_.max_violation(y) < v0) {
        w.sol = std::move(y);
        ++lp_polished_;
      }
    }
  }
  for (const auto& [j, b] : saved) lp.set_col_bounds(j, b.first, b.second);
  lp.restore_state(st);
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
  // The objective's epigraph row (outer approximation of a quadratic objective): eta raised to f(x), exactly feasible
  // at the point's true objective.
  for (const QuadRow& q : p_.quad_rows) {
    if (q.epigraph < 0) continue;
    const double a = q.activity(x.data());
    if (a > q.hi) x[q.epigraph] += a - q.hi;
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
    double lin = 0.0;
    const double v_in = p_.max_violation(x_in), obj_in = p_.objective(x_in) - p_.offset;
    const bool in_ok = v_in <= kFeasTol;
    // Quadratic rows: a point that only violates them (its linear rows, bounds and integers within tolerance once
    // rounded) gets its continuous part from the fixed-integer outer approximation (solve_fixed_ints).
    const bool quad_only = !p_.quad_rows.empty() && (p_.max_violation(x, nullptr, &lin), lin <= kFeasTol);
    if (in_ok ? obj_in < repair_below : (v_in <= kPolishTol || quad_only) && obj_in < polish_below) {
      std::vector<double> ints;
      uint64_t key = 0, xkey = 0;
      bool cached = false;
      auto mix = [](uint64_t h, uint64_t v) { return (h ^ (v + 0x9e3779b97f4a7c15ULL)) * 0x100000001b3ULL; };
      if (quad_only) {
        for (Index j = 0; j < p_.num_cols(); ++j) {
          uint64_t bits;
          std::memcpy(&bits, &x[j], sizeof bits);
          xkey = mix(xkey, bits);
          if (!p_.is_int[j]) continue;
          ints.push_back(x[j]);
          key = mix(key, bits);
        }
        std::lock_guard<std::mutex> lk(repair_mu_);
        if (const auto it = repair_cache_.find(key); it != repair_cache_.end() && it->second.key == ints) {
          fixed = it->second.x;
          cached = true;
        } else if (const auto jt = repair_failed_.find(xkey); jt != repair_failed_.end() && jt->second.key == x) {
          cached = true;
        }
        if (cached) ++repair_hits_;
      }
      if (!cached) {
        ++polish_tried_;
        for (const double tol : {1e-9, 1e-7}) {
          const LpStatus fs = solve_fixed_ints(x, p_.col_lb, p_.col_ub, tol, kInf, fixed);
          if (fs == LpStatus::Optimal && p_.max_violation(fixed) <= kFeasTol) break;
          fixed.clear();
        }
        if (quad_only) {
          std::lock_guard<std::mutex> lk(repair_mu_);
          if (repair_cache_.size() + repair_failed_.size() >= 20000) repair_cache_.clear(), repair_failed_.clear();
          if (!fixed.empty()) repair_cache_[key] = Repair{std::move(ints), fixed};
          else repair_failed_[xkey] = Repair{x, {}};
        }
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
  count_win(source);
  return true;
}

bool Search::try_incumbent(const std::vector<double>& x_in, const char* source) {
  std::vector<double> x;
  double obj;
  const double below = std::min(incumbent_.load(), external_cutoff_);
  if (!verified_point(x_in, x, &obj, below, below)) return false;
  if (opts_.on_incumbent) {
    std::vector<double> reported;
    {
      std::lock_guard<std::mutex> lk(inc_mu_);
      if (!accept_incumbent(x, obj, source)) return false;
      reported = inc_x_;
    }
    // Lower bound: the open nodes' (the root bound before the tree exists or while it is empty).
    // Deterministic mode: the bound after the last merge; the pool alone misses the nodes of the
    // round being merged (taken out of the pool, their children not pushed yet).
    // Subtrees dropped for numerical reasons still limit it.
    double plb = det_running_.load() ? det_lb_ : pool_ ? pool_->lower_bound() : kInf;
    {
      std::lock_guard<std::mutex> lk(stop_mu_);
      plb = std::min(plb, lost_bound_);
    }
    const double rb = root_bound_.load();
    const double lb = std::max(is_finite(plb) ? std::max(plb, rb) : rb, trivial_lb_);
    opts_.on_incumbent(reported, std::min(lb, obj) + p_.offset);
    return true;
  }
  std::lock_guard<std::mutex> lk(inc_mu_);
  return accept_incumbent(x, obj, source);
}

bool Search::accept_incumbent(std::vector<double>& x, double obj, const char* source) {
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
  count_win(source);
  log_.log(2, "  new incumbent %.10g (%s) at %.2fs\n", p_.display_sense * (obj + p_.offset), source, timer_.elapsed());
  if (opts_.on_trace) opts_.on_trace(p_.display_sense * (obj + p_.offset), source, timer_.elapsed());
  // MILP_ROOT_GAP_STOP (default 1; 0: off): opportunistic mode, an incumbent at the root LP bound ends the search
  // at once, also while root heuristics still run (triptim1: after the vector-length dive found the optimum, the
  // coefficient dive ran on for 3.3 s to its 25 s cap). Deterministic mode: the root node's own test.
  static const bool root_gap_stop = knob("MILP_ROOT_GAP_STOP", 1) > 0;
  if (gap_closed_at(trivial_lb_, obj)) trivial_solved(obj);
  else if (root_gap_stop && !params_.deterministic && gap_closed_at(root_bound_.load(), obj)) trivial_solved(obj, true);
  if (opts_.solution_limit > 0 && ++solutions_ >= opts_.solution_limit) {
    sol_limit_hit_ = true;  // reported as Interrupted (search())
    stop_ = true;
  }
  return true;
}

void Search::trivial_solved(double obj, bool root) {
  if (solved_.exchange(true)) return;
  const double bound = root ? root_bound_.load() : trivial_lb_;
  log_.log(1, "Incumbent %.10g meets the %s %.10g: optimal (%.2fs)\n", p_.display_sense * (obj + p_.offset),
           root ? "root LP bound" : "trivial objective bound", p_.display_sense * (bound + p_.offset), timer_.elapsed());
  stop_ = true;
  if (!params_.deterministic) {
    fj_stop_ = true;
    solved_stop_ = true;
  }
}

void Search::maybe_log(bool force) {
  if (log_.verbosity() < 1) return;
  std::lock_guard<std::mutex> lk(log_mu_);
  const double t = timer_.elapsed();
  if (!force && t - last_log_ < 2.0) return;
  last_log_ = t;
  double lb = std::max(std::min(pool_->lower_bound(), incumbent_.load()), trivial_lb_);
  if (obj_integral_ && is_finite(lb)) lb = std::ceil(lb - 1e-6);
  const double inc = incumbent_.load();
  const double gap = is_finite(inc) ? std::abs(inc - lb) / std::max(std::abs(inc + p_.offset), 1e-10) * 100.0 : kInf;
  log_.log(1, "%10lld nodes %8zu open  bound %14.8g  best %14.8g  gap %7.3f%%  %7.1fs\n", nodes_.load(),
           pool_->size(), p_.display_sense * (lb + p_.offset), p_.display_sense * (inc + p_.offset), gap, t);
  static const bool trace = knob("MILP_POOL_TRACE", 0.0) > 0.0;
  if (trace && use_pool_) {
    // Since the last line, per pool-mode node: LP rows (pool rows), first-solve and cut-round LP iterations,
    // pool rows added / removed, and the node time split (ms per node).
    TraceSnap s{popped_n_.load() + plunged_n_.load(), prow_sum_.load(), lprows_sum_.load(), node_it_total_.load(),
                pool_it_total_.load(), ncp_added_.load(), ncp_removed_.load(), sep_ns_.load(), scan_ns_.load(),
                cutlp_ns_.load(), nodelp_ns_.load()};
    const TraceSnap& o = trace_last_;
    const double k = static_cast<double>(std::max(1LL, s.n - o.n));
    size_t live = 0;
    {
      std::lock_guard<std::mutex> lk2(ncp_mu_);
      live = ncp_live_;
    }
    log_.log(2, "  pool trace: %lld nodes, %.0f LP rows (%.0f pool), %.1f + %.1f LP it/node, +%.1f -%.1f pool rows/node,"
             " ms/node: LP %.3f, cut LPs %.3f, separation %.3f, pool scan %.3f; pool %zu cuts\n",
             s.n - o.n, (s.rows - o.rows) / k, (s.prow - o.prow) / k, (s.it - o.it) / k, (s.cit - o.cit) / k,
             (s.add - o.add) / k, (s.rem - o.rem) / k, 1e-6 * (s.nodelp - o.nodelp) / k, 1e-6 * (s.cutlp - o.cutlp) / k,
             1e-6 * (s.sep - o.sep) / k, 1e-6 * (s.scan - o.scan) / k, live);
    trace_last_ = s;
  }
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
  log_.log(2, "  memory: resident %.0f MB (at tree start %.0f), tree %.0f MB (state %d, allocated %.0f MB) = queue %.0f"
              " (allocated %.0f) + bases %.0f + paths %.0f + held cuts %.0f; open %zu (depth %.0f, %.0f B per node: queue %.0f,"
              " bases %.0f, paths %.0f): %lld with a basis (%lld sparse, %.1f pool rows), %lld with groups (%.0f MB);"
              " drain k %d; pool %lld cuts / %lld nnz; conflict log %zu; %d integers fixed globally\n",
           resident_bytes() * mb, base_rss_ * mb, tree_bytes() * mb, mem_state_.load(), tree_alloc_bytes() * mb,
           pool_->bytes() * mb, pool_->alloc_bytes() * mb, basis_bytes_.load() * mb, path_bytes_.load() * mb,
           12.0 * ncp_held_nnz_.load() * mb, b.nodes, b.depth, tree_bytes() * per, pool_->bytes() * per,
           basis_bytes_.load() * per, path_bytes_.load() * per, b.with_basis, b.sparse_bases,
           b.with_basis ? static_cast<double>(b.pool_rows) / b.with_basis : 0.0, b.groups, b.group_bytes * mb,
           pool_->drain_ratio(), cut_live, cut_nnz, conf, static_cast<int>(count_global_fixed()));
}

void Search::memory_control(bool det) {
  const long long limit = mem_limit_bytes();
  const long long base = det ? 0 : base_rss_;
  long long rss = det ? 0 : checked_rss();
  const double now = det ? 0.0 : timer_.elapsed();
  // The allocator keeps freed memory (old slot / key arrays, freed bases) resident for reuse; hand
  // it back to the OS before it counts against the limit (at most every 2 s), and after the queue
  // compacted under memory pressure.
  const bool compacted = pool_->take_compacted();
  if (compacted || (!det && rss >= limit / 20 * 16 && now - mem_relief_t_.load() >= 2.0)) {
    if (!det) mem_relief_t_ = now;
    release_free_memory();
    if (!det) {
      rss_checked_ = -1.0;
      rss = checked_rss();
    }
  }
  int state = mem_state_.load();
  // Under memory pressure (and in a large queue) drop the queued nodes the cutoff has pruned since
  // the last purge (at most every 2 s; deterministic mode: at the barriers, in memory states only).
  const double co = cutoff_.load();
  if (co < purge_cutoff_.load() && (state >= 1 || (!det && pool_->approx_size() > 1000000)) &&
      (det || now - purge_t_.load() >= 2.0)) {
    std::unique_lock<std::mutex> lk(mem_mu_, std::try_to_lock);
    if (lk.owns_lock() && co < purge_cutoff_.load()) {
      purge_cutoff_ = co;
      purge_t_ = now;
      const size_t before = pool_->approx_size();
      const long long removed = pool_->purge(co);
      mem_purged_ += removed;
      if (removed > 0) {
        log_.log(2, "  memory: purged %lld of %zu queued nodes at the cutoff %.10g (%.1fs)\n", removed, before,
                 p_.display_sense * (co + p_.offset), timer_.elapsed());
      }
    }
  }
  // used: what the open nodes use; alloc: incl. the queue's free slots and spare capacity (the
  // hard stop applies to it), which counts like the resident size: drain from 85%, exit below 80%.
  long long used = tree_bytes() + base;
  const long long alloc = tree_alloc_bytes() + base;
  if (alloc - base > mem_peak_.load()) mem_peak_ = alloc - base;  // statistics (racy maximum)
  const long long big = std::max(alloc, rss);
  int want = used >= limit / 20 * 15 || big >= limit / 20 * 17 ? 2 : used >= limit / 20 * 12 ? 1 : 0;
  if (state == 2 && want < 2) {
    // Leaving drain mode brings the estimate heap back: exit only if that stays below 70% / 80%.
    const long long grow = pool_->exit_growth();
    if (used + grow >= limit / 20 * 14 || alloc + grow >= limit / 20 * 16 || rss >= limit / 20 * 15) want = 2;
  }
  // Bytes to free: down to 65% of the limit used, 75% allocated / resident.
  const long long excess = std::max(used >= limit / 20 * 15 ? used - limit / 20 * 13 : 0,
                                    big >= limit / 20 * 17 ? big - limit / 20 * 15 : 0);
  const bool strip = want == 2 && excess > 0 && basis_bytes_.load() > limit / 100 && (det || now - mem_strip_t_.load() >= 1.0);
  // Close to the hard stop (85% used, 90% allocated or resident): drain pops all deepest.
  const bool deep_only = want == 2 && (used >= limit / 20 * 17 || big >= limit / 20 * 18);
  if (want == state && !strip && deep_only == pool_->deep_only()) {
    if (det) det_tree_bytes_ = tree_bytes();
    return;
  }
  std::lock_guard<std::mutex> lk(mem_mu_);
  if (deep_only != pool_->deep_only()) pool_->set_deep_only(deep_only);
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
  state = mem_state_.load();
  if (want != state) {
    const double lb = pool_->lower_bound();
    if (want == 2) {
      ++mem_drains_;
      drain_entry_lb_ = lb;
    } else if (state == 2) {
      ++mem_drain_exits_;
      drain_rise_ += std::max(0.0, lb - drain_entry_lb_);
    }
    pool_->set_drain(want == 2);
    pool_->set_pressure(want >= 1);
    log_.log(2, "  memory: %.0f MB used (resident %.0f MB) of %.0f MB at %.1fs, %zu open, bound %.10g: %s\n",
             used / 1048576.0, rss / 1048576.0, limit / 1048576.0, timer_.elapsed(), pool_->approx_size(),
             p_.display_sense * (lb + p_.offset),
             want == 2 ? "drain mode (k deepest : 1 best bound, no stored bases)" : want == 1 ? "no new stored bases" : "normal");
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
  // MILP_NODE_SEP_WORK = f: the c-MIR path aggregation's work budget per call at nodes is f times the
  // root's (4000 max(nnz(A), 20000)); node separation on tr12-30 took 5-18 ms per node, more than its LP.
  static const double kNodeSepWork = knob("MILP_NODE_SEP_WORK", 1.0);
  ctx.work_scale = kNodeSepWork;
  std::vector<Cut> cand;
  for (size_t k = 0; k < w.seps.size(); ++k) {
    if (use_pool_ && k + 1 == w.seps.size() && !tmir) continue;  // the tableau c-MIR separator
    const long long ns = now_ns();
    w.seps[k]->separate(ctx, cand);
    if (k < kSepStats) sep_k_ns_[k] += now_ns() - ns;
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
    // (With MILP_POOL_DEDUP a cut may map to a live pool cut, possibly one this round re-added already.)
    ids.erase(std::remove_if(ids.begin(), ids.end(), [&](int id) { return id < 0 || w.pool_pos[id] >= 0; }),
              ids.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
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
  static const bool dedup = knob("MILP_POOL_DEDUP", 0.0) > 0.0;
  std::lock_guard<std::mutex> lk(ncp_mu_);
  const long long epoch = nodes_.load();
  for (Cut& c : cuts) {
    const long long nnz = static_cast<long long>(c.idx.size());
    if (c.idx.empty()) {
      if (ids) ids->push_back(-1);
      continue;
    }
    uint64_t h = 0;
    if (dedup) {
      // Exact duplicates: fresh node separation runs on the LP point at which the round's pool cuts were
      // chosen, and the root's dropped list holds cuts that were separated again.
      h = 1469598103934665603ULL;
      for (size_t k = 0; k < c.idx.size(); ++k) {
        uint64_t b;
        std::memcpy(&b, &c.val[k], sizeof b);
        h = (h ^ static_cast<uint64_t>(c.idx[k])) * 1099511628211ULL;
        h = (h ^ b) * 1099511628211ULL;
      }
      const auto it = ncp_by_hash_.find(h);
      if (it != ncp_by_hash_.end() && ncp_[it->second]) {
        const Cut& o = *ncp_[it->second];
        if (o.idx == c.idx && o.val == c.val && c.rhs >= o.rhs) {
          if (ids) ids->push_back(it->second);
          ++ncp_dups_;
          continue;
        }
      }
    }
    if (ncp_live_ >= ncp_max_ || ncp_nnz_ + nnz > ncp_max_nnz_) pool_purge();
    if (ncp_live_ >= ncp_max_ || ncp_nnz_ + nnz > ncp_max_nnz_) {
      if (ids) ids->push_back(-1);
      continue;
    }
    if (ids) ids->push_back(static_cast<int>(ncp_.size()));
    if (dedup) {
      ncp_by_hash_[h] = static_cast<int>(ncp_.size());
      ncp_hash_.resize(ncp_.size() + 1);
      ncp_hash_.back() = h;
    }
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
    if (static_cast<size_t>(id) < ncp_hash_.size()) {
      const auto it = ncp_by_hash_.find(ncp_hash_[id]);
      if (it != ncp_by_hash_.end() && it->second == id) ncp_by_hash_.erase(it);
    }
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

BasisRef Search::store_basis(const Basis& b, Index size, const std::vector<std::pair<int, VarStatus>>* pool_nb) {
  PackedBasis* pb = PackedBasis::make(b, size, ref_basis_.get(), pool_nb);
  basis_bytes_ += pb->bytes();
  if (pb->npool() > 0) pool_hold(*pb, 1);
  return BasisRef(pb, this);
}

void Search::basis_released(const PackedBasis& b) {
  if (b.npool() > 0) pool_hold(b, -1);
  basis_bytes_ -= b.bytes();
}

PathRef Search::branch_segment(Worker& w, const Node& node, bool with_rcfix) {
  const bool rc = with_rcfix && !w.rcfix.empty();
  if (node.change.col < 0 && !rc) return node.path;
  w.seg.clear();
  if (node.change.col >= 0) {
    const Index j = node.change.col;
    PathSeg::encode(node.change, p_.col_lb[j], p_.col_ub[j], w.seg);
  }
  if (rc) {
    for (const BoundChange& f : w.rcfix) PathSeg::encode(f, p_.col_lb[f.col], p_.col_ub[f.col], w.seg);
  }
  return PathRef(PathSeg::make(node.path.get(), w.seg, &path_bytes_));
}

void Search::set_child_change(Node& c, const PathRef& base, const BoundChange& bc, const std::vector<uint8_t>* extra) {
  if (extra) {
    c.path = PathRef(PathSeg::make(base.get(), *extra, &path_bytes_));
    c.change = BoundChange{-1, -kInf, kInf};
    return;
  }
  c.path = base;
  // As PathSeg::for_each decodes it: sides at the original bound (or infinite) are infinite.
  const Index j = bc.col;
  c.change = BoundChange{j, bc.lb != p_.col_lb[j] && bc.lb > -kInf ? PathSeg::canonical(bc.lb) : -kInf,
                         bc.ub != p_.col_ub[j] && bc.ub < kInf ? PathSeg::canonical(bc.ub) : kInf};
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
    w.prow_bind.push_back(0);
  }
  w.lp.add_rows(static_cast<Index>(ids.size()), start, index, value, lb, ub);
}

// Removes the pool rows k with del[k] (their logicals must be basic to keep the basis).
void Search::pool_remove_rows(Worker& w, const std::vector<char>& del) {
  std::vector<Index> rows;
  size_t keep = 0;
  long long never = 0;
  for (size_t k = 0; k < w.prow.size(); ++k) {
    if (del[k]) {
      never += !w.prow_bind[k];
      rows.push_back(w.m_base + static_cast<Index>(k));
      w.pool_pos[w.prow[k]] = -1;
      if (w.pool_dead[w.prow[k]]) w.pool[w.prow[k]].reset();
      continue;
    }
    w.prow[keep] = w.prow[k];
    w.prow_age[keep] = w.prow_age[k];
    w.prow_bind[keep] = w.prow_bind[k];
    w.pool_pos[w.prow[keep]] = static_cast<int>(keep);
    ++keep;
  }
  if (rows.empty()) return;
  w.prow.resize(keep);
  w.prow_age.resize(keep);
  w.prow_bind.resize(keep);
  w.lp.remove_rows(rows);
  ncp_removed_ += static_cast<long long>(rows.size());
  ncp_never_bind_ += never;
}

// Before a node's LP solve: drop aged pool rows and install the node's warm-start basis, adding
// the pool rows that were nonbasic in the parent's final LP.
void Search::pool_prepare(Worker& w, const Node& node) {
  DualSimplex& lp = w.lp;
  const Index n = p_.num_cols();
  const bool warm = node.basis && node.parent != w.last_node;
  if (warm && node.basis->size() != n + w.m_base) {  // cannot happen: base rows are fixed in pool mode
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
  // Aging: rows inactive for kAge node LPs leave; beyond the row cap the oldest leave first. Rows that
  // were never binding since they entered leave after kAgeNew (MILP_POOL_AGE_NEW, default kAge): on
  // tr12-30 almost half of the rows that aged out had never been binding (added together with
  // stronger cuts of the same round).
  static const int kAge = static_cast<int>(knob("MILP_POOL_AGE", 10));
  static const int kAgeNew = static_cast<int>(knob("MILP_POOL_AGE_NEW", kAge));
  const size_t cap = static_cast<size_t>(std::clamp<Index>(p_.num_rows(), 100, 1000));
  std::vector<char> del(np, 0);
  size_t ndel = 0;
  for (size_t k = 0; k < np; ++k) {
    if (!w.pool_need[w.prow[k]] && w.prow_age[k] >= (w.prow_bind[k] ? kAge : kAgeNew)) del[k] = 1, ++ndel;
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
  node.basis->unpack(w.unpacked, ref_basis_.get());
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
  // MILP_POOL_SPARSE=1: rank by violation / (active entries x norm over the active entries), as HiGHS's
  // pool does (entries whose column is not at the bound that the coefficient's sign prefers): sparse cuts
  // first. Default: efficacy.
  static const bool sparse_score = knob("MILP_POOL_SPARSE", 0.0) > 0.0;
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
    double score = eff;
    if (sparse_score) {
      double nrm = 0.0;
      int act = 0;
      for (size_t k = 0; k < c.idx.size(); ++k) {
        const Index j = c.idx[k];
        const bool at = c.val[k] > 0 ? x[j] <= w.lp.col_lower(j) + 1e-6 : x[j] >= w.lp.col_upper(j) - 1e-6;
        if (at) continue;
        ++act;
        nrm += c.val[k] * c.val[k];
      }
      score = viol / (std::max(act, 1) * std::sqrt(std::max(nrm, 1e-24)));
    }
    w.pool_cand.emplace_back(-score, static_cast<int>(id));
  }
  if (w.pool_cand.empty()) return 0;
  static const int kMaxAdd = static_cast<int>(knob("MILP_POOL_ADD", 50));
  // MILP_POOL_PAR = p < 1: by decreasing efficacy, a cut is taken only if it is at most p parallel to
  // the cuts taken before it (as the root's selection; 1: the most efficacious kMaxAdd).
  static const double kPar = knob("MILP_POOL_PAR", 1.0);
  std::vector<int> ids;
  if (kPar < 1.0) {
    std::sort(w.pool_cand.begin(), w.pool_cand.end());
    if (w.par_occ.size() < static_cast<size_t>(p_.num_cols())) w.par_occ.resize(p_.num_cols());
    for (const auto& [neg_eff, id] : w.pool_cand) {
      if (static_cast<int>(ids.size()) >= kMaxAdd) break;
      const Cut& c = *w.pool[id];
      w.par_dot.assign(ids.size(), 0.0);
      for (size_t k = 0; k < c.idx.size(); ++k) {
        for (const auto& [s, v] : w.par_occ[c.idx[k]]) w.par_dot[s] += v * c.val[k];
      }
      bool ok = true;
      for (size_t s = 0; s < ids.size() && ok; ++s) {
        ok = std::abs(w.par_dot[s]) <= kPar * w.pool_norm[id] * w.pool_norm[ids[s]];
      }
      if (!ok) continue;
      const int sid = static_cast<int>(ids.size());
      for (size_t k = 0; k < c.idx.size(); ++k) {
        if (w.par_occ[c.idx[k]].empty()) w.par_cols.push_back(c.idx[k]);
        w.par_occ[c.idx[k]].emplace_back(sid, c.val[k]);
      }
      ids.push_back(id);
    }
    for (Index j : w.par_cols) w.par_occ[j].clear();
    w.par_cols.clear();
  } else {
    const size_t take = std::min<size_t>(w.pool_cand.size(), static_cast<size_t>(kMaxAdd));
    std::partial_sort(w.pool_cand.begin(), w.pool_cand.begin() + static_cast<std::ptrdiff_t>(take), w.pool_cand.end());
    for (size_t q = 0; q < take; ++q) ids.push_back(w.pool_cand[q].second);
  }
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
    if (!slack) w.prow_bind[k] = 1;
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
  bool unsafe = false;
  lp_bound(lp, &unsafe, 0);
  if (unsafe) return;  // z + d is no bound if the LP's safe bound does not confirm z
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
  // Objective cutoff propagation on the global bounds: c x <= cutoff and the rows (continuous bounds
  // only as intermediate steps), integer bounds become global like the reduced-cost fixings above
  // (valid for improving points). neos-4413714-turia (min z, z >= d_ik x_ik per customer): z <=
  // cutoff fixes every assignment with d_ik above it, 95.7% of the 190k binaries at the first
  // incumbent 49.73; node propagation found them per node only, and the root restart counted 195
  // reduced-cost fixings (19020 needed). HiGHS restarts there with 96.5% of the columns inactive.
  // With it: restart at 44 s to 8478 columns, two more restarts, optimal 45.3702 at 91 s (Mac, 4
  // threads; before TL at 120 s, incumbent 46.64). MILP_GLOBAL_OBJPROP=0: off.
  static const bool objprop = knob("MILP_GLOBAL_OBJPROP", 1) > 0 && std::getenv("MILP_NO_OBJPROP") == nullptr;
  Index oprop = 0;
  if (objprop && is_finite(cutoff)) {
    if (!gprop_) {
      for (Index j = 0; j < p_.num_cols(); ++j) {
        if (p_.cost[j] != 0.0) obj_cols_.push_back(j);
      }
      gprop_ = std::make_unique<DomainPropagator>(prop_problem());
      gprop_->enable_objective();
      gprop_->continuous = true;
      // Once per new incumbent: a tenth of the node propagation's budget (turia needs ~0.25 nnz).
      gprop_->work_limit = 2 * static_cast<long long>(p_.A.nnz()) + 10000;
    }
    if (!obj_cols_.empty()) {
      gprop_->set_objective_cutoff(cutoff);
      std::vector<double> lb = glb_, ub = gub_;
      std::vector<Index> t;
      if (gprop_->propagate(lb, ub, obj_cols_, t)) {
        for (Index j : t) {
          if (!p_.is_int[j] || (lb[j] <= glb_[j] && ub[j] >= gub_[j])) continue;
          glb_[j] = std::max(glb_[j], lb[j]);
          gub_[j] = std::max(glb_[j], std::min(gub_[j], ub[j]));
          gfix_log_.push_back({j, glb_[j], gub_[j]});
          ++oprop;
        }
      }
    }
  }
  // Objective propagation with clique implications: a literal whose objective increase, with the literals it
  // forces to 0, exceeds the cutoff's room over the pseudo objective is fixed to 0 (search/obj_cliques.hpp).
  Index ocl = 0;
  if (is_finite(cutoff)) {
    std::vector<double> lb = glb_, ub = gub_;
    std::vector<Index> t;
    if (obj_clique_fix(cutoff, lb, ub, t) > 0) {
      for (Index j : t) {
        if (lb[j] <= glb_[j] && ub[j] >= gub_[j]) continue;
        glb_[j] = std::max(glb_[j], lb[j]);
        gub_[j] = std::max(glb_[j], std::min(gub_[j], ub[j]));
        gfix_log_.push_back({j, glb_[j], gub_[j]});
        ocl_new_.push_back(j);
        ++ocl;
      }
    }
  }
  if (fixed + oprop + ocl > 0) {
    log_.log(2, "  root reduced-cost fixing: %d bounds tightened, %d by objective propagation, %d by objective cliques\n",
             fixed, oprop, ocl);
    restart_check_due_ = true;
  }
  dbg_global("root reduced-cost fixing");
}

Index Search::obj_clique_fix(double cutoff, std::vector<double>& lb, std::vector<double>& ub,
                             std::vector<Index>& changed) {
  static const bool on = knob("MILP_OBJ_CLIQUES", 1) > 0;
  if (!on || !is_finite(cutoff)) return 0;
  const auto t0 = std::chrono::steady_clock::now();
  const long long nnz = static_cast<long long>(p_.A.nnz());
  if (!ocl_built_) {
    ocl_built_ = true;
    // At most 3e6 list entries (~100 MB with the scratch arrays; a list entry is a row nonzero).
    auto oc = std::make_unique<ObjectiveCliques>(p_, 3000000);
    log_.log(2, "  objective cliques: %d conflict lists, %lld entries%s, %.3fs\n", oc->lists(), oc->entries(),
             oc->truncated() ? " (truncated)" : "",
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (oc->lists() > 0) ocl_ = std::move(oc);
  }
  if (!ocl_) return 0;
  // Exact unions of a literal's partners up to about two passes over the matrix per call.
  const Index k = ocl_->propagate(cutoff, lb, ub, changed, 2 * nnz + 100000);
  ++ocl_calls_;
  ocl_fixed_ += k;
  ocl_time_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return k;
}

// Objective-clique fixings since the root LP that cut off its solution (the incumbent arrived after the root cut
// loop: deterministic mode syncs the FJ / local MIP points there, or a root heuristic found it): the root is
// processed again with the global bounds (propagation, LP, a few cut rounds; as after root strong branching
// fixings) instead of heuristics and strong branching on the weaker LP. At most 3 times per search.
// hypothyroid-k1, deterministic: 68 fixings at the root node; without this, root strong branching (40 s) and its
// cut rounds came first (67.8 s in all).
bool Search::ocl_root_redo(Worker& w) {
  std::vector<Index> cols;
  std::lock_guard<std::mutex> lk(gfix_mu_);
  cols.swap(ocl_new_);
  if (cols.empty() || ocl_redos_ >= 3 || w.xnode.size() != static_cast<size_t>(p_.num_cols())) return false;
  bool cut = false;
  for (Index j : cols) cut = cut || w.xnode[j] < glb_[j] - 1e-6 || w.xnode[j] > gub_[j] + 1e-6;
  if (!cut) return false;
  ++ocl_redos_;
  log_.log(2, "  root again: %zu columns fixed by objective cliques cut off the root LP solution\n", cols.size());
  for (; w.gfix_seen < gfix_log_.size(); ++w.gfix_seen) {
    const BoundChange& g = gfix_log_[w.gfix_seen];
    w.glb[g.col] = std::max(w.glb[g.col], g.lb);
    w.gub[g.col] = std::min(w.gub[g.col], g.ub);
    w.changed.push_back(g.col);
  }
  root_seeds_.insert(root_seeds_.end(), cols.begin(), cols.end());  // propagated on the next pass
  return true;
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
  if (genres_ > 0 && w.trail.usable()) {
    GenresOptions go;
    go.max_nnz = conf_max_nnz_;
    Conflict g;
    GenresInfo info;
    ++gr_calls_;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = analyze_genres(w.trail, w.prop->rows(), w.prop->row_lb(), w.prop->row_ub(), p_.is_int, w.glb,
                                   w.gub, go, g, &info);
    gr_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    gr_steps_ += info.steps, gr_reduced_ += info.reduced, gr_mir_ += info.mir;
    if (ok) {
      ++gr_learned_;
      if (info.outcome == GenresInfo::kUip) ++gr_uip_;
      if (info.outcome == GenresInfo::kPartial) ++gr_partial_;
      gr_nnz_ += static_cast<long long>(g.idx.size());
      g.cutoff = std::min(g.cutoff, cutoff(w));  // rows (objective) and global bounds rely on it
      add_conflict(w, std::move(g), "genres");
      if (genres_ == 2) return;
    } else {
      ++gr_fail_[std::min<int>(info.reason, 8)];
    }
  }
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
  static const bool conf_weight = knob("MILP_CONF_WEIGHT", 0.0) > 0.0;
  if (conf_weight || pc_.flat_hybrid) {  // the flat hybrid score uses them too
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

double Search::probe(Worker& w, Index j, double lb, double ub, double obj, const Basis& basis, bool* exact,
                     bool bound) {
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
  if (st == LpStatus::Infeasible || st == LpStatus::CutoffReached) {
    gain = kInf;
  } else if (st == LpStatus::Optimal || st == LpStatus::IterationLimit) {
    double z = lp.objective();
    // An optimal probe at or above the cutoff prunes its child, and its gain may become a bound: both only as
    // far as the probe's safe bound confirms its objective.
    bool below = false;
    if (st == LpStatus::Optimal && (bound || z >= cutoff(w))) z = lp_bound(lp, &below, -1);
    gain = st == LpStatus::Optimal && z >= cutoff(w) ? kInf : std::max(z - obj, 0.0);
  } else {
    gain = 0.0;
  }
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
  static const int sb_dyn = static_cast<int>(knob("MILP_SB_DYN", 2));
  static const int kReliability = static_cast<int>(knob("MILP_REL", 4));
  static const int kMaxProbes = no_sb ? 0 : static_cast<int>(knob("MILP_MAXPROBES", 8));
  static const int kLookahead = static_cast<int>(knob("MILP_LOOKAHEAD", 4));
  // Dynamic strong branching effort (SCIP-like), opt-in MILP_SB_DYN=1: probe iteration limit twice
  // the worker's average node LP (at least MILP_PROBE_ITERS, at most 500), reliability between
  // kMinRel and kReliability by the share of strong branching iterations in the worker's LP work
  // (full while below half of the node LP iterations + 10000, kMinRel above all of them). Not the
  // default: piperout-08 / rmatr100-p10 / mc11 1.2-1.8x slower at 60 s / 2 threads (longer probes on
  // expensive LPs); with MILP_REL=8 drayage-25-23 0/3 solved (degenerate probes), kasai's bound at
  // 60 s better (gap 0.03-0.06% vs 0.07%) but still unsolved. MILP_SB_DYN=2 (default): the reliability
  // part only (miplib_wide_core, 120 s, 4 threads, seeds 1-2: solved 109 -> 114 of 126, shifted
  // geomean 0.975x); 3: the iteration limit only; 0: off.
  static const long long kProbeIters = static_cast<long long>(knob("MILP_PROBE_ITERS", 100));
  static const int kMinRel = static_cast<int>(knob("MILP_REL_MIN", 1));
  int reliability = kReliability;
  w.probe_limit = kProbeIters;
  if (sb_dyn == 1 || sb_dyn == 3) {
    const double avg = w.tree_lps > 0 ? static_cast<double>(w.tree_iters) / static_cast<double>(w.tree_lps) : 0.0;
    w.probe_limit = std::clamp<long long>(static_cast<long long>(2.0 * avg), kProbeIters, std::max<long long>(kProbeIters, 500));
  }
  if (sb_dyn == 1 || sb_dyn == 2) {
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
  const double gap = cutoff(w) - obj;
  const BranchScoreRule rule = w.pc->score_rule(gap);
  bool used_aps = false;
  bool used_flat = false;
  w.pc->scores(w.fracs, reliability, w.scored, &w.est_sum, &ps_scale, degen_factor, rule, &used_aps, &used_flat);
  if (used_aps) ++sb_aps_nodes_;
  if (used_flat) ++sb_flat_nodes_;
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
  // Probabilistic lookahead (MILP_PL=1 exponential, 2 Pareto; off by default): once the best candidate
  // has not changed for MILP_PL_PHI x the lookahead probes, probing stops when one more probe is not
  // expected to shrink the tree (see branch_score.hpp); needs a finite gap and MILP_PL_MIN nonzero gains.
  static const int pl = static_cast<int>(knob("MILP_PL", 0));
  static const double pl_phi = knob("MILP_PL_PHI", 0.6);
  static const int pl_min = static_cast<int>(knob("MILP_PL_MIN", 3));
  const int pl_after = static_cast<int>(std::ceil(pl_phi * kLookahead));
  bool pl_stop = false;
  double best_g = 0.0;  // geometric-mean gain of the best candidate
  if (pl > 0) w.la.clear();
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
    if (!c.reliable && probes < max_probes && no_improve < kLookahead && !pl_stop) {
      const double v = w.xnode[c.j];
      bool ex_dn = false, ex_up = false;
      const double dn = probe(w, c.j, w.lp.col_lower(c.j), std::floor(v), obj, basis, &ex_dn, sb_bounds);
      const double up = probe(w, c.j, std::ceil(v), w.lp.col_upper(c.j), obj, basis, &ex_up, sb_bounds);
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
      score = Pseudocosts::ps_part(branch_score(dn, up, rule), ps_scale) / (ps_scale > 0.0 ? degen_factor : 1.0) +
              c.bonus;
      if (score > best_score) {
        no_improve = 0;
      } else {
        ++no_improve;
      }
      if (pl > 0) {
        const double g = std::sqrt((dn + 1e-6) * (up + 1e-6)) - 1e-6;
        w.la.add(g);
        if (score > best_score) best_g = g;
        if (no_improve >= pl_after && w.la.nonzero() >= pl_min && !w.la.worth_more(gap, best_g, pl - 1)) {
          pl_stop = true;
          ++sb_pl_stops_;
        }
      }
    } else if (pl > 0 && score > best_score) {
      best_g = std::sqrt((c.dn + 1e-6) * (c.up + 1e-6)) - 1e-6;
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

// Global bound change at the root (strong branching fixings, their propagation): global bounds,
// the root worker's bounds and LP; workers pick it up from gfix_log_.
void Search::root_tighten(Worker& w, Index j, double l, double u) {
  l = std::max(l, w.glb[j]);
  u = std::min(u, w.gub[j]);
  if (l <= w.glb[j] && u >= w.gub[j]) return;
  {
    std::lock_guard<std::mutex> lk(gfix_mu_);
    if (!glb_.empty()) {
      glb_[j] = std::max(glb_[j], l);
      gub_[j] = std::min(gub_[j], u);
    }
    gfix_log_.push_back({j, l, u});
  }
  w.glb[j] = l, w.gub[j] = u;
  w.lb[j] = std::max(w.lb[j], l), w.ub[j] = std::min(w.ub[j], u);
  w.lp.set_col_bounds(j, w.lb[j], std::max(w.lb[j], w.ub[j]));
  w.changed.push_back(j);
}

// Root strong branching (after the root cut loop and heuristics), as SCIP's root branching: probes
// the fractional candidates in pseudocost-score order (dual simplex warm starts from the root basis,
// an iteration limit per child) within a work budget relative to the root LP work, and uses them:
//  - a candidate with one infeasible / cut-off child is fixed to the other side globally; then the
//    root is processed again (propagation, LP, a few cut rounds, another pass) while passes fix;
//  - the probes initialise the pseudocosts;
//  - max over the candidates of the smaller child bound (both children solved to optimality) is a
//    valid global bound (every feasible point lies in one child).
// Returns 0 (branch: root_sb_var_ is the best probed candidate or -1), 1 (fixings: redo the root),
// 2 (both children of a candidate pruned: the root is infeasible / cut off).
// MILP_ROOT_SB=0 off; MILP_ROOT_SB_CANDS candidates per pass, MILP_ROOT_SB_WORK x the root LP iterations
// + MILP_ROOT_SB_OFS as the total budget, MILP_ROOT_SB_ITERS per child (0: auto), MILP_ROOT_SB_PASSES.
int Search::root_strong_branching(Worker& w, double obj, const Basis& basis) {
  // Read per call (once per root): tests switch it off for searches that must branch.
  const bool on = knob("MILP_ROOT_SB", 1) > 0 && std::getenv("MILP_NO_SB") == nullptr;
  static const int kCands = static_cast<int>(knob("MILP_ROOT_SB_CANDS", 100));
  static const double kWork = knob("MILP_ROOT_SB_WORK", 0.5);
  static const double kOfs = knob("MILP_ROOT_SB_OFS", 5000);
  static const long long kIters = static_cast<long long>(knob("MILP_ROOT_SB_ITERS", 0));
  static const int kPasses = static_cast<int>(knob("MILP_ROOT_SB_PASSES", 10));
  static const int kLook = static_cast<int>(knob("MILP_ROOT_SB_LOOKAHEAD", 8));
  // Not in sub-MIP heuristics (MILP_ROOT_SB_SUBMIP=1: there too): their short searches pay the probes
  // at every sub-MIP root.
  static const bool in_sub = knob("MILP_ROOT_SB_SUBMIP", 0) > 0;
  // MILP_ROOT_SB_ABORT = k (default 4, 0: off): a pass ends after k probes in a row that learn nothing (a
  // child at the iteration limit, so no bound; no fixing). dano3_5: 23 of 30 children at the limit, 14k
  // iterations; dano3_3: 3 passes, 26k iterations, 44 of 62 children at the limit.
  static const int kAbort = static_cast<int>(knob("MILP_ROOT_SB_ABORT", 4));
  // MILP_ROOT_SB_FIXBONUS = b (opt-in; 0: off): each fixing (a child infeasible or cut off) extends the root's
  // probe budget by b probes at the per-child limit, so passes go on while they keep fixing columns
  // (app1-1, 4 threads: 16 fixings in 22 probes, then the budget of 0.5 x the root's iterations was gone;
  // with 20 x it, 135 fixings in 161 probes over 2 passes and the root closed the gap: 20.7 -> 14.2 s,
  // 434 nodes -> 1; SCIP's root reaches its bound the same way, 78 probes in run 1).
  static const double kFixBonus = knob("MILP_ROOT_SB_FIXBONUS", 0.0);
  root_sb_var_ = -1;
  if (!on || w.fracs.empty() || (!opts_.sub_mip_heuristics && !in_sub)) return 0;
  // A root whose restart is already due (reduced-cost fixing, the fixings of earlier passes): no probes
  // here, the restarted root probes the reduced problem (MILP_RESTART_SKIP_SB, default 1, 0: off; read per
  // call). The rule is the restart's own (root_restart_decision), with this root's cost so far; a refused
  // restart (the last presolve removed too little, the budget) keeps the probes. Only on roots whose LP work
  // so far is at least MILP_RESTART_SKIP_SB_WORK (2e7; work units in both modes): on quick roots the probes
  // cost little and may settle the root (exp-1-500-5-5: 3 passes, 42 fixings, solved at the root in 0.6 s;
  // skipped, 4 restarts and 2.2 s), while a restart costs a new root with its heuristics.
  if (knob("MILP_RESTART_SKIP_SB", 1) > 0 && root_work(w.lp) >= knob("MILP_RESTART_SKIP_SB_WORK", 2e7) &&
      restart_allowed() && !stop_.load()) {
    Index ints = 0, newly_fixed = 0;
    const RestartDecision d = restart_check(root_work(w.lp), root_elapsed(), &ints, &newly_fixed);
    if (d.outcome == RestartDecision::Restart) {
      root_sb_skipped_ = true;
      log_.log(1, "Root strong branching skipped: %d of %d integer variables fixed at the root, restart due\n",
               newly_fixed, ints);
      return 0;
    }
  }
  if (root_sb_budget_ < 0) root_sb_budget_ = static_cast<long long>(kWork * static_cast<double>(root_sb_base_) + kOfs);
  if (root_sb_budget_ <= 0 || root_sb_passes_ >= kPasses) return 0;
  ++root_sb_passes_;
  const long long it_start = w.sb_iters;
  const long long saved_limit = w.probe_limit;
  const long long limit = kIters > 0 ? kIters : std::clamp<long long>(root_sb_base_ / 10, 100, 500);
  w.probe_limit = limit;
  double est_sum = 0.0, ps_scale = 1.0;
  // Restarted roots with carried pseudocosts (opt-in MILP_RESTART_SB_REL = k > 0): candidates with at
  // least k carried observations in both directions are not probed again; they compete with their
  // pseudocost score.
  static const int kSkipRel = static_cast<int>(knob("MILP_RESTART_SB_REL", 0));
  const int skip_rel = kSkipRel > 0 && !opts_.pc_init.empty() ? kSkipRel : std::numeric_limits<int>::max();
  w.pc->scores(w.fracs, skip_rel, w.scored, &est_sum, &ps_scale);
  std::stable_sort(w.scored.begin(), w.scored.end(),
                   [](const Pseudocosts::Scored& a, const Pseudocosts::Scored& b) { return a.score > b.score; });
  int probes = 0, fixed = 0, no_improve = 0, inexact = 0, idle_run = 0;
  double best_score = -1.0, maxmin = -kInf;
  bool pruned = false, aborted = false;
  root_seeds_.clear();
  const double gap = cutoff(w) - obj;
  const BranchScoreRule rule = w.pc->score_rule(gap);
  // Probabilistic lookahead as in select_branch (MILP_PL), from MILP_PL_PHI x the root lookahead on.
  static const int pl = static_cast<int>(knob("MILP_PL", 0));
  static const double pl_phi = knob("MILP_PL_PHI", 0.6);
  static const int pl_min = static_cast<int>(knob("MILP_PL_MIN", 3));
  const int pl_after = static_cast<int>(std::ceil(pl_phi * kLook));
  double best_g = 0.0;
  w.la.clear();
  for (const auto& c : w.scored) {
    if (probes >= kCands || w.sb_iters - it_start >= root_sb_budget_ || (kLook > 0 && no_improve >= kLook)) break;
    if (pl > 0 && no_improve >= pl_after && w.la.nonzero() >= pl_min && !w.la.worth_more(gap, best_g, pl - 1)) {
      ++sb_pl_stops_;
      break;
    }
    if (stop_.load() || timer_.elapsed() >= params_.time_limit) break;
    if (c.reliable) {
      const double score = branch_score(c.dn, c.up, rule);
      if (score > best_score) {
        best_score = score, no_improve = 0;
        root_sb_var_ = c.j;
        root_sb_dn_ = root_sb_up_ = -kInf;
      }
      continue;
    }
    const double v = w.xnode[c.j];
    bool ex_dn = false, ex_up = false;
    const double dn = probe(w, c.j, w.lp.col_lower(c.j), std::floor(v), obj, basis, &ex_dn, true);
    const double up = probe(w, c.j, std::ceil(v), w.lp.col_upper(c.j), obj, basis, &ex_up, true);
    ++probes;
    inexact += (is_finite(dn) && !ex_dn) + (is_finite(up) && !ex_up);
    log_.log(3, "    root probe %d col %d: down %.6g%s, up %.6g%s\n", probes, c.j, dn, is_finite(dn) && !ex_dn ? " (limit)" : "",
             up, is_finite(up) && !ex_up ? " (limit)" : "");
    if (!is_finite(dn) && !is_finite(up)) {
      pruned = true;
      break;
    }
    if (!is_finite(dn) || !is_finite(up)) {
      if (is_finite(dn)) root_tighten(w, c.j, w.lb[c.j], std::floor(v));
      else root_tighten(w, c.j, std::ceil(v), w.ub[c.j]);
      root_seeds_.push_back(c.j);
      ++fixed;
      no_improve = 0;
      idle_run = 0;
      root_sb_budget_ += static_cast<long long>(kFixBonus * 2.0 * static_cast<double>(limit));
      if (is_finite(dn)) pc_update(w, c.j, -1, dn / c.f);
      else pc_update(w, c.j, 1, up / (1.0 - c.f));
      continue;
    }
    pc_update(w, c.j, -1, dn / c.f);
    pc_update(w, c.j, 1, up / (1.0 - c.f));
    if (ex_dn && ex_up) maxmin = std::max(maxmin, std::min(dn, up));
    const double score = branch_score(dn, up, rule);
    const double g = std::sqrt((dn + 1e-6) * (up + 1e-6)) - 1e-6;
    w.la.add(g);
    if (score > best_score) {
      best_score = score, no_improve = 0;
      best_g = g;
      root_sb_var_ = c.j;
      root_sb_dn_ = ex_dn ? dn : -kInf;
      root_sb_up_ = ex_up ? up : -kInf;
    } else {
      ++no_improve;
    }
    idle_run = ex_dn && ex_up ? 0 : idle_run + 1;
    if (kAbort > 0 && idle_run >= kAbort) {
      aborted = true;
      ++root_sb_aborted_;
      break;
    }
  }
  w.lp.restore_state(w.state);
  w.probe_limit = saved_limit;
  const long long used = w.sb_iters - it_start;
  root_sb_budget_ = std::max(0LL, root_sb_budget_ - used);
  root_sb_probes_ += probes;
  root_sb_iters_ += used;
  root_sb_fixed_ += fixed;
  if (is_finite(maxmin) && !w.lp_unsafe) root_sb_bound_ = std::max(root_sb_bound_, obj + maxmin);
  log_.log(2, "  root strong branching pass %d: %d probes, %lld LP iterations (limit %lld per child, %d children at the "
           "limit%s), %d fixings, bound %.10g\n", root_sb_passes_, probes, used, limit, inexact,
           aborted ? ", pass aborted" : "", fixed, p_.display_sense * (std::max(obj, root_sb_bound_) + p_.offset));
  if (pruned) return 2;
  if (fixed > 0) {
    root_sb_var_ = -1;
    return 1;
  }
  return 0;
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
  // Diving may use up to ~10% of the search's LP iterations (plus a base budget).
  const long long iters = w.det ? w.iters_base : lp_iters_.load();
  const long long dived = w.det ? w.dive_base + w.dive_iters : dive_iters_.load();
  // MILP_DIVE_ADAPT (default 1; 0: the 10% above): the share follows this worker's dive successes, as SCIP's
  // diving heuristics, 5% x (1 + 10 (new incumbents + 1) / (dives + 1)), at most MILP_DIVE_ADAPT_MAX
  // (0.1): 10% for the first 10 dives, 5.5% after 100 without a new incumbent. And a dive starts only once
  // the budget left covers a whole dive (tree_dive_budget; a dive cut off by the budget finds nothing:
  // pg5_34 13 of 42, swath3 12 of 29 tree dives; one that never finishes is retried with a doubled budget
  // each time); until then the next nodes ask again.
  static const bool adapt = knob("MILP_DIVE_ADAPT", 1) > 0;
  static const double adapt_max = knob("MILP_DIVE_ADAPT_MAX", 0.1);
  bool wait = false;
  const long long budget = tree_dive_budget(w.dives, iters, dived, adapt, adapt_max, &wait);
  if (wait) return;
  w.nodes_since_dive = 0;
  if (budget <= 0) return;
  const DiveRule rule = kRules[w.next_rule++ % 4];
  std::vector<double> inc;
  incumbent_x(w, inc);
  DiveLimits lim;
  lim.cutoff = cutoff(w);
  lim.iteration_budget = budget;
  lim.time_limit = params_.time_limit - timer_.elapsed();
  long long its = 0;
  DiveOutcome out;
  ++tdive_calls_;
  const bool found = dive(p_, w.lp, rule, rounding_, inc.empty() ? nullptr : &inc, lim, w.sol, &its,
                          prop ? w.prop.get() : nullptr, &out);
  bool win = false;
  if (found) {
    ++tdive_found_;
    win = try_incumbent(w, w.sol, to_string(rule));
    tdive_wins_ += win;
  }
  tdive_limit_ += out.limit_hit;
  w.dives.record(its, win, out.limit_hit);
  static const bool debug = std::getenv("MILP_DIVE_DEBUG") != nullptr;
  if (debug && opts_.sub_mip_heuristics) {
    log_.log(0, "  tree %s: %s, %d steps, %lld of %lld iters, node LP mean %.0f, %.2fs\n", to_string(rule),
             win ? "improved" : found ? "found" : out.limit_hit ? "limit" : "failed", out.steps, its, budget,
             w.tree_lps > 0 ? static_cast<double>(w.tree_iters) / static_cast<double>(w.tree_lps) : 0.0,
             timer_.elapsed());
  }
  lp_iters_ += its;
  dive_iters_ += its;
  tdive_iters_ += its;
  w.dive_iters += its;
  w.lp.restore_state(w.state);
}

int Search::async_heur_threads() const {
  // MILP_ROOT_HEUR_ASYNC = k: with at least k threads (0: off); MILP_ROOT_HEUR_SHARE: the heuristics' share of
  // the threads (default 1/4: 1 of 4, 3 of 12). The first root only: a restarted root is smaller, its
  // heuristics are capped at about its own time, and their incumbent decides the next restart at once (mc11,
  // 4 threads: three root restarts in a row, 23.6 s; with the restarted roots' heuristics beside the tree 27.5 s).
  static const int min_threads = static_cast<int>(knob("MILP_ROOT_HEUR_ASYNC", 0));
  static const double share = knob("MILP_ROOT_HEUR_SHARE", 0.25);
  if (min_threads <= 0 || params_.deterministic || threads_ < std::max(2, min_threads)) return 0;
  if (opts_.restarts_done > 0 || opts_.tree_restarts_done > 0) return 0;
  return std::clamp(static_cast<int>(std::lround(share * threads_)), 1, threads_ - 1);
}

void Search::root_heuristics(Worker& w, int nthreads, double t_root_in) {
  // Time this search spent until the end of the root LP with cuts (after a restart: of the
  // restarted root only).
  const double t_root = t_root_in >= 0.0 ? t_root_in : timer_.elapsed() - t_start_;
  const int nthr = nthreads > 0 ? nthreads : threads_;
  // MILP_SUBHEUR_DEBUG: the root heuristic lines of the sub-MIPs too (their log is silent).
  static const bool sub_debug = std::getenv("MILP_SUBHEUR_DEBUG") != nullptr;
  const int hlog = sub_debug && !opts_.sub_mip_heuristics ? 0 : 3;
  log_.log(hlog, "  root heuristics: root %lld LP iterations (initial LP %lld, barrier root as %lld), %.2fs, %d rows, %d"
           " cols, cutoff %s, time left %.2fs\n", w.lp.iterations(), root_lp0_iters_, root_lp_equiv_iters_, t_root,
           p_.num_rows(), p_.num_cols(), is_finite(cutoff(w)) ? "yes" : "no", params_.time_limit - timer_.elapsed());
  struct Result {
    std::vector<double> x;
    const char* source = nullptr;
    long long iters = 0, dive_iters = 0;
  };
  using Task = std::function<void(Result&)>;
  // Opt-in MILP_ROOT_HEUR_SKIP=1 (opportunistic mode): a heuristic task that would start once the root
  // restart is due (fixings from an incumbent found by an earlier task) is skipped; the restarted
  // root runs its heuristics on the reduced problem. neos-4413714-turia: the first incumbent at 32 s
  // fixes 95.6% of the columns, the tasks after it ran another ~12 s on the full 190k columns (optimal
  // in 79.7 -> 69.4 s, Mac, 4 threads, 1 run each; not A/B tested, hence opt-in).
  static const bool skip_on_restart = knob("MILP_ROOT_HEUR_SKIP", 0) > 0;
  auto restart_due = [&] {
    if (!skip_on_restart || params_.deterministic || !restart_allowed()) return false;
    Index ints = 0, fixed = 0;
    return restart_check(root_work(w.lp), root_elapsed(), &ints, &fixed).outcome == RestartDecision::Restart;
  };
  auto run_tasks = [&](const std::vector<Task>& tasks) {
    std::vector<Result> res(tasks.size());
    std::atomic<size_t> next{0};
    std::atomic<int> early_wins{0};
    // Opportunistic mode: publish each point as soon as its task ends (a quick RENS no longer waits for
    // the slower dives and the pump: blp-ic98 4 s). Deterministic mode: in task order after the join.
    static const bool no_early = std::getenv("MILP_NO_EARLY_PUBLISH") != nullptr;
    const bool early = !w.det && !params_.deterministic && !no_early;
    auto loop = [&] {
      for (size_t k; (k = next++) < tasks.size();) {
        if (restart_due()) continue;
        tasks[k](res[k]);
        if (early && !res[k].x.empty()) {
          early_wins += try_incumbent(res[k].x, res[k].source);
          res[k].x.clear();
        }
      }
    };
    std::vector<std::thread> pool;
    const int nt = std::min(nthr, static_cast<int>(tasks.size()));
    for (int t = 1; t < nt; ++t) pool.emplace_back(inherit_interrupt(loop));
    loop();
    for (auto& th : pool) th.join();
    int wins = early_wins.load();
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
  // MILP_DET_SUB_FACTOR (default 3): that multiple of the root LP work. A sub-MIP's LP iterations cost several
  // times the root's (its presolve, cut loop, propagation), so 3x the work runs far beyond opportunistic mode's
  // 3x the time (Mac, 4 threads, with MILP_DET_LNS_FACTOR=0.25: factor 1 took n5-3 25.1 -> 21.4 s, seymour1
  // 23.7 -> 17.8, kasai 34.6 -> 28.3; air05 / app1-1 / h80x6320d within 1.5 s; qap10 unchanged). A/B pending.
  static const double det_sub_factor = knob("MILP_DET_SUB_FACTOR", 3.0);
  const long long det_sub_iters =
      params_.deterministic
          ? static_cast<long long>(std::max(2e8, det_sub_factor * static_cast<double>(w.lp.iterations()) * dims) / std::max(1.0, dims))
          : -1;

  // Phase 1: one dive per rule, each on its own copy of the root LP. Restarted roots from the second
  // restart on skip the dives, RENS and the root LNS unless the search before the restart improved the
  // incumbent (SearchOptions::root_heuristics).
  const bool no_dive = std::getenv("MILP_NO_DIVE") != nullptr || !opts_.root_heuristics;
  static const bool no_dive_prop = std::getenv("MILP_NO_DIVE_PROP") != nullptr;
  // Without an incumbent the dives run beside the sub-MIP heuristics of phase 2 (queued after them)
  // instead of before them.
  static const bool no_merge = std::getenv("MILP_NO_ROOT_MERGE") != nullptr;
  // MILP_ROOT_MERGE_INC (default 1; 0: off): also with an incumbent, so RENS gets the incumbent before the
  // dives as its cutoff (h80x6320d: local MIP's early point; the dives then find the optimum and RENS, with
  // that as its cutoff, cannot improve it and runs to its 10 s cap).
  static const bool merge_inc = knob("MILP_ROOT_MERGE_INC", 1) > 0;
  const bool merge_phases = !no_merge && (merge_inc || !is_finite(incumbent(w)));
  std::vector<Task> dive_tasks;
  if (!no_dive) {
    constexpr DiveRule kRules[] = {DiveRule::Fractional, DiveRule::Coefficient, DiveRule::Guided,
                                   DiveRule::VectorLength};
    auto inc_p = std::make_shared<std::vector<double>>();
    incumbent_x(w, *inc_p);
    // Each dive may use about twice the root LP effort (the dives run in parallel). A barrier root counts as
    // the dual simplex iterations it stands for: w.lp's own are the crossover cleanup and the cut loop only
    // (triptim1: 995, so 5000 per dive, which ended all three dives at their limit).
    const long long equiv = barrier_equiv() ? root_lp_equiv_iters_ : 0;
    long long budget = std::max<long long>(5000, 2 * (w.lp.iterations() + equiv));
    // MILP_ROOT_DIVE_REL = r (default 0: off): at most max(MILP_ROOT_DIVE_MIN (50000), r x the initial root
    // LP's iterations), the cut loop's not counted. --deterministic has no time cap on the dives, and after a
    // long cut loop the 2x above is large: puriri at 600 s 48 rounds, 182400 iterations, so 364800 per dive
    // and ~490 s of dives (2 threads); opportunistic mode caps each at 5% of the time left (puriri ~11 s,
    // ~15k iterations). The floor keeps the dives of the core set as they are (largest 51376 on qap10, initial
    // LP 25398; successful dives use up to 39x a small initial LP: pg, h80x6320d).
    static const double dive_rel = knob("MILP_ROOT_DIVE_REL", 0.0);
    static const double dive_min = knob("MILP_ROOT_DIVE_MIN", 50000);
    const long long lp0 = (root_lp0_iters_ >= 0 ? root_lp0_iters_ : w.lp.iterations()) + equiv;
    if (dive_rel > 0.0) {
      budget = std::min(budget, static_cast<long long>(std::max({5000.0, dive_min, dive_rel * static_cast<double>(lp0)})));
    }
    // Opportunistic mode caps each dive at 5% of the time left. After a barrier root the vector-length dive
    // (see MILP_BARRIER_VL_FIRST below) gets at least the root LP's own time (MILP_BARRIER_DIVE_TL = f x it,
    // default 1; 0: off): its dual simplex equivalent is in the budget above, and the 5% cut every dive of
    // triptim1 at 120 s (4 s, 6-10k iterations); with 25 s, the vector-length dive found the optimum after a
    // 30 s root LP (155 steps, 45353 iterations, 20 s). Not the other dives: all three at the barrier's time
    // (uccase9, 21 s, 2 threads, 70 s limit) took the root heuristics to the time limit before the tree.
    static const double barrier_dive_tl = knob("MILP_BARRIER_DIVE_TL", 1.0);
    const double tl = capped(std::max(2.0, 0.05 * remaining()));
    const double tl_vl = barrier_equiv() ? capped(std::max(tl, barrier_dive_tl * root_barrier_time_)) : tl;
    const double cut = cutoff(w);
    // MILP_DIVE_DEDUP (default 1; 0: off): no guided dive without an incumbent, where it is the fractional dive
    // again (same LP, same choices: puriri deterministic 21691 iterations each; in the sub-MIPs, which
    // start without an incumbent, every time).
    static const bool dedup = knob("MILP_DIVE_DEDUP", 1) > 0;
    std::vector<Task> tasks;
    for (DiveRule rule : kRules) {
      if (dedup && rule == DiveRule::Guided && inc_p->empty()) continue;
      const double cap = rule == DiveRule::VectorLength ? tl_vl : tl;
      tasks.push_back([&, rule, inc_p, budget, cap, cut](Result& r) {
        const std::vector<double>& inc = *inc_p;
        DualSimplex lp = w.lp;
        std::unique_ptr<DomainPropagator> prop;
        if (!no_dive_prop) prop = std::make_unique<DomainPropagator>(p_);
        DiveLimits lim;
        lim.cutoff = cut;
        lim.iteration_budget = budget;
        // At most the time left when the dive starts: cap is set before the tasks queue up (--deterministic:
        // the whole time left, so with more dives than threads a queued dive ran that long again;
        // ns1116954 2 threads ended 109 s past its limit).
        lim.time_limit = std::min(cap, remaining());
        std::vector<double> sol;
        const double t0 = timer_.elapsed();
        DiveOutcome out;
        const bool ok = dive(p_, lp, rule, rounding_, inc.empty() ? nullptr : &inc, lim, sol, &r.iters, prop.get(), &out);
        r.dive_iters = r.iters;
        if (ok) r.x = std::move(sol), r.source = to_string(rule);
        log_.log(hlog, "  root %s: %s, %d steps, %lld iters (budget %lld), %.2fs (cap %.2fs)\n", to_string(rule),
                 ok ? "found" : out.limit_hit ? "failed at its limit" : "failed", out.steps, r.iters, budget,
                 timer_.elapsed() - t0, lim.time_limit);
      });
    }
    if (merge_phases) dive_tasks = std::move(tasks);
    else run_tasks(tasks);
  }

  // Phase 2: feasibility pump (+ RENS around its closest point) if nothing was
  // found yet, and RENS around the root LP point.
  std::vector<Task> tasks;
  // MILP_SUB_FP (default 0; 1: as before): no pump in an improvement sub-MIP (a search with a cutoff and no
  // sub-MIPs of its own: RINS, mutation, crossover, RENS with an incumbent). The pump does not know the
  // cutoff, and its points are rarely below it (opportunistic, 13 core instances: 6 new sub-MIP
  // incumbents from 94 pumps, 26-47% of the sub-MIPs' LP iterations on pg5_34 / pg / nu25-pr12).
  static const bool sub_fp = knob("MILP_SUB_FP", 0) > 0;
  const bool improvement_sub = !opts_.sub_mip_heuristics && is_finite(external_cutoff_);
  if (!is_finite(incumbent(w)) && std::getenv("MILP_NO_FP") == nullptr && (sub_fp || !improvement_sub)) {
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
      fp_iters_ += r.iters;
      ++fp_runs_;
      log_.log(hlog, "  root feasibility pump: %s, %lld iters, %.2fs\n", ok ? "found" : "failed", r.iters,
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
  static const bool fpr_lp = knob("MILP_FPR_LP", 0.0) != 0.0;  // opt-in until the end-to-end A/B
  if (fpr_lp && !is_finite(incumbent(w))) {
    tasks.push_back([&](Result& r) {  // LP-guided fix-propagate-repair (reads w.xnode only)
      std::vector<double> sol;
      if (fpr_run_lp(w.xnode, sol)) r.x = std::move(sol), r.source = "fix-propagate-repair (LP)";
    });
  }
  static const bool no_submip = std::getenv("MILP_NO_SUBMIP") != nullptr;
  if (!no_submip && opts_.sub_mip_heuristics && opts_.root_heuristics) {
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
  static const bool lp_rens_on = std::getenv("MILP_NO_LP_RENS") == nullptr;
  if (lp_rens_on && !no_submip && opts_.sub_mip_heuristics && !is_finite(incumbent(w))) {
    const double cut = incumbent(w) + p_.offset;
    // MILP_LP_RENS_REL = r (default 1; 0: the old caps): a budget relative to the root's own cost instead of
    // the 2 s / 2x floors (the root waits for this task; enlight_hard: 1.06 s after a 0.3 s root where every
    // other task failed in 0.01 s): at most max(0.5 s, r x the root's time), deterministic max(5e7, 3 r x
    // the root LP work) work units (as det_sub_iters: 3x the LP work ~ the root's time); and no retry after
    // a first sub-MIP proved infeasible (lp_rens stop_blind).
    static const double lp_rens_rel = knob("MILP_LP_RENS_REL", 1.0);
    tasks.push_back([&, cut](Result& r) {
      SubMipLimits lim;
      lim.time_limit = capped(std::min(std::max(2.0, t_root), 0.1 * remaining()));
      lim.node_limit = 1000;
      lim.iteration_limit = det_sub_iters >= 0 ? 2 * det_sub_iters : -1;
      if (lp_rens_rel > 0.0) {
        lim.time_limit = std::min(lim.time_limit, capped(std::max(0.5, lp_rens_rel * t_root)));
        if (lim.iteration_limit >= 0) {
          const double work = std::max(5e7, 3.0 * lp_rens_rel * static_cast<double>(w.lp.iterations()) * dims);
          lim.iteration_limit = std::min(lim.iteration_limit, static_cast<long long>(work / std::max(1.0, dims)));
        }
      }
      SubMipStats ss;
      lim.stats = &ss;
      LpRensStats ls;
      std::vector<double> sol;
      const double t0 = timer_.elapsed();
      static const double target = knob("MILP_LP_RENS_TARGET", 0.7);
      static const int attempts = static_cast<int>(knob("MILP_LP_RENS_ATTEMPTS", 4));
      const bool ok = lp_rens(p_, w.lp, cut, lim, sol, target, attempts, &ls, lp_rens_rel > 0.0);
      r.iters = ls.dive_iterations;
      if (ok) r.x = std::move(sol), r.source = "root LP sub-MIP";
      log_.log(3, "  root LP sub-MIP: %s, %d sub-MIPs, rate %.2f, %lld nodes, %lld iters, %.2fs\n", ok ? "found" : "failed",
               ls.attempts, ls.last_rate, ss.nodes, ss.lp_iterations, timer_.elapsed() - t0);
    });
  }
  // MILP_BARRIER_VL_FIRST (default 1; 0: off): after a barrier root the vector-length dive (the last of
  // dive_tasks) goes first. Its LP re-solves are as expensive as the root LP is large, and it needs the fewest:
  // triptim1 76-155 steps (3-20 s) to the optimum, coefficient 367-434 and fractional 1500-3250 failing; with
  // 2 threads it was queued behind the sub-MIPs and two dives that failed, and started 42 s after the root
  // (solved at 83 s; first: 47 s).
  static const bool vl_first = knob("MILP_BARRIER_VL_FIRST", 1) > 0;
  if (vl_first && barrier_equiv() && root_barrier_time_ > 0.0 && !dive_tasks.empty()) {
    tasks.insert(tasks.begin(), std::move(dive_tasks.back()));
    dive_tasks.pop_back();
  }
  for (Task& t : dive_tasks) tasks.push_back(std::move(t));
  run_tasks(tasks);

  // Phase 3: improve the incumbent around the root LP point after the cuts (HiGHS finds most
  // of its early incumbents this way): rounds of RINS plus mutation sub-MIPs (one per
  // thread), repeated while a round improves.
  static const bool no_root_lns = std::getenv("MILP_NO_ROOT_LNS") != nullptr;
  // Only after a root of some size (>= 2000 LP iterations incl. the cut loop, a barrier root as the iterations
  // it stands for): on quick roots the sub-MIP setup costs more than the tree's LNS would.
  const long long root_iters = w.lp.iterations() + (barrier_equiv() ? root_lp_equiv_iters_ : 0);
  if (no_submip || no_root_lns || !opts_.sub_mip_heuristics || !opts_.root_heuristics || root_iters < 2000) return;
  if (restart_due()) return;
  // After a restart only if the rounds before it improved the incumbent (neos-1445765: ~2.5 s
  // for a round that never succeeds).
  if (!opts_.root_lns) return;
  root_lns_ran_ = true;
  // Budget: about the time spent until the end of the root LP with cuts (at most 10% of the
  // time limit, 20 s), so easy instances pay little. Not the time so far: the dives and RENS
  // above may take 10 s on their own (neos-1445765), which doubled the LNS time for nothing.
  // Deterministic mode: an LP-iteration budget instead of the clock.
  // A poor incumbent (> 1% above the root bound) gets at least kPoorLnsSecs (deterministic: 4x the
  // LP iterations): on quick roots the rounds otherwise end after one round (tr12-30: 0.45 s root,
  // incumbent 3.5% above the bound).
  static const double kPoorLnsSecs = knob("MILP_ROOT_LNS_POOR", 3.0);
  const bool poor = is_finite(incumbent(w)) &&
                    incumbent(w) - root_bound_ > 1e-2 * std::max(1.0, std::abs(incumbent(w) + p_.offset));
  const double lns_secs = poor ? std::max(t_root, kPoorLnsSecs) : t_root;
  const double lns_end = timer_.elapsed() + std::min({20.0, 0.1 * remaining(), lns_secs});
  // MILP_DET_LNS_FACTOR (default 1): factor on that deterministic budget (see MILP_DET_SUB_FACTOR above).
  static const double det_lns_factor = knob("MILP_DET_LNS_FACTOR", 1.0);
  const long long lns_iters = static_cast<long long>(
      det_lns_factor * static_cast<double>(std::max<long long>(50000, 5 * w.lp.iterations()) * (poor && kPoorLnsSecs > 0.0 ? 4 : 1)));
  long long lns_used = 0;
  double fix = 0.7;
  bool rins_ok = true;
  for (int round = 0; round < 12 && is_finite(incumbent(w)) && !solved_.load(); ++round) {
    // Skip when the incumbent is already within 0.5% of the root bound.
    if (incumbent(w) - root_bound_ <= 5e-3 * std::max(1.0, std::abs(incumbent(w) + p_.offset))) break;
    if (params_.deterministic ? lns_used >= lns_iters : timer_.elapsed() >= lns_end) break;
    std::vector<double> inc;
    incumbent_x(w, inc);
    const double before = incumbent(w), cut = before + p_.offset;
    std::vector<SubMipStats> stats(static_cast<size_t>(std::max(nthr, 2)));
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
    for (int t = static_cast<int>(lns.size()), k = 0; t < std::max(nthr, 2); ++t, ++k) {
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
  if (node.change.col >= 0) apply(node.change);  // the node's own branching bound comes last
  // Root strong branching fixings of the previous pass (already global bounds): propagate them.
  const bool root_prop = node.id == 0 && !root_seeds_.empty();
  if (root_prop) w.seeds.insert(w.seeds.end(), root_seeds_.begin(), root_seeds_.end());
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
      // A child proven infeasible by propagation counts as many inferences. The node propagates its
      // whole path, so the bounds it tightened are mostly those of the ancestors' branchings (the count
      // grew with the depth: on fhnw-binpack4-4 the inference score favoured columns branched on deep
      // in the tree). With a trail, only the bounds at the last decision level count: those that
      // depend on this node's own branching (SCIP's inference history; fhnw-binpack4-4 with the flat
      // hybrid score: 63k nodes / 5.9 s, all path bounds 103k / 6.9 s). MILP_INFER_PATH=1: all.
      static const bool infer_path = knob("MILP_INFER_PATH", 0) > 0;
      double inferences = ok ? static_cast<double>(w.changed.size() - changed_before) : 10.0;
      if (ok && use_trail && !w.trail.overflow && !infer_path) {
        long long own = 0;
        for (const BoundTrail::Entry& e : w.trail.e) own += e.kind != BoundTrail::kDecision && e.level == w.trail.level;
        inferences = static_cast<double>(own);
      }
      w.pc->update_inference(node.branch_var, node.branch_dir, inferences);
      if (w.det) w.inf_log.emplace_back(node.branch_var, node.branch_dir, inferences);
    }
    if (ok && root_prop) {
      // Root propagation of global fixings: globally valid (incl. the objective cutoff row). Integer
      // columns only become global bounds (continuous ones stay in the root LP: on dano3_3 they were
      // ~26k entries of the global bound log for every worker to replay).
      const size_t nchg = w.changed.size();
      for (size_t k = changed_before; k < nchg; ++k) {
        const Index j = w.changed[k];
        if (p_.is_int[j] && (w.lb[j] > w.glb[j] || w.ub[j] < w.gub[j])) root_tighten(w, j, w.lb[j], w.ub[j]);
      }
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
    node.basis->unpack(w.unpacked, ref_basis_.get());
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
  const long long ns0 = now_ns();
  LpStatus st = lp.solve(opt);
  const long long lp_ns = now_ns() - ns0;
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
    nodelp_ns_ += lp_ns;
    lprows_sum_ += static_cast<long long>(lp.num_rows());
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
      long long ns1 = now_ns();
      int added = pool_separate(w);
      scan_ns_ += now_ns() - ns1;
      // Fresh separation (into the pool) down to depth kNodeCutDepth at every sep_interval-th node
      // of a worker; deeper only at every kDeepEvery-th of those (0: never). Deep separation at
      // every node helped kasai a lot but cost 15-90% on cflp/lotsize/gap (LP work of the cuts).
      static const int kDeepEvery = static_cast<int>(knob("MILP_POOL_DEEP_EVERY", 16));
      const long long every = static_cast<long long>(sep_interval(w)) *
                              (node.depth <= kNodeCutDepth ? 1 : std::max(kDeepEvery, 1));
      const bool fresh = round == 0 && (node.depth <= kNodeCutDepth || kDeepEvery > 0) &&
                         node_cuts_enabled_.load() && w.nodes % every == 0;
      const bool tmir = fresh && tmir_turn(w);
      if (fresh) {
        ns1 = now_ns();
        added += separate_node(w, tmir);
        sep_ns_ += now_ns() - ns1;
        ++sep_calls_;
      }
      if (added == 0) break;
      ncp_rounds_ += 1;
      const long long it2 = lp.iterations();
      if (dump_numeric) start_basis = lp.get_basis();
      ns1 = now_ns();
      st = lp.solve(opt);
      cutlp_ns_ += now_ns() - ns1;
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
      if (lp.objective() - before <= 1e-6 * std::max(1.0, std::abs(before))) {
        ++ncp_flat_rounds_;
        break;
      }
    }
    if (st == LpStatus::Optimal) pool_age(w);
    // Zero objective (fhnw-binpack4-4): pool rows cannot raise a node's bound, they only help by making
    // its LP infeasible. The evidence above samples pruned nodes only, and nearly all of them close by
    // propagation there, so the pool stayed at every node (40k rounds re-added 1M cuts in 30 s, 166
    // prunes; 90% of the search time). A node that went through pool rounds and stays open is evidence
    // of rows not needed (fhnw-binpack4-4 proof 5.9 s / 63k nodes, without this 49.7 s / 154k nodes).
    // MILP_ZERO_OBJ_POOL_EV=0: off.
    static const bool zero_ev = knob("MILP_ZERO_OBJ_POOL_EV", 1) > 0;
    if (zero_ev && zero_obj_ && rounds && st == LpStatus::Optimal && node.depth > kNodeCutDepth && !w.prow.empty()) {
      const long long share =
          static_cast<long long>(1000.0 * static_cast<double>(w.prow.size()) / static_cast<double>(lp.num_rows()));
      pool_evidence_result(w, false, share);
    }
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
  // Convex quadratic rows (MipProblem::quad_rows, kept out of the LP): while the LP point violates them, gradient cuts
  // (outer approximation) through the cut pool (sub-MIPs: the global cuts) and a re-solve. Deterministic mode applies
  // new cuts only at its round barriers: there a violating integral point is branched on as an unsolved node.
  if (!p_.quad_rows.empty() && st == LpStatus::Optimal && !w.det) {
    // An integral LP point cannot be branched on: there the outer approximation goes on (MILP_OA_INT_ROUNDS) until the
    // point satisfies the rows (its node then closes or branches as usual). A fractional point is branched on after
    // MILP_OA_NODE_ROUNDS rounds, or once its relative violations are below MILP_OA_NODE_TOL (1e-4: CBLIB A/B, 30 s,
    // against 1e-7: solved 96 -> 102 of 157, sgm 0.851 [0.80, 0.90]; sssd-strong-20-8 37.9 -> 4.3 s: at 1e-7 about 13
    // rounds per node; on the server 1e-4 / 1e-3 102 -> 110 / 101 -> 112 at 120 s; QPLIB QCQP + portqc / qcrand
    // neutral, with the best bounds on the hard qcrand instances. Rounds 3 vs 20: neutral).
    static const int kOaRounds = static_cast<int>(knob("MILP_OA_NODE_ROUNDS", 20));
    static const int kOaIntRounds = static_cast<int>(knob("MILP_OA_INT_ROUNDS", 200));
    static const double kOaNodeTol = knob("MILP_OA_NODE_TOL", 1e-4);
    for (int round = 0;; ++round) {
      bool integral = true;
      for (Index j = 0; j < p_.num_cols() && integral; ++j) {
        integral = !p_.is_int[j] || std::abs(lp.x()[j] - std::round(lp.x()[j])) <= params_.tol.integrality;
      }
      if (round >= (integral ? kOaIntRounds : kOaRounds)) break;
      std::vector<Cut> cuts;
      if (quad_oa_cuts(p_, lp.x(), integral ? 1e-7 : kOaNodeTol, cuts) == 0) break;
      ++oa_rounds_;
      oa_cuts_ += static_cast<long long>(cuts.size());
      if (use_pool_ && w.m_base >= 0) {
        std::vector<int> ids;
        pool_publish(cuts, &ids);
        pool_pull(w);
        ids.erase(std::remove_if(ids.begin(), ids.end(), [&](int id) { return id < 0 || w.pool_pos[id] >= 0; }),
                  ids.end());
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        if (ids.empty()) break;  // nothing new for this LP (a numerically repeated cut)
        pool_add_rows(w, ids);
      } else {
        {
          std::lock_guard<std::mutex> lk(gcut_mu_);
          for (Cut& c : cuts) gcuts_.push_back(std::move(c));
        }
        apply_global_cuts(w);
      }
      // The node's opt.time_limit was the time left at its start: each re-solve gets what is left now.
      opt.time_limit = params_.time_limit - timer_.elapsed();
      if (opt.time_limit <= 0.0) {
        st = LpStatus::TimeLimit;
        break;
      }
      const long long it2 = lp.iterations();
      st = lp.solve(opt);
      lp_iters_ += lp.iterations() - it2;
      if (st == LpStatus::Infeasible || st == LpStatus::CutoffReached) {
        count_outcome(5, node.depth);
        return 0;
      }
      if (st != LpStatus::Optimal) break;
    }
  } else if (!p_.quad_rows.empty() && st == LpStatus::Optimal) {
    // Deterministic mode: the node's tangent cuts go to everyone after the round barrier (its integral point, if it
    // violates the rows, is branched on as an unsolved node meanwhile).
    std::vector<Cut> cuts;
    if (quad_oa_cuts(p_, lp.x(), 1e-7, cuts) > 0) {
      ++oa_rounds_;
      oa_cuts_ += static_cast<long long>(cuts.size());
      for (Cut& c : cuts) w.new_cuts.push_back(std::move(c));
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
  // The node's bound: obj, or the LP's safe dual bound where that is lower (a dual infeasibility the scaled
  // tolerances let pass would make obj an invalid bound). Pruning, the children's bounds and the root bound use
  // it; pseudocosts, estimates and branching scores stay on obj.
  bool unsafe = false;
  const double bnd = lp_bound(lp, &unsafe, node.id);
  w.lp_unsafe = unsafe;
  if (node.id == 0) {
    root_bound_ = bnd;
    log_.log(1, "Root LP: objective %.10g, %lld iterations, %.2fs\n", p_.display_sense * (obj + p_.offset), lp.iterations(),
             timer_.elapsed());
    // An incumbent that arrived before the root LP ended (a user start, FJ, local MIP) may already close the gap
    // at this bound: accept_incumbent tests the root bound only when a new incumbent arrives (MILP_ROOT_GAP_STOP).
    static const bool root_gap_stop_lp = knob("MILP_ROOT_GAP_STOP", 1) > 0;
    if (root_gap_stop_lp && !params_.deterministic && gap_closed(bnd)) trivial_solved(incumbent_.load(), true);
  }
  if (node.branch_var >= 0 && node.branch_dist > 0) {
    const double pobj = is_finite(node.parent_obj) ? node.parent_obj : node.bound;
    pc_update(w, node.branch_var, node.branch_dir, std::max(obj - pobj, 0.0) / node.branch_dist);
    if (node.anc_var >= 0 && node.anc_dist != 0.0f && Pseudocosts::aps_gamma() > 0.0) {
      const double g = std::max(obj - pobj, 0.0) / std::abs(static_cast<double>(node.anc_dist));
      w.pc->update_anc(node.anc_var, node.anc_dist > 0.0f ? 1 : -1, g);
      if (w.det) w.anc_log.emplace_back(node.anc_var, node.anc_dist > 0.0f ? 1 : -1, g);
    }
  }
  if (bnd >= cutoff(w)) {
    if (node.id != 0) {
      pool_evidence(w, node.depth);
      learn_conflict(w, LpStatus::CutoffReached);
    }
    count_outcome(6, node.depth);
    return 0;
  }
  if (node.id == 0) {
    root_z_ = bnd;
    root_d_.assign(p_.num_cols(), 0.0);
    root_at_.assign(p_.num_cols(), 0);
    glb_ = w.glb;
    gub_ = w.gub;
    for (Index j = 0; j < p_.num_cols() && !unsafe; ++j) {  // unsafe: no reduced-cost fixing from this root
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
  if (!no_rcfix && !unsafe && is_finite(cutoff(w))) {
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
    if (z >= 0) return implication_branch(w, node, children, z, std::max(bnd, node.bound));
  }
  if (w.fracs.empty()) {
    w.sol.assign(x, x + p_.num_cols());
    static const bool polish_lp = knob("MILP_POLISH_LP", 1) > 0;
    if (polish_lp) polish_integral_lp(w, opt);
    bool feasible = true;
    if (!integral_point_closes(w, w.sol, obj, &feasible)) {
      double iv = 0.0;
      const double viol = p_.max_violation(w.sol, &iv);
      // The LP point is integral within tolerance but violates the model, or is feasible only at a clearly
      // worse objective: we cannot branch on its fractionalities, so branch as for an unsolvable node LP
      // (fresh LP, else a domain split).
      log_.log(2, "  node %lld: integral LP point %s (viol %.3g, int %.3g), branching without it\n", node.id,
               feasible ? "worse than the LP bound once exactly feasible" : "violates model", viol, iv);
      return unsolved_branch(w, node, children);
    }
    if (unsafe && bnd < cutoff(w)) {
      // Integral, but the LP's optimality is not confirmed (its safe bound is lower): better points may remain
      // in the subtree. Branch as for an unsolvable node LP (fresh LP, else a domain split).
      log_.log(3, "  node %lld: integral LP point, safe bound below the cutoff, branching without it\n", node.id);
      return unsolved_branch(w, node, children);
    }
    count_outcome(7, node.depth);
    return 0;
  }
  // MILP_ZIROUND = 0 (default): off, 1: ZI rounding where simple rounding fails (root, every 8th node), 2: at
  // every node. Opt-in: server core A/B (4 seeds, 2 threads, 120 s) 1.001 [0.980, 1.025], solved 212 -> 208 (neos5
  // 4 -> 1, roll3000 3 -> 1: high-variance instances; no gain anywhere measurable). Each call recomputes the row activities (a pass over the matrix): after 1000 calls of a worker
  // with fewer than 1% feasible points it stops there (as SCIP's zirounding; per worker: deterministic).
  static const int ziround = static_cast<int>(knob("MILP_ZIROUND", 0));
  auto zi = [&] {
    if (w.zi_calls >= 1000 && 100 * w.zi_found < w.zi_calls) return;
    ++w.zi_calls;
    ++zi_calls_;
    if (!zi_rounding_.run(x, w.sol)) return;
    ++w.zi_found;
    ++zi_found_;
    zi_wins_ += try_incumbent(w, w.sol, "ZI rounding");
  };
  if (node.id == 0 || w.nodes % 8 == 0) {
    if (rounding_.run(x, w.sol)) try_incumbent(w, w.sol, "rounding");
    else if (ziround > 0) zi();
  } else if (ziround > 1) {
    zi();
  }

  w.tree_iters += lp.iterations() - it0;  // before dives and probes move the LP
  ++w.tree_lps;
  w.xnode.assign(x, x + p_.num_cols());  // strong branching moves the LP
  if (node.id == 0) root_x_ = w.xnode;
  if (node.id == 0 && ocl_root_redo(w)) {
    root_redo_ = true;
    --nodes_;
    --w.nodes;
    return 0;
  }
  if (node.id == 0 && root_sb_budget_ < 0) root_sb_base_ = lp.iterations();
  auto basis = std::make_shared<const Basis>(lp.get_basis());
  lp.save_state(w.state);
  if (node.id == 0) {
    if (!root_heur_done_ && !solved_.load()) {
      if (async_heur_threads() > 0) {
        // Beside the tree (MILP_ROOT_HEUR_ASYNC), from a copy of this root LP and point.
        heur_w_ = std::make_unique<Worker>();
        prompt_restart_check_ = true;
        heur_w_->lp = w.lp;
        heur_w_->xnode = w.xnode;
        heur_t_root_ = timer_.elapsed() - t_start_;
      } else {
        root_heuristics(w);
      }
    }
    root_heur_done_ = true;
  } else {
    const auto c0 = std::chrono::steady_clock::now();
    run_dives(w);
    w.t_dive += std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();  // statistics only
    run_sub_mips(w);
  }
  if (bnd >= cutoff(w) || solved_.load()) {
    count_outcome(9, node.depth);
    return 0;
  }
  w.node_depth = node.depth;
  Index j = -1;
  if (node.id == 0 && ocl_root_redo(w)) {  // an incumbent of the root heuristics
    root_redo_ = true;
    --nodes_;
    --w.nodes;
    return 0;
  }
  if (node.id == 0) {
    const int r = root_strong_branching(w, obj, *basis);
    if (r == 1) {
      // Global fixings: run() re-solves the root (propagation, cut rounds, another pass).
      root_redo_ = true;
      --nodes_;
      --w.nodes;
      return 0;
    }
    if (r == 2 || root_sb_bound_ >= cutoff(w)) {
      count_outcome(10, node.depth);
      return 0;
    }
    j = root_sb_var_;
    if (j >= 0) w.sb_var = j, w.sb_dn = root_sb_dn_, w.sb_up = root_sb_up_;
    if (j < 0 && root_sb_skipped_) {
      // The root restarts (its strong branching was skipped for that): no probes for a branching whose
      // children are dropped; the most fractional candidate.
      double best = -1.0;
      for (const auto& [c, f] : w.fracs) {
        if (std::min(f, 1.0 - f) > best) best = std::min(f, 1.0 - f), j = c;
      }
    }
  }
  if (j < 0) j = select_branch(w, obj, *basis);
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
  BasisRef packed;
  if (keep_basis) {
    if (use_pool_ && w.m_base >= 0) {
      w.nb.clear();
      for (size_t k = 0; k < w.prow.size(); ++k) {
        const VarStatus vs = basis->status[static_cast<size_t>(p_.num_cols() + w.m_base) + k];
        if (vs != VarStatus::Basic) w.nb.emplace_back(w.prow[k], vs);
      }
      std::sort(w.nb.begin(), w.nb.end());
      packed = store_basis(*basis, p_.num_cols() + w.m_base, &w.nb);
    } else {
      packed = store_basis(*basis, -1, nullptr);
    }
  }
  for (Node* c : {&down, &up}) {
    // A child's region lies within its parent's: never report a bound below the parent's (a node LP
    // can come out lower numerically; roll3000 once ended OPTIMAL with a reported bound 0.7% low).
    c->bound = std::max(bnd, node.bound);
    c->parent_obj = std::max(obj, node.bound);  // pseudocost reference (no strong branching bound)
    if (node.id == 0) c->bound = std::max(c->bound, root_sb_bound_);
    c->depth = node.depth + 1;
    if (keep_basis) c->basis = packed;
    c->parent = node.id;
    c->id = next_node_id(w);
    c->branch_var = j;
    if (node.branch_var >= 0 && node.branch_dist > 0.0) {
      c->anc_var = node.branch_var;
      c->anc_dist = static_cast<float>(node.branch_dir > 0 ? node.branch_dist : -node.branch_dist);
    }
  }
  // Bound paths: one segment with this node's own branching bound and its reduced-cost fixings
  // (shared by both children; the fixings are not kept under memory pressure: the children's LPs
  // only start weaker), then per child its branching bound, inline (the down child with orbital
  // fixings in a segment of its own).
  const PathRef base = branch_segment(w, node, keep_basis);
  w.seg.clear();
  PathSeg::encode({j, cur_lb, std::floor(v)}, p_.col_lb[j], p_.col_ub[j], w.seg);
  const size_t own_bytes = w.seg.size();
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
  set_child_change(down, base, {j, cur_lb, std::floor(v)}, w.seg.size() > own_bytes ? &w.seg : nullptr);
  for (const auto& c : w.scored) {
    if (c.j != j) continue;
    const double base_est = obj + w.est_sum - std::min(c.dn, c.up);
    down.estimate = base_est + c.dn;
    up.estimate = base_est + c.up;
    break;
  }
  if (w.sb_var == j && !unsafe) {
    // Opt-in (MILP_SB_BOUNDS=1): strong branching solved the children's LPs to optimality, so their
    // objectives are valid bounds (an infeasible / cutoff probe prunes the child without an LP; the gains
    // come from the probes' safe bounds, relative to obj: not with an unconfirmed obj).
    if (is_finite(w.sb_dn)) down.bound = std::max(down.bound, obj + w.sb_dn);
    if (is_finite(w.sb_up)) up.bound = std::max(up.bound, obj + w.sb_up);
  }
  down.branch_dir = -1;
  down.branch_dist = f;
  set_child_change(up, base, {j, std::ceil(v), cur_ub}, nullptr);
  up.branch_dir = 1;
  up.branch_dist = 1.0 - f;
  if (!debug_sol_.empty()) {
    // Debug: a node containing the optimum must have a child containing it.
    auto in_path = [&](const Node& c, const PathSeg* stop) {
      Index bad = -1;
      auto check = [&](const BoundChange& bc) {
        const double xd = debug_sol_[bc.col];
        if (bad < 0 && (xd < bc.lb - 1e-6 || xd > bc.ub + 1e-6)) bad = bc.col;
      };
      for (const PathSeg* b = c.path.get(); b && b != stop; b = b->parent()) b->for_each(check);
      if (c.change.col >= 0) check(c.change);
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

double Search::lp_bound(const DualSimplex& lp, bool* below, long long node_id) {
  *below = false;
  if (!DualSimplex::safe_bound_enabled()) return lp.objective();
  const long long ns0 = now_ns();
  const double b = lp.safe_objective(params_.tol.dual_feas, below);
  safe_ns_ += now_ns() - ns0;
  ++safe_checks_;
  if (*below) {
    const long long k = ++safe_below_;
    if (k <= 10) {
      const double z = lp.objective();
      char where[48];
      if (node_id >= 0) std::snprintf(where, sizeof(where), "node %lld", node_id);
      else std::snprintf(where, sizeof(where), "strong branching probe");
      log_.log(2, "  %s: LP objective %.10g, safe bound %.10g (%.3g below)%s\n", where,
               p_.display_sense * (z + p_.offset), p_.display_sense * (b + p_.offset), z - b,
               k == 10 ? "; further ones only counted" : "");
    }
  }
  return b;
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

bool Search::integral_point_closes(Worker& w, const std::vector<double>& sol, double lp_obj, bool* feasible) {
  if (feasible) *feasible = true;
  try_incumbent(w, sol, "LP");
  if (lp_obj >= cutoff(w)) return true;  // bound prune (also after the point became the incumbent)
  std::vector<double> chk;
  double chk_obj;
  if (!verified_point(sol, chk, &chk_obj)) {
    if (feasible) *feasible = false;
    return false;
  }
  // MILP_INT_CLOSE_STRICT=0: feasible closes (the old rule).
  static const bool strict = knob("MILP_INT_CLOSE_STRICT", 1) > 0;
  return !strict || chk_obj <= lp_obj + 1e-6 * std::max(1.0, std::abs(chk_obj + p_.offset));
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
  bool unsafe = false;
  if (st == LpStatus::Optimal) {
    bound = std::max(bound, lp_bound(fresh, &unsafe, node.id));
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
      std::vector<double> sol(x.begin(), x.end());
      // Integral: the subtree's optimum is this point if it is feasible at the LP's objective (and the LP's
      // optimality is confirmed by its safe bound; else the domain split below).
      if (integral_point_closes(w, sol, bound) && (!unsafe || bound >= cutoff(w))) return 0;
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
      double relaxed = -kInf;  // the objective of an optimal LP over the tangent cuts: a bound for the subtree
      for (const double tol : {1e-9, params_.tol.primal_feas}) {
        const LpStatus s2 = solve_fixed_ints(w.lb, w.lb, w.ub, tol, cutoff(w), pt);
        if (s2 == LpStatus::Optimal && !p_.quad_rows.empty()) relaxed = std::max(relaxed, p_.objective(pt) - p_.offset);
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
      // Quadratic rows whose point neither the tangents nor the conic method made feasible: the LP over the tangent
      // cuts relaxes the subtree (portqc_hard_1, deterministic mode: node 1927's conic solve failed at 8e-6, the node
      // was dropped with its weaker node bound and the result NUMERIC). Pruned if that bound reaches the cutoff.
      if (relaxed >= cutoff(w)) {
        ++fixed_recovered_;
        return 0;
      }
      // The LP over the continuous columns is unsolvable: give the subtree up.
      numerical_ = true;
      std::lock_guard<std::mutex> lk(stop_mu_);
      lost_bound_ = std::min(lost_bound_, std::max(bound, relaxed));
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
  const PathRef base = branch_segment(w, node, false);
  set_child_change(down, base, {j, w.lb[j], std::floor(v)}, nullptr);
  set_child_change(up, base, {j, std::ceil(v), w.ub[j]}, nullptr);
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
  const PathRef base = branch_segment(w, node, false);
  set_child_change(down, base, {z, w.lb[z], std::floor(w.lb[z])}, nullptr);
  set_child_change(up, base, {z, std::floor(w.lb[z]) + 1.0, w.ub[z]}, nullptr);
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
  const double searched = static_cast<double>(w.det ? w.iters_base / det_workers_ : lp_iters_.load());
  // At least kGap nodes apart, or (slow nodes: nexp, puriri at 1-7 nodes/s) kMinGap nodes and
  // kIterGap LP iterations of the search apart, so the first sub-MIPs do not wait minutes.
  static const long long kIterGap = static_cast<long long>(knob("MILP_LNS_ITER_GAP", 50000));
  constexpr long long kMinGap = 10;
  const long long since = nodes - (w.det ? w.lns.last_node : lns_last_node_.load());
  const long long iters_since = static_cast<long long>(searched) - (w.det ? w.lns.last_iters : lns_last_iters_.load());
  if (since < kGap && (kIterGap <= 0 || since < kMinGap || iters_since < kIterGap)) return;
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
  if (w.det) {
    w.lns.last_node = nodes;
    w.lns.last_iters = static_cast<long long>(searched);
  } else {
    lns_last_node_ = nodes;
    lns_last_iters_ = static_cast<long long>(searched);
  }

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
  int pick = -1;
  if (alns_) {
    // Adaptive selection: proximity search (once the incumbent is not known to be within its smallest
    // step of the optimum) and graph mutation join; all compete by alns_select alone.
    static const bool no_prox = knob("MILP_ALNS_PROX", 1) <= 0;
    static const bool no_graph = knob("MILP_ALNS_GRAPH", 1) <= 0;
    avail[kLnsProximity] = has_inc && prox_ok_ && !no_prox && inc_now != st.prox_inc;
    avail[kLnsGraph] = has_inc && !no_graph;
    pick = alns_select(st, avail);
  } else {
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
    pick = rins_turn ? kLnsRins : -1;
    double best = -kInf;
    // The other calls pick by UCB (before, the loop stopped as soon as RINS, the first entry,
    // was the provisional best: mutation, local branching and RENS never ran).
    for (int k = 0; k < kLns && !rins_turn; ++k) {
      if (!avail[k] || (k == kLnsRins && avail[kLnsMutation] && (rins_alternate || st.calls[kLnsRins] < 3.0))) continue;
      const double score =
          (st.wins[k] + 0.5) / (st.calls[k] + 1.0) + 0.5 * std::sqrt(std::log(total + 1.0) / (st.calls[k] + 1.0));
      if (score > best) best = score, pick = k;
    }
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
  for (int k = 0;; ++k) {
    const int solved = run_lns(*job, st, &w, w.det ? nullptr : &lk);
    // MILP_ALNS_REPICK=1: a rejected neighbourhood (no sub-MIP: too few fixings, or searched before) does not
    // use up the call; the next best one runs instead, at most twice (supportcase33, 60 s: 22 of 82 calls
    // were RENS / crossover rejections, each delaying the next call by kGap nodes). Not the default: measured
    // only together with the exploiting selection (alns_reward).
    if (!alns_repick_ || solved > 0 || k >= 2) break;
    avail[job->pick] = false;
    const int next = alns_select(st, avail);
    if (next < 0) break;
    job->pick = next;
    if (next == kLnsRins || next == kLnsRens) job->xnode = w.xnode;
  }
  w.t_sub += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();  // statistics only
  ++w.sub_runs;
  if (!w.det) --sub_active_;
}

// The sub-MIP attempts of one LNS call (st: the neighbourhood statistics to adapt). With a worker,
// solutions go to it; without (a deterministic-mode job), they are kept in job.sols and count as
// improvements (they beat the incumbent the job started from). lk: opportunistic mode's lns_mu_.
// Returns the number of sub-MIPs solved (0: every neighbourhood was rejected).
int Search::run_lns(SubMipJob& job, LnsState& st, Worker* w, std::unique_lock<std::mutex>* lk) {
  static const bool rins_static = std::getenv("MILP_RINS_STATIC") != nullptr;
  // A neighbourhood that turns out too small (sub-MIP exhausted with almost no LP work, e.g.
  // infeasible after presolve) is retried at once with its adapted size, up to 4 times.
  static const bool no_retry = std::getenv("MILP_NO_LNS_RETRY") != nullptr;
  const int pick = job.pick;
  const std::vector<std::vector<double>>& pool = job.pool;
  std::vector<double> chain;  // proximity search: this call's last point, the next centre (job mode)
  int solved = 0;
  for (int attempt = 0; attempt < 4; ++attempt) {
    // Neighbourhood parameters now; the sub-MIP itself runs without the lock.
    const double rins_fix_used = st.rins_fix, rins_max_used = st.rins_max, mut_fix = st.mut_fix;
    const double graph_fix = st.graph_fix, prox_imp = st.prox_imp;
    const int lb_k = st.lb_k, cx_parents = st.cx_parents;
    const unsigned seq = st.seq++;
    // MILP_ALNS: fixing neighbourhoods searched completely before are skipped (copied: other threads may add).
    const bool dedup = alns_ && (pick == kLnsRins || pick == kLnsMutation || pick == kLnsCrossover || pick == kLnsGraph);
    std::vector<std::uint64_t> searched;
    if (dedup) searched = st.searched;
    std::uint64_t fingerprint = 0;
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
    if (dedup) lim.fingerprint = &fingerprint, lim.skip = &searched;
    std::vector<double> sol;
    std::vector<double> inc = job.inc;
    if (w && !w->det && job.has_inc) incumbent_x(*w, inc);  // the incumbent may have improved meanwhile
    else if (!chain.empty()) inc = chain;
    const double cut = job.inc_now + p_.offset;
    // Objective the call starts from: the cutoff (proximity search: its centre).
    double c_old = job.inc_now;
    bool prox_min = false;  // proximity search asked for its smallest improvement step
    bool found = false;
    const char* source = nullptr;
    switch (pick) {
      case kLnsRins:
        found = rins(p_, job.xnode.data(), inc, cut, lim, sol, rins_fix_used, rins_max_used,
                     params_.seed * 7919u + 7u * seq);
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
      case kLnsProximity: {
        // Required improvement theta: prox_imp x the gap to the root bound (or to 0 when there is none), at
        // least 1 for an integral objective. Stops at the first point (the next attempt recentres there).
        c_old = p_.objective(inc) - p_.offset;
        const double rb = root_bound_.load();
        const double scale = is_finite(rb) && rb < c_old ? c_old - rb : std::max(1.0, std::abs(c_old + p_.offset));
        double theta = prox_imp * scale;
        theta = obj_integral_ ? std::max(1.0, std::floor(theta + 1e-9)) : std::max(theta, 1e-9 * scale);
        prox_min = obj_integral_ ? theta <= 1.0 : prox_imp <= kProxMin;
        if (!job.det) lim.time_limit = std::min(5.0, 0.05 * rem);
        lim.solution_limit = 1;
        found = proximity(p_, inc, c_old + p_.offset - theta, lim, sol);
        source = "proximity";
        break;
      }
      case kLnsGraph:
        found = graph_mutation(p_, inc, graph_fix, params_.seed * 7919u + 104723u + seq, cut, lim, sol);
        source = "graph mutation";
        break;
      default:
        found = rens(p_, job.xnode.data(), cut, lim, sol);
        source = "RENS";
        break;
    }
    double c_new = kInf;
    if (found && alns_) c_new = p_.objective(sol) - p_.offset;
    if (found && pick == kLnsProximity) chain = sol;
    bool win = found;
    if (found && w) {
      win = try_incumbent(*w, sol, source);
    } else if (found) {
      job.sols.emplace_back(std::move(sol), source);
    }
    if (lk) lk->lock();
    solved += ss.solves;
    double value = 0.0;
    if (alns_) {
      // `found`: the neighbourhood held a better point than the one it started from (in opportunistic mode
      // another thread may have found a still better one meanwhile).
      if (found) {
        const double rb = root_bound_.load();
        const double closed = !is_finite(c_old) ? 1.0 : is_finite(rb) && rb < c_old ? (c_old - c_new) / (c_old - rb) : 0.0;
        value = 0.8 + 0.2 * std::clamp(closed, 0.0, 1.0);
      }
      if (dedup && ss.solves > 0 && ss.exhausted && !found) {
        st.searched.push_back(fingerprint);
        if (st.searched.size() > 64) st.searched.erase(st.searched.begin());
      }
      // A rejected neighbourhood (too few fixings or searched before, no sub-MIP) counts a quarter call
      // without iterations and value, as without MILP_ALNS (MILP_ALNS_REPICK: a whole call; a quarter keeps
      // its UCB term large, so it is picked again and again: supportcase33, see run_sub_mips).
      alns_reward(st, pick, value, static_cast<double>(ss.lp_iterations), ss.solves > 0 || alns_repick_ ? 1.0 : 0.25);
      ++arm_calls_[pick];
      arm_found_[pick] += found;
      arm_iters_[pick] += ss.lp_iterations;
      if (ss.solves > 0 && (pick == kLnsMutation || pick == kLnsGraph)) {
        // Fixing rate from the outcome, with steps shrinking from 0.1 by 0.8 per adjustment (to 0.02; SCIP's ALNS:
        // 0.2 x 0.75^k): searched completely (better point or not) -> larger neighbourhood, a limit hit without a
        // better point -> smaller. (Without MILP_ALNS mutation moves by 0.05 after failures only, below.)
        double& fix = pick == kLnsMutation ? st.mut_fix : st.graph_fix;
        double& step = pick == kLnsMutation ? st.mut_step : st.graph_step;
        const double used = pick == kLnsMutation ? mut_fix : graph_fix;
        if (ss.exhausted) fix = std::max(0.2, used - step);
        else if (!found) fix = std::min(0.95, used + step);
        step = std::max(0.02, 0.8 * step);
      }
      if (pick == kLnsProximity && ss.solves > 0) {
        // A point: ask for more next time. None (infeasible: no point improves by theta; or the limits) at the
        // smallest step: leave this incumbent alone. Otherwise a smaller step.
        if (found) st.prox_imp = std::min(0.5, prox_imp * 1.5);
        else if (prox_min) st.prox_inc = c_old;
        else st.prox_imp = std::max(kProxMin, prox_imp * (ss.exhausted ? 0.25 : 0.5));
      }
    }
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
      if (pick == kLnsMutation && !alns_) {
        // Finer steps down: the random fixings decide as much as the rate whether the
        // neighbourhood is empty (fiball: 0.6 infeasible or improving depending on the seed).
        st.mut_fix = ss.exhausted ? std::max(0.2, mut_fix - 0.05) : std::min(0.95, mut_fix + 0.05);
      }
      if (pick == kLnsRins && !rins_static) {
        // Too large (limit hit): fix more next time, at least 10% above this call's rate.
        const double rate = ss.ints > 0 ? static_cast<double>(ss.fixed) / ss.ints : 0.0;
        st.rins_fix = ss.exhausted ? std::max(0.0, st.rins_fix - 0.1) : std::min(0.9, std::max(st.rins_fix, rate) + 0.1);
        // Exhausted at a high rate: the agreement alone fixes too much; cap it below this call's rate.
        // A limit hit lets the cap grow back.
        static const bool no_rins_cap = std::getenv("MILP_NO_RINS_CAP") != nullptr;
        if (!no_rins_cap) {
          if (ss.exhausted && rate > 0.6) st.rins_max = std::max(0.5, std::min(st.rins_max, rate) - 0.1);
          else if (!ss.exhausted) st.rins_max = std::min(1.0, st.rins_max + 0.05);
        }
      }
      if (pick == kLnsLocalBranching) st.lb_k = ss.exhausted ? std::min(lb_k * 2, 80) : std::max(lb_k / 2, 5);
      // More parents agree on fewer variables: a larger neighbourhood.
      if (pick == kLnsCrossover) st.cx_parents = ss.exhausted ? std::min(cx_parents + 1, 6) : std::max(cx_parents - 1, 2);
    }
    if (pick == kLnsCrossover && ss.solves == 0) st.cx_parents = std::max(cx_parents - 1, 2);  // too few fixed
    if (std::getenv("MILP_LNS_DEBUG")) {
      log_.log(1, "  LNS %s: %s, %d solves, fixed %d/%d, %lld iters (dives %lld / %d inc, %d pumps %lld / %d inc), %lld"
               " nodes, %s, boost %d, %.2fs\n", source, win ? "improved" : "no", ss.solves, ss.fixed, ss.ints,
               ss.lp_iterations, ss.dive_iterations, ss.dive_wins, ss.fp_runs, ss.fp_iterations, ss.fp_wins, ss.nodes,
               ss.exhausted ? "exhausted" : "limit", st.boost, timer_.elapsed());
      if (alns_) {
        log_.log(1, "    ALNS value %.3f, ref %.0f iters; mutation %.2f, graph %.2f, proximity %.4f; scores", value,
                 st.iter_ref, st.mut_fix, st.graph_fix, st.prox_imp);
        for (int k = 0; k < kLns; ++k) {
          if (st.rcnt[k] > 0.0) log_.log(1, " %d:%.3f/%.1f", k, st.rsum[k] / std::max(1e-9, st.rcost[k]), st.rcnt[k]);
        }
        log_.log(1, "\n");
      }
    }
    st.iters += ss.lp_iterations;
    if (w && !w->det) sub_iters_ += ss.lp_iterations;
    ++hsub_calls_;
    hsub_wins_ += win;
    hsub_iters_ += ss.lp_iterations;
    hsub_dive_iters_ += ss.dive_iterations;
    hsub_fp_iters_ += ss.fp_iterations;
    hsub_dive_wins_ += ss.dive_wins;
    hsub_fp_runs_ += ss.fp_runs;
    hsub_fp_wins_ += ss.fp_wins;
    const bool cheap_fail = !win && ss.solves > 0 && ss.exhausted && ss.lp_iterations < 1000;
    if (pick == kLnsProximity) {
      // Chains (Fischetti and Monaci recentre at each new point): again from the new point, or at once with the
      // smaller step after a cheap proof that none improves by theta.
      if (job.stop.load() || !(found || (cheap_fail && st.prox_inc != c_old))) break;
      continue;
    }
    if (no_retry || !cheap_fail || pick == kLnsRens || pick == kLnsLocalBranching ||
        (pick == kLnsCrossover && st.cx_parents == cx_parents) ||
        (pick == kLnsRins && st.rins_fix == rins_fix_used && st.rins_max == rins_max_used) ||
        job.stop.load()) {
      break;
    }
  }
  return solved;
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
  if ((Pseudocosts::sd_mode() & 4) && nc == 0 && node.branch_var >= 0 && !stop_ && last_outcome_ >= 0) {
    // Leaf for the last-assignment asymmetry: infeasible (path, propagation, orbitope, LP) or integral
    // vs pruned by bound (LP cutoff, cut rounds, late cutoff, after heuristics, strong branching).
    const int k = last_outcome_;
    const bool inf = k == 0 || k == 1 || k == 2 || k == 3 || k == 7;
    w.pc->update_leaf(node.branch_dir, inf);
    if (w.det) w.leaf_log.emplace_back(node.branch_dir, inf);
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
      tw_.close(cur.depth);
      have = false;
      pool_->done(tid);
      continue;
    }
    pool_->set_worker_bound(tid, cur.bound);
    const int nc = process(w, cur, children);
    if (nc != 2) tw_.close(cur.depth);
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
    else if (prompt_restart_check_ && restart_check_due_.load(std::memory_order_relaxed) && restart_check_due_.exchange(false)) {
      check_tree_restart(false);
    }
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
  det_running_ = true;
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
  // done job_factor (2) times the job's LP iterations per thread (each round adds the round's LP
  // iterations / T to the job's credit): a deterministic point after the job usually finishes. The
  // barrier waits for a job only while it may still finish below its credit (it has used fewer
  // iterations so far).
  static const bool no_jobs = std::getenv("MILP_DET_NO_JOBS") != nullptr;
  static const double job_factor = knob("MILP_DET_JOB_FACTOR", 2.0);
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
    w->own_pc->flat_hybrid = pc_.flat_hybrid;
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
  long long jw_n[3] = {0, 0, 0};
  double jw_s[3] = {0.0, 0.0, 0.0};
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
  // paths, queue arrays; at least NodePool::kNodeBytes), for both measures memory_control uses: the
  // bytes the open nodes use (target 78% of the limit outside drain mode, just past the drain
  // threshold of 75%: drain mode starts about where it does in opportunistic mode, not after a
  // round overshot it; 90% in drain mode) and the allocated bytes (88%, just past their drain
  // threshold of 85%; 90% in drain mode: the search stops at 95%). A target on one measure alone
  // could stall the tree just below the other measure's drain threshold. Where that binds, the
  // growth goes to few workers, each with room for a plunge of
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
    auto room = [&](long long used, long long target) {
      const long long per_node = std::max<long long>(NodePool::kNodeBytes, used / std::max<long long>(1, open)) * 3 / 2;
      return std::max<long long>(0, (target - used) / per_node);
    };
    const long long g = std::min(room(tree_bytes(), drain_round ? limit / 20 * 18 : limit / 100 * 78),
                                 room(tree_alloc_bytes(), drain_round ? limit / 20 * 18 : limit / 100 * 88));
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
    // Past the time limit a round ends at its next node (the result at a time limit depends on timing anyway): nodes
    // whose work the round budget does not see (the fixed-integer repairs of integral points that violate quadratic
    // rows, ~0.4 s each on portqc_hard_1) made one round run 91 s past a 60 s limit.
    auto more = [&](long long open_now) {
      return processed == 0 || (processed < kMaxNodes && clock_of(w) - it0 < round_budget &&
                                open_now - open0 < grow_cap[t] && timer_.elapsed() <= params_.time_limit);
    };
    if (batch <= 0) {
      // Old scheme: depth-first over the subtree of the single assigned node.
      while (!local.empty() && !stop_.load() && more(static_cast<long long>(local.size()))) {
        Node cur = std::move(local.back());
        local.pop_back();
        if (cur.bound >= w.cutoff) {
          tw_.close(cur.depth);
          continue;
        }
        ++processed;
        if (process(w, cur, children) == 2) {
          local.push_back(std::move(children[1]));
          local.push_back(std::move(children[0]));
        } else {
          tw_.close(cur.depth);
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
          tw_.close(cur.depth);
          have = false;
          continue;
        }
        ++processed;
        if (process(w, cur, children) == 2) {
          local.push_back(std::move(children[1]));
          std::push_heap(local.begin(), local.end(), worse);
          cur = std::move(children[0]);
        } else {
          tw_.close(cur.depth);
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
      else tw_.close(n.depth);
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
    threads.emplace_back(inherit_interrupt([&, th] {
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
    }));
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
      if (carried[t] && carry[t].bound >= cutoff_.load()) {
        tw_.close(carry[t].depth);
        carried[t] = 0;  // pruned meanwhile
      }
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
      for (auto [dir, inf] : w.leaf_log) pc_.update_leaf(dir, inf);
      w.leaf_log.clear();
      for (auto& [j, dir, g] : w.anc_log) pc_.update_anc(j, dir, g);
      w.anc_log.clear();
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
      // The round's LP work / T, at most one thread's rate. A finished job counts as pending until it enters
      // while its thread does tree work, so the round's work / (T - pending jobs) (MILP_DET_JOB_SHARE=0) can
      // credit a running job with several threads' work per round, and the barrier then waits for it. With
      // MILP_DET_JOB_FACTOR 2: server deterministic A/B (4 threads, core, 2 seeds) 0.973 [0.955, 0.989].
      static const int share_mode = static_cast<int>(knob("MILP_DET_JOB_SHARE", 1));
      const double share = static_cast<double>(clock - last_clock) /
                           (share_mode == 1 ? T : std::max<int>(1, T - static_cast<int>(jobs.size())));
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
          const Clock::time_point tj = Clock::now();
          std::unique_lock<std::mutex> lk(mu);
          const bool waits = !job->done.load() && job->progress.load() <= due;
          while (!job->done.load() && job->progress.load() <= due) cv.wait_for(lk, std::chrono::milliseconds(1));
          enter = job->done.load() && job->progress.load() <= due;
          if (waits) {
            // Statistics: the wait ended with the job entering, finished above its due point, or still running.
            const int k = enter ? 0 : job->done.load() ? 1 : 2;
            ++jw_n[k];
            jw_s[k] += secs(tj, Clock::now());
          }
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
    log_.log(det_stats ? 0 : 2, "DET   job waits: %lld entered %.2fs, %lld finished above their due point %.2fs, %lld still "
             "running %.2fs\n", jw_n[0], jw_s[0], jw_n[1], jw_s[1], jw_n[2], jw_s[2]);
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
  fj_thread_ = std::thread(inherit_interrupt([this] {
    FjLimits lim;
    // ~3e8 effort units per second; stops early once it stops making progress.
    lim.effort = 10000000000LL;
    lim.stall_effort = 300000000LL;
    // MILP_FJ_REPORT = e (default 3e7; 0: every improving point; default mode only, deterministic mode takes points at
    // its syncs): after the first, FJ reports its best point at most every e effort units (~0.1 s) and at its end, as
    // local MIP does (MILP_LMIP_REPORT): savsched1's 2335 improving points cost 8.4 s of incumbent checks. Server A/B
    // (core, 4 seeds, 120 s, 2 threads): 0.986 [0.955, 1.019], solved 218 -> 219 of 252, WRONG 0.
    lim.report_effort = fj_sync_ ? 0 : static_cast<long long>(knob("MILP_FJ_REPORT", 3e7));
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
    double best_obj = kInf;
    // FPR's point (MILP_FPR) starts as the best one: local MIP and FJ then keep only better points,
    // FJ makes one attempt and the repair sub-MIP is skipped. Its continuous part is LP-optimal
    // already (no polish).
    bool polished = false;
    if (fpr_run_lp_free(publish, deadline, best)) best_obj = p_.objective(best) - p_.offset, polished = true;
    long long min_viol = -1;
    // One FJ attempt (seed a: params_.seed + 1000003 a), keeping the best point and the closest
    // infeasible one.
    auto fj_attempt = [&](int a) {
      lim.seed = params_.seed + 1000003u * static_cast<unsigned>(a);
      lim.time_limit = deadline - timer_.elapsed();
      if (lim.time_limit <= 0.0) return false;
      FjStats st;
      feasibility_jump(p_, lim, [&](const std::vector<double>& x, double obj) {
        log_.log(3, "  FJ: solution %.10g at %.2fs\n", p_.display_sense * (obj + p_.offset), timer_.elapsed());
        if (obj >= best_obj) return;
        best = x;
        best_obj = obj;
        polished = false;
        publish(x, "feasibility jump");
      }, &st);
      base += st.effort;
      log_.log(2, "  FJ: %d solutions, %lld moves, min violated %lld, effort %lld, %.2fs\n", st.solutions, st.moves,
               st.min_violated, st.effort, timer_.elapsed());
      if (best.empty() && !st.closest.empty() && (min_viol < 0 || st.min_violated < min_viol)) {
        min_viol = st.min_violated;
        closest = std::move(st.closest);
      }
      return true;
    };
    // MILP_FJ_FIRST=1 (opt-in): the first FJ attempt runs before local MIP instead of after it. FJ
    // stops after ~1 s without progress; local MIP without a solution restarts until its whole budget
    // is used (5-10 s, longer on a loaded machine), and FJ, which finds neos-3381206-awhea's first
    // points in < 1 s, then only starts after that (no incumbent during the root).
    static const bool fj_first = knob("MILP_FJ_FIRST", 0) > 0;
    if (fj_first && !fj_stop_.load()) fj_attempt(0);
    // Local-MIP style search first (heuristics/lmip.hpp): it finds feasible points where FJ does
    // not and improves them with lift / breakthrough moves. MILP_LMIP=<effort budget> (default 3e9,
    // ~5-10 s; 0 = off).
    static const double lmip_effort = knob("MILP_LMIP", 3e9);
    if (lmip_effort > 0 && !fj_stop_.load()) {
      LocalMipLimits ll;
      ll.effort = static_cast<long long>(lmip_effort);
      // MILP_LMIP_REPORT = e (default 3e7 effort units, ~0.1 s; 0: every point): after the first point, a better
      // one is reported once e effort has passed since the last report (the best point then, and at the end).
      // Local MIP walks through long chains of slightly better feasible points: roll3000 reported 40874 points
      // in 2 s (25904 new incumbents, 0.29 apart), each a pass over the matrix in the search, the incumbent
      // checks, root reduced-cost fixing and objective propagation, and a callback / trace / log entry.
      // Opportunistic mode only: in deterministic mode the effort until the next report can be many rounds of a
      // small model, and the points enter at the syncs anyway.
      ll.report_effort = fj_sync_ ? 0 : static_cast<long long>(knob("MILP_LMIP_REPORT", 3e7));
      ll.stall_effort = static_cast<long long>(knob("MILP_LMIP_STALL", 5e8));
      ll.stall_restart = true;  // no solution yet: random restarts until the budget is used
      ll.seed = params_.seed;
      ll.stop = &fj_stop_;
      ll.checkpoint = lim.checkpoint;
      ll.time_limit = deadline - timer_.elapsed();
      LocalMipStats st;
      local_mip(p_, ll, [&](const std::vector<double>& x, double obj) {
        log_.log(3, "  LMIP: solution %.10g at %.2fs\n", p_.display_sense * (obj + p_.offset), timer_.elapsed());
        if (obj < best_obj) best = x, best_obj = obj, polished = false;
        publish(x, "local mip");
      }, &st);
      base += st.effort;
      log_.log(2, "  LMIP: %d solutions, %lld steps, %lld lifts, min violated %lld, effort %lld, %.2fs\n",
               st.solutions, st.steps, st.lift_moves, st.min_violated, st.effort, timer_.elapsed());
      if (best.empty() && !st.closest.empty() && (min_viol < 0 || st.min_violated < min_viol)) {
        min_viol = st.min_violated;
        closest = std::move(st.closest);
      }
    }
    // Without a solution FJ stalls after ~1 s; fresh seeds often succeed where one failed. One
    // attempt also after a local-MIP solution (its objective row sometimes does better).
    // (With MILP_FJ_FIRST the first attempt ran already: more only while nothing was found.)
    constexpr int kAttempts = 8;
    for (int a = fj_first ? 1 : 0; a < kAttempts && (a == 0 || best.empty()) && !fj_stop_.load(); ++a) {
      if (a > 0 && !fj_sync_ && is_finite(incumbent_.load())) break;  // another heuristic succeeded
      if (!fj_attempt(a)) break;
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
    if (!best.empty() && !polished && !fj_stop_.load() &&
        std::find(p_.is_int.begin(), p_.is_int.end(), 0) != p_.is_int.end()) {
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
    // Opportunistic mode, during the tree: local MIP from each new incumbent (found by others), with
    // the incumbent as cutoff; at most MILP_LMIP_TREE (default 5%) of the elapsed time.
    static const double tree_share = knob("MILP_LMIP_TREE", 0.05);
    if (!fj_sync_ && lmip_effort > 0 && tree_share > 0) {
      double tried = kInf, spent = 0.0;
      for (unsigned seq = 1; !fj_stop_.load(); ++seq) {
        {
          std::unique_lock<std::mutex> lk(fj_mu_);
          fj_cv_.wait_for(lk, std::chrono::milliseconds(200), [&] { return fj_stop_.load(); });
        }
        const double inc_obj = incumbent_.load();
        if (fj_stop_.load() || !is_finite(inc_obj) || inc_obj >= tried) continue;
        if (spent > tree_share * timer_.elapsed()) continue;
        std::vector<double> inc;
        incumbent_x_global(inc);
        if (inc.empty()) continue;
        const double t0 = timer_.elapsed();
        LocalMipLimits ll;
        ll.effort = static_cast<long long>(knob("MILP_LMIP_TREE_EFFORT", 3e8));
        ll.stall_effort = ll.effort / 3;
        ll.report_effort = static_cast<long long>(knob("MILP_LMIP_REPORT", 3e7));
        ll.seed = params_.seed + 7919u * seq;
        ll.stop = &fj_stop_;
        ll.start = &inc;
        ll.cutoff = inc_obj;
        ll.time_limit = std::max(0.0, params_.time_limit - t0);
        LocalMipStats st;
        double found = kInf;
        local_mip(p_, ll, [&](const std::vector<double>& x, double obj) {
          found = obj;
          try_incumbent(x, "local mip");
        }, &st);
        tried = std::min(inc_obj, found);
        spent += timer_.elapsed() - t0;
        log_.log(3, "  LMIP from incumbent: %d solutions, effort %lld, %.2fs\n", st.solutions, st.effort,
                 timer_.elapsed());
      }
    }
    std::lock_guard<std::mutex> lk(fj_mu_);
    fj_done_ = true;
    fj_cv_.notify_all();
  }));
}

bool Search::fpr_run_lp_free(const std::function<void(const std::vector<double>&, const char*)>& publish,
                             double deadline, std::vector<double>& point) {
  // Opt-in (MILP_FPR=1) until the end-to-end A/B; a deterministic work budget (~0.25 s at most,
  // usually far less).
  static const bool enabled = knob("MILP_FPR", 0.0) != 0.0;
  if (!enabled) return false;
  FprLimits fl;
  fl.effort = static_cast<long long>(knob("MILP_FPR_EFFORT", 5e7));
  fl.seed = params_.seed;
  fl.stop = &fj_stop_;
  if (!fj_sync_) fl.time_limit = std::max(0.0, deadline - timer_.elapsed());
  FprStats fs;
  std::vector<double> x;
  const double t0 = timer_.elapsed();
  const bool ok = fix_propagate_repair(p_, fl, x, &fs);
  log_.log(2, "  FPR: %s%s%s, %d runs, %lld nodes, %lld repairs, effort %lld, %.2fs\n", ok ? "found by " : "failed",
           ok ? fs.strategy : "", fs.infeasible ? " (infeasible)" : "", fs.runs, fs.nodes, fs.repairs, fs.effort,
           timer_.elapsed() - t0);
  if (!ok) return false;
  publish(x, "fix-propagate-repair");
  point = std::move(x);
  return true;
}

bool Search::fpr_run_lp(const std::vector<double>& lp_x, std::vector<double>& sol) {
  if (!opts_.sub_mip_heuristics || static_cast<Index>(lp_x.size()) < p_.num_cols()) return false;
  FprLimits fl;
  fl.effort = static_cast<long long>(knob("MILP_FPR_LP_EFFORT", 5e7));
  fl.seed = params_.seed + 7u;
  fl.lp_x = &lp_x;
  fl.strategies = fpr_lp_portfolio();
  if (!params_.deterministic) fl.time_limit = std::max(0.0, std::min(10.0, params_.time_limit - timer_.elapsed()));
  FprStats fs;
  const double t0 = timer_.elapsed();
  const bool ok = fix_propagate_repair(p_, fl, sol, &fs);
  log_.log(3, "  root FPR (LP): %s%s, %d runs, effort %lld, %.2fs\n", ok ? "found by " : "failed", ok ? fs.strategy : "",
           fs.runs, fs.effort, timer_.elapsed() - t0);
  return ok;
}

void Search::fj_sync(long long release, bool lag) {
  if (!fj_sync_ || !fj_thread_.joinable()) return;
  std::vector<FjFound> found;
  {
    std::unique_lock<std::mutex> lk(fj_mu_);
    // Every solution stamped below `target` has been published once FJ passed a checkpoint at or
    // beyond it (or finished): the solutions taken do not depend on thread timing.
    const long long target = lag ? fj_prev_allowed_ : fj_allowed_;
    // Up to the time limit (or an interrupt): a run that reaches it is not deterministic anyway, and the helper's
    // moves can be slow (gfd-schedulen180f7d50m30k18, deterministic, 12 runs at a time: the root LP waited for its
    // syncs past the limit and was killed at 144 s of 120).
    // No time limit (or ~30 years): no deadline (an infinite duration would overflow the clock's integer one).
    const double rem = params_.time_limit - timer_.elapsed();
    const auto deadline = rem < 1e9 ? std::chrono::steady_clock::now() +
                                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                              std::chrono::duration<double>(std::max(0.0, rem)))
                                    : std::chrono::steady_clock::time_point::max();
    for (;;) {
      if (fj_done_ || fj_effort_ >= target) break;
      // The time limit or an interrupt: no sync (the caller's LP stops at its own limit check).
      if (interrupted() || std::chrono::steady_clock::now() >= deadline) return;
      fj_cv_.wait_until(lk, std::min(deadline, std::chrono::steady_clock::now() + std::chrono::milliseconds(200)));
    }
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

int Search::root_sep_threads() {
  int threads = threads_;
  if (fj_thread_.joinable() && !fj_sync_) {
    std::lock_guard<std::mutex> lk(fj_mu_);
    threads -= fj_done_ ? 0 : 1;
  }
  return std::max(1, threads);
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
  if (p_.num_rows() == 0) return false;
  const long long nnz = static_cast<long long>(p_.A.nnz());
  if (std::find(p_.is_int.begin(), p_.is_int.end(), 1) != p_.is_int.end()) {
    // MIP roots: at least MILP_CONCURRENT_MIP_NNZ (default 200000; 0: off, the dual simplex alone) presolved
    // nonzeros. A barrier win changes the root vertex and with it the cut loop and the search after it: with
    // every MIP root concurrent the core set (largest 173k presolved nonzeros, every root LP <= 12 s by the
    // dual simplex) was 1.095x slower (2 threads, 1 seed; the barrier won 10 of 63 roots, e.g.
    // neos-3381206-awhea 32 s -> unsolved, mc11 37 -> 65 s). Large roots the barrier wins clearly: triptim1
    // (466k), uccase9 (259k), rail01 (258k); the simplex wins anyway on app1-2 (236k, 0.3 s),
    // academictimetablesmall (240k, 2 s), satellites2-40 (263k, 4.7 s), snp-02-004-104 (303k, 6.6 s).
    static const long long min_mip_nnz = static_cast<long long>(knob("MILP_CONCURRENT_MIP_NNZ", 200000));
    return min_mip_nnz > 0 && nnz >= min_mip_nnz;
  }
  // Small LPs take milliseconds with the simplex.
  static const long long min_nnz = static_cast<long long>(knob("MILP_CONCURRENT_NNZ", 10000));
  return nnz >= min_nnz;
}

void Search::concurrent_root(DualSimplex& lp) {
  const double t0 = timer_.elapsed();
  DualSimplex blp = lp;
  std::atomic<bool> stop{false};
  BarrierStats bs;
  LpStatus bst = LpStatus::NumericalError;
  double t_barrier = 0.0;
  Logger quiet(0);
  // The simplex runs on this thread; a MIP root also has the FJ / local-MIP helper thread busy.
  const int busy = 1 + (fj_thread_.joinable() ? 1 : 0);
  // MILP_CONCURRENT_DELAY = d (MIP roots; default 2): the barrier starts only once the dual simplex has run
  // for d seconds: a root LP done by then keeps the simplex's vertex and costs no barrier work, and narrow
  // barrier wins (fast0507 1.2 s, neos-787933 0.5 s, physiciansched6-2 0.4 s) do not move the search.
  static const double delay_knob = knob("MILP_CONCURRENT_DELAY", 2.0);
  const bool mip = std::find(p_.is_int.begin(), p_.is_int.end(), 1) != p_.is_int.end();
  const double delay = mip ? delay_knob : 0.0;
  std::mutex mu;
  std::condition_variable cv;
  bool simplex_ended = false;
  std::thread tb(inherit_interrupt([&] {
    if (delay > 0.0) {
      std::unique_lock<std::mutex> lk(mu);
      if (cv.wait_for(lk, std::chrono::duration<double>(delay), [&] { return simplex_ended; })) return;
    }
    BarrierOptions bo;
    bo.log = &quiet;
    bo.stop = &stop;
    bo.time_limit = params_.time_limit - timer_.elapsed();
    bo.ipm.threads = std::max(1, threads_ - busy);
    // Memory: gfd-schedulen180f7d50m30k18, neos-5104907-jarama and neos-5114902-kasavu (8 GB limit) went past 10 GB
    // with the barrier (2.2 / 7.8 / 7.3 GB without it), also with a factor budget checked after the symbolic
    // analysis (its structures and the frontal work are large too). Not started at all when the process holds
    // 40% of the memory limit already; else its factor gets half the limit less that, at ~32 bytes per nonzero.
    if (params_.mem_limit_mb > 0) {
      const double limit = static_cast<double>(params_.mem_limit_mb) * 1048576.0;
      const double held = static_cast<double>(resident_bytes());
      if (held > 0.4 * limit) return;
      bo.ipm.max_factor_nnz = std::max(1LL, static_cast<long long>((0.5 * limit - held) / 32.0));
    }
    bst = barrier_solve(blp, p_.A, p_.cost, p_.col_lb, p_.col_ub, p_.row_lb, p_.row_ub, bo, &bs);
    t_barrier = timer_.elapsed() - t0;
    if (bst == LpStatus::Optimal && bs.used) stop.store(true);
  }));
  LpOptions so;
  so.tol = params_.tol;
  so.time_limit = params_.time_limit - timer_.elapsed();
  so.stop = &stop;
  const long long it0 = lp.iterations();
  const LpStatus sst = lp.solve(so);
  const double t_simplex = timer_.elapsed() - t0;
  const bool simplex_done = sst == LpStatus::Optimal || sst == LpStatus::Infeasible || sst == LpStatus::Unbounded;
  // Also at the time limit or an interrupt: the barrier checks its own limit, but its crossover only this flag
  // (savsched1, 120 s limit: the simplex stopped at 120 s, the join then waited for the crossover's LU
  // factorizations, ~14 s each, until 180-215 s).
  if (simplex_done || timer_.elapsed() >= params_.time_limit || interrupted()) stop.store(true);
  {
    std::lock_guard<std::mutex> lk(mu);
    simplex_ended = true;
  }
  cv.notify_all();
  tb.join();
  if (!simplex_done && bst == LpStatus::Optimal && bs.used) {
    // The simplex had not finished after these iterations: a lower bound on its cost of the root LP.
    root_lp_equiv_iters_ = lp.iterations() - it0;
    root_barrier_time_ = t_barrier;
    lp = blp;
    root_lp_lb_ = std::max(root_lp_lb_, lp.objective());  // before set_basis (barrier_basis_weights)
    if (mip) barrier_basis_weights(lp);
    log_.log(1, "Concurrent LP: barrier finished first (%.2fs: %d iterations, crossover and %lld simplex iterations;"
             " dual simplex %lld iterations)\n", t_barrier, bs.ipm.iterations, bs.cleanup_iterations, root_lp_equiv_iters_);
  } else if (simplex_done) {
    root_lp_race_iters_ = lp.iterations() - it0;
    if (sst == LpStatus::Optimal) root_lp_lb_ = std::max(root_lp_lb_, lp.objective());
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
  if (bs.used) root_barrier_time_ = bs.total_time;
  // The root LP bound also when the cut loop's warm re-solve (a fresh factorization of the crossover basis)
  // does not finish in time; before set_basis (barrier_basis_weights).
  if (bs.used && st == LpStatus::Optimal) root_lp_lb_ = std::max(root_lp_lb_, lp.objective());
  if (bs.used) barrier_basis_weights(lp);
  if (!bs.used) {
    log_.log(1, "Barrier: %s (%s); solving the root LP with the dual simplex\n", to_string(bs.ipm.status),
             to_string(st));
  } else if (st != LpStatus::Optimal) {
    log_.log(1, "Barrier: crossover cleanup ended %s; the dual simplex continues from its basis\n", to_string(st));
  }
}

void Search::detect_symmetry_root() {
  if (std::getenv("MILP_NO_SYMMETRY") != nullptr || !params_.symmetry || !opts_.sub_mip_heuristics) return;
  if (!p_.implications.empty()) return;  // the implications are not part of the symmetry detection
  if (!p_.quad_rows.empty()) return;     // nor the quadratic rows
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

bool Search::restart_allowed() const {
  static const bool no_restart = std::getenv("MILP_NO_RESTART") != nullptr;
  return opts_.allow_restart && !no_restart && root_rc_ready_;
}

RestartDecision Search::restart_check(double root_work, double root_time, Index* ints, Index* newly_fixed) {
  *ints = 0, *newly_fixed = 0;
  {
    std::lock_guard<std::mutex> lk(gfix_mu_);
    for (Index j = 0; j < p_.num_cols(); ++j) {
      if (!p_.is_int[j]) continue;
      ++*ints;
      *newly_fixed += glb_[j] == gub_[j] && p_.col_lb[j] != p_.col_ub[j];
    }
  }
  return root_restart_decision(opts_, params_.deterministic, *ints, *newly_fixed, root_work, root_time);
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
  std::lock_guard<std::mutex> lk(tree_restart_mu_);
  static const double frac = knob("MILP_TREE_RESTART", 0.1);  // <= 0: off
  // Opt-in MILP_EST_RESTART=1 (A/B against off: 1.021x, 125 -> 121 of 136; mas74 restarted at 9-15k nodes
  // and lost its solve in both seeds, eil33-2 restarted at tree weight 0.645, swath3 gained once).
  static const bool est_on = knob("MILP_EST_RESTART", 0) > 0;
  const bool fix_rule = opts_.allow_tree_restart && frac > 0.0;
  const bool est_rule = est_on && opts_.allow_est_restart;
  if (!(fix_rule || est_rule) || tree_restart_.load() || stop_.load()) return;
  Index ints = 0, free_ints = 0;
  for (Index j = 0; j < p_.num_cols(); ++j) {
    if (!p_.is_int[j]) continue;
    ++ints;
    free_ints += p_.col_lb[j] != p_.col_ub[j];
  }
  const Index fixed = count_global_fixed();
  // The estimator votes at every check (also without an incumbent: its gap ratio is then 1).
  const int votes = est_rule ? est_restart_vote(fixed, free_ints) : 0;
  // Not once the tree has done most of the work: at least half of the root gap still open, and in
  // opportunistic mode at most 60% of the time limit used.
  const double inc = incumbent_.load(), lb = pool_->lower_bound();
  if (!is_finite(inc) || !is_finite(lb) || !is_finite(root_bound_)) return;
  if (inc - lb < 0.5 * (inc - root_bound_)) return;
  const double used = is_finite(params_.time_limit) ? (timer_.elapsed() - t_start_) / (params_.time_limit - t_start_) : 0.0;
  if (!det && used > 0.6) return;
  // The later in the tree, the more fixings it takes (the restart throws the tree away: mkp_hard_1
  // restarted after 20-35k nodes with 21% fixed took 4 s instead of 3).
  static const double early = knob("MILP_TREE_RESTART_NODES", 5000);
  const double late = std::max(1.0, static_cast<double>(nodes_.load()) / early);
  const Index need = static_cast<Index>(
      late * static_cast<double>(std::max<Index>(20, static_cast<Index>(frac * static_cast<double>(free_ints - tree_fixed0_)))));
  if (fix_rule && fixed - tree_fixed0_ >= need) {
    log_.log(1, "Restart: %d of %d integer variables fixed globally in the tree (%d at its start), %lld nodes\n", fixed,
             ints, tree_fixed0_, nodes_.load());
    tree_restart_ = true;
    stop_ = true;
    return;
  }
  // Estimation-driven: enough consecutive huge votes (round(r (10 + leaves / 1000) 1.5^k), k the tree
  // restarts so far), and a gain: at least max(10, MILP_EST_GATE (1%) of the free integer columns)
  // fixed since the tree started (tr12-30 and gen-ip054 fix none: their trees are huge, but a restart
  // would bring nothing). In opportunistic mode within the first half of the time limit. (Tests:
  // MILP_EST_GATE < 0, no gain needed.)
  if (votes == 0 || (!det && used > 0.5)) return;
  static const double gate = knob("MILP_EST_GATE", 0.01);
  const Index gain = fixed - tree_fixed0_;
  if (gate >= 0.0 &&
      gain < std::max<Index>(10, static_cast<Index>(gate * static_cast<double>(free_ints - tree_fixed0_)))) {
    return;
  }
  const double share = free_ints > 0 ? static_cast<double>(fixed) / static_cast<double>(free_ints) : 0.0;
  const double r = (1.0 - share) * (1.0 - share);
  const double need_votes = std::round(r * (10.0 + static_cast<double>(tw_.leaves()) / 1000.0) *
                                       std::pow(1.5, static_cast<double>(opts_.tree_restarts_done)));
  if (static_cast<double>(votes) < need_votes) return;
  log_.log(1, "Restart: tree estimated at %.3g nodes after %lld (weight %.3g, %d votes), %d of %d integer variables "
           "fixed globally in the tree (%d at its start)\n", est_nhat_, nodes_.load(), tw_.weight(), votes, fixed,
           ints, tree_fixed0_);
  tree_restart_ = true;
  est_restart_ = true;
  stop_ = true;
}

int Search::est_restart_vote(Index fixed, Index free_ints) {
  static const double factor = knob("MILP_EST_FACTOR", 20);
  static const long long min_nodes = static_cast<long long>(knob("MILP_EST_MIN_NODES", 1000));
  const long long n = nodes_.load();
  if (n < min_nodes || (est_last_ >= 0 && n - est_last_ < 100)) return est_votes_;  // at most one vote per 100 nodes
  est_last_ = n;
  const double w = tw_.weight();
  const double inc = incumbent_.load(), lb = pool_->lower_bound();
  const double gap = is_finite(inc) && is_finite(lb) ? std::max(0.0, inc - lb) : kInf;
  if (est_n_ref_ < 0) {
    est_n_ref_ = n, est_w_ref_ = w, est_gap_ref_ = gap;
    return est_votes_ = 0;
  }
  const double share = free_ints > 0 ? static_cast<double>(fixed) / static_cast<double>(free_ints) : 0.0;
  const double r = std::max(1e-6, (1.0 - share) * (1.0 - share));
  est_nhat_ = w > est_w_ref_ ? static_cast<double>(est_n_ref_) + static_cast<double>(n - est_n_ref_) * (1.0 - w) / (w - est_w_ref_)
                             : kInf;
  // Gap ratio 1 without a finite gap on either side (no incumbent yet).
  const double ratio = is_finite(gap) && is_finite(est_gap_ref_) && gap > 0.0 ? est_gap_ref_ / gap : 1.0;
  const bool huge = ratio < 1.0 + 0.05 / r && est_nhat_ >= factor * r * static_cast<double>(n);
  static const bool trace = knob("MILP_EST_TRACE", 0) > 0;
  if (trace) {
    log_.log(0, "EST n %lld w %.4g ref %lld / %.4g nhat %.3g ratio %.4g r %.3g fixed %d -> %s, votes %d, leaves %lld\n", n, w,
             est_n_ref_, est_w_ref_, est_nhat_, ratio, r, fixed, huge ? "huge" : "reset", huge ? est_votes_ + 1 : 0,
             tw_.leaves());
  }
  if (huge) return ++est_votes_;
  est_n_ref_ = n, est_w_ref_ = w, est_gap_ref_ = gap;
  return est_votes_ = 0;
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
  res.pc = pc_.snapshot();
}

MipResult Search::run() {
  MipResult res;
  t_start_ = timer_.elapsed();
  checks0_ = DualSimplex::check_counts();
  // Everything this search runs stops once it is solved (trivial_solved; opportunistic mode).
  SearchStopScope stop_scope(params_.deterministic ? nullptr : &solved_stop_);
  debug_root_ = opts_.sub_mip_heuristics;
  // Sub-MIP heuristics keep the old permanent node cut rows (short searches; pool separation at
  // every node would mostly add LP work there).
  if (!opts_.sub_mip_heuristics) use_pool_ = false;
  load_debug_solution();
  if (is_finite(trivial_lb_)) log_.log(2, "  trivial objective bound %.10g\n", p_.display_sense * (trivial_lb_ + p_.offset));
  if (is_finite(opts_.cutoff)) {
    external_cutoff_ = opts_.cutoff - p_.offset;
    cutoff_.store(prune_threshold(external_cutoff_));
  }
  // The start is the user's (solver.cpp) or, after a restart, the incumbent before it.
  const char* start_source = opts_.restarts_done + opts_.tree_restarts_done > 0 ? "restart" : "start";
  // A rejected start: the cutoff from its own objective (SearchOptions::start_objective), not from the point.
  const double start_obj =
      opts_.start.empty() ? kInf : is_finite(opts_.start_objective) ? opts_.start_objective : p_.objective(opts_.start);
  bool start_taken = !opts_.start.empty() && try_incumbent(opts_.start, start_source);
  // Start repair: presolve's dual reductions (dominated columns, dual bounds) can make the crushed image of a
  // feasible start infeasible here. Rather than keep only its objective as a cutoff, fix the start's integer
  // values except in the violated rows and complete it with a small sub-MIP, so that the start can still feed
  // the guided dive and RINS. Only the user's start (solver.cpp); MILP_START_REPAIR=0: off.
  static const bool start_repair = knob("MILP_START_REPAIR", 1) > 0;
  if (!opts_.start.empty() && !start_taken && start_repair && opts_.sub_mip_heuristics &&
      opts_.restarts_done + opts_.tree_restarts_done == 0) {  // the user's start only, not a restart's incumbent
    const double t0 = timer_.elapsed();
    std::vector<double> act(p_.num_rows(), 0.0);
    for (Index j = 0; j < p_.num_cols(); ++j)
      for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) act[p_.A.index[k]] += p_.A.value[k] * opts_.start[j];
    std::vector<char> bad(p_.num_rows(), 0);
    Index nbad = 0;
    for (Index i = 0; i < p_.num_rows(); ++i)
      if (act[i] < p_.row_lb[i] - 1e-6 || act[i] > p_.row_ub[i] + 1e-6) bad[i] = 1, ++nbad;
    std::vector<double> lb = p_.col_lb, ub = p_.col_ub;
    Index fixed = 0, ints = 0;
    for (Index j = 0; j < p_.num_cols(); ++j) {
      if (!p_.is_int[j]) continue;
      ++ints;
      bool touches = false;
      for (Index k = p_.A.begin(j); k < p_.A.end(j) && !touches; ++k) touches = bad[p_.A.index[k]];
      const double v = std::round(opts_.start[j]);
      if (touches || v < lb[j] || v > ub[j]) continue;
      lb[j] = ub[j] = v;
      ++fixed;
    }
    SubMipLimits lim;
    lim.node_limit = 500;
    const double left = std::max(0.0, params_.time_limit - timer_.elapsed());
    // Deterministic mode: no wall-clock cap (it would make the result depend on the machine's speed), an LP
    // iteration limit instead.
    lim.time_limit = params_.deterministic ? left : std::min(5.0, 0.1 * left);
    if (params_.deterministic) lim.iteration_limit = 200000;
    std::vector<double> sol;
    const bool ok = fixed > 0 && lim.time_limit > 0.0 && solve_sub_mip(p_, lb, ub, kInf, lim, sol) &&
                    try_incumbent(sol, "start repair");
    start_taken = ok;
    if (ok) {
      log_.log(1, "Start solution: violates %d rows after presolve; repaired (%d of %d integers kept), objective %.10g\n",
               nbad, fixed, ints, p_.display_sense * (incumbent_.load() + p_.offset));
    } else {
      log_.log(1, "Start solution: violates %d rows after presolve; repair failed, its objective %.10g bounds the "
               "search\n", nbad, p_.display_sense * start_obj);
    }
    log_.log(2, "  start repair: %s, %d violated rows, %d of %d integers fixed, %.2fs\n", ok ? "found" : "failed", nbad,
             fixed, ints, timer_.elapsed() - t0);
  }
  // The start's own objective stays a cutoff unless the start itself (or a repair at least as good) is the incumbent.
  if (!opts_.start.empty() && (!start_taken || incumbent_.load() > start_obj - p_.offset)) {
    external_cutoff_ = std::min(external_cutoff_, start_obj - p_.offset);
    cutoff_.store(prune_threshold(external_cutoff_));
  }
  if (!opts_.start.empty()) inc_start_ = start_obj - p_.offset;
  if (!opts_.pc_init.empty()) {
    // Restart: the pseudocosts of the search before, counts capped at the reliability threshold.
    // MILP_RESTART_PC=1: the carried entries are history (counts capped at MILP_RESTART_PC_CAP, default
    // the reliability threshold); 2: estimates only, until a column has observations of its own.
    const int mode = static_cast<int>(knob("MILP_RESTART_PC", kRestartPcDefault));
    const int cap = static_cast<int>(knob("MILP_RESTART_PC_CAP", knob("MILP_REL", 4)));
    pc_.load(opts_.pc_init.capped(std::max(1, cap)), mode >= 2);
    Index carried = 0;
    for (size_t k = 0; k + 1 < opts_.pc_init.cnt.size(); k += 2) carried += opts_.pc_init.cnt[k] + opts_.pc_init.cnt[k + 1] > 0;
    log_.log(2, "  restart: pseudocosts of %d columns carried\n", carried);
  }
  root_lp_.load(p_.A, p_.cost, p_.col_lb, p_.col_ub, p_.row_lb, p_.row_ub);
  root_lp_.set_seed(params_.seed);
  cut_data_.build(p_);
  pool_ = std::make_unique<NodePool>(threads_, this, &path_bytes_);
  pool_->set_tree_weight(&tw_);
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

  // Root node is processed on the calling thread to keep its log ordered (not when the start point
  // solved the problem: trivial_solved).
  res.root_bound = trivial_lb_ + p_.offset;
  if (!solved_.load()) {
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
      // The orbitopes' static fixings (orbitopal fixing on the model bounds: the zeros above the diagonal) on the
      // root LP before the root LP and cut loop, so that its cuts see them (graph20-20-1rand: a 37 x 37 packing
      // orbitope, 666 zeros; without them the cut loop stayed at -37, the root node's propagation reached -19).
      // They hold at every node (node propagation derives them again), so the cuts derived with them are valid
      // for the representatives the search keeps; they are not global bounds, and a restart's problem (its
      // global bounds plus these cuts) still contains a representative of every optimum.
      // Opt-in MILP_ORB_ROOT_FIX=1 (Mac, 4 threads: graph20-20-1rand root -19 -> -17.64 but its tree no shorter in one
      // run, drayage-25-23 2.7 / 5.4 s vs 3.4 / 3.5 s, neos-911970 1.3 / 1.1 vs 2.3 / 1.2 s: not measured broadly).
      static const bool orb_root_fix = knob("MILP_ORB_ROOT_FIX", 0) > 0;
      if (orb_root_fix && !orbs_.empty()) {
        std::vector<double> lb = p_.col_lb, ub = p_.col_ub;
        std::vector<Index> tight;
        std::vector<char> ones;
        bool ok = true;
        for (const Orbitope& o : orbs_) ok = ok && orbitope_propagate(o, lb, ub, tight, ones);
        if (ok) {
          for (Index j : tight) w.lp.set_col_bounds(j, lb[j], ub[j]);
          log_.log(2, "  orbitopes: %zu static fixings on the root LP\n", tight.size());
        }
      }
    }
    Node children[2];
    pool_->set_worker_bound(0, -kInf);
    if (use_barrier_root()) barrier_root(w.lp);
    else if (use_concurrent_root()) concurrent_root(w.lp);
    if (opts_.root_cuts && p_.is_int.end() != std::find(p_.is_int.begin(), p_.is_int.end(), 1)) {
      std::vector<Cut> dropped;
      static const bool no_lurk = std::getenv("MILP_NO_LURKRC") != nullptr;
      const bool lurk = !no_lurk && std::getenv("MILP_NO_RCFIX") == nullptr;
      if (lurk) {
        lurk_lo_.assign(p_.num_cols(), -kInf);
        lurk_up_.assign(p_.num_cols(), -kInf);
      }
      // The first call comes after the initial LP solve: its iterations (the root dive budget).
      std::function<void(const DualSimplex&)> on_lp = [this, lurk](const DualSimplex& lp) {
        if (root_lp0_iters_ < 0) root_lp0_iters_ = lp.iterations();
        // Sub-MIP job progress after every LP of the cut loop, not only at the root's end: the deterministic
        // barrier waits for a running job until its progress passes its due point, and a job reporting nothing
        // until its root was done kept it waiting through the whole root (pk1, 4 threads: 20-70 ms per wait;
        // waits 2.32 -> 2.23 s alone, 2.08 -> 1.38 s with MILP_DET_JOB_SHARE=1). The final progress is the same,
        // so the jobs enter where they did.
        // MILP_DET_ROOT_PROGRESS=0: at the root's end only.
        static const bool root_progress = knob("MILP_DET_ROOT_PROGRESS", 1) > 0;
        if (opts_.progress && root_progress) {
          opts_.progress->fetch_add(lp.iterations() - root_progress_);
          root_progress_ = lp.iterations();
        }
        root_lp_lb_ = std::max(root_lp_lb_, lp.objective());  // optimal LPs only: a global bound
        if (lurk) record_lurking(lp);
      };
      // Reduced-cost fixing between cut rounds against the current cutoff (deterministic: the one at the start).
      const double cutoff0 = cutoff_.load();
      std::function<double()> cutoff_fn = [this, cutoff0, det = params_.deterministic]() {
        return det ? cutoff0 : cutoff_.load();
      };
      // Stop the cut rounds once the incumbent closes the gap (deterministic: the incumbent at the start).
      const double inc0 = incumbent_.load();
      std::function<bool(double)> gap_fn = [this, inc0, det = params_.deterministic](double lb) {
        return det ? gap_closed_at(lb, inc0) : gap_closed(lb);
      };
      // Deterministic mode: the FJ / local-MIP helper's points also enter during a long initial root LP, every
      // MILP_DET_ROOT_SYNC iterations (default 5000; 0: at the end of the root cut loop only), each release
      // paying for those iterations at the tree's rate (iterations x rows, MILP_FJ_RATE). Before, the helper
      // stopped after 1e6 effort units until then (ns1116954: local MIP's point is optimal by the trivial bound,
      // but the 131k-row root LP took 53 s and the root node the rest of the time limit: no incumbent). The
      // solve stops once an incumbent meets the trivial bound.
      static const long long det_root_sync = static_cast<long long>(knob("MILP_DET_ROOT_SYNC", 5000));
      std::function<bool(long long)> lp_checkpoint;
      if (fj_sync_ && det_root_sync > 0) {
        lp_checkpoint = [this](long long) {
          static const double fj_rate = knob("MILP_FJ_RATE", 1.0);
          const double t0 = timer_.elapsed();  // statistics only
          fj_sync(static_cast<long long>(fj_rate * static_cast<double>(det_root_sync) * p_.num_rows()), true);
          root_fj_wait_ += timer_.elapsed() - t0;
          ++root_fj_syncs_;
          return !solved_.load();
        };
      }
      // Objective propagation with clique implications against the cutoff (deterministic: the one at the start)
      // after the initial LP and between the rounds, whenever the cutoff improved: global fixings, kept in the LP
      // and published for the root node (gfix_log_; glb_ is set up there). hypothyroid-k1: local MIP finds the
      // optimum -2851 at 1.3 s, before the initial LP ends; 74 fixings lift the LP from -2902.85 to -2851, while 29
      // cut rounds took it to -2861.5 in 27 s.
      std::function<Index(std::vector<double>&, std::vector<double>&, std::vector<Index>&)> fix_fn =
          [this, cutoff0, det = params_.deterministic](std::vector<double>& lb, std::vector<double>& ub,
                                                        std::vector<Index>& changed) -> Index {
        const double co = det ? cutoff0 : cutoff_.load();
        if (!is_finite(co) || co >= ocl_cut_cutoff_) return 0;
        std::lock_guard<std::mutex> lk(gfix_mu_);
        ocl_cut_cutoff_ = co;
        const Index k = obj_clique_fix(co, lb, ub, changed);
        for (Index j : changed) {
          if (!glb_.empty()) glb_[j] = std::max(glb_[j], lb[j]), gub_[j] = std::min(gub_[j], ub[j]);
          gfix_log_.push_back({j, lb[j], ub[j]});
        }
        return k;
      };
      const CutLoopStats cs = root_cut_loop(p_, w.lp, params_, log_, timer_, inc0, &cut_data_,
                                            !opts_.sub_mip_heuristics || opts_.no_zerohalf,
                                            use_pool_ ? &dropped : nullptr, on_lp, opts_.stop_flag, cutoff_fn,
                                            opts_.root_cut_rounds, gap_fn,
                                            root_lp_race_iters_ + (barrier_equiv() ? root_lp_equiv_iters_ : 0),
                                            lp_checkpoint, det_root_sync, root_sep_threads(), fix_fn);
      if (barrier_equiv() && root_barrier_time_ > 0.0 && root_lp_equiv_iters_ == 0 && !params_.deterministic &&
          cs.resolve_iterations >= 200 && cs.resolve_time > 0.0) {
        // --lp-method barrier (no simplex raced it): the barrier's time at the rate of the cut rounds' warm
        // re-solves (wall clock: opportunistic mode only). triptim1: 18.8 s at ~1500 iterations/s.
        root_lp_equiv_iters_ = static_cast<long long>(root_barrier_time_ * static_cast<double>(cs.resolve_iterations) /
                                                      cs.resolve_time);
      }
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
      if (opts_.progress) opts_.progress->fetch_add(w.lp.iterations() - root_progress_);
      if (use_pool_) {
        const long long nnz = static_cast<long long>(p_.A.index.size());
        ncp_max_ = static_cast<size_t>(knob("MILP_POOL_MAX", 20000));
        ncp_max_nnz_ = static_cast<long long>(knob("MILP_POOL_NNZ_MULT", 1.0) * static_cast<double>(std::clamp<long long>(10 * nnz, 200000, 4000000)));
        pool_publish(dropped, nullptr);
      }
      // The root LP before cuts and before the objective-clique fixings (ROOTSTAT lp=).
      root_cs_before_ = cs.rounds > 0 || cs.global_fixed > 0 ? cs.bound_before : w.lp.objective();
      root_cs_rounds_ = cs.rounds;
      root_cs_cuts_ = cs.cuts_added;
      if (cs.rounds > 0) {
        log_.log(1, "Cuts: %d rounds, %d cuts, bound %.10g -> %.10g\n", cs.rounds, cs.cuts_added,
                 p_.display_sense * (cs.bound_before + p_.offset), p_.display_sense * (cs.bound_after + p_.offset));
      }
    }
    // Deterministic point: FJ results enter before the root node (after syncs during the root LP: those up to
    // the previous release, as in the tree, so that the root does not wait for the latest one).
    if (root_fj_syncs_ > 0) log_.log(2, "  root LP: %lld FJ syncs (waited %.2fs)\n", root_fj_syncs_, root_fj_wait_);
    fj_sync(3 * w.lp.iterations() * p_.num_rows(), root_fj_syncs_ > 0);
    fj_iters_mark_ = lp_iters_.load();
    // Solved (deterministic mode: an incumbent at the trivial bound from a sync above): the root LP was stopped.
    int nc = solved_.load() ? 0 : process(w, root, children);
    while (root_redo_ && !stop_.load()) {
      // Root strong branching fixed variables globally: a few cut rounds on the tightened root LP
      // (global bounds only: the root node's reduced-cost fixings are dropped first), then the root
      // again (propagation of the fixings, LP, another strong branching pass).
      root_redo_ = false;
      static const int kSbCuts = static_cast<int>(knob("MILP_ROOT_SB_CUTS", 5));
      for (Index j : w.changed) w.lp.set_col_bounds(j, w.glb[j], w.gub[j]);
      if (kSbCuts > 0 && root_cs_rounds_ > 0) {
        CutLoopOptions co;
        co.max_rounds = kSbCuts;
        co.max_rounds_cheap = 0;
        co.degenerate_rounds = 0;
        co.keep_rows = true;  // the root cut loop's rows stay (its cuts are what the bound rests on)
        co.tol = params_.tol;
        co.zerohalf = opts_.sub_mip_heuristics && !opts_.no_zerohalf;
        co.stop = opts_.stop_flag;
        co.threads = root_sep_threads();
        if (is_finite(params_.time_limit)) co.time_limit = timer_.elapsed() + 0.1 * (params_.time_limit - timer_.elapsed());
        std::vector<Cut> dropped;
        const CutLoopStats cs = cut_loop(p_, w.lp, co, log_, timer_, cutoff(w), nullptr, &cut_data_,
                                         use_pool_ ? &dropped : nullptr);
        if (use_pool_) pool_publish(dropped, nullptr);
        root_cs_rounds_ += cs.rounds;
        root_cs_cuts_ += cs.cuts_added;
        if (cs.rounds > 0) {
          log_.log(2, "  cuts after root strong branching: %d rounds, %d cuts, bound %.10g -> %.10g\n", cs.rounds,
                   cs.cuts_added, p_.display_sense * (cs.bound_before + p_.offset),
                   p_.display_sense * (cs.bound_after + p_.offset));
        }
      }
      nc = process(w, root, children);
    }
    if (root_sb_probes_ > 0) {
      log_.log(1, "Root strong branching: %d passes%s, %lld probes, %lld LP iterations, %lld fixings, bound %.10g\n",
               root_sb_passes_, root_sb_aborted_ > 0 ? " (aborted: nothing learned)" : "", root_sb_probes_,
               root_sb_iters_, root_sb_fixed_, p_.display_sense * (std::max(root_bound_.load(), root_sb_bound_) + p_.offset));
    }
    root_done_ = true;
    {
      // Machine-readable root summary (benchmarks/tools/rootgap.py): bound after the root node (the
      // incumbent if the root was pruned), the root LP bound before cuts and the root cut loop.
      double rb = incumbent_.load();
      if (nc > 0) {
        rb = kInf;
        for (int k = 0; k < nc; ++k) rb = std::min(rb, std::max(children[k].bound, root_bound_.load()));
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
    res.root_lp_work = root_work(w.lp);
    res.root_time = root_elapsed();
    res.root_bound = std::max(root_bound_.load(), trivial_lb_) + p_.offset;
    res.root_lns_ran = root_lns_ran_;
    res.root_lns_improved = root_lns_improved_;
    for (int k = 0; k < nc; ++k) pool_->push(std::move(children[k]));
  }

  // A root that ended past the time limit (its heuristics run up to it) starts no restart and no tree: the
  // workers check the limit after a node, and a node LP of a large model refactorizes before its own check
  // (ns1116954 deterministic, 2 threads: root RENS to 121.4 s, then a 2 s tree round, 123.4 s in all).
  if (!stop_.load() && timer_.elapsed() > params_.time_limit) {
    std::lock_guard<std::mutex> lk(stop_mu_);
    if (stop_reason_ == Status::Unknown) stop_reason_ = Status::TimeLimit;
    stop_ = true;
  }

  // Root restart: if the root fixed many integer variables globally (reduced-cost fixing, root strong
  // branching, conflicts), hand the tightened problem plus root cuts back for a fresh presolve.
  // Rule and budget: SearchOptions (set by solver.cpp, see RestartRule there), root_restart_decision.
  if (restart_allowed() && !stop_.load() && pool_->size() > 0) {
    Index ints = 0, newly_fixed = 0;
    RestartDecision d = restart_check(res.root_lp_work, res.root_time, &ints, &newly_fixed);
    if (root_sb_skipped_ && d.outcome != RestartDecision::Restart) {
      // Decided before root strong branching, which was skipped for it (opportunistic mode: the root's
      // time grew since; the fixings can only have grown).
      log_.log(2, "  root restart: decided before root strong branching (now: %s)\n",
               d.refused ? d.refused : "too few fixings");
      d.outcome = RestartDecision::Restart;
    }
    if (ints > 0 && d.outcome == RestartDecision::TooFew) {
      log_.log(2, "  root restart: %d of %d integer variables fixed at the root (%d needed)\n", newly_fixed, ints, d.need);
    }
    if (d.outcome != RestartDecision::TooFew) {
      // From the second root restart on: only after a restart whose presolve removed enough, and while
      // the restarted roots stay within the budget (work in deterministic mode, time otherwise).
      if (d.outcome == RestartDecision::Refused) {
        log_.log(1, "Restart: %d of %d integer variables fixed at the root, no restart (%s)\n", newly_fixed, ints, d.refused);
      } else {
        log_.log(1, "Restart %d: %d of %d integer variables (%.1f%%) fixed at the root\n", opts_.restarts_done + 1,
                 newly_fixed, ints, 100.0 * static_cast<double>(newly_fixed) / static_cast<double>(ints));
        fill_restart(res, false);
        stop_ = true;  // skip the tree; the caller restarts
      }
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
      // Root heuristics beside the tree (MILP_ROOT_HEUR_ASYNC): H threads run them while the tree starts on
      // the others, then become tree workers too.
      const int H = heur_w_ ? async_heur_threads() : 0;
      std::vector<std::thread> threads;
      for (int t = 0; t < threads_ - H; ++t) threads.emplace_back(inherit_interrupt([this, t] { worker_loop(t); }));
      std::thread heur;
      if (H > 0) {
        log_.log(2, "  root heuristics beside the tree on %d of %d threads\n", H, threads_);
        heur = std::thread(inherit_interrupt([this, H] {
          {
            LocalStopScope scope(&heur_stop_);
            root_heuristics(*heur_w_, H, heur_t_root_);
          }
          log_.log(2, "  root heuristics done at %.2fs, %lld nodes\n", timer_.elapsed(), nodes_.load());
          // Their incumbents fix columns globally (root reduced-cost fixing): the restart the root would have
          // made after them, now (worker 0 checks only every 128 of its nodes).
          check_tree_restart(false);
          std::vector<std::thread> late;
          for (int t = threads_ - H + 1; t < threads_; ++t) late.emplace_back(inherit_interrupt([this, t] { worker_loop(t); }));
          worker_loop(threads_ - H);
          for (auto& th : late) th.join();
        }));
      }
      for (auto& th : threads) th.join();
      if (heur.joinable()) {
        heur_stop_ = true;
        heur.join();
        res.root_lns_ran = root_lns_ran_;
        res.root_lns_improved = root_lns_improved_;
      }
      heur_w_.reset();
    }
    maybe_log(true);
    log_.log(2, "Tree weight %.6g from %lld leaves (%.6g open), %lld nodes; last size estimate %.3g nodes\n",
             tw_.weight(), tw_.leaves(), pool_->open_weight(), nodes_.load(), est_nhat_);
    if (conf_learned_.load() > 0) {
      log_.log(1, "Conflicts: %lld learned (%lld from propagation), %lld nodes pruned by conflict propagation\n",
               conf_learned_.load(), prop_conf_learned_.load(), conf_pruned_.load());
      if (prop_conf_learned_.load() > 0) {
        log_.log(2, "  clauses: %.1f literals on average, %lld LP proofs not analysed (trail mismatch)\n",
                 static_cast<double>(prop_conf_lits_.load()) / static_cast<double>(prop_conf_learned_.load()),
                 trail_mismatch_.load());
      }
      if (gr_calls_.load() > 0) {
        const long long n = std::max(1LL, gr_learned_.load());
        log_.log(1, "  cut-based conflicts: %lld calls, %lld learned (%lld UIP, %lld partial), %.1f nonzeros,"
                 " %.2f steps/call, %lld reduced (%lld MIR), %.2f s; failures: start %lld, no step %lld, clause %lld,"
                 " reduction %lld, cont. decision %lld, limit %lld, numerics %lld, final %lld\n",
                 gr_calls_.load(), gr_learned_.load(), gr_uip_.load(), gr_partial_.load(),
                 static_cast<double>(gr_nnz_.load()) / static_cast<double>(n),
                 static_cast<double>(gr_steps_.load()) / static_cast<double>(gr_calls_.load()), gr_reduced_.load(),
                 gr_mir_.load(), 1e-9 * static_cast<double>(gr_ns_.load()), gr_fail_[1].load(), gr_fail_[2].load(),
                 gr_fail_[3].load(), gr_fail_[4].load(), gr_fail_[5].load(), gr_fail_[6].load(), gr_fail_[7].load(),
                 gr_fail_[8].load());
      }
    }
    if (oa_rounds_.load() > 0) {
      log_.log(1, "Outer approximation at nodes: %lld rounds, %lld cuts\n", oa_rounds_.load(), oa_cuts_.load());
    }
    if (use_pool_ && ncp_rounds_.load() > 0) {
      log_.log(1, "Cut pool: %zu cuts (%lld deleted), %lld node rounds re-added %lld cuts, %lld rows aged out; rounds"
               " at every %d-th node, %lld strips\n",
               ncp_.size(), ncp_purged_.load(), ncp_rounds_.load(), ncp_added_.load(), ncp_removed_.load(),
               pool_interval_.load(), ncp_stripped_.load());
      if (log_.verbosity() >= 2) {
        // Exact duplicates (same coefficients) among the live pool cuts.
        std::lock_guard<std::mutex> lk(ncp_mu_);
        std::unordered_set<uint64_t> seen;
        size_t live = 0;
        for (const auto& c : ncp_) {
          if (!c) continue;
          ++live;
          uint64_t h = 1469598103934665603ULL;
          for (size_t k = 0; k < c->idx.size(); ++k) {
            uint64_t b;
            std::memcpy(&b, &c->val[k], sizeof b);
            h = (h ^ static_cast<uint64_t>(c->idx[k])) * 1099511628211ULL;
            h = (h ^ b) * 1099511628211ULL;
          }
          seen.insert(h);
        }
        log_.log(2, "  cut pool: %zu live cuts, %zu distinct coefficient vectors (%lld duplicates mapped to live cuts);"
                 " %lld node rounds without a bound change, %lld pool rows removed without ever binding\n", live,
                 seen.size(), ncp_dups_, ncp_flat_rounds_.load(), ncp_never_bind_.load());
      }
    }
    if (mem_peak_.load() > 0) {
      log_.log(mem_drains_.load() > 0 || mem_stripped_.load() > 0 ? 1 : 2,
               "Memory: tree peak %.1f MB (of %.1f MB), %lld drain phases (%lld left, bound +%.6g in them),"
               " %lld stored bases dropped, %lld cut-off nodes purged, %lld queue compactions\n",
               mem_peak_.load() / 1048576.0, mem_limit_bytes() / 1048576.0, mem_drains_.load(), mem_drain_exits_,
               drain_rise_, mem_stripped_.load(), mem_purged_, pool_->compactions());
    }
    if (lp_work_total_.load() > 0) {
      log_.log(2, "  propagation work %.3g, %.1f%% of the node LP work\n", static_cast<double>(prop_work_total_.load()),
               100.0 * static_cast<double>(prop_work_total_.load()) / static_cast<double>(lp_work_total_.load()));
    }
    if (obj_prop_calls_.load() + obj_prop_skipped_.load() > 0) {
      log_.log(2, "  objective propagation: %lld nodes, %lld useful, %lld skipped\n", obj_prop_calls_.load(),
               obj_prop_useful_.load(), obj_prop_skipped_.load());
    }
    if (ocl_calls_ > 0) {
      log_.log(2, "  objective cliques: %lld calls, %lld columns fixed, %.3fs\n", ocl_calls_, ocl_fixed_, ocl_time_);
    }
    if (sb_probes_.load() > 0) {
      log_.log(1, "Strong branching: %lld probes, %lld LP iterations (%.1f%% of all)\n", sb_probes_.load(), sb_iters_.load(),
               100.0 * static_cast<double>(sb_iters_.load()) / static_cast<double>(std::max(1LL, lp_iters_.load())));
      if (sb_pl_stops_.load() > 0) log_.log(1, "Strong branching: %lld early stops (probabilistic lookahead)\n", sb_pl_stops_.load());
      if (sb_aps_nodes_.load() > 0) log_.log(1, "Branching: %lld nodes scored with ancestral pseudocosts\n", sb_aps_nodes_.load());
      if (sb_flat_nodes_.load() > 0) {
        log_.log(2, "Branching: %lld of %lld nodes without pseudocost gains (conflict / inference scores)\n",
                 sb_flat_nodes_.load(), sb_calls_.load());
      }
    }
    if (tdive_calls_.load() + hsub_calls_.load() > 0) {
      // Sub-MIP iterations are not part of the search's LP iterations (shares relative to them).
      const double all = static_cast<double>(std::max(1LL, lp_iters_.load()));
      log_.log(1, "Heuristics: %lld tree dives (%lld found, %lld improved, %lld at their limit), %lld LP iterations"
               " (%.1f%%); %lld sub-MIPs (%lld improved), %lld LP iterations (%.1f%%; dives %lld with %lld incumbents,"
               " %lld pumps %lld with %lld)\n",
               tdive_calls_.load(), tdive_found_.load(), tdive_wins_.load(), tdive_limit_.load(), tdive_iters_.load(),
               100.0 * static_cast<double>(tdive_iters_.load()) / all, hsub_calls_.load(), hsub_wins_.load(),
               hsub_iters_.load(), 100.0 * static_cast<double>(hsub_iters_.load()) / all, hsub_dive_iters_.load(),
               hsub_dive_wins_.load(), hsub_fp_runs_.load(), hsub_fp_iters_.load(), hsub_fp_wins_.load());
    }
    if (zi_calls_.load() > 0) {
      log_.log(2, "  ZI rounding: %lld calls, %lld feasible points, %lld improved\n", zi_calls_.load(), zi_found_.load(),
               zi_wins_.load());
    }
    if (alns_ && hsub_calls_.load() > 0) {
      static const char* const kArm[kLns] = {"RINS", "mutation", "local branching", "RENS", "crossover", "proximity",
                                             "graph mutation"};
      log_.log(1, "ALNS sub-MIPs (calls / better point / LP iterations):");
      for (int k = 0; k < kLns; ++k) {
        if (arm_calls_[k].load() > 0) {
          log_.log(1, " %s %lld / %lld / %lld;", kArm[k], arm_calls_[k].load(), arm_found_[k].load(), arm_iters_[k].load());
        }
      }
      log_.log(1, "\n");
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
      if (popped_n_.load() + plunged_n_.load() > 0) {
        log_.log(2, "Node cut time: first LP solves %.2f s (%.0f rows on average), cut-round LP solves %.2f s, separation"
                 " %.2f s in %lld calls (cover %.2f, c-MIR %.2f, clique %.2f, implied bound %.2f, tableau %.2f), pool scans"
                 " %.2f s\n",
                 1e-9 * static_cast<double>(nodelp_ns_.load()),
                 static_cast<double>(lprows_sum_.load()) / static_cast<double>(popped_n_.load() + plunged_n_.load()),
                 1e-9 * static_cast<double>(cutlp_ns_.load()), 1e-9 * static_cast<double>(sep_ns_.load()),
                 sep_calls_.load(), 1e-9 * static_cast<double>(sep_k_ns_[0].load()), 1e-9 * static_cast<double>(sep_k_ns_[1].load()),
                 1e-9 * static_cast<double>(sep_k_ns_[2].load()), 1e-9 * static_cast<double>(sep_k_ns_[3].load()),
                 1e-9 * static_cast<double>(sep_k_ns_[4].load()), 1e-9 * static_cast<double>(scan_ns_.load()));
      }
      if (tmir_rounds_.load() > 0) {
        log_.log(2, "Tableau cuts: %lld separation rounds, %lld successful\n", tmir_rounds_.load(), tmir_won_.load());
      }
    }
    if (orb_fixed_.load() > 0 || orb_pruned_.load() > 0) {
      log_.log(1, "Orbitopes: %lld fixings, %lld nodes pruned\n", orb_fixed_.load(), orb_pruned_.load());
    }
    if (safe_checks_.load() > 0) {
      log_.log(safe_below_.load() > 0 ? 1 : 2, "Safe dual bounds: %lld LPs checked, %lld below the LP objective, %.3f s\n",
               safe_checks_.load(), safe_below_.load(), 1e-9 * static_cast<double>(safe_ns_.load()));
    }
    const DualSimplex::CheckCounts ck = DualSimplex::check_counts();
    if (ck.farkas_checked > checks0_.farkas_checked) {
      const long long failed = ck.farkas_failed - checks0_.farkas_failed;
      const long long retried = ck.farkas_retried_ok - checks0_.farkas_retried_ok;
      log_.log(failed + retried > 0 ? 1 : 2,
               "Farkas proofs: %lld infeasible LPs checked, %lld failed, %lld passed after a refactorization\n",
               ck.farkas_checked - checks0_.farkas_checked, failed, retried);
    }
    if (ck.unscaled_solves > checks0_.unscaled_solves) {
      const long long its = ck.unscaled_iterations - checks0_.unscaled_iterations;
      log_.log(1, "Unscaled dual feasibility: %lld LP solves continued, %lld iterations\n",
               ck.unscaled_solves - checks0_.unscaled_solves, its);
    }
    if (lp_failed_.load() > 0 || lp_unsolved_.load() > 0) {
      log_.log(1, "Node LP failures: %lld, recovered %lld, branched without LP %lld, all-integers-fixed re-solves %lld\n",
               lp_failed_.load(), lp_recovered_.load(), lp_unsolved_.load(), fixed_recovered_.load());
    }
    if (impl_branches_.load() > 0) {
      log_.log(1, "Logical implications: %lld branchings\n", impl_branches_.load());
    }
    if (polish_tried_.load() > 0) {
      log_.log(2, "Fixed-integer re-solves of near-feasible points: %lld, %lld feasible, %lld LP iterations; %lld repeated "
               "integer assignments from the cache\n", polish_tried_.load(), polish_ok_.load(), polish_iters_.load(),
               repair_hits_.load());
    }
    if (lp_polished_.load() > 0) {
      log_.log(2, "Integral node LP points re-solved warm with the integers fixed: %lld\n", lp_polished_.load());
    }
    if (sym_branches_.load() > 0) {
      log_.log(1, "Symmetry: %lld orbital branchings fixed %lld variables, %lld stabilizers\n", sym_branches_.load(),
               sym_fixed_.load(), sym_stabs_.load());
    }
  }

  if (tree_restart_.load()) {
    fill_restart(res, true);
    res.tree_restart = true;
    res.est_restart = est_restart_;
  }
  fj_finish();
  report_progress();
  res.nodes = nodes_.load();
  res.lp_iterations = lp_iters_.load();
  res.dive_iterations = dive_iters_.load();
  res.fp_iterations = fp_iters_.load();
  res.dive_wins = dive_wins_.load();
  res.fp_runs = fp_runs_.load();
  res.fp_wins = fp_wins_.load();
  res.mem_drains = mem_drains_.load();
  res.mem_drain_exits = mem_drain_exits_;
  res.mem_drain_rise = drain_rise_;
  res.mem_purged = mem_purged_;
  const double inc = incumbent_.load();
  res.has_solution = is_finite(inc);
  // MILP_RESTART_IMPROVED_GAP (default 1; 0: beyond 1e-9 relative): an improvement counts only beyond the
  // gap tolerance (h80x6320d, deterministic: copies of the optimum 9e-6 apart counted, so a third restarted
  // root ran RENS for 11 s with the optimum as its cutoff).
  static const bool improved_gap = knob("MILP_RESTART_IMPROVED_GAP", 1) > 0;
  const double start_mag = std::max(1.0, std::abs(inc_start_ + p_.offset));
  const double improved_tol = improved_gap ? std::max(std::max(1e-9, params_.mip_rel_gap) * start_mag, params_.mip_abs_gap)
                                           : 1e-9 * start_mag;
  res.improved = is_finite(inc) && (!is_finite(inc_start_) || inc < inc_start_ - improved_tol);
  if (res.has_solution) {
    res.x = inc_x_;
    res.objective = inc + p_.offset;
    final_polish(res);
  }
  // The root LP bound holds for the whole problem, also when the search stopped before the root node's
  // children reached the pool (cut loop, root node LP or heuristics at the time limit: the trivial bound was
  // reported then, triptim1 -2.039 instead of 22.8681). Above the incumbent only through reductions against it:
  // capped by the incumbent below.
  static const bool root_floor = knob("MILP_ROOT_BOUND_FLOOR", 1) > 0;  // 0: the trivial bound only (as before)
  const double floor_lb = root_floor ? std::max({trivial_lb_, root_lp_lb_, root_bound_.load()}) : trivial_lb_;
  double lb = std::max(std::min(pool_->lower_bound(), inc), floor_lb);
  // An incumbent that met the root LP (or trivial) bound within the gap tolerances ended the search: the nodes it
  // closed never reached the pool, so the bound is the root's, not the incumbent (QCQP fuzz case: -19.17998 reported
  // as its own bound, the optimum -19.18046 lay within the 1e-4 gap below it).
  if (solved_.load()) lb = std::min(lb, floor_lb);
  if (root_unbounded_) {
    res.status = res.has_solution ? Status::Unbounded : Status::InfeasibleOrUnbounded;
    res.bound = -kInf;
    return res;
  }
  if (sol_limit_hit_.load() && stop_reason_ == Status::Unknown) stop_reason_ = Status::Interrupted;
  if (solved_.load()) {
    // An incumbent met the trivial bound: optimal, whatever stopped the search meanwhile (the root LP
    // interrupted by solved_stop_ reports a time limit), and no restart.
    res.status = Status::Optimal;
    res.restart = false;
  } else if (stop_reason_ == Status::TimeLimit || stop_reason_ == Status::NodeLimit || stop_reason_ == Status::MemoryLimit ||
             stop_reason_ == Status::Interrupted || stop_reason_ == Status::IterationLimit) {
    res.status = stop_reason_;
  } else {
    res.status = res.has_solution ? Status::Optimal : Status::Infeasible;
    if (!res.has_solution) lb = kInf;
  }
  // The trivial bound and the root LP bound hold in dropped subtrees too.
  if (const double lost = std::max(lost_bound_, floor_lb); lost < lb) {
    lb = lost;
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

RestartDecision root_restart_decision(const SearchOptions& o, bool deterministic, Index ints, Index newly_fixed,
                                      double root_work, double root_time) {
  RestartDecision d;
  d.need = std::max<Index>(o.restart_min_cols, static_cast<Index>(std::floor(o.restart_frac * static_cast<double>(ints))));
  if (ints <= 0 || newly_fixed < d.need) return d;
  // From the second root restart on: only after a restart whose presolve removed enough, and while the
  // restarted roots stay within the budget (this root's cost again on top of it so far).
  if (o.restarts_done >= 1) {
    if (o.last_restart_removed < o.restart_min_removed) d.refused = "the last restart removed too little";
    else if (deterministic ? 2.0 * root_work > o.restart_work_left : 2.0 * root_time > o.restart_time_left)
      d.refused = "restart budget used";
  }
  d.outcome = d.refused ? RestartDecision::Refused : RestartDecision::Restart;
  return d;
}

MipResult branch_and_bound(const MipProblem& p, const SolverParams& params, const Logger& log, const Timer& timer,
                           const SearchOptions& opts) {
  Search s(p, params, log, timer, opts);
  return s.run();
}

}  // namespace milp
