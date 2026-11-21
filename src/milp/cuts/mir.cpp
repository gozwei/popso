// Complemented mixed-integer rounding (c-MIR) cuts, Marchand & Wolsey (2001).
//
// Base inequalities are aggregations sum_i w_i (a_i x + s_i) = 0 of LP rows
// with their logicals s_i = -a_i x (bounds [-ub_i, -lb_i]), so any multipliers
// give a valid equation and the row sides enter through the logicals' bounds.
// Aggregations start from a problem row and are extended by path aggregation:
// the continuous variable farthest from its simple bounds (e.g. an arc flow,
// even at its VUB x_a = u y_a) is eliminated with another tight row containing
// it, so flow balance rows chain into cut-set like bases. (Tableau-row bases
// were tried as well: they gave similar bounds at a much higher cost.) Paths
// also start from purely continuous rows (single-commodity balances reach the
// integer capacity modules of multi-commodity models through the capacity
// rows) and yield up to kCutsPerPath cuts from successive aggregations. Once no continuous column
// is left to eliminate, the path continues through the integer column farthest from its bounds
// (general integers first): on pure integer models (mzzv11) single rows give almost no c-MIR cut,
// while aggregations linking a general integer's defining row to its capacity rows do.
//
// The base sum a_j x_j <= b is transformed so that every variable is nonnegative:
//   continuous x_j = l_j + x'_j | u_j - x'_j | d + c y + x'_j (VLB) | d + c y - x'_j (VUB)
//   integer    x_j = l_j + z_j  | u_j - z_j  (complemented)
// (logicals are continuous with simple bounds), giving sum a'_j z_j + sum c_j x'_j <= beta.
// For a divisor delta with f0 = frac(beta / delta) the MIR inequality
//   sum delta F(a'_j / delta) z_j + sum_{c_j < 0} c_j / (1 - f0) x'_j <= delta floor(beta / delta),
//   F(t) = floor(t) + max(0, frac(t) - f0) / (1 - f0),
// is valid; it is mapped back to x space. Positive continuous terms are dropped
// (a relaxation since x' >= 0). Variable bound substitution makes this cover
// flow-cover structures (fixed charge networks, lot sizing): e.g. a node
// inflow x_a <= u y_a with demand d yields x_a <= d y_a + outflow.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "milp/core/interrupt.hpp"
#include "milp/core/worker_pool.hpp"
#include "milp/cuts/cut.hpp"
#include "milp/cuts/mcf.hpp"

namespace milp {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

constexpr int kMaxAggr = 12;          // rows added to a path after its start row
constexpr int kCutsPerPath = 4;       // cuts taken from successive aggregations of one path
constexpr double kWorkPerNnz = 4000;  // separation work budget per call, in units of nnz(A)
constexpr double kTightDist = 1e-4;   // rows farther from their bounds are never eliminated through
constexpr size_t kParChunk = 64;      // start rows per chunk of separate_chunked (MILP_MIR_PAR_CHUNK)

enum class BoundKind : int8_t { Lower, Upper, VarLower, VarUpper };

struct ContTerm {
  Index j;
  BoundKind kind;
  double bound;  // simple bound value, or VB constant
  Index y;       // VB variable
  double vcoef;  // VB coefficient
  double c;      // coefficient of x' in the transformed row
  double xs;     // LP value of x'
};

struct IntTerm {
  Index j;
  double a;   // coefficient of x_j (after VB substitution)
  bool comp;  // z = u - x instead of x - l
  double l, u, x;
  double v = 0.0;  // x-space coefficient of x_j in the continuous part (see prepare_norm)
};

class MirSeparator : public Separator {
 public:
  explicit MirSeparator(bool tableau = false, bool flowcover = true, bool mcf = false)
      : tableau_(tableau), use_flowcover_(flowcover), mcf_(mcf) {}
  const char* name() const override { return mcf_ ? "mcf" : tableau_ ? "tmir" : "mir"; }
  void separate(const SepaContext& ctx, std::vector<Cut>& out) override;
  long long work() const override { return work_count_; }

 private:
  // Variables 0..n-1 are structurals, n + i is the logical of LP row i.
  // (Read from the call's copies lo_ / up_ / cls_, set up in separate(): the transformation of every
  // aggregation asks for each of its columns, and the logicals' bounds were calls into the LP.)
  double lower(Index j) const { return own_->lo_[j]; }
  double upper(Index j) const { return own_->up_[j]; }
  // Integer columns, and implied integers (MipProblem::implied_int) with integral bounds: some optimal
  // solution has them integral, so the c-MIR rounding may treat them as integers.
  bool is_int(Index j) const { return own_->cls_[j] == kClsInt; }
  bool is_int_column(Index j) const {
    const MipProblem& p = ctx_->p;
    if (p.is_int[j]) return true;
    return p.is_implied_int(j) && integral_bound(p.col_lb[j]) && integral_bound(p.col_ub[j]);
  }
  static bool integral_bound(double b) { return !is_finite(b) || b == std::round(b); }

  // Tries a c-MIR cut from the base inequality sum agg_j x_j <= b.
  bool try_cut(double b, Cut* cut);
  // Lifted knapsack cover from the aggregation sum agg_j x_j <= 0 (SCIP's knapsackcover on
  // aggregated rows): continuous terms are replaced by the (variable) bound that lets them be
  // dropped, non-binary integers bounded out, binaries complemented to positive weights.
  bool try_knapsack(Cut* cut, double b0 = 0.0);
  // Lifted simple generalized flow cover (Gu, Nemhauser & Savelsbergh 1999; as SCIP's cuts.c) from
  // the aggregation sum agg_j x_j <= 0: single-node flow relaxation over binary arc variables
  // (variable bounds of continuous columns, binaries of the row; other columns with simple bounds
  // and an arc fixed to 1), a flow cover by a greedy knapsack, sequence independent lifting.
  bool try_flowcover(Cut* cut);
  // Distance of continuous x_j from its closest bound (incl. variable bounds) and that bound.
  // sign > 0 (< 0): only simple bounds and VUBs (VLBs), sign 0: all.
  // Columns with several variable bounds keep the result per call (bd_slot_ / bd_cache_): it depends on
  // the LP point and the bounds only, and every aggregation of a path transforms the column again (swath3:
  // the variable bound lists were 6% of the tree time).
  double bound_distance(Index j, ContTerm* t, int sign = 0);
  double bound_distance_raw(Index j, ContTerm* t, int sign) const;
  // Builds cont_/ints_ for the base sum agg_j x_j <= b; returns beta's b part (nan on failure).
  // Mode 0: closest bound; mode 1: variable bounds only where the x' term is kept (c < 0).
  double transform(double b, int mode, bool* vb_skipped);
  // Best c-MIR cut over divisors and complementations of the current transformation.
  double best_cut(double b, Cut* cut);
  // Efficacy of the cut for divisor delta; builds it into `cut` unless null (then only the efficacy,
  // in O(#terms) using the data from prepare_norm).
  double evaluate(double delta, double beta, Cut* cut);
  // evaluate(delta, beta, nullptr) with the integer terms' quotients a / delta (floor and fraction, for both
  // complementations) cached by cache_quotients(delta): the complementation trials of best_cut keep delta
  // and change only beta, and the quotients were most of their cost (h80x6320d: ~700 terms per trial).
  // The same arithmetic, so the same efficacies.
  void cache_quotients(double delta);
  double evaluate_cached(double delta, double beta);
  // The x-space continuous part of every cut of the current transformation is V / (1 - f0) with
  // V = sum_{c_j < 0} c_j x'_j(x) (fixed); stores ||V||^2 and V on the integer columns.
  void prepare_norm();
  // Adds mult * (a_i x + s_i) to the aggregation.
  void add_row(Index i, double mult);
  void clear_agg();
  // Appends v * x_j to the cut, expanding logicals into structurals (rhs unchanged).
  void emit(Index j, double v, Cut* cut) const;
  // Path aggregation from row i0 (one side); appends at most one cut.
  void aggregate_path(Index i0, double sign, std::vector<Cut>& out);
  // The elimination candidates of column j (tight_beg_ / tight_end_ / tight_k_).
  void build_tight(Index j);
  // The start rows in fixed chunks of kParChunk (MILP_MIR_PAR=1, root cut loops): the starts of a chunk
  // run independently, on SepaContext::threads threads (this separator and helpers with their own scratch
  // state), each with the work at the chunk's start plus its own for the integer elimination budget; their
  // cuts are merged in start order and the work budget is checked between chunks. The cuts depend on the
  // chunk size (MILP_MIR_PAR_CHUNK), not on the thread count.
  void separate_chunked(long long budget, Index max_starts, std::vector<Cut>& out);
  // Tableau rows of fractional basic integers as base equations (HiGHS' tableau separator): the row
  // of B^-1 [A I] is an exact identity over structurals and logicals, so with global bounds
  // the c-MIR / lifted cover cuts from it are globally valid even at a node.
  void separate_tableau(std::vector<Cut>& out);
  // Network cut-sets (as SCIP's mcf separator): for node sets S of the detected multi-commodity flow
  // network, the flow conservation rows of S (commodities with net demand into / supply out of S at the
  // LP point) plus the capacity rows of the arcs whose flow they leave with a negative coefficient; c-MIR
  // and the aggregated knapsack cover on that base. Sets: single nodes and the components formed while
  // contracting arcs in the order of decreasing capacity slack, then increasing capacity dual (so that
  // the cut arcs are the tight, expensive ones).
  void separate_mcf(std::vector<Cut>& out);
  bool tableau_ = false, use_flowcover_ = true, mcf_ = false;
  McfNetwork net_;
  bool net_built_ = false;
  std::vector<double> tab_;
  std::vector<Index> tnz_;

  const SepaContext* ctx_ = nullptr;
  long long work_count_ = 0;
  long long int_budget_ = 0;  // integer path steps only while work_count_ is below this (per call)
  double int_work_per_nnz_ = [] { const char* v = std::getenv("MILP_MIR_INTAGG_WORK"); return v ? std::atof(v) : 20.0; }();
  Index n_ = 0;
  std::vector<double> agg_, icoef_, dense_;
  std::vector<Index> nz_, inz_, used_;
  std::vector<char> mark_, imark_, row_int_;  // row_int_: problem row has an integer column
  std::vector<char> row_used_;                // problem row is in the current path
  std::vector<std::pair<double, Index>> starts_;  // start rows of the paths, best first
  std::vector<double> row_dist_;              // distance of a problem row's logical from its bounds
  std::vector<double> row_dual_;              // |LP dual| of a problem row, relative to ||c|| (path row choice)
  bool dual_choice_ = [] { const char* v = std::getenv("MILP_MIR_DUALROW"); return !(v && *v == '0'); }();
  // Per column (built on first use in a call): positions k in its column of the problem rows whose
  // logical is within kTightDist of a bound, the only candidates for path elimination (app1-1 has
  // continuous columns in 1225 rows, scanned at every aggregation step otherwise).
  // A candidate carries what the row choice reads (its row, the model's coefficient, the row's distance, dual
  // score and integer flag), so the choice walks one array; rows it would always skip (longer than the limit,
  // variable bound rows) are left out when the list is built.
  struct TightRow {
    Index r;
    double val, dist, ds;
    char has_int;
  };
  std::vector<Index> tight_beg_, tight_end_;
  std::vector<TightRow> tight_;
  struct BdEntry {
    ContTerm t[3];
    double d[3];
    const VarBound* kvb[2];
    double kbest[2];
    unsigned char have = 0;  // bits 0-2: t / d by sign + 1; bits 3-4: the knapsack's choice by side
  };
  // Per variable of the call: integer (see is_int_column), continuous with at least kBdCacheMin variable
  // bounds (bound distances kept in bd_cache_), other continuous (logicals too).
  enum : char { kClsCont = 0, kClsInt = 1, kClsVb = 2 };
  std::vector<char> cls_;
  std::vector<double> lo_, up_;
  std::vector<Index> bd_slot_;  // per column: its entry of bd_cache_ in this call, or -1
  std::vector<BdEntry> bd_cache_;
  static constexpr size_t kBdCacheMin = 1;  // variable bounds of a column from which the result is kept
  std::vector<ContTerm> cont_;
  std::vector<IntTerm> ints_;
  std::vector<double> deltas_;
  struct Quot {
    double fl[2], fq[2];  // floor and fraction of a / delta, [1]: complemented (-a / delta)
  };
  std::vector<Quot> quot_;
  std::vector<double> kw_, kz_, kcoef_;
  std::vector<Index> kj_;
  std::vector<char> kcomp_;
  bool knapsack_ = true, flowcover_ = true;
  double last_eff_ = -1.0;  // efficacy of the last c-MIR / flow cover attempt
  // A flow cover replaces the c-MIR cut of an aggregation only if its efficacy is fc_win_ times higher
  // (MILP_FC_WIN; default never: flow covers only where c-MIR finds nothing, see aggregate_path).
  double fc_win_ = 1e30;
  // Path aggregation eliminates integer columns when no continuous one is left (MILP_MIR_INTAGG=0: off).
  bool int_elim_ = [] { const char* v = std::getenv("MILP_MIR_INTAGG"); return !(v && *v == '0'); }();
  double int_elim_min_ = [] { const char* v = std::getenv("MILP_MIR_INTAGG_MIN"); return v ? std::atof(v) : 1e-6; }();
  int int_elim_max_ = [] { const char* v = std::getenv("MILP_MIR_INTAGG_MAX"); return v ? std::atoi(v) : 1000; }();
  // Only general integers are eliminated (MILP_MIR_INTAGG_GENERAL=0: binaries too; rococoC10 / ns1830653 roots
  // gain from binaries, but mas76 / piperout-08 trees were 1.2-1.7x slower with the extra cuts).
  bool int_elim_general_ = [] { const char* v = std::getenv("MILP_MIR_INTAGG_GENERAL"); return !(v && *v == '0'); }();
  struct SnfVar {
    int sign;          // +1: N1 (inflow), -1: N2 (outflow)
    double u;          // capacity: 0 <= y' <= u x
    double xb;         // LP value of the arc binary (1 without one)
    Index bin;         // arc binary (-1: fixed to 1)
    Index cont;        // non-binary column (or logical) of y' (-1: none)
    double cb, cc, k;  // y' = cc x_cont + cb x_bin + k
    int status;        // +1 in the cover, -1 not
  };
  std::vector<SnfVar> snf_;
  std::vector<Index> fc_items_, fc_nonbin_, fc_used_list_;
  std::vector<char> fc_used_;
  std::vector<double> fc_m_, fc_M_;
  std::vector<double> row_minact_, row_maxact_;  // activity range of the problem rows (global bounds)
  Cut cand_, vbuf_;
  double vnorm2_ = 0.0;
  // separate_chunked (opt-in until an A/B: the cuts change with the chunk granularity).
  bool par_ = [] { const char* v = std::getenv("MILP_MIR_PAR"); return v && *v && *v != '0'; }();
  // The separator whose setup (row_dist_, row_dual_, row_int_, tight_*, row_minact_ / row_maxact_) the paths
  // read: this one, or for a helper of separate_chunked the one it works for.
  const MirSeparator* own_ = this;
  std::vector<std::unique_ptr<MirSeparator>> helpers_;
  std::unique_ptr<WorkerPool> pool_;
  std::vector<std::vector<Cut>> chunk_cuts_;  // per start of the chunk
  std::vector<long long> chunk_work_;
};

double MirSeparator::bound_distance(Index j, ContTerm* t, int sign) {
  if (own_->cls_[j] != kClsVb) return bound_distance_raw(j, t, sign);
  if (bd_slot_[j] < 0) bd_slot_[j] = static_cast<Index>(bd_cache_.size()), bd_cache_.emplace_back();
  BdEntry& e = bd_cache_[bd_slot_[j]];
  const int s = sign < 0 ? 0 : sign > 0 ? 2 : 1;
  if (!(e.have >> s & 1)) {
    e.d[s] = bound_distance_raw(j, &e.t[s], sign);
    if (!is_finite(e.d[s])) e.t[s] = ContTerm{j, BoundKind::Lower, 0.0, -1, 0.0, 0.0, 0.0};  // never read
    e.have = static_cast<unsigned char>(e.have | 1 << s);
  }
  *t = e.t[s];
  return e.d[s];
}

double MirSeparator::bound_distance_raw(Index j, ContTerm* t, int sign) const {
  const double* xv = ctx_->x;
  const double x = xv[j];
  double best = kInf, best_key = kInf;
  auto consider = [&](double dist, BoundKind kind, double bound, Index y, double vc) {
    // Prefer variable bounds on near ties: they carry the integer structure.
    const bool vb = kind == BoundKind::VarLower || kind == BoundKind::VarUpper;
    const double key = vb ? dist - 1e-6 : dist;
    if (key < best_key) {
      best_key = key;
      best = std::max(dist, 0.0);
      *t = ContTerm{j, kind, bound, y, vc, 0.0, 0.0};
    }
  };
  if (j < n_ && sign <= 0) {
    for (const VarBound& vb : ctx_->data.vlb[j]) {
      consider(x - (vb.constant + vb.coef * xv[vb.y]), BoundKind::VarLower, vb.constant, vb.y, vb.coef);
    }
  }
  if (j < n_ && sign >= 0) {
    for (const VarBound& vb : ctx_->data.vub[j]) {
      consider(vb.constant + vb.coef * xv[vb.y] - x, BoundKind::VarUpper, vb.constant, vb.y, vb.coef);
    }
  }
  const double l = lower(j), u = upper(j);
  if (is_finite(l)) consider(x - l, BoundKind::Lower, l, -1, 0.0);
  if (is_finite(u)) consider(u - x, BoundKind::Upper, u, -1, 0.0);
  return best;
}

void MirSeparator::add_row(Index i, double mult) {
  const SparseMatrix& R = ctx_->lp.scaled_row_matrix();  // entry * us: the model's coefficient, exactly
  const double us = ctx_->lp.row_unscale(i);
  auto add = [&](Index j, double v) {
    if (!mark_[j]) mark_[j] = 1, nz_.push_back(j);
    agg_[j] += v;
  };
  for (Index k = R.begin(i); k < R.end(i); ++k) add(R.index[k], mult * (R.value[k] * us));
  add(n_ + i, mult);
  work_count_ += R.end(i) - R.begin(i);
}

void MirSeparator::clear_agg() {
  for (Index j : nz_) agg_[j] = 0.0, mark_[j] = 0;
  nz_.clear();
}

void MirSeparator::emit(Index j, double v, Cut* cut) const {
  if (j < n_) {
    cut->idx.push_back(j), cut->val.push_back(v);
    return;
  }
  const SparseMatrix& R = ctx_->lp.scaled_row_matrix();
  const Index i = j - n_;
  const double us = ctx_->lp.row_unscale(i);
  for (Index k = R.begin(i); k < R.end(i); ++k) cut->idx.push_back(R.index[k]), cut->val.push_back(-v * (R.value[k] * us));
}

// MIR cut for divisor delta over the current cont_/ints_ transformation, written
// to `cut` (mapped back to x space). Returns its efficacy in x space (-1 if not
// applicable): after variable bound substitution the transformed-space norm is
// meaningless (with x <= 3950 y the cut x <= 2 y + ... has terms 3950 y - 3950 y').
void MirSeparator::cache_quotients(double delta) {
  quot_.resize(ints_.size());
  for (size_t k = 0; k < ints_.size(); ++k) {
    for (int c = 0; c < 2; ++c) {
      const double q = (c ? -ints_[k].a : ints_[k].a) / delta;
      quot_[k].fl[c] = std::floor(q);
      quot_[k].fq[c] = q - quot_[k].fl[c];
    }
  }
}

double MirSeparator::evaluate_cached(double delta, double beta) {
  const double bd = beta / delta;
  if (std::abs(bd) > 1e8) return -1.0;
  const double f0 = bd - std::floor(bd);
  if (f0 < 1e-4 || f0 > 0.999) return -1.0;
  const double rhs = delta * std::floor(bd);
  double act = 0.0;
  const double s = 1.0 / (1.0 - f0);
  double norm = s * s * vnorm2_;
  for (size_t k = 0; k < ints_.size(); ++k) {
    const IntTerm& t = ints_[k];
    const int c = t.comp ? 1 : 0;
    const double g = delta * (quot_[k].fl[c] + std::max(0.0, quot_[k].fq[c] - f0) / (1.0 - f0));
    if (g == 0.0) continue;
    act += g * (t.comp ? t.u - t.x : t.x - t.l);
    const double gx = t.comp ? -g : g, sv = s * t.v;
    norm += (gx + sv) * (gx + sv) - sv * sv;
  }
  for (const ContTerm& t : cont_) {
    if (t.c < 0) act += t.c * s * t.xs;
  }
  work_count_ += static_cast<long long>(ints_.size() + cont_.size());
  if (norm <= 1e-12) return -1.0;
  return (act - rhs) / std::sqrt(norm);
}

double MirSeparator::evaluate(double delta, double beta, Cut* cut) {
  const double bd = beta / delta;
  if (std::abs(bd) > 1e8) return -1.0;
  const double f0 = bd - std::floor(bd);
  if (f0 < 1e-4 || f0 > 0.999) return -1.0;
  const double rhs = delta * std::floor(bd);
  double act = 0.0;
  if (!cut) {
    const double s = 1.0 / (1.0 - f0);
    double norm = s * s * vnorm2_;
    for (const IntTerm& t : ints_) {
      const double a = t.comp ? -t.a : t.a;
      const double q = a / delta;
      const double fq = q - std::floor(q);
      const double g = delta * (std::floor(q) + std::max(0.0, fq - f0) / (1.0 - f0));
      if (g == 0.0) continue;
      act += g * (t.comp ? t.u - t.x : t.x - t.l);
      const double gx = t.comp ? -g : g, sv = s * t.v;
      norm += (gx + sv) * (gx + sv) - sv * sv;
    }
    for (const ContTerm& t : cont_) {
      if (t.c < 0) act += t.c * s * t.xs;
    }
    work_count_ += static_cast<long long>(ints_.size() + cont_.size());
    if (norm <= 1e-12) return -1.0;
    return (act - rhs) / std::sqrt(norm);
  }
  cut->idx.clear();
  cut->val.clear();
  cut->rhs = rhs;
  for (const IntTerm& t : ints_) {
    const double a = t.comp ? -t.a : t.a;
    const double q = a / delta;
    const double fq = q - std::floor(q);
    const double g = delta * (std::floor(q) + std::max(0.0, fq - f0) / (1.0 - f0));
    if (g == 0.0) continue;
    act += g * (t.comp ? t.u - t.x : t.x - t.l);
    if (t.comp) {  // z = u - x
      cut->idx.push_back(t.j), cut->val.push_back(-g);
      cut->rhs -= g * t.u;
    } else {  // z = x - l
      cut->idx.push_back(t.j), cut->val.push_back(g);
      cut->rhs += g * t.l;
    }
  }
  for (const ContTerm& t : cont_) {
    if (t.c >= 0) continue;
    const double h = t.c / (1.0 - f0);
    act += h * t.xs;
    switch (t.kind) {
      case BoundKind::Lower:  // x' = x - l
        emit(t.j, h, cut);
        cut->rhs += h * t.bound;
        break;
      case BoundKind::Upper:  // x' = u - x
        emit(t.j, -h, cut);
        cut->rhs -= h * t.bound;
        break;
      case BoundKind::VarLower:  // x' = x - d - c y
        emit(t.j, h, cut);
        cut->idx.push_back(t.y), cut->val.push_back(-h * t.vcoef);
        cut->rhs += h * t.bound;
        break;
      case BoundKind::VarUpper:  // x' = d + c y - x
        emit(t.j, -h, cut);
        cut->idx.push_back(t.y), cut->val.push_back(h * t.vcoef);
        cut->rhs -= h * t.bound;
        break;
    }
  }
  work_count_ += static_cast<long long>(cut->idx.size()) + static_cast<long long>(ints_.size() + cont_.size());
  // Norm of the merged x-space coefficients.
  for (size_t k = 0; k < cut->idx.size(); ++k) dense_[cut->idx[k]] += cut->val[k];
  double norm = 0.0;
  for (Index j : cut->idx) norm += dense_[j] * dense_[j], dense_[j] = 0.0;
  if (norm <= 1e-12) return -1.0;
  return (act - rhs) / std::sqrt(norm);
}

void MirSeparator::prepare_norm() {
  Cut& v = vbuf_;
  v.idx.clear();
  v.val.clear();
  for (const ContTerm& t : cont_) {
    if (t.c >= 0) continue;
    switch (t.kind) {
      case BoundKind::Lower:
        emit(t.j, t.c, &v);
        break;
      case BoundKind::Upper:
        emit(t.j, -t.c, &v);
        break;
      case BoundKind::VarLower:
        emit(t.j, t.c, &v);
        v.idx.push_back(t.y), v.val.push_back(-t.c * t.vcoef);
        break;
      case BoundKind::VarUpper:
        emit(t.j, -t.c, &v);
        v.idx.push_back(t.y), v.val.push_back(t.c * t.vcoef);
        break;
    }
  }
  work_count_ += static_cast<long long>(v.idx.size());
  for (size_t k = 0; k < v.idx.size(); ++k) dense_[v.idx[k]] += v.val[k];
  for (IntTerm& t : ints_) t.v = dense_[t.j];
  vnorm2_ = 0.0;
  for (Index j : v.idx) vnorm2_ += dense_[j] * dense_[j], dense_[j] = 0.0;
}

double MirSeparator::transform(double b, int mode, bool* vb_skipped) {
  const double* xv = ctx_->x;
  cont_.clear();
  ints_.clear();
  for (Index j : inz_) icoef_[j] = 0.0, imark_[j] = 0;
  inz_.clear();
  auto add_int = [&](Index j, double v) {
    if (!imark_[j]) imark_[j] = 1, inz_.push_back(j);
    icoef_[j] += v;
  };
  // Continuous variables: substitute the closest (variable) bound.
  work_count_ += static_cast<long long>(nz_.size());
  for (Index j : nz_) {
    const double a = agg_[j];
    if (a == 0.0) continue;
    if (is_int(j)) {
      add_int(j, a);
      continue;
    }
    if (lower(j) == upper(j)) {
      b -= a * lower(j);
      continue;
    }
    ContTerm t;
    if (!is_finite(bound_distance(j, &t, mode == 0 ? 0 : (a > 0 ? 1 : -1)))) return kNaN;
    if (mode == 0 && j < n_ && (t.kind == BoundKind::VarLower ? a > 0 : t.kind == BoundKind::VarUpper && a < 0)) {
      *vb_skipped = true;  // mode 1 would choose differently
    }
    const double x = xv[j];
    switch (t.kind) {
      case BoundKind::Lower:
        t.c = a, t.xs = x - t.bound, b -= a * t.bound;
        break;
      case BoundKind::Upper:
        t.c = -a, t.xs = t.bound - x, b -= a * t.bound;
        break;
      case BoundKind::VarLower:  // x = d + c y + x'
        t.c = a, t.xs = x - t.bound - t.vcoef * xv[t.y], b -= a * t.bound;
        add_int(t.y, a * t.vcoef);
        break;
      case BoundKind::VarUpper:  // x = d + c y - x'
        t.c = -a, t.xs = t.bound + t.vcoef * xv[t.y] - x, b -= a * t.bound;
        add_int(t.y, a * t.vcoef);
        break;
    }
    t.xs = std::max(t.xs, 0.0);
    cont_.push_back(t);
  }
  // Integer variables: complement towards the closest bound.
  const MipProblem& p = ctx_->p;
  for (Index j : inz_) {
    const double a = icoef_[j];
    if (a == 0.0) continue;
    const double l = p.col_lb[j], u = p.col_ub[j], x = xv[j];
    if (l == u) {
      b -= a * l;
      continue;
    }
    if (!is_finite(l) && !is_finite(u)) return kNaN;
    const bool comp = !is_finite(l) || (is_finite(u) && u - x < x - l);
    ints_.push_back({j, a, comp, l, u, x});
  }
  if (ints_.empty() || !std::isfinite(b)) return kNaN;
  return b;
}

double MirSeparator::best_cut(double b, Cut* cut) {
  auto beta_of = [&]() {
    double beta = b;
    for (const IntTerm& t : ints_) beta -= t.a * (t.comp ? t.u : t.l);
    return beta;
  };
  // Divisor candidates: coefficients of integer variables strictly inside their bounds, up to 8
  // (MILP_MIR_MAXDELTA; 64 gave binkar10_1 +5 and rococoC10-001000 +4 points of root gap, but miplib_dev
  // at 60 s / 2 threads 1.06-1.11x slower vs 1.02x with 8, gap_large_1 4 -> 8 s).
  static const size_t max_deltas = [] {
    const char* v = std::getenv("MILP_MIR_MAXDELTA");
    return v ? static_cast<size_t>(std::max(1, std::atoi(v))) : size_t{8};
  }();
  std::vector<double>& deltas = deltas_;
  deltas.clear();
  double amax = 0.0;
  for (const IntTerm& t : ints_) amax = std::max(amax, std::abs(t.a));
  for (const IntTerm& t : ints_) {
    if (t.x <= t.l + 1e-6 || t.x >= t.u - 1e-6) continue;
    const double d = std::abs(t.a);
    if (d < 1e-6 * amax || d < 1e-9) continue;
    bool dup = false;
    for (double e : deltas) dup = dup || std::abs(e - d) <= 1e-9 * d;
    work_count_ += static_cast<long long>(deltas.size());
    if (!dup) deltas.push_back(d);
    if (deltas.size() >= max_deltas) break;
  }
  if (deltas.empty()) return -1.0;
  prepare_norm();
  double beta = beta_of();
  double best_eff = -1.0, best_delta = 0.0;
  for (double d : deltas) {
    const double e = evaluate(d, beta, nullptr);
    if (e > best_eff + 1e-9) best_eff = e, best_delta = d;
  }
  if (best_eff <= 0.0) return -1.0;
  for (double div : {2.0, 4.0, 8.0}) {
    const double d = best_delta / div;
    const double e = evaluate(d, beta, nullptr);
    if (e > best_eff + 1e-9) best_eff = e, best_delta = d;
  }
  // Try complementing integers with LP values inside their bounds (beta updated incrementally:
  // recomputing it per candidate made this loop quadratic outside the work budget).
  static const bool qcache = [] { const char* v = std::getenv("MILP_MIR_QCACHE"); return !(v && *v == '0'); }();
  int trials = 0;  // the cache costs about two plain evaluations
  for (const IntTerm& t : ints_) {
    trials += !(t.x <= t.l + 1e-6 || t.x >= t.u - 1e-6 || !is_finite(t.l) || !is_finite(t.u));
  }
  const bool cached = qcache && trials >= 3;
  if (cached) cache_quotients(best_delta);
  for (IntTerm& t : ints_) {
    if (t.x <= t.l + 1e-6 || t.x >= t.u - 1e-6 || !is_finite(t.l) || !is_finite(t.u)) continue;
    t.comp = !t.comp;
    const double beta2 = beta - t.a * (t.comp ? t.u - t.l : t.l - t.u);
    const double e = cached ? evaluate_cached(best_delta, beta2) : evaluate(best_delta, beta2, nullptr);
    if (e > best_eff + 1e-9) {
      best_eff = e;
      beta = beta2;
    } else {
      t.comp = !t.comp;
    }
  }
  if (best_eff < 1e-4) return -1.0;
  return evaluate(best_delta, beta_of(), cut);
}

bool MirSeparator::try_cut(double b, Cut* cut) {
  double best = -1.0;
  bool vb_skipped = false;
  for (int mode = 0; mode < 2; ++mode) {
    if (mode == 1 && !vb_skipped) break;
    const double bt = transform(b, mode, &vb_skipped);
    if (std::isnan(bt)) continue;
    const double e = best_cut(bt, &cand_);
    if (e > best) best = e, std::swap(*cut, cand_);
  }
  last_eff_ = best;
  if (best < 1e-4) return false;
  cut->origin = "mir";
  return true;
}

bool MirSeparator::try_knapsack(Cut* cut, double b0) {
  const double* xv = ctx_->x;
  const MipProblem& p = ctx_->p;
  for (Index j : inz_) icoef_[j] = 0.0, imark_[j] = 0;
  inz_.clear();
  auto add_int = [&](Index j, double v) {
    if (!imark_[j]) imark_[j] = 1, inz_.push_back(j);
    icoef_[j] += v;
  };
  double b = b0;
  work_count_ += static_cast<long long>(nz_.size());
  for (Index j : nz_) {
    const double a = agg_[j];
    if (a == 0.0) continue;
    const double l = lower(j), u = upper(j);
    if (l == u) {
      b -= a * l;
      continue;
    }
    if (is_int(j)) {
      add_int(j, a);
      continue;
    }
    // a > 0: x >= l or x >= d + c y (VLB); a < 0: x <= u or x <= d + c y (VUB). The remaining
    // a x' term is nonnegative and dropped. Closest bound at the LP point (variable bounds on ties).
    const double x = xv[j];
    double best = kInf;
    const VarBound* vb_best = nullptr;
    // Kept per call for columns with several variable bounds, as bound_distance does.
    BdEntry* e = nullptr;
    const int side = a > 0 ? 0 : 1;
    if (own_->cls_[j] == kClsVb) {
      if (bd_slot_[j] < 0) bd_slot_[j] = static_cast<Index>(bd_cache_.size()), bd_cache_.emplace_back();
      e = &bd_cache_[bd_slot_[j]];
    }
    if (e && (e->have >> (3 + side) & 1)) {
      best = e->kbest[side], vb_best = e->kvb[side];
    } else {
      if (a > 0) {
        if (is_finite(l)) best = x - l;
        if (j < n_) {
          for (const VarBound& vb : ctx_->data.vlb[j]) {
            const double d = x - (vb.constant + vb.coef * xv[vb.y]) - 1e-6;
            if (d < best) best = d, vb_best = &vb;
          }
        }
      } else {
        if (is_finite(u)) best = u - x;
        if (j < n_) {
          for (const VarBound& vb : ctx_->data.vub[j]) {
            const double d = vb.constant + vb.coef * xv[vb.y] - x - 1e-6;
            if (d < best) best = d, vb_best = &vb;
          }
        }
      }
      if (e) e->kbest[side] = best, e->kvb[side] = vb_best, e->have = static_cast<unsigned char>(e->have | 1 << (3 + side));
    }
    if (!is_finite(best)) return false;
    if (vb_best) {
      b -= a * vb_best->constant;
      add_int(vb_best->y, a * vb_best->coef);
    } else {
      b -= a * (a > 0 ? l : u);
    }
  }
  kw_.clear(), kz_.clear(), kj_.clear(), kcomp_.clear();
  for (Index j : inz_) {
    const double a = icoef_[j];
    if (std::abs(a) <= 1e-9) continue;
    const double l = p.col_lb[j], u = p.col_ub[j];
    if (l == u) {
      b -= a * l;
      continue;
    }
    if (l == 0.0 && u == 1.0) {
      if (a > 0) {
        kw_.push_back(a), kz_.push_back(xv[j]), kj_.push_back(j), kcomp_.push_back(0);
      } else {  // a x = a - a (1 - x)
        b -= a;
        kw_.push_back(-a), kz_.push_back(1.0 - xv[j]), kj_.push_back(j), kcomp_.push_back(1);
      }
      continue;
    }
    const double bound = a > 0 ? l : u;  // a x >= a * bound
    if (!is_finite(bound)) return false;
    b -= a * bound;
  }
  if (kw_.size() < 2 || !std::isfinite(b) || b < 0.0) return false;
  double rhs = 0.0;
  if (!knapsack_cover_cut(kw_, kz_, b, kcoef_, rhs)) return false;
  cut->idx.clear(), cut->val.clear();
  cut->rhs = rhs;
  double act = 0.0, norm = 0.0;
  for (size_t k = 0; k < kw_.size(); ++k) {
    const double c = kcoef_[k];
    if (c == 0.0) continue;
    act += c * kz_[k], norm += c * c;
    if (kcomp_[k]) {  // c (1 - x)
      cut->idx.push_back(kj_[k]), cut->val.push_back(-c);
      cut->rhs -= c;
    } else {
      cut->idx.push_back(kj_[k]), cut->val.push_back(c);
    }
  }
  work_count_ += static_cast<long long>(kw_.size());
  if (norm <= 0.0 || (act - rhs) / std::sqrt(norm) < 1e-4) return false;
  cut->origin = "aggcover";
  return true;
}

bool MirSeparator::try_flowcover(Cut* cut) {
  const double* xv = ctx_->x;
  const MipProblem& p = ctx_->p;
  constexpr double kMaxBound = 1e10, kMaxVbCoef = 1e5, kFeas = 1e-6, kEps = 1e-9;
  auto binary = [&](Index j) { return j < n_ && p.is_int[j] && p.col_lb[j] == 0.0 && p.col_ub[j] == 1.0; };
  // Binaries of the aggregation (coefficients in icoef_) and the other columns / logicals.
  for (Index j : inz_) icoef_[j] = 0.0, imark_[j] = 0;
  inz_.clear();
  fc_nonbin_.clear();
  double b = 0.0;
  work_count_ += static_cast<long long>(nz_.size());
  for (Index j : nz_) {
    const double a = agg_[j];
    if (a == 0.0) continue;
    const double l = lower(j), u = upper(j);
    if (l == u) {
      b -= a * l;
      continue;
    }
    if (binary(j)) {
      if (!imark_[j]) imark_[j] = 1, inz_.push_back(j);
      icoef_[j] += a;
      continue;
    }
    fc_nonbin_.push_back(j);
  }
  if (fc_used_.size() < static_cast<size_t>(n_)) fc_used_.assign(n_, 0);
  fc_used_list_.clear();
  auto fail = [&]() {
    for (Index y : fc_used_list_) fc_used_[y] = 0;
    fc_used_list_.clear();
    return false;
  };
  snf_.clear();
  // Non-binary columns: y' from the closest (variable) bound; a variable bound's binary is the arc.
  for (Index j : fc_nonbin_) {
    const double a = agg_[j], x = xv[j];
    double slb = lower(j), sub = upper(j);
    if (j >= n_ && j - n_ < ctx_->num_problem_rows) {
      // Logical s = -a_i x of a one-sided row: bounded by the row's activity range (SCIP bounds
      // the slack of the aggregated rows the same way).
      const Index i = j - n_;
      slb = std::max(slb, -own_->row_maxact_[i]);
      sub = std::min(sub, -own_->row_minact_[i]);
    }
    if (slb <= -kMaxBound) slb = -kInf;
    if (sub >= kMaxBound) sub = kInf;
    if (!is_finite(slb) && !is_finite(sub)) return fail();
    const bool cont = j < n_ && !p.is_int[j];
    const double sgn = a >= 0 ? 1.0 : -1.0;
    double blb = -kInf, bub = kInf;
    const VarBound* vlb = nullptr;
    const VarBound* vub = nullptr;
    if (is_finite(sub)) {
      blb = slb;
      if (cont) {
        double best = -kInf;
        for (const VarBound& vb : ctx_->data.vlb[j]) {
          // x >= d + c y: needs u <= d and the capacity terms below nonnegative.
          if (!binary(vb.y) || fc_used_[vb.y] || std::abs(vb.coef) > kMaxVbCoef || sub > vb.constant) continue;
          const double ay = imark_[vb.y] ? icoef_[vb.y] : 0.0;
          if (sgn * (a * vb.coef + ay) > 0.0 || sgn * (a * (sub - vb.constant) + ay) > 0.0) continue;
          const double v = vb.coef * xv[vb.y] + vb.constant;
          if (v > best) best = v, vlb = &vb;
        }
        if (vlb && best > blb + kEps * std::max(1.0, std::abs(best))) {
          blb = best;
        } else {
          vlb = nullptr;
        }
      }
    }
    if (is_finite(slb)) {
      bub = sub;
      if (cont) {
        double best = kInf;
        for (const VarBound& vb : ctx_->data.vub[j]) {
          // x <= d + c y: needs l >= d.
          if (!binary(vb.y) || fc_used_[vb.y] || std::abs(vb.coef) > kMaxVbCoef || slb < vb.constant) continue;
          const double ay = imark_[vb.y] ? icoef_[vb.y] : 0.0;
          if (sgn * (a * vb.coef + ay) < 0.0 || sgn * (a * (slb - vb.constant) + ay) < 0.0) continue;
          const double v = vb.coef * xv[vb.y] + vb.constant;
          if (v < best) best = v, vub = &vb;
        }
        if (vub && best < bub - kEps * std::max(1.0, std::abs(best))) {
          bub = best;
        } else {
          vub = nullptr;
        }
      }
    }
    if (!is_finite(blb) && !is_finite(bub)) {
      // One-sided column without a variable bound: a (x - l) >= 0 (a > 0) or a (x - u) >= 0 (a < 0)
      // is dropped from the left-hand side (a relaxation); the other direction has no flow model.
      if (a > 0 && is_finite(slb)) {
        b -= a * slb;
        continue;
      }
      if (a < 0 && is_finite(sub)) {
        b -= a * sub;
        continue;
      }
      return fail();
    }
    bool use_lower;
    if (!is_finite(bub)) {
      use_lower = true;
    } else if (!is_finite(blb)) {
      use_lower = false;
    } else {
      const double mid = 0.5 * (blb + bub);
      if (std::abs(x - mid) <= kEps * std::max(1.0, std::abs(mid)) && (vlb || vub)) {
        use_lower = vlb != nullptr;
      } else {
        use_lower = x <= mid;
      }
    }
    SnfVar s{};
    s.bin = -1, s.cont = j, s.xb = 1.0, s.cb = 0.0, s.status = 0;
    const bool pos = a >= 0;
    if (use_lower && !vlb) {  // y' = a (u - x) (a > 0: N2) or a (x - u) (a < 0: N1); capacity |a| (u - l)
      const double val = a * (sub - slb);
      s.sign = pos ? -1 : 1;
      s.u = pos ? val : -val;
      s.cc = pos ? -a : a;
      s.k = pos ? a * sub : -a * sub;
      b -= a * sub;
    } else if (use_lower) {  // x >= d + c y: y' = -(a (x - d) + a_y y) (a > 0: N2) or a (x - d) + a_y y (N1)
      const Index y = vlb->y;
      const double ay = imark_[y] ? icoef_[y] : 0.0, d = vlb->constant;
      const double val = a * vlb->coef + ay;
      s.bin = y, s.xb = xv[y];
      s.sign = pos ? -1 : 1;
      s.u = pos ? -val : val;
      s.cb = pos ? -ay : ay;
      s.cc = pos ? -a : a;
      s.k = pos ? a * d : -a * d;
      b -= a * d;
      fc_used_[y] = 1, fc_used_list_.push_back(y);
    } else if (!vub) {  // y' = a (x - l) (a > 0: N1) or -a (x - l) (N2); capacity |a| (u - l)
      const double val = a * (sub - slb);
      s.sign = pos ? 1 : -1;
      s.u = pos ? val : -val;
      s.cc = pos ? a : -a;
      s.k = pos ? -a * slb : a * slb;
      b -= a * slb;
    } else {  // x <= d + c y: y' = a (x - d) + a_y y (a > 0: N1) or its negation (N2)
      const Index y = vub->y;
      const double ay = imark_[y] ? icoef_[y] : 0.0, d = vub->constant;
      const double val = a * vub->coef + ay;
      s.bin = y, s.xb = xv[y];
      s.sign = pos ? 1 : -1;
      s.u = pos ? val : -val;
      s.cb = pos ? ay : -ay;
      s.cc = pos ? a : -a;
      s.k = pos ? -a * d : a * d;
      b -= a * d;
      fc_used_[y] = 1, fc_used_list_.push_back(y);
    }
    s.u = std::max(s.u, 0.0);
    if (!std::isfinite(s.u) || !std::isfinite(s.k)) return fail();
    snf_.push_back(s);
  }
  // Binaries of the aggregation not taken by a variable bound: y' = |a| y with capacity |a|.
  for (Index y : inz_) {
    const double a = icoef_[y];
    if (fc_used_[y] || a == 0.0) continue;
    SnfVar s{};
    s.sign = a >= 0 ? 1 : -1;
    s.u = std::abs(a);
    s.xb = xv[y];
    s.bin = y, s.cont = -1;
    s.cb = std::abs(a), s.cc = 0.0, s.k = 0.0, s.status = 0;
    snf_.push_back(s);
  }
  fail();  // releases fc_used_
  if (snf_.empty() || !std::isfinite(b)) return false;
  work_count_ += 4 * static_cast<long long>(snf_.size());

  // Flow cover C1 in N1, C2 in N2 with sum_C1 u - sum_C2 u = b + lambda, lambda > 0: arcs at 0 / 1
  // are fixed (x = 1: in the cover), the fractional ones chosen by a greedy knapsack.
  double fcw = 0.0, n1w = 0.0;
  fc_items_.clear();
  for (Index k = 0; k < static_cast<Index>(snf_.size()); ++k) {
    SnfVar& s = snf_[k];
    if (s.u <= kFeas) {
      s.status = -1;
      continue;
    }
    if (s.xb > kFeas && s.xb < 1.0 - kFeas) {
      fc_items_.push_back(k);
      if (s.sign > 0) n1w += s.u;
      continue;
    }
    s.status = s.xb > 0.5 ? 1 : -1;
    if (s.status > 0) fcw += s.sign * s.u;
  }
  const double cap = -b + fcw + n1w;
  if (cap / 10.0 <= kFeas) return false;
  if (!fc_items_.empty()) {
    // max sum profit z s.t. sum u z < cap; profit 1 - x (N1, z = 1: not in C1) or x (N2, z = 1: in C2).
    auto profit = [&](Index k) { return snf_[k].sign > 0 ? 1.0 - snf_[k].xb : snf_[k].xb; };
    std::stable_sort(fc_items_.begin(), fc_items_.end(),
                     [&](Index i, Index j) { return profit(i) * snf_[j].u > profit(j) * snf_[i].u; });
    double w = 0.0;
    const double lim = cap - kFeas * std::max(1.0, std::abs(cap));
    for (Index k : fc_items_) {
      SnfVar& s = snf_[k];
      const bool in = w + s.u < lim;
      if (in) w += s.u;
      s.status = s.sign > 0 ? (in ? -1 : 1) : (in ? 1 : -1);
      if (s.status > 0) fcw += s.sign * s.u;
    }
  }
  const double lambda = fcw - b;
  if (!(lambda > kFeas * std::max(1.0, std::abs(b)))) return false;

  // Lifting data: m = capacities > lambda of C1 and N2 \ C2 (decreasing), M their prefix sums.
  auto gt = [&](double a, double c) { return a - c > kEps * std::max({1.0, std::abs(a), std::abs(c)}); };
  std::vector<double>& m = fc_m_;
  std::vector<double>& M = fc_M_;
  m.clear();
  double n2le = 0.0, c1le = 0.0, c2 = 0.0, mp = kInf;
  for (const SnfVar& s : snf_) {
    if (s.sign < 0 && s.status < 0) {
      if (gt(s.u, lambda)) m.push_back(s.u);
      else n2le += s.u;
    } else if (s.sign < 0) {
      c2 += s.u;
    } else if (s.status > 0) {
      if (gt(s.u, lambda)) m.push_back(s.u), mp = std::min(mp, s.u);
      else c1le += s.u;
    }
  }
  if (!is_finite(mp)) return false;
  const double ml = std::min(lambda, c1le + n2le);
  std::sort(m.begin(), m.end(), std::greater<double>());
  const int r = static_cast<int>(m.size());
  M.assign(static_cast<size_t>(r) + 1, 0.0);
  for (int i = 0; i < r; ++i) M[i + 1] = M[i] + m[i];
  int t = 0;
  while (t < r && m[t] >= mp) ++t;
  auto segment = [&](double u) {
    int i = 0;
    while (i < r && gt(u + lambda, M[i + 1])) ++i;
    return i;
  };
  auto lift = [&](double x) {  // superadditive lifting function g(x) for C2 arcs
    const int i = segment(x);
    if (i < t) return gt(M[i], x) ? x - M[i] + i * lambda : i * lambda;
    if (i < r) {
      const double pp = std::max(0.0, m[i] - (mp - lambda) - ml);
      return gt(x + lambda, M[i] + ml + pp) ? i * lambda : x - M[i] + i * lambda;
    }
    return r * lambda + x - M[r];
  };

  // sum_C1 [y' + (u - lambda)^+ (1 - x)] + sum_{N1\C1} [y' - beta x] (lifted ones)
  //   - sum_{N2\C2, u <= lambda} y' - lambda sum_{N2\C2, u > lambda} x + sum_C2 g(u) (1 - x) <= b + sum_C2 u.
  cut->idx.clear(), cut->val.clear();
  double rhs = b + c2;
  auto add_y = [&](const SnfVar& s, double f) {
    if (s.cont >= 0 && s.cc != 0.0) emit(s.cont, f * s.cc, cut);
    if (s.bin >= 0 && s.cb != 0.0) cut->idx.push_back(s.bin), cut->val.push_back(f * s.cb);
    rhs -= f * s.k;
  };
  for (const SnfVar& s : snf_) {
    if (s.sign < 0 && s.status < 0) {
      if (gt(s.u, lambda)) {
        if (s.bin >= 0) cut->idx.push_back(s.bin), cut->val.push_back(-lambda);
        else rhs += lambda;
      } else {
        add_y(s, -1.0);
      }
    } else if (s.sign < 0) {
      if (s.bin >= 0) {
        const double g = lift(s.u);
        if (g != 0.0) cut->idx.push_back(s.bin), cut->val.push_back(-g), rhs -= g;
      }
    } else if (s.status < 0) {
      const int i = segment(s.u);
      if (gt(M[i], s.u)) {
        const double beta = M[i] - i * lambda;
        add_y(s, 1.0);
        if (s.bin >= 0) cut->idx.push_back(s.bin), cut->val.push_back(-beta);
        else rhs += beta;
      }
    } else {
      add_y(s, 1.0);
      if (s.bin >= 0 && gt(s.u, lambda)) {
        cut->idx.push_back(s.bin), cut->val.push_back(-(s.u - lambda));
        rhs -= s.u - lambda;
      }
    }
  }
  cut->rhs = rhs;
  if (cut->idx.empty() || !std::isfinite(rhs)) return false;
  for (size_t k = 0; k < cut->idx.size(); ++k) dense_[cut->idx[k]] += cut->val[k];
  double act = 0.0, norm = 0.0;
  for (Index j : cut->idx) {
    const double v = dense_[j];
    act += v * xv[j], norm += v * v, dense_[j] = 0.0;
  }
  work_count_ += static_cast<long long>(cut->idx.size());
  last_eff_ = norm > 1e-12 ? (act - rhs) / std::sqrt(norm) : -1.0;
  if (norm <= 1e-12 || last_eff_ < 1e-4) return false;
  cut->origin = "flowcover";
  return true;
}

void MirSeparator::build_tight(Index j) {
  const Index m0 = ctx_->num_problem_rows;
  const SparseMatrix& C = ctx_->lp.scaled_col_matrix();  // pattern as the model's; values scaled by row
  const SparseMatrix& R = ctx_->lp.scaled_row_matrix();  // the pattern only
  const MipProblem& p = ctx_->p;
  const Index max_len = 500;
  // The dual score is used for continuous columns without a variable bound from a two-variable row only
  // (see aggregate_path).
  const bool by_dual = dual_choice_ && !is_int(j) && !ctx_->data.row_vb[j];
  tight_beg_[j] = static_cast<Index>(tight_.size());
  for (Index k = C.begin(j); k < C.end(j); ++k) {
    const Index r = C.index[k];
    if (r >= m0 || row_dist_[r] > kTightDist) continue;
    const Index rlen = R.end(r) - R.begin(r);
    if (rlen > max_len) continue;
    if (rlen == 2 && (p.is_int[R.index[R.begin(r)]] || p.is_int[R.index[R.begin(r) + 1]])) continue;  // VB row
    tight_.push_back({r, C.value[k] / ctx_->lp.row_scale(r), row_dist_[r],
                      by_dual ? row_dual_[r] + 1e-3 / (1.0 + static_cast<double>(rlen)) : 0.0, row_int_[r]});
  }
  tight_end_[j] = static_cast<Index>(tight_.size());
  work_count_ += C.end(j) - C.begin(j);
}

void MirSeparator::aggregate_path(Index i0, double sign, std::vector<Cut>& out) {
  const MirSeparator& o = *own_;
  clear_agg();
  for (Index r : used_) row_used_[r] = 0;
  used_.assign(1, i0);
  row_used_[i0] = 1;
  add_row(i0, sign);
  int found = 0;
  int int_steps = 0;  // integer columns eliminated on this path
  for (int naggr = 0;; ++naggr) {
    Cut cut;
    // A cut does not end the path: larger aggregations give cut-set like cuts over several
    // nodes (h80x6320d, p200x1188c), often stronger than the first one found.
    if (knapsack_ && try_knapsack(&cut)) out.push_back(std::move(cut));
    // Flow cover and c-MIR of the same aggregation: one cut. Both together (2-3x the nodes) and the
    // more efficacious one (SCIP's SCIPcalcBestCut, 1.8x) made cflp_medium_1 slower over 6 seeds; the
    // flow cover only where c-MIR fails is neutral there and gave the best unitcal_7 root bound.
    Cut fc;
    last_eff_ = -1.0;
    const bool have_mir = try_cut(0.0, &cut);
    const double mir_eff = last_eff_;
    // The flow cover only where it can be taken: no c-MIR cut (or MILP_FC_WIN set); it doubled the c-MIR
    // separation time on app1-1 when tried on every aggregation.
    const double fc_eff = flowcover_ && (!have_mir || fc_win_ < 1e29) && try_flowcover(&fc) ? last_eff_ : -1.0;
    last_eff_ = mir_eff;
    if (fc_eff > 0.0 && (!have_mir || fc_eff > fc_win_ * last_eff_)) {
      out.push_back(std::move(fc));
      if (++found >= kCutsPerPath) return;
    } else if (have_mir) {
      out.push_back(std::move(cut));
      if (++found >= kCutsPerPath) return;
    }
    if (naggr >= kMaxAggr) return;
    // Eliminate the continuous structural farthest from its simple bounds (a flow at its
    // variable upper bound x_a = u y_a still links two flow balance rows).
    Index jbest = -1;
    double dbest = 1e-6;
    for (Index j : nz_) {
      if (j >= n_ || is_int(j) || agg_[j] == 0.0) continue;
      const double d = std::min(ctx_->x[j] - lower(j), upper(j) - ctx_->x[j]);
      if (d > dbest) dbest = d, jbest = j;
    }
    if (jbest < 0 && int_elim_ && ctx_->expensive && ctx_->round > 0 && work_count_ < int_budget_ && int_steps < int_elim_max_) {
      ++int_steps;
      // Root cut loop only (round > 0; node separation passes round 0: mas76 / neos5 trees slower with it).
      // No continuous column left (pure integer models, mzzv11): eliminate the integer column
      // farthest from its bounds instead (general integers only by default: their LP values act as
      // quantities linking rows, e.g. x = sum_k c_k z_k with x in a capacity row).
      double ibest = int_elim_min_;
      for (Index j : nz_) {
        if (j >= n_ || !is_int(j) || agg_[j] == 0.0 || (int_elim_general_ && upper(j) - lower(j) <= 1.0)) continue;
        const double d = std::min(ctx_->x[j] - lower(j), upper(j) - ctx_->x[j]);
        const double score = d + (upper(j) - lower(j) > 1.0 ? 1.0 : 0.0);
        if (d > int_elim_min_ && score > ibest) ibest = score, jbest = j;
      }
    }
    if (jbest < 0) return;
    // Pick a tight problem row containing it. A column without variable bounds is bounded through
    // rows with integer columns if at all (n5-3: commodity flows x_ak in the arc capacity rows
    // sum_k x_ak <= 7 y1 + 31 y2 + 255 y3): prefer those then (a column with a VUB/VLB is
    // substituted by it in the transformation, and its path should follow the flow balances, as
    // in lot sizing: x_t <= M y_t, s_t-1 + x_t = d_t + s_t).
    Index rbest = -1;
    double sbest = kInf, mbest = 0.0, dbest_row = -1.0;
    const bool prefer_int = ctx_->data.vub[jbest].empty() && ctx_->data.vlb[jbest].empty();
    if (o.tight_beg_[jbest] < 0) build_tight(jbest);  // own_ == this here (separate_chunked builds them first)
    // Among equally tight rows: the one with the largest LP dual (as SCIP's aggregation row score: the
    // rows that prove the LP bound; neos17 pairs an observation row with a contradicting one), then
    // the sparser one; otherwise the first tight row in column order. Not for integer eliminations
    // (mzzv42z root 100 -> 98%) nor columns with a variable bound from a two-variable row (x <= u y:
    // the balance rows of lot sizing / fixed charge paths are taken in order; by dual, mc11 went from
    // 15-23 s to 24-120 s over 3 seeds, exp-1-500-5-5 1 -> 3 s). (TightRow::ds, set in build_tight.)
    const double aj = agg_[jbest];
    const TightRow* tend = o.tight_.data() + o.tight_end_[jbest];
    for (const TightRow* t = o.tight_.data() + o.tight_beg_[jbest]; t != tend; ++t) {
      const Index r = t->r;
      if (row_used_[r]) continue;
      // The row must be tight (its logical at a bound).
      const double mult = -aj / t->val;
      double slack = t->dist * std::abs(mult);
      if (slack > 1e-6) continue;
      if (prefer_int && !t->has_int) slack += 1.0;
      if (slack < sbest - 1e-12 || (slack <= sbest + 1e-12 && t->ds > dbest_row)) {
        sbest = slack, rbest = r, mbest = mult, dbest_row = t->ds;
      }
    }
    if (rbest < 0) return;
    add_row(rbest, mbest);
    agg_[jbest] = 0.0;  // exact elimination
    used_.push_back(rbest);
    row_used_[rbest] = 1;
  }
}

void MirSeparator::separate_tableau(std::vector<Cut>& out) {
  const MipProblem& p = ctx_->p;
  DualSimplex& lp = ctx_->lp;
  const double* x = ctx_->x;
  const Index m = lp.num_rows();
  std::vector<std::pair<double, Index>> cand;
  for (Index j : lp.basic_vars()) {
    if (j >= n_ || !p.is_int[j]) continue;
    const double f = x[j] - std::floor(x[j]);
    if (f < 0.01 || f > 0.99) continue;
    cand.emplace_back(std::abs(f - 0.5), j);
  }
  std::sort(cand.begin(), cand.end());
  static const size_t kRows = static_cast<size_t>([] {
    const char* v = std::getenv("MILP_TMIR_ROWS");
    return v ? std::atoi(v) : 50;
  }());
  if (cand.size() > kRows) cand.resize(kRows);
  tab_.assign(n_ + m, 0.0);
  tnz_.clear();
  const long long nnz = static_cast<long long>(p.A.index.size());
  const long long budget = static_cast<long long>(kWorkPerNnz * static_cast<double>(std::max<long long>(nnz, 20000)));
  for (const auto& [score, jb] : cand) {
    if (work_count_ >= budget) break;
    for (Index j : tnz_) tab_[j] = 0.0;
    tnz_.clear();
    if (!lp.tableau_row(jb, tab_, tnz_)) continue;
    work_count_ += 4 * static_cast<long long>(tnz_.size());
    double amax = 0.0;
    for (Index j : tnz_) amax = std::max(amax, std::abs(tab_[j]));
    for (double sign : {1.0, -1.0}) {
      clear_agg();
      // The base is the identity sum_j t_j x_j = 0 (structurals and logicals) as sum agg_j x_j
      // <= b. A tiny entry is left out only with its bound: sum_rest <= -a x_j <= -a l_j (a > 0)
      // or -a u_j (a < 0); without a finite bound only at the rounding level of the row itself.
      // Dropping entries up to 1e-11 max |t_j| without the bound made the cuts invalid where a
      // logical has a huge range (neos-4413714-turia, a budget row with rhs 9e8: node cuts
      // violated by the known optimum by up to 0.004, the search reported the bound 46.47 above
      // the optimum 45.37). MILP_TMIR_DROP=1: the old drop.
      double b = 0.0;
      static const bool old_drop = std::getenv("MILP_TMIR_DROP") != nullptr;
      const double tiny = 1e-11 * std::max(1.0, amax);
      for (Index j : tnz_) {
        const double v = tab_[j];
        if (v == 0.0) continue;
        if (std::abs(v) <= tiny) {
          if (old_drop) continue;
          const double a = sign * v;
          const double bnd = a > 0 ? lower(j) : upper(j);
          if (is_finite(bnd)) {
            b -= a * bnd;
            continue;
          }
          if (std::abs(v) <= 1e-4 * tiny) continue;
        }
        mark_[j] = 1;
        nz_.push_back(j);
        agg_[j] = sign * v;
      }
      Cut cut;
      if (try_cut(b, &cut)) {
        cut.origin = "tmir";
        out.push_back(std::move(cut));
      }
      if (knapsack_ && try_knapsack(&cut, b)) {
        cut.origin = "tcover";
        out.push_back(std::move(cut));
      }
    }
  }
  for (Index j : tnz_) tab_[j] = 0.0;
  tnz_.clear();
}

void MirSeparator::separate_mcf(std::vector<Cut>& out) {
  if (!net_built_) {
    const auto t0 = std::chrono::steady_clock::now();
    net_ = detect_mcf_network(ctx_->p);
    net_built_ = true;
    if (std::getenv("MILP_MCF_DEBUG")) {
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      Index ncap = 0;
      for (const auto& a : net_.arcs) ncap += a.cap_row >= 0;
      std::fprintf(stderr, "MCF network: %d nodes, %d commodities, %zu flow rows, %zu arcs (%d capacitated), detection %.4fs\n",
                   static_cast<int>(net_.num_nodes), static_cast<int>(net_.num_comms), net_.flow_rows.size(),
                   net_.arcs.size(), static_cast<int>(ncap), secs);
    }
  }
  if (net_.empty() || ctx_->round <= 0) return;
  const McfNetwork& N = net_;
  const Index V = N.num_nodes;
  const double* x = ctx_->x;
  DualSimplex& lp = ctx_->lp;
  const long long nnz = static_cast<long long>(ctx_->p.A.index.size());
  static const double work_per_nnz = [] { const char* v = std::getenv("MILP_MCF_WORK"); return v ? std::atof(v) : 200.0; }();
  const long long budget = static_cast<long long>(work_per_nnz * static_cast<double>(std::max<long long>(nnz, 20000)));
  // Contraction order of the arcs: uncapacitated first, then by decreasing relative slack of the capacity
  // row, then by increasing |dual| of a tight capacity row.
  std::vector<std::pair<double, Index>> order;
  for (Index a = 0; a < static_cast<Index>(N.arcs.size()); ++a) {
    const McfNetwork::Arc& arc = N.arcs[a];
    if (arc.tail < 0 || arc.head < 0 || arc.tail == arc.head) continue;
    double key;
    if (arc.cap_row < 0) {
      key = -1e30;
    } else {
      const Index c = arc.cap_row;
      const double side = std::max(std::abs(ctx_->p.row_lb[c]) * is_finite(ctx_->p.row_lb[c]),
                                   std::abs(ctx_->p.row_ub[c]) * is_finite(ctx_->p.row_ub[c]));
      const double rel = row_dist_[c] / (1.0 + side);
      key = rel > 1e-6 ? -1.0 - rel : std::abs(lp.reduced_cost(n_ + c));
    }
    order.emplace_back(key, a);
  }
  std::stable_sort(order.begin(), order.end());
  // Node sets: singletons, then the components of the contraction (at most half of the nodes).
  std::vector<Index> set_start{0}, set_nodes;
  for (Index v = 0; v < V; ++v) set_nodes.push_back(v), set_start.push_back(static_cast<Index>(set_nodes.size()));
  {
    std::vector<Index> parent(V);
    std::vector<std::vector<Index>> members(V);
    for (Index v = 0; v < V; ++v) parent[v] = v, members[v].assign(1, v);
    auto find = [&](Index a) {
      while (parent[a] != a) a = parent[a] = parent[parent[a]];
      return a;
    };
    const size_t max_listed = static_cast<size_t>(std::max<long long>(20 * nnz, 100000));
    for (const auto& [key, a] : order) {
      Index r1 = find(N.arcs[a].tail), r2 = find(N.arcs[a].head);
      if (r1 == r2) continue;
      if (members[r1].size() < members[r2].size()) std::swap(r1, r2);
      parent[r2] = r1;
      members[r1].insert(members[r1].end(), members[r2].begin(), members[r2].end());
      std::vector<Index>().swap(members[r2]);
      if (2 * members[r1].size() > static_cast<size_t>(V) || set_nodes.size() + members[r1].size() > max_listed) continue;
      set_nodes.insert(set_nodes.end(), members[r1].begin(), members[r1].end());
      set_start.push_back(static_cast<Index>(set_nodes.size()));
    }
  }
  std::vector<double> supply(N.num_comms, 0.0);
  std::vector<char> comm_listed(N.num_comms, 0);
  std::vector<Index> comms;
  std::vector<char> cap_added(N.arcs.size(), 0);
  std::vector<Index> caps, cross;
  const Index nsets = static_cast<Index>(set_start.size()) - 1;
  for (Index s = 0; s < nsets && work_count_ < budget; ++s) {
    // Net supply of each commodity in S at the LP point (flow row activity = -logical).
    comms.clear();
    for (Index t = set_start[s]; t < set_start[s + 1]; ++t) {
      const Index v = set_nodes[t];
      for (Index q = N.node_start[v]; q < N.node_start[v + 1]; ++q) {
        const McfNetwork::FlowRow& fr = N.flow_rows[q];
        if (!comm_listed[fr.comm]) comm_listed[fr.comm] = 1, comms.push_back(fr.comm);
        supply[fr.comm] += fr.mult * -x[n_ + fr.row];
      }
      work_count_ += N.node_start[v + 1] - N.node_start[v];
    }
    for (double sigma : {1.0, -1.0}) {
      // sigma = 1: sum_S (outflow - inflow) = supply; commodities with net demand (their inflow arcs need
      // capacity). sigma = -1: the negation; commodities with net supply.
      clear_agg();
      bool any = false;
      for (Index t = set_start[s]; t < set_start[s + 1]; ++t) {
        const Index v = set_nodes[t];
        for (Index q = N.node_start[v]; q < N.node_start[v + 1]; ++q) {
          const McfNetwork::FlowRow& fr = N.flow_rows[q];
          if (sigma * supply[fr.comm] > -1e-6) continue;
          add_row(fr.row, sigma * fr.mult);
          any = true;
        }
      }
      if (!any) continue;
      // Crossing flows with a negative coefficient: substitute their capacity rows.
      cross.clear();
      for (Index j : nz_) {
        if (j < n_ && N.col_arc[j] >= 0 && agg_[j] < -1e-9) cross.push_back(j);
      }
      caps.clear();
      for (Index j : cross) {
        const Index a = N.col_arc[j];
        const Index c = N.arcs[a].cap_row;
        if (c < 0 || cap_added[a] || agg_[j] >= -1e-9 || std::abs(N.col_cap[j]) < 1e-6) continue;
        cap_added[a] = 1, caps.push_back(a);
        add_row(c, -agg_[j] / N.col_cap[j]);
        agg_[j] = 0.0;
      }
      for (Index a : caps) cap_added[a] = 0;
      if (caps.empty()) continue;
      Cut cut;
      if (try_cut(0.0, &cut)) {
        cut.origin = "mcf";
        out.push_back(std::move(cut));
      }
      if (knapsack_ && try_knapsack(&cut)) {
        cut.origin = "mcf";
        out.push_back(std::move(cut));
      }
    }
    for (Index k : comms) supply[k] = 0.0, comm_listed[k] = 0;
  }
}

void MirSeparator::separate(const SepaContext& ctx, std::vector<Cut>& out) {
  const MipProblem& p = ctx.p;
  DualSimplex& lp = ctx.lp;
  ctx_ = &ctx;
  n_ = p.num_cols();
  work_count_ = 0;
  if (mcf_ && (ctx.round <= 0 || (net_built_ && net_.empty()))) return;  // no network: skip the setup below
  {
    const char* v = std::getenv("MILP_NO_AGGCOVER");
    knapsack_ = !(v && *v && v[0] != '0');
    // Opt-in (MILP_FLOWCOVER=1): at 120 s / 4 threads unitcal_7 went from 2/3 seeds solved to 0/3 (root bound
    // +0..10k, but a longer root), dev set / miplib_dev neutral.
    const char* f = std::getenv("MILP_FLOWCOVER");
    flowcover_ = use_flowcover_ && f && *f && f[0] != '0';
    const char* w = std::getenv("MILP_FC_WIN");
    fc_win_ = w ? std::atof(w) : 1e30;
  }
  const Index m = lp.num_rows(), m0 = ctx.num_problem_rows;
  const SparseMatrix& R = lp.scaled_row_matrix();  // entry * row_unscale(row): the model's coefficient, exactly
  agg_.assign(n_ + m, 0.0);
  mark_.assign(n_ + m, 0);
  icoef_.assign(n_, 0.0);
  dense_.assign(n_, 0.0);
  imark_.assign(n_, 0);
  nz_.clear();
  inz_.clear();
  const Index max_len = 500;
  Index tried = 0;
  work_count_ = 0;
  row_int_.assign(m0, 0);
  row_used_.assign(m0, 0);
  used_.clear();
  row_dist_.resize(m0);
  tight_beg_.assign(n_, -1);
  tight_end_.assign(n_, -1);
  tight_.clear();
  bd_slot_.assign(n_, -1);
  bd_cache_.clear();
  own_ = this;
  lo_.resize(n_ + m), up_.resize(n_ + m), cls_.resize(n_ + m);
  for (Index j = 0; j < n_; ++j) {
    lo_[j] = p.col_lb[j], up_[j] = p.col_ub[j];
    cls_[j] = is_int_column(j) ? kClsInt : ctx.data.vlb[j].size() + ctx.data.vub[j].size() >= kBdCacheMin ? kClsVb : kClsCont;
  }
  for (Index j = n_; j < n_ + m; ++j) lo_[j] = lp.var_lower(j), up_[j] = lp.var_upper(j), cls_[j] = kClsCont;
  for (Index i = 0; i < m0; ++i) {
    ContTerm t;
    row_dist_[i] = bound_distance(n_ + i, &t);
  }
  if (dual_choice_) {
    double cn = 0.0;
    for (Index j = 0; j < n_; ++j) cn += p.cost[j] * p.cost[j];
    cn = std::max(std::sqrt(cn), 1e-9);
    row_dual_.resize(m0);
    for (Index i = 0; i < m0; ++i) row_dual_[i] = std::min(1.0, std::abs(lp.reduced_cost(n_ + i)) / cn);
  }
  if (flowcover_ && !tableau_) {
    row_minact_.assign(m0, 0.0);
    row_maxact_.assign(m0, 0.0);
    for (Index i = 0; i < m0; ++i) {
      double lo = 0.0, hi = 0.0;
      const double us = lp.row_unscale(i);
      for (Index k = R.begin(i); k < R.end(i); ++k) {
        const Index j = R.index[k];
        const double v = R.value[k] * us;
        lo += v * (v > 0 ? p.col_lb[j] : p.col_ub[j]);
        hi += v * (v > 0 ? p.col_ub[j] : p.col_lb[j]);
      }
      row_minact_[i] = std::isnan(lo) ? -kInf : lo;
      row_maxact_[i] = std::isnan(hi) ? kInf : hi;
    }
    work_count_ += static_cast<long long>(p.A.index.size());
  }
  for (Index i = 0; i < m0; ++i) {
    for (Index k = R.begin(i); k < R.end(i) && !row_int_[i]; ++k) row_int_[i] = p.is_int[R.index[k]];
  }
  if (tableau_) {
    separate_tableau(out);
    clear_agg();
    return;
  }
  if (mcf_) {
    separate_mcf(out);
    clear_agg();
    return;
  }
  const long long nnz = static_cast<long long>(p.A.index.size());
  const double scale = ctx.work_scale * static_cast<double>(std::max<long long>(nnz, 20000));
  const long long budget = static_cast<long long>(kWorkPerNnz * scale);
  int_budget_ = static_cast<long long>(int_work_per_nnz_ * scale);
  {
    // Pure integer models only (neos-4738912-atrato, mixed: 2.3x the root cuts for the same bound, tree 70 s ->
    // TL); on mixed models paths through continuous columns already link the rows.
    bool pure = true;
    for (Index j = 0; j < n_ && pure; ++j) pure = p.is_int[j] || p.col_lb[j] == p.col_ub[j];
    if (!pure) int_budget_ = 0;
  }
  // Pass 0: rows with integer variables (or continuous ones with variable bounds); pass 1: purely
  // continuous rows (e.g. single-commodity flow balances), which reach integers by aggregation only.
  // Start rows used to be the first 5000 in index order: on large models (unitcal_7, 38k rows) the
  // same rows in every round. Now all rows, best first (as SCIP's aggregation separator: tight rows
  // with fractional integers, sparse ones first) until the work budget is used (MILP_MIR_ROW_ORDER=0:
  // index order; MILP_MIR_STARTS: a cap on the start rows). unitcal_7 root 19536281 -> 19564403.
  static const bool scored = [] { const char* v = std::getenv("MILP_MIR_ROW_ORDER"); return !(v && v[0] == '0'); }();
  starts_.clear();
  for (int pass = 0; pass < 2; ++pass) {
    for (Index i0 = 0; i0 < m0; ++i0) {
      const Index len = R.end(i0) - R.begin(i0);
      if (len < 1 || len > max_len) continue;
      bool has_int = false, movable = false;
      int nfrac = 0;
      for (Index k = R.begin(i0); k < R.end(i0); ++k) {
        const Index j = R.index[k];
        has_int = has_int || p.is_int[j] || !ctx.data.vlb[j].empty() || !ctx.data.vub[j].empty();
        movable = movable || std::min(ctx.x[j] - p.col_lb[j], p.col_ub[j] - ctx.x[j]) > 1e-6;
        if (p.is_int[j]) nfrac += std::abs(ctx.x[j] - std::round(ctx.x[j])) > 1e-6;
      }
      if (has_int != (pass == 0) || (pass == 1 && !movable)) continue;
      double score = 0.0;
      if (scored) {
        const double tight = row_dist_[i0] <= kTightDist ? 1.0 : 1.0 / (1.0 + row_dist_[i0]);
        score = tight * (1.0 + static_cast<double>(nfrac)) / (1.0 + 0.01 * static_cast<double>(len));
      }
      starts_.push_back({static_cast<double>(pass) * 1e30 - score, i0});
    }
  }
  if (scored) std::stable_sort(starts_.begin(), starts_.end());
  static const Index max_starts = [] { const char* v = std::getenv("MILP_MIR_STARTS"); return v ? std::atoi(v) : 1 << 30; }();
  if (par_ && ctx.round > 0) {
    // Root cut loops only: node separation (round 0) stays sequential.
    separate_chunked(budget, max_starts, out);
    clear_agg();
    return;
  }
  // The stop flag and the deadline are polled per start (neos-5104907-jarama: a node's separation ran 0.6 s past
  // the time limit).
  for (size_t s = 0; s < starts_.size() && tried < max_starts && work_count_ < budget && !interrupted_or_expired(); ++s) {
    const Index i0 = starts_[s].second;
    ++tried;
    if (is_finite(p.row_ub[i0])) aggregate_path(i0, 1.0, out);
    if (is_finite(p.row_lb[i0])) aggregate_path(i0, -1.0, out);
  }

  clear_agg();
}

void MirSeparator::separate_chunked(long long budget, Index max_starts, std::vector<Cut>& out) {
  const MipProblem& p = ctx_->p;
  // The paths share only read-only state: the context's problem and LP (row / column matrices, bounds, x),
  // the CutData and this separator's setup. tight_* is built on first use otherwise: all columns first.
  for (Index j = 0; j < n_; ++j) {
    if (tight_beg_[j] < 0) build_tight(j);
  }
  const int threads = std::max(1, ctx_->threads);
  while (static_cast<int>(helpers_.size()) < threads - 1) {
    helpers_.push_back(std::make_unique<MirSeparator>(tableau_, use_flowcover_, mcf_));
  }
  for (int t = 0; t + 1 < threads; ++t) {
    MirSeparator& h = *helpers_[t];
    h.own_ = this;
    h.ctx_ = ctx_;
    h.n_ = n_;
    h.knapsack_ = knapsack_, h.flowcover_ = flowcover_, h.fc_win_ = fc_win_;
    h.int_budget_ = int_budget_;
    h.int_elim_ = int_elim_, h.int_elim_min_ = int_elim_min_, h.int_elim_max_ = int_elim_max_;
    h.int_elim_general_ = int_elim_general_, h.dual_choice_ = dual_choice_;
    h.agg_.assign(agg_.size(), 0.0);
    h.mark_.assign(mark_.size(), 0);
    h.icoef_.assign(n_, 0.0);
    h.dense_.assign(n_, 0.0);
    h.imark_.assign(n_, 0);
    h.bd_slot_.assign(n_, -1);
    h.bd_cache_.clear();
    h.nz_.clear();
    h.inz_.clear();
    h.row_used_.assign(row_used_.size(), 0);
    h.used_.clear();
  }
  if (threads > 1 && (!pool_ || pool_->size() != threads)) pool_ = std::make_unique<WorkerPool>(threads);
  static const size_t chunk = [] {
    const char* v = std::getenv("MILP_MIR_PAR_CHUNK");
    return v ? static_cast<size_t>(std::max(1, std::atoi(v))) : kParChunk;
  }();
  chunk_cuts_.resize(chunk);
  chunk_work_.resize(chunk);
  size_t s = 0;
  Index tried = 0;
  while (s < starts_.size() && tried < max_starts && work_count_ < budget && !interrupted_or_expired()) {
    const size_t len = std::min({chunk, starts_.size() - s, static_cast<size_t>(max_starts - tried)});
    const long long w0 = work_count_;
    std::atomic<size_t> next{0};
    // Task t runs on worker t's scratch state (this one or a helper), taking the chunk's starts in turn.
    auto run = [&](int t) {
      MirSeparator& w = t == 0 ? *this : *helpers_[t - 1];
      for (size_t c = next.fetch_add(1); c < len; c = next.fetch_add(1)) {
        const Index i0 = starts_[s + c].second;
        std::vector<Cut>& cuts = chunk_cuts_[c];
        cuts.clear();
        w.work_count_ = w0;
        if (is_finite(p.row_ub[i0])) w.aggregate_path(i0, 1.0, cuts);
        if (is_finite(p.row_lb[i0])) w.aggregate_path(i0, -1.0, cuts);
        chunk_work_[c] = w.work_count_ - w0;
      }
    };
    if (threads > 1) {
      pool_->run(threads, run);
    } else {
      run(0);
    }
    work_count_ = w0;
    for (size_t c = 0; c < len; ++c) {
      for (Cut& cut : chunk_cuts_[c]) out.push_back(std::move(cut));
      chunk_cuts_[c].clear();
      work_count_ += chunk_work_[c];
    }
    s += len;
    tried += static_cast<Index>(len);
  }
}

}  // namespace

std::unique_ptr<Separator> make_mir_separator(bool flowcover) { return std::make_unique<MirSeparator>(false, flowcover); }
std::unique_ptr<Separator> make_tableau_mir_separator() { return std::make_unique<MirSeparator>(true); }
std::unique_ptr<Separator> make_mcf_separator() { return std::make_unique<MirSeparator>(false, false, true); }

}  // namespace milp
