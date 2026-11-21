#pragma once
// Cutting-plane infrastructure: the cut representation, the separator plugin
// interface and the cut pool (numerical cleanup, efficacy, parallelism-based
// selection).

#include <functional>
#include <memory>
#include <vector>

#include "milp/core/types.hpp"
#include "milp/lp/simplex.hpp"
#include "milp/search/mip_problem.hpp"

namespace milp {

// A valid inequality sum_k val[k] * x[idx[k]] <= rhs over structural variables.
struct Cut {
  std::vector<Index> idx;
  std::vector<double> val;
  double rhs = 0.0;
  const char* origin = "";  // separator name
  double efficacy = 0.0;    // (a x* - rhs) / ||a||, set by the pool
  double score = 0.0;
  double objpar = 0.0, intsupport = 0.0;  // set by the pool
  int pool_id = -1;         // cut loop bookkeeping: index in the pool of dropped cuts
  int sepa = -1;            // cut loop bookkeeping: the separator that found it (this round)

  double activity(const double* x) const {
    double s = 0.0;
    for (size_t k = 0; k < idx.size(); ++k) s += val[k] * x[idx[k]];
    return s;
  }
};

// Variable bound x_j <= constant + coef * y (VUB) or x_j >= constant + coef * y
// (VLB) with x_j continuous and y integer, implied by a two-variable row.
struct VarBound {
  Index y = -1;
  double coef = 0.0, constant = 0.0;
};

// Bounds on a general integer column x implied by fixing a binary y: x <= / >= v0 + (v1 - v0) y.
struct ProbeImplication {
  Index x, y;
  double v0, v1;  // bound on x when y = 0 / y = 1
  bool upper;
};

// Problem data derived once per cut loop and shared by the separators.
struct CutData {
  std::vector<std::vector<VarBound>> vlb, vub;  // per column
  std::vector<char> row_vb;  // per column: has a variable bound from a two-variable row (x <= u y)
  // Probing on binaries (fix to 0 and to 1, propagate the rows): implied bounds on general
  // integers and conflicting binary literal pairs (literal 2j: x_j = 1, 2j + 1: x_j = 0), the
  // latter as adjacency lists (CSR over literals).
  std::vector<ProbeImplication> probe_imps;
  std::vector<Index> conflict_start, conflict_lit;
  bool probed = false;
  // Implied variable bounds of continuous columns on binaries (single longer rows in build(), probing
  // in probe()), stored in vlb / vub next to those of two-variable rows (MILP_NO_IMPLIED_VB=1 /
  // MILP_NO_PROBE_VB=1 off).
  bool implied_vbs = true, probe_vbs = true;
  size_t num_implied_vbs = 0;
  static constexpr size_t kMaxVbPerCol = 8;
  // Adds x <= / >= constant + coef y (binary y) unless dominated by a stored bound on the same y.
  bool add_vb(Index x, const VarBound& vb, bool upper);
  void build(const MipProblem& p);
  // Probes binaries fractional in x first (x may be null), within `work_limit` matrix entries; stops early
  // (the results so far are valid) once `stop` returns true (checked every 64 probes, e.g. the time limit).
  void probe(const MipProblem& p, const double* x, long long work_limit, const std::function<bool()>& stop = {});
};

struct SepaContext {
  const MipProblem& p;
  DualSimplex& lp;      // solved to optimality; rows >= num_problem_rows are cuts
  const double* x;      // LP values of all n + m variables (structurals first)
  Index num_problem_rows;
  const CutData& data;
  int round = 0;
  Index max_nnz = 0;  // > 0: cuts with more nonzeros are rejected by the pool (separators may skip them)
  bool expensive = true;  // false: skip expensive separation variants (flat bound on a degenerate LP)
  double work_scale = 1.0;  // factor on the separators' per-call work budgets (node separation: MILP_NODE_SEP_WORK)
  // Threads a separator may use (the caller's included; the main search's root cut loops only). The cuts must
  // not depend on it (c-MIR start rows with MILP_MIR_PAR=1).
  int threads = 1;
};

class Separator {
 public:
  virtual ~Separator() = default;
  virtual const char* name() const = 0;
  // Appends candidate cuts (not necessarily cleaned or violated) to `out`.
  virtual void separate(const SepaContext& ctx, std::vector<Cut>& out) = 0;
  // Work of the last separate() call in deterministic units (about the matrix entries it touched; 0: not
  // counted, a cheap separator).
  virtual long long work() const { return 0; }
};

std::unique_ptr<Separator> make_gomory_separator();
std::unique_ptr<Separator> make_cover_separator();
// flowcover: also lifted flow cover cuts on the aggregations (off for node separation by default).
std::unique_ptr<Separator> make_mir_separator(bool flowcover = true);
// c-MIR / lifted cover cuts from simplex tableau rows (globally valid, also at nodes).
std::unique_ptr<Separator> make_tableau_mir_separator();
// Network cut-set c-MIR cuts on detected multi-commodity flow structure (cuts/mcf.hpp; root only).
std::unique_ptr<Separator> make_mcf_separator();
std::unique_ptr<Separator> make_clique_separator();
std::unique_ptr<Separator> make_implied_bound_separator();
std::unique_ptr<Separator> make_zerohalf_separator();
// Gradient cuts of MipProblem::quad_rows at the LP point (outer approximation; quad_oa.cpp).
std::unique_ptr<Separator> make_quad_oa_separator();
// Gradient cuts of the quadratic rows violated at x by more than tol (absolute, as max_violation): for a row with g
// convex in the direction of its finite side, g(x^) + grad g(x^) (x - x^) <= hi (>= lo), valid for every x. Returns
// the number appended to `out`.
int quad_oa_cuts(const MipProblem& p, const double* x, double tol, std::vector<Cut>& out);

// Lifted cover inequality for the binary knapsack sum_k w_k z_k <= b (w_k > 0, LP values zs_k):
// a minimal cover C chosen greedily by (1 - zs) / w, lifted with the superadditive function of
// Gu, Nemhauser & Savelsbergh (sequence independent). On success coef (one per item, >= 0) and
// rhs = |C| - 1 give sum coef_k z_k <= rhs, violated by zs by more than 1e-6.
bool knapsack_cover_cut(const std::vector<double>& w, const std::vector<double>& zs, double b,
                        std::vector<double>& coef, double& rhs);

// Numerical cleanup of a cut against global column bounds [lb, ub]:
// fixed variables are moved to the rhs, coefficients tiny relative to the
// largest one are removed while relaxing the rhs with the variable's bound
// (rejecting the cut if that bound is infinite), the cut is scaled by a power
// of two to max |a| ~ 1 and the rhs gets a tiny safety relaxation. Returns
// false if the cut is rejected (non-finite data, empty, too dense).
bool clean_cut(Cut& c, const std::vector<double>& lb, const std::vector<double>& ub, Index max_nnz);

struct CutPoolOptions {
  double min_efficacy = 1e-4;
  double min_violation = 1e-6;
  double max_parallelism = 0.98;  // reject cuts with cos(angle) above this to a selected cut
  int max_cuts = 100;
  Index max_nnz = 1 << 30;
  // Hybrid scoring (SCIP-like): score = efficacy / max efficacy + obj_weight * objective parallelism
  // + int_weight * integral support; cuts scoring below good_score x the best score must also be at
  // most max_parallelism_weak parallel to every selected cut. Off (hybrid = false): score =
  // efficacy (1 + 0.1 objective parallelism) and max_parallelism for all.
  bool hybrid = false;
  double obj_weight = 0.1, int_weight = 0.1;
  double good_score = 0.9;
  double max_parallelism_weak = 0.98;
  // Density rule (MILP_CUT_DENSE = x, default 8, off at 0): a cut with more than dense_len nonzeros competes with its score
  // times sqrt(dense_len / nonzeros), and is not selected when its efficacy scaled that way is below dense_share
  // of the round's best efficacy. Dense cut rows stay in the tree's LPs and make every basis solve denser.
  Index dense_len = 0;
  double dense_share = 0.5;
  bool dense_gomory_only = false;  // mode 3: the rule applies to tableau cuts (gomory, tmir, zerohalf) only
  bool dense_root = false;         // statistics: the pool belongs to a root cut loop
  // Sets dense_len from MILP_CUT_DENSE = x: max(MILP_CUT_DENSE_FLOOR (10), x times the model's average row length).
  // root: the pool of a root cut loop (else a node cut round). MILP_CUT_DENSE_MODE (default 6): 0 everywhere, 1 node rounds
  // only, 2 root loops only, 3 tableau cuts only, 4 as 0 but a root loop applies it only once its bound has
  // stopped moving (dense_mode() == 4: the loop switches dense_len per round), 5 as 0 with a length limit of at
  // least sqrt(columns). 6: tableau cuts always, the other families in a root loop only once its bound
  // has stopped moving (3 and 4 together).
  void set_dense(const MipProblem& p, bool root);
  static int dense_mode();
};

class CutPool {
 public:
  CutPool(const MipProblem& p, const CutPoolOptions& opt) : p_(p), opt_(opt) {}

  // Cleans the cut and keeps it if it is violated enough by x. Returns true if kept.
  bool add(Cut c, const double* x);
  // Picks up to max_cuts cuts by score, skipping near-parallel ones. Clears the pool.
  std::vector<Cut> select();
  size_t size() const { return cuts_.size(); }
  // The last select(): candidates, rejected as too parallel, left over at the max_cuts limit.
  struct SelectStats {
    int candidates = 0, parallel = 0, over_limit = 0;
  };
  const SelectStats& select_stats() const { return sstats_; }

 private:
  const MipProblem& p_;
  CutPoolOptions opt_;
  std::vector<Cut> cuts_;
  double cnorm_ = -1.0;  // ||c||^2, computed on the first add
  SelectStats sstats_;
};

}  // namespace milp
