#pragma once
// Cutting-plane infrastructure: the cut representation, the separator plugin
// interface and the cut pool (numerical cleanup, efficacy, parallelism-based
// selection).

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
  // Probes binaries fractional in x first (x may be null), within `work_limit` matrix entries.
  void probe(const MipProblem& p, const double* x, long long work_limit);
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
};

class Separator {
 public:
  virtual ~Separator() = default;
  virtual const char* name() const = 0;
  // Appends candidate cuts (not necessarily cleaned or violated) to `out`.
  virtual void separate(const SepaContext& ctx, std::vector<Cut>& out) = 0;
};

std::unique_ptr<Separator> make_gomory_separator();
std::unique_ptr<Separator> make_cover_separator();
// flowcover: also lifted flow cover cuts on the aggregations (off for node separation by default).
std::unique_ptr<Separator> make_mir_separator(bool flowcover = true);
// c-MIR / lifted cover cuts from simplex tableau rows (globally valid, also at nodes).
std::unique_ptr<Separator> make_tableau_mir_separator();
std::unique_ptr<Separator> make_clique_separator();
std::unique_ptr<Separator> make_implied_bound_separator();
std::unique_ptr<Separator> make_zerohalf_separator();

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
};

class CutPool {
 public:
  CutPool(const MipProblem& p, const CutPoolOptions& opt) : p_(p), opt_(opt) {}

  // Cleans the cut and keeps it if it is violated enough by x. Returns true if kept.
  bool add(Cut c, const double* x);
  // Picks up to max_cuts cuts by score, skipping near-parallel ones. Clears the pool.
  std::vector<Cut> select();
  size_t size() const { return cuts_.size(); }

 private:
  const MipProblem& p_;
  CutPoolOptions opt_;
  std::vector<Cut> cuts_;
  double cnorm_ = -1.0;  // ||c||^2, computed on the first add
};

}  // namespace milp
