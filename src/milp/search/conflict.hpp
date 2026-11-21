#pragma once
// Conflict analysis by dual proofs (Witzig, Berthold, Heinz): when a node LP
// is infeasible or exceeds the cutoff, the LP multipliers that prove it are
// aggregated into one constraint over the problem rows (and globally valid
// cut rows). The constraint is valid for every solution better than the
// cutoff, so it is stored in a conflict pool and propagated at other nodes,
// pruning subtrees that fail for the same reason without an LP solve.

#include <memory>
#include <vector>

#include "milp/core/types.hpp"
#include "milp/lp/simplex.hpp"
#include "milp/search/trail.hpp"

namespace milp {

// A conflict constraint sum_k val[k] * x[idx[k]] <= rhs (indices sorted), or,
// if `upper` is nonempty, a clause over integer bound literals: not all of
// (upper[k] ? x[idx[k]] <= val[k] : x[idx[k]] >= val[k]) hold (rhs unused).
struct Conflict {
  std::vector<Index> idx;
  std::vector<double> val;
  std::vector<char> upper;
  double rhs = 0.0;
  double cutoff = kInf;  // objective cutoff it relies on (kInf: none)
  int owner = -1;        // worker that found it
  bool is_clause() const { return !upper.empty(); }
  // Whether point x violates it (by more than tol in the linear case).
  bool violated_by(const std::vector<double>& x, double tol = 1e-6) const;
};

struct ConflictOptions {
  Index max_nnz = 1 << 30;
  bool relax_ints = false;      // replace all variables at their global bound (sparsest proof)
  double min_violation = 1e-6;  // the proof must exceed the rhs by this (relative) amount
};

// Derives a conflict from multipliers mu of the rows of `lp` (all rows,
// including cut rows, must be valid for the solutions of interest) and an
// objective weight lambda >= 0 (with the objective bound c^T x <= cutoff):
//
//   (lambda c + sum_i mu_i a_i) x <= lambda cutoff + sum_i (mu_i > 0 ? mu_i row_ub_i : mu_i row_lb_i).
//
// Continuous variables whose local bound in `lp` equals their global bound are
// replaced by that bound; the result is cleaned like a cut against the global
// bounds [glb, gub]. Returns false unless the constraint is infeasible under the
// local column bounds of `lp` (i.e. it really proves the node can be pruned).
bool derive_conflict(const DualSimplex& lp, const std::vector<double>& mu, double lambda, const std::vector<double>& cost,
                     double cutoff, const std::vector<char>& is_int, const std::vector<double>& glb,
                     const std::vector<double>& gub, const ConflictOptions& opt, Conflict& out);

// Proof for an infeasible LP from its Farkas multipliers (tries both signs).
bool infeasibility_conflict(const DualSimplex& lp, const std::vector<double>& y, const std::vector<double>& cost,
                            const std::vector<char>& is_int, const std::vector<double>& glb,
                            const std::vector<double>& gub, const ConflictOptions& opt, Conflict& out);

// Activity-based propagation of a single conflict over bounds [lb, ub]:
// tightens integer bounds in place (appending tightened columns to `tightened`
// if given) and returns false if the conflict is infeasible.
bool propagate_conflict(const Conflict& c, const std::vector<char>& is_int, std::vector<double>& lb,
                        std::vector<double>& ub, std::vector<Index>* tightened, bool* changed,
                        BoundTrail* trail = nullptr,
                        double idle_room = kInf);

// Conflict analysis of a node proved infeasible by domain propagation (the
// failure recorded in `trail`): walks the implication graph (reasons: rows of
// `rows` with sides [row_lb, row_ub], and conflicts) backwards and appends to
// `out` up to max_clauses clauses "not all of these bounds": the first-UIP
// cut of the deepest decision level, then those of the next lower levels.
// Bounds are relaxed towards the global bounds [glb, gub] as far as the
// reasons allow. Clauses with a literal on a continuous column or more than
// max_lits literals are skipped (first_only: the analysis then ends instead of going on to the lower levels).
// minimize: the first clause is minimized (literals implied by its other literals are dropped). stats (two
// counters, or null): the entries read and the number of dropped literals are added to stats[0] and [1].
// Returns the number of clauses appended.
int analyze_propagation_conflict(const BoundTrail& trail, const SparseMatrix& rows, const std::vector<double>& row_lb,
                                 const std::vector<double>& row_ub, const std::vector<char>& is_int,
                                 const std::vector<double>& glb, const std::vector<double>& gub, Index max_lits,
                                 int max_clauses, std::vector<Conflict>& out, bool first_only = false,
                                 long long* stats = nullptr, bool minimize = false);

// Per-worker pool of conflicts with column occurrence lists for propagation.
// Bounded size: conflicts that have not propagated anything for the longest
// time are dropped first.
class ConflictPool {
 public:
  ConflictPool(Index num_cols, size_t capacity)
      : occ_(2 * static_cast<size_t>(num_cols)), wocc_(num_cols), capacity_(capacity) {}

  void add(std::shared_ptr<const Conflict> c, long long now);
  // Propagates the conflicts that contain a seed column (and, transitively,
  // columns they tighten). Returns false if some conflict is infeasible.
  // Columns whose bounds equal the global bounds [glb, gub] do not trigger propagation.
  bool propagate(const std::vector<char>& is_int, const std::vector<double>& glb, const std::vector<double>& gub,
                 std::vector<double>& lb, std::vector<double>& ub, const std::vector<Index>& seeds,
                 std::vector<Index>& tightened, long long now, BoundTrail* trail = nullptr);
  size_t size() const { return alive_; }
  long long work_limit = 0;         // linear conflicts: nonzeros touched per call (0 = 2e5, < 0: skip them)
  long long clause_work_limit = 0;  // clauses: literals touched per call (0 = 2e5, < 0: skip them)
  long long last_work() const { return last_work_; }
  long long last_clause_work() const { return last_clause_work_; }
  long long last_clause_hits() const { return last_clause_hits_; }  // bounds implied / failures by clauses
  const Conflict* culprit() const { return culprit_; }  // conflict that failed in the last propagate()

 private:
  struct Entry {
    std::shared_ptr<const Conflict> c;
    long long last_used = 0;
    int w0 = 0, w1 = 0;        // clauses: positions of the watched literals
    Index wk0 = -1, wk1 = -1;  // and their bound keys 2 * column + (literal is x <= v)
    double idle_room = -1.0;   // linear: the slack from which nothing can be tightened (-1: not computed yet)
  };
  void shrink();
  void add_occ(Index id);
  bool watches(const Entry& e, Index j, int bit) const;
  bool propagate_watched(Index id, std::vector<double>& lb, std::vector<double>& ub, std::vector<Index>& tightened,
                         bool& changed, BoundTrail* trail, long long& work);
  std::vector<Entry> entries_;  // null c: removed
  // Linear conflicts: at 2 * column the entries triggered by its lower bound, at 2 * column + 1 by its upper.
  std::vector<std::vector<Index>> occ_;
  std::vector<std::vector<Index>> wocc_;  // clause watches, per column: 2 * entry + (literal is x <= v)
  std::vector<char> queued_;
  std::vector<Index> queue_;
  size_t capacity_ = 0, alive_ = 0;
  long long last_work_ = 0, last_clause_work_ = 0, last_clause_hits_ = 0;
  const Conflict* culprit_ = nullptr;
};

}  // namespace milp
