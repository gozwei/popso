// Gomory mixed-integer cuts from simplex tableau rows.
//
// The tableau row of a basic integer variable x_B reads
//   x_B + sum_{j nonbasic} alpha_j x_j = 0      (over structurals and logicals).
// Nonbasic variables are shifted to their active bound (complemented at upper),
// y_j = x_j - l_j or y_j = u_j - x_j >= 0, giving x_B + sum a_j y_j = beta, from
// which the GMI cut sum g_j y_j >= 1 follows. Logicals s_i = -a_i x are then
// substituted back to obtain a cut over structural variables only.

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "milp/cuts/cut.hpp"

namespace milp {

namespace {

class GomorySeparator : public Separator {
 public:
  const char* name() const override { return "gomory"; }
  void separate(const SepaContext& ctx, std::vector<Cut>& out) override;
  long long work() const override { return work_; }

 private:
  long long work_ = 0;
  bool all_dense_ = false;  // the last call's tableau rows gave no cut and some were too dense
  std::vector<double> alpha_, coef_;
  std::vector<Index> anz_;  // nonzero positions of alpha_ (ascending)
  std::vector<char> int_row_, mark_;
};

bool near_int(double v, double tol) { return std::abs(v - std::round(v)) <= tol; }

// Integer, or implied integer (MipProblem::implied_int: integral in some optimal solution) with integral
// global bounds.
bool int_col(const MipProblem& p, Index j) {
  if (p.is_int[j]) return true;
  auto ib = [](double b) { return !is_finite(b) || b == std::round(b); };
  return p.is_implied_int(j) && ib(p.col_lb[j]) && ib(p.col_ub[j]);
}

void GomorySeparator::separate(const SepaContext& ctx, std::vector<Cut>& out) {
  const MipProblem& p = ctx.p;
  DualSimplex& lp = ctx.lp;
  const Index n = lp.num_cols(), m = lp.num_rows();
  const SparseMatrix& R = lp.scaled_row_matrix();  // entry * row_unscale(row): the model's coefficient, exactly
  const double* x = ctx.x;
  const size_t out_size0 = out.size();
  work_ = R.nnz();  // the pass below

  // Rows whose logical is integer for integer x: integer coefficients on integer columns.
  int_row_.assign(m, 1);
  for (Index i = 0; i < m; ++i) {
    const double us = lp.row_unscale(i);
    for (Index k = R.begin(i); k < R.end(i) && int_row_[i]; ++k) {
      const double a = R.value[k] * us;
      int_row_[i] = int_col(p, R.index[k]) && a == std::round(a);
    }
  }

  // Candidate rows: fractional basic integer structurals, most fractional first.
  std::vector<std::pair<double, Index>> cand;
  for (Index j : lp.basic_vars()) {
    if (j >= n || !p.is_int[j]) continue;
    const double f = x[j] - std::floor(x[j]);
    if (f < 0.01 || f > 0.99) continue;
    cand.emplace_back(std::abs(f - 0.5), j);
  }
  std::sort(cand.begin(), cand.end());
  const size_t max_rows = 500;
  if (cand.size() > max_rows) cand.resize(max_rows);

  coef_.assign(n, 0.0);
  mark_.assign(n, 0);
  alpha_.assign(n + m, 0.0);
  anz_.clear();
  // Rows with more than twice the pool's density limit in nonbasic structurals give cuts the pool
  // rejects (cancellation through the logicals is rare): skip them before the second pass
  // (neos-960392: ~40% of the rows have 50k+ of 58k columns).
  const Index max_struct = ctx.max_nnz > 0 ? 2 * ctx.max_nnz : n + 1;
  // Tableau entries below 1e-11 are left out of the row. That is not a relaxation: the GMI term
  // g_j y_j >= 0 of a left-out y_j is missing from the >= cut, and a nonbasic logical of a row with a
  // huge side (neos-4413714-turia: rhs 9e8) or a one-sided row has an (almost) unbounded y_j. The node
  // tableau c-MIR cuts had the same drop and cut off turia's optimum (cuts/mir.cpp). So they are kept
  // (their tiny g_j leave in clean_cut with their bounds); MILP_GOMORY_KEEP_TINY=0: the old drop. Server
  // core A/B (4 seeds): 0.987 [0.951, 1.021], solved 210 -> 215 of 252.
  static const double kZero = [] {
    const char* v = std::getenv("MILP_GOMORY_KEEP_TINY");
    return v && std::atoi(v) == 0 ? 1e-11 : 0.0;
  }();
  // Once rows come out too dense, the basis inverse usually is (dense cut rows in the basis). After a call
  // whose rows gave no cut and were partly too dense, the next call skips the rest of its candidates after
  // dense_stop rows in a row that were too dense. Each costs a pass over the rows its B^{-1} row touches and
  // a full pass over the tableau row (neos-4413714-turia from round 5: all 500 rows of 190k columns too
  // dense, 0.5 s per round for nothing; fast0507 likewise). Not in a call after one with cuts: the most
  // fractional rows can be dense while later ones give cuts (cmflsp50-24-8-8's first round: 80 cuts). Nor
  // before the call's work reaches 1e7 units (as the cut loop's separator backoff; ~10 ms): on small models
  // the rows are cheap and stopping only changes the cuts (rococoC10-001000 lost a late bound jump).
  // MILP_GOMORY_DENSE_STOP = k (default 20, 0: off).
  static const int dense_stop = [] {
    const char* v = std::getenv("MILP_GOMORY_DENSE_STOP");
    return v ? std::atoi(v) : 20;
  }();
  int dense_run = 0, dense_rows = 0;
  const bool may_stop = dense_stop > 0 && all_dense_;
  std::vector<Index> nz;
  for (const auto& [score, jb] : cand) {
    if (may_stop && dense_run >= dense_stop && work_ >= 10000000) break;
    for (Index j : anz_) alpha_[j] = 0.0;
    anz_.clear();
    if (!lp.tableau_row(jb, alpha_, anz_)) continue;
    work_ += lp.tableau_row_work();
    // First pass: beta and validity of the row (positions outside anz_ are zero).
    double beta = 0.0, amax = 0.0;
    bool ok = true;
    Index nstruct = 0;
    for (Index j : anz_) {
      if (lp.status(j) == VarStatus::Basic) continue;
      const double a = alpha_[j];
      if (std::abs(a) < kZero || a == 0.0) continue;  // numerically zero tableau entry
      amax = std::max(amax, std::abs(a));
      const VarStatus st = lp.status(j);
      const double l = lp.var_lower(j), u = lp.var_upper(j);
      if (l == u) {
        beta -= a * l;
        continue;
      }
      if (j < n && ++nstruct > max_struct) {
        ok = false;  // too dense
        break;
      }
      if (st == VarStatus::AtZero) {
        ok = false;  // free nonbasic variable
        break;
      }
      const double b = st == VarStatus::AtUpper ? u : l;
      if (!is_finite(b)) ok = false;
      beta -= a * b;
    }
    dense_run = !ok && nstruct > max_struct ? dense_run + 1 : 0;
    dense_rows += !ok && nstruct > max_struct;
    if (!ok || amax > 1e6) continue;
    if (std::abs(beta - x[jb]) > 1e-6 * (1.0 + std::abs(x[jb]))) continue;
    const double f0 = beta - std::floor(beta);
    if (f0 < 0.01 || f0 > 0.99) continue;

    // Second pass: GMI coefficients g_j for y_j, accumulated in x space.
    //   sum_j g_j y_j >= 1,  y_j = sign_j (x_j - bound_j).
    double rhs = 1.0;  // of the >= cut in x space
    auto add = [&](Index j, double v) {
      if (!mark_[j]) mark_[j] = 1, nz.push_back(j);
      coef_[j] += v;
    };
    for (Index j : anz_) {
      if (lp.status(j) == VarStatus::Basic) continue;
      const double a0 = alpha_[j];
      if (std::abs(a0) < kZero || a0 == 0.0) continue;
      const double l = lp.var_lower(j), u = lp.var_upper(j);
      if (l == u) continue;
      const bool at_upper = lp.status(j) == VarStatus::AtUpper;
      const double sign = at_upper ? -1.0 : 1.0;
      const double bound = at_upper ? u : l;
      const double a = sign * a0;  // coefficient of y_j
      bool integer;
      if (j < n) {
        integer = p.is_int[j] || (p.is_implied_int(j) && bound == std::round(bound));
      } else {
        integer = int_row_[j - n] && near_int(bound, 1e-9);
      }
      double g;
      if (integer) {
        const double fj = a - std::floor(a);
        g = fj <= f0 ? fj / f0 : (1.0 - fj) / (1.0 - f0);
      } else {
        g = a > 0 ? a / f0 : -a / (1.0 - f0);
      }
      if (g == 0.0) continue;
      // g * y_j = g * sign * x_j - g * sign * bound.
      rhs += g * sign * bound;
      const double gx = g * sign;
      if (j < n) {
        add(j, gx);
      } else {
        // x_j = s_i = -a_i x.
        const Index i = j - n;
        const double us = lp.row_unscale(i);
        for (Index k = R.begin(i); k < R.end(i); ++k) add(R.index[k], -gx * (R.value[k] * us));
      }
    }
    Cut c;
    c.origin = "gomory";
    for (Index k : nz) {
      c.idx.push_back(k);
      c.val.push_back(-coef_[k]);
      coef_[k] = 0.0, mark_[k] = 0;
    }
    nz.clear();
    c.rhs = -rhs;
    out.push_back(std::move(c));
  }
  all_dense_ = out.size() == out_size0 && dense_rows > 0;
}

}  // namespace

std::unique_ptr<Separator> make_gomory_separator() { return std::make_unique<GomorySeparator>(); }

}  // namespace milp
