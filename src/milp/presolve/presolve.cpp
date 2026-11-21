#include "milp/presolve/presolve.hpp"

#include "milp/core/interrupt.hpp"
#include "milp/presolve/clique_merge.hpp"
#include "milp/presolve/group_prop.hpp"
#include "milp/presolve/implied_int.hpp"
#include "milp/presolve/probing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>

#include "milp/core/timer.hpp"

namespace milp {

namespace {

constexpr double kBig = 1e12;        // bounds beyond this count as infinite in activities
constexpr double kMaxContBound = 1e9;  // never derive continuous bounds larger than this
constexpr double kRedundantTol = 1e-9;
constexpr Index kMaxSubstRow = 1000;      // longest equation used for non-singleton substitution
constexpr Index kMaxSubstCol = 8;         // most rows of a substituted non-singleton column
constexpr long long kMaxSubstFill = 32;   // (row length - 1) * (column length - 1) limit
constexpr double kMaxImpliedRatio = 10.0;  // |a_ij| / |a_rj| limit for a bound of x_j implied by row r
constexpr Index kMaxSparsifyRow = 50;     // longest equation added to other rows by sparsify
constexpr double kMaxSparsifyScale = 1e3;  // largest |lambda| in r += lambda e
constexpr double kMaxSparsifyRange = 1e6;  // a row's coefficient range may not grow beyond this

struct Entry {
  Index idx;
  double val;
};

inline bool small_bound(double v) { return std::abs(v) < kBig; }

}  // namespace

// Working copy of the problem with row- and column-wise storage. Removed
// rows/columns are only flagged; their entries are purged lazily (compact()).
class PresolveWork {
 public:
  PresolveWork(const MipProblem& in, const PresolveOptions& opt, const Logger& log, Presolver& out)
      : opt_(opt), log_(log), out_(out), stats_(out.stats_) {
    m_ = in.num_rows();
    n_ = in.num_cols();
    rows_.resize(m_);
    cols_.resize(n_);
    {
      // Exact sizes first: grown by push_back the lists end up with up to twice the room they need (16 bytes per
      // nonzero on each side: square47, 27M nonzeros, 0.9 GB exact).
      std::vector<Index> len(m_, 0);
      for (Index j = 0; j < n_; ++j) {
        Index cnt = 0;
        for (Index p = in.A.begin(j); p < in.A.end(j); ++p) {
          if (in.A.value[p] == 0.0) continue;
          ++cnt, ++len[in.A.index[p]];
        }
        cols_[j].reserve(static_cast<size_t>(cnt));
      }
      for (Index i = 0; i < m_; ++i) rows_[i].reserve(static_cast<size_t>(len[i]));
    }
    for (Index j = 0; j < n_; ++j) {
      for (Index p = in.A.begin(j); p < in.A.end(j); ++p) {
        if (in.A.value[p] == 0.0) continue;
        cols_[j].push_back({in.A.index[p], in.A.value[p]});
        rows_[in.A.index[p]].push_back({j, in.A.value[p]});
      }
    }
    row_alive_.assign(m_, 1);
    col_alive_.assign(n_, 1);
    merged_.assign(n_, 0);
    row_size_.resize(m_);
    col_size_.resize(n_);
    for (Index i = 0; i < m_; ++i) row_size_[i] = static_cast<Index>(rows_[i].size());
    for (Index j = 0; j < n_; ++j) col_size_[j] = static_cast<Index>(cols_[j].size());
    lb_ = in.col_lb;
    ub_ = in.col_ub;
    cost_ = in.cost;
    rlb_ = in.row_lb;
    rub_ = in.row_ub;
    is_int_ = in.is_int;
    offset_ = in.offset;
    display_sense_ = in.display_sense;
  }

  PresolveStatus run() {
    for (Index j = 0; j < n_; ++j) {
      if (is_int_[j]) {
        lb_[j] = std::ceil(lb_[j] - 1e-6);
        ub_[j] = std::floor(ub_[j] + 1e-6);
      }
      if (lb_[j] > ub_[j]) return PresolveStatus::Infeasible;
    }
    for (Index i = 0; i < m_; ++i) {
      if (rlb_[i] > rub_[i]) return PresolveStatus::Infeasible;
    }
    const int max_passes = opt_.probing ? std::max(1, opt_.probing_passes) : 0;
    bool probe_pending = max_passes > 0;  // another probing pass is due
    keep_binaries_ = probe_pending && opt_.probing_keep_binaries;
    defer_fill_ = probe_pending;
    int clique_runs = 0;
    long long since_clique = 0;
    long long since_group = 0;
    // Sparsify whenever the other reductions are exhausted and something changed since its last run, at most
    // kSparsifyRuns times; up to 10 more rounds after a run that changed something (also beyond max_rounds).
    constexpr int kSparsifyRuns = 3;
    long long since_sparsify = 0;
    int sparsify_until = 0;
    for (int round = 0;
         round < opt_.max_rounds + 10 * std::max(1, max_passes) + 10 * kSparsifyRuns && status_ == PresolveStatus::Reduced;
         ++round) {
      if (round >= opt_.max_rounds && !probe_pending && round >= sparsify_until) break;
      if (halted()) break;  // the reductions so far are valid
      ++stats_.rounds;
      compact();
      long long ch = 0;
      // Each step is timed (verbosity 3 prints the totals).
      auto step = [&](int k, bool on, auto&& f) {
        if (!on || status_ != PresolveStatus::Reduced || halted()) return;
        Timer t;
        ch += f();
        step_time_[k] += t.elapsed();
      };
      step(0, true, [&] { return column_pass(); });
      step(1, true, [&] { return row_pass(true); });
      step(2, opt_.bound_tightening, [&] { return propagate(); });
      step(0, true, [&] { return column_pass(); });
      step(3, opt_.coef_tightening, [&] { return coef_tightening(); });
      step(3, opt_.int_divisors, [&] { return int_divisors(); });
      step(3, opt_.unit_rows, [&] { return unit_rows(); });
      step(4, opt_.dual_fixing, [&] { return dual_fixing(); });
      step(13, opt_.dual_aggregation, [&] { return dual_aggregate(); });
      step(5, opt_.dominated_cols, [&] { return dominated_cols(); });
      step(6, opt_.parallel_rows, [&] { return parallel_rows(); });
      step(7, opt_.parallel_cols, [&] { return parallel_cols(); });
      step(8, opt_.aggregation, [&] { return aggregate(); });
      step(9, opt_.substitution, [&] { return substitute_equations(); });
      log_.log(3, "Presolve round %d: %lld changes\n", round + 1, ch);
      if (halted()) break;
      since_clique += ch;
      since_sparsify += ch;
      since_group += ch;
      const bool last = round + 1 >= opt_.max_rounds;
      if ((ch == 0 || last) && opt_.sparsify && stats_.sparsify_runs < kSparsifyRuns &&
          (stats_.sparsify_runs == 0 || since_sparsify > 0) && status_ == PresolveStatus::Reduced) {
        // Before the clique extension and probing: probing on the sparser rows finds more (app1-1: 1.2% -> 9.4% of
        // the root gap closed, 5.5% when sparsify runs after probing). The rounds after it remove the singleton and
        // doubleton rows it creates.
        since_sparsify = 0;
        compact();
        Timer ts;
        ch = sparsify();
        step_time_[12] += ts.elapsed();
        if (ch > 0) {
          sparsify_until = round + 11;
          continue;
        }
      }
      if ((ch == 0 || last) && opt_.group_prop && status_ == PresolveStatus::Reduced && stats_.group_runs < 3 &&
          (stats_.group_runs == 0 || since_group > 0)) {
        // One-of group propagation once the cheap reductions are exhausted, before the clique extension and
        // probing (its fixings shrink both), and again after rounds that changed something.
        ++stats_.group_runs;
        since_group = 0;
        compact();
        Timer tg;
        ch = group_prop();
        step_time_[14] += tg.elapsed();
        since_sparsify += ch;
        since_clique += ch;
        if (ch > 0 && !last) continue;
      }
      if ((ch == 0 || last) && opt_.clique_extension && status_ == PresolveStatus::Reduced && clique_runs < 4 &&
          (clique_runs == 0 || since_clique > 0) && (stats_.probing_passes <= 1 || opt_.probing_cliques)) {
        // Clique extension whenever the other reductions are exhausted (and something changed since its last run):
        // its fixings and dominated rows feed further rounds. Before probing and after its first pass, not after the
        // later ones (the few changes they cause would rerun it each time: cod105 after the root restart 4 runs,
        // 1.2 s, for 0.5 s of probing).
        ++clique_runs;
        since_clique = 0;
        compact();
        Timer tc;
        ch = clique_extension();
        step_time_[11] += tc.elapsed();
        since_sparsify += ch;
        if (ch > 0 && !last) continue;
      }
      if ((ch == 0 || last) && probe_pending && status_ == PresolveStatus::Reduced) {
        // Probing once the cheap reductions are (nearly) exhausted, then a few more rounds; again while a pass
        // fixes, substitutes or lifts something, with half the previous work limit and at most the work the
        // previous pass used (puriri: pass 2 would use 2.5x the work of pass 1).
        const int pass = ++stats_.probing_passes;
        keep_binaries_ = false;
        defer_fill_ = false;
        compact();
        Timer tp;
        long long useful = 0;
        ch = probing(opt_.probing_effort * std::ldexp(1.0, 1 - pass), &useful);
        // Substitutions deferred until now (binaries, fill-in): at least one more round.
        if (deferred_ > 0 && status_ == PresolveStatus::Reduced) ch += 1, deferred_ = 0;
        step_time_[10] += tp.elapsed();
        probe_pending = pass < max_passes && useful > 0;
        // With new conflicts in its graph (opt-in) the clique extension runs again.
        if (opt_.probing_cliques) since_clique += new_conflicts_;
        log_.log(3, "Presolve probing pass %d: %lld changes (%lld fixings / equivalences / liftings), work %lld / %lld, %.3fs\n",
                 pass, ch, useful, last_probe_work_, last_probe_limit_, tp.elapsed());
        since_sparsify += ch;
      }
      if (ch == 0) break;
    }
    if (log_.verbosity() >= 3) {
      static const char* names[] = {"columns", "rows", "propagate", "coef", "dual", "dominated", "parallel_rows",
                                    "parallel_cols", "aggregate", "substitute", "probing", "cliqueext", "sparsify",
                                    "dualagg", "groups"};
      std::string line = "Presolve step times:";
      for (int k = 0; k < 15; ++k) line += " " + std::string(names[k]) + " " + std::to_string(step_time_[k]).substr(0, 5);
      log_.log(3, "%s\n", line.c_str());
    }
    if (status_ == PresolveStatus::Reduced) column_pass();
    if (status_ == PresolveStatus::Reduced) row_pass(false);
    if (status_ == PresolveStatus::Reduced) column_pass();
    if (status_ != PresolveStatus::Reduced) return status_;
    build();
    return status_;
  }

 private:
  // Sub-MIP presolves stop early (PresolveOptions::stop / time_limit), all of them on an interrupt.
  bool halted() const {
    return interrupted() || (opt_.stop && opt_.stop->load(std::memory_order_relaxed)) ||
           (opt_.time_limit < kInf && run_timer_.elapsed() > opt_.time_limit);
  }

  // ---- structural helpers -------------------------------------------------

  void compact() {
    for (Index i = 0; i < m_; ++i) {
      auto& r = rows_[i];
      if (!row_alive_[i]) {
        r.clear();
        r.shrink_to_fit();
        continue;
      }
      r.erase(std::remove_if(r.begin(), r.end(), [&](const Entry& e) { return !col_alive_[e.idx]; }), r.end());
    }
    for (Index j = 0; j < n_; ++j) {
      auto& c = cols_[j];
      if (!col_alive_[j]) {
        c.clear();
        c.shrink_to_fit();
        continue;
      }
      c.erase(std::remove_if(c.begin(), c.end(), [&](const Entry& e) { return !row_alive_[e.idx]; }), c.end());
    }
  }

  void remove_row(Index i) {
    row_alive_[i] = 0;
    ++stats_.removed_rows;
    for (const Entry& e : rows_[i]) {
      if (col_alive_[e.idx]) --col_size_[e.idx];
    }
  }

  // Removes column j at value v, moving its contribution into row bounds and offset.
  void fix_col(Index j, double v) {
    col_alive_[j] = 0;
    ++stats_.fixed_cols;
    offset_ += cost_[j] * v;
    for (const Entry& e : cols_[j]) {
      const Index i = e.idx;
      if (!row_alive_[i]) continue;
      const double d = e.val * v;
      if (std::isfinite(rlb_[i])) rlb_[i] -= d;
      if (std::isfinite(rub_[i])) rub_[i] -= d;
      if (rlb_[i] > rub_[i] && rlb_[i] - rub_[i] < 1e-12 * std::max(1.0, std::abs(rub_[i]))) rlb_[i] = rub_[i];
      --row_size_[i];
    }
    out_.stack_.push_back({j, v, 1.0, {}});
  }

  // Sets A(i, j) = v in both storages (inserting or erasing as needed). Row i and column j must be alive.
  void set_coef(Index i, Index j, double v) {
    auto& r = rows_[i];
    auto& c = cols_[j];
    auto rit = std::find_if(r.begin(), r.end(), [&](const Entry& e) { return e.idx == j; });
    auto cit = std::find_if(c.begin(), c.end(), [&](const Entry& e) { return e.idx == i; });
    const bool present = rit != r.end();
    if (v == 0.0) {
      if (!present) return;
      r.erase(rit);
      c.erase(cit);
      --row_size_[i];
      --col_size_[j];
    } else if (present) {
      rit->val = v;
      cit->val = v;
    } else {
      r.push_back({j, v});
      c.push_back({i, v});
      ++row_size_[i];
      ++col_size_[j];
    }
  }

  // ---- bound changes --------------------------------------------------------

  // Tightens bounds of column j to [l, u] (exact implications). xtol: tolerance in x space
  // below which an empty domain is treated as a touching one. Returns number of changes.
  int tighten(Index j, double l, double u, double xtol) {
    int ch = 0;
    if (log_.verbosity() >= 4) log_.log(4, "  col %d [%.12g, %.12g] tighten to [%.12g, %.12g]\n", j, lb_[j], ub_[j], l, u);
    if (l > lb_[j]) {
      lb_[j] = l;
      ++ch;
    }
    if (u < ub_[j]) {
      ub_[j] = u;
      ++ch;
    }
    if (lb_[j] > ub_[j]) {
      if (is_int_[j] || lb_[j] - ub_[j] > xtol) {
        status_ = PresolveStatus::Infeasible;
        log_.log(3, "Presolve: column %d has empty domain [%g, %g]\n", j, lb_[j], ub_[j]);
        return ch;
      }
      // Within tolerance: keep whichever side is the original column bound.
      if (l > u) {
        lb_[j] = ub_[j] = (l + u) * 0.5;
      } else if (l == lb_[j]) {
        lb_[j] = ub_[j];
      } else {
        ub_[j] = lb_[j];
      }
    }
    stats_.bound_changes += ch;
    return ch;
  }

  // v relaxed upwards by at most margin, preferring a nearby integer (keeps data clean).
  static double relax_up(double v, double margin) {
    const double r = std::round(v);
    return r >= v && r - v <= margin ? r : v + margin;
  }

  // Applies an implied upper bound v (from a kept row): rounded for integers, only
  // significant improvements for continuous columns. eps: safety margin (generous for
  // integers; for continuous columns just above rounding error, since an LP solution
  // sitting on a relaxed bound would violate the row that implied it).
  int implied_ub(Index j, double v, double eps, double xtol) {
    if (is_int_[j]) {
      if (!(v < kBig)) return 0;
      v = std::floor(v + eps);
      if (v > ub_[j] - 0.5) return 0;
      return tighten(j, -kInf, v, xtol);
    }
    v = relax_up(v, eps + 1e-11 * std::abs(v));
    if (!(std::abs(v) <= kMaxContBound)) return 0;
    if (std::isfinite(ub_[j]) && ub_[j] - v <= 1e-3 * std::max(1.0, std::abs(ub_[j]))) return 0;
    return tighten(j, -kInf, v, xtol);
  }

  int implied_lb(Index j, double v, double eps, double xtol) {
    if (is_int_[j]) {
      if (!(v > -kBig)) return 0;
      v = std::ceil(v - eps);
      if (v < lb_[j] + 0.5) return 0;
      return tighten(j, v, kInf, xtol);
    }
    v = -relax_up(-v, eps + 1e-11 * std::abs(v));
    if (!(std::abs(v) <= kMaxContBound)) return 0;
    if (std::isfinite(lb_[j]) && v - lb_[j] <= 1e-3 * std::max(1.0, std::abs(lb_[j]))) return 0;
    return tighten(j, v, kInf, xtol);
  }

  // ---- reductions -----------------------------------------------------------

  // Fixed columns and empty columns.
  long long column_pass() {
    long long ch = 0;
    for (Index j = 0; j < n_ && status_ == PresolveStatus::Reduced; ++j) {
      if (!col_alive_[j]) continue;
      if (lb_[j] == ub_[j]) {
        fix_col(j, lb_[j]);
        ++ch;
      } else if (col_size_[j] == 0) {
        double v;
        if (cost_[j] > 0.0) {
          v = lb_[j];
        } else if (cost_[j] < 0.0) {
          v = ub_[j];
        } else {
          v = std::min(std::max(0.0, lb_[j]), ub_[j]);
        }
        if (!std::isfinite(v)) {
          status_ = PresolveStatus::Unbounded;
          log_.log(3, "Presolve: empty column %d is unbounded\n", j);
          return ch;
        }
        fix_col(j, v);
        ++ch;
      }
    }
    return ch;
  }

  // Empty rows and (optionally) singleton rows.
  long long row_pass(bool singletons) {
    long long ch = 0;
    for (Index i = 0; i < m_ && status_ == PresolveStatus::Reduced; ++i) {
      if (!row_alive_[i]) continue;
      if (row_size_[i] == 0) {
        const double tol = opt_.feas_tol;
        if (rlb_[i] > tol || rub_[i] < -tol) {
          status_ = PresolveStatus::Infeasible;
          log_.log(3, "Presolve: empty row %d infeasible [%g, %g]\n", i, rlb_[i], rub_[i]);
          return ch;
        }
        remove_row(i);
        ++ch;
      } else if (singletons && row_size_[i] == 1) {
        Index j = -1;
        double a = 0.0;
        for (const Entry& e : rows_[i]) {
          if (col_alive_[e.idx]) {
            j = e.idx;
            a = e.val;
            break;
          }
        }
        // A continuous column with a tiny coefficient: the bound would be numerical noise; keep the row.
        if (!is_int_[j] && std::abs(a) < 1e-7) continue;
        double l = a > 0 ? rlb_[i] / a : rub_[i] / a;
        double u = a > 0 ? rub_[i] / a : rlb_[i] / a;
        // The row tolerates violations up to feas_tol, i.e. feas_tol / |a| in x.
        const double xtol = opt_.feas_tol / std::abs(a);
        if (is_int_[j]) {
          l = std::ceil(l - xtol - 1e-12 * std::abs(l));
          u = std::floor(u + xtol + 1e-12 * std::abs(u));
        }
        if (log_.verbosity() >= 4) log_.log(4, "  singleton row %d: %.17g * col %d in [%.17g, %.17g]\n", i, a, j, rlb_[i], rub_[i]);
        tighten(j, l, u, xtol);
        remove_row(i);
        ++ch;
      }
    }
    return ch;
  }

  // Activity-based bound tightening, redundant and forcing rows.
  long long propagate() {
    long long ch = 0;
    std::vector<double> lo, hi;  // per-entry min/max contribution (snapshot)
    std::vector<char> lo_inf, hi_inf;
    for (Index i = 0; i < m_ && status_ == PresolveStatus::Reduced; ++i) {
      if (!row_alive_[i] || row_size_[i] == 0) continue;
      const auto& r = rows_[i];
      const size_t len = r.size();
      lo.assign(len, 0.0);
      hi.assign(len, 0.0);
      lo_inf.assign(len, 0);
      hi_inf.assign(len, 0);
      double minact = 0.0, maxact = 0.0, mag = 0.0;
      int nmin_inf = 0, nmax_inf = 0;
      for (size_t k = 0; k < len; ++k) {
        const Index j = r[k].idx;
        if (!col_alive_[j]) continue;
        const double a = r[k].val;
        const double bl = a > 0 ? lb_[j] : ub_[j];
        const double bh = a > 0 ? ub_[j] : lb_[j];
        if (small_bound(bl)) {
          lo[k] = a * bl;
          minact += lo[k];
          mag += std::abs(lo[k]);
        } else {
          lo_inf[k] = 1;
          ++nmin_inf;
        }
        if (small_bound(bh)) {
          hi[k] = a * bh;
          maxact += hi[k];
          mag += std::abs(hi[k]);
        } else {
          hi_inf[k] = 1;
          ++nmax_inf;
        }
      }
      const bool has_lb = std::isfinite(rlb_[i]), has_ub = std::isfinite(rub_[i]);
      double scale = 1.0;
      if (has_lb) scale = std::max(scale, std::abs(rlb_[i]));
      if (has_ub) scale = std::max(scale, std::abs(rub_[i]));
      const double tol = opt_.feas_tol * scale + 1e-12 * mag;
      if ((has_ub && nmin_inf == 0 && minact > rub_[i] + tol) || (has_lb && nmax_inf == 0 && maxact < rlb_[i] - tol)) {
        status_ = PresolveStatus::Infeasible;
        log_.log(3, "Presolve: row %d infeasible (activity [%g, %g], bounds [%g, %g])\n", i, minact, maxact, rlb_[i],
                 rub_[i]);
        return ch;
      }
      const double rtol = kRedundantTol * scale;
      const bool low_red = !has_lb || (nmin_inf == 0 && minact >= rlb_[i] - rtol);
      const bool up_red = !has_ub || (nmax_inf == 0 && maxact <= rub_[i] + rtol);
      if (low_red && up_red) {
        if (log_.verbosity() >= 4) log_.log(4, "  row %d redundant\n", i);
        remove_row(i);
        ++ch;
        continue;
      }
      // Forcing rows: the only (near-)feasible point puts every column at one bound.
      const bool force_min = has_ub && nmin_inf == 0 && minact >= rub_[i] - rtol;
      const bool force_max = has_lb && nmax_inf == 0 && maxact <= rlb_[i] + rtol;
      if (force_min || force_max) {
        for (const Entry& e : r) {
          const Index j = e.idx;
          if (!col_alive_[j]) continue;
          if ((e.val > 0) == force_min) {
            if (ub_[j] != lb_[j]) ub_[j] = lb_[j], ++stats_.bound_changes;
          } else {
            if (lb_[j] != ub_[j]) lb_[j] = ub_[j], ++stats_.bound_changes;
          }
        }
        remove_row(i);
        ++ch;
        continue;
      }
      if (low_red && has_lb) {
        rlb_[i] = -kInf;
        ++ch;
      }
      if (up_red && has_ub) {
        rub_[i] = kInf;
        ++ch;
      }
      if (!opt_.bound_tightening) continue;
      const bool use_ub = std::isfinite(rub_[i]) && nmin_inf <= 1;
      const bool use_lb = std::isfinite(rlb_[i]) && nmax_inf <= 1;
      if (!use_ub && !use_lb) continue;
      if (log_.verbosity() >= 4) log_.log(4, "  propagate row %d [%.17g, %.17g] act [%.17g, %.17g] len %d\n", i, rlb_[i], rub_[i], minact, maxact, (int)len);
      for (size_t k = 0; k < len && status_ == PresolveStatus::Reduced; ++k) {
        const Index j = r[k].idx;
        if (!col_alive_[j]) continue;
        const double a = r[k].val;
        const double absa = std::abs(a);
        const double xtol = opt_.feas_tol * scale / absa;
        const double eps_fp = 1e-12 * (mag + scale) / absa;
        const double eps = is_int_[j] ? std::max(1e-6, xtol) + eps_fp : eps_fp;
        if (use_ub && (nmin_inf == 0 || lo_inf[k])) {
          // a x_j <= rub - (min activity of the others)
          const double res = nmin_inf == 0 ? minact - lo[k] : minact;
          const double v = (rub_[i] - res) / a;
          ch += a > 0 ? implied_ub(j, v, eps, xtol) : implied_lb(j, v, eps, xtol);
        }
        if (status_ != PresolveStatus::Reduced) break;
        if (use_lb && (nmax_inf == 0 || hi_inf[k])) {
          // a x_j >= rlb - (max activity of the others)
          const double res = nmax_inf == 0 ? maxact - hi[k] : maxact;
          const double v = (rlb_[i] - res) / a;
          ch += a > 0 ? implied_lb(j, v, eps, xtol) : implied_ub(j, v, eps, xtol);
        }
      }
    }
    return ch;
  }

  // GCD tightening: in a row of integer columns with integral coefficients (|a| <= 1e9), the activity is a
  // multiple of g = gcd(a): the sides are rounded to multiples of g (an equation with b not divisible by g is
  // infeasible).
  long long gcd_tightening() {
    long long ch = 0;
    for (Index i = 0; i < m_ && status_ == PresolveStatus::Reduced; ++i) {
      if (!row_alive_[i] || row_size_[i] < 1) continue;
      const bool has_lb = std::isfinite(rlb_[i]), has_ub = std::isfinite(rub_[i]);
      if (!has_lb && !has_ub) continue;
      long long g = 0;
      for (const Entry& e : rows_[i]) {
        if (!col_alive_[e.idx]) continue;
        const double a = std::abs(e.val);
        if (!is_int_[e.idx] || a > 1e9 || a != std::round(a)) {
          g = -1;
          break;
        }
        long long x = static_cast<long long>(a), y = g;
        while (y != 0) {
          const long long t = x % y;
          x = y;
          y = t;
        }
        g = x;
      }
      if (g <= 1) continue;
      const double gd = static_cast<double>(g);
      bool changed = false;
      // Also a multiple within feas_tol beyond a side replaces it: a side just short of one by round-off of earlier
      // steps (14.9999999999972 for 15) would otherwise stay while the other is rounded past it (lb 15 > ub).
      if (has_ub && std::abs(rub_[i]) < 1e12) {
        const double v = gd * std::floor((rub_[i] + opt_.feas_tol) / gd);
        if (v != rub_[i]) rub_[i] = v, changed = true;
      }
      if (has_lb && std::abs(rlb_[i]) < 1e12) {
        const double v = gd * std::ceil((rlb_[i] - opt_.feas_tol) / gd);
        if (v != rlb_[i]) rlb_[i] = v, changed = true;
      }
      if (rlb_[i] > rub_[i]) {
        status_ = PresolveStatus::Infeasible;
        log_.log(3, "Presolve: row %d has no integral solution (gcd %lld)\n", i, g);
        return ch;
      }
      if (changed) {
        ++stats_.coef_changes;
        ++ch;
      }
    }
    return ch;
  }

  // Integer columns whose values are all congruent modulo a divisor. In an equation over integer columns with
  // integral coefficients, let G be the gcd of the coefficients of the columns other than x_j: if |a_j| divides
  // G, then x_j is congruent to b / a_j modulo g = G / |a_j|. For g >= 2 the column is replaced by
  // x_j = g x' + r (x' integer, in the same place): its range shrinks by the factor g, and rows in which it
  // stands opposite other integers become divisible, which probing and the gcd rule use. Repeated: a rescaled
  // column changes the gcd of its other equations. traininstance6: the value columns of three 501-member one-of
  // groups are sums of terms 10 (a - b); as 10 x' the groups lose the 450 members that are no multiple of 10 to
  // probing (presolved 2804 x 1172 -> 1909 x 272).
  long long int_divisors() {
    constexpr double kMaxCoef = 1e9;
    constexpr long long kMaxDivisor = 1000000;
    auto gcd = [](long long x, long long y) {
      while (y != 0) {
        const long long t = x % y;
        x = y;
        y = t;
      }
      return x;
    };
    long long total = 0, work = 0;
    long long nnz = 0;
    for (Index i = 0; i < m_; ++i) nnz += row_alive_[i] ? row_size_[i] : 0;
    const long long limit = 20 * nnz + 100000;
    std::vector<long long> coef, pre;
    std::vector<Index> col;
    for (int sweep = 0; sweep < 20 && status_ == PresolveStatus::Reduced && work < limit; ++sweep) {
      long long ch = 0;
      for (Index i = 0; i < m_ && status_ == PresolveStatus::Reduced && work < limit; ++i) {
        if (!row_alive_[i] || row_size_[i] < 2 || rlb_[i] != rub_[i]) continue;
        const double b = rub_[i];
        if (std::abs(b) > 1e12 || b != std::round(b)) continue;
        coef.clear();
        col.clear();
        bool ok = true, wide = false;
        for (const Entry& e : rows_[i]) {
          if (!col_alive_[e.idx]) continue;
          const double a = std::abs(e.val);
          if (!is_int_[e.idx] || a > kMaxCoef || a != std::round(a) || a == 0.0) {
            ok = false;
            break;
          }
          coef.push_back(static_cast<long long>(e.val));
          col.push_back(e.idx);
          wide = wide || ub_[e.idx] - lb_[e.idx] > 1.5;
        }
        work += static_cast<long long>(coef.size()) + 1;
        if (!ok || !wide || coef.size() < 2) continue;
        const size_t len = coef.size();
        // pre[k]: gcd of the coefficients before k; suf: of those after k.
        pre.assign(len + 1, 0);
        for (size_t k = 0; k < len; ++k) pre[k + 1] = gcd(std::abs(coef[k]), pre[k]);
        long long suf = 0;
        for (size_t k = len; k-- > 0; suf = gcd(std::abs(coef[k]), suf)) {
          const Index j = col[k];
          if (ub_[j] - lb_[j] < 1.5) continue;  // binaries and fixed columns: nothing to gain
          const long long a = std::abs(coef[k]);
          const long long G = gcd(pre[k], suf);
          if (G == 0 || G % a != 0 || G / a < 2 || G / a > kMaxDivisor) continue;
          const long long g = G / a;
          const double q = b / static_cast<double>(coef[k]);
          if (q != std::round(q)) continue;  // no integral solution: left to the gcd rule
          long long r = static_cast<long long>(q) % g;
          if (r < 0) r += g;
          const double gd = static_cast<double>(g), rd = static_cast<double>(r);
          bool fits = std::abs(cost_[j]) * gd <= 1e12;
          for (const Entry& e : cols_[j]) {
            if (row_alive_[e.idx] && std::abs(e.val) * gd > kMaxCoef) fits = false;
          }
          if (!fits) continue;
          const double lo = std::isfinite(lb_[j]) ? std::ceil((lb_[j] - rd) / gd - 1e-9) : -kInf;
          const double hi = std::isfinite(ub_[j]) ? std::floor((ub_[j] - rd) / gd + 1e-9) : kInf;
          if (lo > hi) {
            status_ = PresolveStatus::Infeasible;
            log_.log(3, "Presolve: column %d has no value congruent to %lld modulo %lld\n", j, r, g);
            return total;
          }
          if (log_.verbosity() >= 4) {
            log_.log(4, "  col %d [%.12g, %.12g] = %lld x' + %lld (row %d), x' in [%.12g, %.12g]\n", j, lb_[j], ub_[j],
                     g, r, i, lo, hi);
          }
          for (Entry& e : cols_[j]) {
            const Index i2 = e.idx;
            if (!row_alive_[i2]) continue;
            const double d = e.val * rd;
            if (std::isfinite(rlb_[i2])) rlb_[i2] -= d;
            if (std::isfinite(rub_[i2])) rub_[i2] -= d;
            e.val *= gd;
            for (Entry& f : rows_[i2]) {
              if (f.idx == j) {
                f.val = e.val;
                break;
              }
            }
            work += row_size_[i2];
          }
          offset_ += cost_[j] * rd;
          cost_[j] *= gd;
          lb_[j] = lo;
          ub_[j] = hi;
          out_.stack_.push_back({j, rd, 1.0, {{j, -gd}}});  // x_j = r + g x'
          ++stats_.int_divisors;
          ++ch;
          break;  // the row's coefficients changed: next sweep
        }
      }
      total += ch;
      if (ch == 0) break;
    }
    return total;
  }

  // Rows whose coefficients all have one absolute value c other than 1 are divided by c: sum of +-x on one side.
  // The clique steps, the one-of groups and the clique cuts look for coefficients 1 and a side 1; coefficient
  // tightening and liftings leave rows like 0.25 (x1 + x2 + x3 + x4) <= 0.25 or 238 (x1 + ... + xk) >= 238
  // (roll3000: 83 packing rows with 0.25, 139 covering rows with 238). Rows of integer columns only, with
  // integral sides afterwards: a row with a continuous column divided by c > 1 would be held to c times the
  // feasibility tolerance in its original form (fuzz seed 58, case 205 of the sparsify models: a violation of
  // 4e-6 through a row divided by 4).
  long long unit_rows() {
    long long ch = 0;
    for (Index i = 0; i < m_ && status_ == PresolveStatus::Reduced; ++i) {
      if (!row_alive_[i] || row_size_[i] < 2) continue;
      double c = 0.0;
      bool ok = true, all_int = true;
      for (const Entry& e : rows_[i]) {
        if (!col_alive_[e.idx]) continue;
        const double a = std::abs(e.val);
        if (c == 0.0) c = a;
        if (a != c) {
          ok = false;
          break;
        }
        all_int = all_int && is_int_[e.idx];
      }
      if (!ok || !all_int || c == 0.0 || c == 1.0 || c < 1e-6 || c > 1e6) continue;
      double lo = std::isfinite(rlb_[i]) ? rlb_[i] / c : rlb_[i], hi = std::isfinite(rub_[i]) ? rub_[i] / c : rub_[i];
      if (std::isfinite(lo)) lo = std::ceil(lo - opt_.feas_tol);
      if (std::isfinite(hi)) hi = std::floor(hi + opt_.feas_tol);
      if (lo > hi) {
        status_ = PresolveStatus::Infeasible;
        log_.log(3, "Presolve: row %d has no integral solution (coefficients %g)\n", i, c);
        return ch;
      }
      for (Entry& e : rows_[i]) {
        if (!col_alive_[e.idx]) continue;
        const double v = e.val > 0 ? 1.0 : -1.0;
        e.val = v;
        for (Entry& f : cols_[e.idx]) {
          if (f.idx == i) {
            f.val = v;
            break;
          }
        }
      }
      rlb_[i] = lo;
      rub_[i] = hi;
      ++stats_.coef_changes;
      ++ch;
    }
    return ch;
  }

  // Coefficient strengthening for integer columns in one-sided rows
  // (normalized to sum a_k x_k <= b with finite max activity M):
  //   a_j > 0, M - a_j < b:  a_j -= d, b -= d u_j  with d = b - (M - a_j)
  //   a_j < 0, M + a_j < b:  a_j += d, b += d l_j  with d = b - (M + a_j)
  long long coef_tightening() {
    long long ch = gcd_tightening();
    for (Index i = 0; i < m_ && status_ == PresolveStatus::Reduced; ++i) {
      if (!row_alive_[i] || row_size_[i] < 2) continue;
      const bool has_lb = std::isfinite(rlb_[i]), has_ub = std::isfinite(rub_[i]);
      if (has_lb == has_ub) continue;
      const double s = has_ub ? 1.0 : -1.0;
      double b = has_ub ? rub_[i] : -rlb_[i];
      double maxact = 0.0, mag = std::abs(b);
      // Exact sums: integral data, or (lift_dyadic) multiples of 1/1024 with integral bounds.
      const double unit = opt_.lift_dyadic ? 1024.0 : 1.0;
      bool ok = true, any_int = false, exact = std::abs(b) < 1e9 && b * unit == std::round(b * unit);
      for (const Entry& e : rows_[i]) {
        const Index j = e.idx;
        if (!col_alive_[j]) continue;
        const double a = s * e.val;
        const double bh = a > 0 ? ub_[j] : lb_[j];
        if (!small_bound(bh) || !small_bound(lb_[j]) || !small_bound(ub_[j])) {
          ok = false;
          break;
        }
        maxact += a * bh;
        mag += std::abs(a * bh);
        any_int = any_int || is_int_[j];
        exact = exact && a * unit == std::round(a * unit) && bh == std::round(bh) && std::abs(a * bh) < 1e9;
      }
      if (!ok || !any_int) continue;
      // Margin against rounding errors in M (none when all data are moderate integers).
      const double safety = exact ? 0.0 : 1e-13 * std::max(1.0, mag) * static_cast<double>(row_size_[i]);
      if (maxact - b <= opt_.feas_tol) continue;  // (nearly) redundant
      // Collect candidates first: set_coef may reorder the row.
      std::vector<Entry> cand;
      for (const Entry& e : rows_[i]) {
        if (col_alive_[e.idx] && is_int_[e.idx]) cand.push_back(e);
      }
      bool changed = false;
      for (const Entry& e : cand) {
        const Index j = e.idx;
        const double a = s * e.val;
        const double gap = maxact - b + safety;  // > 0, preserved by each step
        const double absa = std::abs(a);
        if (absa <= gap + 1e-3 * absa + 1e-9) continue;
        const double d = absa - gap;
        if (a > 0) {
          b -= d * ub_[j];
          maxact -= d * ub_[j];
          set_coef(i, j, s * (a - d));
        } else {
          b += d * lb_[j];
          maxact += d * lb_[j];
          set_coef(i, j, s * (a + d));
        }
        ++stats_.coef_changes;
        if (log_.verbosity() >= 4) log_.log(4, "  row %d col %d coef %.12g -> %.12g\n", i, j, s * a, s * (a > 0 ? a - d : a + d));
        changed = true;
        ++ch;
      }
      if (changed) {
        if (has_ub) {
          rub_[i] = b;
        } else {
          rlb_[i] = -b;
        }
      }
    }
    return ch;
  }

  // Dual fixing: a column whose objective never prefers moving against its
  // free direction (no row locks it) is fixed at the corresponding bound.
  long long dual_fixing() {
    long long ch = 0;
    for (Index j = 0; j < n_ && status_ == PresolveStatus::Reduced; ++j) {
      if (!col_alive_[j] || lb_[j] == ub_[j]) continue;
      int down = 0, up = 0;
      for (const Entry& e : cols_[j]) {
        const Index i = e.idx;
        if (!row_alive_[i]) continue;
        const bool lo = std::isfinite(rlb_[i]), hi = std::isfinite(rub_[i]);
        if (e.val > 0) {
          down += lo;
          up += hi;
        } else {
          down += hi;
          up += lo;
        }
      }
      const double c = cost_[j];
      if (c >= 0.0 && down == 0) {
        if (std::isfinite(lb_[j])) {
          ub_[j] = lb_[j];
          ++stats_.bound_changes;
          ++ch;
        } else if (c > 0.0) {
          status_ = PresolveStatus::Unbounded;
          log_.log(3, "Presolve: column %d unbounded below by dual argument\n", j);
        }
      } else if (c <= 0.0 && up == 0) {
        if (std::isfinite(ub_[j])) {
          lb_[j] = ub_[j];
          ++stats_.bound_changes;
          ++ch;
        } else if (c < 0.0) {
          status_ = PresolveStatus::Unbounded;
          log_.log(3, "Presolve: column %d unbounded above by dual argument\n", j);
        }
      } else {
        // Dual bound strengthening: with c_j >= 0, x_j never needs to exceed the value at which every row
        // locking it downwards holds for all values of the other columns (lowering x_j to it keeps those rows,
        // only relaxes the others and does not increase the objective); symmetrically for c_j <= 0.
        if (minact_.empty()) row_activities();
        if (c >= 0.0 && std::isfinite(lb_[j])) ch += dual_bound(j, true);
        else if (c < 0.0 && std::isfinite(ub_[j])) ch += dual_bound(j, false);  // one side per pass (snapshot)
      }
    }
    minact_.clear();
    return ch;
  }

  // Row activity snapshot (minact_ / maxact_ with counts of infinite contributions).
  void row_activities() {
    minact_.assign(m_, 0.0);
    maxact_.assign(m_, 0.0);
    nmin_inf_.assign(m_, 0);
    nmax_inf_.assign(m_, 0);
    for (Index j = 0; j < n_; ++j) {
      if (!col_alive_[j]) continue;
      for (const Entry& e : cols_[j]) {
        if (!row_alive_[e.idx]) continue;
        const double bl = e.val > 0 ? lb_[j] : ub_[j], bh = e.val > 0 ? ub_[j] : lb_[j];
        if (small_bound(bl)) minact_[e.idx] += e.val * bl; else ++nmin_inf_[e.idx];
        if (small_bound(bh)) maxact_[e.idx] += e.val * bh; else ++nmax_inf_[e.idx];
      }
    }
  }

  // down: new upper bound of x_j from its down-locking rows (c_j >= 0); otherwise a lower bound from its
  // up-locking rows (c_j <= 0). The snapshot activities are conservative: bounds only tighten during the pass.
  int dual_bound(Index j, bool down) {
    const double own = down ? lb_[j] : ub_[j];  // x_j's contribution at that bound is part of the snapshot
    if (!small_bound(own)) return 0;
    double v = down ? -kInf : kInf;
    for (const Entry& e : cols_[j]) {
      const Index i = e.idx;
      if (!row_alive_[i]) continue;
      const double a = e.val;
      // Lowering x_j hurts a lower side for a > 0, an upper side for a < 0 (raising: the other way round).
      const bool use_lb = down == (a > 0);
      const double side = use_lb ? rlb_[i] : rub_[i];
      if (!std::isfinite(side)) continue;
      double w;
      if (use_lb) {  // a x_j + minact(others) >= rlb
        if (nmin_inf_[i] != 0) return 0;
        w = (rlb_[i] - (minact_[i] - a * own)) / a;
      } else {  // a x_j + maxact(others) <= rub
        if (nmax_inf_[i] != 0) return 0;
        w = (rub_[i] - (maxact_[i] - a * own)) / a;
      }
      v = down ? std::max(v, w) : std::min(v, w);
    }
    const double tol = 1e-9 * (1.0 + std::abs(v));
    if (down) {
      if (is_int_[j]) {
        v = std::ceil(v - 1e-9 * (1.0 + std::abs(v)));
      } else {
        v = relax_up(v, tol);  // an integral value stays integral (csched008's makespan: 190, not 190 + 1.9e-7)
        if (std::isfinite(ub_[j]) && v > ub_[j] - 1e-3 * std::max(1.0, std::abs(ub_[j]))) return 0;
        if (std::abs(v) > kMaxContBound && v > lb_[j]) return 0;
      }
      v = std::max(v, lb_[j]);
      if (!(v < ub_[j])) return 0;
      if (log_.verbosity() >= 4) log_.log(4, "  dual bound: col %d ub %g -> %g\n", j, ub_[j], v);
      ub_[j] = v;
    } else {
      if (is_int_[j]) {
        v = std::floor(v + 1e-9 * (1.0 + std::abs(v)));
      } else {
        v = -relax_up(-v, tol);
        if (std::isfinite(lb_[j]) && v < lb_[j] + 1e-3 * std::max(1.0, std::abs(lb_[j]))) return 0;
        if (std::abs(v) > kMaxContBound && v < ub_[j]) return 0;
      }
      v = std::min(v, ub_[j]);
      if (!(v > lb_[j])) return 0;
      if (log_.verbosity() >= 4) log_.log(4, "  dual bound: col %d lb %g -> %g\n", j, lb_[j], v);
      lb_[j] = v;
    }
    ++stats_.bound_changes;
    ++stats_.dual_bounds;
    return 1;
  }

  // Dominated columns: x_j dominates x_k if c_j <= c_k and moving a unit from x_k to x_j never hurts a row
  // (a_ij <= a_ik in rows with only an upper side, >= with only a lower side, = in ranged rows) or integrality.
  // Some optimal solution then has x_k = lb_k or x_j = ub_j: with ub_j = inf, x_k is fixed to lb_k; otherwise
  // x_k is bounded by what the rows allow with x_j = ub_j (fixed if that excludes every value above lb_k).
  long long dominated_cols() {
    long long ch = 0;
    // Row activities; a snapshot stays conservative since bounds only tighten during the pass.
    std::vector<double> minact(m_, 0.0), maxact(m_, 0.0);
    std::vector<int> nmin_inf(m_, 0), nmax_inf(m_, 0);
    long long nnz = 0;
    for (Index j = 0; j < n_; ++j) {
      if (!col_alive_[j]) continue;
      for (const Entry& e : cols_[j]) {
        if (!row_alive_[e.idx]) continue;
        ++nnz;
        const double bl = e.val > 0 ? lb_[j] : ub_[j], bh = e.val > 0 ? ub_[j] : lb_[j];
        if (small_bound(bl)) minact[e.idx] += e.val * bl; else ++nmin_inf[e.idx];
        if (small_bound(bh)) maxact[e.idx] += e.val * bh; else ++nmax_inf[e.idx];
      }
    }
    std::vector<double> coefk(m_, 0.0);
    std::vector<Index> stamp_k(m_, -1), stamp_j(m_, -1);
    std::vector<char> touched(n_, 0);  // changed in this pass: neither dominated again nor dominating
    const long long limit = 20 * nnz + 1000000;
    long long work = 0;
    // Moving delta from x_k to x_j changes row i by (aj - ak) delta.
    auto row_ok = [&](Index i, double aj, double ak) {
      const bool lo = std::isfinite(rlb_[i]), hi = std::isfinite(rub_[i]);
      if (lo && hi) return aj == ak;
      if (hi) return aj <= ak;
      if (lo) return aj >= ak;
      return true;
    };
    for (Index k = 0; k < n_ && work < limit && status_ == PresolveStatus::Reduced; ++k) {
      if (!col_alive_[k] || lb_[k] == ub_[k] || !small_bound(lb_[k]) || touched[k]) continue;
      Index cand = -1;
      for (const Entry& e : cols_[k]) {
        if (row_alive_[e.idx] && (cand < 0 || row_size_[e.idx] < row_size_[cand])) cand = e.idx;
        if (row_alive_[e.idx]) stamp_k[e.idx] = k, coefk[e.idx] = e.val;
      }
      if (cand < 0 || row_size_[cand] > 1000) continue;
      for (const Entry& c : rows_[cand]) {
        const Index j = c.idx;
        if (j == k || !col_alive_[j] || touched[j] || lb_[j] == ub_[j]) continue;
        if (cost_[j] > cost_[k]) continue;
        const bool j_unbounded = ub_[j] == kInf;
        // The shift must keep both integral: same type, or any x_k into an unbounded continuous x_j.
        if (is_int_[j] != is_int_[k] && !(is_int_[k] && j_unbounded)) continue;
        work += col_size_[j] + col_size_[k];
        bool dom = true;
        for (const Entry& e : cols_[j]) {
          if (!row_alive_[e.idx]) continue;
          stamp_j[e.idx] = j;
          dom = dom && row_ok(e.idx, e.val, stamp_k[e.idx] == k ? coefk[e.idx] : 0.0);
        }
        for (const Entry& e : cols_[k]) {
          if (dom && row_alive_[e.idx] && stamp_j[e.idx] != j) dom = row_ok(e.idx, 0.0, e.val);
        }
        if (!dom) continue;
        double nub = ub_[k];
        // With x_j = ub_j, lowering x_k to lb_k never hurts if c_k >= 0 and every row that locks x_k downwards
        // still holds at x_k = lb_k for all values of the other columns: x_k = lb_k in some optimal solution
        // either way (set covering: x_j covers every row of x_k at no more cost).
        bool down_free = !j_unbounded && cost_[k] >= 0.0 && small_bound(ub_[j]);
        for (const Entry& e : cols_[k]) {
          if (!down_free) break;
          const Index i = e.idx;
          if (!row_alive_[i]) continue;
          const double a = e.val;
          const double aj = stamp_j[i] == j ? get_coef(i, j) : 0.0;
          if (a > 0 && std::isfinite(rlb_[i])) {
            const double jmin = aj > 0 ? lb_[j] : ub_[j];
            const int inf = nmin_inf[i] - (aj != 0.0 && !small_bound(jmin) ? 1 : 0);
            const double jc = aj != 0.0 && small_bound(jmin) ? aj * jmin : 0.0;
            down_free = inf == 0 && minact[i] - jc + aj * ub_[j] >= rlb_[i] - 1e-9 * (1.0 + std::abs(rlb_[i]));
          } else if (a < 0 && std::isfinite(rub_[i])) {
            const double jmax = aj > 0 ? ub_[j] : lb_[j];
            const int inf = nmax_inf[i] - (aj != 0.0 && !small_bound(jmax) ? 1 : 0);
            const double jc = aj != 0.0 && small_bound(jmax) ? aj * jmax : 0.0;
            down_free = inf == 0 && maxact[i] - jc + aj * ub_[j] <= rub_[i] + 1e-9 * (1.0 + std::abs(rub_[i]));
          }
        }
        if (down_free) {
          nub = lb_[k];
        } else if (!j_unbounded) {
          // Bounds on x_k from its rows with x_j = ub_j (and the others at their activity bounds).
          for (const Entry& e : cols_[k]) {
            const Index i = e.idx;
            if (!row_alive_[i]) continue;
            const double a = e.val;
            const double aj = stamp_j[i] == j ? get_coef(i, j) : 0.0;
            if (a > 0 && std::isfinite(rub_[i])) {
              const double jmin = aj > 0 ? lb_[j] : ub_[j];
              const int inf = nmin_inf[i] - (aj != 0.0 && !small_bound(jmin) ? 1 : 0);
              if (inf != 0) continue;
              const double jc = aj != 0.0 && small_bound(jmin) ? aj * jmin : 0.0;
              const double rest = minact[i] - a * lb_[k] - jc + aj * ub_[j];
              nub = std::min(nub, (rub_[i] - rest) / a);
            } else if (a < 0 && std::isfinite(rlb_[i])) {
              const double jmax = aj > 0 ? ub_[j] : lb_[j];
              const int inf = nmax_inf[i] - (aj != 0.0 && !small_bound(jmax) ? 1 : 0);
              if (inf != 0) continue;
              const double jc = aj != 0.0 && small_bound(jmax) ? aj * jmax : 0.0;
              const double rest = maxact[i] - a * lb_[k] - jc + aj * ub_[j];
              nub = std::min(nub, (rlb_[i] - rest) / a);
            }
          }
          const double tol = 1e-9 * (1.0 + std::abs(nub));
          if (is_int_[k]) {
            nub = std::floor(nub + 1e-6);
          } else if (nub > lb_[k] + tol) {
            continue;  // continuous columns are only fixed
          }
          nub = std::max(nub, lb_[k]);
          if (!(nub < ub_[k])) continue;
        } else {
          nub = lb_[k];
        }
        if (log_.verbosity() >= 4) log_.log(4, "  col %d dominates col %d: ub %g -> %g\n", j, k, ub_[k], nub);
        ub_[k] = nub;
        touched[k] = 1;
        ++stats_.bound_changes;
        ++stats_.dominated;
        ++ch;
        break;
      }
    }
    return ch;
  }

  // Dual aggregation onto a variable upper bound: a binary x_j with c_j <= 0 whose only up-lock is a one-sided row
  // r with a binary y such that r holds at y = 1 for all values of its other columns, and admits no x_j = 1 at
  // y = 0 (sum_{k in S} a_k x_k <= (sum_{k in S} a_k) y, or x_j <= y). Some optimal solution has x_j = y: y = 0
  // forces x_j = 0, and with y = 1 raising x_j to 1 keeps row r (redundant then) and x_j's other rows (none locks
  // it upwards) and does not increase the objective. x_j is substituted by y (neos-787933: 1764 rows "sum of 133
  // x_k <= 133 y" after dual fixing).
  long long dual_aggregate() {
    long long ch = 0, nnz = 0;
    for (Index i = 0; i < m_; ++i) nnz += row_alive_[i] ? row_size_[i] : 0;
    const long long limit = 20 * nnz + 1000000;
    long long work = 0;
    auto binary = [&](Index k) { return is_int_[k] && lb_[k] == 0.0 && ub_[k] == 1.0; };
    for (Index j = 0; j < n_ && work < limit && status_ == PresolveStatus::Reduced; ++j) {
      if (!col_alive_[j] || !binary(j) || cost_[j] > 0.0) continue;
      Index r = -1;
      double s = 0.0;  // row r as s * (row) <= b
      bool one = true;
      for (const Entry& e : cols_[j]) {
        const Index i = e.idx;
        if (!row_alive_[i] || !(e.val > 0 ? std::isfinite(rub_[i]) : std::isfinite(rlb_[i]))) continue;
        if (r >= 0 || (std::isfinite(rlb_[i]) && std::isfinite(rub_[i]))) {
          one = false;
          break;
        }
        r = i;
        s = e.val > 0 ? 1.0 : -1.0;
      }
      if (!one || r < 0 || row_size_[r] < 2 || row_size_[r] > 1000) continue;
      work += row_size_[r];
      const double b = s > 0 ? rub_[r] : -rlb_[r];
      double minact = 0.0, maxact = 0.0, aj = 0.0;
      bool finite = true;
      for (const Entry& e : rows_[r]) {
        const Index k = e.idx;
        if (!col_alive_[k]) continue;
        if (!small_bound(lb_[k]) || !small_bound(ub_[k])) {
          finite = false;
          break;
        }
        const double a = s * e.val;
        minact += a * (a > 0 ? lb_[k] : ub_[k]);
        maxact += a * (a > 0 ? ub_[k] : lb_[k]);
        if (k == j) aj = a;
      }
      if (!finite || !(aj > 0.0)) continue;
      // y: at y = 1 the maximum activity is maxact + a_y (y adds a_y lb_y = 0 to maxact); at y = 0 with x_j = 1 the
      // minimum one is minact - a_y + a_j (y adds a_y ub_y = a_y to minact, x_j a_j lb_j = 0).
      const double tol = 1e-9 * (1.0 + std::abs(b));
      Index y = -1;
      for (const Entry& e : rows_[r]) {
        const Index k = e.idx;
        const double a = s * e.val;
        if (!col_alive_[k] || k == j || !(a < 0.0) || !binary(k)) continue;
        if (maxact + a <= b + tol && minact - a + aj > b + opt_.feas_tol * (1.0 + std::abs(b))) {
          y = k;
          break;
        }
      }
      if (y < 0) continue;
      for (const Entry& e : cols_[j]) work += row_alive_[e.idx] ? row_size_[e.idx] : 0;
      if (log_.verbosity() >= 4) log_.log(4, "  dual aggregation: col %d = col %d (row %d)\n", j, y, r);
      substitute(j, 0.0, 1.0, {{y, -1.0}}, -1);
      ++stats_.dual_aggregations;
      ++ch;
    }
    return ch;
  }

  // Parallel rows: rows that are scalar multiples of each other are merged.
  long long parallel_rows() {
    long long ch = 0;
    // The normalized rows (sorted by column, scaled so the first value is 1) lie in one array and the rows with
    // equal hashes are found by sorting (hash, row) pairs: a hash map of row lists and one vector per row cost
    // more than the comparisons (neos-3402294-bobin: 2.1 s of a 3.2 s presolve in this step, most of it in the
    // map). The groups are independent of each other, so the order in which they are visited does not matter.
    std::vector<Entry>& norm = par_norm_;
    std::vector<Index>& off = par_off_;
    std::vector<double>& first = par_first_;
    std::vector<std::pair<uint64_t, Index>>& keys = par_keys_;
    norm.clear();
    keys.clear();
    off.assign(m_ + 1, 0);
    first.assign(m_, 0.0);
    for (Index i = 0; i < m_; ++i) {
      off[i] = static_cast<Index>(norm.size());
      if (!row_alive_[i] || row_size_[i] < 2) continue;
      for (const Entry& e : rows_[i]) {
        if (col_alive_[e.idx]) norm.push_back(e);
      }
      const auto vb = norm.begin() + off[i], ve = norm.end();
      std::sort(vb, ve, [](const Entry& x, const Entry& y) { return x.idx < y.idx; });
      first[i] = vb->val;
      uint64_t h = 1469598103934665603ULL ^ static_cast<size_t>(ve - vb);
      for (auto it = vb; it != ve; ++it) {
        Entry& e = *it;
        e.val /= first[i];
        const auto q = static_cast<int64_t>(std::llround(e.val * 1e6));
        h = (h ^ static_cast<uint64_t>(e.idx)) * 1099511628211ULL;
        h = (h ^ static_cast<uint64_t>(q)) * 1099511628211ULL;
      }
      keys.push_back({h, i});
    }
    off[m_] = static_cast<Index>(norm.size());
    std::sort(keys.begin(), keys.end());  // equal hashes together, rows in increasing order within
    for (size_t g0 = 0, g1; g0 < keys.size(); g0 = g1) {
      for (g1 = g0 + 1; g1 < keys.size() && keys[g1].first == keys[g0].first; ++g1) {}
      if (g1 - g0 < 2) continue;
      for (size_t x = g0; x < g1 && status_ == PresolveStatus::Reduced; ++x) {
        const Index r1 = keys[x].second;
        if (!row_alive_[r1]) continue;
        const Index len1 = off[r1 + 1] - off[r1];
        for (size_t y = x + 1; y < g1; ++y) {
          const Index r2 = keys[y].second;
          if (!row_alive_[r2] || off[r2 + 1] - off[r2] != len1) continue;
          bool same = true;
          for (Index k = 0; k < len1 && same; ++k) {
            const Entry &e1 = norm[off[r1] + k], &e2 = norm[off[r2] + k];
            same = e1.idx == e2.idx && std::abs(e1.val - e2.val) <= 1e-12 * std::max(1.0, std::abs(e1.val));
          }
          if (!same) continue;
          // Keep the row with larger coefficients: its absolute tolerance is the strictest.
          const bool keep_first = std::abs(first[r1]) >= std::abs(first[r2]);
          const Index keep = keep_first ? r1 : r2, drop = keep_first ? r2 : r1;
          // drop = lambda * keep
          const double lambda = first[drop] / first[keep];
          const double l2 = lambda > 0 ? rlb_[drop] / lambda : rub_[drop] / lambda;
          const double u2 = lambda > 0 ? rub_[drop] / lambda : rlb_[drop] / lambda;
          double l = std::max(rlb_[keep], l2), u = std::min(rub_[keep], u2);
          if (l > u) {
            const double scale = std::max({1.0, std::abs(l), std::abs(u)});
            if (l - u > opt_.feas_tol * scale / std::max(1.0, std::abs(first[keep]))) {
              status_ = PresolveStatus::Infeasible;
              log_.log(3, "Presolve: parallel rows %d, %d contradict\n", r1, r2);
              return ch;
            }
            l = u = 0.5 * (l + u);
          }
          rlb_[keep] = l;
          rub_[keep] = u;
          remove_row(drop);
          ++stats_.parallel_rows;
          ++ch;
          if (!keep_first) break;
        }
      }
    }
    return ch;
  }

  // Live entries of row i.
  void live_row(Index i, std::vector<Entry>& out) const {
    out.clear();
    for (const Entry& e : rows_[i]) {
      if (col_alive_[e.idx]) out.push_back(e);
    }
  }

  double get_coef(Index i, Index j) const {
    for (const Entry& e : rows_[i]) {
      if (e.idx == j) return e.val;
    }
    return 0.0;
  }

  static bool integral(double v) { return std::abs(v - std::round(v)) <= 1e-9 * std::max(1.0, std::abs(v)); }

  // Substitutes x_j = (rhs - sum terms) / coef into the objective and every live row except `skip`, removes
  // column j and records the substitution for postsolve. The terms must be live columns other than j.
  void substitute(Index j, double rhs, double coef, const std::vector<Entry>& terms, Index skip) {
    offset_ += cost_[j] * rhs / coef;
    for (const Entry& t : terms) cost_[t.idx] -= cost_[j] * t.val / coef;
    std::vector<Entry> colj;
    for (const Entry& x : cols_[j]) {
      if (row_alive_[x.idx] && x.idx != skip) colj.push_back(x);
    }
    for (const Entry& x : colj) {
      const Index r = x.idx;
      const double f = x.val / coef;
      if (std::isfinite(rlb_[r])) rlb_[r] -= f * rhs;
      if (std::isfinite(rub_[r])) rub_[r] -= f * rhs;
      for (const Entry& t : terms) {
        const Index k = t.idx;
        const double cur = get_coef(r, k);
        double nv = cur - f * t.val;
        // Near-cancellation leaves numerical noise: drop it, relaxing the row by its largest effect. The round-off of
        // an exact cancellation (a few ulps) goes without relaxing, also for unbounded columns: kept, it could become
        // a pivot (0.1 - 0.1 = 1.4e-17 for a free integer column, whose "integral" ratios then substituted it).
        const double kmax = std::max(std::abs(lb_[k]), std::abs(ub_[k]));
        const double big = std::max(std::abs(cur), std::abs(f * t.val));
        if (std::abs(nv) <= 1e-14 * big) {
          nv = 0.0;
        } else if (std::abs(nv) <= 1e-9 * big && kmax < kBig) {
          if (std::isfinite(rlb_[r])) rlb_[r] -= std::abs(nv) * kmax;
          if (std::isfinite(rub_[r])) rub_[r] += std::abs(nv) * kmax;
          nv = 0.0;
        }
        set_coef(r, k, nv);
      }
      set_coef(r, j, 0.0);
    }
    out_.stack_.push_back({j, rhs, coef, {}, lb_[j], ub_[j]});
    for (const Entry& t : terms) out_.stack_.back().terms.push_back({t.idx, t.val});
    col_alive_[j] = 0;
    for (const Entry& x : cols_[j]) {
      if (row_alive_[x.idx]) --row_size_[x.idx];
    }
  }

  // Entries that substituting column j out of row i (live entries r) would create in its other rows.
  long long new_entries(Index j, Index i, const std::vector<Entry>& r) {
    if (mark_.size() != static_cast<size_t>(n_)) mark_.assign(n_, -1);
    long long cnt = 0;
    // A row much longer than the columns of r is not walked: whether it holds a column of r is read from that
    // column instead (triptim1: the rows of a candidate are over a thousand entries long, its columns about 20).
    long long col_len = 0;
    for (const Entry& e : r) col_len += e.idx != j ? static_cast<long long>(cols_[e.idx].size()) : 0;
    for (const Entry& x : cols_[j]) {
      if (!row_alive_[x.idx] || x.idx == i) continue;
      if (static_cast<long long>(rows_[x.idx].size()) > col_len) {
        for (const Entry& e : r) {
          if (e.idx == j) continue;
          bool found = false;
          for (const Entry& y : cols_[e.idx]) {
            if (y.idx == x.idx) {
              found = true;
              break;
            }
          }
          cnt += !found;
        }
        continue;
      }
      for (const Entry& e : rows_[x.idx]) mark_[e.idx] = x.idx;
      for (const Entry& e : r) cnt += e.idx != j && mark_[e.idx] != x.idx;
    }
    return cnt;
  }

  // Parallel columns A_j = s A_k with c_j = s c_k (same type; s = +-1 for integers): x_k and x_j only appear as
  // y = x_k + s x_j, so x_j is removed and x_k becomes y with the bounds of that sum.
  long long parallel_cols() {
    long long ch = 0;
    // As parallel_rows: the normalized columns (sorted by row, scaled so the first value is 1) in one array,
    // equal hashes found by sorting. A merge touches only its two columns, so the group order does not matter.
    std::vector<Entry>& norm = par_norm_;
    std::vector<Index>& off = par_off_;
    std::vector<double>& first = par_first_;
    std::vector<std::pair<uint64_t, Index>>& keys = par_keys_;
    norm.clear();
    keys.clear();
    off.assign(n_ + 1, 0);
    first.assign(n_, 0.0);
    for (Index j = 0; j < n_; ++j) {
      off[j] = static_cast<Index>(norm.size());
      if (!col_alive_[j] || col_size_[j] < 1 || lb_[j] == ub_[j]) continue;
      for (const Entry& e : cols_[j]) {
        if (row_alive_[e.idx]) norm.push_back(e);
      }
      const auto vb = norm.begin() + off[j], ve = norm.end();
      std::sort(vb, ve, [](const Entry& x, const Entry& y) { return x.idx < y.idx; });
      first[j] = vb->val;
      uint64_t h = 1469598103934665603ULL ^ (static_cast<size_t>(ve - vb) * 2 + is_int_[j]);
      for (auto it = vb; it != ve; ++it) {
        Entry& e = *it;
        e.val /= first[j];
        const auto q = static_cast<int64_t>(std::llround(e.val * 1e6));
        h = (h ^ static_cast<uint64_t>(e.idx)) * 1099511628211ULL;
        h = (h ^ static_cast<uint64_t>(q)) * 1099511628211ULL;
      }
      keys.push_back({h, j});
    }
    off[n_] = static_cast<Index>(norm.size());
    std::sort(keys.begin(), keys.end());
    for (size_t g0 = 0, g1; g0 < keys.size(); g0 = g1) {
      for (g1 = g0 + 1; g1 < keys.size() && keys[g1].first == keys[g0].first; ++g1) {}
      for (size_t x = g0; x < g1 && g1 - g0 > 1; ++x) {
        const Index k = keys[x].second;
        if (!col_alive_[k]) continue;
        const Index lenk = off[k + 1] - off[k];
        for (size_t y = x + 1; y < g1; ++y) {
          const Index j = keys[y].second;
          if (!col_alive_[j] || is_int_[j] != is_int_[k] || off[j + 1] - off[j] != lenk) continue;
          bool same = true;
          for (Index t = 0; t < lenk && same; ++t) {
            const Entry &e1 = norm[off[k] + t], &e2 = norm[off[j] + t];
            same = e1.idx == e2.idx && std::abs(e1.val - e2.val) <= 1e-12 * std::max(1.0, std::abs(e1.val));
          }
          if (!same) continue;
          double s = first[j] / first[k];  // A_j = s A_k
          if (is_int_[k]) {
            if (std::abs(std::abs(s) - 1.0) > 1e-12) continue;
            s = s > 0 ? 1.0 : -1.0;
          } else if (std::abs(s) > 1e3 || std::abs(s) < 1e-3) {
            continue;
          }
          if (std::abs(cost_[j] - s * cost_[k]) > 1e-12 * std::max(1.0, std::abs(cost_[j]))) continue;
          // Bounds of y = x_k + s x_j.
          const double jl = s > 0 ? lb_[j] : ub_[j], ju = s > 0 ? ub_[j] : lb_[j];
          const double yl = std::isfinite(lb_[k]) && std::isfinite(jl) ? lb_[k] + s * jl : -kInf;
          const double yu = std::isfinite(ub_[k]) && std::isfinite(ju) ? ub_[k] + s * ju : kInf;
          if (std::abs(yl) > kBig && std::isfinite(yl)) continue;
          if (std::abs(yu) > kBig && std::isfinite(yu)) continue;
          Presolver::Step st;
          st.col = j;
          st.other = k;
          st.coef = s;
          st.lb = lb_[j];
          st.ub = ub_[j];
          st.olb = lb_[k];
          st.oub = ub_[k];
          st.integer = is_int_[k];
          out_.stack_.push_back(st);
          if (log_.verbosity() >= 4) log_.log(4, "  parallel cols: col %d += %g col %d\n", k, s, j);
          lb_[k] = yl;
          ub_[k] = yu;
          merged_[k] = 1;
          col_alive_[j] = 0;
          for (const Entry& e : cols_[j]) {
            if (row_alive_[e.idx]) --row_size_[e.idx];
          }
          ++stats_.parallel_cols;
          ++ch;
        }
      }
    }
    return ch;
  }

  bool binary(Index j) const { return is_int_[j] && lb_[j] == 0.0 && ub_[j] == 1.0; }

  // Whether column j is in a row (other than skip) of three or more binaries with one coefficient a and the side
  // a on the bounded side of a: sum x <= 1 or sum x = 1 after scaling. Rows of up to 64 columns of j are looked
  // at, each scanned in full.
  bool packing_member(Index j, Index skip) const {
    if (col_size_[j] > 64) return false;
    for (const Entry& c : cols_[j]) {
      const Index r = c.idx;
      if (r == skip || !row_alive_[r] || row_size_[r] < 3) continue;
      const double a = c.val;
      if (a > 0 ? rub_[r] != a : rlb_[r] != a) continue;
      bool ok = true;
      for (const Entry& e : rows_[r]) {
        if (!col_alive_[e.idx]) continue;
        if (e.val != a || !binary(e.idx)) {
          ok = false;
          break;
        }
      }
      if (ok) return true;
    }
    return false;
  }

  // Doubleton equations a_j x_j + a_k x_k = b: substitute x_j = (b - a_k x_k) / a_j everywhere, move the bounds
  // of x_j onto x_k and drop the row. x_j is continuous, or both are integer with a_k / a_j and b / a_j integral.
  long long aggregate() {
    long long ch = 0;
    for (Index i = 0; i < m_ && status_ == PresolveStatus::Reduced; ++i) {
      if (!row_alive_[i] || row_size_[i] != 2 || rlb_[i] != rub_[i]) continue;
      Entry e[2];
      int cnt = 0;
      for (const Entry& x : rows_[i]) {
        if (col_alive_[x.idx] && cnt < 2) e[cnt++] = x;
      }
      if (cnt != 2) continue;
      const double b = rlb_[i];
      if (!std::isfinite(b) || std::abs(b) > kBig) continue;
      // Eliminate a continuous column; prefer the larger coefficient (stable).
      int el;
      if (!is_int_[e[0].idx] && !is_int_[e[1].idx]) {
        el = std::abs(e[0].val) >= std::abs(e[1].val) ? 0 : 1;
      } else if (!is_int_[e[0].idx]) {
        el = 0;
      } else if (!is_int_[e[1].idx]) {
        el = 1;
      } else {
        // Both integer: x_j stays integral iff a_k / a_j and b / a_j are integers.
        auto ok = [&](int t) { return integral(e[1 - t].val / e[t].val) && integral(b / e[t].val); };
        const bool ok0 = ok(0), ok1 = ok(1);
        if (!ok0 && !ok1) continue;
        el = ok0 && ok1 ? (col_size_[e[0].idx] <= col_size_[e[1].idx] ? 0 : 1) : (ok0 ? 0 : 1);
        // Complements x + y = 1 of two binaries: the one that is a member of a packing or partitioning row stays,
        // so that the row keeps its form for the clique steps and the one-of groups (traininstance6: the 501
        // members of a group were replaced by their complements, sum y = 500, and no step recognized the group).
        if (opt_.keep_members && ok0 && ok1 && e[0].val == e[1].val && b == e[0].val && binary(e[0].idx) &&
            binary(e[1].idx)) {
          const bool m0 = packing_member(e[0].idx, i), m1 = packing_member(e[1].idx, i);
          if (m0 != m1) el = m0 ? 1 : 0;
        }
      }
      const Index j = e[el].idx, k = e[1 - el].idx;
      const double aj = e[el].val, ak = e[1 - el].val;
      const double ratio = std::abs(ak / aj);
      if (ratio > 1e4 || ratio < 1e-4 || col_size_[j] > 64) continue;
      // Bounds of x_j translated to x_k: x_k = (b - aj x_j) / ak.
      const double v1 = std::isfinite(lb_[j]) ? (b - aj * lb_[j]) / ak : (aj / ak > 0 ? kInf : -kInf);
      const double v2 = std::isfinite(ub_[j]) ? (b - aj * ub_[j]) / ak : (aj / ak > 0 ? -kInf : kInf);
      double kl = std::min(v1, v2), ku = std::max(v1, v2);
      const double xtol = opt_.feas_tol / std::max(1.0, std::abs(ak / aj));
      if (is_int_[k]) {
        kl = std::ceil(kl - xtol);
        ku = std::floor(ku + xtol);
      }
      tighten(k, kl, ku, xtol);
      if (status_ != PresolveStatus::Reduced) break;
      if (log_.verbosity() >= 4) log_.log(4, "  aggregate col %d = (%.12g - %.12g col %d) / %.12g\n", j, b, ak, k, aj);
      substitute(j, b, aj, {{k, ak}}, i);
      remove_row(i);
      ++stats_.aggregations;
      ++ch;
    }
    return ch;
  }

  // Whether the rows of column j other than `skip` imply its bounds, given the current bounds of their other
  // columns: lo (hi) is set when one of them implies x_j >= lb_j (x_j <= ub_j). Row r: rlb_r - (max activity of
  // the others) <= a_rj x_j <= rub_r - (min activity of the others). When x_j is then substituted out of row
  // skip, these rows stay (with x_j replaced by its expression) and so do the bounds of their other columns, so
  // the bounds of x_j are redundant. The bound is evaluated in the current problem each time (a snapshot could
  // let two columns imply each other's bounds, e.g. x_j = x_k with both bounds derived from that row, and both
  // be dropped). Tolerances as for the substitution row (1e-9 relative, integers rounded by 1e-6), plus the
  // round-off of the activity (1e-12 of the magnitudes summed, as propagate's margin: a continuous bound
  // propagate tightened from row r is recognized, a bound the row misses by more is not). Only rows with
  // |a_rj| >= |a_j| / kMaxImpliedRatio (a_j: x_j's coefficient in row skip): a reduced point violating row r by v
  // puts x_j up to v / |a_rj| beyond its bound, and postsolve's clamp moves that into row skip as |a_j| v / |a_rj|
  // (milo-v12-6-r2-40-1 has ratios up to 100; 10 keeps 232 of the 270 rows removed without a limit).
  struct RowAct {
    double minact = 0.0, maxact = 0.0, mag = 0.0;  // over all live columns of the row
    int nmin = 0, nmax = 0;                        // terms with a bound beyond kBig
    int stamp = -1;                                // valid if equal to row_act_stamp_
    bool exact = false;                            // integer terms, totals below 2^52: the sums carry no rounding
  };
  // Sums of row r under the current bounds, kept until the row or its columns change (substitute_equations
  // starts a new stamp at every call and drops the rows of a column it substitutes; nothing else changes them
  // while implied_by_rows is in use).
  const RowAct& row_act(Index r) const {
    RowAct& c = row_act_[r];
    if (c.stamp == row_act_stamp_) return c;
    c = RowAct{};
    c.stamp = row_act_stamp_;
    c.exact = true;
    for (const Entry& e : rows_[r]) {
      const Index k = e.idx;
      if (!col_alive_[k]) continue;
      const double bl = e.val > 0 ? lb_[k] : ub_[k], bh = e.val > 0 ? ub_[k] : lb_[k];
      if (small_bound(bl)) {
        const double t = e.val * bl;
        c.minact += t, c.mag += std::abs(t), c.exact = c.exact && t == std::round(t);
      } else {
        ++c.nmin;
      }
      if (small_bound(bh)) {
        const double t = e.val * bh;
        c.maxact += t, c.mag += std::abs(t), c.exact = c.exact && t == std::round(t);
      } else {
        ++c.nmax;
      }
      c.exact = c.exact && c.mag <= 4503599627370496.0;
    }
    return c;
  }

  void implied_by_rows(Index j, Index skip, double aj, bool& lo, bool& hi) const {
    for (const Entry& x : cols_[j]) {
      const Index r = x.idx;
      if (lo && hi) return;
      if (r == skip || !row_alive_[r]) continue;
      const double a = x.val;
      if (std::abs(aj) > kMaxImpliedRatio * std::abs(a)) continue;
      const bool has_lb = std::isfinite(rlb_[r]), has_ub = std::isfinite(rub_[r]);
      // Sides that would give the missing bound: a > 0: rub gives an upper bound, rlb a lower one.
      const bool want_ub = !hi && (a > 0 ? has_ub : has_lb), want_lb = !lo && (a > 0 ? has_lb : has_ub);
      if (!want_ub && !want_lb) continue;
      double minact = 0.0, maxact = 0.0, mag = 0.0;
      int nmin_inf = 0, nmax_inf = 0;
      // The sums over the row without x_j. One long row is asked about by many of its columns (triptim1: rows of
      // over a thousand entries, 43 million entries walked per call of substitute_equations, twelve calls with no
      // substitution at all), so the sums over the whole row are kept while the row is unchanged (row_act). The
      // counts of unbounded terms without x_j follow from them; the sums only when every term is an integer and
      // the totals stay below 2^52: then total minus x_j's term is the same number as the sum that skips x_j.
      const RowAct& c = row_act(r);
      const double jl = a > 0 ? lb_[j] : ub_[j], jh = a > 0 ? ub_[j] : lb_[j];
      const int jmin_inf = small_bound(jl) ? 0 : 1, jmax_inf = small_bound(jh) ? 0 : 1;
      if (c.nmin - jmin_inf > 0 && c.nmax - jmax_inf > 0) {
        nmin_inf = nmax_inf = 1;  // neither side gives a bound (the walk below would stop at once)
      } else if (c.exact) {
        nmin_inf = c.nmin - jmin_inf, nmax_inf = c.nmax - jmax_inf;
        minact = c.minact - (jmin_inf ? 0.0 : a * jl);
        maxact = c.maxact - (jmax_inf ? 0.0 : a * jh);
        mag = c.mag - (jmin_inf ? 0.0 : std::abs(a * jl)) - (jmax_inf ? 0.0 : std::abs(a * jh));
      } else {
        for (const Entry& e : rows_[r]) {
          const Index k = e.idx;
          if (k == j || !col_alive_[k]) continue;
          const double bl = e.val > 0 ? lb_[k] : ub_[k], bh = e.val > 0 ? ub_[k] : lb_[k];
          if (small_bound(bl)) minact += e.val * bl, mag += std::abs(e.val * bl); else ++nmin_inf;
          if (small_bound(bh)) maxact += e.val * bh, mag += std::abs(e.val * bh); else ++nmax_inf;
          if (nmin_inf > 0 && nmax_inf > 0) break;
        }
      }
      double scale = 1.0;
      if (has_lb) scale = std::max(scale, std::abs(rlb_[r]));
      if (has_ub) scale = std::max(scale, std::abs(rub_[r]));
      const double err = 1e-12 * (mag + scale) / std::abs(a);
      // From rub: a x_j <= rub - minact; from rlb: a x_j >= rlb - maxact.
      double ilo = -kInf, ihi = kInf;
      if (has_ub && nmin_inf == 0) (a > 0 ? ihi : ilo) = (rub_[r] - minact) / a;
      if (has_lb && nmax_inf == 0) (a > 0 ? ilo : ihi) = (rlb_[r] - maxact) / a;
      ilo -= err;
      ihi += err;
      if (is_int_[j]) {
        ilo = std::ceil(ilo - 1e-6);
        ihi = std::floor(ihi + 1e-6);
      }
      lo = lo || ilo >= lb_[j] - 1e-9 * (1.0 + std::abs(lb_[j]));
      hi = hi || ihi <= ub_[j] + 1e-9 * (1.0 + std::abs(ub_[j]));
    }
  }

  // Whether continuous column j has a variable bound: an inequality with integer columns whose continuous
  // coefficients all have one sign and the integer ones the other (x <= u y; as sparsify_protect).
  bool has_variable_bound(Index j) const {
    for (const Entry& x : cols_[j]) {
      const Index i = x.idx;
      if (!row_alive_[i] || rlb_[i] == rub_[i]) continue;
      int cs = 0, is = 0;
      bool ok = true;
      for (const Entry& e : rows_[i]) {
        if (!col_alive_[e.idx]) continue;
        int& sg = is_int_[e.idx] ? is : cs;
        const int v = e.val > 0 ? 1 : -1;
        if (sg == 0) sg = v;
        ok = ok && sg == v;
      }
      if (ok && cs != 0 && is != 0 && cs != is) return true;
    }
    return false;
  }

  // Equations sum_k a_k x_k = b: a column j is substituted out, x_j = (b - sum_{k != j} a_k x_k) / a_j. The row
  // stays as lb_j <= (b - sum_{k != j} a_k x_k) / a_j <= ub_j, without the sides that the row or the column's
  // other rows already imply (implied_by_rows; the row is removed if j is implied free). Column singletons cause
  // no fill-in and are preferred (implied free ones first); they are taken from equations of any length: the
  // work is O(row length) for the objective update (objective-defining rows z = c^T x with a free z). Other
  // columns are tried only when no singleton qualifies, in rows of at most kMaxSubstRow entries, if the fill-in
  // is small: a substitution that keeps the row, or removes it by implied_by_rows, may add at most
  // substitution_fill nonzeros net for a continuous column (csched007/008, neos-4954672-berkel, milo-v12-6-r2-40-1;
  // without the limit for the latter milo kept 236 more rows than before: the longer rows no longer became
  // doubletons), none for an integer column (binaries out of set partitioning rows: chromaticindex512-7 9215
  // columns fewer for 27.6k more nonzeros, the cliques gone). One that keeps the row waits for the first probing
  // pass (defer_fill_), and is not done for a column with a variable bound (has_variable_bound: the flows of
  // fixed-charge networks, which the c-MIR and flow cover separators aggregate through; beasleyC3 and mc11 went
  // from 15-17 s to the 120 s limit, p200x1188c 0.6 -> 5 s). Integer columns qualify only if the row keeps them
  // integral (integer columns, a_k / a_j and b / a_j integral).
  long long substitute_equations() {
    long long ch = 0;
    std::vector<Entry> r, terms;
    if (row_act_.size() != static_cast<size_t>(m_)) row_act_.assign(m_, RowAct{});
    if (++row_act_stamp_ == std::numeric_limits<int>::max()) {
      row_act_.assign(m_, RowAct{});
      row_act_stamp_ = 0;
    }
    for (Index i = 0; i < m_ && status_ == PresolveStatus::Reduced; ++i) {
      if (!row_alive_[i] || rlb_[i] != rub_[i] || row_size_[i] < 2) continue;
      const double b = rlb_[i];
      if (!(std::abs(b) < kBig)) continue;
      live_row(i, r);
      double amax = 0.0, amin = kInf, minact = 0.0, maxact = 0.0;
      int nmin_inf = 0, nmax_inf = 0;
      bool all_int = true, any_single = false;
      for (const Entry& e : r) {
        const Index j = e.idx;
        amax = std::max(amax, std::abs(e.val));
        amin = std::min(amin, std::abs(e.val));
        all_int = all_int && is_int_[j];
        any_single = any_single || col_size_[j] == 1;
        const double bl = e.val > 0 ? lb_[j] : ub_[j], bh = e.val > 0 ? ub_[j] : lb_[j];
        if (small_bound(bl)) minact += e.val * bl; else ++nmin_inf;
        if (small_bound(bh)) maxact += e.val * bh; else ++nmax_inf;
      }
      // Integer pivots must divide every coefficient and b, hence have the smallest magnitude.
      bool int_ok = all_int && integral(b / amin);
      for (size_t t = 0; t < r.size() && int_ok; ++t) int_ok = integral(r[t].val / amin);
      int best = -1;
      long long best_score = 0;
      bool best_lo = false, best_hi = false;  // whether the rows imply lb_j / ub_j
      bool best_by_rows = false;              // ... both, with the column's other rows
      const bool long_row = row_size_[i] > kMaxSubstRow;
      if (long_row && !any_single) continue;
      // Pass 0: column singletons; pass 1 (none qualified; short rows only): the other columns.
      for (int pass = long_row ? 0 : (any_single ? 0 : 1); pass < (long_row ? 1 : 2) && best < 0; ++pass) {
        for (size_t t = 0; t < r.size(); ++t) {
          const Index j = r[t].idx;
          const double a = r[t].val;
          const bool single = col_size_[j] == 1;
          if (single != (pass == 0)) continue;
          if (keep_binaries_ && is_int_[j] && lb_[j] == 0.0 && ub_[j] == 1.0) {
            ++deferred_;
            continue;
          }
          // Continuous pivots need a coefficient within 1e-2 of the largest (stability of the rows they are
          // substituted into), except implied free singletons (checked below): their substitution only moves
          // their cost to the others and drops the row (blp-ic98's objective row: z with 1 vs 1973).
          const bool small = !is_int_[j] && std::abs(a) < 1e-2 * amax;
          if (is_int_[j] ? !(int_ok && std::abs(a) == amin) : (small && !single)) continue;
          if (!single && col_size_[j] > kMaxSubstCol) continue;
          if (static_cast<long long>(r.size() - 1) * (col_size_[j] - 1) > kMaxSubstFill) continue;
          // Implied bounds of x_j from the row: a x_j = b - (activity of the others).
          const double bl = a > 0 ? lb_[j] : ub_[j], bh = a > 0 ? ub_[j] : lb_[j];
          const bool rmin_ok = nmin_inf - (small_bound(bl) ? 0 : 1) == 0;
          const bool rmax_ok = nmax_inf - (small_bound(bh) ? 0 : 1) == 0;
          const double rmin = minact - (small_bound(bl) ? a * bl : 0.0);
          const double rmax = maxact - (small_bound(bh) ? a * bh : 0.0);
          // a > 0: x_j >= (b - rmax) / a, x_j <= (b - rmin) / a; a < 0: the other way round.
          double ilo = -kInf, ihi = kInf;
          if (a > 0 ? rmax_ok : rmin_ok) ilo = (b - (a > 0 ? rmax : rmin)) / a;
          if (a > 0 ? rmin_ok : rmax_ok) ihi = (b - (a > 0 ? rmin : rmax)) / a;
          if (is_int_[j]) {
            ilo = std::ceil(ilo - 1e-6);
            ihi = std::floor(ihi + 1e-6);
          }
          bool lo = !std::isfinite(lb_[j]) || ilo >= lb_[j] - 1e-9 * (1.0 + std::abs(lb_[j]));
          bool hi = !std::isfinite(ub_[j]) || ihi <= ub_[j] + 1e-9 * (1.0 + std::abs(ub_[j]));
          if (small && !(lo && hi)) continue;
          // The bounds the row does not imply may be implied by the column's other rows.
          bool by_rows = false;
          if (!single && !(lo && hi) && opt_.substitution_implied) {
            implied_by_rows(j, i, a, lo, hi);
            by_rows = lo && hi;
          }
          // Net change of nonzeros (limits above).
          const long long score =
              single ? (lo && hi ? -static_cast<long long>(row_size_[i]) : -1)
                     : new_entries(j, i, r) - col_size_[j] - (lo && hi ? row_size_[i] - 1 : 0);
          const bool limited = !(lo && hi) || by_rows;
          if (limited && score > (is_int_[j] ? 0 : opt_.substitution_fill)) continue;
          if (!(lo && hi) && !single && !is_int_[j] && cost_[j] != 0.0 && !opt_.substitution_bounded) continue;
          if (!(lo && hi) && score > 0 && (defer_fill_ || has_variable_bound(j))) {
            deferred_ += defer_fill_;
            continue;
          }
          if (best < 0 || score < best_score) {
            best = static_cast<int>(t), best_score = score, best_lo = lo, best_hi = hi, best_by_rows = by_rows;
          }
        }
      }
      if (best < 0) continue;
      stats_.subst_implied += best_by_rows;
      stats_.subst_fill += !(best_lo && best_hi) && best_score > 0;
      const Index j = r[best].idx;
      const double a = r[best].val;
      terms.clear();
      for (const Entry& e : r) {
        if (e.idx != j) terms.push_back(e);
      }
      if (log_.verbosity() >= 4) {
        log_.log(4, "  substitute col %d from row %d (size %d, implied %d%d)\n", j, i, col_size_[j], best_lo, best_hi);
      }
      // The other columns keep the row: their activity is b - a x_j for x_j in [lb_j, ub_j].
      const double side_ub = std::isfinite(ub_[j]) && !best_hi ? b - a * ub_[j] : (a > 0 ? -kInf : kInf);
      const double side_lb = std::isfinite(lb_[j]) && !best_lo ? b - a * lb_[j] : (a > 0 ? kInf : -kInf);
      for (const Entry& x : cols_[j]) row_act_[x.idx].stamp = -1;  // the rows that change (row i among them)
      substitute(j, b, a, terms, i);
      rlb_[i] = a > 0 ? side_ub : side_lb;
      rub_[i] = a > 0 ? side_lb : side_ub;
      if (!std::isfinite(rlb_[i]) && !std::isfinite(rub_[i])) remove_row(i);
      ++stats_.substitutions;
      ++ch;
    }
    return ch;
  }

  // Sparsification (Achterberg et al. 2020, two-row nonzero cancellation; HiGHS HPresolve::sparsify): an equation e
  // of at most kMaxSparsifyRow entries is added to a row r that shares columns with it, r += lambda e, when lambda
  // cancels more entries of r than the columns of e missing in r add. The candidates are the rows of e's shortest
  // column (of at least two rows), as in HiGHS: the rows of e's ceil(len / 2) shortest columns (every row that can
  // gain) found more combinations, but they coupled rows probing relies on (neos-3216931-puriri: 12.3% -> 1.0% of
  // the root gap closed by the presolved LP; mzzv11 35.8% -> 33.1%). lambda is the ratio -a_rk / a_ek shared by the
  // most entries (ties: the smallest |lambda|), at most kMaxSparsifyScale in magnitude; r's coefficient range may not
  // grow beyond kMaxSparsifyRange, and a row of integer columns with integral coefficients stays one (GCD
  // tightening, knapsack cuts). Clique structure is kept: all-binary set partitioning equations are not used, and set
  // packing / covering / partitioning rows (binaries, equal |a|) are not changed (puriri: x1 + .. + x5 - y + z = 1
  // would turn x1 + .. + x5 <= 1 into y - z <= 0, and probing then closes 1.7% of the root gap instead of 12.3%).
  // Shortest equations first. Rows only: the feasible set is the same, no postsolve (duals come from the original
  // problem, solver/duals.cpp). Cancelled entries are those within 1e-13 relative of zero (round-off) and are dropped.
  // Only the row storage changes during the pass: the candidates come from the column storage as it was at the start
  // (an entry may stand for a row that has lost the column; rows that gained it are missed), rebuilt at the end.
  long long sparsify() {
    Timer timer;
    std::vector<Index> src;
    long long nnz = 0;
    for (Index i = 0; i < m_; ++i) {
      if (!row_alive_[i]) continue;
      nnz += row_size_[i];
      if (rlb_[i] == rub_[i] && row_size_[i] >= 2 && row_size_[i] <= kMaxSparsifyRow) src.push_back(i);
    }
    std::stable_sort(src.begin(), src.end(), [&](Index a, Index b) { return row_size_[a] < row_size_[b]; });
    const long long limit = 20 * nnz + 1000000;
    long long work = 0, combos = 0, removed = 0, protected_eqs = 0;
    // Continuous columns with a variable bound: in an inequality with integer columns whose continuous coefficients
    // all have one sign and the integer ones the other (x <= u y; sum of flows <= 7 y1 + 31 y2 + 255 y3). Equations
    // with such a column are not used (sparsify_protect): the cut separators aggregate through these columns
    // (c-MIR paths over flow balances, flow covers), and combined rows hide them (sp150x300d: root bound 69 -> 65,
    // aggregated covers 50 -> 34, MIR 76 -> 40; n5-3: the arc capacity rows lost their flows to z = sum of flows).
    std::vector<char> vb;
    if (opt_.sparsify_protect) {
      vb.assign(n_, 0);
      for (Index i = 0; i < m_; ++i) {
        if (!row_alive_[i] || rlb_[i] == rub_[i] || row_size_[i] < 2) continue;
        int cs = 0, is = 0;
        bool ok = true;
        for (const Entry& x : rows_[i]) {
          int& sg = is_int_[x.idx] ? is : cs;
          const int v = x.val > 0 ? 1 : -1;
          if (sg == 0) sg = v;
          ok = ok && sg == v;
        }
        if (!ok || cs == 0 || is == 0 || cs == is) continue;
        for (const Entry& x : rows_[i]) {
          if (!is_int_[x.idx]) vb[x.idx] = 1;
        }
      }
    }
    std::vector<Index> emark(n_, -1), epos(n_, 0), seen(m_, -1);
    std::vector<Entry> er;
    std::vector<double> rv, nv;
    std::vector<char> shared;
    std::vector<std::pair<double, Index>> ratios;
    // r += lambda e if that pays (e: er, right-hand side b; emark / epos: its columns).
    auto combine = [&](Index e, double b, Index r) {
      const Index len = static_cast<Index>(er.size());
      std::vector<Entry>& row = rows_[r];
      work += static_cast<long long>(row.size()) + len;
      // r's entries in the columns of e (rv), its integrality, clique form and coefficient range (all entries; the
      // others).
      rv.assign(len, 0.0);
      shared.assign(len, 0);
      Index nshared = 0;
      bool r_int = true, r_clique = true;
      double omin = kInf, omax = 0.0, kmin = kInf, kmax = 0.0;
      for (const Entry& x : row) {
        const double a = std::abs(x.val);
        omin = std::min(omin, a);
        omax = std::max(omax, a);
        r_int = r_int && is_int_[x.idx] && x.val == std::round(x.val);
        r_clique = r_clique && is_int_[x.idx] && lb_[x.idx] == 0.0 && ub_[x.idx] == 1.0 && a == std::abs(row[0].val);
        if (emark[x.idx] == e) {
          rv[epos[x.idx]] = x.val;
          shared[epos[x.idx]] = 1;
          ++nshared;
        } else {
          kmin = std::min(kmin, a);
          kmax = std::max(kmax, a);
        }
      }
      const Index misses = len - nshared;
      if (nshared < 2 || nshared <= misses || r_clique) return;
      ratios.clear();
      for (Index t = 0; t < len; ++t) {
        if (shared[t]) ratios.push_back({-rv[t] / er[t].val, t});
      }
      std::sort(ratios.begin(), ratios.end());
      double lambda = 0.0;
      Index best = 0;
      for (size_t s = 0; s < ratios.size();) {
        size_t u = s + 1;
        while (u < ratios.size() && ratios[u].first - ratios[s].first <= 1e-13 * std::abs(ratios[s].first)) ++u;
        double lam = ratios[s].first;
        if (std::abs(lam - std::round(lam)) <= 1e-13 * std::abs(lam)) lam = std::round(lam);
        const Index cnt = static_cast<Index>(u - s);
        if (std::abs(lam) <= kMaxSparsifyScale && (!r_int || lam == std::round(lam)) &&
            (cnt > best || (cnt == best && std::abs(lam) < std::abs(lambda)))) {
          lambda = lam;
          best = cnt;
        }
        s = u;
      }
      if (best <= misses) return;
      // The combined row: cancellations, integrality, coefficient range.
      nv.assign(len, 0.0);
      Index cancel = 0;
      bool ok = true;
      double nmin = kmin, nmax = kmax;
      for (Index t = 0; t < len && ok; ++t) {
        const double ae = lambda * er[t].val;
        double v = rv[t] + ae;
        if (std::abs(v) <= 1e-13 * std::max(std::abs(rv[t]), std::abs(ae))) v = 0.0;
        if (r_int && v != 0.0) {
          ok = is_int_[er[t].idx] && integral(v) && std::abs(v) <= 1e9;
          v = std::round(v);
        }
        if (v == 0.0) {
          cancel += shared[t];
        } else {
          nmin = std::min(nmin, std::abs(v));
          nmax = std::max(nmax, std::abs(v));
        }
        nv[t] = v;
      }
      if (!ok || cancel <= misses) return;
      if (nmax > kMaxSparsifyRange * nmin && nmax * omin > omax * nmin) return;
      double db = lambda * b;
      if (r_int) {
        if (!integral(db)) return;
        db = std::round(db);
      }
      if (log_.verbosity() >= 4) {
        log_.log(4, "  sparsify: row %d += %.12g row %d (%d cancelled, %d new)\n", r, lambda, e, cancel, misses);
      }
      Index added = 0;
      for (Entry& x : row) {
        if (emark[x.idx] == e) x.val = nv[epos[x.idx]];
        if (x.val == 0.0) --col_size_[x.idx];
      }
      row.erase(std::remove_if(row.begin(), row.end(), [](const Entry& x) { return x.val == 0.0; }), row.end());
      for (Index t = 0; t < len; ++t) {
        if (shared[t] || nv[t] == 0.0) continue;
        row.push_back({er[t].idx, nv[t]});
        ++col_size_[er[t].idx];
        ++added;
      }
      row_size_[r] = static_cast<Index>(row.size());
      if (std::isfinite(rlb_[r])) rlb_[r] += db;
      if (std::isfinite(rub_[r])) rub_[r] += db;
      ++combos;
      removed += cancel - added;
    };
    for (const Index e : src) {
      if (work >= limit || interrupted()) break;
      if (!row_alive_[e] || rlb_[e] != rub_[e]) continue;
      const double b = rlb_[e];
      if (!small_bound(b)) continue;
      live_row(e, er);
      const Index len = static_cast<Index>(er.size());
      if (len < 2 || len > kMaxSparsifyRow) continue;
      bool part = b == er[0].val;
      for (const Entry& x : er) {
        part = part && x.val == er[0].val && is_int_[x.idx] && lb_[x.idx] == 0.0 && ub_[x.idx] == 1.0;
      }
      if (part) continue;
      if (!vb.empty() && std::any_of(er.begin(), er.end(), [&](const Entry& x) { return vb[x.idx] != 0; })) {
        ++protected_eqs;
        continue;
      }
      Index c = -1;
      for (Index t = 0; t < len; ++t) {
        const Index j = er[t].idx;
        emark[j] = e;
        epos[j] = t;
        if (col_size_[j] >= 2 && (c < 0 || col_size_[j] < col_size_[c])) c = j;
      }
      work += len;
      if (c < 0) continue;
      for (const Entry& ce : cols_[c]) {
        if (work >= limit) break;
        const Index r = ce.idx;
        if (r == e || !row_alive_[r] || seen[r] == e) continue;
        seen[r] = e;
        combine(e, b, r);
      }
    }
    if (combos > 0) {
      for (Index j = 0; j < n_; ++j) cols_[j].clear();
      for (Index i = 0; i < m_; ++i) {
        if (!row_alive_[i]) continue;
        for (const Entry& x : rows_[i]) cols_[x.idx].push_back({i, x.val});
      }
      for (Index j = 0; j < n_; ++j) {
        if (col_alive_[j]) col_size_[j] = static_cast<Index>(cols_[j].size());
      }
    }
    ++stats_.sparsify_runs;
    stats_.sparsify_rows += combos;
    stats_.sparsify_removed += removed;
    stats_.sparsify_protected += protected_eqs;
    log_.log(3, "Presolve sparsify: %lld rows combined, %lld nonzeros removed, %zu equations (%lld skipped for "
             "variable-bound columns), work %lld / %lld, %.3fs\n", combos, removed, src.size(), protected_eqs, work, limit,
             timer.elapsed());
    return combos;
  }

  // Propagation over one-of groups with shaving of integer cost columns (group_prop.hpp); applies the bounds.
  long long group_prop() {
    MipProblem q;
    std::vector<Index> cols;
    assemble(q, cols);
    const GroupPropResult gr = group_propagate(q);
    if (gr.groups == 0) {
      stats_.group_runs = 99;  // no groups: no further run
      return 0;
    }
    if (gr.infeasible) {
      status_ = PresolveStatus::Infeasible;
      log_.log(3, "Presolve: one-of group propagation proved infeasibility\n");
      return 0;
    }
    long long ch = 0;
    for (size_t t = 0; t < cols.size() && status_ == PresolveStatus::Reduced; ++t) {
      const Index j = cols[t];
      if (gr.lb[t] <= lb_[j] && gr.ub[t] >= ub_[j]) continue;
      ch += tighten(j, gr.lb[t], gr.ub[t], 0.0);
    }
    stats_.group_fixed += gr.fixed;
    stats_.group_bounds += gr.tightened;
    stats_.group_shaved += gr.shaved;
    log_.log(3, "Presolve one-of groups: %d groups, %d members fixed, %d bounds, %d shaving steps (%d probes), work "
             "%lld\n", gr.groups, gr.fixed, gr.tightened, gr.shaved, gr.shave_probes, gr.work);
    return ch;
  }

  // Probing on the binaries of the current problem; applies fixings, bounds, liftings and equivalences.
  // *useful: fixings + equivalences + liftings.
  long long probing(double effort, long long* useful) {
    MipProblem q;
    std::vector<Index> cols;
    assemble(q, cols);
    long long limit = static_cast<long long>(effort * std::min(5e8, 5e7 + 2000.0 * q.A.nnz()));
    if (stats_.probing_passes > 1) limit = std::min(limit, last_probe_work_);
    ProbingOptions po;
    po.lift = opt_.probing_lift;
    po.lift_dyadic = opt_.lift_dyadic;
    // Default: as many candidates side by side as there are threads, at most 4 (measured at 4 on 4 threads);
    // one thread probes one candidate at a time (a batch on one thread costs more than it finds).
    po.batch = opt_.probing_batch >= 0 ? opt_.probing_batch : opt_.threads >= 2 ? std::min(4, opt_.threads) : 0;
    po.threads = opt_.threads;
    // MILP_PROBE_STOP=1 (trial, off by default): a pass gives up after a shorter stretch without a useful probe,
    // 5% of its work limit in the first pass and 2.5% in the later ones instead of 10% (each counted twice: the
    // order is reversed once). nw04's first pass used 0.66 s for one fixing, and a later pass that finds nothing
    // costs 20% of the pass before it (neos-957323 0.17 s, k1mushroom 0.14 s).
    static const bool probe_stop = [] { const char* v = std::getenv("MILP_PROBE_STOP"); return v && v[0] == '1'; }();
    if (probe_stop) po.idle_frac = stats_.probing_passes > 1 ? 0.025 : 0.05;
    // Implications between binaries (conflicting literal pairs), kept over the passes: at most 8 per nonzero
    // (3e6 pairs, ~50 MB with the prober's adjacency lists; mzzv42z: 1.2e6). The previous passes' ones seed the
    // propagation (live, unmerged columns still binary).
    std::vector<std::pair<Index, Index>> seeds;
    if (opt_.probing_implications) {
      po.max_conflicts = std::min<size_t>(3000000, 8 * static_cast<size_t>(q.A.nnz()) + 100000);
      std::vector<Index> inv(n_, -1);
      for (size_t t = 0; t < cols.size(); ++t) inv[cols[t]] = static_cast<Index>(t);
      for (const auto& [a, b] : probe_lits_) {
        const Index u = inv[a >> 1], v = inv[b >> 1];
        if (u >= 0 && v >= 0 && !merged_[a >> 1] && !merged_[b >> 1]) seeds.push_back({2 * u + (a & 1), 2 * v + (b & 1)});
      }
      po.seeds = &seeds;
    }
    const ProbingResult pr = probe_binaries(q, limit, po);
    stats_.probes += pr.probes;
    last_probe_work_ = pr.work;
    last_probe_limit_ = limit;
    new_conflicts_ = 0;
    if (!pr.conflicts.empty()) {
      for (const auto& [a, b] : pr.conflicts) {
        const Index u = 2 * cols[a >> 1] + (a & 1), v = 2 * cols[b >> 1] + (b & 1);
        probe_lits_.push_back({std::min(u, v), std::max(u, v)});
        new_conflicts_ += (a & 1) && (b & 1);
      }
      std::sort(probe_lits_.begin(), probe_lits_.end());
      probe_lits_.erase(std::unique(probe_lits_.begin(), probe_lits_.end()), probe_lits_.end());
      stats_.probing_conflicts = static_cast<long long>(probe_lits_.size());
    }
    if (pr.infeasible) {
      status_ = PresolveStatus::Infeasible;
      log_.log(3, "Presolve: probing proved infeasibility\n");
      return 0;
    }
    long long ch = 0;
    for (size_t t = 0; t < cols.size() && status_ == PresolveStatus::Reduced; ++t) {
      const Index j = cols[t];
      if (pr.lb[t] <= lb_[j] && pr.ub[t] >= ub_[j]) continue;
      const int c = tighten(j, pr.lb[t], pr.ub[t], 0.0);
      ch += c;
      if (lb_[j] == ub_[j]) ++stats_.probing_fixed, ++*useful; else stats_.probing_bounds += c;
    }
    if (status_ != PresolveStatus::Reduced) return ch;
    // Liftings before the equivalences (they refer to the rows as probed; a substitution afterwards rewrites the
    // lifted rows like any other).
    const long long lifted = apply_lifts(pr, q.A.nnz(), cols);
    ch += lifted;
    *useful += lifted;
    // y = v0 + (v1 - v0) x; x may have been replaced by an equivalent binary: rep_of[c] = (column, negated).
    std::unordered_map<Index, std::pair<Index, bool>> rep_of;
    auto find = [&](Index c) {
      bool neg = false;
      for (auto it = rep_of.find(c); it != rep_of.end(); it = rep_of.find(c)) {
        neg = neg != it->second.second;
        c = it->second.first;
      }
      return std::make_pair(c, neg);
    };
    for (const ProbingResult::Equiv& e : pr.equiv) {
      const Index y = cols[e.y];
      const auto [x, neg] = find(cols[e.x]);
      if (y == x || !col_alive_[y] || !col_alive_[x] || lb_[x] != 0.0 || ub_[x] != 1.0) continue;
      // Both values must still be allowed for y (its bounds disappear with it).
      const double tol = is_int_[y] ? 0.0 : 1e-9;
      if (std::min(e.v0, e.v1) < lb_[y] - tol * (1.0 + std::abs(lb_[y])) ||
          std::max(e.v0, e.v1) > ub_[y] + tol * (1.0 + std::abs(ub_[y])))
        continue;
      // A column that takes (nearly) the same value in both probes is not equivalent to x: the difference is
      // round-off (a continuous column within [0.4999999999, 0.5000000002]: "y = 0.5 + 3.4e-11 x" left the
      // coefficient 3.4e-11 in a row, then x in [-29091, 29090] and sides of 2e11; fuzz seed 22, case 1184 of
      // the sparsify models, wrong since v0.10.0 behind the solver's check of the postsolved point).
      if (std::abs(e.v1 - e.v0) <= 1e-7 * std::max({1.0, std::abs(e.v0), std::abs(e.v1)})) continue;
      const double d = e.v1 - e.v0, c0 = neg ? e.v0 + d : e.v0, c1 = neg ? -d : d;  // y = c0 + c1 x
      if (log_.verbosity() >= 4) log_.log(4, "  probing: col %d = %.12g + %.12g col %d\n", y, c0, c1, x);
      substitute(y, c0, 1.0, {{x, -c1}}, -1);
      if (is_int_[y] && std::min(e.v0, e.v1) == 0.0 && std::max(e.v0, e.v1) == 1.0) rep_of[y] = {x, c1 < 0};
      ++stats_.probing_equivs;
      ++ch;
      ++*useful;
    }
    if (new_conflicts_ > 0 && opt_.probing_cliques && opt_.clique_extension && status_ == PresolveStatus::Reduced) {
      // Clique extension with the new conflicts before the fixed columns leave the rows: a packing row whose
      // members probing fixed at 0 but one is extended into a clique of that one (instead of being dropped as a
      // singleton row). Its time counts as probing time.
      ch += clique_extension();
    }
    return ch;
  }

  // Applies the liftings of a probing pass (probing.hpp; at most one per row; rows of the probed problem are the
  // live rows in order). New entries are capped at 10% of the nonzeros per pass, best (largest delta / max |a|)
  // first. Rows only: no postsolve.
  long long apply_lifts(const ProbingResult& pr, long long nnz, const std::vector<Index>& cols) {
    if (pr.lifts.empty()) return 0;
    std::vector<Index> rows;
    for (Index i = 0; i < m_; ++i) {
      if (row_alive_[i]) rows.push_back(i);
    }
    std::vector<size_t> order(pr.lifts.size());
    for (size_t t = 0; t < order.size(); ++t) order[t] = t;
    std::stable_sort(order.begin(), order.end(), [&](size_t u, size_t v) {
      const ProbingResult::Lift &a = pr.lifts[u], &b = pr.lifts[v];
      return a.fill != b.fill ? !a.fill : (a.fill && a.score > b.score);
    });
    long long fill_left = nnz / 10;
    long long ch = 0;
    for (size_t t : order) {
      const ProbingResult::Lift& l = pr.lifts[t];
      if (l.fill && fill_left <= 0) break;  // the new entries come last
      const Index i = rows[l.row], j = cols[l.col];
      if (!row_alive_[i] || !col_alive_[j]) continue;
      // The row is still as probed (one-sided; tighten() changes no row).
      if (!std::isfinite(l.upper ? rub_[i] : rlb_[i]) || std::isfinite(l.upper ? rlb_[i] : rub_[i])) continue;
      const double s = l.upper ? 1.0 : -1.0;
      double a = 0.0, amax = 0.0;
      bool present = false;
      for (const Entry& e : rows_[i]) {
        if (e.idx == j) present = true;
        if (!col_alive_[e.idx]) continue;
        if (e.idx == j) a = e.val;
        amax = std::max(amax, std::abs(e.val));
      }
      if (opt_.lift_live) {
        // A row that the fixings of this pass made redundant is not lifted: the lift would keep it alive with
        // entries of other binaries (traininstance6: 1300 big-M rows of one-of members fixed at 0 by the same
        // pass stayed as four-entry rows). The next round removes it.
        double maxact = 0.0;
        bool finite = true;
        for (const Entry& e : rows_[i]) {
          if (!col_alive_[e.idx]) continue;
          const double v = s * e.val > 0 ? ub_[e.idx] : lb_[e.idx];
          if (!std::isfinite(v)) {
            finite = false;
            break;
          }
          maxact += s * e.val * v;
        }
        const double side = s * (l.upper ? rub_[i] : rlb_[i]);
        if (finite && maxact <= side + 1e-9 * std::max(1.0, std::abs(side))) continue;
      }
      // In <= form: c x + ... <= b.
      double c = s * a + (l.one ? l.delta : -l.delta);
      double b = s * (l.upper ? rub_[i] : rlb_[i]) - (l.one ? 0.0 : l.delta);
      if (std::abs(c) <= 1e-9 * amax) {
        if (c < 0.0) b -= c;  // dropping c x (x <= 1) relaxes the row by at most |c|
        c = 0.0;
      }
      if (log_.verbosity() >= 4) {
        log_.log(4, "  probing: lift row %d col %d (x = %d): coef %.12g -> %.12g, side %.12g -> %.12g\n", i, j,
                 l.one ? 1 : 0, a, s * c, l.upper ? rub_[i] : rlb_[i], s * b);
      }
      if (present || c == 0.0) {
        set_coef(i, j, s * c);
      } else {
        // A new entry (known absent from the row scan): appended without set_coef's search of column j,
        // which the lifts of one probed column fill quadratically (ns1760995: ~200k new entries per pass,
        // 15 s per pass).
        rows_[i].push_back({j, s * c});
        cols_[j].push_back({i, s * c});
        ++row_size_[i];
        ++col_size_[j];
      }
      if (l.upper) rub_[i] = b; else rlb_[i] = -b;
      if (l.fill && c != 0.0) --fill_left, ++stats_.probing_lift_fill;
      ++stats_.probing_lifts;
      ++ch;
    }
    return ch;
  }

  // Clique extension of set-packing rows (clique_merge.hpp): fixings, added entries, dominated rows dropped.
  long long clique_extension() {
    MipProblem q;
    std::vector<Index> cols, rows;
    assemble(q, cols);
    for (Index i = 0; i < m_; ++i) {
      if (row_alive_[i]) rows.push_back(i);
    }
    static const double scale = [] {
      const char* v = std::getenv("MILP_CLIQUE_EXT_WORK");
      return v ? std::atof(v) : 1.0;
    }();
    const long long limit = static_cast<long long>(scale * (1.2e8 + 400.0 * q.A.nnz()));
    Timer timer;
    // Probing conflicts x_u + x_v <= 1 between live, unmerged columns (a merged parallel column stands for a sum).
    std::vector<std::pair<Index, Index>> edges;
    if (!probe_lits_.empty() && opt_.probing_cliques) {
      std::vector<Index> inv(n_, -1);
      for (size_t t = 0; t < cols.size(); ++t) inv[cols[t]] = static_cast<Index>(t);
      for (const auto& [a, b] : probe_lits_) {
        if (!(a & 1) || !(b & 1)) continue;
        const Index u = a >> 1, v = b >> 1;
        if (inv[u] >= 0 && inv[v] >= 0 && !merged_[u] && !merged_[v]) edges.push_back({inv[u], inv[v]});
      }
    }
    const CliqueExtension ce = find_clique_extensions(q, limit, &edges);
    const CliqueExtendStats& cs = ce.stats;
    stats_.clique_extended += cs.extended;
    stats_.clique_dominated += cs.dominated;
    stats_.clique_fixed += cs.fixed;
    log_.log(3, "Clique extension: %d cliques, %d rows extended by %lld entries, %d dominated, %d fixed, work %lld, %.3fs\n",
             cs.cliques, cs.extended, cs.added, cs.dominated, cs.fixed, cs.work, timer.elapsed());
    long long ch = 0;
    for (Index t : ce.fixed) ch += tighten(cols[t], lb_[cols[t]], 0.0, 0.0);
    for (Index r : ce.removed) remove_row(rows[r]), ++ch;
    for (const auto& [r, add] : ce.ext) {
      const Index i = rows[r];
      if (!row_alive_[i]) continue;
      for (Index t : add) {
        const Index j = cols[t];
        if (col_alive_[j]) set_coef(i, j, 1.0);
      }
      rlb_[i] = -kInf;
      ++ch;
    }
    return ch;
  }

  void build() { assemble(out_.reduced_, out_.col_map_); }

  // The live problem as a MipProblem; cols receives the original index of each column.
  void assemble(MipProblem& q, std::vector<Index>& cols) const {
    q = MipProblem{};
    std::vector<Index> rmap(m_, -1), cmap(n_, -1);
    Index nr = 0, nc = 0;
    for (Index i = 0; i < m_; ++i) {
      if (row_alive_[i]) rmap[i] = nr++;
    }
    cols.clear();
    for (Index j = 0; j < n_; ++j) {
      if (!col_alive_[j]) continue;
      cmap[j] = nc++;
      cols.push_back(j);
    }
    std::vector<Index> tc, tr;
    std::vector<double> tv;
    for (Index j = 0; j < n_; ++j) {
      if (!col_alive_[j]) continue;
      q.cost.push_back(cost_[j]);
      q.col_lb.push_back(lb_[j]);
      q.col_ub.push_back(ub_[j]);
      q.is_int.push_back(is_int_[j]);
      for (const Entry& e : cols_[j]) {
        if (!row_alive_[e.idx] || e.val == 0.0) continue;
        tc.push_back(cmap[j]);
        tr.push_back(rmap[e.idx]);
        tv.push_back(e.val);
      }
    }
    for (Index i = 0; i < m_; ++i) {
      if (!row_alive_[i]) continue;
      q.row_lb.push_back(rlb_[i]);
      q.row_ub.push_back(rub_[i]);
    }
    q.A = SparseMatrix::from_triplets(nc, nr, tc, tr, tv);
    q.offset = offset_;
    q.display_sense = display_sense_;
  }

  const PresolveOptions& opt_;
  const Logger& log_;
  Timer run_timer_;  // from construction (run() follows at once): PresolveOptions::time_limit
  Presolver& out_;
  PresolveStats& stats_;
  PresolveStatus status_ = PresolveStatus::Reduced;
  Index m_ = 0, n_ = 0;
  std::vector<std::vector<Entry>> rows_, cols_;  // rows: (col, val); cols: (row, val)
  std::vector<char> row_alive_, col_alive_;
  std::vector<Index> row_size_, col_size_;  // live entries
  std::vector<double> lb_, ub_, cost_, rlb_, rub_;
  std::vector<char> is_int_;
  std::vector<Index> mark_;  // scratch (new_entries)
  mutable std::vector<RowAct> row_act_;  // implied_by_rows
  mutable int row_act_stamp_ = 0;
  // Scratch of parallel_rows / parallel_cols: normalized vectors in one array (off: start of each), first
  // values before scaling, (hash, index) pairs.
  std::vector<Entry> par_norm_;
  std::vector<Index> par_off_;
  std::vector<double> par_first_;
  std::vector<std::pair<uint64_t, Index>> par_keys_;
  double offset_ = 0.0, display_sense_ = 1.0;
  double step_time_[15] = {};
  // Implications between binaries found by probing, as conflicting literal pairs a < b (2 j + v: x_j = v; columns of
  // this problem): seeds of later passes; the pairs of two literals x_j = 1 extend the cliques. Columns that
  // absorbed a parallel column (merged_) no longer stand for themselves.
  std::vector<std::pair<Index, Index>> probe_lits_;
  long long new_conflicts_ = 0;  // x_u + x_v <= 1 pairs added by the last probing pass
  long long last_probe_work_ = 0, last_probe_limit_ = 0;
  // Binaries stay until the first probing pass (no substitution): probing may fix them, e.g. y = sum x_k with
  // y = 0 infeasible gives sum x_k = 1, while y substituted out leaves only 0 <= sum x_k <= 1 (puriri).
  bool keep_binaries_ = false;
  // Substitutions that add nonzeros also wait for the first probing pass: they changed the rows probing works on
  // (neos-1445765: 18423 -> 16283 equivalences, presolved 955 x 1134 -> 1077 x 3465).
  bool defer_fill_ = false;
  long long deferred_ = 0;  // substitutions skipped for these two
  std::vector<char> merged_;
  std::vector<double> minact_, maxact_;  // row activity snapshot (dual_fixing)
  std::vector<int> nmin_inf_, nmax_inf_;
};

PresolveOptions PresolveOptions::from_env() {
  PresolveOptions o;
  if (const char* e = std::getenv("MILP_PROBING_EFFORT")) o.probing_effort = std::atof(e);
  if (const char* e = std::getenv("MILP_PROBING_PASSES")) o.probing_passes = std::atoi(e);
  if (const char* e = std::getenv("MILP_PROBE_PAR")) o.probing_batch = std::atoi(e);
  if (const char* e = std::getenv("MILP_GROUP_PROP")) o.group_prop = std::atoi(e) != 0;
  if (const char* e = std::getenv("MILP_INT_DIV")) o.int_divisors = std::atoi(e) != 0;
  if (const char* e = std::getenv("MILP_KEEP_MEMBERS")) o.keep_members = std::atoi(e) != 0;
  if (const char* e = std::getenv("MILP_LIFT_LIVE")) o.lift_live = std::atoi(e) != 0;
  if (const char* e = std::getenv("MILP_LIFT_DYADIC")) o.lift_dyadic = std::atoi(e) != 0;
  if (const char* e = std::getenv("MILP_UNIT_ROWS")) o.unit_rows = std::atoi(e) != 0;
  if (const char* on = std::getenv("MILP_PRESOLVE_ON")) {
    if ((std::string(",") + on + ",").find(",subst_bounded,") != std::string::npos) o.substitution_bounded = true;
    if ((std::string(",") + on + ",").find(",probing_cliques,") != std::string::npos) o.probing_cliques = true;
    if ((std::string(",") + on + ",").find(",sparsify,") != std::string::npos) o.sparsify = true;
    if ((std::string(",") + on + ",").find(",sparsify_protect,") != std::string::npos) o.sparsify_protect = true;
  }
  const char* env = std::getenv("MILP_PRESOLVE_OFF");
  if (!env) return o;
  const std::string s = std::string(",") + env + ",";
  auto off = [&](const char* name) {
    return s.find(std::string(",") + name + ",") != std::string::npos || s.find(",all,") != std::string::npos;
  };
  if (off("bounds")) o.bound_tightening = false;
  if (off("coef")) o.coef_tightening = false;
  if (off("parallel")) o.parallel_rows = false;
  if (off("parallel_cols")) o.parallel_cols = false;
  if (off("dominated")) o.dominated_cols = false;
  if (off("dual")) o.dual_fixing = false;
  if (off("dual_agg")) o.dual_aggregation = false;
  if (off("aggregation")) o.aggregation = false;
  if (off("cliques")) o.clique_merging = false;
  if (off("cliqueext")) o.clique_extension = false;
  if (off("substitution")) o.substitution = false;
  if (off("subst_implied")) o.substitution_implied = false;
  if (off("subst_fill")) o.substitution_fill = 0;
  if (off("probing")) o.probing = false;
  if (off("probing_lift")) o.probing_lift = false;
  if (off("probing_passes")) o.probing_passes = 1;
  if (off("probing_implications")) o.probing_implications = false;
  if (off("probing_keep")) o.probing_keep_binaries = false;
  if (off("sparsify")) o.sparsify = false;
  if (off("sparsify_protect")) o.sparsify_protect = false;
  if (off("implint")) o.implied_int = false;
  if (off("implint_dual")) o.implied_int_dual = false;
  return o;
}

PresolveStatus Presolver::run(const MipProblem& in, const PresolveOptions& opt, const Logger& log) {
  Timer timer;
  reduced_ = MipProblem{};
  stats_ = PresolveStats{};
  stack_.clear();
  col_map_.clear();
  orig_cols_ = in.num_cols();
  orig_is_int_ = in.is_int;
  stats_.rows_before = in.num_rows();
  stats_.cols_before = in.num_cols();
  stats_.nnz_before = in.A.nnz();
  PresolveWork work(in, opt, log, *this);
  const PresolveStatus st = work.run();
  if (st == PresolveStatus::Reduced && opt.clique_merging) {
    const Index rows = reduced_.num_rows();
    const Index removed = merge_cliques(reduced_);
    if (removed > 0) {
      log.log(2, "Clique merging: %d set-packing rows replaced by %d clique rows\n", removed,
              reduced_.num_rows() - rows + removed);
    }
  }
  if (st == PresolveStatus::Reduced && opt.implied_int) {
    // Flags of the input carry over to the kept columns (their values are those of the input's columns),
    // except for columns that absorbed a merged parallel column.
    if (!in.implied_int.empty()) {
      std::vector<char> merged(static_cast<size_t>(orig_cols_), 0);
      for (const Step& s : stack_) {
        if (s.other >= 0) merged[s.other] = 1;
      }
      reduced_.implied_int.assign(col_map_.size(), 0);
      for (size_t k = 0; k < col_map_.size(); ++k) {
        reduced_.implied_int[k] = in.is_implied_int(col_map_[k]) && !merged[col_map_[k]] && !reduced_.is_int[k];
      }
    }
    ImpliedIntStats is;
    stats_.implied_int = mark_implied_integers(reduced_, opt.implied_int_dual, &is);
    stats_.implied_primal = is.primal;
    stats_.implied_dual = is.dual;
    bool obj_int = true;
    for (Index j = 0; j < reduced_.num_cols() && obj_int; ++j) {
      const double c = reduced_.cost[j];
      obj_int = c == 0.0 || ((reduced_.is_int[j] || reduced_.is_implied_int(j)) && c == std::round(c));
    }
    log.log(2, "Presolve implied integers: %lld columns (new: %lld primal, %lld dual rule); objective %s\n",
            stats_.implied_int, stats_.implied_primal, stats_.implied_dual, obj_int ? "integral" : "not integral");
  }
  stats_.rows_after = reduced_.num_rows();
  stats_.cols_after = reduced_.num_cols();
  stats_.nnz_after = reduced_.A.nnz();
  stats_.time = timer.elapsed();
  log.log(2,
          "Presolve details: %d rounds, %lld fixed cols, %lld removed rows, %lld bound changes, %lld coef changes, "
          "%lld parallel rows, %lld parallel cols, %lld dominated cols, %lld dual bounds, %lld aggregations, "
          "%lld substitutions, %lld dual aggregations\n",
          stats_.rounds, stats_.fixed_cols, stats_.removed_rows, stats_.bound_changes, stats_.coef_changes,
          stats_.parallel_rows, stats_.parallel_cols, stats_.dominated, stats_.dual_bounds, stats_.aggregations,
          stats_.substitutions, stats_.dual_aggregations);
  if (opt.substitution && (opt.substitution_implied || opt.substitution_fill > 0)) {
    log.log(2, "Presolve substitutions: %lld implied free by other rows, %lld kept rows grew\n", stats_.subst_implied,
            stats_.subst_fill);
  }
  if (opt.sparsify) {
    log.log(2, "Presolve sparsify: %lld rows combined, %lld nonzeros removed, %d runs, %lld equations skipped for "
            "variable-bound columns (summed over the runs)\n", stats_.sparsify_rows, stats_.sparsify_removed,
            stats_.sparsify_runs, stats_.sparsify_protected);
  }
  if (opt.clique_extension) {
    log.log(2, "Presolve cliques: %lld rows extended, %lld dominated rows, %lld fixed\n", stats_.clique_extended,
            stats_.clique_dominated, stats_.clique_fixed);
  }
  if (opt.group_prop && stats_.group_runs > 0) {
    log.log(2, "Presolve one-of groups: %lld members fixed, %lld bounds, %lld shaving steps, %d runs\n", stats_.group_fixed,
            stats_.group_bounds, stats_.group_shaved, stats_.group_runs == 99 ? 0 : stats_.group_runs);
  }
  if (opt.int_divisors && stats_.int_divisors > 0) {
    log.log(2, "Presolve integer divisors: %lld columns rescaled\n", stats_.int_divisors);
  }
  if (opt.probing) {
    log.log(2, "Presolve probing: %lld probes, %lld fixed, %lld bounds, %lld equivalences, %lld lifted rows (%lld new "
            "entries), %lld conflicts, %d passes\n", stats_.probes, stats_.probing_fixed, stats_.probing_bounds,
            stats_.probing_equivs, stats_.probing_lifts, stats_.probing_lift_fill, stats_.probing_conflicts,
            stats_.probing_passes);
  }
  return st;
}

std::vector<double> Presolver::postsolve(const std::vector<double>& x_reduced) const {
  std::vector<double> x(orig_cols_, 0.0);
  // Near-integral values are not rounded: that could break rows the reduced solution
  // satisfies only thanks to the tiny deviation.
  // Small bound violations are clamped: a derived bound may stand for a row with a large coefficient.
  for (size_t k = 0; k < col_map_.size(); ++k) {
    x[col_map_[k]] = std::min(std::max(x_reduced[k], reduced_.col_lb[k]), reduced_.col_ub[k]);
  }
  for (auto it = stack_.rbegin(); it != stack_.rend(); ++it) {
    if (it->other >= 0) {
      // y = x_k + s x_j: x_j in [lb, ub] with x_k = y - s x_j in [olb, oub].
      const double y = x[it->other], s = it->coef;
      double lo = s > 0 ? (y - it->oub) / s : (y - it->olb) / s, hi = s > 0 ? (y - it->olb) / s : (y - it->oub) / s;
      if (it->integer) {
        lo = std::ceil(lo - 1e-9);
        hi = std::floor(hi + 1e-9);
      }
      lo = std::max(lo, it->lb);
      hi = std::min(hi, it->ub);
      double xj = std::isfinite(lo) ? lo : (std::isfinite(hi) ? std::min(hi, 0.0) : 0.0);
      if (lo > hi) xj = std::abs(lo - it->lb) < std::abs(hi - it->ub) ? hi : lo;  // y slightly out of range
      x[it->col] = xj;
      x[it->other] = y - s * xj;
      continue;
    }
    double v = it->rhs;
    for (const auto& t : it->terms) v -= t.second * x[t.first];
    x[it->col] = std::min(std::max(v / it->coef, it->lb), it->ub);
  }
  return x;
}

std::vector<double> Presolver::crush(const std::vector<double>& x_orig) const {
  std::vector<double> x = x_orig;
  for (const Step& st : stack_) {
    if (st.other >= 0) {
      x[st.other] += st.coef * x[st.col];  // merged parallel columns: y = x_k + s x_j
    } else if (st.terms.size() == 1 && st.terms[0].first == st.col) {
      x[st.col] = (x[st.col] - st.rhs) / -st.terms[0].second;  // rescaled integer column: x = r + g x'
    }
  }
  std::vector<double> out(col_map_.size());
  for (size_t k = 0; k < col_map_.size(); ++k) out[k] = x[col_map_[k]];
  return out;
}

}  // namespace milp
