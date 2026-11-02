#include "milp/presolve/presolve.hpp"

#include "milp/presolve/clique_merge.hpp"
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
constexpr Index kMaxSubstRow = 1000;      // longest equation used for substitution
constexpr Index kMaxSubstCol = 8;         // most rows of a substituted non-singleton column
constexpr long long kMaxSubstFill = 32;   // (row length - 1) * (column length - 1) limit

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
    for (Index j = 0; j < n_; ++j) {
      for (Index p = in.A.begin(j); p < in.A.end(j); ++p) {
        if (in.A.value[p] == 0.0) continue;
        cols_[j].push_back({in.A.index[p], in.A.value[p]});
        rows_[in.A.index[p]].push_back({j, in.A.value[p]});
      }
    }
    row_alive_.assign(m_, 1);
    col_alive_.assign(n_, 1);
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
    bool probed = !opt_.probing;
    int clique_runs = 0;
    long long since_clique = 0;
    for (int round = 0; round < opt_.max_rounds + 10 && status_ == PresolveStatus::Reduced; ++round) {
      if (round >= opt_.max_rounds && probed) break;
      ++stats_.rounds;
      compact();
      long long ch = 0;
      // Each step is timed (verbosity 3 prints the totals).
      auto step = [&](int k, bool on, auto&& f) {
        if (!on || status_ != PresolveStatus::Reduced) return;
        Timer t;
        ch += f();
        step_time_[k] += t.elapsed();
      };
      step(0, true, [&] { return column_pass(); });
      step(1, true, [&] { return row_pass(true); });
      step(2, opt_.bound_tightening, [&] { return propagate(); });
      step(0, true, [&] { return column_pass(); });
      step(3, opt_.coef_tightening, [&] { return coef_tightening(); });
      step(4, opt_.dual_fixing, [&] { return dual_fixing(); });
      step(5, opt_.dominated_cols, [&] { return dominated_cols(); });
      step(6, opt_.parallel_rows, [&] { return parallel_rows(); });
      step(7, opt_.parallel_cols, [&] { return parallel_cols(); });
      step(8, opt_.aggregation, [&] { return aggregate(); });
      step(9, opt_.substitution, [&] { return substitute_equations(); });
      log_.log(3, "Presolve round %d: %lld changes\n", round + 1, ch);
      since_clique += ch;
      const bool last = round + 1 >= opt_.max_rounds;
      if ((ch == 0 || last) && opt_.clique_extension && status_ == PresolveStatus::Reduced && clique_runs < 4 &&
          (clique_runs == 0 || since_clique > 0)) {
        // Clique extension whenever the other reductions are exhausted (and something changed since its last run):
        // its fixings and dominated rows feed further rounds.
        ++clique_runs;
        since_clique = 0;
        compact();
        Timer tc;
        ch = clique_extension();
        step_time_[11] += tc.elapsed();
        if (ch > 0 && !last) continue;
      }
      if ((ch == 0 || last) && !probed && status_ == PresolveStatus::Reduced) {
        // Probing once the cheap reductions are (nearly) exhausted, then a few more rounds.
        probed = true;
        compact();
        Timer tp;
        ch = probing();
        step_time_[10] += tp.elapsed();
        log_.log(3, "Presolve probing: %lld changes\n", ch);
      }
      if (ch == 0) break;
    }
    if (log_.verbosity() >= 3) {
      static const char* names[] = {"columns", "rows", "propagate", "coef", "dual", "dominated", "parallel_rows",
                                    "parallel_cols", "aggregate", "substitute", "probing", "cliqueext"};
      std::string line = "Presolve step times:";
      for (int k = 0; k < 12; ++k) line += " " + std::string(names[k]) + " " + std::to_string(step_time_[k]).substr(0, 5);
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
      if (has_ub && std::abs(rub_[i]) < 1e12) {
        const double v = gd * std::floor((rub_[i] + opt_.feas_tol) / gd);
        if (v < rub_[i] - 1e-9) rub_[i] = v, changed = true;
      }
      if (has_lb && std::abs(rlb_[i]) < 1e12) {
        const double v = gd * std::ceil((rlb_[i] - opt_.feas_tol) / gd);
        if (v > rlb_[i] + 1e-9) rlb_[i] = v, changed = true;
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
      bool ok = true, any_int = false, exact = std::abs(b) < 1e9 && b == std::round(b);
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
        exact = exact && a == std::round(a) && bh == std::round(bh) && std::abs(a * bh) < 1e9;
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
        v += tol;
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
        v -= tol;
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

  // Parallel rows: rows that are scalar multiples of each other are merged.
  long long parallel_rows() {
    long long ch = 0;
    std::unordered_map<uint64_t, std::vector<Index>> buckets;
    std::vector<std::vector<Entry>> norm(m_);  // sorted by column, scaled so the first value is 1
    std::vector<double> first(m_, 0.0);
    for (Index i = 0; i < m_; ++i) {
      if (!row_alive_[i] || row_size_[i] < 2) continue;
      auto& v = norm[i];
      for (const Entry& e : rows_[i]) {
        if (col_alive_[e.idx]) v.push_back(e);
      }
      std::sort(v.begin(), v.end(), [](const Entry& x, const Entry& y) { return x.idx < y.idx; });
      first[i] = v[0].val;
      uint64_t h = 1469598103934665603ULL ^ v.size();
      for (Entry& e : v) {
        e.val /= first[i];
        const auto q = static_cast<int64_t>(std::llround(e.val * 1e6));
        h = (h ^ static_cast<uint64_t>(e.idx)) * 1099511628211ULL;
        h = (h ^ static_cast<uint64_t>(q)) * 1099511628211ULL;
      }
      buckets[h].push_back(i);
    }
    for (auto& kv : buckets) {
      auto& rs = kv.second;
      if (rs.size() < 2) continue;
      for (size_t x = 0; x < rs.size() && status_ == PresolveStatus::Reduced; ++x) {
        const Index r1 = rs[x];
        if (!row_alive_[r1]) continue;
        for (size_t y = x + 1; y < rs.size(); ++y) {
          const Index r2 = rs[y];
          if (!row_alive_[r2] || norm[r2].size() != norm[r1].size()) continue;
          bool same = true;
          for (size_t k = 0; k < norm[r1].size() && same; ++k) {
            const Entry &e1 = norm[r1][k], &e2 = norm[r2][k];
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
        // Near-cancellation leaves numerical noise: drop it, relaxing the row by its largest effect.
        const double kmax = std::max(std::abs(lb_[k]), std::abs(ub_[k]));
        if (std::abs(nv) <= 1e-9 * std::max(std::abs(cur), std::abs(f * t.val)) && kmax < kBig) {
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
    for (const Entry& x : cols_[j]) {
      if (!row_alive_[x.idx] || x.idx == i) continue;
      for (const Entry& e : rows_[x.idx]) mark_[e.idx] = x.idx;
      for (const Entry& e : r) cnt += e.idx != j && mark_[e.idx] != x.idx;
    }
    return cnt;
  }

  // Parallel columns A_j = s A_k with c_j = s c_k (same type; s = +-1 for integers): x_k and x_j only appear as
  // y = x_k + s x_j, so x_j is removed and x_k becomes y with the bounds of that sum.
  long long parallel_cols() {
    long long ch = 0;
    std::unordered_map<uint64_t, std::vector<Index>> buckets;
    std::vector<std::vector<Entry>> norm(n_);  // sorted by row, scaled so the first value is 1
    std::vector<double> first(n_, 0.0);
    for (Index j = 0; j < n_; ++j) {
      if (!col_alive_[j] || col_size_[j] < 1 || lb_[j] == ub_[j]) continue;
      auto& v = norm[j];
      for (const Entry& e : cols_[j]) {
        if (row_alive_[e.idx]) v.push_back(e);
      }
      std::sort(v.begin(), v.end(), [](const Entry& x, const Entry& y) { return x.idx < y.idx; });
      first[j] = v[0].val;
      uint64_t h = 1469598103934665603ULL ^ (v.size() * 2 + is_int_[j]);
      for (Entry& e : v) {
        e.val /= first[j];
        const auto q = static_cast<int64_t>(std::llround(e.val * 1e6));
        h = (h ^ static_cast<uint64_t>(e.idx)) * 1099511628211ULL;
        h = (h ^ static_cast<uint64_t>(q)) * 1099511628211ULL;
      }
      buckets[h].push_back(j);
    }
    for (auto& kv : buckets) {
      auto& cs = kv.second;
      for (size_t x = 0; x < cs.size() && cs.size() > 1; ++x) {
        const Index k = cs[x];
        if (!col_alive_[k]) continue;
        for (size_t y = x + 1; y < cs.size(); ++y) {
          const Index j = cs[y];
          if (!col_alive_[j] || is_int_[j] != is_int_[k] || norm[j].size() != norm[k].size()) continue;
          bool same = true;
          for (size_t t = 0; t < norm[k].size() && same; ++t) {
            const Entry &e1 = norm[k][t], &e2 = norm[j][t];
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

  // Equations sum_k a_k x_k = b: a column j is substituted out, x_j = (b - sum_{k != j} a_k x_k) / a_j. The row
  // stays as lb_j <= (b - sum_{k != j} a_k x_k) / a_j <= ub_j, without the sides the row already implies (it is
  // removed if j is implied free). Column singletons cause no fill-in and are preferred; other columns are taken
  // if the fill-in is small. Integer columns qualify only if the row keeps them integral (integer columns,
  // a_k / a_j and b / a_j integral).
  long long substitute_equations() {
    long long ch = 0;
    std::vector<Entry> r, terms;
    for (Index i = 0; i < m_ && status_ == PresolveStatus::Reduced; ++i) {
      if (!row_alive_[i] || rlb_[i] != rub_[i] || row_size_[i] < 2 || row_size_[i] > kMaxSubstRow) continue;
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
      bool best_lo = false, best_hi = false;  // whether the row implies lb_j / ub_j
      for (size_t t = 0; t < r.size(); ++t) {
        const Index j = r[t].idx;
        const double a = r[t].val;
        if (is_int_[j] ? !(int_ok && std::abs(a) == amin) : std::abs(a) < 1e-2 * amax) continue;
        const bool single = col_size_[j] == 1;
        if (!single && (any_single || col_size_[j] > kMaxSubstCol)) continue;
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
        const bool lo = !std::isfinite(lb_[j]) || ilo >= lb_[j] - 1e-9 * (1.0 + std::abs(lb_[j]));
        const bool hi = !std::isfinite(ub_[j]) || ihi <= ub_[j] + 1e-9 * (1.0 + std::abs(ub_[j]));
        // Net change of nonzeros; a kept row must not grow the matrix.
        const long long score = single ? -1 : new_entries(j, i, r) - col_size_[j] - (lo && hi ? row_size_[i] - 1 : 0);
        if (!(lo && hi) && score > 0) continue;
        if (best < 0 || score < best_score) best = static_cast<int>(t), best_score = score, best_lo = lo, best_hi = hi;
      }
      if (best < 0) continue;
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
      substitute(j, b, a, terms, i);
      rlb_[i] = a > 0 ? side_ub : side_lb;
      rub_[i] = a > 0 ? side_lb : side_ub;
      if (!std::isfinite(rlb_[i]) && !std::isfinite(rub_[i])) remove_row(i);
      ++stats_.substitutions;
      ++ch;
    }
    return ch;
  }

  // Probing on the binaries of the current problem; applies fixings, bounds and equivalences.
  long long probing() {
    MipProblem q;
    std::vector<Index> cols;
    assemble(q, cols);
    const long long limit = static_cast<long long>(opt_.probing_effort * std::min(5e8, 5e7 + 2000.0 * q.A.nnz()));
    const ProbingResult pr = probe_binaries(q, limit);
    stats_.probes += pr.probes;
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
      if (lb_[j] == ub_[j]) ++stats_.probing_fixed; else stats_.probing_bounds += c;
    }
    if (status_ != PresolveStatus::Reduced) return ch;
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
      const double d = e.v1 - e.v0, c0 = neg ? e.v0 + d : e.v0, c1 = neg ? -d : d;  // y = c0 + c1 x
      if (log_.verbosity() >= 4) log_.log(4, "  probing: col %d = %.12g + %.12g col %d\n", y, c0, c1, x);
      substitute(y, c0, 1.0, {{x, -c1}}, -1);
      if (is_int_[y] && std::min(e.v0, e.v1) == 0.0 && std::max(e.v0, e.v1) == 1.0) rep_of[y] = {x, c1 < 0};
      ++stats_.probing_equivs;
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
    const CliqueExtension ce = find_clique_extensions(q, limit);
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
  double offset_ = 0.0, display_sense_ = 1.0;
  double step_time_[12] = {};
  std::vector<double> minact_, maxact_;  // row activity snapshot (dual_fixing)
  std::vector<int> nmin_inf_, nmax_inf_;
};

PresolveOptions PresolveOptions::from_env() {
  PresolveOptions o;
  if (const char* e = std::getenv("MILP_PROBING_EFFORT")) o.probing_effort = std::atof(e);
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
  if (off("aggregation")) o.aggregation = false;
  if (off("cliques")) o.clique_merging = false;
  if (off("cliqueext")) o.clique_extension = false;
  if (off("substitution")) o.substitution = false;
  if (off("probing")) o.probing = false;
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
  stats_.rows_after = reduced_.num_rows();
  stats_.cols_after = reduced_.num_cols();
  stats_.nnz_after = reduced_.A.nnz();
  stats_.time = timer.elapsed();
  log.log(2,
          "Presolve details: %d rounds, %lld fixed cols, %lld removed rows, %lld bound changes, %lld coef changes, "
          "%lld parallel rows, %lld parallel cols, %lld dominated cols, %lld dual bounds, %lld aggregations, "
          "%lld substitutions\n",
          stats_.rounds, stats_.fixed_cols, stats_.removed_rows, stats_.bound_changes, stats_.coef_changes,
          stats_.parallel_rows, stats_.parallel_cols, stats_.dominated, stats_.dual_bounds, stats_.aggregations,
          stats_.substitutions);
  if (opt.clique_extension) {
    log.log(2, "Presolve cliques: %lld rows extended, %lld dominated rows, %lld fixed\n", stats_.clique_extended,
            stats_.clique_dominated, stats_.clique_fixed);
  }
  if (opt.probing) {
    log.log(2, "Presolve probing: %lld probes, %lld fixed, %lld bounds, %lld equivalences\n", stats_.probes,
            stats_.probing_fixed, stats_.probing_bounds, stats_.probing_equivs);
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
    if (st.other >= 0) x[st.other] += st.coef * x[st.col];  // merged parallel columns: y = x_k + s x_j
  }
  std::vector<double> out(col_map_.size());
  for (size_t k = 0; k < col_map_.size(); ++k) out[k] = x[col_map_[k]];
  return out;
}

}  // namespace milp
