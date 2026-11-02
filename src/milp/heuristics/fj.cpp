#include "milp/heuristics/fj.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>

namespace milp {

namespace {

constexpr double kFeasTol = 1e-7;  // row violation tolerance (try_incumbent checks 1e-6)

class Fj {
 public:
  Fj(const MipProblem& p, const FjLimits& lim) : p_(p), lim_(lim), rng_(lim.seed * 2654435761u + 12345u) {
    n_ = p.num_cols();
    m_ = p.num_rows();
    obj_row_ = m_;
    // Columns with the objective as an extra row m_.
    cstart_.assign(n_ + 1, 0);
    std::vector<Index> rcount(m_ + 1, 0);
    for (Index j = 0; j < n_; ++j) {
      for (Index k = p.A.begin(j); k < p.A.end(j); ++k) {
        if (p.A.value[k] == 0.0) continue;
        crow_.push_back(p.A.index[k]);
        cval_.push_back(p.A.value[k]);
        ++rcount[p.A.index[k]];
      }
      if (p.cost[j] != 0.0) {
        crow_.push_back(obj_row_);
        cval_.push_back(p.cost[j]);
        ++rcount[obj_row_];
      }
      cstart_[j + 1] = static_cast<Index>(crow_.size());
    }
    // Rows (CSR) for candidate selection.
    rstart_.assign(m_ + 2, 0);
    for (Index i = 0; i <= m_; ++i) rstart_[i + 1] = rstart_[i] + rcount[i];
    rcol_.resize(crow_.size());
    std::vector<Index> fill(rstart_.begin(), rstart_.end() - 1);
    for (Index j = 0; j < n_; ++j) {
      for (Index k = cstart_[j]; k < cstart_[j + 1]; ++k) rcol_[fill[crow_[k]]++] = j;
    }
    lo_.assign(p.row_lb.begin(), p.row_lb.end());
    hi_.assign(p.row_ub.begin(), p.row_ub.end());
    lo_.push_back(-kInf);
    hi_.push_back(kInf);
    scale_.assign(m_ + 1, 0.0);
    for (Index k = 0; k < static_cast<Index>(crow_.size()); ++k) {
      scale_[crow_[k]] = std::max(scale_[crow_[k]], std::abs(cval_[k]));
    }
    for (double& s : scale_) s = s > 0.0 ? 1.0 / s : 1.0;
    obj_integral_ = true;
    for (Index j = 0; j < n_ && obj_integral_; ++j) {
      if (p.cost[j] != 0.0) obj_integral_ = p.is_int[j] && p.cost[j] == std::round(p.cost[j]);
    }
    has_cost_ = rcount[obj_row_] > 0;
  }

  bool run(const std::function<void(const std::vector<double>&, double)>& on_solution, FjStats* stats) {
    const auto t0 = std::chrono::steady_clock::now();
    x_.resize(n_);
    for (Index j = 0; j < n_; ++j) x_[j] = clamp_var(j, 0.0);
    init_state();
    size_t best_nviol = viol_.size();
    long long since_best = 0, step = 0;
    // Restarts are rare: weights do most of the diversification.
    const long long restart_after = std::max<long long>(2000000, 400LL * (n_ + m_));
    long long progress_effort = 0;  // effort at the last new solution / fewest violated rows
    while (effort_ < lim_.effort) {
      if ((++step & 1023) == 0) {
        if (lim_.stop && lim_.stop->load(std::memory_order_relaxed)) break;
        if (lim_.checkpoint && !lim_.checkpoint(effort_)) break;
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el > lim_.time_limit) break;
      }
      if (effort_ - progress_effort > lim_.stall_effort) break;
      if (viol_.empty()) {
        if (!found_point(on_solution)) break;
        progress_effort = effort_;
        best_nviol = viol_.size();
        since_best = 0;
        continue;
      }
      if (static_cast<long long>(viol_.size()) < min_viol_) {
        min_viol_ = static_cast<long long>(viol_.size());
        progress_effort = effort_;
        if (stats && solutions_ == 0) {
          stats->closest = x_;
          effort_ += n_;
        }
      }
      if (viol_.size() < best_nviol) {
        best_nviol = viol_.size();
        since_best = 0;
      } else if (++since_best > restart_after) {
        restart();
        best_nviol = viol_.size();
        since_best = 0;
        continue;
      }
      const Index r = viol_[rng_() % viol_.size()];
      Index best_j = -1;
      double best_v = 0.0, best_score = 1e-9;
      const Index len = rstart_[r + 1] - rstart_[r];
      const Index samples = std::min<Index>(len, kSample);
      effort_ += 1 + samples;  // every step costs effort (a row of fixed columns must not spin)
      for (Index s = 0; s < samples; ++s) {
        const Index j = rcol_[rstart_[r] + (len > kSample ? static_cast<Index>(rng_() % len) : s)];
        if (p_.col_lb[j] == p_.col_ub[j]) continue;
        double v, score;
        jump(j, &v, &score);
        if (score > best_score) {
          best_score = score;
          best_j = j;
          best_v = v;
        }
      }
      if (best_j >= 0) {
        move(best_j, best_v);
        ++moves_;
      } else {
        // Local minimum w.r.t. this row: make it more important.
        w_[r] += 1.0;
      }
    }
    if (stats) {
      stats->effort = effort_;
      stats->moves = moves_;
      stats->restarts = restarts_;
      stats->solutions = solutions_;
      stats->min_violated = solutions_ > 0 ? 0 : min_viol_;
    }
    return solutions_ > 0;
  }

 private:
  static constexpr Index kSample = 24;

  double clamp_var(Index j, double v) const {
    v = std::clamp(v, p_.col_lb[j], p_.col_ub[j]);
    return p_.is_int[j] ? std::round(v) : v;
  }
  double row_viol(Index i, double a) const {
    const double d = std::max({0.0, lo_[i] - a, a - hi_[i]});
    return d > kFeasTol ? d : 0.0;
  }
  void set_viol(Index i) {
    const bool v = row_viol(i, act_[i]) > 0.0;
    if (v && pos_[i] < 0) {
      pos_[i] = static_cast<Index>(viol_.size());
      viol_.push_back(i);
    } else if (!v && pos_[i] >= 0) {
      const Index last = viol_.back();
      viol_[pos_[i]] = last;
      pos_[last] = pos_[i];
      viol_.pop_back();
      pos_[i] = -1;
    }
  }
  void init_state() {
    act_.assign(m_ + 1, 0.0);
    for (Index j = 0; j < n_; ++j) {
      for (Index k = cstart_[j]; k < cstart_[j + 1]; ++k) act_[crow_[k]] += cval_[k] * x_[j];
    }
    effort_ += static_cast<long long>(crow_.size());
    w_.assign(m_ + 1, 1.0);
    pos_.assign(m_ + 1, -1);
    viol_.clear();
    for (Index i = 0; i <= m_; ++i) set_viol(i);
  }

  // Best value for x_j given the other variables (minimum of the convex
  // piecewise-linear weighted violation, closest to the current value).
  void jump(Index j, double* out_v, double* out_score) {
    const Index b = cstart_[j], e = cstart_[j + 1];
    const double xj = x_[j];
    ev_.clear();
    double slope = 0.0;
    for (Index k = b; k < e; ++k) {
      const Index i = crow_[k];
      const double a = cval_[k], res = act_[i] - a * xj, ww = w_[i] * scale_[i] * std::abs(a);
      double l = (lo_[i] - res) / a, u = (hi_[i] - res) / a;
      if (a < 0) std::swap(l, u);
      if (is_finite(l)) {
        ev_.emplace_back(l, ww);
        slope -= ww;
      }
      if (is_finite(u)) ev_.emplace_back(u, ww);
    }
    effort_ += 2 * (e - b);
    std::sort(ev_.begin(), ev_.end());
    // Minimiser region [v1, v2]: slope crosses zero.
    double v1 = -kInf, v2 = kInf;
    size_t k = 0;
    if (slope < 0.0) {
      while (k < ev_.size() && slope < 0.0) slope += ev_[k++].second;
      v1 = ev_[k - 1].first;
    }
    if (slope <= 0.0) {
      while (k < ev_.size() && slope <= 0.0) slope += ev_[k++].second;
      if (slope > 0.0) v2 = ev_[k - 1].first;
    } else {
      v2 = v1;
    }
    double t = std::clamp(xj, v1, v2);
    t = std::clamp(t, p_.col_lb[j], p_.col_ub[j]);
    if (p_.is_int[j]) {
      const double fl = clamp_var(j, std::floor(t + 1e-9)), cl = clamp_var(j, std::ceil(t - 1e-9));
      if (fl != cl) {
        const double ffl = eval(j, fl), fcl = eval(j, cl);
        t = ffl < fcl ? fl : fcl < ffl ? cl : (std::abs(fl - xj) <= std::abs(cl - xj) ? fl : cl);
      } else {
        t = fl;
      }
    }
    *out_v = t;
    *out_score = t == xj ? 0.0 : eval(j, xj) - eval(j, t);
  }
  double eval(Index j, double v) {
    double f = 0.0;
    const double xj = x_[j];
    for (Index k = cstart_[j]; k < cstart_[j + 1]; ++k) {
      const Index i = crow_[k];
      f += w_[i] * scale_[i] * row_viol(i, act_[i] + cval_[k] * (v - xj));
    }
    effort_ += cstart_[j + 1] - cstart_[j];
    return f;
  }
  void move(Index j, double v) {
    const double d = v - x_[j];
    x_[j] = v;
    for (Index k = cstart_[j]; k < cstart_[j + 1]; ++k) {
      act_[crow_[k]] += cval_[k] * d;
      set_viol(crow_[k]);
    }
    effort_ += cstart_[j + 1] - cstart_[j];
  }

  // All rows satisfied: report the point, then tighten the objective row.
  bool found_point(const std::function<void(const std::vector<double>&, double)>& on_solution) {
    // Recompute activities to remove drift before trusting the point.
    const std::vector<double> w = w_;
    init_state();
    w_ = w;
    if (!viol_.empty()) return true;
    effort_ += static_cast<long long>(crow_.size());
    if (p_.max_violation(x_) > 1e-6) return false;
    double obj = 0.0;
    for (Index j = 0; j < n_; ++j) obj += p_.cost[j] * x_[j];
    if (obj < best_obj_) {
      best_obj_ = obj;
      best_x_ = x_;
      ++solutions_;
      on_solution(x_, obj);
    }
    if (!has_cost_) return false;
    const double delta = obj_integral_ ? 1.0 : std::max(1e-6, 1e-4 * std::abs(obj));
    hi_[obj_row_] = best_obj_ - delta + kFeasTol;
    std::fill(w_.begin(), w_.end(), 1.0);  // fresh weights for the improvement phase
    set_viol(obj_row_);
    return true;
  }

  // Restart from the best point (or a perturbed start) with fresh weights.
  void restart() {
    ++restarts_;
    if (!best_x_.empty()) {
      x_ = best_x_;
    } else {
      for (Index j = 0; j < n_; ++j) {
        if (!p_.is_int[j] || rng_() % 8 != 0) continue;
        const double l = p_.col_lb[j], u = std::min(p_.col_ub[j], l + 10.0);
        if (is_finite(l)) x_[j] = clamp_var(j, l + static_cast<double>(rng_() % static_cast<uint32_t>(u - l + 1)));
      }
    }
    init_state();
  }

  const MipProblem& p_;
  FjLimits lim_;
  std::mt19937 rng_;
  Index n_ = 0, m_ = 0, obj_row_ = 0;
  std::vector<Index> cstart_, crow_, rstart_, rcol_;
  std::vector<double> cval_, lo_, hi_, scale_;
  std::vector<double> x_, act_, w_;
  std::vector<Index> pos_, viol_;
  std::vector<std::pair<double, double>> ev_;
  std::vector<double> best_x_;
  double best_obj_ = kInf;
  bool obj_integral_ = false, has_cost_ = false;
  long long effort_ = 0, moves_ = 0, restarts_ = 0, min_viol_ = 1LL << 60;
  int solutions_ = 0;
};

}  // namespace

bool feasibility_jump(const MipProblem& p, const FjLimits& limits,
                      const std::function<void(const std::vector<double>&, double)>& on_solution, FjStats* stats) {
  if (p.num_cols() == 0) return false;
  Fj fj(p, limits);
  return fj.run(on_solution, stats);
}

}  // namespace milp
