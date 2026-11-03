#include "milp/lp/crossover.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "milp/core/timer.hpp"
#include "milp/lp/lu.hpp"

namespace milp {

namespace {

class Crossover {
 public:
  Crossover(const SparseMatrix& A, const std::vector<double>& cost, const std::vector<double>& col_lb,
            const std::vector<double>& col_ub, const std::vector<double>& row_lb, const std::vector<double>& row_ub)
      : A_(A), n_(A.num_major), m_(A.num_minor) {
    const Index N = n_ + m_;
    lo_.resize(static_cast<size_t>(N));
    hi_.resize(static_cast<size_t>(N));
    cost_.assign(static_cast<size_t>(N), 0.0);
    for (Index j = 0; j < n_; ++j) lo_[j] = col_lb[j], hi_[j] = col_ub[j], cost_[j] = cost[j];
    for (Index i = 0; i < m_; ++i) lo_[n_ + i] = -row_ub[i], hi_[n_ + i] = -row_lb[i];
    At_ = A.transpose();
  }

  Basis run(const std::vector<double>& x, const std::vector<double>& y, CrossoverStats* st,
            const std::atomic<bool>* stop);

 private:
  void column(Index k, HVector& v) const {
    v.clear();
    if (k < n_) {
      for (Index p = A_.begin(k); p < A_.end(k); ++p) {
        v.dense[A_.index[p]] = A_.value[p];
        v.nz[v.count++] = A_.index[p];
      }
    } else {
      v.dense[k - n_] = 1.0;
      v.nz[v.count++] = k - n_;
    }
  }
  // Factorizes basic_; dependent columns are replaced by logicals (the old variable goes to its
  // nearest bound). Returns the number replaced.
  Index factorize();
  void compute_basic_values();
  double ptol(Index k, double bound) const { return 1e-9 * (1.0 + std::abs(bound)); }
  // A basic variable this close to a bound may leave the basis there in the dual push.
  double btol(Index k, double bound) const { return 1e-7 * (1.0 + std::abs(bound)); }
  void make_nonbasic(Index k);  // status from the value (snapped to a bound if close)
  // Dual push: basic variables with a nonzero reduced cost (w.r.t. the interior point's duals y)
  // leave the basis at their bound, or get a zero reduced cost by a dual step; returns the pivots.
  long long dual_push(std::vector<double> y, const std::atomic<bool>* stop, Index* candidates);

  const SparseMatrix& A_;
  SparseMatrix At_;
  Index n_, m_;
  std::vector<double> lo_, hi_, cost_;
  std::vector<double> v_;
  std::vector<double> wlo_, whi_;  // working bounds of the push (shifted for infeasible basics)
  std::vector<VarStatus> status_;
  std::vector<Index> basic_;
  LuFactor lu_;
  HVector col_;
};

Index Crossover::factorize() {
  std::vector<Index> start(static_cast<size_t>(m_) + 1, 0), index;
  std::vector<double> value;
  for (Index p = 0; p < m_; ++p) {
    column(basic_[p], col_);
    for (Index k = 0; k < col_.count; ++k) {
      index.push_back(col_.nz[k]);
      value.push_back(col_.dense[col_.nz[k]]);
    }
    start[p + 1] = static_cast<Index>(index.size());
  }
  col_.clear();
  std::vector<std::pair<Index, Index>> replaced;
  lu_.factorize(m_, start, index, value, &replaced);
  for (auto [p, row] : replaced) {
    const Index old = basic_[p];
    basic_[p] = n_ + row;
    status_[n_ + row] = VarStatus::Basic;
    make_nonbasic(old);
  }
  return static_cast<Index>(replaced.size());
}

void Crossover::make_nonbasic(Index k) {
  const double l = lo_[k], u = hi_[k];
  double& v = v_[k];
  if (is_finite(l) && v <= l + ptol(k, l)) v = l, status_[k] = VarStatus::AtLower;
  else if (is_finite(u) && v >= u - ptol(k, u)) v = u, status_[k] = VarStatus::AtUpper;
  else if (!is_finite(l) && !is_finite(u) && std::abs(v) <= 1e-9) v = 0.0, status_[k] = VarStatus::AtZero;
  else status_[k] = VarStatus::AtZero;  // superbasic: the caller pushes it (value kept)
}

void Crossover::compute_basic_values() {
  // B v_B = -N v_N.
  std::vector<double> rhs(static_cast<size_t>(m_), 0.0);
  for (Index k = 0; k < n_ + m_; ++k) {
    if (status_[k] == VarStatus::Basic || v_[k] == 0.0) continue;
    if (k < n_) {
      for (Index p = A_.begin(k); p < A_.end(k); ++p) rhs[A_.index[p]] -= A_.value[p] * v_[k];
    } else {
      rhs[k - n_] -= v_[k];
    }
  }
  lu_.ftran(rhs);
  for (Index p = 0; p < m_; ++p) v_[basic_[p]] = rhs[p];
}

Basis Crossover::run(const std::vector<double>& x, const std::vector<double>& y, CrossoverStats* st,
                     const std::atomic<bool>* stop) {
  const Index N = n_ + m_;
  col_.resize(m_);
  v_.assign(static_cast<size_t>(N), 0.0);
  std::vector<double> d(static_cast<size_t>(N), 0.0);
  for (Index j = 0; j < n_; ++j) {
    v_[j] = std::clamp(x[j], lo_[j], hi_[j]);
    double s = cost_[j];
    for (Index p = A_.begin(j); p < A_.end(j); ++p) {
      v_[n_ + A_.index[p]] -= A_.value[p] * v_[j];
      s -= A_.value[p] * y[A_.index[p]];
    }
    d[j] = s;
  }
  for (Index i = 0; i < m_; ++i) d[n_ + i] = -y[i];

  // Rank: distance to the nearest bound over |reduced cost| (free variables first).
  std::vector<double> score(static_cast<size_t>(N));
  for (Index k = 0; k < N; ++k) {
    const double l = lo_[k], u = hi_[k];
    double dist = 1e30;
    if (is_finite(l)) dist = std::min(dist, v_[k] - l);
    if (is_finite(u)) dist = std::min(dist, u - v_[k]);
    dist = std::max(dist, 0.0);
    if (l == u) dist = -1.0;  // fixed: never basic by choice
    score[k] = dist / std::max(std::abs(d[k]), 1e-12);
    if (dist < 0.0) score[k] = -1.0;
  }
  std::vector<Index> order(static_cast<size_t>(N));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) { return score[a] > score[b]; });

  status_.assign(static_cast<size_t>(N), VarStatus::AtLower);
  basic_.assign(order.begin(), order.begin() + m_);
  for (Index p = 0; p < m_; ++p) status_[basic_[p]] = VarStatus::Basic;
  for (Index k = 0; k < N; ++k) {
    if (status_[k] != VarStatus::Basic) make_nonbasic(k);
  }
  Index replaced = 0;
  // A strict pivot tolerance keeps the starting basis well conditioned: nearly dependent columns
  // are replaced by logicals (an ill-conditioned basis would turn the interior point's small
  // residuals into large basic infeasibilities).
  lu_.pivot_tolerance = 1e-7;
  for (int round = 0; round < 5; ++round) {
    const Index r = factorize();
    replaced += r;
    if (r == 0) break;
  }
  lu_.pivot_tolerance = LuFactor().pivot_tolerance;
  compute_basic_values();

  // Dual push first: it exchanges basic variables at a bound with a nonzero reduced cost for
  // nonbasic ones without changing the primal point; the primal push after it keeps the reduced
  // costs (its entering variables have zero reduced costs), so the result is close to optimal.
  {
    Index cand = 0;
    const long long dp = dual_push(y, stop, &cand);
    if (st) st->dual_pivots = dp, st->dual_candidates = cand;
    factorize();
    compute_basic_values();
  }
  // Primal push of the superbasics.
  std::vector<Index> super;
  for (Index k = 0; k < N; ++k) {
    if (status_[k] == VarStatus::AtZero && !(v_[k] == 0.0 && !is_finite(lo_[k]) && !is_finite(hi_[k]))) super.push_back(k);
  }
  if (st) st->superbasics = static_cast<Index>(super.size()), st->replaced = replaced;
  long long pushes = 0, pivots = 0;
  double dens = -1.0;
  // Working bounds: a basic variable that is infeasible (the interior point's residuals) gets its
  // violated bound shifted to its value, so it can leave the basis there without changing the
  // primal point (snapping it would push the error into the other basic variables at every such
  // pivot). The shifts are removed once at the end.
  wlo_ = lo_;
  whi_ = hi_;
  auto shift_basics = [&] {
    for (Index p = 0; p < m_; ++p) {
      const Index b = basic_[p];
      wlo_[b] = std::min(wlo_[b], v_[b]);
      whi_[b] = std::max(whi_[b], v_[b]);
    }
  };
  shift_basics();
  for (Index q : super) {
    if (stop && (pushes & 63) == 0 && stop->load(std::memory_order_relaxed)) break;
    if (status_[q] != VarStatus::AtZero) continue;
    const double l = lo_[q], u = hi_[q];
    double target = 0.0;
    if (is_finite(l) && is_finite(u)) {
      if (d[q] > 1e-9) target = l;
      else if (d[q] < -1e-9) target = u;
      else target = (v_[q] - l <= u - v_[q]) ? l : u;
    } else if (is_finite(l)) {
      target = l;
    } else if (is_finite(u)) {
      target = u;
    } else {
      target = 0.0;
    }
    const double dist = std::abs(target - v_[q]);
    const double dir = target >= v_[q] ? 1.0 : -1.0;
    ++pushes;
    column(q, col_);
    lu_.ftran(col_, dens);
    dens = 0.9 * (dens < 0 ? 0.0 : dens) + 0.1 * static_cast<double>(col_.count) / std::max<Index>(1, m_);
    // Basic variable at position p changes by rate * t, rate = -dir * alpha_p. Pivots below
    // piv_tol (relative to the column's largest entry) are not used.
    double amax = 0.0;
    for (Index e = 0; e < col_.count; ++e) amax = std::max(amax, std::abs(col_.dense[col_.nz[e]]));
    const double piv_tol = std::max(1e-9, 1e-7 * amax);
    double tmax = kInf;
    for (Index e = 0; e < col_.count; ++e) {
      const Index p = col_.nz[e];
      const double a = col_.dense[p];
      if (std::abs(a) <= piv_tol) continue;
      const Index b = basic_[p];
      const double rate = -dir * a;
      if (rate < 0.0 && is_finite(wlo_[b])) tmax = std::min(tmax, std::max(0.0, v_[b] - wlo_[b] + ptol(b, lo_[b])) / -rate);
      else if (rate > 0.0 && is_finite(whi_[b])) tmax = std::min(tmax, std::max(0.0, whi_[b] - v_[b] + ptol(b, hi_[b])) / rate);
    }
    if (tmax >= dist) {
      for (Index e = 0; e < col_.count; ++e) {
        const Index p = col_.nz[e];
        v_[basic_[p]] -= dir * dist * col_.dense[p];
      }
      v_[q] = target;
      status_[q] = !is_finite(l) && !is_finite(u) ? VarStatus::AtZero : target == l ? VarStatus::AtLower : VarStatus::AtUpper;
      continue;
    }
    // Harris pass 2: the largest pivot among the blocking candidates.
    Index r = -1;
    double best = 0.0, t = 0.0;
    bool to_lower = true;
    for (Index e = 0; e < col_.count; ++e) {
      const Index p = col_.nz[e];
      const double a = col_.dense[p];
      if (std::abs(a) <= piv_tol) continue;
      const Index b = basic_[p];
      const double rate = -dir * a;
      double ratio = 0.0;
      bool lower = true;
      if (rate < 0.0 && is_finite(wlo_[b])) ratio = std::max(0.0, v_[b] - wlo_[b]) / -rate, lower = true;
      else if (rate > 0.0 && is_finite(whi_[b])) ratio = std::max(0.0, whi_[b] - v_[b]) / rate, lower = false;
      else continue;
      if (ratio <= tmax && std::abs(a) > best) best = std::abs(a), r = p, t = ratio, to_lower = lower;
    }
    if (r < 0) continue;  // cannot happen (tmax finite); keep the superbasic
    const Index leave = basic_[r];
    for (Index e = 0; e < col_.count; ++e) {
      const Index p = col_.nz[e];
      v_[basic_[p]] -= dir * t * col_.dense[p];
    }
    v_[q] += dir * t;
    v_[leave] = to_lower ? wlo_[leave] : whi_[leave];
    status_[leave] = to_lower ? VarStatus::AtLower : VarStatus::AtUpper;
    lu_.update(r, col_);
    basic_[r] = q;
    status_[q] = VarStatus::Basic;
    ++pivots;
    if (lu_.num_updates() >= 1000 || lu_.wants_refactor()) {
      factorize();
      compute_basic_values();
      shift_basics();
    }
  }
  col_.clear();
  // Remove the shifts: nonbasic variables go to their true bounds.
  for (Index k = 0; k < N; ++k) {
    if (status_[k] == VarStatus::AtLower && v_[k] != lo_[k]) v_[k] = lo_[k];
    else if (status_[k] == VarStatus::AtUpper && v_[k] != hi_[k]) v_[k] = hi_[k];
  }
  if (st) st->pushes = pushes, st->pivots = pivots;
  if (lu_.num_updates() > 0) factorize();
  compute_basic_values();
  {
    // Second dual push: variables that the primal push moved to a bound while basic.
    Index cand = 0;
    const long long dp = dual_push(y, stop, &cand);
    if (st) st->dual_pivots += dp, st->dual_candidates += cand;
    factorize();
    compute_basic_values();
  }
  {
    double worst = 0.0;
    Index cnt = 0;
    for (Index p = 0; p < m_; ++p) {
      const Index b = basic_[p];
      const double viol = std::max(lo_[b] - v_[b], v_[b] - hi_[b]);
      if (viol > 1e-9 * (1.0 + std::abs(v_[b]))) ++cnt;
      worst = std::max(worst, viol);
    }
    if (st) st->primal_infeasibilities = cnt, st->max_primal_infeasibility = worst;
  }

  Basis basis;
  basis.status = status_;
  return basis;
}

long long Crossover::dual_push(std::vector<double> y, const std::atomic<bool>* stop, Index* candidates) {
  const Index N = n_ + m_;
  // Reduced costs with the interior point's duals: d = c - [A I]^T y.
  std::vector<double> d(static_cast<size_t>(N));
  double cmax = 0.0;
  for (Index j = 0; j < n_; ++j) {
    double s = cost_[j];
    for (Index p = A_.begin(j); p < A_.end(j); ++p) s -= A_.value[p] * y[A_.index[p]];
    d[j] = s;
    cmax = std::max(cmax, std::abs(cost_[j]));
  }
  for (Index i = 0; i < m_; ++i) d[n_ + i] = -y[i];
  const double dtol = 1e-9 * (1.0 + cmax);
  // Candidates: basic variables at a bound with a reduced cost pointing away from it (they belong
  // there as nonbasics), largest first.
  std::vector<std::pair<double, Index>> cand;
  for (Index p = 0; p < m_; ++p) {
    const Index j = basic_[p];
    if (std::abs(d[j]) <= dtol) continue;
    const bool at_lo = is_finite(lo_[j]) && v_[j] <= lo_[j] + btol(j, lo_[j]);
    const bool at_hi = is_finite(hi_[j]) && v_[j] >= hi_[j] - btol(j, hi_[j]);
    if ((at_lo && d[j] > 0.0) || (at_hi && d[j] < 0.0)) cand.emplace_back(-std::abs(d[j]), j);
  }
  std::sort(cand.begin(), cand.end());
  *candidates = static_cast<Index>(cand.size());
  if (cand.empty()) return 0;
  std::vector<Index> pos_of(static_cast<size_t>(N), -1);
  for (Index p = 0; p < m_; ++p) pos_of[basic_[p]] = p;
  HVector rho;
  rho.resize(m_);
  std::vector<double> alpha(static_cast<size_t>(N), 0.0);
  std::vector<Index> touched;
  std::vector<char> mark(static_cast<size_t>(N), 0);
  long long pivots = 0, done = 0;
  for (const auto& c : cand) {
    if (stop && (++done & 63) == 0 && stop->load(std::memory_order_relaxed)) break;
    const Index j = c.second;
    const Index p = pos_of[j];
    if (p < 0 || basic_[p] != j || std::abs(d[j]) <= dtol) continue;
    const bool at_lo = is_finite(lo_[j]) && v_[j] <= lo_[j] + btol(j, lo_[j]);
    const bool at_hi = is_finite(hi_[j]) && v_[j] >= hi_[j] - btol(j, hi_[j]);
    if (!((at_lo && d[j] > 0.0) || (at_hi && d[j] < 0.0))) continue;
    // rho = row p of B^-1; alpha_k = rho^T a_k for the nonbasic k.
    rho.clear();
    rho.dense[p] = 1.0;
    rho.nz[rho.count++] = p;
    lu_.btran(rho);
    touched.clear();
    for (Index e = 0; e < rho.count; ++e) {
      const Index i = rho.nz[e];
      const double r = rho.dense[i];
      if (r == 0.0) continue;
      for (Index t = At_.begin(i); t < At_.end(i); ++t) {
        const Index k = At_.index[t];
        if (!mark[k]) mark[k] = 1, touched.push_back(k);
        alpha[k] += r * At_.value[t];
      }
      const Index k = n_ + i;
      if (!mark[k]) mark[k] = 1, touched.push_back(k);
      alpha[k] += r;
    }
    // Move d_j to zero: d_k(tau) = d_k - tau * s * alpha_k, s = sign(d_j), tau in [0, |d_j|].
    const double s = d[j] > 0.0 ? 1.0 : -1.0;
    double amax = 0.0;
    for (Index k : touched) {
      if (status_[k] != VarStatus::Basic) amax = std::max(amax, std::abs(alpha[k]));
    }
    const double piv_tol = std::max(1e-9, 1e-7 * amax);
    double tmax = std::abs(d[j]);
    for (Index k : touched) {
      if (status_[k] == VarStatus::Basic || lo_[k] == hi_[k] || std::abs(alpha[k]) <= piv_tol) continue;
      const double rate = -s * alpha[k];
      if (status_[k] == VarStatus::AtLower && rate < 0.0) tmax = std::min(tmax, std::max(0.0, d[k] + dtol) / -rate);
      else if (status_[k] == VarStatus::AtUpper && rate > 0.0) tmax = std::min(tmax, std::max(0.0, -d[k] + dtol) / rate);
      else if (status_[k] == VarStatus::AtZero) tmax = std::min(tmax, dtol / std::abs(rate));
    }
    Index q = -1;
    double tau = std::abs(d[j]);
    if (tmax < std::abs(d[j])) {
      // Harris pass 2: the largest |alpha| among the blocking candidates.
      double best = 0.0;
      for (Index k : touched) {
        if (status_[k] == VarStatus::Basic || lo_[k] == hi_[k] || std::abs(alpha[k]) <= piv_tol) continue;
        const double rate = -s * alpha[k];
        double ratio = 0.0;
        if (status_[k] == VarStatus::AtLower && rate < 0.0) ratio = std::max(0.0, d[k]) / -rate;
        else if (status_[k] == VarStatus::AtUpper && rate > 0.0) ratio = std::max(0.0, -d[k]) / rate;
        else if (status_[k] == VarStatus::AtZero) ratio = 0.0;
        else continue;
        if (ratio <= tmax && std::abs(alpha[k]) > best) best = std::abs(alpha[k]), q = k, tau = ratio;
      }
    }
    // Dual step.
    for (Index k : touched) {
      if (status_[k] != VarStatus::Basic) d[k] -= tau * s * alpha[k];
    }
    d[j] -= tau * s;
    for (Index e = 0; e < rho.count; ++e) y[rho.nz[e]] += tau * s * rho.dense[rho.nz[e]];
    if (q < 0) {
      d[j] = 0.0;
    } else {
      // q enters at its bound value, j leaves at its bound (primal values unchanged up to the
      // snapping of j).
      column(q, col_);
      lu_.ftran(col_);
      if (std::abs(col_.dense[p]) > 1e-9) {
        lu_.update(p, col_);
        basic_[p] = q;
        pos_of[q] = p;
        pos_of[j] = -1;
        status_[q] = VarStatus::Basic;
        v_[j] = at_lo ? lo_[j] : hi_[j];
        status_[j] = at_lo ? VarStatus::AtLower : VarStatus::AtUpper;
        d[q] = 0.0;
        ++pivots;
        if (lu_.num_updates() >= 1000 || lu_.wants_refactor()) {
          factorize();
          for (Index t = 0; t < N; ++t) pos_of[t] = -1;
          for (Index t = 0; t < m_; ++t) pos_of[basic_[t]] = t;
        }
      }
      col_.clear();
    }
    for (Index k : touched) alpha[k] = 0.0, mark[k] = 0;
  }
  return pivots;
}

}  // namespace

Basis crossover(const SparseMatrix& A, const std::vector<double>& cost, const std::vector<double>& col_lb,
                const std::vector<double>& col_ub, const std::vector<double>& row_lb, const std::vector<double>& row_ub,
                const std::vector<double>& x, const std::vector<double>& y, CrossoverStats* stats,
                const std::atomic<bool>* stop) {
  Timer t;
  Crossover c(A, cost, col_lb, col_ub, row_lb, row_ub);
  Basis b = c.run(x, y, stats, stop);
  if (stats) stats->time = t.elapsed();
  return b;
}

LpStatus barrier_solve(DualSimplex& lp, const SparseMatrix& A, const std::vector<double>& cost,
                       const std::vector<double>& col_lb, const std::vector<double>& col_ub,
                       const std::vector<double>& row_lb, const std::vector<double>& row_ub,
                       const BarrierOptions& opt, BarrierStats* stats) {
  Timer timer;
  BarrierStats local;
  BarrierStats& s = stats ? *stats : local;
  s = BarrierStats{};
  IpmOptions io = opt.ipm;
  io.log = opt.log;
  io.stop = opt.stop;
  io.time_limit = std::min(io.time_limit, opt.time_limit);
  s.ipm = ipm_solve(A, cost, col_lb, col_ub, row_lb, row_ub, io);
  if (s.ipm.augmented && (s.ipm.status == IpmStatus::NumericalError || s.ipm.status == IpmStatus::IterationLimit) &&
      timer.elapsed() < 0.5 * io.time_limit) {
    // The augmented system is less robust than the normal equations: retry with those.
    if (opt.log) opt.log->log(1, "Barrier: retrying with the normal equations\n");
    io.kkt = 1;
    io.time_limit -= timer.elapsed();
    s.ipm = ipm_solve(A, cost, col_lb, col_ub, row_lb, row_ub, io);
  }
  if (s.ipm.status != IpmStatus::Optimal) {
    s.total_time = timer.elapsed();
    switch (s.ipm.status) {
      case IpmStatus::PrimalInfeasible: return LpStatus::Infeasible;
      case IpmStatus::DualInfeasible: return LpStatus::Unbounded;
      case IpmStatus::TimeLimit: return LpStatus::TimeLimit;
      case IpmStatus::IterationLimit: return LpStatus::IterationLimit;
      default: return LpStatus::NumericalError;
    }
  }
  const Basis b = crossover(A, cost, col_lb, col_ub, row_lb, row_ub, s.ipm.x, s.ipm.y, &s.crossover, opt.stop);
  if (opt.stop && opt.stop->load()) {
    s.total_time = timer.elapsed();
    return LpStatus::TimeLimit;
  }
  lp.set_basis(b);
  s.used = true;
  LpOptions lo;
  lo.perturb = true;  // the dual loop may still have work on degenerate problems
  lo.primal_first = true;
  lo.stop = opt.stop;
  lo.time_limit = std::max(0.0, opt.time_limit - timer.elapsed());
  const long long it0 = lp.iterations();
  const LpStatus st = lp.solve(lo);
  s.cleanup_iterations = lp.iterations() - it0;
  s.total_time = timer.elapsed();
  if (opt.log) {
    const CrossoverStats& c = s.crossover;
    opt.log->log(1, "Crossover: primal push %d superbasics (%lld pivots), dual push %d candidates (%lld pivots), %d "
                 "replaced, %.2fs; simplex cleanup %s, %lld iterations; %.2fs in total\n", c.superbasics, c.pivots,
                 c.dual_candidates, c.dual_pivots, c.replaced, c.time, to_string(st), s.cleanup_iterations,
                 s.total_time);
  }
  return st;
}

}  // namespace milp
