#include "milp/lp/simplex.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "milp/core/timer.hpp"

namespace milp {

const char* to_string(LpStatus s) {
  switch (s) {
    case LpStatus::Optimal: return "optimal";
    case LpStatus::Infeasible: return "infeasible";
    case LpStatus::Unbounded: return "unbounded";
    case LpStatus::CutoffReached: return "cutoff";
    case LpStatus::IterationLimit: return "iteration limit";
    case LpStatus::TimeLimit: return "time limit";
    case LpStatus::NumericalError: return "numerical error";
  }
  return "?";
}

namespace {
thread_local Timer t_timer;
}

void DualSimplex::load(const SparseMatrix& A, const std::vector<double>& cost, const std::vector<double>& col_lb,
                       const std::vector<double>& col_ub, const std::vector<double>& row_lb,
                       const std::vector<double>& row_ub) {
  n_ = A.num_major;
  m_ = A.num_minor;
  A_orig_ = A;
  At_orig_ = A.transpose();
  // Geometric row scaling by powers of two (exact; makes badly scaled rows tractable).
  rscale_.assign(m_, 1.0);
  static const bool no_scale = std::getenv("MILP_NO_LP_SCALING") != nullptr;
  if (!no_scale) {
    for (Index i = 0; i < m_; ++i) {
      compute_row_scale(i, At_orig_.index.data() + At_orig_.begin(i), At_orig_.value.data() + At_orig_.begin(i),
                        At_orig_.end(i) - At_orig_.begin(i));
    }
  }
  A_ = A;
  for (Index j = 0; j < n_; ++j)
    for (Index p = A_.begin(j); p < A_.end(j); ++p) A_.value[p] *= rscale_[A_.index[p]];
  At_ = A_.transpose();
  const Index N = n_ + m_;
  cost_.assign(N, 0.0);
  lb_.resize(N);
  ub_.resize(N);
  for (Index j = 0; j < n_; ++j) {
    cost_[j] = cost[j];
    lb_[j] = col_lb[j];
    ub_[j] = col_ub[j];
  }
  ftol_.assign(N, 1.0);
  for (Index i = 0; i < m_; ++i) {
    lb_[n_ + i] = -row_ub[i] * rscale_[i];
    ub_[n_ + i] = -row_lb[i] * rscale_[i];
    ftol_[n_ + i] = std::min(1.0, rscale_[i]);
  }
  wcost_ = cost_;
  wlb_ = lb_;
  wub_ = ub_;
  x_.assign(N, 0.0);
  d_.assign(N, 0.0);
  arow_.assign(N, 0.0);
  arow_mark_.assign(N, 0);
  arow_count_ = 0;
  rho_.resize(m_);
  alpha_.resize(m_);
  tau_.resize(m_);
  flipcol_.resize(m_);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  perturb_rand_.resize(n_);
  for (double& r : perturb_rand_) r = unif(rng_);
  compute_pert_scale();
  reset_basis();
}

void DualSimplex::compute_row_scale(Index i, const Index* idx, const double* val, Index len) {
  (void)idx;
  double mx = 0.0, mn = kInf;
  for (Index k = 0; k < len; ++k) {
    const double a = std::abs(val[k]);
    if (a == 0.0) continue;
    mx = std::max(mx, a);
    mn = std::min(mn, a);
  }
  rscale_[i] = mx > 0.0 ? std::ldexp(1.0, -static_cast<int>(std::lround(std::log2(std::sqrt(mx * mn))))) : 1.0;
}

const double* DualSimplex::x() const {
  xout_.assign(x_.begin(), x_.end());
  for (Index i = 0; i < m_; ++i) xout_[n_ + i] /= rscale_[i];
  return xout_.data();
}

void DualSimplex::reset_basis() {
  const Index N = n_ + m_;
  status_.assign(N, VarStatus::AtLower);
  basic_.resize(m_);
  for (Index i = 0; i < m_; ++i) {
    basic_[i] = n_ + i;
    status_[n_ + i] = VarStatus::Basic;
  }
  for (Index j = 0; j < n_; ++j) place_nonbasic(j);
  dse_.assign(m_, 1.0);
  dse_new_.clear();
  has_factor_ = false;
  values_ok_ = false;
  moves_.clear();
}

void DualSimplex::set_basis(const Basis& b) {
  const Index N = n_ + m_;
  if (static_cast<Index>(b.status.size()) != N) {
    reset_basis();
    return;
  }
  Index nb = 0;
  for (VarStatus s : b.status) nb += s == VarStatus::Basic;
  if (nb != m_) {
    reset_basis();
    return;
  }
  // DSE weights belong to the basis (||e_p B^-1||^2 for position p): a new basis gets exact weights,
  // computed at the next solve (one unit btran per row, as HiGHS). Unit weights made the warm starts
  // of nodes taken from the queue (a parent's basis, other rows) 3-10x longer than plunge steps
  // (roll3000: ~2300 iterations per such node LP with tableau cuts, 210 with exact weights).
  // MILP_LP_DSE_KEEP: 0 unit weights; 1 keep the weights of variables basic before and after;
  // 2 exact (default); 3 / 4 exact for the rows primal infeasible at the start only, kept / unit
  // weights for the others (cheaper, but 2-3x the iterations of 2 on roll3000).
  static const int keep = [] {
    const char* v = std::getenv("MILP_LP_DSE_KEEP");
    return v ? std::atoi(v) : 2;
  }();
  std::vector<double>& old_w = renum_dse_;
  if ((keep == 1 || keep == 3) && static_cast<Index>(dse_.size()) == m_ && static_cast<Index>(basic_.size()) == m_ &&
      static_cast<Index>(status_.size()) == N) {
    old_w.assign(N, 0.0);
    for (Index q = 0; q < m_; ++q) {
      if (basic_[q] >= 0 && basic_[q] < N) old_w[basic_[q]] = dse_[q];
    }
  } else {
    old_w.clear();
  }
  status_ = b.status;
  Index p = 0;
  for (Index j = 0; j < N; ++j) {
    if (status_[j] == VarStatus::Basic) basic_[p++] = j;
  }
  for (Index j = 0; j < N; ++j) place_nonbasic(j);
  dse_.assign(m_, 1.0);
  dse_new_.clear();
  if (!old_w.empty()) {
    for (Index q = 0; q < m_; ++q) {
      if (old_w[basic_[q]] > 0.0) dse_[q] = old_w[basic_[q]];
    }
  }
  dse_exact_ = keep >= 2 ? keep : 0;
  has_factor_ = false;
  values_ok_ = false;
  moves_.clear();
}

void DualSimplex::save_state(State& s) const {
  s.status = status_;
  s.basic = basic_;
  s.dse = dse_;
  s.has_factor = has_factor_;
  if (has_factor_) s.lu = lu_;
  s.has_values = values_ok_ && moves_.empty();
  if (s.has_values) {
    s.x = x_;
    s.d = d_;
  }
}

void DualSimplex::restore_state(const State& s) {
  status_ = s.status;
  basic_ = s.basic;
  dse_ = s.dse;
  dse_new_.clear();
  has_factor_ = s.has_factor;
  if (has_factor_) lu_ = s.lu;
  moves_.clear();
  values_ok_ = s.has_values && has_factor_ && s.x.size() == x_.size();
  if (values_ok_) {
    x_ = s.x;
    d_ = s.d;
  }
  // Bounds may have changed since the state was saved (and a phase 1 may
  // have left working bounds behind).
  wlb_ = lb_;
  wub_ = ub_;
  for (Index j = 0; j < n_ + m_; ++j) {
    if (status_[j] == VarStatus::Basic) continue;
    const double old = x_[j];
    place_nonbasic(j);
    note_move(j, old);
  }
}

void DualSimplex::note_move(Index j, double old_x) {
  if (!values_ok_ || x_[j] == old_x) return;
  if (static_cast<Index>(moves_.size()) > 16 + m_ / 8) {
    values_ok_ = false;  // many moves: recompute from scratch instead
    moves_.clear();
    return;
  }
  moves_.emplace_back(j, x_[j] - old_x);
}

void DualSimplex::apply_moves() {
  // x_B -= B^{-1} sum_j a_j delta_j
  if (moves_.empty()) return;
  std::vector<double>& v = alpha_.dense;
  alpha_.clear();
  for (auto [j, delta] : moves_) {
    if (j < n_) {
      for (Index p = A_.begin(j); p < A_.end(j); ++p) v[A_.index[p]] += A_.value[p] * delta;
    } else {
      v[j - n_] += delta;
    }
  }
  moves_.clear();
  lu_.ftran(v);
  for (Index p = 0; p < m_; ++p) x_[basic_[p]] -= v[p];
  std::fill(v.begin(), v.end(), 0.0);
}

Basis DualSimplex::get_basis() const { return Basis{status_}; }

void DualSimplex::set_col_bounds(Index j, double lb, double ub) {
  lb_[j] = wlb_[j] = lb;
  ub_[j] = wub_[j] = ub;
  const double old = x_[j];
  place_nonbasic(j);
  if (status_[j] != VarStatus::Basic) note_move(j, old);
}

void DualSimplex::set_row_bounds(Index i, double lb, double ub) {
  const Index j = n_ + i;
  lb_[j] = wlb_[j] = -ub * rscale_[i];
  ub_[j] = wub_[j] = -lb * rscale_[i];
  const double old = x_[j];
  place_nonbasic(j);
  if (status_[j] != VarStatus::Basic) note_move(j, old);
}

void DualSimplex::place_nonbasic(Index j) {
  VarStatus& s = status_[j];
  if (s == VarStatus::Basic) return;
  const double l = wlb_[j], u = wub_[j];
  if (l == u) {
    s = VarStatus::AtLower;
    x_[j] = l;
  } else if (s == VarStatus::AtUpper && is_finite(u)) {
    x_[j] = u;
  } else if (is_finite(l)) {
    s = VarStatus::AtLower;
    x_[j] = l;
  } else if (is_finite(u)) {
    s = VarStatus::AtUpper;
    x_[j] = u;
  } else {
    s = VarStatus::AtZero;
    x_[j] = 0.0;
  }
}

void DualSimplex::place_by_dual(Index j) {
  if (status_[j] == VarStatus::Basic) return;
  const double l = wlb_[j], u = wub_[j];
  if (is_finite(l) && is_finite(u)) {
    if (d_[j] >= 0.0 || l == u) {
      status_[j] = VarStatus::AtLower;
      x_[j] = l;
    } else {
      status_[j] = VarStatus::AtUpper;
      x_[j] = u;
    }
  } else {
    place_nonbasic(j);
  }
}

void DualSimplex::column(Index j, HVector& v) const {
  v.clear();
  if (j < n_) {
    for (Index p = A_.begin(j); p < A_.end(j); ++p) {
      v.dense[A_.index[p]] = A_.value[p];
      v.nz[v.count++] = A_.index[p];
    }
  } else {
    v.dense[j - n_] = 1.0;
    v.nz[v.count++] = j - n_;
  }
}

void DualSimplex::unit_btran(Index r) {
  rho_.clear();
  rho_.dense[r] = 1.0;
  rho_.nz[rho_.count++] = r;
  lu_.btran(rho_, dens_rho_);
}

bool DualSimplex::refactor(bool renumber) {
  std::vector<Index>&start = fac_start_, &index = fac_index_;
  std::vector<double>& value = fac_value_;
  start.assign(m_ + 1, 0);
  index.clear();
  value.clear();
  for (Index p = 0; p < m_; ++p) {
    const Index j = basic_[p];
    if (j < n_) {
      for (Index k = A_.begin(j); k < A_.end(j); ++k) {
        index.push_back(A_.index[k]);
        value.push_back(A_.value[k]);
      }
    } else {
      index.push_back(j - n_);
      value.push_back(1.0);
    }
    start[p + 1] = static_cast<Index>(index.size());
  }
  std::vector<std::pair<Index, Index>>& replaced = fac_replaced_;
  lu_.factorize(m_, start, index, value, &replaced);
  if (!replaced.empty()) values_ok_ = false, moves_.clear();
  for (auto [p, row] : replaced) {
    const Index jold = basic_[p];
    status_[jold] = VarStatus::AtLower;
    place_nonbasic(jold);
    basic_[p] = n_ + row;
    status_[n_ + row] = VarStatus::Basic;
    dse_[p] = 1.0;
  }
  if (renumber) {
    // Position p becomes the column pivoted in row p: ftran/btran results need no permutation.
    lu_.renumber_positions(renum_old_);
    bool identity = true;
    for (Index p = 0; p < m_ && identity; ++p) identity = renum_old_[p] == p;
    if (!identity) {
      std::vector<Index>& nb = renum_new_;
      nb.resize(m_);
      renum_dse_.resize(m_);
      for (Index p = 0; p < m_; ++p) {
        nb[p] = basic_[renum_old_[p]];
        renum_dse_[p] = dse_[renum_old_[p]];
      }
      basic_.swap(nb);
      dse_.swap(renum_dse_);
      if (!dse_new_.empty()) {
        for (Index p = 0; p < m_; ++p) nb[renum_old_[p]] = p;  // old position -> new
        for (Index& p : dse_new_)
          if (p < m_) p = nb[p];
      }
      infeas_dirty_ = true;
    }
  }
  has_factor_ = true;
  return replaced.empty();
}

void DualSimplex::rebuild_infeasible(double ptol) {
  infeas_pos_.assign(m_, -1);
  infeas_list_.clear();
  for (Index p = 0; p < m_; ++p) update_infeasible(p, ptol);
  infeas_dirty_ = false;
}

void DualSimplex::update_infeasible(Index p, double ptol) {
  const Index j = basic_[p];
  const double v = x_[j];
  const double t = ptol * ftol_[j];
  const bool inf = v < wlb_[j] - t || v > wub_[j] + t;
  Index& k = infeas_pos_[p];
  if (inf == (k >= 0)) return;
  if (inf) {
    k = static_cast<Index>(infeas_list_.size());
    infeas_list_.push_back(p);
  } else {
    const Index last = infeas_list_.back();
    infeas_list_[k] = last;
    infeas_pos_[last] = k;
    infeas_list_.pop_back();
    k = -1;
  }
}

void DualSimplex::compute_primal() {
  infeas_dirty_ = true;
  alpha_.clear();
  std::vector<double>& rhs = alpha_.dense;
  for (Index j = 0; j < n_; ++j) {
    if (status_[j] == VarStatus::Basic || x_[j] == 0.0) continue;
    const double v = x_[j];
    for (Index p = A_.begin(j); p < A_.end(j); ++p) rhs[A_.index[p]] -= A_.value[p] * v;
  }
  for (Index i = 0; i < m_; ++i) {
    if (status_[n_ + i] != VarStatus::Basic) rhs[i] -= x_[n_ + i];
  }
  lu_.ftran(rhs);
  for (Index p = 0; p < m_; ++p) x_[basic_[p]] = rhs[p];
  std::fill(rhs.begin(), rhs.end(), 0.0);
}

bool DualSimplex::compute_duals(bool check) {
  rho_.clear();
  std::vector<double>& y = rho_.dense;
  for (Index p = 0; p < m_; ++p) y[p] = wcost_[basic_[p]];
  lu_.btran(y);
  // With `check`, basic reduced costs must vanish to a fresh-factor level.
  bool ok = true;
  for (Index j = 0; j < n_; ++j) {
    const bool basic = status_[j] == VarStatus::Basic;
    if (basic && !check) {
      d_[j] = 0.0;
      continue;
    }
    double dj = wcost_[j], sc = std::abs(dj);
    for (Index p = A_.begin(j); p < A_.end(j); ++p) {
      const double t = A_.value[p] * y[A_.index[p]];
      dj -= t;
      sc += std::abs(t);
    }
    if (basic) {
      ok = ok && std::abs(dj) <= 1e-9 * (1.0 + sc);
      dj = 0.0;
    }
    d_[j] = dj;
  }
  for (Index i = 0; i < m_; ++i) {
    const Index j = n_ + i;
    if (status_[j] != VarStatus::Basic) {
      d_[j] = wcost_[j] - y[i];
    } else {
      if (check) ok = ok && std::abs(wcost_[j] - y[i]) <= 1e-9 * (1.0 + std::abs(wcost_[j]) + std::abs(y[i]));
      d_[j] = 0.0;
    }
  }
  std::fill(y.begin(), y.end(), 0.0);
  return ok;
}

bool DualSimplex::primal_residuals_small() {
  // [A I] x = 0 row by row, relative to the magnitude of the terms.
  std::vector<double>& r = tau_.dense;
  std::vector<double>& scale = flipcol_.dense;
  tau_.clear();
  flipcol_.clear();
  for (Index j = 0; j < n_; ++j) {
    const double v = x_[j];
    if (v == 0.0) continue;
    for (Index p = A_.begin(j); p < A_.end(j); ++p) {
      const double t = A_.value[p] * v;
      r[A_.index[p]] += t;
      scale[A_.index[p]] += std::abs(t);
    }
  }
  bool ok = true;
  for (Index i = 0; i < m_ && ok; ++i) {
    const double sv = x_[n_ + i];
    ok = std::abs(r[i] + sv) <= 1e-9 * (1.0 + scale[i] + std::abs(sv));
  }
  std::fill(r.begin(), r.end(), 0.0);
  std::fill(scale.begin(), scale.end(), 0.0);
  return ok;
}

Index DualSimplex::count_dual_infeasibilities(bool flip_boxed) {
  const double tol = 1e-7;
  Index count = 0;
  bool flipped = false;
  for (Index j = 0; j < n_ + m_; ++j) {
    const VarStatus s = status_[j];
    if (s == VarStatus::Basic || wlb_[j] == wub_[j]) continue;
    const double dj = d_[j];
    const bool infeasible = (s == VarStatus::AtLower && dj < -tol) || (s == VarStatus::AtUpper && dj > tol) ||
                            (s == VarStatus::AtZero && std::abs(dj) > tol);
    if (!infeasible) continue;
    if (flip_boxed && is_finite(wlb_[j]) && is_finite(wub_[j])) {
      place_by_dual(j);
      flipped = true;
    } else {
      ++count;
    }
  }
  if (flipped) compute_primal();
  return count;
}

void DualSimplex::perturb_costs(double mul) {
  // Nonbasic structurals only, pushing d_j further into its feasible side, so
  // the duals stay valid without a recomputation (d_j moves with c_j).
  // Magnitude as in HiGHS: 5e-7 (1 + |c_j|) (1 + rand) times a cost scale (max |c|, its fourth
  // root above 100, at most 1 when hardly any variable is boxed). Smaller perturbations made
  // degenerate LPs much slower (mzzv11 root: 30.6k iterations with 5e-7 alone, 12.9k with this).
  const double base = 5e-7 * pert_scale_ * mul;
  for (Index j = 0; j < n_; ++j) {
    const VarStatus st = status_[j];
    if (st == VarStatus::Basic || st == VarStatus::AtZero || lb_[j] == ub_[j]) continue;
    const double xi = (1.0 + std::abs(cost_[j])) * base * (1.0 + perturb_rand_[j]);
    const double delta = st == VarStatus::AtLower ? xi : -xi;
    wcost_[j] = cost_[j] + delta;
    d_[j] += delta;
  }
}

void DualSimplex::boost_perturbation(double extra) {
  // Adds `extra` times the perturbation to the nonbasic structurals, again into their feasible side.
  const double base = 5e-7 * pert_scale_ * extra;
  for (Index j = 0; j < n_; ++j) {
    const VarStatus st = status_[j];
    if (st == VarStatus::Basic || st == VarStatus::AtZero || lb_[j] == ub_[j]) continue;
    const double xi = (1.0 + std::abs(cost_[j])) * base * (1.0 + perturb_rand_[j]);
    const double delta = st == VarStatus::AtLower ? xi : -xi;
    wcost_[j] += delta;
    d_[j] += delta;
  }
}

void DualSimplex::compute_pert_scale() {
  double bigc = 0.0;
  for (Index j = 0; j < n_; ++j) bigc = std::max(bigc, std::abs(cost_[j]));
  if (bigc > 100.0) bigc = std::sqrt(std::sqrt(bigc));
  Index boxed = 0;
  for (Index j = 0; j < n_ + m_; ++j) boxed += is_finite(lb_[j]) && is_finite(ub_[j]);
  if (boxed < 0.01 * static_cast<double>(n_ + m_)) bigc = std::min(bigc, 1.0);
  static const double mult = [] {
    const char* v = std::getenv("MILP_LP_PERTURB");
    return v ? std::atof(v) : 1.0;
  }();
  pert_scale_ = std::max(bigc, 1e-3) * mult;
}

void DualSimplex::set_seed(unsigned seed) {
  rng_.seed(12345u + seed * 7919u);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  perturb_rand_.resize(n_);
  for (double& r : perturb_rand_) r = unif(rng_);
}

void DualSimplex::pivot_row(const HVector& rho, bool nonbasic_only) {
  // Clear the previous row via its index list, then accumulate row-wise.
  arow_count_ = 0;
  if (arow_mark_.size() != arow_.size() || ++arow_gen_ == 0) {
    arow_mark_.assign(arow_.size(), 0);
    arow_gen_ = 1;
  }
  const uint32_t gen = arow_gen_;
  uint32_t* mark = arow_mark_.data();
  if (arow_nz_.size() < arow_.size()) arow_nz_.resize(arow_.size());
  Index* nz = arow_nz_.data();
  Index cnt = 0;
  if (nonbasic_only) {
    // For a dense rho, column-wise dot products over the nonbasic structurals
    // beat row-wise scattering (no marks, no basic columns).
    Index row_work = 0;
    for (Index k = 0; k < rho.count; ++k) row_work += At_.end(rho.nz[k]) - At_.begin(rho.nz[k]);
    if (2 * row_work > A_.nnz() + n_) {
      const double* r = rho.dense.data();
      for (Index j = 0; j < n_; ++j) {
        if (status_[j] == VarStatus::Basic) continue;
        double v = 0.0;
        for (Index p = A_.begin(j); p < A_.end(j); ++p) v += A_.value[p] * r[A_.index[p]];
        if (v == 0.0) continue;
        arow_[j] = v;
        mark[j] = gen;
        nz[cnt++] = j;
      }
      for (Index k = 0; k < rho.count; ++k) {
        const Index i = rho.nz[k];
        if (rho.dense[i] == 0.0) continue;
        arow_[n_ + i] = rho.dense[i];
        mark[n_ + i] = gen;
        nz[cnt++] = n_ + i;
      }
      arow_count_ = cnt;
      return;
    }
  }
  for (Index k = 0; k < rho.count; ++k) {
    const Index i = rho.nz[k];
    const double r = rho.dense[i];
    if (r == 0.0) continue;
    // Branch-free: whether j was touched before is unpredictable.
    double* ar = arow_.data();
    const Index* idx = At_.index.data();
    const double* val = At_.value.data();
    for (Index p = At_.begin(i), e = At_.end(i); p < e; ++p) {
      const Index j = idx[p];
      const bool seen = mark[j] == gen;
      nz[cnt] = j;
      cnt += !seen;
      mark[j] = gen;
      ar[j] = (seen ? ar[j] : 0.0) + r * val[p];
    }
    arow_[n_ + i] = r;
    mark[n_ + i] = gen;
    nz[cnt++] = n_ + i;
  }
  arow_count_ = cnt;
}

bool DualSimplex::limits_hit(const LpOptions& opt, LpStatus* st) const {
  if (opt.iteration_limit >= 0 && total_iterations_ - iter_start_ >= opt.iteration_limit) {
    *st = LpStatus::IterationLimit;
    return true;
  }
  if (opt.time_limit < kInf && (total_iterations_ & 15) == 0 && t_timer.elapsed() - solve_start_ > opt.time_limit) {
    *st = LpStatus::TimeLimit;
    return true;
  }
  if (opt.stop && opt.stop->load(std::memory_order_relaxed)) {
    *st = LpStatus::TimeLimit;
    return true;
  }
  return false;
}

double DualSimplex::lagrangian_bound() {
  // Valid lower bound for any multipliers y: min over the box of (c - A^T y)^T x.
  tau_.clear();
  std::vector<double>& y = tau_.dense;
  for (Index p = 0; p < m_; ++p) y[p] = cost_[basic_[p]];
  lu_.btran(y);
  double bound = 0.0;
  for (Index j = 0; j < n_ + m_; ++j) {
    double dj;
    if (j < n_) {
      dj = cost_[j];
      for (Index p = A_.begin(j); p < A_.end(j); ++p) dj -= A_.value[p] * y[A_.index[p]];
    } else {
      dj = -y[j - n_];
    }
    if (std::abs(dj) < 1e-11) continue;
    const double b = dj > 0 ? lb_[j] : ub_[j];
    if (!is_finite(b)) {
      bound = -kInf;
      break;
    }
    bound += dj * b;
  }
  std::fill(y.begin(), y.end(), 0.0);
  return bound;
}

LpStatus DualSimplex::fail(const char* why) {
  fail_reason_ = why;
  return LpStatus::NumericalError;
}

LpStatus DualSimplex::solve(const LpOptions& opt) {
  fail_reason_ = "";
  solve_start_ = t_timer.elapsed();
  iter_start_ = total_iterations_;
  has_farkas_ = false;
  const LpStatus st = run(opt);
  if (st != LpStatus::Infeasible) has_farkas_ = false;
  return st;
}

LpStatus DualSimplex::run(const LpOptions& opt) {
  wcost_ = cost_;
  if (values_ok_ && has_factor_ && lu_.num_updates() <= opt.refactor_interval / 2) {
    // Warm: same basis and costs as the last optimal solve, only bounds moved
    // (the working bounds equal the true ones outside of run()).
    apply_moves();
  } else {
    wlb_ = lb_;
    wub_ = ub_;
    for (Index j = 0; j < n_ + m_; ++j) place_nonbasic(j);
    if (!has_factor_ || lu_.num_updates() > opt.refactor_interval / 2) refactor();
    compute_duals();
    compute_primal();
  }
  values_ok_ = false;
  moves_.clear();
  if (dse_exact_) {
    // 2: all rows; 3, 4: only the rows that are primal infeasible now (the dual simplex prices
    // those; rows becoming infeasible later keep their kept (3) or unit (4) weights).
    const bool all = dse_exact_ == 2;
    dse_exact_ = 0;
    for (Index p = 0; p < m_; ++p) {
      if (!all) {
        const Index j = basic_[p];
        const double t = opt.tol.primal_feas * ftol_[j];
        if (x_[j] >= lb_[j] - t && x_[j] <= ub_[j] + t) continue;
      }
      unit_btran(p);
      double w = 0.0;
      for (Index e = 0; e < rho_.count; ++e) w += rho_.dense[rho_.nz[e]] * rho_.dense[rho_.nz[e]];
      dse_[p] = std::max(w, 1e-6);
    }
    dse_new_.clear();
  }
  if (!dse_new_.empty()) {
    // A new row's basic logical has DSE weight ||e_r B^{-1}||^2 = 1 + ||a_B B^{-1}||^2, often far
    // above 1 (dense cuts); a weight of 1 would make the dual simplex favour the new rows.
    for (Index p : dse_new_) {
      if (p >= m_ || basic_[p] != n_ + p) continue;
      unit_btran(p);
      double w = 0.0;
      for (Index e = 0; e < rho_.count; ++e) w += rho_.dense[rho_.nz[e]] * rho_.dense[rho_.nz[e]];
      dse_[p] = std::max(w, 1.0);
    }
    dse_new_.clear();
  }
  if (opt.perturb) {
    // A long cold solve (started from a slack basis, still running after max(m/2, 5000) iterations) gets a 100x
    // larger perturbation from then on. Long cold solves of dual degenerate LPs stall with the HiGHS
    // magnitude (mzzv11's root LP: 11k-100k iterations depending on the random factors, mean 48k over
    // 12 seeds; 7.6k-14k, mean 11k with this). Applied from the start it changed the optimal vertex
    // (and the cuts) of short root LPs too (netdesign_large 0.9 -> 1.6 s; with m/2 alone neos-3381206-awhea,
    // 1.1k iterations, got a root bound 451.65 instead of 452.0 after cuts: 3 s -> time limit); warm solves keep the
    // HiGHS magnitude (larger made node LP sequences slower: more cleanup after removal).
    static const double kColdMul = [] { const char* v = std::getenv("MILP_LP_COLD_PERTURB"); return v ? std::atof(v) : 100.0; }();
    static const double kColdIters = [] { const char* v = std::getenv("MILP_LP_COLD_ITERS"); return v ? std::atof(v) : 0.5; }();
    bool cold = true;
    for (Index p = 0; p < m_ && cold; ++p) cold = basic_[p] >= n_;
    boost_at_ = -1;
    if (cold && kColdIters > 0.0) {
      perturb_costs(1.0);
      boost_at_ = iter_start_ + std::max(5000LL, static_cast<long long>(kColdIters * m_));
      boost_mul_ = kColdMul - 1.0;
    } else {
      perturb_costs(cold ? kColdMul : 1.0);
    }
  }
  const Index dual_infeasible = opt.primal_first ? count_dual_infeasibilities(false) : 0;
  if (dual_infeasible > 0) {
    bool primal_feasible = true;
    for (Index p = 0; p < m_ && primal_feasible; ++p) {
      const Index j = basic_[p];
      primal_feasible = x_[j] >= lb_[j] - opt.tol.primal_feas * ftol_[j] && x_[j] <= ub_[j] + opt.tol.primal_feas * ftol_[j];
    }
    if (primal_feasible) {
      // Capped (Dantzig pricing without anti-degeneracy measures can stall): a few pivots per dual
      // infeasibility, then the dual simplex takes over from the basis reached (rmatr200-p5's
      // relaxation: 20k stalling primal pivots with the cap of the primal cleanup below).
      const long long cap = std::min<long long>(1000 + m_ / 2, 1000 + 3LL * dual_infeasible);
      LpOptions popt = opt;
      popt.iteration_limit = opt.iteration_limit >= 0 ? std::min(opt.iteration_limit, cap) : cap;
      const LpStatus st = primal_loop(popt);
      if (st == LpStatus::IterationLimit && !(opt.iteration_limit >= 0 && opt.iteration_limit <= cap)) {
        dse_.assign(m_, 1.0);  // the primal pivots do not maintain the weights
      } else if (st != LpStatus::Optimal && st != LpStatus::NumericalError) {
        return st;
      }
      if (st == LpStatus::NumericalError) {
        refactor();
        compute_duals();
        compute_primal();
      }
      if (st != LpStatus::Optimal) {
        // The dual simplex continues: exact DSE weights (unit weights made it far slower).
        for (Index p = 0; p < m_; ++p) {
          unit_btran(p);
          double w = 0.0;
          for (Index e = 0; e < rho_.count; ++e) w += rho_.dense[rho_.nz[e]] * rho_.dense[rho_.nz[e]];
          dse_[p] = std::max(w, 1e-6);
        }
      }
    }
  }
  if (count_dual_infeasibilities(true) > 0) {
    // Dual phase 1 on the auxiliary problem where every variable is boxed.
    for (Index j = 0; j < n_ + m_; ++j) {
      const double l = lb_[j], u = ub_[j];
      if (is_finite(l) && is_finite(u)) {
        wlb_[j] = wub_[j] = 0.0;
      } else if (is_finite(l)) {
        wlb_[j] = 0.0, wub_[j] = 1.0;
      } else if (is_finite(u)) {
        wlb_[j] = -1.0, wub_[j] = 0.0;
      } else {
        wlb_[j] = -1000.0, wub_[j] = 1000.0;
      }
      if (status_[j] == VarStatus::AtZero) status_[j] = VarStatus::AtLower;
      place_by_dual(j);
    }
    compute_primal();
    LpStatus st = dual_loop(opt, true);
    if (st != LpStatus::Optimal && st != LpStatus::Infeasible) return st;
    wlb_ = lb_;
    wub_ = ub_;
    for (Index j = 0; j < n_ + m_; ++j) {
      if (status_[j] == VarStatus::Basic) continue;
      if (!is_finite(lb_[j]) && !is_finite(ub_[j])) status_[j] = VarStatus::AtZero;
      place_by_dual(j);
    }
    compute_duals();
    compute_primal();
    if (count_dual_infeasibilities(true) > 0) {
      // Dual infeasible (possibly only numerically): find a primal feasible
      // point with zero costs, then let the primal simplex either reach the
      // optimum or prove unboundedness with a ray.
      std::fill(wcost_.begin(), wcost_.end(), 0.0);
      compute_duals();
      st = dual_loop(opt, false);
      if (st != LpStatus::Optimal) {
        wcost_ = cost_;
        return st;
      }
      wcost_ = cost_;
      compute_duals();
      st = primal_loop(opt);
      if (st != LpStatus::Optimal) return st;
    }
  }
  static const bool trace = std::getenv("MILP_LP_TRACE") != nullptr;
  for (int round = 0; round < 20; ++round) {
    const long long it_r = total_iterations_;
    LpStatus st = dual_loop(opt, false);
    if (trace && round > 0) std::fprintf(stderr, "LPTRACE round %d dual loop %s %lld it\n", round, to_string(st), total_iterations_ - it_r);
    if (st != LpStatus::Optimal) return st;
    // Remove perturbation / cost shifts and check optimality w.r.t. true costs.
    // Shifts of nonbasic costs move only their own d_j; shifted basic costs
    // change the duals.
    bool basic_shifted = false;
    for (Index p = 0; p < m_ && !basic_shifted; ++p) basic_shifted = wcost_[basic_[p]] != cost_[basic_[p]];
    if (basic_shifted) {
      wcost_ = cost_;
      compute_duals();
    } else {
      for (Index j = 0; j < n_ + m_; ++j) {
        if (wcost_[j] == cost_[j]) continue;
        d_[j] -= wcost_[j] - cost_[j];
        wcost_[j] = cost_[j];
      }
    }
    if (trace) {
      double worst = 0.0;
      Index cnt = 0, boxed = 0;
      for (Index j = 0; j < n_ + m_; ++j) {
        const VarStatus vs = status_[j];
        if (vs == VarStatus::Basic || wlb_[j] == wub_[j]) continue;
        const double dj = d_[j];
        const double inf = vs == VarStatus::AtLower ? -dj : vs == VarStatus::AtUpper ? dj : std::abs(dj);
        if (inf > 1e-7) ++cnt, boxed += is_finite(wlb_[j]) && is_finite(wub_[j]), worst = std::max(worst, inf);
      }
      std::fprintf(stderr, "LPTRACE round %d: after unperturb %d dual infeasible (%d boxed), max %.3g, basic_shifted %d\n", round, cnt, boxed, worst, (int)basic_shifted);
    }
    // Primal cleanup (as HiGHS): the dual loop ended primal feasible, so small dual infeasibilities
    // left by removing the perturbation / cost shifts are repaired by the primal simplex from this
    // basis. Flipping the boxed ones instead makes the basis primal infeasible, and on degenerate LPs
    // the following dual loop (with new Harris shifts) leaves new dual infeasibilities: ns1830653's node
    // LPs alternated like that for 20 rounds and failed.
    static const int kFlipRounds = [] {
      const char* v = std::getenv("MILP_LP_FLIP_ROUNDS");
      return v ? std::atoi(v) : 0;
    }();
    if (round >= kFlipRounds && count_dual_infeasibilities(false) > 0) {
      bool primal_feasible = true;
      for (Index p = 0; p < m_ && primal_feasible; ++p) {
        const Index j = basic_[p];
        primal_feasible = x_[j] >= lb_[j] - opt.tol.primal_feas * ftol_[j] && x_[j] <= ub_[j] + opt.tol.primal_feas * ftol_[j];
      }
      if (primal_feasible) {
        // Capped (Dantzig pricing, no anti-cycling): past the cap the boxed variables are flipped instead.
        const long long done = total_iterations_ - iter_start_, cap = done + 5000 + 2LL * m_;
        LpOptions copt = opt;
        copt.iteration_limit = opt.iteration_limit >= 0 ? std::min(opt.iteration_limit, cap) : cap;
        const long long it_p = total_iterations_;
        st = primal_loop(copt);
        if (trace) std::fprintf(stderr, "LPTRACE round %d: primal cleanup %s %lld it\n", round, to_string(st), total_iterations_ - it_p);
        if (st == LpStatus::Optimal) continue;
        if (st != LpStatus::IterationLimit || (opt.iteration_limit >= 0 && opt.iteration_limit <= cap)) return st;
        dse_.assign(m_, 1.0);  // the primal pivots do not maintain the weights
      }
    }
    if (count_dual_infeasibilities(true) == 0) {
      // Never trust incrementally updated values for the final verdict: the
      // primal values must pass a residual check (else they are recomputed),
      // the duals are recomputed, and if the updated factorization leaves
      // residuals a fresh one would not, it is refactorized.
      std::vector<double> x_old;
      static const bool debug_drift = std::getenv("MILP_LP_DRIFT") != nullptr;
      if (debug_drift) x_old = x_;
      const bool fresh = lu_.num_updates() == 0;
      bool ok = !debug_drift && primal_residuals_small();
      if (!ok) {
        compute_primal();
        ok = fresh || primal_residuals_small();
      }
      ok = compute_duals(!fresh) && ok;
      if (!ok && !fresh) {
        refactor();
        compute_primal();
        compute_duals();
      }
      if (debug_drift) {
        double drift = 0.0;
        for (Index p = 0; p < m_; ++p) drift = std::max(drift, std::abs(x_[basic_[p]] - x_old[basic_[p]]));
        if (drift > 1e-6) std::fprintf(stderr, "LP drift %.3g at optimality (round %d)\n", drift, round);
      }
      if (count_dual_infeasibilities(true) > 0) {
        if (trace) std::fprintf(stderr, "LPTRACE round %d: dual infeasible after recompute (fresh %d ok %d)\n", round, (int)fresh, (int)ok);
        continue;
      }
      bool primal_ok = true;
      for (Index p = 0; p < m_ && primal_ok; ++p) {
        const Index j = basic_[p];
        primal_ok = x_[j] >= lb_[j] - opt.tol.primal_feas * ftol_[j] && x_[j] <= ub_[j] + opt.tol.primal_feas * ftol_[j];
      }
      if (primal_ok) {
        values_ok_ = true;
        return LpStatus::Optimal;
      }
      if (trace) std::fprintf(stderr, "LPTRACE round %d: primal infeasible after recompute/flips\n", round);
      continue;
    }
    // The primal simplex needs a primal feasible start, which bound flips of
    // boxed variables may have destroyed. If so, shift the costs of the
    // dual infeasible variables (making the basis dual feasible) and let the
    // dual simplex restore primal feasibility first; the shifts are removed
    // at the top of the next round.
    bool primal_feasible = true;
    for (Index p = 0; p < m_ && primal_feasible; ++p) {
      const Index j = basic_[p];
      primal_feasible = x_[j] >= lb_[j] - opt.tol.primal_feas * ftol_[j] && x_[j] <= ub_[j] + opt.tol.primal_feas * ftol_[j];
    }
    if (trace) std::fprintf(stderr, "LPTRACE round %d: primal feasible %d -> %s\n", round, (int)primal_feasible, primal_feasible ? "primal loop" : "cost shifts");
    if (!primal_feasible) {
      for (Index j = 0; j < n_ + m_; ++j) {
        const VarStatus vs = status_[j];
        if (vs == VarStatus::Basic || wlb_[j] == wub_[j]) continue;
        const double dj = d_[j];
        if ((vs == VarStatus::AtLower && dj < 0) || (vs == VarStatus::AtUpper && dj > 0) || vs == VarStatus::AtZero) {
          wcost_[j] -= dj;
          d_[j] = 0.0;
        }
      }
      continue;
    }
    const long long it_p = total_iterations_;
    st = primal_loop(opt);
    if (trace) std::fprintf(stderr, "LPTRACE round %d: primal loop %s %lld it\n", round, to_string(st), total_iterations_ - it_p);
    if (st != LpStatus::Optimal) return st;
  }
  return fail("no convergence after 20 rounds");
}

LpStatus DualSimplex::infeasible(bool phase1) {
  // rho_ is the row of B^{-1} of the leaving variable whose bound cannot be reached.
  if (!phase1) {
    farkas_.assign(rho_.dense.begin(), rho_.dense.begin() + m_);
    has_farkas_ = true;
  }
  return LpStatus::Infeasible;
}

bool DualSimplex::farkas_proof(std::vector<double>& y) const {
  if (!has_farkas_) return false;
  y = farkas_;
  for (Index i = 0; i < static_cast<Index>(y.size()) && i < m_; ++i) y[i] *= rscale_[i];
  return true;
}

void DualSimplex::build_move() {
  const Index N = n_ + m_;
  move_.resize(N);
  for (Index j = 0; j < N; ++j) move_[j] = move_of(j);
}

LpStatus DualSimplex::dual_loop(const LpOptions& opt, bool phase1) {
  const double ptol = opt.tol.primal_feas;
  const double dtol = opt.tol.dual_feas;
  int mismatch_retries = 0;
  infeas_dirty_ = true;
  bool move_ok = false;  // refactor() may replace columns (status changes): rebuild after it
  for (;;) {
    LpStatus lim;
    if (limits_hit(opt, &lim)) return lim;
    if (lu_.num_updates() >= opt.refactor_interval || lu_.wants_refactor()) {
      refactor();
      compute_primal();
      compute_duals();
      move_ok = false;
    }
    if (infeas_dirty_) rebuild_infeasible(ptol);
    if (boost_at_ >= 0 && !phase1 && total_iterations_ >= boost_at_) {
      boost_at_ = -1;
      boost_perturbation(boost_mul_);
    }
    // Objective cutoff check (phase 2 only, every few iterations).
    if (!phase1 && opt.objective_cutoff < kInf && (total_iterations_ % 10) == 0) {
      double obj = 0.0;
      for (Index j = 0; j < n_; ++j) obj += cost_[j] * x_[j];
      if (obj > opt.objective_cutoff) {
        const double lb = lagrangian_bound();
        if (lb > opt.objective_cutoff) return LpStatus::CutoffReached;
      }
    }

    // ---- Pricing: dual steepest edge over the primal infeasible positions ----
    Index r = -1;
    double best = 0.0;
    for (const Index p : infeas_list_) {
      const Index j = basic_[p];
      const double v = x_[j];
      double infeas = 0.0;
      if (v < wlb_[j] - ptol * ftol_[j]) infeas = wlb_[j] - v;
      else if (v > wub_[j] + ptol * ftol_[j]) infeas = v - wub_[j];
      if (infeas == 0.0) continue;
      const double score = infeas * infeas / dse_[p];
      if (score > best) best = score, r = p;
    }
    if (r < 0) {
      // NaN primal values compare false above; never report them as optimal.
      bool finite = true;
      for (Index p = 0; p < m_ && finite; ++p) finite = std::isfinite(x_[basic_[p]]);
      if (finite) return LpStatus::Optimal;
      if (lu_.num_updates() > 0 && mismatch_retries < 3) {
        ++mismatch_retries;
        refactor();
        compute_primal();
        compute_duals();
        move_ok = false;
        continue;
      }
      return fail("non-finite primal values");
    }

    const Index jl = basic_[r];
    const bool to_lower = x_[jl] < wlb_[jl];
    const double target = to_lower ? wlb_[jl] : wub_[jl];
    const double s = to_lower ? 1.0 : -1.0;

    // ---- BTRAN and pivot row ----
    unit_btran(r);
    note_density(dens_rho_, rho_);
    pivot_row(rho_, true);

    // ---- Bound flipping ratio test with Harris tolerance ----
    // cands_ only grows (resizing it down and up again would re-zero it).
    if (cands_.size() < static_cast<size_t>(arow_count_)) cands_.resize(arow_count_);
    // Candidates: nonbasic, not fixed, |alpha| >= 1e-9, and moving away from their bound helps
    // (at lower: s alpha < 0, at upper: s alpha > 0, free: either). The Harris bound of the first
    // group is taken along.
    if (!move_ok) build_move(), move_ok = true;
    size_t nc = 0;
    double theta0 = kInf;
    {
      const Index* anz = arow_nz_.data();
      const double* ar = arow_.data();
      const int8_t* mvv = move_.data();
      Candidate* cd = cands_.data();
      for (Index t = 0; t < arow_count_; ++t) {
        const Index j = anz[t];
        const int mv = mvv[j];
        if (mv == 0) continue;
        const double a = ar[j];
        double dj_eff, aa;
        if (mv == 2) {
          aa = std::abs(a);
          if (aa < 1e-9) continue;
          dj_eff = 0.0;
        } else {
          aa = -s * mv * a;  // = |a| when the sign is right
          if (aa < 1e-9) continue;
          dj_eff = mv * d_[j];
        }
        const double ratio = std::max(dj_eff, 0.0) / aa;
        cd[nc++] = {j, ratio, aa};
        theta0 = std::min(theta0, ratio + dtol / aa);
      }
    }
    if (nc == 0) return infeasible(phase1);

    // Process breakpoints group by group (Harris groups). The first group is
    // found by partitioning; if bound flips continue past it, the remaining
    // candidates go into a min-heap by ratio and each group is popped from it:
    // a group is everything with ratio <= min(ratio + dtol/|alpha|) over the
    // remaining candidates, and popping stops once the next ratio exceeds that
    // bound (O(C + popped log C) instead of O(C * groups) or a full sort).
    double slope = std::abs(x_[jl] - target);
    bool numerical_trouble = !std::isfinite(slope);
    Index q = -1;
    size_t k = 0, flip_end = 0;
    bool heap_mode = false;
    const auto heap_cmp = [](const Candidate& x, const Candidate& y) { return x.ratio > y.ratio; };
    const auto pop_candidate = [&]() {
      std::pop_heap(cand_heap_.begin(), cand_heap_.end(), heap_cmp);
      const Candidate& c = cands_[nc++] = cand_heap_.back();
      cand_heap_.pop_back();
      return c.ratio + dtol / c.abs_alpha;
    };
    while (!numerical_trouble) {
      size_t g_end;
      if (!heap_mode) {
        if (k >= nc) break;
        double theta_max = kInf;
        if (k == 0) {
          theta_max = theta0;
        } else {
          for (size_t t = k; t < nc; ++t) theta_max = std::min(theta_max, cands_[t].ratio + dtol / cands_[t].abs_alpha);
        }
        g_end = static_cast<size_t>(std::partition(cands_.begin() + k, cands_.begin() + nc,
                                                   [&](const Candidate& c) { return c.ratio <= theta_max; }) -
                                    cands_.begin());
        if (g_end == k) {
          numerical_trouble = true;  // NaN ratios: no candidate qualifies
          break;
        }
      } else {
        if (cand_heap_.empty()) break;
        double theta = pop_candidate();
        while (!cand_heap_.empty() && cand_heap_.front().ratio <= theta) theta = std::min(theta, pop_candidate());
        g_end = nc;
      }
      double best_alpha = -1.0, total = 0.0;
      Index gq = -1;
      for (size_t t = k; t < g_end; ++t) {
        const Candidate& c = cands_[t];
        if (c.abs_alpha > best_alpha) best_alpha = c.abs_alpha, gq = c.j;
        total += c.abs_alpha * (wub_[c.j] - wlb_[c.j]);
      }
      if (slope - total > ptol * ftol_[jl] && std::isfinite(total)) {
        slope -= total;
        k = g_end;
        flip_end = g_end;
        if (!heap_mode && k < nc) {
          for (size_t t = k; t < nc; ++t) numerical_trouble |= std::isnan(cands_[t].ratio);
          if (numerical_trouble) break;
          cand_heap_.assign(cands_.begin() + k, cands_.begin() + nc);
          nc = k;
          std::make_heap(cand_heap_.begin(), cand_heap_.end(), heap_cmp);
          heap_mode = true;
        }
        continue;
      }
      q = gq;
      break;
    }
    if (numerical_trouble || !std::isfinite(slope)) {
      // NaN/inf from an ill-conditioned eta file: retry from a fresh factorization.
      if (lu_.num_updates() > 0 && mismatch_retries < 3) {
        ++mismatch_retries;
        refactor();
        compute_primal();
        compute_duals();
        move_ok = false;
        continue;
      }
      return fail("non-finite ratio test");
    }
    if (q < 0) return infeasible(phase1);

    // ---- FTRAN: entering column, DSE tau = B^{-1} rho and bound flips, in one pass ----
    column(q, alpha_);
    double wr = 0.0;
    tau_.clear();
    for (Index e = 0; e < rho_.count; ++e) {
      const Index i = rho_.nz[e];
      const double v = rho_.dense[i];
      wr += v * v;
      tau_.dense[i] = v;
      tau_.nz[tau_.count++] = i;
    }
    if (flip_end > 0) {
      flipcol_.clear();
      std::vector<double>& fc = flipcol_.dense;
      Index* fnz = flipcol_.nz.data();
      Index fcnt = 0;
      // An entry that cancels to zero may be listed twice (the solve drops duplicates); if the list
      // would overflow, it is rebuilt from the values.
      for (size_t t = 0; t < flip_end; ++t) {
        const Index j = cands_[t].j;
        const double delta = status_[j] == VarStatus::AtLower ? wub_[j] - wlb_[j] : wlb_[j] - wub_[j];
        if (j < n_) {
          for (Index p = A_.begin(j); p < A_.end(j); ++p) {
            const Index i = A_.index[p];
            if (fc[i] == 0.0 && fcnt < m_) fnz[fcnt++] = i;
            fc[i] += A_.value[p] * delta;
          }
        } else {
          if (fc[j - n_] == 0.0 && fcnt < m_) fnz[fcnt++] = j - n_;
          fc[j - n_] += delta;
        }
      }
      flipcol_.count = fcnt;
      if (fcnt >= m_) flipcol_.rebuild_index(0.0);
      lu_.ftran(alpha_, tau_, flipcol_, dens_col_, dens_tau_, dens_flip_);
      note_density(dens_flip_, flipcol_);
    } else {
      lu_.ftran(alpha_, tau_, dens_col_, dens_tau_);
    }
    note_density(dens_col_, alpha_);
    note_density(dens_tau_, tau_);
    const double alpha_rq = alpha_.dense[r];
    if (std::abs(alpha_rq - arow_[q]) > 1e-6 * std::max(1.0, std::abs(alpha_rq)) ||
        std::abs(alpha_rq) < 1e-11) {
      if (lu_.num_updates() > 0 && mismatch_retries < 3) {
        ++mismatch_retries;
        refactor();
        compute_primal();
        compute_duals();
        move_ok = false;
        continue;
      }
      if (std::abs(alpha_rq) < 1e-11) return fail("tiny pivot");
    }
    mismatch_retries = 0;

    // ---- Bound flips ----
    if (flip_end > 0) {
      for (size_t t = 0; t < flip_end; ++t) {
        const Index j = cands_[t].j;
        if (status_[j] == VarStatus::AtLower) {
          status_[j] = VarStatus::AtUpper;
          x_[j] = wub_[j];
        } else {
          status_[j] = VarStatus::AtLower;
          x_[j] = wlb_[j];
        }
        move_[j] = move_of(j);
      }
      for (Index e = 0; e < flipcol_.count; ++e) {
        const Index p = flipcol_.nz[e];
        x_[basic_[p]] -= flipcol_.dense[p];
      }
    }

    // ---- Primal update ----
    const double theta_p = (x_[jl] - target) / alpha_rq;
    const std::vector<double>& alpha = alpha_.dense;
    for (Index e = 0; e < alpha_.count; ++e) {
      const Index p = alpha_.nz[e];
      x_[basic_[p]] -= theta_p * alpha[p];
    }
    x_[q] += theta_p;
    x_[jl] = target;

    // ---- Dual update ----
    double dq = d_[q];
    // Shift cost if d_q has the wrong sign (Harris may admit that).
    if ((status_[q] == VarStatus::AtLower && dq < 0) || (status_[q] == VarStatus::AtUpper && dq > 0) ||
        status_[q] == VarStatus::AtZero) {
      wcost_[q] -= dq;
      dq = 0.0;
      dbg_shifts_ += opt.verbosity >= 3;
    }
    const double theta_d = dq / alpha_rq;
    if (opt.verbosity >= 3) {
      dbg_degen_ += theta_d == 0.0;
      dbg_flips_ += static_cast<long long>(flip_end);
    }
    if (theta_d != 0.0) {
      for (Index t = 0; t < arow_count_; ++t) {
        const Index j = arow_nz_[t];
        if (status_[j] != VarStatus::Basic) d_[j] -= theta_d * arow_[j];
      }
    }
    d_[q] = 0.0;
    d_[jl] = -theta_d;

    // ---- DSE weight update ----
    for (Index e = 0; e < alpha_.count; ++e) {
      const Index p = alpha_.nz[e];
      if (p == r) continue;
      const double ratio = alpha[p] / alpha_rq;
      dse_[p] = std::max(dse_[p] - 2.0 * ratio * tau_.dense[p] + ratio * ratio * wr, 1e-6);
    }
    dse_[r] = std::max(wr / (alpha_rq * alpha_rq), 1e-6);

    // ---- Basis update ----
    lu_.update(r, alpha_);
    basic_[r] = q;
    status_[q] = VarStatus::Basic;
    status_[jl] = (to_lower || wlb_[jl] == wub_[jl]) ? VarStatus::AtLower : VarStatus::AtUpper;
    move_[q] = 0;
    move_[jl] = move_of(jl);
    if (!infeas_dirty_) {
      for (Index e = 0; e < alpha_.count; ++e) update_infeasible(alpha_.nz[e], ptol);
      if (flip_end > 0)
        for (Index e = 0; e < flipcol_.count; ++e) update_infeasible(flipcol_.nz[e], ptol);
      update_infeasible(r, ptol);
    }
    ++total_iterations_;
    if (opt.verbosity >= 3 && total_iterations_ % 100 == 0) {
      double obj = 0.0;
      for (Index j = 0; j < n_ + m_; ++j) obj += wcost_[j] * x_[j];
      std::printf("  dual%s it %lld obj %.10g (last 100: shifts %lld, zero steps %lld, flips %lld), infeasible %zu\n", phase1 ? "1" : "2", total_iterations_, obj, dbg_shifts_, dbg_degen_, dbg_flips_, infeas_list_.size());
      dbg_shifts_ = dbg_degen_ = dbg_flips_ = 0;
    }
  }
}

LpStatus DualSimplex::primal_loop(const LpOptions& opt) {
  const double ptol = opt.tol.primal_feas;
  const double dtol = opt.tol.dual_feas;
  for (;;) {
    LpStatus lim;
    if (limits_hit(opt, &lim)) return lim;
    if (lu_.num_updates() >= opt.refactor_interval || lu_.wants_refactor()) {
      refactor();
      compute_primal();
      compute_duals();
    }
    // Dantzig pricing.
    Index q = -1;
    double best = dtol;
    for (Index j = 0; j < n_ + m_; ++j) {
      const VarStatus st = status_[j];
      if (st == VarStatus::Basic || wlb_[j] == wub_[j]) continue;
      const double dj = d_[j];
      double infeas = 0.0;
      if (st == VarStatus::AtLower) infeas = -dj;
      else if (st == VarStatus::AtUpper) infeas = dj;
      else infeas = std::abs(dj);
      if (infeas > best) best = infeas, q = j;
    }
    if (q < 0) {
      dse_.assign(m_, 1.0);
      return LpStatus::Optimal;
    }
    const double dir = d_[q] < 0 ? 1.0 : -1.0;
    column(q, alpha_);
    lu_.ftran(alpha_, dens_col_);
    note_density(dens_col_, alpha_);
    const std::vector<double>& alpha = alpha_.dense;
    // Harris two-pass ratio test.
    double theta_max = kInf;
    for (Index k = 0; k < alpha_.count; ++k) {
      const Index p = alpha_.nz[k];
      const double a = dir * alpha[p];
      const Index j = basic_[p];
      if (a > 1e-9 && is_finite(wlb_[j])) theta_max = std::min(theta_max, (x_[j] - wlb_[j] + ptol * ftol_[j]) / a);
      else if (a < -1e-9 && is_finite(wub_[j])) theta_max = std::min(theta_max, (wub_[j] - x_[j] + ptol * ftol_[j]) / -a);
    }
    const double range = wub_[q] - wlb_[q];
    if (!is_finite(theta_max) && !is_finite(range)) return LpStatus::Unbounded;
    if (range <= theta_max) {
      // Entering variable just moves to its other bound.
      for (Index k = 0; k < alpha_.count; ++k) x_[basic_[alpha_.nz[k]]] -= dir * range * alpha[alpha_.nz[k]];
      if (status_[q] == VarStatus::AtLower) status_[q] = VarStatus::AtUpper, x_[q] = wub_[q];
      else status_[q] = VarStatus::AtLower, x_[q] = wlb_[q];
      ++total_iterations_;
      continue;
    }
    Index r = -1;
    double best_a = 0.0, theta = 0.0;
    bool leave_lower = true;
    for (Index k = 0; k < alpha_.count; ++k) {
      const Index p = alpha_.nz[k];
      const double a = dir * alpha[p];
      const Index j = basic_[p];
      double ratio;
      bool lower;
      if (a > 1e-9 && is_finite(wlb_[j])) ratio = (x_[j] - wlb_[j]) / a, lower = true;
      else if (a < -1e-9 && is_finite(wub_[j])) ratio = (wub_[j] - x_[j]) / -a, lower = false;
      else continue;
      if (ratio <= theta_max && std::abs(a) > best_a) best_a = std::abs(a), r = p, theta = ratio, leave_lower = lower;
    }
    if (r < 0) return fail("primal ratio test");
    theta = std::max(theta, 0.0);
    const Index jl = basic_[r];
    for (Index k = 0; k < alpha_.count; ++k) x_[basic_[alpha_.nz[k]]] -= dir * theta * alpha[alpha_.nz[k]];
    x_[q] += dir * theta;
    x_[jl] = leave_lower ? wlb_[jl] : wub_[jl];

    unit_btran(r);
    note_density(dens_rho_, rho_);
    pivot_row(rho_, true);
    const double alpha_rq = alpha[r];
    const double theta_d = d_[q] / alpha_rq;
    for (Index t = 0; t < arow_count_; ++t) {
      const Index j = arow_nz_[t];
      if (status_[j] != VarStatus::Basic) d_[j] -= theta_d * arow_[j];
    }
    d_[q] = 0.0;
    d_[jl] = -theta_d;
    lu_.update(r, alpha_);
    basic_[r] = q;
    status_[q] = VarStatus::Basic;
    status_[jl] = (leave_lower || wlb_[jl] == wub_[jl]) ? VarStatus::AtLower : VarStatus::AtUpper;
    ++total_iterations_;
  }
}

Index DualSimplex::pivot_out_rows(const std::vector<Index>& rows) {
  if (!has_factor_ && !refactor(false)) return 0;
  Index entered = 0;
  for (Index i : rows) {
    const Index q = n_ + i;
    if (i < 0 || i >= m_ || status_[q] == VarStatus::Basic) continue;
    if (lu_.num_updates() >= 50 || lu_.wants_refactor()) {
      if (!refactor(false)) break;
    }
    column(q, alpha_);
    lu_.ftran(alpha_, dens_col_);
    Index r = -1;
    double best = 1e-7;
    for (Index k = 0; k < alpha_.count; ++k) {
      const Index p = alpha_.nz[k];
      if (std::abs(alpha_.dense[p]) > best) best = std::abs(alpha_.dense[p]), r = p;
    }
    if (r < 0) continue;
    const Index jl = basic_[r];
    lu_.update(r, alpha_);
    basic_[r] = q;
    status_[q] = VarStatus::Basic;
    const double l = wlb_[jl], u = wub_[jl];
    if (!is_finite(l) && !is_finite(u)) status_[jl] = VarStatus::AtZero;
    else if (is_finite(u) && (!is_finite(l) || std::abs(x_[jl] - u) < std::abs(x_[jl] - l))) status_[jl] = VarStatus::AtUpper;
    else status_[jl] = VarStatus::AtLower;
    dse_[r] = 1.0;
    ++entered;
  }
  if (entered > 0) {
    values_ok_ = false;
    moves_.clear();
  }
  return entered;
}

void DualSimplex::add_rows(Index count, const std::vector<Index>& start, const std::vector<Index>& index,
                           const std::vector<double>& value, const std::vector<double>& lb,
                           const std::vector<double>& ub) {
  if (count == 0) return;
  // Rebuild CSR then CSC with the new rows appended (scaled internally).
  static const bool no_scale = std::getenv("MILP_NO_LP_SCALING") != nullptr;
  const Index m_base = m_;
  std::vector<double> scaled;
  for (Index r = 0; r < count; ++r) {
    const Index len = start[r + 1] - start[r];
    At_orig_.append(index.data() + start[r], value.data() + start[r], len);
    rscale_.push_back(1.0);
    if (!no_scale) compute_row_scale(m_base + r, index.data() + start[r], value.data() + start[r], len);
    scaled.assign(value.data() + start[r], value.data() + start[r] + len);
    for (double& v : scaled) v *= rscale_[m_base + r];
    At_.append(index.data() + start[r], scaled.data(), len);
  }
  At_.num_minor = n_;
  At_orig_.num_minor = n_;
  A_ = At_.transpose();
  A_orig_ = At_orig_.transpose();
  const Index m_old = m_;
  m_ += count;
  for (Index r = 0; r < count; ++r) {
    const Index j = n_ + m_old + r;
    const double rs = rscale_[m_old + r];
    cost_.push_back(0.0);
    lb_.push_back(-ub[r] * rs);
    ub_.push_back(-lb[r] * rs);
    wcost_.push_back(0.0);
    wlb_.push_back(-ub[r] * rs);
    wub_.push_back(-lb[r] * rs);
    ftol_.push_back(std::min(1.0, rs));
    status_.push_back(VarStatus::Basic);
    basic_.push_back(j);
    // Logical value: s = -a x.
    double act = 0.0;
    for (Index k = start[r]; k < start[r + 1]; ++k) act += value[k] * x_[index[k]];
    x_.push_back(-act * rs);
    d_.push_back(0.0);
    arow_.push_back(0.0);
    arow_mark_.push_back(0);
    dse_.push_back(1.0);
    dse_new_.push_back(m_old + r);
  }
  rho_.resize(m_);
  alpha_.resize(m_);
  tau_.resize(m_);
  flipcol_.resize(m_);
  has_factor_ = false;
  values_ok_ = false;
  moves_.clear();
}

void DualSimplex::remove_rows(const std::vector<Index>& rows) {
  if (rows.empty()) return;
  dse_new_.clear();
  std::vector<char> del(m_, 0);
  for (Index i : rows) del[i] = 1;
  bool basis_ok = true;
  for (Index i : rows) basis_ok = basis_ok && status_[n_ + i] == VarStatus::Basic;
  // Row-wise rebuild.
  SparseMatrix At, Ato;
  At.num_minor = n_;
  Ato.num_minor = n_;
  std::vector<Index> newpos(m_, -1);
  std::vector<double> rs;
  Index m_new = 0;
  for (Index i = 0; i < m_; ++i) {
    if (del[i]) continue;
    newpos[i] = m_new++;
    At.append(At_.index.data() + At_.begin(i), At_.value.data() + At_.begin(i), At_.end(i) - At_.begin(i));
    Ato.append(At_orig_.index.data() + At_orig_.begin(i), At_orig_.value.data() + At_orig_.begin(i),
               At_orig_.end(i) - At_orig_.begin(i));
    rs.push_back(rscale_[i]);
  }
  At_ = std::move(At);
  A_ = At_.transpose();
  At_orig_ = std::move(Ato);
  A_orig_ = At_orig_.transpose();
  rscale_ = std::move(rs);
  auto compact = [&](auto& v) {
    Index w = n_;
    for (Index i = 0; i < m_; ++i) {
      if (!del[i]) v[w++] = v[n_ + i];
    }
    v.resize(n_ + m_new);
  };
  compact(cost_);
  compact(lb_);
  compact(ub_);
  compact(wcost_);
  compact(wlb_);
  compact(wub_);
  compact(status_);
  compact(x_);
  compact(d_);
  compact(ftol_);
  arow_.assign(n_ + m_new, 0.0);
  arow_mark_.assign(n_ + m_new, 0);
  arow_count_ = 0;
  std::vector<Index> basic;
  std::vector<double> dse;
  for (Index p = 0; p < m_; ++p) {
    Index j = basic_[p];
    if (j >= n_) {
      if (del[j - n_]) continue;
      j = n_ + newpos[j - n_];
    }
    basic.push_back(j);
    dse.push_back(dse_[p]);
  }
  m_ = m_new;
  basic_ = std::move(basic);
  dse_ = std::move(dse);
  rho_.resize(m_);
  alpha_.resize(m_);
  tau_.resize(m_);
  flipcol_.resize(m_);
  if (!basis_ok || static_cast<Index>(basic_.size()) != m_) reset_basis();
  has_factor_ = false;
  values_ok_ = false;
  moves_.clear();
}

bool DualSimplex::tableau_row(Index basic_var, std::vector<double>& out) {
  if (status_[basic_var] != VarStatus::Basic) return false;
  if (!has_factor_) refactor(false);
  Index r = -1;
  for (Index p = 0; p < m_; ++p) {
    if (basic_[p] == basic_var) r = p;
  }
  if (r < 0) return false;
  unit_btran(r);
  pivot_row(rho_);
  out.assign(arow_.size(), 0.0);
  for (Index t = 0; t < arow_count_; ++t) out[arow_nz_[t]] = arow_[arow_nz_[t]];
  // Unscale: logical columns refer to scaled logicals s' = r s; a logical basic row is scaled by r_k.
  for (Index i = 0; i < m_; ++i) out[n_ + i] *= rscale_[i];
  if (basic_var >= n_) {
    const double inv = 1.0 / rscale_[basic_var - n_];
    for (double& v : out) v *= inv;
  }
  return true;
}

bool DualSimplex::tableau_row(Index basic_var, std::vector<double>& out, std::vector<Index>& nz) {
  if (status_[basic_var] != VarStatus::Basic) return false;
  if (!has_factor_) refactor(false);
  Index r = -1;
  for (Index p = 0; p < m_; ++p) {
    if (basic_[p] == basic_var) r = p;
  }
  if (r < 0) return false;
  unit_btran(r);
  pivot_row(rho_);
  out.resize(arow_.size(), 0.0);
  const Index total = static_cast<Index>(arow_.size());
  if (16 * arow_count_ > total) {
    nz.clear();  // dense row: a scan is cheaper than sorting the index list
    for (Index j = 0; j < total; ++j) {
      if (arow_mark_[j] == arow_gen_) nz.push_back(j);
    }
  } else {
    nz.assign(arow_nz_.begin(), arow_nz_.begin() + arow_count_);
    std::sort(nz.begin(), nz.end());
  }
  const double inv = basic_var >= n_ ? 1.0 / rscale_[basic_var - n_] : 1.0;
  for (Index j : nz) {
    double v = arow_[j];
    if (j >= n_) v *= rscale_[j - n_];
    if (basic_var >= n_) v *= inv;
    out[j] = v;
  }
  return true;
}

double DualSimplex::objective() const {
  double v = 0.0;
  for (Index j = 0; j < n_; ++j) v += cost_[j] * x_[j];
  return v;
}

std::vector<double> DualSimplex::col_values() const { return std::vector<double>(x_.begin(), x_.begin() + n_); }

std::vector<double> DualSimplex::row_activities() const {
  std::vector<double> a(m_);
  for (Index i = 0; i < m_; ++i) a[i] = -x_[n_ + i] / rscale_[i];
  return a;
}

std::vector<double> DualSimplex::row_duals() const {
  std::vector<double> y(m_);
  for (Index p = 0; p < m_; ++p) y[p] = cost_[basic_[p]];
  lu_.btran(y);
  for (Index i = 0; i < m_; ++i) y[i] *= rscale_[i];  // duals of the unscaled rows
  return y;
}

std::vector<double> DualSimplex::reduced_costs() const {
  std::vector<double> y = row_duals();
  std::vector<double> d(n_);
  for (Index j = 0; j < n_; ++j) {
    double dj = cost_[j];
    for (Index p = A_orig_.begin(j); p < A_orig_.end(j); ++p) dj -= A_orig_.value[p] * y[A_orig_.index[p]];
    d[j] = dj;
  }
  return d;
}

}  // namespace milp
