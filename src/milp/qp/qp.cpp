#include "milp/qp/qp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "milp/core/timer.hpp"
#include "milp/lp/barrier.hpp"
#include "milp/lp/cholesky.hpp"
#include "milp/lp/simplex.hpp"

namespace milp {

const char* to_string(QpStatus s) {
  switch (s) {
    case QpStatus::Optimal: return "optimal";
    case QpStatus::Infeasible: return "infeasible";
    case QpStatus::Unbounded: return "unbounded";
    case QpStatus::InfeasibleOrUnbounded: return "infeasible or unbounded";
    case QpStatus::TimeLimit: return "time limit";
    case QpStatus::NumericalError: return "numerical error";
  }
  return "?";
}

ConvexityResult check_convexity(const SparseMatrix& Q) {
  ConvexityResult r;
  const Index n = Q.num_major;
  std::vector<double> diag(static_cast<size_t>(n), 0.0);
  double maxabs = 0.0;
  bool offdiag = false;
  for (Index j = 0; j < n; ++j) {
    for (Index k = Q.begin(j); k < Q.end(j); ++k) {
      const double v = Q.value[k];
      maxabs = std::max(maxabs, std::abs(v));
      if (Q.index[k] == j) diag[j] += v;
      else if (v != 0.0) offdiag = true;
    }
  }
  if (maxabs == 0.0) return r;
  const double tiny = 1e-12 * maxabs;
  for (Index j = 0; j < n; ++j) {
    if (diag[j] < -tiny) {
      r.convex = false;
      r.message = "the quadratic objective is not convex: the diagonal entry of column " + std::to_string(j) +
                  " has the wrong sign (" + std::to_string(diag[j]) + ")";
      return r;
    }
    if (diag[j] < 0.0) {
      // Round-off: shift to zero.
      if (r.diag_shift.empty()) r.diag_shift.assign(static_cast<size_t>(n), 0.0);
      r.diag_shift[j] = -diag[j];
      r.perturbed = true;
      diag[j] = 0.0;
    }
  }
  if (!offdiag) {
    if (r.perturbed) r.message = "tiny negative diagonal entries of Q were set to zero";
    return r;
  }
  // Unit-diagonal scaling over the columns with a positive diagonal; a column with a zero
  // diagonal must have no (significant) off-diagonal entries.
  std::vector<Index> cmap(static_cast<size_t>(n), -1);
  std::vector<double> s(static_cast<size_t>(n), 0.0);
  Index nc = 0;
  for (Index j = 0; j < n; ++j) {
    if (diag[j] > 0.0) {
      cmap[j] = nc++;
      s[j] = 1.0 / std::sqrt(diag[j]);
    }
  }
  std::vector<Index> J, I;
  std::vector<double> V;
  for (Index j = 0; j < n; ++j) {
    for (Index k = Q.begin(j); k < Q.end(j); ++k) {
      const Index i = Q.index[k];
      const double v = Q.value[k];
      if (i == j || v == 0.0) continue;
      if (cmap[i] < 0 || cmap[j] < 0) {
        if (std::abs(v) <= tiny) continue;
        if (std::abs(v) <= 1e-6 * maxabs) {
          // A tiny cross term on a column without a square term (QPLIB_8515: 1.5625e-8 x2 x3, no x3^2; an eigenvalue
          // of -1.5e-17): |v| on both diagonals makes [|v| v; v |v|] PSD, the rest is tested without the term.
          if (r.diag_shift.empty()) r.diag_shift.assign(static_cast<size_t>(n), 0.0);
          r.diag_shift[i] += std::abs(v);
          r.diag_shift[j] += std::abs(v);
          r.perturbed = true;
          r.message = "Q is positive semidefinite only up to a relative 1e-6; its diagonal was increased by that much";
          continue;
        }
        r.convex = false;
        r.message = "the quadratic objective is not convex: columns " + std::to_string(std::min(i, j)) + " and " +
                    std::to_string(std::max(i, j)) + " have a cross term but one of them has no square term";
        return r;
      }
      J.push_back(cmap[j]);
      I.push_back(cmap[i]);
      V.push_back(v * s[i] * s[j]);
    }
  }
  // Lower triangle with the diagonal first in each column.
  for (Index c = 0; c < nc; ++c) J.push_back(c), I.push_back(c), V.push_back(0.0);
  std::vector<Index> start(static_cast<size_t>(nc) + 1, 0), index;
  std::vector<double> value;
  {
    std::vector<Index> cnt(static_cast<size_t>(nc) + 1, 0);
    for (size_t t = 0; t < J.size(); ++t) {
      const Index lo = std::min(J[t], I[t]);
      ++cnt[lo + 1];
    }
    for (Index c = 0; c < nc; ++c) cnt[c + 1] += cnt[c];
    start = cnt;
    index.assign(J.size(), 0);
    value.assign(J.size(), 0.0);
    std::vector<Index> pos(start.begin(), start.end() - 1);
    // Diagonal entries first (they were appended last; place them in a first pass).
    for (size_t t = 0; t < J.size(); ++t) {
      if (J[t] != I[t]) continue;
      const Index p = pos[J[t]]++;
      index[p] = I[t];
      value[p] = 0.0;
    }
    for (size_t t = 0; t < J.size(); ++t) {
      if (J[t] == I[t]) continue;
      const Index lo = std::min(J[t], I[t]), hi = std::max(J[t], I[t]);
      const Index p = pos[lo]++;
      index[p] = hi;
      value[p] = V[t];
    }
  }
  // Duplicate (i, j) pairs cannot occur: Q is a CSC matrix without duplicates.
  SparseCholesky chol;
  chol.pivot_tol = 0.0;
  if (!chol.analyze(nc, start, index)) {
    r.convex = false;
    r.message = "the convexity test of the quadratic objective failed";
    return r;
  }
  for (const double shift : {1e-10, 1e-6}) {
    for (Index c = 0; c < nc; ++c) value[start[c]] = 1.0 + shift;
    if (!chol.factorize(value)) break;
    if (chol.num_wrong_sign() == 0) {
      if (shift > 1e-9) {
        if (r.diag_shift.empty()) r.diag_shift.assign(static_cast<size_t>(n), 0.0);
        for (Index j = 0; j < n; ++j)
          if (cmap[j] >= 0) r.diag_shift[j] += shift * diag[j];
        r.perturbed = true;
        r.message = "Q is positive semidefinite only up to a relative 1e-6; its diagonal was increased by that much";
      }
      return r;
    }
  }
  r.convex = false;
  r.message = "the quadratic objective is not convex (Q has a negative eigenvalue)";
  return r;
}

namespace {

// Row activities of x.
std::vector<double> activities(const MipProblem& p, const std::vector<double>& x) {
  std::vector<double> act(static_cast<size_t>(p.num_rows()), 0.0);
  for (Index j = 0; j < p.num_cols(); ++j)
    for (Index k = p.A.begin(j); k < p.A.end(j); ++k) act[p.A.index[k]] += p.A.value[k] * x[j];
  return act;
}

double row_violation(const MipProblem& p, const std::vector<double>& act) {
  double viol = 0.0;
  for (Index i = 0; i < p.num_rows(); ++i) viol = std::max({viol, p.row_lb[i] - act[i], act[i] - p.row_ub[i]});
  return viol;
}

// The largest row violation relative to its row's own side, max(1, |side|): the measure of the MIP search's
// solution check and of the benchmark harness.
double row_violation_rel(const MipProblem& p, const std::vector<double>& act) {
  double viol = 0.0;
  for (Index i = 0; i < p.num_rows(); ++i) {
    if (act[i] < p.row_lb[i]) viol = std::max(viol, (p.row_lb[i] - act[i]) / std::max(1.0, std::abs(p.row_lb[i])));
    if (act[i] > p.row_ub[i]) viol = std::max(viol, (act[i] - p.row_ub[i]) / std::max(1.0, std::abs(p.row_ub[i])));
  }
  return viol;
}

// Feasibility polish of an interior-point result that violates rows (by more than tol): a projection onto the rows
// that are violated or active (equalities, and rows within 1e-9 relative of a side): dx = A_V' (A_V A_V')^-1 r_V with
// r_V the distance to the side (0 for an active row that holds), over the columns not clipped at a bound in an earlier
// round, then clipped to the bounds; a few rounds, kept only while the violation falls. Without the active rows the
// correction breaks them (QPLIB_10034: 2.2e-6 -> 7e-3). The objective moves by about grad' dx, small next to the
// point's own error (QPLIB_9002: the barrier's point violated a row by 5e-5 at objective 5.7e9).
void polish_rows(const MipProblem& p, const std::vector<double>& lb, const std::vector<double>& ub, std::vector<double>& x,
                 double tol) {
  const Index n = p.num_cols(), m = p.num_rows();
  std::vector<double> act = activities(p, x);
  double viol = row_violation(p, act);
  if (!(viol > tol)) return;
  const bool debug = std::getenv("MILP_POLISH_DEBUG") != nullptr;
  std::vector<char> clipped(static_cast<size_t>(n), 0);
  for (int round = 0; round < 6 && viol > tol; ++round) {
    std::vector<Index> V, pos(static_cast<size_t>(m), -1);
    std::vector<double> r;
    for (Index i = 0; i < m; ++i) {
      const double lo = p.row_lb[i], hi = p.row_ub[i];
      const double near_lo = 1e-9 * (1.0 + std::abs(lo)), near_hi = 1e-9 * (1.0 + std::abs(hi));
      if (is_finite(lo) && act[i] < lo + near_lo) pos[i] = static_cast<Index>(V.size()), V.push_back(i), r.push_back(lo - act[i]);
      else if (is_finite(hi) && act[i] > hi - near_hi) pos[i] = static_cast<Index>(V.size()), V.push_back(i), r.push_back(hi - act[i]);
    }
    if (V.empty()) return;
    // M = A_V A_V' over the movable columns (lower triangle triplets, by columns of A).
    const Index v = static_cast<Index>(V.size());
    std::vector<Index> J, I;
    std::vector<double> Vv;
    double work = 0.0;
    std::vector<std::pair<Index, double>> colv;
    for (Index j = 0; j < n; ++j) {
      if (clipped[j] || !(lb[j] < ub[j])) continue;
      colv.clear();
      for (Index k = p.A.begin(j); k < p.A.end(j); ++k) {
        if (pos[p.A.index[k]] >= 0) colv.emplace_back(pos[p.A.index[k]], p.A.value[k]);
      }
      work += static_cast<double>(colv.size()) * colv.size();
      if (work > 5e7) return;
      for (size_t a = 0; a < colv.size(); ++a) {
        for (size_t b = 0; b <= a; ++b) {
          const Index ra = std::max(colv[a].first, colv[b].first), rb = std::min(colv[a].first, colv[b].first);
          J.push_back(rb), I.push_back(ra), Vv.push_back(colv[a].second * colv[b].second);
        }
      }
    }
    double dmax = 0.0;
    for (size_t t = 0; t < Vv.size(); ++t) {
      if (I[t] == J[t]) dmax = std::max(dmax, Vv[t]);
    }
    if (!(dmax > 0.0)) return;
    for (Index a = 0; a < v; ++a) J.push_back(a), I.push_back(a), Vv.push_back(1e-12 * dmax);
    const SparseMatrix M = SparseMatrix::from_triplets(v, v, J, I, Vv);  // CSC, minor >= major: lower triangle
    SparseCholesky chol;
    if (!chol.analyze(v, M.start, M.index) || !chol.factorize(M.value)) return;
    std::vector<double> w = r;
    chol.solve(w);
    std::vector<double> y = x;
    for (Index j = 0; j < n; ++j) {
      if (clipped[j] || !(lb[j] < ub[j])) continue;
      double t = 0.0;
      for (Index k = p.A.begin(j); k < p.A.end(j); ++k) {
        if (pos[p.A.index[k]] >= 0) t += p.A.value[k] * w[pos[p.A.index[k]]];
      }
      y[j] += t;
    }
    for (Index j = 0; j < n; ++j) {
      const double c = std::min(std::max(y[j], lb[j]), ub[j]);
      if (c != y[j]) clipped[j] = 1;
      y[j] = c;
    }
    std::vector<double> act2 = activities(p, y);
    const double viol2 = row_violation(p, act2);
    if (debug) std::fprintf(stderr, "polish round %d: %d rows, violation %.3g -> %.3g\n", round, static_cast<int>(v), viol, viol2);
    if (!(viol2 < viol)) continue;  // the clipped columns are out next round
    x.swap(y), act.swap(act2), viol = viol2;
  }
}

// Feasibility of the rows and bounds by the dual simplex: 1 feasible, 0 infeasible, -1 unknown.
int lp_feasible(const MipProblem& p, const std::vector<double>& lb, const std::vector<double>& ub, double time_limit) {
  DualSimplex lp;
  const std::vector<double> zero(static_cast<size_t>(p.num_cols()), 0.0);
  lp.load(p.A, zero, lb, ub, p.row_lb, p.row_ub);
  LpOptions o;
  o.time_limit = std::max(1.0, time_limit);
  const LpStatus st = lp.solve(o);
  if (st == LpStatus::Optimal) return 1;
  if (st == LpStatus::Infeasible) return 0;
  return -1;
}

// Is there a recession direction d (A d within the recession cone of the rows, d of the bounds,
// |d| <= 1) with Q d = 0 and c^T d < 0? Then a feasible convex QP is unbounded (and only then).
// 1 yes, 0 no, -1 unknown.
int has_descent_ray(const MipProblem& p, const SparseMatrix& Q, const std::vector<double>& lb,
                    const std::vector<double>& ub, double time_limit) {
  const Index n = p.num_cols(), m = p.num_rows();
  std::vector<double> dl(static_cast<size_t>(n)), du(static_cast<size_t>(n));
  for (Index j = 0; j < n; ++j) {
    dl[j] = is_finite(lb[j]) ? 0.0 : -1.0;
    du[j] = is_finite(ub[j]) ? 0.0 : 1.0;
  }
  std::vector<Index> J, I;
  std::vector<double> V;
  for (Index j = 0; j < n; ++j)
    for (Index k = p.A.begin(j); k < p.A.end(j); ++k) J.push_back(j), I.push_back(p.A.index[k]), V.push_back(p.A.value[k]);
  std::vector<double> rl(static_cast<size_t>(m)), ru(static_cast<size_t>(m));
  for (Index i = 0; i < m; ++i) {
    rl[i] = is_finite(p.row_lb[i]) ? 0.0 : -kInf;
    ru[i] = is_finite(p.row_ub[i]) ? 0.0 : kInf;
  }
  // Rows Q d = 0 (symmetric Q from its lower triangle), one per column with Hessian entries.
  std::vector<Index> qrow(static_cast<size_t>(n), -1);
  Index rows = m;
  for (Index j = 0; j < Q.num_major; ++j) {
    for (Index k = Q.begin(j); k < Q.end(j); ++k) {
      const Index i = Q.index[k];
      if (qrow[i] < 0) qrow[i] = rows++;
      if (qrow[j] < 0) qrow[j] = rows++;
      J.push_back(j), I.push_back(qrow[i]), V.push_back(Q.value[k]);
      if (i != j) J.push_back(i), I.push_back(qrow[j]), V.push_back(Q.value[k]);
    }
  }
  rl.resize(static_cast<size_t>(rows), 0.0);
  ru.resize(static_cast<size_t>(rows), 0.0);
  const SparseMatrix M = SparseMatrix::from_triplets(n, rows, J, I, V);
  DualSimplex lp;
  lp.load(M, p.cost, dl, du, rl, ru);
  LpOptions o;
  o.time_limit = std::max(1.0, time_limit);
  const LpStatus st = lp.solve(o);
  if (st != LpStatus::Optimal) return -1;
  double cmax = 0.0;
  for (double c : p.cost) cmax = std::max(cmax, std::abs(c));
  return lp.objective() < -1e-7 * std::max(1.0, cmax) ? 1 : 0;
}

}  // namespace

std::vector<double> qp_reduced_costs(const MipProblem& p, const std::vector<double>& x, const std::vector<double>& y) {
  std::vector<double> d = p.cost;
  for (Index j = 0; j < p.Q.num_major; ++j) {
    for (Index k = p.Q.begin(j); k < p.Q.end(j); ++k) {
      const Index i = p.Q.index[k];
      d[i] += p.Q.value[k] * x[j];
      if (i != j) d[j] += p.Q.value[k] * x[i];
    }
  }
  for (Index j = 0; j < p.num_cols(); ++j)
    for (Index k = p.A.begin(j); k < p.A.end(j); ++k) d[j] -= p.A.value[k] * y[p.A.index[k]];
  return d;
}

QpSolution solve_qp(const MipProblem& p, const std::vector<double>& lb, const std::vector<double>& ub,
                    const QpOptions& opt, const std::vector<double>& diag_shift) {
  Timer timer;
  QpSolution sol;
  const Index n = p.num_cols();
  SparseMatrix Q = p.Q;
  if (Q.num_major != n) Q = SparseMatrix::from_triplets(n, n, {}, {}, {});
  if (!diag_shift.empty()) {
    std::vector<Index> J, I;
    std::vector<double> V;
    for (Index j = 0; j < n; ++j) {
      for (Index k = Q.begin(j); k < Q.end(j); ++k) J.push_back(j), I.push_back(Q.index[k]), V.push_back(Q.value[k]);
      if (diag_shift[j] != 0.0) J.push_back(j), I.push_back(j), V.push_back(diag_shift[j]);
    }
    Q = SparseMatrix::from_triplets(n, n, J, I, V);
  }
  int feasible = -1;  // rows and bounds: 1 feasible, 0 infeasible (simplex), -1 not known
  auto remaining = [&] { return opt.time_limit - timer.elapsed(); };
  // Attempts: default settings, then without scaling on the augmented system. An optimal point that violates the rows
  // a little is kept for the case that the second attempt does no better.
  IpmResult kept;
  bool have_kept = false;
  for (int attempt = 0; attempt < 2; ++attempt) {
    IpmOptions io;
    io.tol = opt.tol;
    io.max_iterations = opt.max_iterations;
    io.time_limit = remaining();
    io.threads = opt.threads;
    io.log = opt.log;
    io.stop = opt.stop;
    io.log_offset = p.offset;
    io.log_sense = p.display_sense;
    io.gap_with_offset = true;
    if (attempt == 1) {
      io.scale = false;
      io.kkt = 2;
    }
    IpmResult r = ipm_solve(p.A, p.cost, lb, ub, p.row_lb, p.row_ub, io, &Q);
    sol.iterations += r.iterations;
    bool finite = r.status == IpmStatus::Optimal;
    for (double v : r.x) finite = finite && std::isfinite(v);
    if (r.status == IpmStatus::Optimal && !finite) r.status = IpmStatus::NumericalError;
    if (attempt == 1 && r.status != IpmStatus::Optimal && have_kept && r.status != IpmStatus::TimeLimit) {
      if (opt.log) opt.log->log(2, "QP: the second attempt ended with '%s': the first point\n", to_string(r.status));
      r = std::move(kept);
      r.approximate = true;
    }
    // A point that violates a row of the original problem (the barrier's measures are scaled) is not accepted;
    // the next attempt runs without scaling. Judged per row, by more than 1e-6 relative to the row's own side
    // (until v0.9: relative to the largest side of all rows, which let a row with a small side be violated
    // by 1e-6 x that largest side; QPLIB_9002 has sides up to 5e9).
    auto rel_violation = [&](const std::vector<double>& x) {
      std::vector<double> xc = x;
      for (Index j = 0; j < n; ++j) xc[j] = std::min(std::max(xc[j], lb[j]), ub[j]);
      return row_violation_rel(p, activities(p, xc));
    };
    const bool keep_first = attempt == 1 && r.approximate && have_kept;  // the first attempt's point, taken back above
    double rv = r.status == IpmStatus::Optimal && !keep_first ? rel_violation(r.x) : 0.0;
    if (rv > 1e-6) {
      // The projection onto the active rows often repairs it (then no retry is needed).
      std::vector<double> xp = r.x;
      for (Index j = 0; j < n; ++j) xp[j] = std::min(std::max(xp[j], lb[j]), ub[j]);
      polish_rows(p, lb, ub, xp, 1e-7);
      const std::vector<double> act = activities(p, xp);
      const double pv = row_violation_rel(p, act);
      if (pv <= 1e-6) {
        if (opt.log) opt.log->log(2, "QP: barrier point violated the rows by %.3g (relative), %.3g after the projection\n", rv, pv);
        r.x = std::move(xp);
        r.primal_infeas = row_violation(p, act);
        rv = pv;
      }
    }
    if (rv > 1e-6) {
      if (opt.log) opt.log->log(2, "QP: barrier point violates the rows by %.3g (relative); retrying\n", rv);
      if (attempt == 0 && rv <= 1e-4) kept = r, have_kept = true;
      continue;
    }
    if (r.status == IpmStatus::Optimal) {
      sol.x = r.x;
      sol.y = r.y;
      for (Index j = 0; j < n; ++j) sol.x[j] = std::min(std::max(sol.x[j], lb[j]), ub[j]);
      polish_rows(p, lb, ub, sol.x, 1e-7);
      sol.objective = p.objective(sol.x);
      sol.dual_bound = sol.objective - r.rel_gap * (1.0 + std::abs(sol.objective - p.offset));
      sol.approximate = r.approximate;
      double viol = 0.0;
      std::vector<double> act(static_cast<size_t>(p.num_rows()), 0.0);
      for (Index j = 0; j < n; ++j)
        for (Index k = p.A.begin(j); k < p.A.end(j); ++k) act[p.A.index[k]] += p.A.value[k] * sol.x[j];
      for (Index i = 0; i < p.num_rows(); ++i) viol = std::max({viol, p.row_lb[i] - act[i], act[i] - p.row_ub[i]});
      sol.primal_infeas = viol;
      sol.status = QpStatus::Optimal;
      return sol;
    }
    if (r.status == IpmStatus::TimeLimit || remaining() <= 0.0 || (opt.stop && opt.stop->load())) {
      sol.status = QpStatus::TimeLimit;
      return sol;
    }
    if (r.status == IpmStatus::PrimalInfeasible || r.status == IpmStatus::DualInfeasible) {
      if (!opt.confirm) {
        sol.status = r.status == IpmStatus::PrimalInfeasible ? QpStatus::Infeasible : QpStatus::InfeasibleOrUnbounded;
        return sol;
      }
      if (feasible < 0) feasible = lp_feasible(p, lb, ub, remaining());
      if (feasible == 0) {
        sol.status = QpStatus::Infeasible;
        return sol;
      }
      if (r.status == IpmStatus::DualInfeasible && feasible == 1) {
        const int ray = has_descent_ray(p, Q, lb, ub, remaining());
        if (ray == 1) {
          sol.status = QpStatus::Unbounded;
          return sol;
        }
      }
    }
    if (opt.log) opt.log->log(2, "QP: barrier attempt %d ended with '%s'\n", attempt + 1, to_string(r.status));
  }
  // Both attempts failed: an infeasible or unbounded problem often ends this way (the barrier's
  // divergence tests did not trigger); the simplex decides.
  if (opt.confirm && feasible < 0 && remaining() > 0.0) feasible = lp_feasible(p, lb, ub, remaining());
  const int ray = feasible == 1 && opt.confirm && remaining() > 0.0 ? has_descent_ray(p, Q, lb, ub, remaining()) : -1;
  if (opt.log) opt.log->log(2, "QP: simplex checks: feasible %d, descent ray %d\n", feasible, ray);
  sol.status = feasible == 0 ? QpStatus::Infeasible : ray == 1 ? QpStatus::Unbounded : QpStatus::NumericalError;
  return sol;
}

}  // namespace milp
