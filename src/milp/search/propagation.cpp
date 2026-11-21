#include "milp/search/propagation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace milp {

namespace {
constexpr double kFeasTol = 1e-6;
constexpr double kBigBound = 1e12;  // larger bounds are treated as infinite
}  // namespace

namespace {
// The costs as a row: indices and values of the nonzero ones.
void objective_row(const MipProblem& p, std::vector<Index>& idx, std::vector<double>& val) {
  for (Index j = 0; j < p.num_cols(); ++j) {
    if (p.cost[j] != 0.0) idx.push_back(j), val.push_back(p.cost[j]);
  }
}
}  // namespace

DomainPropagator::DomainPropagator(const MipProblem& p, std::shared_ptr<const SparseMatrix> rows)
    : p_(p), rows_(rows ? std::move(rows) : std::make_shared<const SparseMatrix>(p.A.transpose())), row_lb_(p.row_lb),
      row_ub_(p.row_ub) {
  queued_.assign(p.num_rows(), 0);
}

std::shared_ptr<const SparseMatrix> DomainPropagator::shared_rows(const MipProblem& p) {
  auto rows = std::make_shared<SparseMatrix>(p.A.transpose());
  std::vector<Index> idx;
  std::vector<double> val;
  objective_row(p, idx, val);
  if (!idx.empty()) rows->append(idx.data(), val.data(), static_cast<Index>(idx.size()));
  return rows;
}

void DomainPropagator::enable_objective() {
  if (obj_row_ >= 0) return;
  if (rows_->num_major == p_.num_rows() + 1) {
    // shared_rows: the objective row is there already.
    obj_row_ = p_.num_rows();
  } else {
    std::vector<Index> idx;
    std::vector<double> val;
    objective_row(p_, idx, val);
    if (idx.empty()) return;
    auto rows = std::make_shared<SparseMatrix>(*rows_);
    obj_row_ = rows->num_major;
    rows->append(idx.data(), val.data(), static_cast<Index>(idx.size()));
    rows_ = std::move(rows);
  }
  row_lb_.push_back(-kInf);
  row_ub_.push_back(kInf);
  queued_.push_back(0);
}

void DomainPropagator::set_objective_cutoff(double cutoff) {
  if (obj_row_ >= 0) row_ub_[obj_row_] = cutoff;
}

bool DomainPropagator::propagate(std::vector<double>& lb, std::vector<double>& ub, const std::vector<Index>& seeds,
                                 std::vector<Index>& tightened) {
  const SparseMatrix& rows = *rows_;
  static const double wf = [] { const char* v = std::getenv("MILP_PROP_WORK"); return v ? std::atof(v) : 20.0; }();
  const long long limit = work_limit > 0 ? work_limit : static_cast<long long>(wf * p_.A.nnz()) + 1000;
  long long work = 0;
  queue_.clear();
  // The objective row is dense on many models: processed at most once per call.
  bool obj_seen = false;
  auto enqueue_col = [&](Index j) {
    for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
      const Index i = p_.A.index[k];
      if (!queued_[i]) queued_[i] = 1, queue_.push_back(i);
    }
    if (obj_row_ >= 0 && objective && !obj_seen && p_.cost[j] != 0.0 && is_finite(row_ub_[obj_row_]) && !queued_[obj_row_]) {
      queued_[obj_row_] = 1, queue_.push_back(obj_row_);
    }
  };
  for (Index j : seeds) enqueue_col(j);
  // Rows found idle in this call (see the skip below) keep how much room they have left, per side, before a bound
  // could be tightened; a tightened column takes its share from the rooms of its rows. Valid within one call only
  // (the caller's bounds change freely between calls): a stamp per row.
  if (room_stamp_.size() != queued_.size()) {
    room_stamp_.assign(queued_.size(), 0);
    room_hi_.assign(queued_.size(), 0.0);
    room_lo_.assign(queued_.size(), 0.0);
    room_call_ = 0;
  }
  if (++room_call_ == 0) {
    std::fill(room_stamp_.begin(), room_stamp_.end(), 0u);
    room_call_ = 1;
  }
  const uint32_t call = room_call_;
  // Column j went from [l0, u0] to its current bounds: its rows are queued, and the rooms of the idle ones shrink
  // by the change of their activities (rounded up; a bound that was infinite invalidates the room).
  auto tightened_col = [&](Index j, double l0, double u0) {
    const bool finite = std::abs(l0) <= kBigBound && std::abs(u0) <= kBigBound;
    const double dl = lb[j] - l0, du = u0 - ub[j];  // both >= 0
    auto take = [&](Index i, double a) {
      if (room_stamp_[i] != call) return;
      if (!finite) {
        room_stamp_[i] = 0;
        return;
      }
      const double xmin = a > 0 ? a * dl : -a * du, xmax = a > 0 ? a * du : -a * dl;
      room_hi_[i] -= xmin * (1.0 + 1e-9) + xmax * 1e-9;
      room_lo_[i] -= xmax * (1.0 + 1e-9) + xmin * 1e-9;
    };
    for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) {
      const Index i = p_.A.index[k];
      take(i, p_.A.value[k]);
      if (!queued_[i]) queued_[i] = 1, queue_.push_back(i);
    }
    if (obj_row_ >= 0 && p_.cost[j] != 0.0) {
      take(obj_row_, p_.cost[j]);
      if (objective && !obj_seen && is_finite(row_ub_[obj_row_]) && !queued_[obj_row_]) {
        queued_[obj_row_] = 1, queue_.push_back(obj_row_);
      }
    }
  };
  last_cont_useful = 0;
  if (continuous && cont_mark_.size() != static_cast<size_t>(p_.num_cols())) cont_mark_.assign(p_.num_cols(), 0);

  bool feasible = true;
  for (size_t head = 0; head < queue_.size() && feasible; ++head) {
    const Index i = queue_[head];
    queued_[i] = 0;
    if (i == obj_row_) obj_seen = true;
    if (work > limit) continue;  // drain the queue flags
    const Index rb = rows.begin(i), re = rows.end(i);
    work += re - rb;
    // Idle at an earlier visit of this call, and the tightenings since left room on both sides: the scan below
    // would find it idle again (and so not infeasible). The work is charged as if scanned.
    const bool known_idle = room_stamp_[i] == call && room_hi_[i] >= 0.0 && room_lo_[i] >= 0.0;
#ifndef MILP_PROP_CHECK
    if (known_idle) continue;
#endif
    if (!known_idle) room_stamp_[i] = 0;
    // Activity bounds with counts of infinite contributions.
    double minact = 0.0, maxact = 0.0;
    int mininf = 0, maxinf = 0;
    // With a trail: the two highest decision levels among the bounds entering
    // the minimal (side 0) and maximal (side 1) activity, and the column of the highest.
    int lv1[2] = {0, 0}, lv2[2] = {0, 0};
    Index lvj[2] = {-1, -1};
    bool uses_cont = false;  // a continuous bound tightened in this call enters the activities
    // Largest |a_j| (u_j - l_j) over the row's columns (infinite with a one-sided column, whose missing bound
    // the row may supply): no bound of the row can be tightened from a side whose slack is at least that (see
    // the skip below).
    double maxspan = 0.0;
    double asum = 0.0;  // sum of |a l| + |a u| over the finite bounds: scale of the rounding in the activities
    auto note_level = [&](int side, Index j, int lv) {
      if (lv > lv1[side]) lv2[side] = lv1[side], lv1[side] = lv, lvj[side] = j;
      else if (lv > lv2[side]) lv2[side] = lv;
    };
    for (Index k = rb; k < re; ++k) {
      const Index j = rows.index[k];
      const double a = rows.value[k];
      const double lj = lb[j], uj = ub[j];
      if (std::abs(lj) <= kBigBound && std::abs(uj) <= kBigBound) {
        // Both bounds finite (nearly every entry): the same sums as below without the tests.
        const double al = a * lj, au = a * uj;
        if (a > 0) minact += al, maxact += au;
        else minact += au, maxact += al;
        maxspan = std::max(maxspan, std::abs(a) * (uj - lj));
        asum += std::abs(al) + std::abs(au);
      } else {
        const double l = lj < -kBigBound ? -kInf : lj, u = uj > kBigBound ? kInf : uj;
        if (a > 0) {
          if (is_finite(l)) minact += a * l; else ++mininf;
          if (is_finite(u)) maxact += a * u; else ++maxinf;
        } else {
          if (is_finite(u)) minact += a * u; else ++mininf;
          if (is_finite(l)) maxact += a * l; else ++maxinf;
        }
        maxspan = is_finite(l) && is_finite(u) ? std::max(maxspan, std::abs(a) * (u - l)) : kInf;
        if (is_finite(l)) asum += std::abs(a * l);
        if (is_finite(u)) asum += std::abs(a * u);
      }
      if (continuous && cont_mark_[j]) uses_cont = true;
    }
    // The levels are needed only once the row tightens a bound: computed then, before the first push of this
    // visit (so from the same bounds as the activities above).
    bool levels_ready = false;
    auto ensure_levels = [&] {
      if (levels_ready) return;
      levels_ready = true;
      for (Index k = rb; k < re; ++k) {
        const Index j = rows.index[k];
        const double a = rows.value[k];
        note_level(0, j, trail->bound_level(j, a < 0));
        note_level(1, j, trail->bound_level(j, a > 0));
      }
    };
    // Level of a bound of column j implied by this row: from the minimal
    // activity (row_ub side) or the maximal one (row_lb side) of the others.
    auto implied_level = [&](Index j, int side) { return lvj[side] == j ? lv2[side] : lv1[side]; };
    const double rlo = row_lb_[i], rhi = row_ub_[i];
    const double scale = 1.0 + std::max(std::abs(minact), std::abs(maxact));
    if ((mininf == 0 && minact > rhi + kFeasTol * scale) || (maxinf == 0 && maxact < rlo - kFeasTol * scale)) {
      if (known_idle) std::abort();  // MILP_PROP_CHECK only: the skip would have missed this
      if (trail) trail->fail_row_at(i, mininf == 0 && minact > rhi + kFeasTol * scale);
      if (uses_cont) ++last_cont_useful;
      if (i == obj_row_) last_obj_useful = true;
      feasible = false;
      break;
    }
    // The row tightens nothing when, on each side, it is infinite, two or more columns are unbounded towards
    // it (no residual activity is finite), or all are bounded and the side's slack covers the largest span: the
    // loop below would then find new_ub >= u and new_lb <= l for every column (its tolerances only widen the
    // implied bounds). Most visits of a dense row end here (mas74: 13 rows of 150 entries, the propagator took
    // 30% of the tree's time).
    const bool hi_idle = !is_finite(rhi) || mininf >= 2 || (mininf == 0 && maxspan <= rhi - minact);
    const bool lo_idle = !is_finite(rlo) || maxinf >= 2 || (maxinf == 0 && maxspan <= maxact - rlo);
    if (known_idle && !(hi_idle && lo_idle)) std::abort();  // MILP_PROP_CHECK only
    if (known_idle) continue;                               // MILP_PROP_CHECK only: the room stays as it was
    if (hi_idle && lo_idle) {
      // Room per side: unlimited where the side is idle whatever finite bounds do (infinite side, two or more
      // unbounded columns: only a bound that was infinite changes that, which invalidates the room), else the
      // slack beyond the largest span, less a bound on the rounding of this scan and of a later one (n terms).
      // (Rows of 500000 entries and more are not kept: the shares taken later assume n * 1e-15 <= 1e-9.)
      if (re - rb >= 500000) continue;
      const double eps = (static_cast<double>(re - rb) + 8.0) * 1e-15;
      auto room = [&](bool fixed, double slack, double rhs) {
        if (fixed) return kInf;
        const double r = slack - maxspan;
        return r - eps * (asum + std::abs(rhs)) - 1e-9 * std::abs(r);
      };
      room_hi_[i] = room(!is_finite(rhi) || mininf >= 2, rhi - minact, rhi);
      room_lo_[i] = room(!is_finite(rlo) || maxinf >= 2, maxact - rlo, rlo);
      room_stamp_[i] = call;
      continue;
    }
    // Tighten integer variables only (continuous tightening is left to the LP) unless `continuous`.
    // Per column the same argument as for the row: with both bounds finite, a side tightens nothing while the
    // column's span is within that side's slack (or the side cannot act: infinite, or an unbounded column
    // elsewhere in the row). Most columns of a long row end at this test. The activities are those of the scan,
    // also after tightenings in this loop, as below. The margin covers the rounding of the residual activities,
    // so that the implied bound computed below would lie at or beyond the column's own; an integer column with a
    // fractional bound goes through the loop (it is rounded there).
    const double col_room = std::min(is_finite(rhi) && mininf == 0 ? rhi - minact : kInf,
                                     is_finite(rlo) && maxinf == 0 ? maxact - rlo : kInf) -
                            1e-12 * (asum + (is_finite(rhi) ? std::abs(rhi) : 0.0) + (is_finite(rlo) ? std::abs(rlo) : 0.0));
    for (Index k = rb; k < re && feasible; ++k) {
      const Index j = rows.index[k];
      const double a = rows.value[k];
      bool known_same = false;
      {
        const double lj = lb[j], uj = ub[j];
        known_same = std::abs(lj) <= kBigBound && std::abs(uj) <= kBigBound && uj >= lj && std::abs(a) * (uj - lj) <= col_room &&
                     (!p_.is_int[j] || (lj == std::floor(lj) && uj == std::floor(uj)));
      }
#ifndef MILP_PROP_CHECK
      if (known_same) continue;
#endif
      const bool is_int = p_.is_int[j];
      if (!is_int && !continuous) continue;
      const double l = lb[j] < -kBigBound ? -kInf : lb[j], u = ub[j] > kBigBound ? kInf : ub[j];
      // Residual activities without j.
      double rmin, rmax;
      bool rmin_ok, rmax_ok;
      if (a > 0) {
        rmin_ok = mininf - (is_finite(l) ? 0 : 1) == 0;
        rmin = is_finite(l) ? minact - a * l : minact;
        rmax_ok = maxinf - (is_finite(u) ? 0 : 1) == 0;
        rmax = is_finite(u) ? maxact - a * u : maxact;
      } else {
        rmin_ok = mininf - (is_finite(u) ? 0 : 1) == 0;
        rmin = is_finite(u) ? minact - a * u : minact;
        rmax_ok = maxinf - (is_finite(l) ? 0 : 1) == 0;
        rmax = is_finite(l) ? maxact - a * l : maxact;
      }
      double new_lb = l, new_ub = u;
      const double slack = kFeasTol * (1.0 + std::abs(rmin) + std::abs(rmax));
      // Integers round; continuous bounds keep the slack and must move by a meaningful amount.
      auto down = [&](double v) { return is_int ? std::floor(v + 1e-9) : v; };
      auto up = [&](double v) { return is_int ? std::ceil(v - 1e-9) : v; };
      if (is_finite(rhi) && rmin_ok) {
        const double bound = (rhi - rmin) / a;
        if (a > 0) new_ub = std::min(new_ub, down(bound + slack / a));
        else new_lb = std::max(new_lb, up(bound - slack / -a));
      }
      if (is_finite(rlo) && rmax_ok) {
        const double bound = (rlo - rmax) / a;
        if (a > 0) new_lb = std::max(new_lb, up(bound - slack / a));
        else new_ub = std::min(new_ub, down(bound + slack / -a));
      }
      if (known_same && (new_lb > lb[j] || new_ub < ub[j])) std::abort();  // MILP_PROP_CHECK only
      if (new_lb > new_ub + (is_int ? 0.0 : kFeasTol * (1.0 + std::abs(new_lb)))) {
        if (trail) {
          ensure_levels();
          if (new_lb > lb[j]) trail->push(j, false, new_lb, BoundTrail::kRow, i, nullptr, implied_level(j, a > 0));
          if (new_ub < ub[j]) trail->push(j, true, new_ub, BoundTrail::kRow, i, nullptr, implied_level(j, a < 0));
          trail->fail_crossing(j);
        }
        if (uses_cont) ++last_cont_useful;
        if (i == obj_row_) last_obj_useful = true;
        feasible = false;
        break;
      }
      if (!is_int) {
        // Meaningful changes only (no long chains of tiny steps); infinite bounds always count.
        const double width = (is_finite(u) && is_finite(l)) ? u - l : kInf;
        static const double ce = [] { const char* v = std::getenv("MILP_PROP_CONT_EPS"); return v ? std::atof(v) : 1e-3; }();
        const double eps = is_finite(width) ? ce * std::max(width, 1e-3) : ce * (1.0 + std::abs(new_lb > l ? new_lb : new_ub));
        if (new_lb < l + eps) new_lb = l;
        if (new_ub > u - eps) new_ub = u;
        if (new_lb > new_ub) new_lb = l, new_ub = u;  // within tolerance: leave it
      }
      if (new_lb > lb[j] || new_ub < ub[j]) {
        if (trail) {
          ensure_levels();
          // With a > 0: x >= from the maximal activity, x <= from the minimal one (the reverse if a < 0).
          if (new_lb > lb[j]) trail->push(j, false, new_lb, BoundTrail::kRow, i, nullptr, implied_level(j, a > 0));
          if (new_ub < ub[j]) trail->push(j, true, new_ub, BoundTrail::kRow, i, nullptr, implied_level(j, a < 0));
        }
        const double l0 = lb[j], u0 = ub[j];
        lb[j] = std::max(lb[j], new_lb);
        ub[j] = std::min(ub[j], new_ub);
        if (i == obj_row_) last_obj_useful = true;
        if (is_int) {
          if (uses_cont) ++last_cont_useful;
        } else if (!cont_mark_[j]) {
          cont_mark_[j] = 1;
          cont_marked_.push_back(j);
        }
        tightened.push_back(j);
        tightened_col(j, l0, u0);
        // The activity of this row changed too; it is re-queued by enqueue_col.
      }
    }
  }
  for (size_t h = 0; h < queue_.size(); ++h) queued_[queue_[h]] = 0;
  for (Index j : cont_marked_) cont_mark_[j] = 0;
  cont_marked_.clear();
  last_work = work;
  return feasible;
}

}  // namespace milp
