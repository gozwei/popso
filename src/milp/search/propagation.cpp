#include "milp/search/propagation.hpp"

#include <cmath>
#include <cstdlib>

namespace milp {

namespace {
constexpr double kFeasTol = 1e-6;
constexpr double kBigBound = 1e12;  // larger bounds are treated as infinite
}  // namespace

DomainPropagator::DomainPropagator(const MipProblem& p)
    : p_(p), rows_(p.A.transpose()), row_lb_(p.row_lb), row_ub_(p.row_ub) {
  queued_.assign(p.num_rows(), 0);
}

void DomainPropagator::enable_objective() {
  if (obj_row_ >= 0) return;
  std::vector<Index> idx;
  std::vector<double> val;
  for (Index j = 0; j < p_.num_cols(); ++j) {
    if (p_.cost[j] != 0.0) idx.push_back(j), val.push_back(p_.cost[j]);
  }
  if (idx.empty()) return;
  obj_row_ = rows_.num_major;
  rows_.append(idx.data(), val.data(), static_cast<Index>(idx.size()));
  row_lb_.push_back(-kInf);
  row_ub_.push_back(kInf);
  queued_.push_back(0);
}

void DomainPropagator::set_objective_cutoff(double cutoff) {
  if (obj_row_ >= 0) row_ub_[obj_row_] = cutoff;
}

bool DomainPropagator::propagate(std::vector<double>& lb, std::vector<double>& ub, const std::vector<Index>& seeds,
                                 std::vector<Index>& tightened) {
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
  last_cont_useful = 0;
  if (continuous && cont_mark_.size() != static_cast<size_t>(p_.num_cols())) cont_mark_.assign(p_.num_cols(), 0);

  bool feasible = true;
  for (size_t head = 0; head < queue_.size() && feasible; ++head) {
    const Index i = queue_[head];
    queued_[i] = 0;
    if (i == obj_row_) obj_seen = true;
    if (work > limit) continue;  // drain the queue flags
    const Index rb = rows_.begin(i), re = rows_.end(i);
    work += re - rb;
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
    auto note_level = [&](int side, Index j, int lv) {
      if (lv > lv1[side]) lv2[side] = lv1[side], lv1[side] = lv, lvj[side] = j;
      else if (lv > lv2[side]) lv2[side] = lv;
    };
    for (Index k = rb; k < re; ++k) {
      const Index j = rows_.index[k];
      const double a = rows_.value[k];
      const double lj = lb[j], uj = ub[j];
      if (std::abs(lj) <= kBigBound && std::abs(uj) <= kBigBound) {
        // Both bounds finite (nearly every entry): the same sums as below without the tests.
        const double al = a * lj, au = a * uj;
        if (a > 0) minact += al, maxact += au;
        else minact += au, maxact += al;
        maxspan = std::max(maxspan, std::abs(a) * (uj - lj));
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
        const Index j = rows_.index[k];
        const double a = rows_.value[k];
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
    if (hi_idle && lo_idle) continue;
    // Tighten integer variables only (continuous tightening is left to the LP) unless `continuous`.
    for (Index k = rb; k < re && feasible; ++k) {
      const Index j = rows_.index[k];
      const bool is_int = p_.is_int[j];
      if (!is_int && !continuous) continue;
      const double a = rows_.value[k];
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
        enqueue_col(j);
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
