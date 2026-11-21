#include "milp/presolve/group_prop.hpp"

#include <algorithm>
#include <cmath>

namespace milp {

namespace {

class GroupProp {
 public:
  GroupProp(const MipProblem& p, const GroupPropOptions& opt, GroupPropResult& res)
      : p_(p), opt_(opt), res_(res), n_(p.num_cols()), m_(p.num_rows()) {
    rows_ = p.A.transpose();
    lb_ = p.col_lb;
    ub_ = p.col_ub;
    limit_ = opt.work_limit > 0 ? opt.work_limit : 50LL * p.A.nnz() + 100000;
  }

  void run() {
    find_groups();
    res_.groups = static_cast<Index>(gexact_.size());
    if (gexact_.empty()) return;
    queued_.assign(m_, 0);
    for (Index i = 0; i < m_; ++i) push_row(i);
    for (Index g = 0; g < res_.groups; ++g) mark_dirty(g);
    if (!propagate()) {
      res_.infeasible = true;
      return;
    }
    if (opt_.shave && !limit_hit()) shave();
  }

  std::vector<double>& lb() { return lb_; }
  std::vector<double>& ub() { return ub_; }
  long long work() const { return work_; }

 private:
  struct TrailEntry {
    Index j;
    double l, u;
  };

  bool limit_hit() const { return work_ >= limit_; }
  bool is_free(Index j) const { return lb_[j] == 0.0 && ub_[j] == 1.0; }

  void find_groups() {
    group_of_.assign(n_, -1);
    // Candidate rows: all coefficients 1 on integer columns within [0, 1], side 1 (= or <=), no member at 1.
    struct Cand {
      Index row;
      Index free;
      bool exact;
    };
    std::vector<Cand> cand;
    for (Index i = 0; i < m_; ++i) {
      if (p_.row_ub[i] != 1.0) continue;
      const bool exact = p_.row_lb[i] == 1.0;
      if (!exact && p_.row_lb[i] > 0.0) continue;
      Index free = 0;
      bool ok = true;
      for (Index k = rows_.begin(i); k < rows_.end(i) && ok; ++k) {
        const Index j = rows_.index[k];
        ok = rows_.value[k] == 1.0 && p_.is_int[j] && lb_[j] >= 0.0 && ub_[j] <= 1.0 && lb_[j] < 1.0;
        free += ok && ub_[j] == 1.0;
      }
      if (ok && free >= 2) cand.push_back({i, free, exact});
    }
    std::stable_sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) {
      return a.exact != b.exact ? a.exact : a.free > b.free;
    });
    gstart_.assign(1, 0);
    for (const Cand& c : cand) {
      const Index g = static_cast<Index>(gexact_.size());
      Index taken = 0, mine = 0;
      for (Index k = rows_.begin(c.row); k < rows_.end(c.row); ++k) {
        const Index j = rows_.index[k];
        if (ub_[j] != 1.0) continue;  // fixed at 0
        if (group_of_[j] >= 0) {
          ++taken;
          continue;
        }
        gcols_.push_back(j);
        ++mine;
      }
      if (mine < 2) {
        gcols_.resize(gstart_.back());
        continue;
      }
      for (Index k = gstart_.back(); k < static_cast<Index>(gcols_.size()); ++k) group_of_[gcols_[k]] = g;
      gstart_.push_back(static_cast<Index>(gcols_.size()));
      gexact_.push_back(c.exact && taken == 0);
      gfree_.push_back(mine);
      gone_.push_back(0);
    }
    const size_t ng = gexact_.size();
    gdirty_.assign(ng, 0);
    gstamp_.assign(ng, 0);
    gminc_.assign(ng, 0.0);
    gmaxc_.assign(ng, 0.0);
    gcnt_.assign(ng, 0);
    colmark_.assign(n_, 0);
  }

  void push_row(Index i) {
    if (queued_[i]) return;
    queued_[i] = 1;
    queue_.push_back(i);
  }

  void mark_dirty(Index g) {
    if (gdirty_[g]) return;
    gdirty_[g] = 1;
    dirty_.push_back(g);
  }

  // Sets the bounds of a column (no rounding here), keeps the group counters, queues its rows.
  void set_bounds(Index j, double l, double u) {
    const Index g = group_of_[j];
    if (g >= 0) {
      gfree_[g] -= is_free(j);
      gone_[g] -= lb_[j] >= 1.0;
    }
    if (recording_) trail_.push_back({j, lb_[j], ub_[j]});
    lb_[j] = l;
    ub_[j] = u;
    if (g >= 0) {
      gfree_[g] += is_free(j);
      gone_[g] += lb_[j] >= 1.0;
      mark_dirty(g);
    }
    for (Index k = p_.A.begin(j); k < p_.A.end(j); ++k) push_row(p_.A.index[k]);
    work_ += p_.A.end(j) - p_.A.begin(j);
  }

  void undo() {
    for (size_t t = trail_.size(); t-- > 0;) {
      const TrailEntry& e = trail_[t];
      const Index g = group_of_[e.j];
      if (g >= 0) {
        gfree_[g] -= is_free(e.j);
        gone_[g] -= lb_[e.j] >= 1.0;
      }
      lb_[e.j] = e.l;
      ub_[e.j] = e.u;
      if (g >= 0) {
        gfree_[g] += is_free(e.j);
        gone_[g] += lb_[e.j] >= 1.0;
      }
    }
    trail_.clear();
    for (Index i : queue_) queued_[i] = 0;
    queue_.clear();
    for (Index g : dirty_) gdirty_[g] = 0;
    dirty_.clear();
  }

  bool fix(Index j, double v) {
    if (v < lb_[j] || v > ub_[j]) return false;
    if (lb_[j] == ub_[j]) return true;
    set_bounds(j, v, v);
    if (!recording_) ++res_.fixed;
    return true;
  }

  // A group whose counters changed: one member at 1 fixes the others at 0; an exact group needs a free member
  // and its last one is 1. The rows of the remaining free members see a changed group and are queued.
  bool resolve(Index g) {
    if (gone_[g] > 1) return false;
    if (gone_[g] == 1) {
      for (Index k = gstart_[g]; k < gstart_[g + 1]; ++k) {
        const Index j = gcols_[k];
        if (is_free(j)) set_bounds(j, 0.0, 0.0), res_.fixed += !recording_;
      }
      work_ += gstart_[g + 1] - gstart_[g];
      return true;
    }
    if (gexact_[g] && gfree_[g] == 0) return false;
    for (Index k = gstart_[g]; k < gstart_[g + 1]; ++k) {
      const Index j = gcols_[k];
      if (!is_free(j)) continue;
      if (gexact_[g] && gfree_[g] == 1) {
        set_bounds(j, 1.0, 1.0);
        res_.fixed += !recording_;
        break;
      }
      for (Index q = p_.A.begin(j); q < p_.A.end(j); ++q) push_row(p_.A.index[q]);
      work_ += p_.A.end(j) - p_.A.begin(j);
    }
    work_ += gstart_[g + 1] - gstart_[g];
    return true;
  }

  // To the fixpoint or the work limit. False: infeasible.
  bool propagate() {
    size_t head = 0;
    for (;;) {
      if (limit_hit()) break;
      if (head >= queue_.size()) {
        // The rows are done: the groups that changed, each once (a group queues the rows of all its free
        // members, so one pass per sweep, not one per fixed member).
        if (dirty_.empty()) break;
        std::vector<Index> todo;
        todo.swap(dirty_);
        for (Index g : todo) gdirty_[g] = 0;
        for (Index g : todo) {
          if (!resolve(g)) return false;
        }
        continue;
      }
      const Index i = queue_[head++];
      queued_[i] = 0;
      if (!process_row(i)) return false;
      if (head > 4096 && head * 2 > queue_.size()) {
        queue_.erase(queue_.begin(), queue_.begin() + static_cast<std::ptrdiff_t>(head));
        head = 0;
      }
    }
    // Rows still queued at the work limit stay unprocessed: the bounds so far are valid.
    for (size_t t = head; t < queue_.size(); ++t) queued_[queue_[t]] = 0;
    queue_.clear();
    for (Index g : dirty_) gdirty_[g] = 0;
    dirty_.clear();
    return true;
  }

  bool process_row(Index i) {
    const Index b = rows_.begin(i), e = rows_.end(i);
    work_ += e - b;
    const double rlb = p_.row_lb[i], rub = p_.row_ub[i];
    double mn = 0.0, mx = 0.0;
    int inf_mn = 0, inf_mx = 0;
    touched_.clear();
    ++stamp_;
    for (Index k = b; k < e; ++k) {
      const Index j = rows_.index[k];
      const double a = rows_.value[k];
      const Index g = group_of_[j];
      if (g >= 0 && gone_[g] == 0 && is_free(j)) {
        if (gstamp_[g] != stamp_) {
          gstamp_[g] = stamp_;
          gminc_[g] = gmaxc_[g] = a;
          gcnt_[g] = 1;
          touched_.push_back(g);
        } else {
          gminc_[g] = std::min(gminc_[g], a);
          gmaxc_[g] = std::max(gmaxc_[g], a);
          ++gcnt_[g];
        }
        continue;
      }
      const double lo = a > 0.0 ? lb_[j] : ub_[j], hi = a > 0.0 ? ub_[j] : lb_[j];
      if (std::isfinite(lo)) mn += a * lo; else ++inf_mn;
      if (std::isfinite(hi)) mx += a * hi; else ++inf_mx;
    }
    for (Index g : touched_) {
      // 0 is a candidate when a free member is absent from the row, or none has to be chosen.
      if (!gexact_[g] || gcnt_[g] < gfree_[g]) {
        gminc_[g] = std::min(gminc_[g], 0.0);
        gmaxc_[g] = std::max(gmaxc_[g], 0.0);
      }
      mn += gminc_[g];
      mx += gmaxc_[g];
    }
    const double tol_ub = 1e-6 * std::max(1.0, std::abs(rub)), tol_lb = 1e-6 * std::max(1.0, std::abs(rlb));
    const bool use_ub = std::isfinite(rub) && inf_mn == 0, use_lb = std::isfinite(rlb) && inf_mx == 0;
    if (use_ub && mn > rub + tol_ub) return false;
    if (use_lb && mx < rlb - tol_lb) return false;
    // Slack of each side: nothing to learn from a side that even the worst case satisfies.
    const bool tight_ub = std::isfinite(rub) && inf_mn <= 1, tight_lb = std::isfinite(rlb) && inf_mx <= 1;
    if (!tight_ub && !tight_lb) return true;

    // Group members.
    for (Index g : touched_) {
      const bool zero_option = !gexact_[g] || gcnt_[g] < gfree_[g];
      // "None of the row's members": violated -> the absent members are 0 (an at-most-one group then has one of
      // the row's members at 1, which the group's own row cannot say).
      if (zero_option && ((use_ub && mn - gminc_[g] > rub + tol_ub) || (use_lb && mx - gmaxc_[g] < rlb - tol_lb))) {
        for (Index k = b; k < e; ++k) {
          if (group_of_[rows_.index[k]] == g) colmark_[rows_.index[k]] = 1;
        }
        bool ok = true;
        for (Index k = gstart_[g]; k < gstart_[g + 1] && ok; ++k) {
          const Index j = gcols_[k];
          if (!colmark_[j] && is_free(j)) ok = fix(j, 0.0);
        }
        work_ += gstart_[g + 1] - gstart_[g];
        for (Index k = b; k < e; ++k) colmark_[rows_.index[k]] = 0;
        if (!ok) return false;
      }
    }
    for (Index k = b; k < e; ++k) {
      const Index j = rows_.index[k];
      const double a = rows_.value[k];
      const Index g = group_of_[j];
      if (g >= 0 && gstamp_[g] == stamp_ && gone_[g] == 0 && is_free(j)) {
        // Choosing this member.
        if ((use_ub && mn - gminc_[g] + a > rub + tol_ub) || (use_lb && mx - gmaxc_[g] + a < rlb - tol_lb)) {
          if (!fix(j, 0.0)) return false;
        }
        continue;
      }
      if (!p_.is_int[j] || lb_[j] == ub_[j]) continue;
      // The usual bounds from the residual activities.
      const double lo = a > 0.0 ? lb_[j] : ub_[j], hi = a > 0.0 ? ub_[j] : lb_[j];
      double nl = lb_[j], nu = ub_[j];
      if (std::isfinite(rub)) {
        double res = 0.0;
        bool have = false;
        if (inf_mn == 0) res = mn - a * lo, have = true;
        else if (inf_mn == 1 && !std::isfinite(lo)) res = mn, have = true;
        if (have) {
          const double v = (rub - res) / a;
          if (a > 0.0) nu = std::min(nu, std::floor(v + 1e-6 * std::max(1.0, std::abs(v))));
          else nl = std::max(nl, std::ceil(v - 1e-6 * std::max(1.0, std::abs(v))));
        }
      }
      if (std::isfinite(rlb)) {
        double res = 0.0;
        bool have = false;
        if (inf_mx == 0) res = mx - a * hi, have = true;
        else if (inf_mx == 1 && !std::isfinite(hi)) res = mx, have = true;
        if (have) {
          const double v = (rlb - res) / a;
          if (a > 0.0) nl = std::max(nl, std::ceil(v - 1e-6 * std::max(1.0, std::abs(v))));
          else nu = std::min(nu, std::floor(v + 1e-6 * std::max(1.0, std::abs(v))));
        }
      }
      if (nl > lb_[j] || nu < ub_[j]) {
        if (nl > nu) return false;
        set_bounds(j, nl, nu);
        if (!recording_) ++res_.tightened;
      }
    }
    return true;
  }

  // Integer columns with a cost, at their cheapest value: infeasible by propagation -> the bound moves by one.
  void shave() {
    std::vector<Index> cols;
    for (Index j = 0; j < n_; ++j) {
      if (p_.cost[j] == 0.0) continue;
      if (static_cast<int>(cols.size()) >= opt_.max_shave_cols) return;
      cols.push_back(j);
    }
    bool again = true;
    for (int sweep = 0; sweep < 4 && again; ++sweep) {
      again = false;
      for (Index j : cols) {
        if (!p_.is_int[j]) continue;
        for (int step = 0; step < opt_.max_shave_steps; ++step) {
          if (limit_hit() || lb_[j] == ub_[j]) break;
          const bool up = p_.cost[j] > 0.0;  // the cheap value is the lower bound
          const double v = up ? lb_[j] : ub_[j];
          if (!std::isfinite(v)) break;
          ++res_.shave_probes;
          recording_ = true;
          set_bounds(j, v, v);
          const bool ok = propagate();
          const bool complete = !limit_hit();
          undo();
          recording_ = false;
          if (ok || !complete) break;
          set_bounds(j, up ? v + 1.0 : lb_[j], up ? ub_[j] : v - 1.0);
          ++res_.shaved;
          again = true;
          if (!propagate()) {
            res_.infeasible = true;
            return;
          }
        }
      }
    }
  }

  const MipProblem& p_;
  const GroupPropOptions& opt_;
  GroupPropResult& res_;
  Index n_, m_;
  SparseMatrix rows_;  // CSR
  std::vector<double> lb_, ub_;
  std::vector<Index> group_of_, gstart_, gcols_, gfree_, gone_, gcnt_;
  std::vector<char> gexact_, gdirty_, queued_, colmark_;
  std::vector<unsigned> gstamp_;
  std::vector<double> gminc_, gmaxc_;
  std::vector<Index> queue_, dirty_, touched_;
  std::vector<TrailEntry> trail_;
  bool recording_ = false;
  unsigned stamp_ = 0;
  long long work_ = 0, limit_ = 0;
};

}  // namespace

GroupPropResult group_propagate(const MipProblem& p, const GroupPropOptions& opt) {
  GroupPropResult res;
  GroupProp gp(p, opt, res);
  gp.run();
  res.work = gp.work();
  res.lb = std::move(gp.lb());
  res.ub = std::move(gp.ub());
  return res;
}

}  // namespace milp
