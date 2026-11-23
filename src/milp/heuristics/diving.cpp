#include "milp/heuristics/diving.hpp"

#include <algorithm>
#include <cmath>

#include "milp/core/interrupt.hpp"
#include "milp/core/timer.hpp"

namespace milp {

const char* to_string(DiveRule r) {
  switch (r) {
    case DiveRule::Fractional: return "fractional diving";
    case DiveRule::Coefficient: return "coefficient diving";
    case DiveRule::Guided: return "guided diving";
    case DiveRule::VectorLength: return "vector-length diving";
  }
  return "diving";
}

namespace {

struct Choice {
  Index j = -1;
  bool up = false;
};

// Picks the variable to round and the direction according to the rule.
Choice choose(const MipProblem& p, const DualSimplex& lp, const double* x, DiveRule rule,
              const SimpleRounding& rounding, const std::vector<double>* inc) {
  Choice best;
  double best_score = kInf;
  for (Index j = 0; j < p.num_cols(); ++j) {
    if (!p.is_int[j]) continue;
    const double f = x[j] - std::floor(x[j]);
    if (f <= 1e-6 || f >= 1.0 - 1e-6) continue;
    // Just outside an integer bound (x = ub + 1e-6, within the LP's tolerance): rounding cannot
    // change the domain, and choosing it again and again made the dive loop forever.
    if (std::floor(x[j]) >= lp.col_upper(j) || std::ceil(x[j]) <= lp.col_lower(j)) continue;
    bool up = false;
    double score = kInf;
    switch (rule) {
      case DiveRule::Coefficient: {
        // Round in the direction with fewer locks; prefer few locks, then small fractionality.
        const Index dl = rounding.down_locks(j), ul = rounding.up_locks(j);
        up = ul < dl || (ul == dl && f > 0.5);
        const double dist = up ? 1.0 - f : f;
        score = static_cast<double>(up ? ul : dl) + dist;
        break;
      }
      case DiveRule::Guided: {
        if (inc) {
          up = (*inc)[j] > x[j];
          score = std::abs((*inc)[j] - x[j]);
          break;
        }
        [[fallthrough]];
      }
      case DiveRule::Fractional: {
        up = f > 0.5;
        score = up ? 1.0 - f : f;
        break;
      }
      case DiveRule::VectorLength: {
        // Objective deterioration per row touched (Eckstein-Nediak style).
        const Index len = p.A.end(j) - p.A.begin(j) + 1;
        up = p.cost[j] < 0;  // round in the objective-improving direction
        const double dist = up ? 1.0 - f : f;
        score = (std::abs(p.cost[j]) * dist + 1e-6) / static_cast<double>(len);
        break;
      }
    }
    if (score < best_score) best_score = score, best.j = j, best.up = up;
  }
  return best;
}

}  // namespace

void TreeDiveHistory::record(long long iters, bool win, bool limit_hit) {
  ++calls;
  wins += win;
  if (limit_hit) {
    need = std::max(need, 2 * iters);
  } else {
    ++done;
    done_iters += iters;
    need = 0;
  }
}

long long tree_dive_budget(const TreeDiveHistory& h, long long iters, long long dived, bool adapt, double adapt_max,
                           bool* wait) {
  if (wait) *wait = false;
  double share = 0.1;
  if (adapt) {
    share = std::min(adapt_max, 0.05 * (1.0 + 10.0 * static_cast<double>(h.wins + 1) / static_cast<double>(h.calls + 1)));
  }
  const long long budget = 1000 + static_cast<long long>(share * static_cast<double>(iters)) - dived;
  if (budget <= 0 || !adapt) return budget;
  const long long whole = std::max(h.need, h.done > 0 ? h.done_iters / h.done : 0);
  if (budget >= whole) return budget;
  if (wait) *wait = true;
  return 0;
}

bool dive(const MipProblem& p, DualSimplex& lp, DiveRule rule, const SimpleRounding& rounding,
          const std::vector<double>* incumbent, const DiveLimits& limits, std::vector<double>& sol,
          long long* iterations, DomainPropagator* prop, DiveOutcome* outcome) {
  Timer timer;
  const long long it0 = lp.iterations();
  const Index n = p.num_cols();
  struct Saved {
    Index j;
    double lb, ub;
  };
  std::vector<Saved> saved;
  // Local integer domains for propagation (mirror the LP column bounds).
  std::vector<double> lb, ub;
  std::vector<Index> seeds(1), tightened;
  if (prop) {
    lb.resize(n);
    ub.resize(n);
    for (Index j = 0; j < n; ++j) lb[j] = lp.col_lower(j), ub[j] = lp.col_upper(j);
  }
  auto set_bounds = [&](Index j, double l, double u) {
    saved.push_back({j, lp.col_lower(j), lp.col_upper(j)});
    lp.set_col_bounds(j, l, u);
    if (prop) lb[j] = l, ub[j] = u;
  };
  auto undo = [&](size_t mark) {
    for (; saved.size() > mark; saved.pop_back()) {
      const Saved& s = saved.back();
      lp.set_col_bounds(s.j, s.lb, s.ub);
      if (prop) lb[s.j] = s.lb, ub[s.j] = s.ub;
    }
  };
  bool found = false;
  DiveOutcome out;
  for (;;) {
    const double* x = lp.x();
    if (rounding.run(x, sol)) {
      found = true;
      break;
    }
    const Choice c = choose(p, lp, x, rule, rounding, incumbent);
    if (c.j < 0) {
      sol.assign(x, x + n);
      found = true;
      break;
    }
    const long long used = lp.iterations() - it0;
    if (used >= limits.iteration_budget || timer.elapsed() > limits.time_limit || interrupted() ||
        (limits.stop && limits.stop->load())) {
      out.limit_hit = true;
      break;
    }

    const double v = x[c.j];
    const double lo = lp.col_lower(c.j), hi = lp.col_upper(c.j);
    LpOptions opt;
    opt.objective_cutoff = limits.cutoff;
    opt.iteration_limit = limits.iteration_budget - used;
    opt.time_limit = limits.time_limit - timer.elapsed();
    // Try the preferred direction, then backtrack once to the other one.
    bool ok = false;
    for (int attempt = 0; attempt < 2 && !ok; ++attempt) {
      const bool up = (attempt == 0) == c.up;
      const size_t mark = saved.size();
      if (up) set_bounds(c.j, std::ceil(v), hi);
      else set_bounds(c.j, lo, std::floor(v));
      if (prop) {
        // Fix-and-propagate: implied integer bounds go into the LP as well.
        seeds[0] = c.j;
        tightened.clear();
        if (!prop->propagate(lb, ub, seeds, tightened)) {
          for (Index t : tightened) lb[t] = lp.col_lower(t), ub[t] = lp.col_upper(t);
          undo(mark);
          continue;
        }
        for (Index t : tightened) {
          const double l = lb[t], u = ub[t];
          if (l != lp.col_lower(t) || u != lp.col_upper(t)) set_bounds(t, l, u);
        }
      }
      const LpStatus st = lp.solve(opt);
      ++out.steps;
      ok = st == LpStatus::Optimal && lp.objective() < limits.cutoff;
      if (!ok) undo(mark);
      if (!ok && st != LpStatus::Infeasible && st != LpStatus::CutoffReached) {  // limits
        out.limit_hit = st != LpStatus::Optimal;
        break;
      }
    }
    if (!ok) break;
  }
  undo(0);
  if (iterations) *iterations = lp.iterations() - it0;
  if (outcome) *outcome = out;
  return found;
}

}  // namespace milp
