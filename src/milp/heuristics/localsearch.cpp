#include "milp/heuristics/localsearch.hpp"

#include <algorithm>
#include <cmath>

namespace milp {

bool one_opt(const MipProblem& p, std::vector<double>& x) {
  const Index n = p.num_cols();
  std::vector<double> act(p.num_rows(), 0.0);
  for (Index j = 0; j < n; ++j) {
    for (Index k = p.A.begin(j); k < p.A.end(j); ++k) act[p.A.index[k]] += p.A.value[k] * x[j];
  }
  bool improved = false;
  for (int pass = 0; pass < 3; ++pass) {
    bool changed = false;
    for (Index j = 0; j < n; ++j) {
      const double c = p.cost[j];
      if (c == 0.0) continue;
      const double dir = c > 0 ? -1.0 : 1.0;
      double step = dir < 0 ? x[j] - p.col_lb[j] : p.col_ub[j] - x[j];
      for (Index k = p.A.begin(j); k < p.A.end(j) && step > 0.0; ++k) {
        const Index i = p.A.index[k];
        const double a = p.A.value[k] * dir;  // activity change per unit step
        const double slack = a > 0 ? p.row_ub[i] - act[i] : act[i] - p.row_lb[i];
        step = std::min(step, std::max(0.0, slack) / std::abs(a));
      }
      if (!is_finite(step)) continue;  // unbounded ray: leave it to the LP
      if (p.is_int[j]) step = std::floor(step + 1e-9);
      if (step <= 1e-9) continue;
      x[j] += dir * step;
      if (p.is_int[j]) x[j] = std::round(x[j]);
      for (Index k = p.A.begin(j); k < p.A.end(j); ++k) act[p.A.index[k]] += p.A.value[k] * dir * step;
      changed = improved = true;
    }
    if (!changed) break;
  }
  return improved;
}

bool local_branching(const MipProblem& p, const std::vector<double>& incumbent, int k, double cutoff,
                     const SubMipLimits& limits, std::vector<double>& sol) {
  const Index n = p.num_cols(), m = p.num_rows();
  MipProblem q;
  q.cost = p.cost;
  q.col_lb = p.col_lb;
  q.col_ub = p.col_ub;
  q.is_int = p.is_int;
  q.offset = p.offset;
  q.display_sense = p.display_sense;
  q.implications = p.implications;
  q.row_lb = p.row_lb;
  q.row_ub = p.row_ub;
  q.A.num_major = n;
  q.A.num_minor = m + 1;
  q.A.start.assign(1, 0);
  q.A.index.reserve(p.A.nnz() + n);
  q.A.value.reserve(p.A.nnz() + n);
  Index binaries = 0, ones = 0;
  for (Index j = 0; j < n; ++j) {
    for (Index t = p.A.begin(j); t < p.A.end(j); ++t) {
      q.A.index.push_back(p.A.index[t]);
      q.A.value.push_back(p.A.value[t]);
    }
    if (p.is_int[j] && p.col_lb[j] == 0.0 && p.col_ub[j] == 1.0) {
      const bool one = incumbent[j] > 0.5;
      q.A.index.push_back(m);
      q.A.value.push_back(one ? -1.0 : 1.0);
      ++binaries;
      ones += one;
    }
    q.A.start.push_back(static_cast<Index>(q.A.index.size()));
  }
  if (binaries < 2 * k) return false;
  q.row_lb.push_back(-kInf);
  q.row_ub.push_back(static_cast<double>(k - ones));
  return solve_sub_mip(q, q.col_lb, q.col_ub, cutoff, limits, sol);
}

}  // namespace milp
