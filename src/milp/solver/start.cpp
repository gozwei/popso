#include "milp/solver/start.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "milp/heuristics/submip.hpp"

namespace milp {

namespace {

std::string format(const char* fmt, double a, double b = 0.0, double c = 0.0) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), fmt, a, b, c);
  return buf;
}

}  // namespace

StartOutcome prepare_start(const MipProblem& p, const std::vector<double>& start, const SolverParams& params,
                           const Timer& timer) {
  StartOutcome out;
  const Index n = p.num_cols();
  if (static_cast<Index>(start.size()) != n) {
    out.message = "Start solution: wrong number of values; ignored";
    return out;
  }
  constexpr double kTol = 1e-6;
  Index given = 0, ints = 0, ints_given = 0;
  for (Index j = 0; j < n; ++j) {
    const bool g = !std::isnan(start[j]);
    given += g;
    if (p.is_int[j]) ++ints, ints_given += g;
  }
  if (given == 0) {
    out.message = "Start solution: no values given; ignored";
    return out;
  }
  std::string prefix;
  if (given == n) {
    const double viol = p.max_violation(start);
    if (viol <= kTol) {
      out.x = start;
      out.message = format("Start solution: feasible, objective %.10g", p.display_sense * p.objective(start));
      return out;
    }
    prefix = format("Start solution: infeasible (violation %.3g)", viol);
  } else {
    prefix = format("Start solution: partial (%.0f of %.0f values)", given, n);
  }
  if (ints == 0) {
    out.message = prefix + "; ignored (the model has no integer variables)";
    return out;
  }
  if (ints_given == 0) {
    out.message = prefix + "; ignored (no integer values given)";
    return out;
  }
  // Fix the given integers (rounded, within bounds) and complete the rest.
  std::vector<double> lb = p.col_lb, ub = p.col_ub;
  for (Index j = 0; j < n; ++j) {
    if (!p.is_int[j] || std::isnan(start[j])) continue;
    const double v = std::clamp(std::round(start[j]), lb[j], ub[j]);
    lb[j] = ub[j] = v;
  }
  SubMipLimits lim;
  lim.node_limit = 1000;
  const double remaining = params.time_limit - timer.elapsed();
  lim.time_limit = std::max(0.0, std::min(5.0, 0.1 * remaining));
  lim.mem_limit_mb = static_cast<int>(std::clamp<long long>(params.mem_limit_mb, 1, 256));
  // Deterministic mode: an iteration budget makes the outcome independent of timing (the time
  // limit stays as a safeguard).
  if (params.deterministic) lim.iteration_limit = 200000;
  std::vector<double> sol;
  if (lim.time_limit > 0.0 && solve_sub_mip(p, lb, ub, kInf, lim, sol) && p.max_violation(sol) <= kTol) {
    out.x = std::move(sol);
    out.message = prefix + format("; completed with %.0f of %.0f integers fixed, objective %.10g", ints_given, ints,
                                  p.display_sense * p.objective(out.x));
    return out;
  }
  out.message = prefix + "; could not be completed to a feasible solution, ignored";
  return out;
}

}  // namespace milp
