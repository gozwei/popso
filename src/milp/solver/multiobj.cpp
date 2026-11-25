#include "milp/solver/multiobj.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

#include "milp/core/log.hpp"
#include "milp/core/timer.hpp"

namespace milp {

std::vector<ObjectiveStage> objective_stages(const Model& model) {
  std::map<int, ObjectiveStage, std::greater<int>> by_priority;
  for (int k = 0; k < model.num_objectives(); ++k) {
    const int prio = k == 0 ? model.obj_priority : model.objectives[static_cast<size_t>(k - 1)].priority;
    const double at = k == 0 ? model.obj_abs_tol : model.objectives[static_cast<size_t>(k - 1)].abs_tol;
    const double rt = k == 0 ? model.obj_rel_tol : model.objectives[static_cast<size_t>(k - 1)].rel_tol;
    ObjectiveStage& s = by_priority[prio];
    s.priority = prio;
    s.members.push_back(k);
    s.abs_tol = std::max(s.abs_tol, at);
    s.rel_tol = std::max(s.rel_tol, rt);
  }
  std::vector<ObjectiveStage> out;
  for (auto& [prio, s] : by_priority) out.push_back(std::move(s));
  return out;
}

namespace {

// Blended objective of a stage: dense coefficients and the constant.
void blend(const Model& model, const ObjectiveStage& stage, std::vector<double>& c, double* offset) {
  c.assign(static_cast<size_t>(model.num_cols()), 0.0);
  *offset = 0.0;
  for (int k : stage.members) {
    if (k == 0) {
      for (Index j = 0; j < model.num_cols(); ++j) c[static_cast<size_t>(j)] += model.obj_weight * model.obj[static_cast<size_t>(j)];
      *offset += model.obj_weight * model.obj_offset;
    } else {
      const Objective& o = model.objectives[static_cast<size_t>(k - 1)];
      for (size_t t = 0; t < o.cols.size(); ++t) c[static_cast<size_t>(o.cols[t])] += o.weight * o.coefs[t];
      *offset += o.weight * o.offset;
    }
  }
}

// Appends the row lb <= c x <= ub to a model stored by column.
void append_row(Model& m, const std::string& name, const std::vector<double>& c, double lb, double ub) {
  const Index row = m.num_rows();
  SparseMatrix a;
  a.num_major = m.A.num_major;
  a.num_minor = row + 1;
  a.start.assign(1, 0);
  a.index.reserve(m.A.index.size() + c.size());
  a.value.reserve(m.A.value.size() + c.size());
  for (Index j = 0; j < m.A.num_major; ++j) {
    for (Index k = m.A.begin(j); k < m.A.end(j); ++k) {
      a.index.push_back(m.A.index[static_cast<size_t>(k)]);
      a.value.push_back(m.A.value[static_cast<size_t>(k)]);
    }
    if (c[static_cast<size_t>(j)] != 0.0) {
      a.index.push_back(row);
      a.value.push_back(c[static_cast<size_t>(j)]);
    }
    a.start.push_back(static_cast<Index>(a.index.size()));
  }
  m.A = std::move(a);
  m.row_names.push_back(name);
  m.row_lb.push_back(lb);
  m.row_ub.push_back(ub);
}

const char* objective_name(const Model& model, int k, std::string& scratch) {
  const std::string& n = k == 0 ? model.obj_name : model.objectives[static_cast<size_t>(k - 1)].name;
  scratch = n.empty() ? "objective " + std::to_string(k) : n;
  return scratch.c_str();
}

}  // namespace

SolveResult solve_multi_objective(const Model& model, const SolverParams& params_in, const std::vector<double>& start) {
  model.validate();  // the objectives' column indices, weights and tolerances (and: no quadratic objective)
  Timer timer;
  Logger log(params_in.verbosity);
  const std::vector<ObjectiveStage> stages = objective_stages(model);
  const bool maximize = model.sense == ObjSense::Maximize;
  log.log(1, "Multiple objectives: %d objectives in %zu stages (by descending priority)\n", model.num_objectives(), stages.size());

  // The stage model: the model as given, one objective, and a row per finished stage.
  Model sm = model;
  sm.objectives.clear();
  sm.obj_priority = 0;
  sm.obj_weight = 1.0;

  SolveResult out;
  out.status = Status::Unknown;
  std::vector<double> x = start;
  bool have_x = false;
  std::vector<double> c;
  std::string scratch;
  for (size_t s = 0; s < stages.size(); ++s) {
    const ObjectiveStage& st = stages[s];
    double offset = 0.0;
    blend(model, st, c, &offset);
    SolveResult::Stage info;
    info.priority = st.priority;
    if (log.verbosity() >= 1) {
      std::string names;
      for (int k : st.members) names += (names.empty() ? "" : ", ") + std::string(objective_name(model, k, scratch));
      log.log(1, "Objective stage %zu of %zu: priority %d (%s)\n", s + 1, stages.size(), st.priority, names.c_str());
    }
    const bool constant = std::all_of(c.begin(), c.end(), [](double v) { return v == 0.0; });
    if (constant && have_x) {
      // Nothing to optimize, and no row to keep: every point has the stage's value.
      info.status = Status::Optimal;
      info.objective = info.bound = offset;
      out.stages.push_back(info);
      continue;
    }
    SolverParams params = params_in;
    params.duals = 0;
    if (std::isfinite(params_in.time_limit)) {
      params.time_limit = params_in.time_limit - timer.elapsed();
      if (params.time_limit <= 0.0) {
        out.status = Status::TimeLimit;
        break;
      }
    }
    sm.obj = c;
    sm.obj_offset = offset;
    const double t0 = timer.elapsed();
    SolveResult r = solve(sm, params, have_x ? x : (s == 0 ? start : std::vector<double>()));
    info.status = r.status;
    info.objective = r.objective;
    info.bound = r.bound;
    info.time = timer.elapsed() - t0;
    info.nodes = r.stats.nodes;
    out.stages.push_back(info);
    out.stats.nodes += r.stats.nodes;
    out.stats.lp_iterations += r.stats.lp_iterations;
    out.stats.barrier_iterations += r.stats.barrier_iterations;
    out.stats.restarts += r.stats.restarts;
    out.stats.tree_restarts += r.stats.tree_restarts;
    if (s == 0) {
      out.stats.root_bound = r.stats.root_bound;
      out.stats.presolve = r.stats.presolve;
    }
    for (IncumbentEvent e : r.incumbent_trace) {
      e.time += t0;
      e.last_time += t0;
      out.incumbent_trace.push_back(std::move(e));
    }
    if (r.has_solution) {
      x = r.x;
      x.resize(static_cast<size_t>(model.num_cols()));
      have_x = true;
    }
    if (r.status != Status::Optimal) {
      // No solution: the first stage's verdict stands (infeasible, unbounded, a limit); in a later stage the
      // point of the stage before remains. An unbounded later stage is reported as such; an infeasible one
      // contradicts that point and can only be numerical.
      out.status = r.status;
      if (s > 0 && !r.has_solution && (r.status == Status::Infeasible || r.status == Status::InfeasibleOrUnbounded))
        out.status = Status::NumericalError;
      break;
    }
    out.status = Status::Optimal;
    if (s + 1 == stages.size() || constant) continue;
    // Keep this stage's value (the value at x, in the model's sense) within its tolerance.
    const double v = r.objective;
    // (plus a few units of round-off, so that the point found satisfies its own row as computed).
    const double tol = std::max(st.abs_tol, st.rel_tol * std::abs(v)) + 1e-12 * std::max(1.0, std::abs(v));
    const std::string name = "__objective_stage_" + std::to_string(s + 1);
    if (maximize) append_row(sm, name, c, v - offset - tol, kInf);
    else append_row(sm, name, c, -kInf, v - offset + tol);
  }
  out.has_solution = have_x;
  if (have_x) {
    out.x = x;
    out.objective_values.resize(static_cast<size_t>(model.num_objectives()));
    for (int k = 0; k < model.num_objectives(); ++k) out.objective_values[static_cast<size_t>(k)] = model.objective_value(k, x);
    out.objective = out.objective_values[0];
    if (log.verbosity() >= 1) {
      for (int k = 0; k < model.num_objectives(); ++k)
        log.log(1, "  %s: %.10g\n", objective_name(model, k, scratch), out.objective_values[static_cast<size_t>(k)]);
    }
  }
  // The bound of the model's own objective is known when it is the whole first stage.
  out.bound = maximize ? kInf : -kInf;
  if (!out.stages.empty() && stages[0].members.size() == 1 && stages[0].members[0] == 0 && model.obj_weight > 0.0)
    out.bound = out.stages[0].bound / model.obj_weight;
  out.stats.time = timer.elapsed();
  return out;
}

}  // namespace milp
