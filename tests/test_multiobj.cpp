// Tests for models with several objectives (solver/multiobj.hpp).

#include <cmath>
#include <random>
#include <stdexcept>

#include "milp/core/model.hpp"
#include "milp/solver/multiobj.hpp"
#include "milp/solver/solver.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

SolverParams quiet() {
  SolverParams p;
  p.verbosity = 0;
  p.threads = 1;
  p.mip_rel_gap = 0.0;
  p.mip_abs_gap = 0.0;
  return p;
}

Objective objective(std::string name, std::vector<Index> cols, std::vector<double> coefs, int priority, double weight = 1.0,
                    double abs_tol = 0.0, double rel_tol = 0.0, double offset = 0.0) {
  Objective o;
  o.name = std::move(name);
  o.cols = std::move(cols);
  o.coefs = std::move(coefs);
  o.priority = priority;
  o.weight = weight;
  o.abs_tol = abs_tol;
  o.rel_tol = rel_tol;
  o.offset = offset;
  return o;
}

// max  first = x + 2y + 3 (priority 2, tolerances 0.5 / 1%), then 2 (z - x) - (y - z) (priority 1, blended)
// s.t. x + y + z <= 7, x - y >= -2, x integer in [0, 10], y in [0, 5], z binary.
Model three_objectives() {
  ModelBuilder b;
  const Index x = b.add_col("x", 1.0, 0.0, 10.0, VarType::Integer);
  const Index y = b.add_col("y", 2.0, 0.0, 5.0);
  const Index z = b.add_col("z", 0.0, 0.0, 1.0, VarType::Binary);
  const Index c1 = b.add_row("c1", -kInf, 7.0);
  const Index c2 = b.add_row("c2", -2.0, kInf);
  b.add_coef(c1, x, 1.0), b.add_coef(c1, y, 1.0), b.add_coef(c1, z, 1.0);
  b.add_coef(c2, x, 1.0), b.add_coef(c2, y, -1.0);
  Model m = b.build();
  m.sense = ObjSense::Maximize;
  m.obj_offset = 3.0;
  m.obj_name = "first";
  m.obj_priority = 2;
  m.obj_abs_tol = 0.5;
  m.obj_rel_tol = 0.01;
  m.objectives.push_back(objective("second", {x, z}, {-1.0, 1.0}, 1, 2.0));
  m.objectives.push_back(objective("third", {y, z}, {1.0, -1.0}, 1, -1.0));
  return m;
}

// Random small pure-integer model with `nobj` objectives of distinct priorities and zero tolerances.
Model random_lex_model(std::mt19937& rng, int nobj, bool maximize) {
  std::uniform_int_distribution<int> coef(-4, 4), rhs(4, 12), nv(3, 5), nr(2, 3);
  const int n = nv(rng), m = nr(rng);
  ModelBuilder b;
  for (int j = 0; j < n; ++j) b.add_col("x" + std::to_string(j), static_cast<double>(coef(rng)), 0.0, 3.0, VarType::Integer);
  for (int i = 0; i < m; ++i) {
    const Index r = b.add_row("r" + std::to_string(i), -kInf, static_cast<double>(rhs(rng)));
    for (int j = 0; j < n; ++j) {
      const int a = coef(rng);
      if (a != 0) b.add_coef(r, j, static_cast<double>(std::abs(a)));
    }
  }
  Model mod = b.build();
  mod.sense = maximize ? ObjSense::Maximize : ObjSense::Minimize;
  mod.obj_priority = nobj;
  mod.obj_abs_tol = 0.0;
  for (int k = 1; k < nobj; ++k) {
    std::vector<Index> cols;
    std::vector<double> coefs;
    for (int j = 0; j < n; ++j) {
      const int a = coef(rng);
      if (a != 0) cols.push_back(j), coefs.push_back(static_cast<double>(a));
    }
    mod.objectives.push_back(objective("o" + std::to_string(k), cols, coefs, nobj - k));
  }
  return mod;
}

// Lexicographic optimum by enumeration of the integer box [0, 3]^n.
bool enumerate_lex(const Model& m, std::vector<double>* best_vals) {
  const int n = m.num_cols();
  std::vector<double> x(static_cast<size_t>(n), 0.0);
  bool found = false;
  const double sign = m.sense == ObjSense::Maximize ? -1.0 : 1.0;
  for (;;) {
    if (m.violation(x).max() <= 1e-9) {
      std::vector<double> v(static_cast<size_t>(m.num_objectives()));
      for (int k = 0; k < m.num_objectives(); ++k) v[static_cast<size_t>(k)] = m.objective_value(k, x);
      bool better = !found;
      if (found) {
        for (size_t k = 0; k < v.size(); ++k) {
          if (sign * v[k] < sign * (*best_vals)[k] - 1e-9) {
            better = true;
            break;
          }
          if (sign * v[k] > sign * (*best_vals)[k] + 1e-9) break;
        }
      }
      if (better) *best_vals = v, found = true;
    }
    int j = 0;
    while (j < n && x[static_cast<size_t>(j)] == 3.0) x[static_cast<size_t>(j++)] = 0.0;
    if (j == n) break;
    x[static_cast<size_t>(j)] += 1.0;
  }
  return found;
}

}  // namespace

TEST(multiobj_stages_group_by_priority) {
  const Model m = three_objectives();
  const std::vector<ObjectiveStage> st = objective_stages(m);
  CHECK_EQ(st.size(), static_cast<size_t>(2));
  CHECK_EQ(st[0].priority, 2);
  CHECK_EQ(st[0].members.size(), static_cast<size_t>(1));
  CHECK_EQ(st[0].members[0], 0);
  CHECK_NEAR(st[0].abs_tol, 0.5, 0.0);
  CHECK_EQ(st[1].priority, 1);
  CHECK_EQ(st[1].members.size(), static_cast<size_t>(2));
  CHECK_EQ(st[1].members[0], 1);
  CHECK_EQ(st[1].members[1], 2);
}

TEST(multiobj_tolerance_and_blend) {
  const Model m = three_objectives();
  m.validate();
  const SolveResult r = solve(m, quiet());
  CHECK(r.status == Status::Optimal);
  CHECK(r.has_solution);
  // Stage 1: 14 at (3, 4, 0). Stage 2 may give up 0.5 of it: x = 3, y = 3.75, z = 0.
  CHECK_EQ(r.stages.size(), static_cast<size_t>(2));
  CHECK_NEAR(r.stages[0].objective, 14.0, 1e-6);
  CHECK_EQ(r.objective_values.size(), static_cast<size_t>(3));
  CHECK_NEAR(r.objective_values[0], 13.5, 1e-6);
  CHECK_NEAR(r.objective_values[1], -3.0, 1e-6);
  CHECK_NEAR(r.objective_values[2], 3.75, 1e-6);
  CHECK_NEAR(r.objective, r.objective_values[0], 0.0);
  CHECK_NEAR(r.stages[1].objective, 2.0 * -3.0 - 3.75, 1e-6);
  CHECK_NEAR(r.x[0], 3.0, 1e-6);
  CHECK_NEAR(r.x[2], 0.0, 1e-6);
  CHECK_NEAR(r.bound, 14.0, 1e-6);
  CHECK(m.violation(r.x).max() <= 1e-6);
}

TEST(multiobj_lexicographic_matches_enumeration) {
  std::mt19937 rng(20261123);
  int solved = 0;
  for (int it = 0; it < 60; ++it) {
    const Model m = random_lex_model(rng, 2 + it % 3, it % 2 == 1);
    std::vector<double> want;
    const bool feasible = enumerate_lex(m, &want);
    SolverParams p = quiet();
    p.deterministic = it % 4 == 0;
    p.presolve = it % 5 != 0;
    const SolveResult r = solve(m, p);
    CHECK(feasible);  // x = 0 satisfies every row
    CHECK(r.status == Status::Optimal);
    if (r.status != Status::Optimal) continue;
    ++solved;
    CHECK_EQ(r.objective_values.size(), want.size());
    for (size_t k = 0; k < want.size(); ++k) CHECK_NEAR(r.objective_values[k], want[k], 1e-6);
    CHECK(m.violation(r.x).max() <= 1e-6);
  }
  CHECK_EQ(solved, 60);
}

// Stage values by enumeration for any priorities, weights and tolerances: stage by stage the best blended value
// over the points that kept the earlier stages within their tolerances.
std::vector<double> enumerate_stages(const Model& m) {
  const int n = m.num_cols();
  const double sign = m.sense == ObjSense::Maximize ? -1.0 : 1.0;
  std::vector<std::vector<double>> pts;
  std::vector<double> x(static_cast<size_t>(n), 0.0);
  for (;;) {
    if (m.violation(x).max() <= 1e-9) pts.push_back(x);
    int j = 0;
    while (j < n && x[static_cast<size_t>(j)] == 3.0) x[static_cast<size_t>(j++)] = 0.0;
    if (j == n) break;
    x[static_cast<size_t>(j)] += 1.0;
  }
  std::vector<double> out;
  for (const ObjectiveStage& st : objective_stages(m)) {
    auto blended = [&](const std::vector<double>& p) {
      double v = 0.0;
      for (int k : st.members) v += (k == 0 ? m.obj_weight : m.objectives[static_cast<size_t>(k - 1)].weight) * m.objective_value(k, p);
      return v;
    };
    double best = kInf;
    for (const auto& p : pts) best = std::min(best, sign * blended(p));
    best *= sign;
    out.push_back(best);
    const double tol = std::max(st.abs_tol, st.rel_tol * std::abs(best)) + 1e-9;
    std::vector<std::vector<double>> keep;
    for (const auto& p : pts)
      if (sign * blended(p) <= sign * best + tol) keep.push_back(p);
    pts.swap(keep);
  }
  return out;
}

TEST(multiobj_tolerances_and_weights_match_enumeration) {
  std::mt19937 rng(771);
  std::uniform_int_distribution<int> prio(0, 2), wt(-2, 3), tol(0, 3);
  for (int it = 0; it < 60; ++it) {
    Model m = random_lex_model(rng, 3 + it % 2, it % 2 == 0);
    // Priorities with ties, weights of both signs, absolute and relative tolerances.
    m.obj_priority = prio(rng);
    m.obj_weight = wt(rng) == 0 ? 1.0 : 1.5;
    m.obj_abs_tol = static_cast<double>(tol(rng));
    for (Objective& o : m.objectives) {
      o.priority = prio(rng);
      const int w = wt(rng);
      o.weight = w == 0 ? 1.0 : static_cast<double>(w);
      o.abs_tol = static_cast<double>(tol(rng));
      o.rel_tol = it % 3 == 0 ? 0.25 : 0.0;
      o.offset = static_cast<double>(tol(rng)) - 1.0;
    }
    const std::vector<double> want = enumerate_stages(m);
    SolverParams p = quiet();
    p.deterministic = it % 4 == 1;
    const SolveResult r = solve(m, p);
    CHECK(r.status == Status::Optimal);
    CHECK_EQ(r.stages.size(), want.size());
    if (r.stages.size() != want.size()) continue;
    for (size_t k = 0; k < want.size(); ++k) CHECK_NEAR(r.stages[k].objective, want[k], 1e-6);
    CHECK(m.violation(r.x).max() <= 1e-6);
  }
}

TEST(multiobj_one_priority_is_the_weighted_sum) {
  Model m = three_objectives();
  m.obj_priority = 0;
  for (Objective& o : m.objectives) o.priority = 0;
  const SolveResult r = solve(m, quiet());
  CHECK(r.status == Status::Optimal);
  CHECK_EQ(r.stages.size(), static_cast<size_t>(1));
  // The same model with the blended objective x + 2y + 3 + 2 (z - x) - (y - z) = -x + y + 3z + 3.
  Model one = m;
  one.objectives.clear();
  one.obj = {-1.0, 1.0, 3.0};
  const SolveResult s = solve(one, quiet());
  CHECK(s.status == Status::Optimal);
  CHECK_NEAR(r.stages[0].objective, s.objective, 1e-7);
  CHECK_NEAR(r.objective_values[0] + 2.0 * r.objective_values[1] - r.objective_values[2], s.objective, 1e-7);
}

TEST(multiobj_infeasible_and_unbounded_stages) {
  // First stage infeasible: x + y <= -1 with x, y >= 0.
  {
    ModelBuilder b;
    const Index x = b.add_col("x", 1.0, 0.0, 5.0), y = b.add_col("y", 0.0, 0.0, 5.0);
    const Index r = b.add_row("r", -kInf, -1.0);
    b.add_coef(r, x, 1.0), b.add_coef(r, y, 1.0);
    Model m = b.build();
    m.obj_priority = 1;
    m.objectives.push_back(objective("second", {y}, {1.0}, 0));
    const SolveResult res = solve(m, quiet());
    CHECK(res.status == Status::Infeasible);
    CHECK(!res.has_solution);
    CHECK_EQ(res.stages.size(), static_cast<size_t>(1));
  }
  // Second stage unbounded: min x (x >= 0), then min -y with y unbounded above.
  {
    ModelBuilder b;
    const Index x = b.add_col("x", 1.0, 0.0, 5.0), y = b.add_col("y", 0.0, 0.0, kInf);
    const Index r = b.add_row("r", 0.0, kInf);
    b.add_coef(r, x, 1.0), b.add_coef(r, y, 1.0);
    Model m = b.build();
    m.obj_priority = 1;
    m.objectives.push_back(objective("second", {y}, {-1.0}, 0));
    const SolveResult res = solve(m, quiet());
    CHECK(res.status == Status::Unbounded || res.status == Status::InfeasibleOrUnbounded);
    CHECK(res.has_solution);  // the first stage's point
    CHECK_EQ(res.stages.size(), static_cast<size_t>(2));
    CHECK(res.stages[0].status == Status::Optimal);
    CHECK_NEAR(res.objective_values[0], 0.0, 1e-9);
  }
}

TEST(multiobj_time_limit_ends_the_sequence) {
  Model m = three_objectives();
  SolverParams p = quiet();
  p.time_limit = 0.0;
  const SolveResult r = solve(m, p);
  CHECK(r.status == Status::TimeLimit);
  CHECK(r.stages.size() <= static_cast<size_t>(1));
}

TEST(multiobj_validation) {
  Model m = three_objectives();
  m.objectives[0].cols.push_back(17);
  m.objectives[0].coefs.push_back(1.0);
  CHECK_THROWS(m.validate(), std::runtime_error);
  Model q = three_objectives();
  q.add_q(0, 0, 2.0);
  CHECK_THROWS(q.validate(), std::runtime_error);
  Model t = three_objectives();
  t.objectives[1].abs_tol = -1.0;
  CHECK_THROWS(t.validate(), std::runtime_error);
}

// ---- MPS files with several objectives ------------------------------------------------------------------

#include "milp/io/mps_reader.hpp"

namespace {

const char* kThreeObjectivesMps = R"(NAME t
OBJSENSE MAX
ROWS
 N  first       2  1  0.5  0.01
 N  second       1  2  1e-06  0
 N  third       1  -1  1e-06  0
 L  c1
 G  c2
COLUMNS
    MARKER    'MARKER'                 'INTORG'
    x         first     1
    x         second    -1
    x         c1        1
    x         c2        1
    MARKER    'MARKER'                 'INTEND'
    y         first     2
    y         third     1
    y         c1        1
    y         c2        -1
    MARKER    'MARKER'                 'INTORG'
    z         second    1
    z         third     -1
    z         c1        1
    MARKER    'MARKER'                 'INTEND'
RHS
    RHS1      first     -3
    RHS1      c1        7
    RHS1      c2        -2
BOUNDS
 UP BND1      x         10
 UP BND1      y         5
 BV BND1      z
ENDATA
)";

void check_same_objectives(const Model& a, const Model& b) {
  CHECK_EQ(a.objectives.size(), b.objectives.size());
  CHECK_EQ(a.obj_priority, b.obj_priority);
  CHECK_NEAR(a.obj_weight, b.obj_weight, 0.0);
  CHECK_NEAR(a.obj_abs_tol, b.obj_abs_tol, 0.0);
  CHECK_NEAR(a.obj_rel_tol, b.obj_rel_tol, 0.0);
  CHECK_NEAR(a.obj_offset, b.obj_offset, 0.0);
  CHECK(a.obj == b.obj);
  for (size_t k = 0; k < a.objectives.size() && k < b.objectives.size(); ++k) {
    const Objective &p = a.objectives[k], &q = b.objectives[k];
    CHECK_EQ(p.name, q.name);
    CHECK_EQ(p.priority, q.priority);
    CHECK_NEAR(p.weight, q.weight, 0.0);
    CHECK_NEAR(p.abs_tol, q.abs_tol, 0.0);
    CHECK_NEAR(p.rel_tol, q.rel_tol, 0.0);
    CHECK_NEAR(p.offset, q.offset, 0.0);
    CHECK(p.cols == q.cols);
    CHECK(p.coefs == q.coefs);
  }
}

}  // namespace

TEST(multiobj_mps_read) {
  const Model m = read_mps_string(kThreeObjectivesMps);
  m.validate();
  CHECK(m.sense == ObjSense::Maximize);
  CHECK_EQ(m.num_rows(), 2);
  CHECK_EQ(m.objectives.size(), static_cast<size_t>(2));
  CHECK_EQ(m.obj_name, "first");
  CHECK_EQ(m.obj_priority, 2);
  CHECK_NEAR(m.obj_abs_tol, 0.5, 0.0);
  CHECK_NEAR(m.obj_rel_tol, 0.01, 0.0);
  CHECK_NEAR(m.obj_offset, 3.0, 0.0);
  CHECK_EQ(m.objectives[0].name, "second");
  CHECK_EQ(m.objectives[0].priority, 1);
  CHECK_NEAR(m.objectives[0].weight, 2.0, 0.0);
  CHECK_NEAR(m.objectives[1].weight, -1.0, 0.0);
  CHECK_EQ(m.objectives[1].cols.size(), static_cast<size_t>(2));
  // The same model as the one built in code: the same answer.
  const SolveResult r = solve(m, quiet());
  CHECK(r.status == Status::Optimal);
  CHECK_NEAR(r.objective_values[0], 13.5, 1e-5);
  CHECK_NEAR(r.objective_values[1], -3.0, 1e-5);
  CHECK_NEAR(r.objective_values[2], 3.75, 1e-5);
}

TEST(multiobj_mps_round_trip) {
  const Model m = three_objectives();
  const std::string text = write_mps_string(m);
  const Model back = read_mps_string(text);
  back.validate();
  check_same_objectives(m, back);
  CHECK_EQ(back.obj_name, "first");
  CHECK_EQ(back.num_rows(), m.num_rows());
  // A second pass gives the same text.
  CHECK_EQ(write_mps_string(back), text);
}

TEST(multiobj_mps_plain_extra_n_row_is_still_dropped) {
  // Without the four attributes a further N row is a free row, as before.
  const Model m = read_mps_string(R"(NAME t
ROWS
 N  cost
 N  other
 L  c1
COLUMNS
    x  cost  1  other  5
    x  c1  1
RHS
    RHS  c1  4
ENDATA
)");
  CHECK(m.objectives.empty());
  CHECK_EQ(m.num_rows(), 1);
  CHECK_EQ(m.obj_priority, 0);
}

TEST(multiobj_mps_bad_attributes) {
  CHECK_THROWS(read_mps_string("NAME t\nROWS\n N  a  1.5  1  0  0\nCOLUMNS\n    x  a  1\nENDATA\n"), std::runtime_error);
  CHECK_THROWS(read_mps_string("NAME t\nROWS\n N  a  1  1  -1  0\nCOLUMNS\n    x  a  1\nENDATA\n"), std::runtime_error);
  CHECK_THROWS(read_mps_string("NAME t\nROWS\n L  a  1  1  0  0\nCOLUMNS\n    x  a  1\nENDATA\n"), std::runtime_error);
}

// ---- LP files with several objectives -------------------------------------------------------------------

#include "milp/io/lp_reader.hpp"

TEST(multiobj_lp_read) {
  const Model m = read_lp_string(R"(\ Model t
Maximize multi-objectives
  first: Priority=2 Weight=1 AbsTol=0.5 RelTol=0.01
   x + 2 y + 3 Constant0
  second: Priority=1 Weight=2 AbsTol=1e-06 RelTol=0
   - x + z
  third: Priority=1 Weight=-1 AbsTol=1e-06 RelTol=0
   y - z
Subject To
 c1: x + y + z <= 7
 c2: x - y >= -2
Bounds
 x <= 10
 y <= 5
 Constant0 = 1
Binaries
 z
Generals
 x
End
)");
  m.validate();
  CHECK(m.sense == ObjSense::Maximize);
  CHECK_EQ(m.objectives.size(), static_cast<size_t>(2));
  CHECK_EQ(m.obj_name, "first");
  CHECK_EQ(m.obj_priority, 2);
  CHECK_NEAR(m.obj_abs_tol, 0.5, 0.0);
  CHECK_NEAR(m.obj_rel_tol, 0.01, 0.0);
  CHECK_EQ(m.objectives[0].name, "second");
  CHECK_NEAR(m.objectives[0].weight, 2.0, 0.0);
  CHECK_NEAR(m.objectives[1].weight, -1.0, 0.0);
  CHECK_EQ(m.objectives[1].priority, 1);
  const SolveResult r = solve(m, quiet());
  CHECK(r.status == Status::Optimal);
  CHECK_NEAR(r.objective_values[0], 13.5, 1e-5);
  CHECK_NEAR(r.objective_values[1], -3.0, 1e-5);
  CHECK_NEAR(r.objective_values[2], 3.75, 1e-5);
}

TEST(multiobj_lp_defaults_and_constants) {
  // Attributes are optional (priority 0, weight 1, tolerances 1e-6 / 0); a constant may stand in the expression.
  const Model m = read_lp_string(R"(Minimize multi-objectives
  a: Priority=1
   x + y + 2
  b:
   - y - 1.5
Subject To
 c: x + y >= 1
Bounds
 x <= 3
 y <= 3
End
)");
  CHECK_EQ(m.objectives.size(), static_cast<size_t>(1));
  CHECK_EQ(m.obj_priority, 1);
  CHECK_NEAR(m.obj_offset, 2.0, 0.0);
  CHECK_NEAR(m.obj_abs_tol, 1e-6, 0.0);
  CHECK_EQ(m.objectives[0].priority, 0);
  CHECK_NEAR(m.objectives[0].weight, 1.0, 0.0);
  CHECK_NEAR(m.objectives[0].offset, -1.5, 0.0);
  const SolveResult r = solve(m, quiet());
  CHECK(r.status == Status::Optimal);
  CHECK_NEAR(r.objective_values[0], 3.0, 1e-5);   // x + y = 1
  CHECK_NEAR(r.objective_values[1], -2.5, 1e-5);  // y = 1 (within 1e-6 of the first stage's value)
}

TEST(multiobj_lp_errors) {
  CHECK_THROWS(read_lp_string("Minimize multi-objectives\nSubject To\n c: x >= 1\nEnd\n"), std::runtime_error);
  CHECK_THROWS(read_lp_string("Minimize multi-objectives\n a: Priority=1.5\n x\nSubject To\n c: x >= 1\nEnd\n"), std::runtime_error);
  CHECK_THROWS(read_lp_string("Minimize multi-objectives\n a: AbsTol=-1\n x\nSubject To\n c: x >= 1\nEnd\n"), std::runtime_error);
}

// ---- Other model tools -------------------------------------------------------------------------------------

#include "milp/io/anonymize.hpp"

TEST(multiobj_anonymized_model_has_the_same_values) {
  const Model m = three_objectives();
  AnonOptions opt;
  opt.seed = 7;
  const Model a = anonymize(m, opt);
  a.validate();
  CHECK_EQ(a.objectives.size(), m.objectives.size());
  CHECK_EQ(a.obj_priority, m.obj_priority);
  const SolveResult r = solve(m, quiet()), s = solve(a, quiet());
  CHECK(s.status == Status::Optimal);
  CHECK_EQ(s.objective_values.size(), r.objective_values.size());
  for (size_t k = 0; k < r.objective_values.size(); ++k) CHECK_NEAR(s.objective_values[k], r.objective_values[k], 1e-6);
}
