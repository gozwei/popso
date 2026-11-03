// Tests for the cutting-plane separators and the root cut loop. The key test
// is validity: every generated cut must be satisfied by every feasible
// (mixed-)integer point of small random problems.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <random>
#include <string>

#include "milp/core/model.hpp"
#include "milp/cuts/cut_loop.hpp"
#include "milp/io/mps_reader.hpp"
#include "milp/solver/solver.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

// Number of random instances; scaled by MILP_TEST_SCALE for long stress runs.
int iters(int base) {
#ifdef _MSC_VER
#pragma warning(suppress : 4996)
#endif
  const char* v = std::getenv("MILP_TEST_SCALE");
  return v ? base * std::max(1, std::atoi(v)) : base;
}

std::map<std::string, int> g_origin_count;

MipProblem make_problem(const Model& m) {
  MipProblem p;
  bool ok = MipProblem::from_model(m, &p);
  (void)ok;
  return p;
}

// Calls f(x_int) for every assignment of the (bounded) integer variables;
// continuous entries of x are left at 0.
template <class F>
void enumerate_ints(const MipProblem& p, F&& f) {
  const Index n = p.num_cols();
  std::vector<double> x(n, 0.0);
  for (Index j = 0; j < n; ++j) x[j] = p.is_int[j] ? p.col_lb[j] : 0.0;
  for (;;) {
    f(x);
    Index j = 0;
    for (; j < n; ++j) {
      if (!p.is_int[j]) continue;
      if (x[j] < p.col_ub[j]) {
        x[j] += 1.0;
        break;
      }
      x[j] = p.col_lb[j];
    }
    if (j == n) return;
  }
}

// Solves min cost x over the continuous variables with the integers fixed at x_int.
// Returns +inf if infeasible, -inf if unbounded.
double solve_fixed(const MipProblem& p, const std::vector<double>& x_int, const std::vector<double>& cost,
                   std::vector<double>* xout = nullptr) {
  std::vector<double> lb = p.col_lb, ub = p.col_ub;
  for (Index j = 0; j < p.num_cols(); ++j) {
    if (p.is_int[j]) lb[j] = ub[j] = x_int[j];
  }
  DualSimplex lp;
  lp.load(p.A, cost, lb, ub, p.row_lb, p.row_ub);
  LpOptions o;
  o.perturb = false;
  const LpStatus st = lp.solve(o);
  if (st == LpStatus::Infeasible) return kInf;
  if (st == LpStatus::Unbounded) return -kInf;
  if (st != LpStatus::Optimal) return kInf;
  if (xout) *xout = lp.col_values();
  return lp.objective();
}

// Max violation of `cut` over all feasible mixed-integer points (<= 0 means valid).
double cut_max_violation(const MipProblem& p, const Cut& cut) {
  std::vector<double> c(p.num_cols(), 0.0);
  for (size_t k = 0; k < cut.idx.size(); ++k) c[cut.idx[k]] -= cut.val[k];
  double worst = -kInf;
  enumerate_ints(p, [&](const std::vector<double>& x) {
    const double v = solve_fixed(p, x, c);
    if (v == kInf) return;  // infeasible assignment
    worst = std::max(worst, -v - cut.rhs);
  });
  return worst;
}

// Optimal objective (internal min sense, with offset) by enumeration.
double enumerate_optimum(const MipProblem& p) {
  double best = kInf;
  enumerate_ints(p, [&](const std::vector<double>& x) { best = std::min(best, solve_fixed(p, x, p.cost)); });
  return best == kInf || best == -kInf ? best : best + p.offset;
}

// Small random MILP: integers in [-1, 3] or binaries, bounded continuous
// variables, random rows (some knapsack-like), variable bounds and a demand
// row. Rows are built around a random point x0, so most instances are feasible.
Model random_mip(std::mt19937& rng, bool mixed, bool big = false) {
  ModelBuilder b;
  const int ni = big ? 8 + rng() % 12 : 2 + rng() % 4;
  const int nc = mixed ? (big ? 2 + rng() % 8 : 1 + rng() % 3) : 0;
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  const bool frac = rng() % 2;
  auto coef = [&](int range) {
    double v = static_cast<int>(rng() % (2 * range + 1)) - range;
    if (frac) v += std::round(U(rng) * 100) / 100.0;
    return v;
  };
  std::vector<Index> ints, conts;
  std::vector<double> x0;
  for (int j = 0; j < ni; ++j) {
    const bool bin = rng() % 2;
    const double lo = bin ? 0 : -1.0 + static_cast<int>(rng() % 2), up = bin ? 1 : 3;
    ints.push_back(b.add_col("i" + std::to_string(j), coef(10), lo, up, bin ? VarType::Binary : VarType::Integer));
    x0.push_back(lo + static_cast<int>(rng() % static_cast<unsigned>(up - lo + 1)));
  }
  for (int j = 0; j < nc; ++j) {
    const double lo = rng() % 3 == 0 ? -2.0 : 0.0, up = 5.0 + rng() % 10;
    conts.push_back(b.add_col("c" + std::to_string(j), coef(10), lo, up));
    x0.push_back(lo + (up - lo) * (0.5 + 0.5 * U(rng)));
  }
  const int nr = big ? 4 + rng() % 12 : 1 + rng() % 4;
  for (int r = 0; r < nr; ++r) {
    std::vector<std::pair<Index, double>> terms;
    for (Index j : ints) {
      if (rng() % 3 == 0) continue;
      terms.emplace_back(j, rng() % 4 == 0 ? coef(9) : std::abs(coef(9)) + 1.0);
    }
    for (Index j : conts) {
      if (rng() % 2) terms.emplace_back(j, coef(5));
    }
    if (terms.empty()) continue;
    double act = 0.0;
    for (auto& t : terms) act += t.second * x0[t.first];
    const double slack = rng() % 3 == 0 ? 0.0 : std::round(std::abs(U(rng)) * 800) / 100.0;
    const int kind = rng() % 4;
    double lo = -kInf, up = act + slack;
    if (kind == 1) lo = act - slack, up = kInf;
    if (kind == 2) lo = act - slack - std::abs(coef(5));
    if (kind == 3) lo = up = std::round(act * 100) / 100.0;
    const Index row = b.add_row("r" + std::to_string(r), lo, up);
    for (auto& t : terms) b.add_coef(row, t.first, t.second);
  }
  // Variable upper / lower bounds x_c <= u y + d, x_c >= l y.
  for (Index c : conts) {
    if (rng() % 3 == 0) continue;
    const Index y = ints[rng() % ints.size()];
    const double u = 1.0 + rng() % 8, d = rng() % 3 == 0 ? 1.0 : 0.0;
    if (rng() % 4 == 0) {
      const Index row = b.add_row("vlb", -kInf, 0.0);  // l y - x <= 0
      b.add_coef(row, c, -1.0);
      b.add_coef(row, y, 1.0 + rng() % 3);
    } else {
      const Index row = b.add_row("vub", rng() % 2 ? -kInf : -20.0, d);
      b.add_coef(row, c, 1.0);
      b.add_coef(row, y, -u);
    }
  }
  // A demand-like row to make fixed-charge structure binding.
  if (!conts.empty() && rng() % 2) {
    const Index row = b.add_row("dem", 1.0 + rng() % 6, kInf);
    for (Index c : conts) b.add_coef(row, c, 1.0);
  }
  b.model().sense = rng() % 2 ? ObjSense::Minimize : ObjSense::Maximize;
  return b.build();
}

// Small random fixed-charge network: flows x_a >= 0 with x_a <= u_a y_a (y binary or
// integer "modules"), node balances with supplies/demands (equalities or inequalities),
// sometimes a minimum flow x_a >= l_a y_a and a budget row on the y. Built around a
// feasible point (flow along a random spanning arborescence plus random extra flow).
Model random_fcnf(std::mt19937& rng, bool big = false) {
  ModelBuilder b;
  const int nn = big ? 5 + rng() % 6 : 3 + rng() % 2;
  const int na = big ? nn + 4 + rng() % 10 : nn + rng() % 3;
  std::vector<std::pair<int, int>> arcs;
  for (int v = 1; v < nn; ++v) arcs.emplace_back(static_cast<int>(rng() % v), v);  // arborescence from 0
  while (static_cast<int>(arcs.size()) < na) {
    const int i = rng() % nn, j = rng() % nn;
    if (i != j) arcs.emplace_back(i, j);
  }
  std::vector<double> flow(arcs.size(), 0.0), dem(nn, 0.0);
  for (size_t a = 0; a < arcs.size(); ++a) {
    flow[a] = rng() % 3 == 0 ? 0.0 : static_cast<double>(rng() % 7);
    dem[arcs[a].second] += flow[a];
    dem[arcs[a].first] -= flow[a];
  }
  const bool modules = rng() % 3 == 0;
  std::vector<Index> x, y;
  for (size_t a = 0; a < arcs.size(); ++a) {
    const double u = modules ? 2.0 + rng() % 4 : std::max(flow[a], 1.0) + rng() % 12;
    const double ymax = modules ? 3.0 : 1.0;
    x.push_back(b.add_col("x" + std::to_string(a), 0.1 * (rng() % 10), 0.0, rng() % 4 == 0 ? 20.0 : kInf));
    y.push_back(b.add_col("y" + std::to_string(a), 1.0 + rng() % 20, 0.0, ymax,
                          modules ? VarType::Integer : VarType::Binary));
    if (modules && flow[a] > ymax * u) flow[a] = ymax * u;  // keep the point roughly feasible
    const Index r = b.add_row("vub" + std::to_string(a), -kInf, 0.0);
    b.add_coef(r, x[a], 1.0), b.add_coef(r, y[a], -u);
    if (rng() % 5 == 0) {
      const Index r2 = b.add_row("vlb" + std::to_string(a), -kInf, 0.0);  // x >= l y
      b.add_coef(r2, x[a], -1.0), b.add_coef(r2, y[a], 1.0);
    }
  }
  for (int v = 0; v < nn; ++v) {
    // inflow - outflow (=|>=|<=) demand
    const int kind = rng() % 4;
    const double d = dem[v];
    const Index r = b.add_row("bal" + std::to_string(v), kind == 2 ? -kInf : d, kind == 1 ? kInf : d);
    for (size_t a = 0; a < arcs.size(); ++a) {
      if (arcs[a].second == v) b.add_coef(r, x[a], 1.0);
      if (arcs[a].first == v) b.add_coef(r, x[a], -1.0);
    }
  }
  if (rng() % 3 == 0) {
    const Index r = b.add_row("budget", -kInf, static_cast<double>(arcs.size()) - 1.0);
    for (Index j : y) b.add_coef(r, j, 1.0);
  }
  b.model().sense = ObjSense::Minimize;
  return b.build();
}

// Small random multi-commodity network design (n5-3 like): per-commodity flows x_ak >= 0 on a
// bidirected tree plus extra arcs, single-commodity balances (purely continuous rows), arc
// capacity rows sum_k x_ak <= c1 y_a1 (+ c2 y_a2) with integer modules y, sometimes an
// aggregated arc flow column f_a = sum_k x_ak with f_a <= capacity instead.
Model random_mcnd(std::mt19937& rng) {
  ModelBuilder b;
  const int nn = 3 + rng() % 2;
  std::vector<std::pair<int, int>> arcs;
  for (int v = 1; v < nn; ++v) {
    const int u = rng() % v;
    arcs.emplace_back(u, v);
    if (rng() % 2) arcs.emplace_back(v, u);
  }
  while (arcs.size() < static_cast<size_t>(nn) + 1) {
    const int i = rng() % nn, j = rng() % nn;
    if (i != j) arcs.emplace_back(i, j);
  }
  const int nk = 2 + rng() % 2;
  const double c1 = 2.0 + rng() % 3, c2 = 5.0 + rng() % 5;
  const bool two_modules = rng() % 2;
  const bool aggregated = rng() % 3 == 0;
  std::vector<std::vector<Index>> x(arcs.size());
  for (size_t a = 0; a < arcs.size(); ++a) {
    const Index y1 = b.add_col("y1_" + std::to_string(a), 3.0 + rng() % 5, 0.0, two_modules ? 1.0 : 2.0,
                               VarType::Integer);
    const Index y2 = two_modules ? b.add_col("y2_" + std::to_string(a), 6.0 + rng() % 5, 0.0, 1.0, VarType::Integer) : -1;
    const Index cap = b.add_row("cap" + std::to_string(a), -kInf, 0.0);
    for (int k = 0; k < nk; ++k) x[a].push_back(b.add_col("x" + std::to_string(a) + "_" + std::to_string(k), 0.1 * (rng() % 5), 0.0, kInf));
    if (aggregated) {
      const Index f = b.add_col("f" + std::to_string(a), 0.0, 0.0, kInf);
      const Index def = b.add_row("def" + std::to_string(a), 0.0, 0.0);
      b.add_coef(def, f, 1.0);
      for (Index j : x[a]) b.add_coef(def, j, -1.0);
      b.add_coef(cap, f, 1.0);
    } else {
      for (Index j : x[a]) b.add_coef(cap, j, 1.0);
    }
    b.add_coef(cap, y1, -c1);
    if (y2 >= 0) b.add_coef(cap, y2, -c2);
  }
  for (int k = 0; k < nk; ++k) {
    const int o = rng() % nn;
    int d = rng() % nn;
    if (d == o) d = (o + 1) % nn;
    const double dem = 1.0 + rng() % 4;
    for (int v = 0; v < nn; ++v) {
      // outflow - inflow = supply (+dem at the origin, -dem at the destination)
      const double s = v == o ? dem : v == d ? -dem : 0.0;
      const int kind = rng() % 5;  // mostly equalities, sometimes >= at the origin
      const Index r = b.add_row("bal" + std::to_string(k) + "_" + std::to_string(v), s, kind == 0 && v == o ? kInf : s);
      for (size_t a = 0; a < arcs.size(); ++a) {
        if (arcs[a].first == v) b.add_coef(r, x[a][k], 1.0);
        if (arcs[a].second == v) b.add_coef(r, x[a][k], -1.0);
      }
    }
  }
  b.model().sense = ObjSense::Minimize;
  return b.build();
}

// Runs the cut loop with the given separators and checks every cut.
void check_cuts_valid(const MipProblem& p, const CutLoopOptions& opt, int* ncuts, int* ninvalid) {
  // Flow cover cuts are opt-in in the solver; the validity tests always include them.
#ifdef _WIN32
  _putenv_s("MILP_FLOWCOVER", "1");
  _putenv_s("MILP_MIR_INTAGG", "1");
  _putenv_s("MILP_MIR_INTAGG_GENERAL", "0");  // integer path aggregation through binaries too
#else
  setenv("MILP_FLOWCOVER", "1", 1);
  setenv("MILP_MIR_INTAGG", "1", 1);
  setenv("MILP_MIR_INTAGG_GENERAL", "0", 1);  // integer path aggregation through binaries too
#endif
  DualSimplex lp;
  lp.load(p.A, p.cost, p.col_lb, p.col_ub, p.row_lb, p.row_ub);
  Logger log(0);
  Timer timer;
  std::vector<Cut> cuts;
  cut_loop(p, lp, opt, log, timer, kInf, &cuts);
  for (const Cut& c : cuts) {
    ++*ncuts;
    ++g_origin_count[c.origin];
    const double viol = cut_max_violation(p, c);
    if (viol > 1e-6 * std::max(1.0, std::abs(c.rhs))) {
      ++*ninvalid;
      std::fprintf(stderr, "  invalid %s cut (violation %g):", c.origin, viol);
      for (size_t k = 0; k < c.idx.size(); ++k) std::fprintf(stderr, " %+g x%d", c.val[k], c.idx[k]);
      std::fprintf(stderr, " <= %g\n", c.rhs);
    }
  }
}

CutLoopOptions only(bool gomory, bool cover, bool mir, bool clique = false) {
  CutLoopOptions o;
  o.gomory = gomory, o.cover = cover, o.mir = mir, o.clique = clique;
  o.max_rounds = 10;
  o.stall_rounds = 100;
  return o;
}

}  // namespace

TEST(cuts_cover_knapsack) {
  // max x1 + x2 + x3 + x4  s.t. 5x1 + 5x2 + 5x3 + 9x4 <= 9: cover x1+x2+x3 (+ lifted x4) <= 1.
  ModelBuilder b;
  for (int j = 0; j < 4; ++j) b.add_col("x" + std::to_string(j), 1.0, 0, 1, VarType::Binary);
  Index r = b.add_row("k", -kInf, 9.0);
  for (int j = 0; j < 4; ++j) b.add_coef(r, j, j == 3 ? 9.0 : 5.0);
  b.model().sense = ObjSense::Maximize;
  MipProblem p = make_problem(b.build());
  DualSimplex lp;
  lp.load(p.A, p.cost, p.col_lb, p.col_ub, p.row_lb, p.row_ub);
  Logger log(0);
  Timer timer;
  std::vector<Cut> cuts;
  CutLoopStats st = cut_loop(p, lp, only(false, true, false), log, timer, kInf, &cuts);
  CHECK_NEAR(st.bound_before, -1.8, 1e-9);
  CHECK_NEAR(st.bound_after, -1.0, 1e-6);
  REQUIRE(!cuts.empty());
  for (const Cut& c : cuts) CHECK(cut_max_violation(p, c) <= 1e-9);
}

TEST(cuts_clique_from_conflicts) {
  // max x0 + x1 + x2  s.t. x0 + x1 <= 1, x1 + x2 <= 1, x0 + x2 <= 1: LP 1.5, clique x0 + x1 + x2 <= 1.
  ModelBuilder b;
  for (int j = 0; j < 3; ++j) b.add_col("x" + std::to_string(j), 1.0, 0, 1, VarType::Binary);
  for (int r = 0; r < 3; ++r) {
    Index row = b.add_row("p", -kInf, 1.0);
    b.add_coef(row, r, 1.0);
    b.add_coef(row, (r + 1) % 3, 1.0);
  }
  b.model().sense = ObjSense::Maximize;
  MipProblem p = make_problem(b.build());
  DualSimplex lp;
  lp.load(p.A, p.cost, p.col_lb, p.col_ub, p.row_lb, p.row_ub);
  Logger log(0);
  Timer timer;
  std::vector<Cut> cuts;
  CutLoopStats st = cut_loop(p, lp, only(false, false, false, true), log, timer, kInf, &cuts);
  CHECK_NEAR(st.bound_before, -1.5, 1e-9);
  CHECK_NEAR(st.bound_after, -1.0, 1e-6);
  for (const Cut& c : cuts) CHECK(cut_max_violation(p, c) <= 1e-9);
}

TEST(cuts_clique_extended_to_maximal) {
  // max x0 + x1 + x2 - 0.1 x3, all pairs conflict (x3 with each other one too): LP x0 = x1 = x2 = 1/2,
  // x3 = 0. The violated clique x0 + x1 + x2 <= 1 is extended by x3 (LP value 0) to a maximal one.
  ModelBuilder b;
  for (int j = 0; j < 4; ++j) b.add_col("x" + std::to_string(j), j < 3 ? 1.0 : -0.1, 0, 1, VarType::Binary);
  for (int u = 0; u < 4; ++u) {
    for (int v = u + 1; v < 4; ++v) {
      Index row = b.add_row("p", -kInf, 1.0);
      b.add_coef(row, u, 1.0);
      b.add_coef(row, v, 1.0);
    }
  }
  b.model().sense = ObjSense::Maximize;
  MipProblem p = make_problem(b.build());
  DualSimplex lp;
  lp.load(p.A, p.cost, p.col_lb, p.col_ub, p.row_lb, p.row_ub);
  Logger log(0);
  Timer timer;
  std::vector<Cut> cuts;
  CutLoopStats st = cut_loop(p, lp, only(false, false, false, true), log, timer, kInf, &cuts);
  CHECK_NEAR(st.bound_before, -1.5, 1e-9);
  CHECK_NEAR(st.bound_after, -1.0, 1e-6);
  REQUIRE(!cuts.empty());
  size_t clique_len = 0;
  for (const Cut& c : cuts) {
    if (std::string(c.origin) == "clique") clique_len = std::max(clique_len, c.idx.size());
  }
  CHECK_EQ(clique_len, 4u);
  for (const Cut& c : cuts) CHECK(cut_max_violation(p, c) <= 1e-9);
}

TEST(cuts_gomory_textbook) {
  // max x2  s.t. 3x1 + 2x2 <= 6, -3x1 + 2x2 <= 0, x integer >= 0. LP (1, 1.5), IP value 1.
  ModelBuilder b;
  Index x1 = b.add_col("x1", 0.0, 0, kInf, VarType::Integer);
  Index x2 = b.add_col("x2", 1.0, 0, kInf, VarType::Integer);
  Index r0 = b.add_row("r0", -kInf, 6.0), r1 = b.add_row("r1", -kInf, 0.0);
  b.add_coef(r0, x1, 3.0), b.add_coef(r0, x2, 2.0), b.add_coef(r1, x1, -3.0), b.add_coef(r1, x2, 2.0);
  b.model().sense = ObjSense::Maximize;
  MipProblem p = make_problem(b.build());
  p.col_ub = {3.0, 4.0};  // bounded for the enumeration (not binding)
  DualSimplex lp;
  lp.load(p.A, p.cost, p.col_lb, p.col_ub, p.row_lb, p.row_ub);
  Logger log(0);
  Timer timer;
  std::vector<Cut> cuts;
  CutLoopStats st = cut_loop(p, lp, only(true, false, false), log, timer, kInf, &cuts);
  CHECK_NEAR(st.bound_before, -1.5, 1e-9);
  CHECK(st.bound_after > -1.5 + 1e-3);
  CHECK(st.bound_after <= -1.0 + 1e-6);
  REQUIRE(!cuts.empty());
  for (const Cut& c : cuts) CHECK(cut_max_violation(p, c) <= 1e-6);
}

TEST(cuts_mir_fixed_charge) {
  // min y + 0.01 x  s.t. x >= 4, x <= 10 y, y binary: MIR on the aggregation gives y >= 1.
  ModelBuilder b;
  Index x = b.add_col("x", 0.01, 0, 10);
  Index y = b.add_col("y", 1.0, 0, 1, VarType::Binary);
  Index r0 = b.add_row("dem", 4.0, kInf), r1 = b.add_row("vub", -kInf, 0.0);
  b.add_coef(r0, x, 1.0), b.add_coef(r1, x, 1.0), b.add_coef(r1, y, -10.0);
  MipProblem p = make_problem(b.build());
  DualSimplex lp;
  lp.load(p.A, p.cost, p.col_lb, p.col_ub, p.row_lb, p.row_ub);
  Logger log(0);
  Timer timer;
  std::vector<Cut> cuts;
  CutLoopStats st = cut_loop(p, lp, only(false, false, true), log, timer, kInf, &cuts);
  CHECK_NEAR(st.bound_before, 0.44, 1e-9);
  CHECK_NEAR(st.bound_after, 1.04, 1e-6);
  for (const Cut& c : cuts) CHECK(cut_max_violation(p, c) <= 1e-9);
}

TEST(cuts_clean_relaxes_tiny_coefficients) {
  std::vector<double> lb{0.0, -2.0, 1.0}, ub{4.0, 3.0, 1.0};
  Cut c;
  c.idx = {0, 1, 2};
  c.val = {1.0, -1e-8, 5.0};  // x2 fixed at 1, tiny coefficient on x1
  c.rhs = 10.0;
  REQUIRE(clean_cut(c, lb, ub, 100));
  REQUIRE(c.idx.size() == 1u);
  CHECK_EQ(c.idx[0], 0);
  // x0 <= 10 - 5 - (-1e-8 * 3) (+ safety), scaled by 1/2 (power of two to max |a| in [0.5, 1)).
  CHECK(c.rhs * 2.0 >= 5.0 + 3e-8 - 1e-12);
  CHECK(c.rhs * 2.0 <= 5.0 + 1e-7);
  // Tiny coefficient on a variable without a finite bound in the needed direction: rejected.
  std::vector<double> lb2{0.0, -kInf}, ub2{4.0, 3.0};
  Cut d;
  d.idx = {0, 1};
  d.val = {1.0, 1e-8};
  d.rhs = 1.0;
  CHECK(!clean_cut(d, lb2, ub2, 100));
}

TEST(cuts_random_validity_pure_integer) {
  std::mt19937 rng(11);
  int ncuts = 0, ninvalid = 0;
  for (int t = 0; t < iters(500); ++t) {
    MipProblem p = make_problem(random_mip(rng, false));
    check_cuts_valid(p, only(true, false, false), &ncuts, &ninvalid);
    check_cuts_valid(p, only(false, true, false), &ncuts, &ninvalid);
    check_cuts_valid(p, only(false, false, true), &ncuts, &ninvalid);
    check_cuts_valid(p, only(false, false, false, true), &ncuts, &ninvalid);
    check_cuts_valid(p, only(true, true, true, true), &ncuts, &ninvalid);
  }
  std::printf("  pure integer: %d cuts checked, %d invalid (", ncuts, ninvalid);
  for (auto& [k, v] : g_origin_count) std::printf(" %s %d", k.c_str(), v);
  std::printf(" )\n");
  g_origin_count.clear();
  CHECK(ncuts > 100);
  CHECK_EQ(ninvalid, 0);
}

TEST(cuts_random_validity_mixed) {
  std::mt19937 rng(23);
  int ncuts = 0, ninvalid = 0;
  for (int t = 0; t < iters(500); ++t) {
    MipProblem p = make_problem(random_mip(rng, true));
    check_cuts_valid(p, only(true, false, false), &ncuts, &ninvalid);
    check_cuts_valid(p, only(false, true, false), &ncuts, &ninvalid);
    check_cuts_valid(p, only(false, false, true), &ncuts, &ninvalid);
    check_cuts_valid(p, only(false, false, false, true), &ncuts, &ninvalid);
    check_cuts_valid(p, only(true, true, true, true), &ncuts, &ninvalid);
  }
  std::printf("  mixed: %d cuts checked, %d invalid (", ncuts, ninvalid);
  for (auto& [k, v] : g_origin_count) std::printf(" %s %d", k.c_str(), v);
  std::printf(" )\n");
  g_origin_count.clear();
  CHECK(ncuts > 100);
  CHECK_EQ(ninvalid, 0);
}

// Every probing implication and literal conflict holds at every feasible mixed-integer point.
TEST(cuts_probing_implications_valid) {
  std::mt19937 rng(37);
  long long pairs = 0, imps = 0;
  int bad = 0;
  for (int t = 0; t < iters(300); ++t) {
    MipProblem p = make_problem(random_mip(rng, t % 2 == 1));
    CutData d;
    d.build(p);
    d.probe(p, nullptr, 1000000);
    const std::vector<double> zero(p.num_cols(), 0.0);
    enumerate_ints(p, [&](const std::vector<double>& x) {
      if (solve_fixed(p, x, zero) == kInf) return;  // infeasible assignment
      for (Index l = 0; l + 1 < static_cast<Index>(d.conflict_start.size()); ++l) {
        for (Index k = d.conflict_start[l]; k < d.conflict_start[l + 1]; ++k) {
          const Index m = d.conflict_lit[k];
          auto val = [&](Index lit) { return (lit & 1) ? 1.0 - x[lit / 2] : x[lit / 2]; };
          if (val(l) + val(m) > 1.5) ++bad;
        }
      }
      for (const ProbeImplication& im : d.probe_imps) {
        const double b = im.v0 + (im.v1 - im.v0) * x[im.y];
        if (im.upper ? x[im.x] > b + 1e-6 : x[im.x] < b - 1e-6) ++bad;
      }
    });
    pairs += static_cast<long long>(d.conflict_lit.size()) / 2;
    imps += static_cast<long long>(d.probe_imps.size());
  }
  std::printf("  %lld literal conflicts, %lld implied bounds checked\n", pairs, imps);
  CHECK(pairs > 50);
  CHECK(imps > 10);
  CHECK_EQ(bad, 0);
}

namespace {

// Small random unit commitment (unitcal like): commitment binaries y_gt, power split into segments
// p_gtk with sum_k p_gtk <= P_g y_gt (several continuous columns on one binary), a minimum output
// row, a reserve column r_gt in r + sum_k p <= P_g y (VUB of r only through a longer row), startup
// s_gt >= y_t - y_{t-1}, ramping rows on the segment sums and per-period demand / reserve rows.
Model random_unit_commitment(std::mt19937& rng) {
  ModelBuilder b;
  const int G = 2, T = 2 + rng() % 2;
  std::vector<std::vector<Index>> y(G), r(G);
  std::vector<std::vector<std::vector<Index>>> pw(G);
  double total_cap = 0.0;
  for (int g = 0; g < G; ++g) {
    const double P = 5.0 + rng() % 10, Pmin = 1.0 + rng() % 3, ramp = 2.0 + rng() % 6;
    const int K = 2 + rng() % 2;
    total_cap += P;
    pw[g].resize(T);
    for (int t = 0; t < T; ++t) {
      const std::string gt = std::to_string(g) + "_" + std::to_string(t);
      y[g].push_back(b.add_col("y" + gt, 2.0 + rng() % 10, 0, 1, VarType::Binary));
      r[g].push_back(b.add_col("r" + gt, 0.1 * (rng() % 3), 0.0, rng() % 2 ? kInf : 0.5 * P));
      const Index cap = b.add_row("cap" + gt, -kInf, 0.0), mn = b.add_row("min" + gt, 0.0, kInf);
      const Index res = b.add_row("res" + gt, -kInf, 0.0);
      b.add_coef(cap, y[g][t], -P), b.add_coef(mn, y[g][t], -Pmin), b.add_coef(res, y[g][t], -P);
      b.add_coef(res, r[g][t], 1.0);
      for (int k = 0; k < K; ++k) {
        const double seg = std::round(P / K + (rng() % 3)) ;
        const Index pk = b.add_col("p" + gt + "_" + std::to_string(k), 1.0 + rng() % 4 + 0.5 * k, 0.0, seg);
        pw[g][t].push_back(pk);
        b.add_coef(cap, pk, 1.0), b.add_coef(mn, pk, 1.0), b.add_coef(res, pk, 1.0);
      }
      if (t > 0) {
        const Index s = b.add_col("s" + gt, 1.0 + rng() % 5, 0.0, 1.0);
        const Index su = b.add_row("su" + gt, -kInf, 0.0);  // y_t - y_{t-1} - s <= 0
        b.add_coef(su, y[g][t], 1.0), b.add_coef(su, y[g][t - 1], -1.0), b.add_coef(su, s, -1.0);
        const Index rp = b.add_row("ramp" + gt, -ramp, ramp);
        for (Index pk : pw[g][t]) b.add_coef(rp, pk, 1.0);
        for (Index pk : pw[g][t - 1]) b.add_coef(rp, pk, -1.0);
      }
    }
  }
  for (int t = 0; t < T; ++t) {
    const double dem = std::round((0.2 + 0.6 * (rng() % 100) / 100.0) * total_cap) + 0.5 * (rng() % 2);
    const Index d = b.add_row("dem" + std::to_string(t), dem, kInf);
    const Index rr = b.add_row("rreq" + std::to_string(t), 1.0 + rng() % 4, kInf);
    for (int g = 0; g < G; ++g) {
      for (Index pk : pw[g][t]) b.add_coef(d, pk, 1.0);
      b.add_coef(rr, r[g][t], 1.0);
    }
  }
  b.model().sense = ObjSense::Minimize;
  return b.build();
}

}  // namespace

// Variable bounds in the cut data store (two-variable rows, implied by longer rows, probing) must
// hold at every feasible point, and the c-MIR / implied bound / flow cover cuts using them must be
// valid (enumeration over the binaries).
TEST(cuts_implied_variable_bounds_valid) {
  std::mt19937 rng(4242);
  long long vbs = 0;
  int bad = 0, ncuts = 0, ninvalid = 0, models = 0;
  for (int t = 0; t < iters(200); ++t) {
    MipProblem p = make_problem(t % 2 == 0 ? random_unit_commitment(rng) : random_mip(rng, true));
    CutData d;
    d.build(p);
    d.probe(p, nullptr, 1000000);
    const Index n = p.num_cols();
    for (Index j = 0; j < n; ++j) {
      for (int side = 0; side < 2; ++side) {
        for (const VarBound& vb : side ? d.vlb[j] : d.vub[j]) {
          Cut c;  // x - coef y <= constant (VUB) or -x + coef y <= -constant (VLB)
          const double s = side ? -1.0 : 1.0;
          c.idx = {j, vb.y};
          c.val = {s, -s * vb.coef};
          c.rhs = s * vb.constant;
          ++vbs;
          const double viol = cut_max_violation(p, c);
          if (viol > 1e-6 * std::max(1.0, std::abs(c.rhs))) {
            ++bad;
            std::fprintf(stderr, "  invalid %s x%d %s %g + %g y%d (violation %g)\n", side ? "VLB" : "VUB", j,
                         side ? ">=" : "<=", vb.constant, vb.coef, vb.y, viol);
          }
        }
      }
    }
    if (enumerate_optimum(p) == kInf) continue;
    ++models;
    CutLoopOptions o = only(false, false, true);
    o.implied_bound = true;
    check_cuts_valid(p, o, &ncuts, &ninvalid);
    check_cuts_valid(p, only(true, true, true, true), &ncuts, &ninvalid);
  }
  std::printf("  %lld variable bounds checked (%d invalid); %d models, %d cuts checked, %d invalid (", vbs, bad, models,
              ncuts, ninvalid);
  for (auto& [k, v] : g_origin_count) std::printf(" %s %d", k.c_str(), v);
  std::printf(" )\n");
  g_origin_count.clear();
  CHECK(vbs > 500);
  CHECK(ncuts > 100);
  CHECK_EQ(bad, 0);
  CHECK_EQ(ninvalid, 0);
}

namespace {

CutLoopOptions only_zerohalf() {
  CutLoopOptions o = only(false, false, false, false);
  o.implied_bound = false;
  o.zerohalf = true;
  return o;
}

// Small pure/mixed integer problems with integral coefficients (the zero-half case):
// odd-cycle like pair rows, equations with small coefficients, ranged rows, general
// integers with shifted bounds and a few continuous columns.
Model random_zerohalf_mip(std::mt19937& rng) {
  ModelBuilder b;
  const int ni = 3 + rng() % 4, nc = rng() % 3 == 0 ? 1 + rng() % 2 : 0;
  std::vector<Index> cols;
  std::vector<double> x0;
  for (int j = 0; j < ni; ++j) {
    const bool bin = rng() % 2;
    const double lo = bin ? 0.0 : -1.0 + static_cast<int>(rng() % 3), up = bin ? 1.0 : lo + 1 + rng() % 3;
    cols.push_back(b.add_col("i" + std::to_string(j), static_cast<int>(rng() % 11) - 5.0, lo, up,
                             bin ? VarType::Binary : VarType::Integer));
    x0.push_back(lo + static_cast<int>(rng() % static_cast<unsigned>(up - lo + 1)));
  }
  for (int j = 0; j < nc; ++j) {
    cols.push_back(b.add_col("c" + std::to_string(j), static_cast<int>(rng() % 5) - 2.0, 0.0, 1.0 + rng() % 3));
    x0.push_back(0.5);
  }
  const int nr = 2 + rng() % 5;
  for (int r = 0; r < nr; ++r) {
    std::vector<std::pair<Index, double>> terms;
    for (Index j = 0; j < static_cast<Index>(cols.size()); ++j) {
      if (rng() % 2) continue;
      double a = static_cast<int>(rng() % 7) - 3.0;
      if (a == 0.0) a = 1.0;
      if (j >= ni && rng() % 2) a *= 0.5;  // non-integral continuous coefficients are allowed
      terms.emplace_back(j, a);
    }
    if (terms.empty()) continue;
    double act = 0.0;
    for (auto& t : terms) act += t.second * x0[t.first];
    act = std::floor(act);
    const int kind = rng() % 4;
    const double slack = rng() % 3;
    double lo = -kInf, up = act + slack + (rng() % 2);
    if (kind == 1) lo = act - slack, up = kInf;
    if (kind == 2) lo = act - slack - 1;
    if (kind == 3 && nc == 0) lo = up = act;
    // Some rows scaled by 1/2 or 1/3 (integral only after scaling).
    const double sc = rng() % 4 == 0 ? 0.5 : rng() % 6 == 0 ? 1.0 / 3.0 : 1.0;
    const Index row = b.add_row("r" + std::to_string(r), is_finite(lo) ? lo * sc : lo, is_finite(up) ? up * sc : up);
    for (auto& t : terms) b.add_coef(row, t.first, t.second * sc);
  }
  b.model().sense = rng() % 2 ? ObjSense::Minimize : ObjSense::Maximize;
  return b.build();
}

}  // namespace

TEST(cuts_zerohalf_odd_cycle) {
  // max x0 + x1 + x2  s.t. x0 + x1 <= 1, x1 + x2 <= 1, x0 + x2 <= 1: LP 1.5; the sum of the three
  // rows halved gives x0 + x1 + x2 <= 1.
  ModelBuilder b;
  for (int j = 0; j < 3; ++j) b.add_col("x" + std::to_string(j), 1.0, 0, 1, VarType::Binary);
  for (int r = 0; r < 3; ++r) {
    Index row = b.add_row("p", -kInf, 1.0);
    b.add_coef(row, r, 1.0);
    b.add_coef(row, (r + 1) % 3, 1.0);
  }
  b.model().sense = ObjSense::Maximize;
  MipProblem p = make_problem(b.build());
  DualSimplex lp;
  lp.load(p.A, p.cost, p.col_lb, p.col_ub, p.row_lb, p.row_ub);
  Logger log(0);
  Timer timer;
  std::vector<Cut> cuts;
  CutLoopStats st = cut_loop(p, lp, only_zerohalf(), log, timer, kInf, &cuts);
  CHECK_NEAR(st.bound_before, -1.5, 1e-9);
  CHECK_NEAR(st.bound_after, -1.0, 1e-6);
  REQUIRE(!cuts.empty());
  for (const Cut& c : cuts) CHECK(cut_max_violation(p, c) <= 1e-9);
}

TEST(cuts_zerohalf_random_validity) {
  std::mt19937 rng(77);
  int ncuts = 0, ninvalid = 0;
  for (int t = 0; t < iters(1500); ++t) {
    MipProblem p = make_problem(random_zerohalf_mip(rng));
    check_cuts_valid(p, only_zerohalf(), &ncuts, &ninvalid);
    if (t % 3 == 0) {
      MipProblem q = make_problem(random_mip(rng, t % 2 == 0));
      check_cuts_valid(q, only_zerohalf(), &ncuts, &ninvalid);
    }
  }
  std::printf("  zerohalf: %d cuts checked, %d invalid\n", ncuts, ninvalid);
  g_origin_count.clear();
  CHECK(ncuts > 50);
  CHECK_EQ(ninvalid, 0);
}

TEST(cuts_random_validity_fixed_charge_network) {
  std::mt19937 rng(57);
  int ncuts = 0, ninvalid = 0;
  for (int t = 0; t < iters(400); ++t) {
    MipProblem p = make_problem(random_fcnf(rng));
    check_cuts_valid(p, only(false, false, true), &ncuts, &ninvalid);
    check_cuts_valid(p, only(true, true, true, true), &ncuts, &ninvalid);
  }
  std::printf("  fixed-charge network: %d cuts checked, %d invalid (", ncuts, ninvalid);
  for (auto& [k, v] : g_origin_count) std::printf(" %s %d", k.c_str(), v);
  std::printf(" )\n");
  g_origin_count.clear();
  CHECK(ncuts > 300);
  CHECK_EQ(ninvalid, 0);
}

TEST(cuts_random_validity_multicommodity_network) {
  // c-MIR paths from purely continuous balance rows through module capacity rows.
  std::mt19937 rng(91);
  int ncuts = 0, ninvalid = 0, feasible = 0;
  for (int t = 0; t < iters(150); ++t) {
    Model m = random_mcnd(rng);
    MipProblem p = make_problem(m);
    if (enumerate_optimum(p) == kInf) continue;
    ++feasible;
    check_cuts_valid(p, only(false, false, true), &ncuts, &ninvalid);
    check_cuts_valid(p, only(true, true, true, true), &ncuts, &ninvalid);
  }
  std::printf("  multi-commodity network: %d feasible, %d cuts checked, %d invalid (", feasible, ncuts, ninvalid);
  for (auto& [k, v] : g_origin_count) std::printf(" %s %d", k.c_str(), v);
  std::printf(" )\n");
  g_origin_count.clear();
  CHECK(ncuts > 100);
  CHECK_EQ(ninvalid, 0);
}

TEST(cuts_mir_multicommodity_cut_set) {
  // Two commodities from node 0 to node 1 (demands 3 and 4) over one arc with modules of size 5:
  // x_1 + x_2 <= 5 y, x_1 = 3, x_2 = 4. The LP takes y = 1.4; the balance rows aggregated with the
  // capacity row give 5 y >= 7, and its c-MIR y >= 2.
  ModelBuilder b;
  const Index y = b.add_col("y", 1.0, 0, 10, VarType::Integer);
  const Index x1 = b.add_col("x1", 0.0, 0, kInf), x2 = b.add_col("x2", 0.0, 0, kInf);
  const Index cap = b.add_row("cap", -kInf, 0.0);
  b.add_coef(cap, x1, 1.0), b.add_coef(cap, x2, 1.0), b.add_coef(cap, y, -5.0);
  const Index b1 = b.add_row("bal1", 3.0, 3.0), b2 = b.add_row("bal2", 4.0, 4.0);
  b.add_coef(b1, x1, 1.0), b.add_coef(b2, x2, 1.0);
  MipProblem p = make_problem(b.build());
  DualSimplex lp;
  lp.load(p.A, p.cost, p.col_lb, p.col_ub, p.row_lb, p.row_ub);
  Logger log(0);
  Timer timer;
  std::vector<Cut> cuts;
  CutLoopStats st = cut_loop(p, lp, only(false, false, true), log, timer, kInf, &cuts);
  CHECK_NEAR(st.bound_before, 1.4, 1e-9);
  CHECK_NEAR(st.bound_after, 2.0, 1e-6);
  for (const Cut& c : cuts) CHECK(cut_max_violation(p, c) <= 1e-9);
}

TEST(cuts_mir_single_node_flow_cover) {
  // Demand node with inflow arcs from a transshipment node: x1 + x2 - x3 = 2, x_a <= 10 y_a.
  // The LP takes y1 = 0.2 + ...; the flow cover x1 <= 2 y1 + x3 (+ x2 terms) must close it.
  ModelBuilder b;
  Index x1 = b.add_col("x1", 0.0, 0, kInf), x2 = b.add_col("x2", 0.0, 0, kInf), x3 = b.add_col("x3", 0.0, 0, kInf);
  Index y1 = b.add_col("y1", 5.0, 0, 1, VarType::Binary), y2 = b.add_col("y2", 7.0, 0, 1, VarType::Binary);
  Index y3 = b.add_col("y3", 1.0, 0, 1, VarType::Binary);
  Index r = b.add_row("bal", 2.0, 2.0);
  b.add_coef(r, x1, 1.0), b.add_coef(r, x2, 1.0), b.add_coef(r, x3, -1.0);
  for (auto [xa, ya] : {std::pair{x1, y1}, {x2, y2}, {x3, y3}}) {
    Index v = b.add_row("vub", -kInf, 0.0);
    b.add_coef(v, xa, 1.0), b.add_coef(v, ya, -10.0);
  }
  MipProblem p = make_problem(b.build());
  DualSimplex lp;
  lp.load(p.A, p.cost, p.col_lb, p.col_ub, p.row_lb, p.row_ub);
  Logger log(0);
  Timer timer;
  std::vector<Cut> cuts;
  CutLoopStats st = cut_loop(p, lp, only(false, false, true), log, timer, kInf, &cuts);
  CHECK_NEAR(st.bound_before, 1.0, 1e-9);  // y1 = 0.2
  CHECK_NEAR(st.bound_after, 5.0, 1e-6);   // y1 = 1
  for (const Cut& c : cuts) CHECK(cut_max_violation(p, c) <= 1e-9);
}

// Lifted flow cover cuts on random single-node flow sets: inflow arcs x_j <= u_j y_j (some with a
// minimum flow x_j >= l_j y_j, some without an arc binary), outflow arcs, binaries directly in the
// balance row, b as <=, >= or =. Every flow cover cut must be valid (enumeration), and some must be
// produced.
TEST(cuts_flow_cover_random_single_node) {
  // Flow covers replace c-MIR cuts here whenever more efficacious (by default only where c-MIR fails).
#ifdef _WIN32
  _putenv_s("MILP_FC_WIN", "1");
#else
  setenv("MILP_FC_WIN", "1", 1);
#endif
  std::mt19937 rng(2026);
  int ncuts = 0, ninvalid = 0, nfc = 0;
  for (int t = 0; t < iters(300); ++t) {
    ModelBuilder b;
    const int nin = 2 + rng() % 4, nout = rng() % 3;
    std::vector<std::pair<Index, double>> terms;
    for (int k = 0; k < nin + nout; ++k) {
      const bool in = k < nin;
      const double u = 1.0 + rng() % 12 + (rng() % 2 ? 0.5 : 0.0);
      const Index x = b.add_col("x" + std::to_string(k), in ? -1.0 - rng() % 5 : 0.5 * (rng() % 3), 0.0,
                                rng() % 4 == 0 ? u - 1.0 + 0.25 * (rng() % 4) : kInf);
      terms.emplace_back(x, in ? 1.0 : -1.0);
      if (rng() % 6 == 0) continue;  // no arc binary: only the simple bound (if any) or none
      const Index y = b.add_col("y" + std::to_string(k), 1.0 + rng() % 9, 0.0, 1.0, VarType::Binary);
      const Index r = b.add_row("vub" + std::to_string(k), -kInf, 0.0);
      b.add_coef(r, x, 1.0), b.add_coef(r, y, -u);
      if (rng() % 4 == 0) {
        const Index r2 = b.add_row("vlb" + std::to_string(k), -kInf, 0.0);
        b.add_coef(r2, x, -1.0), b.add_coef(r2, y, 0.5 + rng() % 2);
      }
    }
    for (int k = 0; k < static_cast<int>(rng() % 2); ++k) {  // a binary directly in the balance row
      const Index y = b.add_col("z" + std::to_string(k), -1.0 - rng() % 4, 0.0, 1.0, VarType::Binary);
      terms.emplace_back(y, rng() % 2 ? 3.0 + rng() % 4 : -(3.0 + rng() % 4));
    }
    const double d = 2.0 + rng() % 15 + (rng() % 2 ? 0.5 : 0.0);
    const int kind = rng() % 3;
    const Index r = b.add_row("bal", kind == 0 ? -kInf : d, kind == 1 ? kInf : d);  // <=, >=, =
    for (auto& [j, v] : terms) b.add_coef(r, j, v);
    b.model().sense = rng() % 2 ? ObjSense::Minimize : ObjSense::Maximize;
    MipProblem p = make_problem(b.build());
    if (enumerate_optimum(p) == kInf || enumerate_optimum(p) == -kInf) continue;
    check_cuts_valid(p, only(false, false, true), &ncuts, &ninvalid);
  }
  nfc = g_origin_count["flowcover"];
#ifdef _WIN32
  _putenv_s("MILP_FC_WIN", "");
#else
  unsetenv("MILP_FC_WIN");
#endif
  std::printf("  single-node flow sets: %d cuts checked (%d flow cover), %d invalid\n", ncuts, nfc, ninvalid);
  g_origin_count.clear();
  CHECK(nfc > 30);
  CHECK_EQ(ninvalid, 0);
}

// Tableau c-MIR cuts at "nodes": random integer bounds tightened in the LP (not in the problem), so
// the tableau rows mix local bounds into the LP point; the cuts use global bounds only and must be
// valid for every feasible point of the problem.
TEST(cuts_tableau_mir_valid_at_nodes) {
  std::mt19937 rng(71);
  int ncuts = 0, ninvalid = 0, nodes = 0;
  for (int t = 0; t < iters(400); ++t) {
    MipProblem p = make_problem(t % 3 == 2 ? random_fcnf(rng) : random_mip(rng, t % 3 == 1));
    DualSimplex lp;
    lp.load(p.A, p.cost, p.col_lb, p.col_ub, p.row_lb, p.row_ub);
    for (Index j = 0; j < p.num_cols(); ++j) {
      if (!p.is_int[j] || rng() % 3 != 0 || !is_finite(p.col_lb[j]) || !is_finite(p.col_ub[j])) continue;
      const double l = p.col_lb[j], u = p.col_ub[j];
      if (u <= l) continue;
      const double v = l + static_cast<double>(rng() % static_cast<unsigned>(u - l + 1));
      if (rng() % 2) lp.set_col_bounds(j, l, v);
      else lp.set_col_bounds(j, v, u);
    }
    LpOptions o;
    if (lp.solve(o) != LpStatus::Optimal) continue;
    ++nodes;
    CutData data;
    data.build(p);
    const double* x = lp.x();
    std::vector<double> xs(x, x + p.num_cols() + lp.num_rows());
    SepaContext ctx{p, lp, xs.data(), p.num_rows(), data, 0};
    std::vector<Cut> cuts;
    make_tableau_mir_separator()->separate(ctx, cuts);
    for (Cut& c : cuts) {
      if (!clean_cut(c, p.col_lb, p.col_ub, 1 << 30)) continue;
      ++ncuts;
      const double viol = cut_max_violation(p, c);
      if (viol > 1e-6 * std::max(1.0, std::abs(c.rhs))) {
        ++ninvalid;
        std::fprintf(stderr, "  invalid %s cut (violation %g)\n", c.origin, viol);
      }
    }
  }
  std::printf("  tableau c-MIR at nodes: %d node LPs, %d cuts checked, %d invalid\n", nodes, ncuts, ninvalid);
  CHECK(ncuts > 100);
  CHECK_EQ(ninvalid, 0);
}

TEST(cuts_knapsack_cover_lifting_valid) {
  // GNS-lifted covers checked against every 0/1 point of random knapsacks.
  std::mt19937 rng(17);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  int ncuts = 0, ninvalid = 0, nfrac = 0;
  for (int t = 0; t < iters(3000); ++t) {
    const int n = 2 + static_cast<int>(rng() % 9);
    std::vector<double> w(n), z(n);
    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
      w[k] = rng() % 3 == 0 ? 1.0 + rng() % 20 : 1.0 + std::round(U(rng) * 2000) / 100.0;
      sum += w[k];
      z[k] = rng() % 3 == 0 ? 1.0 : rng() % 3 == 0 ? 0.0 : U(rng);
    }
    const double b = std::round(U(rng) * sum * 100) / 100.0;
    std::vector<double> coef;
    double rhs = 0.0;
    if (!knapsack_cover_cut(w, z, b, coef, rhs)) continue;
    ++ncuts;
    for (double c : coef) nfrac += c != std::floor(c);
    double worst = -kInf;
    for (int mask = 0; mask < (1 << n); ++mask) {
      double lw = 0.0, lc = 0.0;
      for (int k = 0; k < n; ++k) {
        if (mask >> k & 1) lw += w[k], lc += coef[k];
      }
      if (lw <= b + 1e-9) worst = std::max(worst, lc - rhs);
    }
    if (worst > 1e-9) ++ninvalid;
  }
  std::printf("  lifted knapsack covers: %d cuts checked, %d with fractional lifting, %d invalid\n", ncuts, nfrac,
              ninvalid);
  CHECK(ncuts > 300);
  CHECK(nfrac > 0);
  CHECK_EQ(ninvalid, 0);
}

TEST(cuts_aggregated_knapsack_cover) {
  // Demand 7 at a node fed by three arcs of capacity 4 (x_a <= 4 y_a, cost y_a): the balance row with
  // the VUBs substituted is the knapsack 4 y1 + 4 y2 + 4 y3 >= 7, whose cover cut y1 + y2 + y3 >= 2
  // closes the LP bound 1.75 -> 2.
  ModelBuilder b;
  std::vector<Index> x, y;
  for (int a = 0; a < 3; ++a) x.push_back(b.add_col("x" + std::to_string(a), 0.0, 0, kInf));
  for (int a = 0; a < 3; ++a) y.push_back(b.add_col("y" + std::to_string(a), 1.0, 0, 1, VarType::Binary));
  const Index r = b.add_row("bal", 7.0, 7.0);
  for (int a = 0; a < 3; ++a) {
    b.add_coef(r, x[a], 1.0);
    const Index v = b.add_row("vub" + std::to_string(a), -kInf, 0.0);
    b.add_coef(v, x[a], 1.0), b.add_coef(v, y[a], -4.0);
  }
  MipProblem p = make_problem(b.build());
  DualSimplex lp;
  lp.load(p.A, p.cost, p.col_lb, p.col_ub, p.row_lb, p.row_ub);
  Logger log(0);
  Timer timer;
  std::vector<Cut> cuts;
  CutLoopStats st = cut_loop(p, lp, only(false, false, true), log, timer, kInf, &cuts);
  CHECK_NEAR(st.bound_before, 1.75, 1e-9);
  CHECK_NEAR(st.bound_after, 2.0, 1e-6);
  bool aggcover = false;
  for (const Cut& c : cuts) {
    aggcover = aggcover || std::string(c.origin) == "aggcover";
    CHECK(cut_max_violation(p, c) <= 1e-9);
  }
  CHECK(aggcover);
}

TEST(cuts_degenerate_bin_packing) {
  // Identical bins k (capacity 10): 6 a_k + 5 b_k <= 10 y_k, a_k <= 1, b_k <= 2, demands
  // sum a = 3, sum b = 2. LP 2.8 = 28 / 10 with every bin interchangeable (dual degenerate);
  // the optimum is 4. With the default loop policy (stall rule, slack removal, re-added
  // dropped cuts) the bound must move and every cut, re-added ones included, stay valid.
  const int bins = 4;
  ModelBuilder b;
  std::vector<Index> a, bb, y;
  for (int k = 0; k < bins; ++k) {
    a.push_back(b.add_col("a" + std::to_string(k), 0.0, 0, 1, VarType::Integer));
    bb.push_back(b.add_col("b" + std::to_string(k), 0.0, 0, 2, VarType::Integer));
    y.push_back(b.add_col("y" + std::to_string(k), 1.0, 0, 1, VarType::Binary));
    Index r = b.add_row("cap", -kInf, 0.0);
    b.add_coef(r, a[k], 6.0), b.add_coef(r, bb[k], 5.0), b.add_coef(r, y[k], -10.0);
  }
  Index da = b.add_row("da", 3.0, 3.0), db = b.add_row("db", 2.0, 2.0);
  for (int k = 0; k < bins; ++k) b.add_coef(da, a[k], 1.0), b.add_coef(db, bb[k], 1.0);
  MipProblem p = make_problem(b.build());
  DualSimplex lp;
  lp.load(p.A, p.cost, p.col_lb, p.col_ub, p.row_lb, p.row_ub);
  Logger log(0);
  Timer timer;
  std::vector<Cut> cuts;
  CutLoopOptions opt;
  CutLoopStats st = cut_loop(p, lp, opt, log, timer, kInf, &cuts);
  CHECK_NEAR(st.bound_before, 2.8, 1e-9);
  CHECK(st.bound_after > 2.8 + 1e-3);
  CHECK(st.bound_after <= 4.0 + 1e-6);
  REQUIRE(!cuts.empty());
  // Pure integer: check each cut against every feasible point directly.
  int invalid = 0;
  enumerate_ints(p, [&](const std::vector<double>& x) {
    if (p.max_violation(x) > 1e-9) return;
    for (const Cut& c : cuts) invalid += c.activity(x.data()) > c.rhs + 1e-9;
  });
  CHECK(invalid == 0);
}

TEST(cuts_end_to_end_matches_enumeration) {
  std::mt19937 rng(37);
  SolverParams params;
  params.verbosity = 0;
  params.threads = 1;
  params.mip_rel_gap = 1e-9;
  params.mip_abs_gap = 1e-9;
  int solved = 0;
  for (int t = 0; t < iters(500); ++t) {
    Model m = t % 7 == 6 ? random_mcnd(rng) : t % 3 == 2 ? random_fcnf(rng) : random_mip(rng, t % 2 == 1);
    MipProblem p = make_problem(m);
    const double opt = enumerate_optimum(p);  // internal min sense
    SolveResult r = solve(m, params);
    if (opt == kInf) {
      CHECK(r.status == Status::Infeasible);
      continue;
    }
    REQUIRE(r.status == Status::Optimal);
    const double sense = m.sense == ObjSense::Maximize ? -1.0 : 1.0;
    CHECK_NEAR(sense * r.objective, opt, 1e-6 * std::max(1.0, std::abs(opt)));
    ++solved;
  }
  std::printf("  end to end: %d feasible instances solved\n", solved);
  CHECK(solved > 100);
}

namespace {

void set_no_cuts(bool on) {
#ifdef _WIN32
  _putenv_s("MILP_NO_CUTS", on ? "1" : "");
#else
  if (on) setenv("MILP_NO_CUTS", "1", 1);
  else unsetenv("MILP_NO_CUTS");
#endif
}

}  // namespace

TEST(cuts_larger_random_satisfied_by_optimum) {
  // Bigger instances (no enumeration): every cut must hold at the optimum found
  // without cuts, and solving with cuts must give the same objective.
  std::mt19937 rng(41);
  SolverParams params;
  params.verbosity = 0;
  params.threads = 1;
  params.mip_rel_gap = 1e-9;
  params.mip_abs_gap = 1e-9;
  params.node_limit = 5000;
  int checked = 0, bad = 0;
  for (int t = 0; t < iters(150); ++t) {
    Model m = t % 4 == 3 ? random_fcnf(rng, true) : random_mip(rng, t % 3 != 0, true);
    set_no_cuts(true);
    SolveResult r0 = solve(m, params);
    set_no_cuts(false);
    if (r0.status != Status::Optimal) continue;
    MipProblem p = make_problem(m);
    DualSimplex lp;
    lp.load(p.A, p.cost, p.col_lb, p.col_ub, p.row_lb, p.row_ub);
    Logger log(0);
    Timer timer;
    std::vector<Cut> cuts;
    cut_loop(p, lp, only(true, true, true, true), log, timer, kInf, &cuts);
    for (const Cut& c : cuts) {
      ++checked;
      const double viol = c.activity(r0.x.data()) - c.rhs;
      if (viol > 1e-6 * std::max(1.0, std::abs(c.rhs))) {
        ++bad;
        std::fprintf(stderr, "  %s cut violated by optimum: %g\n", c.origin, viol);
      }
    }
    SolveResult r1 = solve(m, params);
    if (r1.status != Status::Optimal) continue;
    if (std::abs(r1.objective - r0.objective) > 1e-6 * std::max(1.0, std::abs(r0.objective))) {
      std::fprintf(stderr, "  instance %d: objective %.12g with cuts, %.12g without\n", t, r1.objective, r0.objective);
      if (const char* f = std::getenv("MILP_TEST_DUMP")) {
        Model mm = m;
        for (size_t i = 0; i < mm.row_names.size(); ++i) mm.row_names[i] = "r" + std::to_string(i);
        write_mps(mm, f + std::to_string(t) + ".mps");
      }
    }
    CHECK_NEAR(r1.objective, r0.objective, 1e-6 * std::max(1.0, std::abs(r0.objective)));
  }
  std::printf("  larger: %d cuts checked against optima, %d violated\n", checked, bad);
  CHECK(checked > 200);
  CHECK_EQ(bad, 0);
}

TEST(search_implied_integer_objective_bound) {
  // min z with z = c * (y1 + ... + y5) (z continuous), 2 (y1 + ... + y5) >= 3: the LP gives 1.5 c.
  // For c = 1, z is implied integer, so the bound rounds up to the optimum 2 at the root; for
  // c = 0.3 it must not be rounded (optimum 0.6).
  set_no_cuts(true);
  for (double c : {1.0, 0.3}) {
    ModelBuilder b;
    const Index z = b.add_col("z", 1.0, 0.0, 100.0);
    const Index eq = b.add_row("eq", 0.0, 0.0), cover = b.add_row("cover", 3.0, kInf);
    b.add_coef(eq, z, 1.0);
    for (int k = 0; k < 5; ++k) {
      const Index y = b.add_col("y" + std::to_string(k), 0.0, 0, 1, VarType::Binary);
      b.add_coef(eq, y, -c);
      b.add_coef(cover, y, 2.0);
    }
    SolverParams params;
    params.verbosity = 0;
    params.threads = 1;
    params.presolve = false;
    params.node_limit = 1;
    SolveResult r = solve(b.build(), params);
    REQUIRE(r.has_solution);
    CHECK_NEAR(r.objective, 2.0 * c, 1e-6);
    CHECK(r.bound <= 2.0 * c + 1e-6);
    if (c == 1.0) CHECK(r.bound >= 2.0 - 1e-6);
  }
  set_no_cuts(false);
}
