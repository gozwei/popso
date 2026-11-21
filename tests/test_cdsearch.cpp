// Tests for the conflict search (heuristics/cdsearch.hpp). On small random pure-integer problems, with a budget
// that lets the search finish, it must find a point exactly when enumeration finds one and report infeasibility
// exactly otherwise: a learned clause that cut off a feasible point would show as a missed point or a wrong
// infeasibility. Every returned point must be feasible, the bounds must be restored exactly at every return to the
// root, and runs must be reproducible.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "milp/core/model.hpp"
#include "milp/heuristics/cdsearch.hpp"
#include "milp/search/mip_problem.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

MipProblem build(const Model& m) {
  MipProblem p;
  bool ok = MipProblem::from_model(m, &p);
  REQUIRE(ok);
  return p;
}

// Random rows over binaries and small general integers; `tight` rows of each kind.
MipProblem random_ip(unsigned seed, int cols, int rows, bool general) {
  std::mt19937 rng(seed);
  ModelBuilder b;
  for (int j = 0; j < cols; ++j) {
    const bool gen = general && rng() % 3 == 0;
    b.add_col("x" + std::to_string(j), static_cast<double>(rng() % 7) - 3.0, gen ? -1.0 : 0.0, gen ? 2.0 : 1.0,
              gen ? VarType::Integer : VarType::Binary);
  }
  for (int i = 0; i < rows; ++i) {
    const int len = 2 + static_cast<int>(rng() % 4);
    std::vector<int> js;
    std::vector<double> as;
    double lo = 0.0, hi = 0.0;
    for (int t = 0; t < len; ++t) {
      const int j = static_cast<int>(rng() % cols);
      if (std::find(js.begin(), js.end(), j) != js.end()) continue;
      const double a = static_cast<double>(1 + rng() % 4) * (rng() % 3 == 0 ? -1.0 : 1.0);
      js.push_back(j), as.push_back(a);
      lo += std::min(-a, 2 * a), hi += std::max(-a, 2 * a);  // generous range of the activity
    }
    const double mid = std::round(lo + (hi - lo) * (0.25 + 0.5 * static_cast<double>(rng() % 100) / 100.0));
    const int kind = static_cast<int>(rng() % 4);
    const Index r = kind == 0   ? b.add_row("r" + std::to_string(i), -kInf, mid)
                    : kind == 1 ? b.add_row("r" + std::to_string(i), mid, kInf)
                    : kind == 2 ? b.add_row("r" + std::to_string(i), mid, mid)
                                : b.add_row("r" + std::to_string(i), mid - 1.0, mid + 1.0);
    for (size_t t = 0; t < js.size(); ++t) b.add_coef(r, static_cast<Index>(js[t]), as[t]);
  }
  return build(b.build());
}

// Harder for propagation: long equality and narrow range rows over 16 columns; with `planted` the sides are
// taken from a hidden point (feasible), else at random (mostly infeasible, proven only by search).
MipProblem subset_rows(unsigned seed, bool planted, bool general) {
  std::mt19937 rng(seed);
  const int cols = 15 + static_cast<int>(rng() % 4), rows = 8 + static_cast<int>(rng() % 8);
  ModelBuilder b;
  std::vector<double> hidden(cols), lo(cols), hi(cols);
  for (int j = 0; j < cols; ++j) {
    const bool gen = general && rng() % 4 == 0;
    lo[j] = 0.0, hi[j] = gen ? 2.0 : 1.0;
    hidden[j] = lo[j] + static_cast<double>(rng() % static_cast<unsigned>(hi[j] - lo[j] + 1.0));
    b.add_col("x" + std::to_string(j), 0.0, lo[j], hi[j], gen ? VarType::Integer : VarType::Binary);
  }
  for (int i = 0; i < rows; ++i) {
    const int len = 5 + static_cast<int>(rng() % 4);
    std::vector<int> js;
    std::vector<double> as;
    double act = 0.0, amin = 0.0, amax = 0.0;
    for (int t = 0; t < len; ++t) {
      const int j = static_cast<int>(rng() % cols);
      if (std::find(js.begin(), js.end(), j) != js.end()) continue;
      const double a = rng() % 5 == 0 ? 2.0 : 1.0;  // cardinality-like rows: weak propagation
      js.push_back(j), as.push_back(a);
      act += a * hidden[j], amin += std::min(a * lo[j], a * hi[j]), amax += std::max(a * lo[j], a * hi[j]);
    }
    const double side = planted ? act : std::round(amin + (amax - amin) * static_cast<double>(30 + rng() % 40) / 100.0);
    const double width = 0.0;
    const Index r = b.add_row("r" + std::to_string(i), side - width, side);
    for (size_t t = 0; t < js.size(); ++t) b.add_coef(r, static_cast<Index>(js[t]), as[t]);
  }
  return build(b.build());
}

// Whether some integer point within the bounds satisfies all rows (odometer over the columns).
bool enumerate_feasible(const MipProblem& p) {
  const Index n = p.num_cols();
  std::vector<double> x(p.col_lb);
  for (;;) {
    if (p.max_violation(x) <= 1e-6) return true;
    Index j = 0;
    while (j < n && x[j] >= p.col_ub[j]) x[j] = p.col_lb[j], ++j;
    if (j == n) return false;
    x[j] += 1.0;
  }
}

}  // namespace

TEST(cdsearch_complete_on_small_problems) {
  int feasible = 0, infeasible = 0;
  long long conflicts = 0, learned = 0;
  for (unsigned seed = 1; seed <= 600; ++seed) {
    const bool general = seed % 2 == 0;
    MipProblem p = seed % 3 == 0 ? random_ip(seed, 9 + static_cast<int>(seed % 4), 3 + static_cast<int>(seed % 8), general)
                                 : subset_rows(seed, seed % 3 == 1, general);
    const bool want = enumerate_feasible(p);
    CdsLimits lim;
    lim.effort = 200000000;
    lim.seed = seed;
    lim.check = true;
    lim.minimize = (seed / 5) % 2 == 1;
    lim.restart_unit = 1 + static_cast<int>(seed % 5);  // frequent restarts: the root commit is exercised
    std::vector<double> x;
    CdsStats st;
    const bool got = conflict_search(p, lim, x, &st);
    CHECK(st.applicable);
    CHECK(!st.check_failed);
    CHECK(got == want);
    if (got) CHECK(p.max_violation(x) <= 1e-6);
    if (!got) CHECK(st.infeasible == !want);
    if (got) CHECK(!st.infeasible);
    (want ? feasible : infeasible)++;
    conflicts += st.conflicts, learned += st.learned;
  }
  std::printf("  cdsearch: %d feasible, %d infeasible, %lld conflicts, %lld learned\n", feasible, infeasible, conflicts, learned);
  CHECK(feasible >= 100);
  CHECK(infeasible >= 100);
  CHECK(learned > 500);
  CHECK(conflicts >= learned);
}

// Clauses over the length limit are not kept, only followed (jump back, other side as a decision). The search
// is then not complete, but what it returns must still be right: a point is feasible, and "infeasible" is
// claimed only when enumeration agrees.
TEST(cdsearch_long_clauses_sound) {
  int found = 0, missed = 0;
  long long longc = 0;
  for (unsigned seed = 1; seed <= 300; ++seed) {
    const bool general = seed % 2 == 0;
    MipProblem p = seed % 3 == 0 ? random_ip(seed, 9 + static_cast<int>(seed % 4), 3 + static_cast<int>(seed % 8), general)
                                 : subset_rows(seed, seed % 3 == 1, general);
    const bool want = enumerate_feasible(p);
    CdsLimits lim;
    lim.effort = 20000000;
    lim.seed = seed;
    lim.check = true;
    lim.max_lits = 1 + static_cast<Index>(seed % 3);
    std::vector<double> x;
    CdsStats st;
    const bool got = conflict_search(p, lim, x, &st);
    CHECK(!st.check_failed);
    if (got) CHECK(want && p.max_violation(x) <= 1e-6);
    if (st.infeasible) CHECK(!want && !got);
    if (want) (got ? found : missed)++;
    longc += st.long_clauses;
  }
  std::printf("  cdsearch long clauses: %d found, %d missed, %lld long clauses\n", found, missed, longc);
  CHECK(longc > 500);
  CHECK(found > 5 * missed);
}

TEST(cdsearch_reproducible_and_hint) {
  for (unsigned seed = 1; seed <= 40; ++seed) {
    MipProblem p = random_ip(1000 + seed, 30, 34, seed % 2 == 0);
    CdsLimits lim;
    lim.effort = 20000000;
    lim.seed = seed;
    std::vector<double> x1, x2;
    CdsStats s1, s2;
    const bool g1 = conflict_search(p, lim, x1, &s1), g2 = conflict_search(p, lim, x2, &s2);
    CHECK(g1 == g2);
    CHECK(s1.effort == s2.effort && s1.conflicts == s2.conflicts && s1.decisions == s2.decisions);
    if (g1) {
      CHECK(x1 == x2);
      // Started from a feasible point, the search walks straight to it.
      CdsLimits hl = lim;
      hl.hint = &x1;
      std::vector<double> y;
      CdsStats sh;
      CHECK(conflict_search(p, hl, y, &sh));
      CHECK(sh.conflicts == 0);
      CHECK(y == x1);
    }
  }
}

TEST(cdsearch_out_of_scope) {
  ModelBuilder b;
  b.add_col("x", 1.0, 0, 1, VarType::Binary);
  b.add_col("y", 1.0, 0, 5, VarType::Continuous);
  const Index r = b.add_row("r", 1.0, kInf);
  b.add_coef(r, 0, 1.0), b.add_coef(r, 1, 1.0);
  MipProblem p = build(b.build());
  std::vector<double> x;
  CdsStats st;
  CHECK(!conflict_search(p, CdsLimits{}, x, &st));
  CHECK(!st.applicable && !st.infeasible);
}
