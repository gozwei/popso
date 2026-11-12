// Tests for quadratic objectives: model, readers / writer, convexity test, the barrier on
// convex QPs (against a dense active-set enumeration) and the MIQP branch-and-bound (against
// enumeration of the integer values).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "milp/core/model.hpp"
#include "milp/io/lp_reader.hpp"
#include "milp/io/mps_reader.hpp"
#include "milp/lp/barrier.hpp"
#include "milp/qp/qp.hpp"
#include "milp/qp/qp_active.hpp"
#include "milp/search/mip_problem.hpp"
#include "milp/solver/solver.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

using Dense = std::vector<std::vector<double>>;

// Full symmetric Q from the model's lower triangle triplets.
Dense dense_q(const Model& m) {
  const Index n = m.num_cols();
  Dense Q(n, std::vector<double>(n, 0.0));
  for (size_t k = 0; k < m.q_val.size(); ++k) {
    Q[m.q_row[k]][m.q_col[k]] += m.q_val[k];
    if (m.q_row[k] != m.q_col[k]) Q[m.q_col[k]][m.q_row[k]] += m.q_val[k];
  }
  return Q;
}

double q_entry(const Model& m, Index i, Index j) { return dense_q(m)[i][j]; }

// Solves the dense system M z = r by Gaussian elimination with partial pivoting; false if singular.
bool dense_solve(Dense M, std::vector<double> r, std::vector<double>& z) {
  const size_t n = r.size();
  for (size_t c = 0; c < n; ++c) {
    size_t p = c;
    for (size_t i = c + 1; i < n; ++i)
      if (std::abs(M[i][c]) > std::abs(M[p][c])) p = i;
    if (std::abs(M[p][c]) < 1e-11) return false;
    std::swap(M[p], M[c]);
    std::swap(r[p], r[c]);
    for (size_t i = c + 1; i < n; ++i) {
      const double f = M[i][c] / M[c][c];
      if (f == 0.0) continue;
      for (size_t k = c; k < n; ++k) M[i][k] -= f * M[c][k];
      r[i] -= f * r[c];
    }
  }
  z.assign(n, 0.0);
  for (size_t c = n; c-- > 0;) {
    double s = r[c];
    for (size_t k = c + 1; k < n; ++k) s -= M[c][k] * z[k];
    z[c] = s / M[c][c];
  }
  return true;
}

// Reference for a small strictly convex QP (dense rows): every combination of active sides of
// the rows and bounds is solved as an equality-constrained QP; the best feasible solution is the
// optimum (Q positive definite). Returns +inf if infeasible.
double reference_qp(const Model& m, std::vector<double>* xbest = nullptr) {
  const Index n = m.num_cols(), mr = m.num_rows();
  const Dense Q = dense_q(m);
  Dense A(mr, std::vector<double>(n, 0.0));
  for (Index j = 0; j < n; ++j)
    for (Index p = m.A.begin(j); p < m.A.end(j); ++p) A[m.A.index[p]][j] = m.A.value[p];
  // Constraint sides: rows then columns; state 0 inactive, 1 at lower, 2 at upper.
  const Index K = mr + n;
  std::vector<int> state(K, 0);
  double best = kInf;
  for (;;) {
    std::vector<std::vector<double>> rows;
    std::vector<double> rhs;
    bool ok = true;
    for (Index k = 0; k < K && ok; ++k) {
      if (state[k] == 0) continue;
      std::vector<double> a(n, 0.0);
      double b;
      if (k < mr) {
        a = A[k];
        b = state[k] == 1 ? m.row_lb[k] : m.row_ub[k];
      } else {
        a[k - mr] = 1.0;
        b = state[k] == 1 ? m.col_lb[k - mr] : m.col_ub[k - mr];
      }
      if (!std::isfinite(b)) ok = false;
      rows.push_back(a);
      rhs.push_back(b);
    }
    if (ok && rows.size() <= static_cast<size_t>(n)) {
      const size_t s = rows.size(), N = static_cast<size_t>(n) + s;
      Dense M(N, std::vector<double>(N, 0.0));
      std::vector<double> r(N, 0.0), z;
      for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) M[i][j] = Q[i][j];
        r[i] = -m.obj[i];
      }
      for (size_t t = 0; t < s; ++t) {
        for (Index j = 0; j < n; ++j) M[n + t][j] = M[j][n + t] = rows[t][j];
        r[n + t] = rhs[t];
      }
      if (dense_solve(M, r, z)) {
        std::vector<double> x(z.begin(), z.begin() + n);
        if (m.violation(x).max() <= 1e-9) {
          const double v = m.objective_value(x);
          if (v < best) {
            best = v;
            if (xbest) *xbest = x;
          }
        }
      }
    }
    Index k = 0;
    while (k < K && state[k] == 2) state[k++] = 0;
    if (k == K) break;
    ++state[k];
  }
  return best;
}

// Random small strictly convex QP: Q = F F^T + 0.1 I, a few ranged / one-sided rows, boxes.
Model random_qp(std::mt19937& rng, Index n, Index m) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  ModelBuilder b;
  std::vector<double> x0(n);
  for (Index j = 0; j < n; ++j) {
    x0[j] = u(rng);
    const int kind = static_cast<int>(rng() % 4);
    const double lo = kind == 3 ? -kInf : x0[j] - std::abs(u(rng)) - 0.1;
    const double hi = kind == 2 ? kInf : x0[j] + std::abs(u(rng)) + 0.1;
    b.add_col("x" + std::to_string(j), 3.0 * u(rng), lo, hi);
  }
  for (Index i = 0; i < m; ++i) {
    double act = 0.0;
    std::vector<double> a(n);
    for (Index j = 0; j < n; ++j) a[j] = rng() % 3 == 0 ? 0.0 : u(rng), act += a[j] * x0[j];
    const int kind = static_cast<int>(rng() % 3);
    const double lo = kind == 1 ? -kInf : act - std::abs(u(rng));
    const double hi = kind == 2 ? kInf : act + std::abs(u(rng));
    const Index r = b.add_row("r" + std::to_string(i), lo, hi);
    for (Index j = 0; j < n; ++j)
      if (a[j] != 0.0) b.add_coef(r, j, a[j]);
  }
  Model md = b.build();
  Dense F(n, std::vector<double>(n));
  for (auto& row : F)
    for (double& v : row) v = u(rng);
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j <= i; ++j) {
      double s = i == j ? 0.1 : 0.0;
      for (Index k = 0; k < n; ++k) s += F[i][k] * F[j][k];
      md.add_q(i, j, s);
    }
  }
  return md;
}

SolverParams quiet() {
  SolverParams p;
  p.verbosity = 0;
  p.threads = 1;
  return p;
}

bool expect_error(const std::string& text, bool lp, const std::string& part) {
  try {
    if (lp) read_lp_string(text);
    else read_mps_string(text);
  } catch (const std::exception& e) {
    if (std::string(e.what()).find(part) != std::string::npos) return true;
    std::printf("  error message was: %s\n", e.what());
    return false;
  }
  std::printf("  no error\n");
  return false;
}

}  // namespace

TEST(qp_model_objective_value) {
  ModelBuilder b;
  b.add_col("x", 1.0, -kInf, kInf);
  b.add_col("y", -2.0, -kInf, kInf);
  Model m = b.build();
  CHECK(!m.has_quadratic());
  m.add_q(0, 0, 2.0);  // x^2
  m.add_q(0, 1, 3.0);  // stored as (1, 0): 3 x y
  m.add_q(1, 1, 4.0);  // 2 y^2
  CHECK(m.has_quadratic());
  CHECK_EQ(m.q_row[1], 1);
  CHECK_EQ(m.q_col[1], 0);
  // 1*1 - 2*2 + 1 + 3*2 + 2*4 = 12
  CHECK_NEAR(m.objective_value({1.0, 2.0}), 12.0, 1e-12);
  const SparseMatrix L = m.q_lower();
  CHECK_EQ(L.nnz(), 3);
  m.q_row.push_back(0);  // upper triangle entry: invalid
  m.q_col.push_back(1);
  m.q_val.push_back(1.0);
  CHECK_THROWS(m.validate(), std::runtime_error);
}

TEST(qp_mps_sections) {
  const std::string head = "NAME q\nROWS\n N obj\n G c\nCOLUMNS\n x obj 1 c 1\n y obj 1 c 1\nRHS\n rhs c 1\n";
  const Model a = read_mps_string(head + "QUADOBJ\n x x 2\n x y 1\n y y 4\nENDATA\n");
  const Model b = read_mps_string(head + "QMATRIX\n x x 2\n x y 1\n y x 1\n y y 4\nENDATA\n");
  const Model c = read_mps_string(head + "QSECTION obj\n x x 2\n x y 1\n y x 1\n y y 4\nENDATA\n");
  const Model d = read_mps_string(head + "QSECTION\n x x 2\n y x 1\n x y 1\n y y 4\nENDATA\n");
  for (const Model* m : {&a, &b, &c, &d}) {
    CHECK_NEAR(q_entry(*m, 0, 0), 2.0, 1e-15);
    CHECK_NEAR(q_entry(*m, 0, 1), 1.0, 1e-15);
    CHECK_NEAR(q_entry(*m, 1, 1), 4.0, 1e-15);
  }
  CHECK(read_mps_string(head + "QSECTION c\n x x 1\nENDATA\n").has_quadratic_rows());  // a quadratic constraint
  CHECK(expect_error(head + "QSECTION z\n x x 1\nENDATA\n", false, "QSECTION for unknown row 'z'"));
  CHECK(expect_error(head + "QUADOBJ\n x z 1\nENDATA\n", false, "column 'z' in QUADOBJ"));
  CHECK(expect_error(head + "QUADOBJ\n x y\nENDATA\n", false, "QUADOBJ: expected"));
  // Round trip through the writer.
  const Model e = read_mps_string(write_mps_string(a));
  CHECK_EQ(e.q_lower().nnz(), 3);
  for (Index i = 0; i < 2; ++i)
    for (Index j = 0; j < 2; ++j) CHECK_NEAR(q_entry(e, i, j), q_entry(a, i, j), 0.0);
}

TEST(qp_lp_format_objective) {
  const Model m = read_lp_string("min\n obj: 3 x + [ 2 x ^ 2 + 3 x * y + y*y - z * x ] / 2 + 1\nst\n c: x + y + z >= 1\nend\n");
  // Objective 3x + x^2 + 1.5 xy + 0.5 y^2 - 0.5 xz + 1: Q_xx = 2, Q_xy = 1.5, Q_yy = 1, Q_xz = -0.5.
  CHECK_NEAR(m.obj[0], 3.0, 0.0);
  CHECK_NEAR(m.obj_offset, 1.0, 0.0);
  CHECK_NEAR(q_entry(m, 0, 0), 2.0, 1e-15);
  CHECK_NEAR(q_entry(m, 0, 1), 1.5, 1e-15);
  CHECK_NEAR(q_entry(m, 1, 1), 1.0, 1e-15);
  CHECK_NEAR(q_entry(m, 0, 2), -0.5, 1e-15);
  const std::vector<double> x = {1.0, 2.0, 3.0};
  CHECK_NEAR(m.objective_value(x), 3.0 + 1.0 + 3.0 + 2.0 - 1.5 + 1.0, 1e-12);
  // No divisor, and a negative bracket.
  const Model n = read_lp_string("max\n - [ x ^ 2 ]\nst\n x >= 1\nend\n");
  CHECK_NEAR(q_entry(n, 0, 0), -2.0, 1e-15);
  // The same model in MPS form.
  const Model r = read_mps_string(write_mps_string(m));
  CHECK_NEAR(r.objective_value(x), m.objective_value(x), 1e-12);
}

TEST(qp_convexity_test) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int trial = 0; trial < 40; ++trial) {
    const Index n = 2 + static_cast<Index>(rng() % 7), k = 1 + static_cast<Index>(rng() % static_cast<unsigned>(n));
    Dense F(n, std::vector<double>(k));
    for (auto& row : F)
      for (double& v : row) v = u(rng);
    // PSD (possibly singular): F F^T; indefinite: F F^T - G G^T with a clearly negative part.
    const bool indefinite = trial % 2 == 1;
    std::vector<Index> J, I;
    std::vector<double> V;
    std::vector<double> g(n);
    for (double& v : g) v = u(rng);
    for (Index j = 0; j < n; ++j) {
      for (Index i = j; i < n; ++i) {
        double s = 0.0;
        for (Index t = 0; t < k; ++t) s += F[i][t] * F[j][t];
        if (indefinite) s -= 0.5 * g[i] * g[j] * n;  // a rank-one negative term outside span(F) in general
        J.push_back(j), I.push_back(i), V.push_back(s);
      }
    }
    const SparseMatrix Q = SparseMatrix::from_triplets(n, n, J, I, V);
    const ConvexityResult r = check_convexity(Q);
    if (!indefinite) {
      CHECK(r.convex);
    } else if (k < n) {
      CHECK(!r.convex);  // F F^T is singular: subtracting any rank-one term outside its range goes negative
    }
  }
  // Hand-made cases.
  auto make = [](Index n, std::vector<Index> J, std::vector<Index> I, std::vector<double> V) {
    return SparseMatrix::from_triplets(n, n, J, I, V);
  };
  CHECK(!check_convexity(make(2, {0, 0}, {0, 1}, {0.0, 1.0})).convex);         // [[0 1][1 0]]
  CHECK(!check_convexity(make(2, {0, 1}, {0, 1}, {1.0, -1.0})).convex);        // diag(1, -1)
  CHECK(check_convexity(make(2, {0, 0, 1}, {0, 1, 1}, {1.0, 1.0, 1.0})).convex);  // [[1 1][1 1]]
  CHECK(!check_convexity(make(2, {0, 0, 1}, {0, 1, 1}, {1.0, 1.1, 1.0})).convex);
  const ConvexityResult p = check_convexity(make(2, {0, 0, 1}, {0, 1, 1}, {1.0, 1.0 + 1e-8, 1.0}));
  CHECK(p.convex && p.perturbed);  // PSD up to 1e-8: accepted with a diagonal shift
  CHECK(check_convexity(make(3, {}, {}, {})).convex);
  // Badly scaled but PSD: diag(1e-6, 1e6) with the largest cross term.
  CHECK(check_convexity(make(2, {0, 0, 1}, {0, 1, 1}, {1e-6, 1.0, 1e6})).convex);
  CHECK(!check_convexity(make(2, {0, 0, 1}, {0, 1, 1}, {1e-6, 1.001, 1e6})).convex);
}

TEST(qp_barrier_matches_dense_reference) {
  std::mt19937 rng(11);
  int solved = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const Index n = 2 + static_cast<Index>(rng() % 4), m = static_cast<Index>(rng() % 4);
    Model md = random_qp(rng, n, m);
    std::vector<double> xr;
    const double ref = reference_qp(md, &xr);
    SolveResult r = solve(md, quiet());
    if (!std::isfinite(ref)) {
      CHECK(r.status == Status::Infeasible);
      continue;
    }
    CHECK(r.status == Status::Optimal);
    if (r.status != Status::Optimal) continue;
    ++solved;
    CHECK(md.violation(r.x).max() <= 1e-7);
    CHECK_NEAR(r.objective, ref, 1e-7 * (1.0 + std::abs(ref)));
    // Maximization of the negated objective.
    Model mx = md;
    mx.sense = ObjSense::Maximize;
    for (double& c : mx.obj) c = -c;
    for (double& v : mx.q_val) v = -v;
    const SolveResult rx = solve(mx, quiet());
    CHECK(rx.status == Status::Optimal);
    CHECK_NEAR(rx.objective, -ref, 1e-7 * (1.0 + std::abs(ref)));
  }
  CHECK(solved >= 40);
}

TEST(qp_kkt_paths) {
  // A diagonal Hessian (normal equations allowed) and the same problem forced onto the augmented
  // system agree; free variables with a positive curvature need no bounds.
  ModelBuilder b;
  for (int j = 0; j < 5; ++j) b.add_col("x" + std::to_string(j), j - 2.0, -kInf, kInf);
  const Index r = b.add_row("s", 1.0, 1.0);
  for (int j = 0; j < 5; ++j) b.add_coef(r, j, 1.0);
  Model m = b.build();
  for (int j = 0; j < 5; ++j) m.add_q(j, j, 1.0 + j);
  const double ref = reference_qp(m);
  const SolveResult a = solve(m, quiet());
  CHECK(a.status == Status::Optimal);
  CHECK_NEAR(a.objective, ref, 1e-8);
  // Same with a zero off-diagonal entry structurally present (augmented system).
  m.add_q(4, 0, 1e-300);
  m.add_q(4, 0, -1e-300);
  m.add_q(3, 1, 0.25);
  const double ref2 = reference_qp(m);
  const SolveResult c = solve(m, quiet());
  CHECK(c.status == Status::Optimal);
  CHECK_NEAR(c.objective, ref2, 1e-8);
}

TEST(qp_statuses) {
  // Nonconvex: x^2 - y^2.
  {
    ModelBuilder b;
    b.add_col("x", 0.0, -1.0, 1.0);
    b.add_col("y", 0.0, -1.0, 1.0);
    Model m = b.build();
    m.add_q(0, 0, 2.0);
    m.add_q(1, 1, -2.0);
    // Bounded factors: spatial branch-and-bound (v0.10). min x^2 - y^2 over the box is -1, its max 1.
    const SolveResult lo = solve(m, quiet());
    CHECK(lo.status == Status::Optimal);
    CHECK_NEAR(lo.objective, -1.0, 1e-5);
    m.sense = ObjSense::Maximize;
    const SolveResult hi = solve(m, quiet());
    CHECK(hi.status == Status::Optimal);
    CHECK_NEAR(hi.objective, 1.0, 1e-5);
    m.q_val[0] = -2.0;  // concave: -x^2 - y^2, maximized
    const SolveResult r = solve(m, quiet());
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, 0.0, 1e-7);
  }
  // Infeasible rows.
  {
    ModelBuilder b;
    b.add_col("x", 1.0, 0.0, kInf);
    b.add_col("y", 1.0, 0.0, kInf);
    const Index r = b.add_row("c", -kInf, -1.0);
    b.add_coef(r, 0, 1.0);
    b.add_coef(r, 1, 1.0);
    Model m = b.build();
    m.add_q(0, 0, 1.0);
    CHECK(solve(m, quiet()).status == Status::Infeasible);
  }
  // Unbounded: min x^2 - y, y free above (Q singular along y).
  {
    ModelBuilder b;
    b.add_col("x", 0.0, -kInf, kInf);
    b.add_col("y", -1.0, 0.0, kInf);
    const Index r = b.add_row("c", -kInf, 5.0);
    b.add_coef(r, 0, 1.0);
    b.add_coef(r, 1, -1.0);
    Model m = b.build();
    m.add_q(0, 0, 1.0);
    const Status st = solve(m, quiet()).status;
    CHECK(st == Status::Unbounded);
  }
  // Bounded thanks to the quadratic term along the ray direction: min (x - y)^2 - y + x, x - y <= 5.
  {
    ModelBuilder b;
    b.add_col("x", 1.0, -kInf, kInf);
    b.add_col("y", -1.0, 0.0, kInf);
    Model m = b.build();
    m.add_q(0, 0, 2.0);
    m.add_q(1, 0, -2.0);
    m.add_q(1, 1, 2.0);
    // Objective (x-y)^2 + (x - y) has its minimum -1/4 along the whole line x - y = -1/2.
    const SolveResult r = solve(m, quiet());
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, -0.25, 1e-6);
  }
}

namespace {

// Brute force over the integer values (small boxes) of a model whose continuous part is solved
// by reference_qp with the integers fixed.
double reference_miqp(const Model& m) {
  std::vector<Index> ints;
  for (Index j = 0; j < m.num_cols(); ++j)
    if (is_integral_type(m.col_type[j])) ints.push_back(j);
  std::vector<double> v(ints.size());
  for (size_t t = 0; t < ints.size(); ++t) v[t] = m.col_lb[ints[t]];
  double best = kInf;
  for (;;) {
    Model f = m;
    for (size_t t = 0; t < ints.size(); ++t) {
      f.col_lb[ints[t]] = f.col_ub[ints[t]] = v[t];
      f.col_type[ints[t]] = VarType::Continuous;
    }
    best = std::min(best, reference_qp(f));
    size_t t = 0;
    while (t < ints.size() && v[t] == m.col_ub[ints[t]]) v[t] = m.col_lb[ints[t]], ++t;
    if (t == ints.size()) break;
    v[t] += 1.0;
  }
  return best;
}

}  // namespace

TEST(qp_active_set_matches_reference) {
  // The dual active-set method of the MIQP nodes: cold start (empty working set), warm starts
  // from the previous solve after tightening a bound (cached factorization reused), and the safe
  // bound never above the optimum.
  std::mt19937 rng(23);
  int solved = 0, warm = 0;
  for (int trial = 0; trial < 80; ++trial) {
    const Index n = 2 + static_cast<Index>(rng() % 4), m = static_cast<Index>(rng() % 4);
    Model md = random_qp(rng, n, m);
    if (trial % 3 == 0) {
      // A column without curvature (singular Q) with a finite box.
      md.q_row.clear(), md.q_col.clear(), md.q_val.clear();
      Model tmp = random_qp(rng, n, m);
      for (size_t k = 0; k < tmp.q_val.size(); ++k)
        if (tmp.q_row[k] != 0 && tmp.q_col[k] != 0) md.add_q(tmp.q_row[k], tmp.q_col[k], tmp.q_val[k]);
      md.col_lb[0] = -1.0, md.col_ub[0] = 2.0;
    }
    MipProblem p;
    CHECK(MipProblem::from_model(md, &p));
    ActiveSetQp as(p, {});
    std::vector<double> lb = p.col_lb, ub = p.col_ub;
    WorkingSet ws;
    std::vector<double> center(static_cast<size_t>(n), 0.0);
    uint64_t state = 0;
    for (int step = 0; step < 3; ++step) {
      Model mt = md;
      mt.col_lb = lb, mt.col_ub = ub;
      double ref;
      if (trial % 3 == 0) {
        // Singular Q: the barrier is the reference.
        const SolveResult r = solve(mt, quiet());
        ref = r.status == Status::Optimal ? r.objective : r.status == Status::Infeasible ? kInf : std::nan("");
        if (std::isnan(ref)) break;
      } else {
        ref = reference_qp(mt);
      }
      const ActiveSetResult a = as.solve(lb, ub, ws, center, 1000000000LL, state);
      if (!std::isfinite(ref)) {
        CHECK(a.status != ActiveSetResult::Status::Optimal);
        break;
      }
      CHECK(a.status == ActiveSetResult::Status::Optimal);
      if (a.status != ActiveSetResult::Status::Optimal) break;
      ++solved;
      if (step > 0) ++warm;
      CHECK(mt.violation(a.x).max() <= 1e-7);
      CHECK_NEAR(a.objective, ref, 1e-6 * (1.0 + std::abs(ref)));
      CHECK(a.safe.bound <= ref + 1e-7 * (1.0 + std::abs(ref)));
      CHECK(a.safe.bound >= ref - 1e-5 * (1.0 + std::abs(ref)));
      // Tighten the bound of a column through its current value (a branching).
      const Index j = static_cast<Index>(rng() % n);
      const double v = a.x[j];
      if (rng() % 2) ub[j] = v - 0.3;
      else lb[j] = v + 0.3;
      if (lb[j] > ub[j]) break;
      ws = a.ws, center = a.x, state = a.state;
    }
  }
  CHECK(solved >= 120);
  CHECK(warm >= 40);
}

TEST(miqp_matches_enumeration) {
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  int checked = 0;
  for (int trial = 0; trial < 30; ++trial) {
    const Index n = 3 + static_cast<Index>(rng() % 3), m = static_cast<Index>(rng() % 3);
    Model md = random_qp(rng, n, m);
    // Two or three integer columns with small boxes around the feasible region.
    const Index nint = 2 + static_cast<Index>(rng() % 2);
    for (Index j = 0; j < nint; ++j) {
      md.col_type[j] = VarType::Integer;
      md.col_lb[j] = -2.0;
      md.col_ub[j] = 2.0;
    }
    md.obj[0] += 2.0 * u(rng);  // move the continuous optimum off the integers
    const double ref = reference_miqp(md);
    SolverParams p = quiet();
    p.mip_rel_gap = 1e-9;
    p.mip_abs_gap = 1e-9;
    const SolveResult r = solve(md, p);
    if (std::getenv("QPDUMP") && (r.status != Status::Optimal || std::abs(r.objective - ref) > 1e-6)) {
      write_mps(md, std::string(std::getenv("QPDUMP")) + "/miqp" + std::to_string(trial) + ".mps");
      std::printf("trial %d status %s obj %.10g ref %.10g\n", trial, to_string(r.status), r.objective, ref);
    }
    if (!std::isfinite(ref)) {
      CHECK(r.status == Status::Infeasible);
      continue;
    }
    CHECK(r.status == Status::Optimal);
    if (r.status != Status::Optimal) continue;
    ++checked;
    CHECK(md.violation(r.x).max() <= 1e-6);
    CHECK_NEAR(r.objective, ref, 1e-6 * (1.0 + std::abs(ref)));
    CHECK(r.bound <= r.objective + 1e-9);
  }
  CHECK(checked >= 15);
}

TEST(miqp_pure_binary_and_limits) {
  // min sum (x_j - t_j)^2 + coupling, sum x <= 3, binary x: enumeration of all 2^8 points.
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  ModelBuilder b;
  const Index n = 8;
  for (Index j = 0; j < n; ++j) b.add_col("x" + std::to_string(j), -2.0 * u(rng), 0.0, 1.0, VarType::Binary);
  const Index r = b.add_row("card", -kInf, 3.0);
  for (Index j = 0; j < n; ++j) b.add_coef(r, j, 1.0);
  Model m = b.build();
  for (Index j = 0; j < n; ++j) {
    m.add_q(j, j, 1.0);
    if (j > 0) m.add_q(j, j - 1, 0.3);
  }
  double best = kInf;
  for (unsigned mask = 0; mask < (1u << n); ++mask) {
    std::vector<double> x(n);
    for (Index j = 0; j < n; ++j) x[j] = (mask >> j) & 1u;
    if (m.violation(x).max() <= 0.0) best = std::min(best, m.objective_value(x));
  }
  SolverParams p = quiet();
  p.mip_rel_gap = 0.0;
  p.mip_abs_gap = 1e-9;
  const SolveResult s = solve(m, p);
  CHECK(s.status == Status::Optimal);
  CHECK_NEAR(s.objective, best, 1e-7);
  // A node limit stops the search with a valid bound.
  p.node_limit = 1;
  const SolveResult t = solve(m, p);
  CHECK(t.bound <= best + 1e-7);
  CHECK(t.status == Status::NodeLimit || t.status == Status::Optimal);
  // Deterministic: the same result twice.
  p.node_limit = -1;
  const SolveResult a = solve(m, p), c = solve(m, p);
  CHECK(a.stats.nodes == c.stats.nodes && a.objective == c.objective);
}

TEST(miqp_semicontinuous_and_start) {
  // min (x - 0.5)^2 + (y - 3)^2 with y semi-continuous in {0} u [2, 5], x + y <= 4.
  ModelBuilder b;
  b.add_col("x", -1.0, 0.0, 10.0);
  b.add_col("y", -6.0, 2.0, 5.0, VarType::SemiContinuous);
  const Index r = b.add_row("c", -kInf, 4.0);
  b.add_coef(r, 0, 1.0);
  b.add_coef(r, 1, 1.0);
  Model m = b.build();
  m.obj_offset = 0.25 + 9.0;
  m.add_q(0, 0, 2.0);
  m.add_q(1, 1, 2.0);
  const SolveResult s = solve(m, quiet());
  CHECK(s.status == Status::Optimal);
  // Optimum: y = 3, x = 0.5 (x + y = 3.5 <= 4): objective 0.
  CHECK_NEAR(s.objective, 0.0, 1e-6);
  // A start solution is accepted (and is not worse than the optimum).
  const SolveResult t = solve(m, quiet(), {0.5, 3.0});
  CHECK(t.status == Status::Optimal);
  CHECK_NEAR(t.objective, 0.0, 1e-6);
}

TEST(barrier_fixed_columns_in_inequality_rows) {
  // min x + y + z s.t. 2 <= x + y <= 5, x + z >= 4 with x fixed at 1: y = 1, z = 3 (the slack
  // bounds of a row with a fixed column used to be shifted twice).
  const SparseMatrix A = SparseMatrix::from_triplets(3, 2, {0, 0, 1, 2}, {0, 1, 0, 1}, {1.0, 1.0, 1.0, 1.0});
  IpmOptions o;
  const IpmResult r = ipm_solve(A, {1.0, 1.0, 1.0}, {1.0, 0.0, 0.0}, {1.0, kInf, kInf}, {2.0, 4.0}, {5.0, kInf}, o);
  CHECK(r.status == IpmStatus::Optimal);
  CHECK_NEAR(r.objective, 5.0, 1e-7);
  CHECK(r.primal_infeas <= 1e-8);
  // The same with a Hessian on the free columns (QP): min x + y + z + y^2 + z^2.
  const SparseMatrix Q = SparseMatrix::from_triplets(3, 3, {1, 2}, {1, 2}, {2.0, 2.0});
  const IpmResult q = ipm_solve(A, {1.0, 1.0, 1.0}, {1.0, 0.0, 0.0}, {1.0, kInf, kInf}, {2.0, 4.0}, {5.0, kInf}, o, &Q);
  CHECK(q.status == IpmStatus::Optimal);
  CHECK_NEAR(q.objective, 1.0 + 1.0 + 1.0 + 3.0 + 9.0, 1e-6);
}

TEST(qp_large_duals_yao) {
  // Maros-Meszaros YAO, generated: min 1/2 |x - t|^2, t_j = sin((j + 1) / n), with convex second
  // differences x_j - 2 x_{j+1} + x_{j+2} >= 0, x_0 >= 0.08 and the last two columns fixed at 0.
  // The optimal duals are ~1e5 against costs <= 1: with the dual regularization in the primal
  // equations the barrier stalled at the iteration limit. Optimum: 197.70425594 - 273.125.
  const Index n = 2002;
  ModelBuilder b;
  for (Index j = 0; j < n; ++j) {
    const double lb = j == 0 ? 0.08 : j >= n - 2 ? 0.0 : -kInf, ub = j >= n - 2 ? 0.0 : kInf;
    b.add_col("x" + std::to_string(j), -std::sin((j + 1.0) / n), lb, ub);
  }
  for (Index i = 0; i + 2 < n; ++i) {
    const Index r = b.add_row("r" + std::to_string(i), 0.0, kInf);
    b.add_coef(r, i, 1.0);
    b.add_coef(r, i + 1, -2.0);
    b.add_coef(r, i + 2, 1.0);
  }
  Model m = b.build();
  for (Index j = 0; j < n; ++j) m.add_q(j, j, 1.0);
  const SolveResult r = solve(m, quiet());
  CHECK(r.status == Status::Optimal);
  CHECK_NEAR(r.objective, 197.70425594031065 - 273.125, 1e-6 * 75.42);
  CHECK(m.violation(r.x).max() <= 1e-7);
  CHECK(r.stats.barrier_iterations <= 100);
}

namespace {

// Random convex QP with structure the QP presolve acts on: fixed columns (also inside Q), columns
// without Hessian entries (boxed), an empty column, singleton rows, a free row and rows implied
// by the bounds, besides general ranged / one-sided rows. Feasible by construction (x0).
Model random_structured_qp(std::mt19937& rng, Index n, Index m) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  ModelBuilder b;
  std::vector<double> x0(n), lo(n), hi(n);
  std::vector<char> inq(n);
  for (Index j = 0; j < n; ++j) {
    x0[j] = u(rng);
    const int kind = static_cast<int>(rng() % 6);
    inq[j] = rng() % 3 != 0;
    lo[j] = kind == 3 && inq[j] ? -kInf : x0[j] - std::abs(u(rng)) - 0.1;
    hi[j] = kind == 2 && inq[j] ? kInf : x0[j] + std::abs(u(rng)) + 0.1;
    if (kind == 5) lo[j] = hi[j] = x0[j];  // fixed
    b.add_col("x" + std::to_string(j), 3.0 * u(rng), lo[j], hi[j]);
  }
  const Index empty = b.add_col("e", u(rng), -1.0, 1.0);
  for (Index i = 0; i < m; ++i) {
    const int kind = static_cast<int>(rng() % 8);
    if (kind == 0) {  // singleton row
      const Index j = static_cast<Index>(rng() % n);
      const double a = rng() % 2 ? 2.0 : -0.5;
      const double r = b.add_row("s" + std::to_string(i), a * x0[j] - 0.3 * std::abs(u(rng)),
                                 rng() % 2 ? kInf : a * x0[j] + 0.3 * std::abs(u(rng)));
      b.add_coef(static_cast<Index>(r), j, a);
      continue;
    }
    double act = 0.0, amax = 0.0;
    std::vector<double> a(n);
    for (Index j = 0; j < n; ++j) {
      a[j] = rng() % 3 == 0 ? 0.0 : u(rng);
      act += a[j] * x0[j];
      amax += a[j] > 0.0 ? a[j] * hi[j] : a[j] * lo[j];
    }
    double rl = act - std::abs(u(rng)), rh = act + std::abs(u(rng));
    if (kind == 1) rl = -kInf;
    if (kind == 2) rh = kInf;
    if (kind == 3) rl = -kInf, rh = kInf;                            // free row
    if (kind == 4 && is_finite(amax)) rl = -kInf, rh = amax + 0.5;  // implied by the bounds
    const Index r = b.add_row("r" + std::to_string(i), rl, rh);
    for (Index j = 0; j < n; ++j)
      if (a[j] != 0.0) b.add_coef(r, j, a[j]);
  }
  Model md = b.build();
  Dense F(n, std::vector<double>(n));
  for (auto& row : F)
    for (double& v : row) v = u(rng);
  for (Index i = 0; i < n; ++i) {
    if (!inq[i]) continue;
    for (Index j = 0; j <= i; ++j) {
      if (!inq[j]) continue;
      double s = i == j ? 0.1 : 0.0;
      for (Index k = 0; k < n; ++k) s += F[i][k] * F[j][k];
      md.add_q(i, j, s);
    }
  }
  md.add_q(empty, empty, 0.5 + std::abs(u(rng)));
  return md;
}

// Complementarity of a value v in [lo, hi] with its multiplier z (minimization form: z >= 0
// pushes against lo, z <= 0 against hi): z+ (v - lo) + z- (hi - v), and |z| on a side without bound.
double complementarity(double v, double lo, double hi, double z) {
  const double zp = std::max(z, 0.0), zm = std::max(-z, 0.0);
  return (is_finite(lo) ? zp * std::abs(v - lo) : zp) + (is_finite(hi) ? zm * std::abs(hi - v) : zm);
}

// KKT residual of (x, y, d) for a minimization (sign = 1) or maximization (sign = -1) model:
// max of the relative |d - (c + Q x - A^T y)| and the complementarity of the columns with d and
// of the row activities with y (an interior point's final iterate: products ~ mu).
double kkt_residual(const Model& m, const SolveResult& r, double sign) {
  const Index n = m.num_cols(), rows = m.num_rows();
  if (!r.has_duals || static_cast<Index>(r.row_dual.size()) != rows || static_cast<Index>(r.reduced_cost.size()) != n) return kInf;
  const Dense Q = dense_q(m);
  std::vector<double> act(rows, 0.0);
  double res = 0.0;
  for (Index j = 0; j < n; ++j) {
    double g = m.obj[j];
    for (Index k = 0; k < n; ++k) g += Q[j][k] * r.x[k];
    for (Index k = m.A.begin(j); k < m.A.end(j); ++k) {
      g -= m.A.value[k] * r.row_dual[m.A.index[k]];
      act[m.A.index[k]] += m.A.value[k] * r.x[j];
    }
    res = std::max(res, std::abs(g - r.reduced_cost[j]) / (1.0 + std::abs(g)));
    const double xj = r.x[j];
    res = std::max(res, complementarity(xj, m.col_lb[j], m.col_ub[j], sign * r.reduced_cost[j]));
  }
  for (Index i = 0; i < rows; ++i)
    res = std::max(res, complementarity(act[i], m.row_lb[i], m.row_ub[i], sign * r.row_dual[i]));
  return res;
}

}  // namespace

TEST(qp_duals_kkt) {
  // Row duals and reduced costs of continuous QPs satisfy the KKT conditions, with and without
  // presolve, for minimization and maximization; MIQPs and LPs report none.
  std::mt19937 rng(23);
  int checked = 0;
  for (int trial = 0; trial < 80; ++trial) {
    const Index n = 2 + static_cast<Index>(rng() % 6), m = 1 + static_cast<Index>(rng() % 6);
    const Model md = trial % 2 ? random_structured_qp(rng, n, m) : random_qp(rng, n, m);
    const double ref = reference_qp(md);
    for (int presolve = 0; presolve < 2; ++presolve) {
      SolverParams sp = quiet();
      sp.presolve = presolve != 0;
      const SolveResult r = solve(md, sp);
      if (!std::isfinite(ref)) {
        CHECK(r.status == Status::Infeasible);
        CHECK(!r.has_duals && r.row_dual.empty());
        continue;
      }
      CHECK(r.status == Status::Optimal);
      if (r.status != Status::Optimal) continue;
      CHECK_NEAR(r.objective, ref, 1e-7 * (1.0 + std::abs(ref)));
      const double kkt = kkt_residual(md, r, 1.0);
      CHECK(kkt <= 1e-6);
      if (kkt > 1e-6) std::printf("  trial %d presolve %d: KKT residual %.3g\n", trial, presolve, kkt);
      Model mx = md;
      mx.sense = ObjSense::Maximize;
      for (double& c : mx.obj) c = -c;
      for (double& v : mx.q_val) v = -v;
      const SolveResult rx = solve(mx, sp);
      CHECK(rx.status == Status::Optimal);
      if (rx.status == Status::Optimal) CHECK(kkt_residual(mx, rx, -1.0) <= 1e-6);
      ++checked;
    }
  }
  CHECK(checked >= 100);
  // No duals for an MIQP.
  Model mi = random_qp(rng, 3, 2);
  mi.col_type[0] = VarType::Integer;
  const SolveResult ri = solve(mi, quiet());
  CHECK(!ri.has_duals && ri.row_dual.empty() && ri.reduced_cost.empty());
}

TEST(qp_presolve_reductions) {
  // min (x - 1)^2 + (y - 2)^2 + x y + z + w^2 with z fixed at 3 (in Q with x), w only in Q,
  // a singleton row 2 y <= 1 (active: y = 0.5 with the dual on the row), a row x + y <= 10
  // implied by the bounds, an empty row and a free row.
  ModelBuilder b;
  b.add_col("x", -2.0, 0.0, 4.0);
  b.add_col("y", -4.0, -1.0, 4.0);
  b.add_col("z", 1.0, 3.0, 3.0);
  b.add_col("w", 0.0, -5.0, 5.0);
  const Index r0 = b.add_row("single", -kInf, 1.0);
  b.add_coef(r0, 1, 2.0);
  const Index r1 = b.add_row("implied", -kInf, 10.0);
  b.add_coef(r1, 0, 1.0);
  b.add_coef(r1, 1, 1.0);
  b.add_row("empty", -1.0, 1.0);
  const Index r3 = b.add_row("free", -kInf, kInf);
  b.add_coef(r3, 0, 1.0);
  const Index r4 = b.add_row("withfixed", 4.0, kInf);  // x + z >= 4: x >= 1
  b.add_coef(r4, 0, 1.0);
  b.add_coef(r4, 2, 1.0);
  Model m = b.build();
  m.add_q(0, 0, 2.0);
  m.add_q(1, 1, 2.0);
  m.add_q(1, 0, 1.0);
  m.add_q(2, 0, 0.5);
  m.add_q(3, 3, 2.0);
  m.add_q(2, 2, 1.0);
  m.obj_offset = 5.0;
  const double ref = reference_qp(m);
  for (int presolve = 0; presolve < 2; ++presolve) {
    SolverParams sp = quiet();
    sp.presolve = presolve != 0;
    const SolveResult r = solve(m, sp);
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, ref, 1e-7);
    CHECK_NEAR(r.x[1], 0.5, 1e-7);
    CHECK_NEAR(r.x[2], 3.0, 0.0);
    CHECK_NEAR(r.x[3], 0.0, 1e-7);
    CHECK(kkt_residual(m, r, 1.0) <= 1e-7);
    CHECK(r.row_dual[r0] < -1e-3);  // the singleton row holds y at its upper bound
    CHECK_NEAR(r.row_dual[r1], 0.0, 1e-7);
  }
  // A cross term with a fixed column that has no square term is linear once presolve removes
  // the column; without presolve the convexity test rejects it.
  m.q_val.pop_back();
  m.q_row.pop_back();
  m.q_col.pop_back();
  CHECK(solve(m, quiet()).status == Status::Optimal);
  SolverParams np = quiet();
  np.presolve = false;
  // (v0.10: by spatial branch-and-bound where the factors are bounded, else NonConvex; never a wrong optimum.)
  const SolveResult rn = solve(m, np);
  CHECK(rn.status == Status::NonConvex || rn.status == Status::Optimal);
  if (rn.status == Status::Optimal) CHECK_NEAR(rn.objective, solve(m, quiet()).objective, 1e-5);
  // Infeasible singleton rows: 2 <= x <= 1 through two rows.
  ModelBuilder bi;
  bi.add_col("x", 0.0, 0.0, 5.0);
  bi.add_col("y", 0.0, 0.0, 5.0);
  const Index a0 = bi.add_row("a", 2.0, kInf);
  bi.add_coef(a0, 0, 1.0);
  const Index a1 = bi.add_row("b", -kInf, 1.0);
  bi.add_coef(a1, 0, 1.0);
  Model mi = bi.build();
  mi.add_q(1, 1, 1.0);
  CHECK(solve(mi, quiet()).status == Status::Infeasible);
  // Everything fixed by presolve: min (x - 1)^2 + y over 0 <= y <= 2 (y dual-fixed at 0).
  ModelBuilder bf;
  bf.add_col("x", -2.0, -kInf, kInf);
  bf.add_col("y", 1.0, 0.0, 2.0);
  Model mf = bf.build();
  mf.add_q(0, 0, 2.0);
  const SolveResult rf = solve(mf, quiet());
  CHECK(rf.status == Status::Optimal);
  CHECK_NEAR(rf.objective, -1.0, 1e-12);
  CHECK(kkt_residual(mf, rf, 1.0) <= 1e-12);
}

TEST(qp_maros_meszaros_samples) {
  // A few small problems of the Maros-Meszaros convex QP test set, converted to MPS by
  // benchmarks/tools/maros_meszaros.py (from OSQP's .mat versions). Expected optima: HiGHS's QP
  // solver. VALUES is the set's nonconvex problem.
  struct Case {
    const char* name;
    double opt;
  };
  const Case cases[] = {{"TAME", 0.0},
                        {"HS21", -99.96},
                        {"ZECEVIC2", -4.125},
                        {"HS35", 1.0 / 9.0},
                        {"HS51", 0.0},
                        {"HS76", -4.68181818181818},
                        {"GENHS28", 0.9271736937663911},
                        {"LOTSCHD", 2398.4158914488944},
                        {"QAFIRO", -1.5907817938917632},
                        {"CVXQP1_S", 11590.718119426765},
                        {"CVXQP2_S", 8120.940477250689},
                        {"QPCBLEND", -0.007842543074500372},
                        {"QADLITTL", 480318.8585447984},
                        {"DUAL1", 0.035012965733469216}};
  const std::string dir = std::string(MILP_SOURCE_DIR) + "/tests/data/maros_meszaros/";
  for (const Case& c : cases) {
    const Model m = read_mps(dir + c.name + ".mps");
    const SolveResult r = solve(m, quiet());
    CHECK(r.status == Status::Optimal);
    CHECK_NEAR(r.objective, c.opt, 1e-6 * std::max(1.0, std::abs(c.opt)));
    CHECK(m.violation(r.x).max() <= 1e-6);
  }
  // VALUES is not convex: spatial branch-and-bound (v0.10) under a short limit; whatever it reports is consistent.
  SolverParams vp = quiet();
  vp.time_limit = 2.0;
  const Model values = read_mps(dir + "VALUES.mps");
  const SolveResult rv = solve(values, vp);
  CHECK(rv.status == Status::Optimal || rv.status == Status::TimeLimit || rv.status == Status::NonConvex);
  if (rv.has_solution) {
    CHECK(values.violation(rv.x).max() <= 1e-6);
    CHECK(rv.bound <= rv.objective + 1e-6 * std::max(1.0, std::abs(rv.objective)));
  }
}

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value ? value : "");
#else
  if (value) setenv(name, value, 1);
  else unsetenv(name);
#endif
}

}  // namespace

// Separable convex objectives on integer columns (1/2 q_j x_j^2 + c x on general integers, free integers and
// binaries, both senses) are solved as a MILP with tangent rows (solver.cpp, solve_square_milp); the MIQP
// search (MILP_SQ_MILP=0) must agree, also with one-wide tangent windows around 0 (MILP_SQ_WINDOW=1,
// MILP_SQ_CENTER=0: several tangent rounds).
TEST(square_milp_matches_miqp) {
  std::mt19937 rng(11);
  auto uni = [&](int lo, int hi) { return lo + static_cast<int>(rng() % static_cast<unsigned>(hi - lo + 1)); };
  int checked = 0, infeasible = 0;
  for (int trial = 0; trial < 40; ++trial) {
    const bool maximize = trial % 4 == 3;
    ModelBuilder b;
    const Index n = static_cast<Index>(uni(4, 8));
    for (Index j = 0; j < n; ++j) {
      const int kind = uni(0, 5);
      const std::string name = "x" + std::to_string(j);
      if (kind == 0) b.add_col(name, uni(-20, 20), 0.0, 1.0, VarType::Binary);
      else if (kind == 1) b.add_col(name, uni(-20, 20), -kInf, kInf, VarType::Integer);
      else b.add_col(name, uni(-20, 20), -uni(0, 15), uni(0, 25), VarType::Integer);
    }
    const int m = uni(1, 3);
    for (int i = 0; i < m; ++i) {
      const Index r = b.add_row("r" + std::to_string(i), -uni(0, 30), uni(0, 30));
      for (Index j = 0; j < n; ++j) {
        if (uni(0, 2) > 0) b.add_coef(r, j, uni(-3, 3));
      }
    }
    Model md = b.build();
    md.sense = maximize ? ObjSense::Maximize : ObjSense::Minimize;
    for (Index j = 0; j < n; ++j) md.add_q(j, j, (maximize ? -1.0 : 1.0) * uni(1, 4));
    SolverParams p = quiet();
    p.mip_rel_gap = 0.0;
    p.mip_abs_gap = 1e-9;
    const SolveResult a = solve(md, p);
    set_env("MILP_SQ_MILP", "0");
    const SolveResult c = solve(md, p);
    set_env("MILP_SQ_MILP", nullptr);
    set_env("MILP_SQ_WINDOW", "1");
    set_env("MILP_SQ_CENTER", "0");
    const SolveResult w = solve(md, p);
    set_env("MILP_SQ_WINDOW", nullptr);
    set_env("MILP_SQ_CENTER", nullptr);
    // The MIQP search is the reference; a numerical failure of it is not a mismatch (MSVC, trial 10: NUMERIC there,
    // OPTIMAL for both tangent MILPs). The two tangent MILPs must still agree.
    if (c.status == Status::NumericalError) {
      CHECK(a.status == w.status);
      if (a.status == Status::Optimal && w.status == Status::Optimal) CHECK_NEAR(a.objective, w.objective, 1e-6 * (1.0 + std::abs(a.objective)));
      continue;
    }
    if (!CHECK(a.status == c.status) || !CHECK(w.status == c.status)) {
      std::printf("  trial %d: %s (tangent MILP) / %s (window 1) vs %s (MIQP)\n", trial, to_string(a.status),
                  to_string(w.status), to_string(c.status));
      continue;
    }
    if (c.status == Status::Infeasible) {
      ++infeasible;
      continue;
    }
    if (c.status != Status::Optimal) continue;
    ++checked;
    for (const SolveResult* s : {&a, &w}) {
      CHECK(md.violation(s->x).max() <= 1e-6);
      CHECK_NEAR(s->objective, c.objective, 1e-6 * (1.0 + std::abs(c.objective)));
      CHECK_NEAR(s->objective, md.objective_value(s->x), 1e-9 * (1.0 + std::abs(s->objective)));
      CHECK_NEAR(s->bound, s->objective, 1e-6 * (1.0 + std::abs(s->objective)));
    }
  }
  CHECK(checked >= 20);
  if (std::getenv("MILP_TEST_VERBOSE")) std::printf("  square MILP: %d optimal, %d infeasible\n", checked, infeasible);
}

// Squared errors declared continuous but implied integer (e_p = sum_f x[f,p] - T_p over integer x): the
// tangent MILP route still applies, with the same optimum as for integer errors and as the MIQP search.
TEST(square_milp_implied_integer_errors) {
  for (int trial = 0; trial < 10; ++trial) {
    double obj[3] = {0.0, 0.0, 0.0};
    for (int variant = 0; variant < 3; ++variant) {  // integer errors, continuous errors, continuous + MIQP search
      std::mt19937 r2(static_cast<unsigned>(100 + trial));
      auto u2 = [&](int lo, int hi) { return lo + static_cast<int>(r2() % static_cast<unsigned>(hi - lo + 1)); };
      ModelBuilder b;
      const int plants = u2(2, 4), products = u2(3, 6);
      std::vector<std::vector<Index>> x(plants, std::vector<Index>(products));
      for (int f = 0; f < plants; ++f) {
        const Index cap = b.add_row("cap" + std::to_string(f), -kInf, u2(10, 40));
        for (int p = 0; p < products; ++p) {
          x[f][p] = b.add_col("x" + std::to_string(f) + "_" + std::to_string(p), 0, 0, u2(5, 20), VarType::Integer);
          b.add_coef(cap, x[f][p], u2(1, 4));
        }
      }
      std::vector<Index> e(products);
      for (int p = 0; p < products; ++p) {
        e[p] = b.add_col("e" + std::to_string(p), 0, -kInf, kInf, variant == 0 ? VarType::Integer : VarType::Continuous);
        const double target = u2(5, 40);
        const Index r = b.add_row("t" + std::to_string(p), target, target);
        for (int f = 0; f < plants; ++f) b.add_coef(r, x[f][p], 1.0);
        b.add_coef(r, e[p], -1.0);
      }
      Model md = b.build();
      for (int p = 0; p < products; ++p) md.add_q(e[p], e[p], 2.0);
      SolverParams prm = quiet();
      prm.mip_rel_gap = 0.0;
      prm.mip_abs_gap = 1e-9;
      if (variant == 2) set_env("MILP_SQ_MILP", "0");
      const SolveResult s = solve(md, prm);
      if (variant == 2) set_env("MILP_SQ_MILP", nullptr);
      REQUIRE(s.status == Status::Optimal);
      CHECK(md.violation(s.x).max() <= 1e-6);
      obj[variant] = s.objective;
    }
    CHECK_NEAR(obj[1], obj[0], 1e-6 * (1.0 + std::abs(obj[0])));
    CHECK_NEAR(obj[2], obj[0], 1e-6 * (1.0 + std::abs(obj[0])));
  }
}
