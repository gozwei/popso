// Tests for the sparse Cholesky factorization and the AMD ordering.

#include <algorithm>
#include <cmath>
#include <random>

#include "milp/lp/cholesky.hpp"
#include "milp/lp/ordering.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

// Dense symmetric matrix (row-major) and its lower triangle in CSC form.
struct SymMatrix {
  Index n = 0;
  std::vector<double> dense;
  std::vector<Index> start, index;
  std::vector<double> value;
};

void to_lower_csc(SymMatrix& M) {
  M.start.assign(1, 0);
  M.index.clear();
  M.value.clear();
  for (Index j = 0; j < M.n; ++j) {
    for (Index i = j; i < M.n; ++i) {
      const double v = M.dense[static_cast<size_t>(i) * M.n + j];
      if (v != 0.0 || i == j) {
        M.index.push_back(i);
        M.value.push_back(v);
      }
    }
    M.start.push_back(static_cast<Index>(M.index.size()));
  }
}

// M = B B^T + shift I with B sparse random (n x k).
SymMatrix random_spd(std::mt19937& rng, Index n, Index k, double density, double shift) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> B(static_cast<size_t>(n) * k, 0.0);
  for (auto& v : B) {
    if (u(rng) * 0.5 + 0.5 < density) v = u(rng);
  }
  SymMatrix M;
  M.n = n;
  M.dense.assign(static_cast<size_t>(n) * n, 0.0);
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j < n; ++j) {
      double s = 0.0;
      for (Index t = 0; t < k; ++t) s += B[static_cast<size_t>(i) * k + t] * B[static_cast<size_t>(j) * k + t];
      M.dense[static_cast<size_t>(i) * n + j] = s + (i == j ? shift : 0.0);
    }
  to_lower_csc(M);
  return M;
}

double residual(const SymMatrix& M, const std::vector<double>& x, const std::vector<double>& b) {
  double r = 0.0;
  for (Index i = 0; i < M.n; ++i) {
    double s = 0.0;
    for (Index j = 0; j < M.n; ++j) s += M.dense[static_cast<size_t>(i) * M.n + j] * x[j];
    r = std::max(r, std::abs(s - b[i]));
  }
  return r;
}

// Dense Gaussian elimination with partial pivoting (reference solution).
std::vector<double> dense_solve(const SymMatrix& M, std::vector<double> b) {
  const Index n = M.n;
  std::vector<double> A = M.dense;
  for (Index c = 0; c < n; ++c) {
    Index p = c;
    for (Index i = c + 1; i < n; ++i)
      if (std::abs(A[static_cast<size_t>(i) * n + c]) > std::abs(A[static_cast<size_t>(p) * n + c])) p = i;
    for (Index j = 0; j < n; ++j) std::swap(A[static_cast<size_t>(c) * n + j], A[static_cast<size_t>(p) * n + j]);
    std::swap(b[c], b[p]);
    for (Index i = c + 1; i < n; ++i) {
      const double f = A[static_cast<size_t>(i) * n + c] / A[static_cast<size_t>(c) * n + c];
      for (Index j = c; j < n; ++j) A[static_cast<size_t>(i) * n + j] -= f * A[static_cast<size_t>(c) * n + j];
      b[i] -= f * b[c];
    }
  }
  std::vector<double> x(static_cast<size_t>(n));
  for (Index i = n - 1; i >= 0; --i) {
    double s = b[i];
    for (Index j = i + 1; j < n; ++j) s -= A[static_cast<size_t>(i) * n + j] * x[j];
    x[i] = s / A[static_cast<size_t>(i) * n + i];
  }
  return x;
}

bool is_permutation(const std::vector<Index>& p, Index n) {
  if (static_cast<Index>(p.size()) != n) return false;
  std::vector<char> seen(static_cast<size_t>(n), 0);
  for (Index v : p) {
    if (v < 0 || v >= n || seen[v]) return false;
    seen[v] = 1;
  }
  return true;
}

}  // namespace

TEST(cholesky_random_spd_vs_dense) {
  std::mt19937 rng(11);
  for (int t = 0; t < 60; ++t) {
    const Index n = 1 + static_cast<Index>(rng() % 120);
    const Index k = 1 + static_cast<Index>(rng() % 150);
    const double density = 0.02 + 0.2 * (rng() % 100) / 100.0;
    SymMatrix M = random_spd(rng, n, k, density, 0.1 + (rng() % 3));
    std::vector<double> b(static_cast<size_t>(n));
    std::uniform_real_distribution<double> u(-5.0, 5.0);
    for (auto& v : b) v = u(rng);
    const std::vector<double> ref = dense_solve(M, b);
    for (int mode = 0; mode < 3; ++mode) {
      SparseCholesky ch;
      std::vector<Index> ident(static_cast<size_t>(n));
      for (Index i = 0; i < n; ++i) ident[i] = i;
      if (mode == 2) {
        ch.relax_small = 1;  // fundamental supernodes only
        ch.relax_cols = 1;
      }
      REQUIRE(ch.analyze(n, M.start, M.index, mode == 1 ? &ident : nullptr));
      REQUIRE(is_permutation(ch.perm(), n));
      REQUIRE(ch.factorize(M.value));
      CHECK(ch.num_dropped() == 0);
      std::vector<double> x = b;
      ch.solve(x);
      double err = 0.0;
      for (Index i = 0; i < n; ++i) err = std::max(err, std::abs(x[i] - ref[i]) / (1.0 + std::abs(ref[i])));
      CHECK(err < 1e-8);
      CHECK(residual(M, x, b) < 1e-8);
    }
  }
}

TEST(cholesky_refactorize_same_pattern) {
  std::mt19937 rng(5);
  SymMatrix M = random_spd(rng, 80, 60, 0.08, 1.0);
  SparseCholesky ch;
  REQUIRE(ch.analyze(M.n, M.start, M.index));
  for (int t = 0; t < 5; ++t) {
    // New values on the same pattern: scale the diagonal.
    std::vector<double> val = M.value;
    SymMatrix M2 = M;
    for (Index j = 0; j < M.n; ++j) {
      const double extra = 0.5 * t;
      val[M.start[j]] += extra;  // the diagonal comes first in each column
      M2.dense[static_cast<size_t>(j) * M.n + j] += extra;
    }
    REQUIRE(ch.factorize(val));
    std::vector<double> b(static_cast<size_t>(M.n), 1.0), x = b;
    ch.solve(x);
    CHECK(residual(M2, x, b) < 1e-9);
  }
}

TEST(cholesky_semidefinite_drops_pivots) {
  // M = B B^T with B of rank 5 (n = 30): consistent right-hand sides are still solved.
  std::mt19937 rng(7);
  const Index n = 30, k = 5;
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> B(static_cast<size_t>(n) * k);
  for (auto& v : B) v = u(rng);
  SymMatrix M;
  M.n = n;
  M.dense.assign(static_cast<size_t>(n) * n, 0.0);
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j < n; ++j)
      for (Index t = 0; t < k; ++t)
        M.dense[static_cast<size_t>(i) * n + j] += B[static_cast<size_t>(i) * k + t] * B[static_cast<size_t>(j) * k + t];
  to_lower_csc(M);
  SparseCholesky ch;
  ch.pivot_tol = 1e-10;
  REQUIRE(ch.analyze(n, M.start, M.index));
  REQUIRE(ch.factorize(M.value));
  CHECK(ch.num_dropped() == n - k);
  // b = M z for a random z is in the range of M.
  std::vector<double> z(static_cast<size_t>(n)), b(static_cast<size_t>(n), 0.0);
  for (auto& v : z) v = u(rng);
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j < n; ++j) b[i] += M.dense[static_cast<size_t>(i) * n + j] * z[j];
  std::vector<double> x = b;
  ch.solve(x);
  CHECK(residual(M, x, b) < 1e-6);
}

TEST(cholesky_quasidefinite_signed_pivots) {
  // K = [-D1 A^T; A D2] (D1, D2 > 0) is quasi-definite: L S L^T exists for every ordering.
  std::mt19937 rng(19);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int t = 0; t < 40; ++t) {
    const Index n = 1 + static_cast<Index>(rng() % 50), m = 1 + static_cast<Index>(rng() % 40), N = n + m;
    SymMatrix K;
    K.n = N;
    K.dense.assign(static_cast<size_t>(N) * N, 0.0);
    for (Index j = 0; j < n; ++j) K.dense[static_cast<size_t>(j) * N + j] = -(0.01 + std::abs(u(rng)) * (t % 2 ? 1e4 : 1.0));
    for (Index i = 0; i < m; ++i) K.dense[static_cast<size_t>(n + i) * N + n + i] = 1e-6 + std::abs(u(rng)) * 1e-2;
    for (Index i = 0; i < m; ++i)
      for (Index j = 0; j < n; ++j)
        if (rng() % 100 < 25) {
          const double a = u(rng);
          K.dense[static_cast<size_t>(n + i) * N + j] = a;
          K.dense[static_cast<size_t>(j) * N + n + i] = a;
        }
    to_lower_csc(K);
    std::vector<int8_t> signs(static_cast<size_t>(N), 1);
    for (Index j = 0; j < n; ++j) signs[j] = -1;
    std::vector<double> b(static_cast<size_t>(N));
    for (auto& v : b) v = u(rng);
    const std::vector<double> ref = dense_solve(K, b);
    for (int mode = 0; mode < 2; ++mode) {
      SparseCholesky ch;
      std::vector<Index> ident(static_cast<size_t>(N));
      for (Index i = 0; i < N; ++i) ident[i] = i;
      REQUIRE(ch.analyze(N, K.start, K.index, mode ? &ident : nullptr, &signs));
      REQUIRE(ch.factorize(K.value));
      CHECK(ch.num_dropped() == 0);
      std::vector<double> x = b;
      ch.solve(x);
      double err = 0.0;
      for (Index i = 0; i < N; ++i) err = std::max(err, std::abs(x[i] - ref[i]) / (1.0 + std::abs(ref[i])));
      CHECK(err < 1e-6);
      CHECK(residual(K, x, b) < 1e-8 * (1.0 + 1e4));
    }
  }
}

TEST(amd_grid_reduces_fill) {
  // 2D grid Laplacian (5-point stencil), 30 x 30: AMD fill is far below the natural ordering's.
  const Index g = 30, n = g * g;
  SymMatrix M;
  M.n = n;
  M.start.assign(1, 0);
  for (Index j = 0; j < n; ++j) {
    const Index r = j / g, c = j % g;
    M.index.push_back(j);
    M.value.push_back(4.0);
    if (c + 1 < g) M.index.push_back(j + 1), M.value.push_back(-1.0);
    if (r + 1 < g) M.index.push_back(j + g), M.value.push_back(-1.0);
    M.start.push_back(static_cast<Index>(M.index.size()));
  }
  std::vector<Index> ident(static_cast<size_t>(n));
  for (Index i = 0; i < n; ++i) ident[i] = i;
  SparseCholesky nat, amd;
  REQUIRE(nat.analyze(n, M.start, M.index, &ident));
  REQUIRE(amd.analyze(n, M.start, M.index));
  CHECK(is_permutation(amd.perm(), n));
  CHECK(amd.flops() < 0.5 * nat.flops());
  REQUIRE(amd.factorize(M.value));
  std::vector<double> b(static_cast<size_t>(n), 1.0), x = b;
  amd.solve(x);
  // Residual of the stencil.
  double r = 0.0;
  for (Index j = 0; j < n; ++j) {
    const Index rr = j / g, c = j % g;
    double s = 4.0 * x[j];
    if (c > 0) s -= x[j - 1];
    if (c + 1 < g) s -= x[j + 1];
    if (rr > 0) s -= x[j - g];
    if (rr + 1 < g) s -= x[j + g];
    r = std::max(r, std::abs(s - 1.0));
  }
  CHECK(r < 1e-10);
}

TEST(amd_permutation_edge_cases) {
  // Empty, diagonal-only, a star (one dense node) and disconnected pieces.
  {
    std::vector<Index> st{0}, idx;
    CHECK(amd_order(0, st, idx).empty());
  }
  {
    const Index n = 10;
    std::vector<Index> st, idx;
    for (Index j = 0; j <= n; ++j) st.push_back(j);
    for (Index j = 0; j < n; ++j) idx.push_back(j);
    CHECK(is_permutation(amd_order(n, st, idx), n));
  }
  {
    // Star: node 0 adjacent to everybody; it must be ordered last (no fill at all).
    const Index n = 400;
    std::vector<Index> st{0}, idx;
    for (Index i = 1; i < n; ++i) idx.push_back(i);
    st.push_back(static_cast<Index>(idx.size()));
    for (Index j = 1; j < n; ++j) st.push_back(static_cast<Index>(idx.size()));
    const std::vector<Index> p = amd_order(n, st, idx);
    REQUIRE(is_permutation(p, n));
    CHECK(p.back() == 0);
  }
  {
    std::mt19937 rng(3);
    for (int t = 0; t < 30; ++t) {
      const Index n = 1 + static_cast<Index>(rng() % 200);
      std::vector<Index> st{0}, idx;
      for (Index j = 0; j < n; ++j) {
        const Index deg = static_cast<Index>(rng() % 6);
        for (Index d = 0; d < deg; ++d) idx.push_back(static_cast<Index>(rng() % n));
        st.push_back(static_cast<Index>(idx.size()));
      }
      CHECK(is_permutation(amd_order(n, st, idx), n));
    }
  }
}
