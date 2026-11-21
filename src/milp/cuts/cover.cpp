// Lifted knapsack cover cuts.
//
// A row side a x <= b is relaxed to a binary knapsack sum a_j z_j <= b', a_j > 0:
// non-binary variables are bounded out (a_j x_j >= a_j l_j for a_j > 0, a_j u_j
// for a_j < 0) and binaries with negative coefficients are complemented. For a
// cover C (sum_C a_j > b') with cover weights sorted a_1 >= a_2 >= ... and prefix
// sums A_h, the inequality
//   sum_{C} z_j + sum_{j not in C} h_j z_j <= |C| - 1,   h_j = max{h : A_h <= a_j}
// is valid: a point with S in C and T outside C chosen and |S| + sum_T h_j >= |C|
// weighs at least (A_|C| - A_K) + A_K > b' with K = sum_T h_j (A is superadditive).

#include <algorithm>
#include <cmath>

#include "milp/cuts/cut.hpp"

namespace milp {

namespace {

class CoverSeparator : public Separator {
 public:
  const char* name() const override { return "cover"; }
  void separate(const SepaContext& ctx, std::vector<Cut>& out) override;

 private:
  struct Item {
    Index j;
    double a;    // > 0
    bool comp;   // z = 1 - x
    double z;    // LP value of z
  };
  void separate_side(const SepaContext& ctx, Index row, double sign, std::vector<Cut>& out);
  std::vector<Item> items_;
};

bool is_binary(const MipProblem& p, Index j) { return p.is_int[j] && p.col_lb[j] == 0.0 && p.col_ub[j] == 1.0; }

void CoverSeparator::separate(const SepaContext& ctx, std::vector<Cut>& out) {
  const MipProblem& p = ctx.p;
  const SparseMatrix& R = ctx.lp.scaled_row_matrix();  // the pattern only
  for (Index i = 0; i < ctx.num_problem_rows; ++i) {
    // Need at least two binaries, one of them fractional.
    Index nbin = 0;
    bool frac = false;
    for (Index k = R.begin(i); k < R.end(i); ++k) {
      const Index j = R.index[k];
      if (!is_binary(p, j)) continue;
      ++nbin;
      frac = frac || (ctx.x[j] > 1e-6 && ctx.x[j] < 1 - 1e-6);
    }
    if (nbin < 2 || !frac) continue;
    if (is_finite(p.row_ub[i])) separate_side(ctx, i, 1.0, out);
    if (is_finite(p.row_lb[i])) separate_side(ctx, i, -1.0, out);
  }
}

void CoverSeparator::separate_side(const SepaContext& ctx, Index row, double sign, std::vector<Cut>& out) {
  const MipProblem& p = ctx.p;
  const SparseMatrix& R = ctx.lp.scaled_row_matrix();  // entry * us: the model's coefficient, exactly
  const double us = ctx.lp.row_unscale(row);
  double b = sign > 0 ? p.row_ub[row] : -p.row_lb[row];
  items_.clear();
  for (Index k = R.begin(row); k < R.end(row); ++k) {
    const Index j = R.index[k];
    const double a = sign * (R.value[k] * us);
    if (is_binary(p, j)) {
      if (a > 0) {
        items_.push_back({j, a, false, ctx.x[j]});
      } else {
        items_.push_back({j, -a, true, 1.0 - ctx.x[j]});
        b -= a;
      }
    } else {
      const double bound = a > 0 ? p.col_lb[j] : p.col_ub[j];
      if (!is_finite(bound)) return;
      b -= a * bound;
    }
  }
  if (b < 0 || items_.size() < 2) return;
  double total = 0.0;
  for (const Item& it : items_) total += it.a;
  const double eps = 1e-9 * std::max(1.0, std::abs(b));
  if (total <= b + eps) return;

  // Greedy cover: largest z first (ties: larger weight), then make it minimal.
  std::sort(items_.begin(), items_.end(), [](const Item& u, const Item& v) {
    if (u.z != v.z) return u.z > v.z;
    return u.a > v.a;
  });
  size_t nc = 0;
  double w = 0.0;
  while (nc < items_.size() && w <= b + eps) w += items_[nc++].a;
  if (w <= b + eps) return;
  // Drop cover items with the smallest z while the rest is still a cover.
  std::vector<char> in(items_.size(), 0);
  for (size_t k = 0; k < nc; ++k) in[k] = 1;
  for (size_t k = nc; k-- > 0;) {
    if (w - items_[k].a > b + eps) {
      w -= items_[k].a;
      in[k] = 0;
    }
  }
  std::vector<double> cw;
  double zsum = 0.0;
  for (size_t k = 0; k < items_.size(); ++k) {
    if (in[k]) cw.push_back(items_[k].a), zsum += items_[k].z;
  }
  const double csize = static_cast<double>(cw.size());
  std::sort(cw.rbegin(), cw.rend());
  std::vector<double> A(cw.size() + 1, 0.0);
  for (size_t h = 0; h < cw.size(); ++h) A[h + 1] = A[h] + cw[h];

  // Lifted coefficients; quick violation estimate before building the cut.
  std::vector<double> coef(items_.size(), 1.0);
  double lhs = zsum;
  for (size_t k = 0; k < items_.size(); ++k) {
    if (in[k]) continue;
    // A_h <= a_j up to a tolerance covered by the cover's excess margin eps.
    const double a = items_[k].a + eps / (csize + 1.0);
    size_t h = std::upper_bound(A.begin(), A.end(), a) - A.begin() - 1;
    coef[k] = static_cast<double>(h);
    lhs += coef[k] * items_[k].z;
  }
  if (lhs <= csize - 1.0 + 1e-6) return;

  Cut c;
  c.origin = "cover";
  c.rhs = csize - 1.0;
  for (size_t k = 0; k < items_.size(); ++k) {
    if (coef[k] == 0.0) continue;
    if (items_[k].comp) {  // coef * (1 - x)
      c.idx.push_back(items_[k].j);
      c.val.push_back(-coef[k]);
      c.rhs -= coef[k];
    } else {
      c.idx.push_back(items_[k].j);
      c.val.push_back(coef[k]);
    }
  }
  out.push_back(std::move(c));
}

}  // namespace

// GNS superadditive lifting (Gu, Nemhauser & Savelsbergh 2000): for a minimal cover with weights
// a_1 >= ... >= a_r, excess lambda = mu_r - b, mu_h = a_1 + ... + a_h and
// rho_h = max(0, a_{h+1} - (a_1 - lambda)),
//   g(z) = 0                                   for 0 <= z <= mu_1 - lambda,
//   g(z) = h                                   for mu_h - lambda + rho_h <= z <= mu_{h+1} - lambda,
//   g(z) = h - (mu_h - lambda + rho_h - z) / rho_1   in between (h = 1..r-1),
// is superadditive and sum_C z_k + sum_{k not in C} g(w_k) z_k <= r - 1 is valid. Items heavier than
// b cannot be 1 at all; they get r - 1.
bool knapsack_cover_cut(const std::vector<double>& w, const std::vector<double>& zs, double b,
                        std::vector<double>& coef, double& rhs) {
  const size_t n = w.size();
  if (n < 2 || !(b >= 0.0)) return false;
  const double eps = 1e-9 * std::max(1.0, std::abs(b));
  double total = 0.0;
  for (double v : w) total += v;
  if (total <= b + 1e-6 * std::max(1.0, std::abs(b))) return false;
  // Greedy cover: items at 1 first, then by (1 - z) / w (cheapest violation per weight).
  std::vector<size_t> order(n);
  for (size_t k = 0; k < n; ++k) order[k] = k;
  auto key = [&](size_t k) { return std::max(0.0, 1.0 - zs[k]) / w[k]; };
  std::sort(order.begin(), order.end(), [&](size_t u, size_t v) {
    const double ku = key(u), kv = key(v);
    if (ku != kv) return ku < kv;
    return u < v;
  });
  std::vector<char> in(n, 0);
  double cw = 0.0;
  size_t nc = 0;
  while (nc < n && cw <= b + eps) cw += w[order[nc]], in[order[nc]] = 1, ++nc;
  if (cw <= b + eps) return false;
  // Minimal: drop items with the smallest z (last in the greedy order) while still a cover.
  for (size_t t = nc; t-- > 0;) {
    const size_t k = order[t];
    if (cw - w[k] > b + eps) cw -= w[k], in[k] = 0;
  }
  std::vector<double> a;
  double zsum = 0.0;
  for (size_t k = 0; k < n; ++k) {
    if (in[k]) a.push_back(w[k]), zsum += zs[k];
  }
  const size_t r = a.size();
  if (r < 2) return false;  // a single item heavier than b: a bound, not a cut
  std::sort(a.rbegin(), a.rend());
  std::vector<double> mu(r + 1, 0.0);
  for (size_t h = 0; h < r; ++h) mu[h + 1] = mu[h] + a[h];
  const double lambda = mu[r] - b;
  if (lambda <= 1e-6 * std::max(1.0, std::abs(b))) return false;
  auto rho = [&](size_t h) { return std::max(0.0, a[h] - (a[0] - lambda)); };  // rho_h, h = 0..r-1
  const double rho1 = rho(1);
  auto g = [&](double z) -> double {
    if (z <= mu[1] - lambda) return 0.0;
    for (size_t h = 1; h < r; ++h) {
      if (z <= mu[h + 1] - lambda) {
        const double lo = mu[h] - lambda + rho(h);
        if (z >= lo || rho1 <= 0.0) return static_cast<double>(h);
        return std::max(static_cast<double>(h - 1), static_cast<double>(h) - (lo - z) / rho1);
      }
    }
    return static_cast<double>(r - 1);
  };
  coef.assign(n, 0.0);
  double lhs = zsum;
  for (size_t k = 0; k < n; ++k) {
    if (in[k]) {
      coef[k] = 1.0;
      continue;
    }
    // Slightly smaller weight: g is nondecreasing, so rounding errors only weaken the cut.
    coef[k] = g(w[k] - 1e-9 * std::max(1.0, w[k]) - eps);
    lhs += coef[k] * zs[k];
  }
  rhs = static_cast<double>(r - 1);
  return lhs > rhs + 1e-6;
}

std::unique_ptr<Separator> make_cover_separator() { return std::make_unique<CoverSeparator>(); }

}  // namespace milp
