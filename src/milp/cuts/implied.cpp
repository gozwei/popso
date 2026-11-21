// Implied bound cuts. For a row and a binary y in it, fixing y to 0 or to 1
// implies (by activity bounds of the rest of the row) bounds v0 / v1 on each
// other variable x of the row. Since y is binary, x <= v0 + (v1 - v0) y holds
// (similarly for lower bounds); it is a cut when tighter than x's own bound,
// e.g. the disaggregated facility constraints x_ij <= u_ij y_i implied by a
// capacity row sum_j x_ij <= cap_i y_i. Bounds on general integers implied
// over several rows come from probing (CutData::probe).

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "milp/cuts/cut.hpp"

namespace milp {

namespace {

struct Implication {
  Index x, y;
  double v0, v1;  // bound on x when y = 0 / y = 1
  bool upper;     // x <= ... (true) or x >= ... (false)
};

class ImpliedBoundSeparator : public Separator {
 public:
  const char* name() const override { return "implbd"; }
  void separate(const SepaContext& ctx, std::vector<Cut>& out) override;
  long long work() const override { return work_; }

 private:
  long long work_ = 0;  // implications checked by the last call
  void build(const MipProblem& p);
  bool built_ = false;
  bool vb_store_ = !std::getenv("MILP_NO_IMPLBD_VB");
  std::vector<Implication> imps_;
};

bool binary(const MipProblem& p, Index j) { return p.is_int[j] && p.col_lb[j] == 0.0 && p.col_ub[j] == 1.0; }

void ImpliedBoundSeparator::build(const MipProblem& p) {
  built_ = true;
  const SparseMatrix R = p.A.transpose();
  constexpr Index kMaxRowLen = 500;
  constexpr size_t kMaxImplications = 2000000;
  for (Index i = 0; i < p.num_rows() && imps_.size() < kMaxImplications; ++i) {
    const Index rb = R.begin(i), re = R.end(i);
    if (re - rb < 2 || re - rb > kMaxRowLen) continue;
    for (double sign : {1.0, -1.0}) {  // the row side as  sum (sign * a) x <= b
      const double b = sign > 0 ? p.row_ub[i] : -p.row_lb[i];
      if (!is_finite(b)) continue;
      // Minimal activity of the row; rows with unbounded terms are skipped.
      double minact = 0.0;
      bool finite = true;
      for (Index k = rb; k < re && finite; ++k) {
        const Index j = R.index[k];
        const double a = sign * R.value[k];
        const double bnd = a > 0 ? p.col_lb[j] : p.col_ub[j];
        finite = is_finite(bnd);
        minact += a * bnd;
      }
      if (!finite) continue;
      for (Index ky = rb; ky < re; ++ky) {
        const Index y = R.index[ky];
        if (!binary(p, y)) continue;
        const double ay = sign * R.value[ky];
        const double ymin = std::min(ay, 0.0);  // y's share of minact
        for (Index kx = rb; kx < re; ++kx) {
          const Index x = R.index[kx];
          if (x == y) continue;
          const double ax = sign * R.value[kx];
          const double xmin = ax > 0 ? ax * p.col_lb[x] : ax * p.col_ub[x];
          const double rest = minact - ymin - xmin;  // minimal activity without x and y
          // ax * x <= b - rest - ay * y  for y in {0, 1}
          const double s0 = b - rest, s1 = b - rest - ay;
          if (ax > 0) {
            const double ub = p.col_ub[x];
            const double c0 = std::min(s0 / ax, ub), c1 = std::min(s1 / ax, ub);
            if (is_finite(c0) && is_finite(c1) && std::min(c0, c1) < ub - 1e-6) imps_.push_back({x, y, c0, c1, true});
          } else {
            const double lb = p.col_lb[x];
            const double c0 = std::max(s0 / ax, lb), c1 = std::max(s1 / ax, lb);
            if (is_finite(c0) && is_finite(c1) && std::max(c0, c1) > lb + 1e-6) imps_.push_back({x, y, c0, c1, false});
          }
          if (imps_.size() >= kMaxImplications) break;
        }
      }
    }
  }
}

void ImpliedBoundSeparator::separate(const SepaContext& ctx, std::vector<Cut>& out) {
  if (!built_) build(ctx.p);
  const double* xv = ctx.x;
  auto check = [&](const auto& im) {
    const double bound = im.v0 + (im.v1 - im.v0) * xv[im.y];
    // The test first: nearly every implication holds at the LP point, and building the cut before the test
    // allocated its index array for each of them (neos-3046615-murg: 5% of the thread time in malloc / free).
    if (im.upper ? xv[im.x] <= bound + 1e-6 : xv[im.x] >= bound - 1e-6) return;
    Cut c;
    c.origin = "implbd";
    c.idx = {im.x, im.y};
    if (im.upper) {
      c.val = {1.0, -(im.v1 - im.v0)};  // x - (v1 - v0) y <= v0
      c.rhs = im.v0;
    } else {
      c.val = {-1.0, im.v1 - im.v0};  // -x + (v1 - v0) y <= -v0
      c.rhs = -im.v0;
    }
    out.push_back(std::move(c));
  };
  for (const Implication& im : imps_) check(im);
  for (const ProbeImplication& im : ctx.data.probe_imps) check(im);
  work_ = static_cast<long long>(imps_.size() + ctx.data.probe_imps.size());
  // Variable bounds of the store (two-variable rows, implied by single rows, probing) on binaries.
  const MipProblem& p = ctx.p;
  if (!vb_store_) return;
  for (Index j = 0; j < static_cast<Index>(ctx.data.vub.size()); ++j) {
    work_ += 1 + static_cast<long long>(ctx.data.vlb[j].size() + ctx.data.vub[j].size());
    for (int side = 0; side < 2; ++side) {
      for (const VarBound& vb : side ? ctx.data.vlb[j] : ctx.data.vub[j]) {
        if (!binary(p, vb.y)) continue;
        check(Implication{j, vb.y, vb.constant, vb.constant + vb.coef, side == 0});
      }
    }
  }
}

}  // namespace

std::unique_ptr<Separator> make_implied_bound_separator() { return std::make_unique<ImpliedBoundSeparator>(); }

}  // namespace milp
