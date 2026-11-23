#include "milp/cuts/cut_loop.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_set>

#include "milp/core/interrupt.hpp"

namespace milp {

namespace {

bool cuts_disabled_by_env() {
#ifdef _MSC_VER
#pragma warning(suppress : 4996)
#endif
  const char* v = std::getenv("MILP_NO_CUTS");
  return v && *v && std::strcmp(v, "0") != 0;
}

// MILP_DEBUG_SOL: file with a known optimal point, one value per column.
std::vector<double> load_debug_solution(Index n) {
#ifdef _MSC_VER
#pragma warning(suppress : 4996)
#endif
  const char* f = std::getenv("MILP_DEBUG_SOL");
  std::vector<double> x;
  if (!f) return x;
#ifdef _MSC_VER
#pragma warning(suppress : 4996)
#endif
  std::FILE* fp = std::fopen(f, "r");
  if (!fp) return x;
  double v;
  while (std::fscanf(fp, "%lf", &v) == 1) x.push_back(v);
  std::fclose(fp);
  if (static_cast<Index>(x.size()) != n) x.clear();
  return x;
}

// The known point is only meaningful if it lies within the problem's column
// bounds (sub-MIPs restrict bounds, so the global optimum is usually outside).
void drop_if_outside(const MipProblem& p, std::vector<double>& x) {
  for (Index j = 0; j < static_cast<Index>(x.size()); ++j) {
    if (x[j] < p.col_lb[j] - 1e-6 || x[j] > p.col_ub[j] + 1e-6) {
      x.clear();
      return;
    }
  }
  // Sub-MIPs may also add rows (local branching) that exclude the point.
  if (!x.empty() && p.max_violation(x) > 1e-6) x.clear();
}

}  // namespace

CutLoopStats root_cut_loop(const MipProblem& p, DualSimplex& lp, const SolverParams& params, const Logger& log,
                           const Timer& timer, double cutoff, CutData* data, bool sub_mip,
                           std::vector<Cut>* dropped, const std::function<void(const DualSimplex&)>& on_lp,
                           const std::atomic<bool>* stop, const std::function<double()>& cutoff_fn,
                           int max_rounds, const std::function<bool(double)>& gap_closed, long long prior_iterations,
                           const std::function<bool(long long)>& lp_checkpoint, long long checkpoint_every,
                           int threads,
                           const std::function<Index(std::vector<double>&, std::vector<double>&, std::vector<Index>&)>&
                               fix_fn) {
  if (cuts_disabled_by_env()) return CutLoopStats{};
  CutLoopOptions opt;
  opt.threads = threads;
  opt.fix_fn = fix_fn;
  opt.on_lp = on_lp;
  opt.initial_checkpoint = lp_checkpoint;
  opt.checkpoint_every = checkpoint_every;
  opt.prior_iterations = prior_iterations;
  // Reduced-cost fixing between rounds: opt-in (MILP_CUTLOOP_RCFIX=1). Small root gains only
  // (mzzv42z, ns1830653), and Gomory cuts derived from the fixed bounds are valid below the cutoff
  // only, which root restarts (cut rows become model rows) would have to respect.
  static const bool rcfix = [] { const char* v = std::getenv("MILP_CUTLOOP_RCFIX"); return v && *v && *v != '0'; }();
  if (rcfix) opt.cutoff_fn = cutoff_fn;
  static const bool gap_stop = [] { const char* v = std::getenv("MILP_CUT_GAP_STOP"); return !v || *v != '0'; }();
  if (gap_stop) opt.gap_closed = gap_closed;
  opt.stop = stop;
  opt.zerohalf = !sub_mip;
  if (const char* v = std::getenv("MILP_MCF")) opt.mcf = *v != '0';
  opt.tol = params.tol;
  // Spend at most a fifth of the remaining time on cuts, not counting the initial LP solve
  // (needed anyway; on large models it can take most of that fifth).
  if (is_finite(params.time_limit)) {
    opt.time_limit = timer.elapsed() + 0.2 * (params.time_limit - timer.elapsed());
    opt.time_share = 0.2;
    opt.total_time_limit = params.time_limit;
  }
  // Flat rounds on large models with expensive separation (neos-960392: 12 rounds of ~0.5-1 s, bound
  // never moved) are capped by time; the bin packing like models they are for have cheap rounds.
  if (!params.deterministic) opt.flat_time_factor = 3.0;
  if (const char* v = std::getenv("MILP_FLAT_TIME")) opt.flat_time_factor = std::atof(v);
  // MILP_FLAT_WORK = f (default 3; 0: the time cap above): that cap in work units, in both modes (a
  // wall-clock cap moves with the machine's load: neos-3381206-awhea's 7 flat rounds before its bound moves
  // take 0.4-0.8 s unloaded and more than the 1 s floor loaded, and then the tree cannot close 415 -> 453;
  // --deterministic had no cap: neos-960392 15 rounds, now 4). MILP_FLAT_WORK_MIN: the floor.
  opt.flat_work_factor = 3.0;
  if (const char* v = std::getenv("MILP_FLAT_WORK")) opt.flat_work_factor = std::atof(v);
  if (const char* v = std::getenv("MILP_FLAT_WORK_MIN")) opt.flat_work_min = std::atof(v);
  // MILP_SEPA_WORK=0: the loop's work estimates the separation as Gomory's (no Separator::work() counts);
  // MILP_PLATEAU_WORK=0: no work cap on plateaus after the bound moved; MILP_SEPA_BACKOFF=0: no separator
  // backoff, MILP_SEPA_BACKOFF_WORK: its minimum call work (CutLoopOptions).
  if (const char* v = std::getenv("MILP_SEPA_WORK")) opt.sepa_work = *v != '0';
  if (const char* v = std::getenv("MILP_PLATEAU_WORK")) opt.plateau_work = *v != '0';
  if (const char* v = std::getenv("MILP_SEPA_BACKOFF")) opt.backoff = *v != '0';
  if (const char* v = std::getenv("MILP_SEPA_BACKOFF_WORK")) opt.backoff_work = std::atof(v);
  if (const char* v = std::getenv("MILP_CHEAP_ROUNDS")) opt.max_rounds_cheap = std::atoi(v);
  // MILP_CHEAP_SEP = f (opt-in; 0: off): CutLoopOptions::cheap_sep_factor.
  if (const char* v = std::getenv("MILP_CHEAP_SEP")) opt.cheap_sep_factor = std::atof(v);
  if (const char* v = std::getenv("MILP_MAX_CUTS")) opt.max_cuts_per_round = std::atoi(v);
  if (const char* v = std::getenv("MILP_CHEAP_RECENT_REL")) opt.cheap_recent_rel = std::atof(v);
  // MILP_CUT_AGE = k: slack cut rows stay for k further slack re-solves (0: removed at the first).
  if (const char* v = std::getenv("MILP_CUT_AGE")) opt.cut_age = std::atoi(v);
  // Slow roots (default; MILP_SLOW_ROOT=0 switches it off): CutLoopOptions::resolve_work_min
  // (MILP_SLOW_ROOT_CUT, default 5e9) and resolve_work_rel (MILP_SLOW_ROOT_CUT_REL, default 1). supportcase22
  // (259962 rows, ~240 iterations/s): the first round's re-solve ran 59 s into the loop's time limit and was
  // discarded.
  if (const char* v = std::getenv("MILP_SLOW_ROOT"); !v || !*v || *v != '0') {
    opt.resolve_work_min = 5e9;
    if (const char* c = std::getenv("MILP_SLOW_ROOT_CUT")) opt.resolve_work_min = std::atof(c);
    if (const char* c = std::getenv("MILP_SLOW_ROOT_CUT_REL")) opt.resolve_work_rel = std::atof(c);
  }
  if (!sub_mip) {
    // Flat root loops (default; MILP_DEGEN_MOVES=0 MILP_FLAT_CUTS=0 switch them off): degenerate moves after the
    // rounds while the bound has not moved (bit 4), and the binding cuts of such a loop stay when the moves
    // at least halved the fractional count. Only on models with at least flat_frac of the free integer columns
    // fractional at the first LP (CMS750_4: 19%, 60-113 s and 34-126k nodes -> 3-6 s and 45-277 nodes).
    opt.degen_moves = 4;
    opt.flat_cuts = 1;
    if (const char* v = std::getenv("MILP_DEGEN_MOVES")) opt.degen_moves = std::atoi(v);
    if (const char* v = std::getenv("MILP_FLAT_CUTS")) opt.flat_cuts = std::atoi(v);
    if (const char* v = std::getenv("MILP_FLAT_KEEP")) opt.flat_keep = std::atof(v);
    if (const char* v = std::getenv("MILP_FLAT_MORE")) opt.flat_more = std::atof(v);
    if (const char* v = std::getenv("MILP_FLAT_ROWS")) opt.flat_rows = std::atof(v);
    if (const char* v = std::getenv("MILP_FLAT_FRAC")) opt.flat_frac = std::atof(v);
  }
  if (max_rounds >= 0) {
    opt.max_rounds = std::min(opt.max_rounds, max_rounds);
    opt.max_rounds_cheap = 0;
  }
  return cut_loop(p, lp, opt, log, timer, cutoff, nullptr, data, dropped);
}

CutLoopStats cut_loop(const MipProblem& p, DualSimplex& lp, const CutLoopOptions& opt_in, const Logger& log,
                      const Timer& timer, double cutoff, std::vector<Cut>* all_cuts, CutData* shared_data,
                      std::vector<Cut>* dropped_out) {
  CutLoopOptions opt = opt_in;
  CutLoopStats stats;
  const Index n = p.num_cols(), m0 = p.num_rows();
  LpOptions lpo;
  lpo.tol = opt.tol;
  lpo.objective_cutoff = cutoff;
  auto solve = [&]() {
    lpo.time_limit = std::max(0.0, opt.time_limit - timer.elapsed());
    return lp.solve(lpo);
  };
  const long long iters_start = lp.iterations();
  const long long lp_work_start = lp.work();
  const double t_lp0 = timer.elapsed();
  lpo.checkpoint = opt.initial_checkpoint;
  lpo.checkpoint_every = opt.checkpoint_every;
  if (opt.time_share > 0.0 && is_finite(opt.total_time_limit)) {
    // The initial solve gets the whole remaining time; the share counts from its end.
    lpo.time_limit = std::max(0.0, opt.total_time_limit - timer.elapsed());
    if (lp.solve(lpo) != LpStatus::Optimal) return stats;
    const double now = timer.elapsed();
    opt.time_limit = std::max(opt.time_limit, now + opt.time_share * (opt.total_time_limit - now));
  } else if (solve() != LpStatus::Optimal) {
    return stats;
  }
  lpo.checkpoint = nullptr;
  const long long iters0 = lp.iterations();
  const long long lp_work_initial = lp.work() - lp_work_start;
  if (opt.on_lp) opt.on_lp(lp);
  const double t_lp_end = timer.elapsed(), initial_lp_time = t_lp_end - t_lp0;
  stats.initial_lp_time = initial_lp_time;
  log.log(2, "  initial LP: %lld iterations, %.2fs (%d rows, %d cols)\n", iters0 - iters_start, initial_lp_time,
          static_cast<int>(lp.num_rows()), static_cast<int>(lp.num_cols()));
  // Work of the loop after the initial solve (flat_work_factor): the cut probing budget (its time counts for
  // the time cap too), per round the re-solve's iterations x (rows + columns) plus the separation: the work
  // the separators count (Separator::work(): Gomory's tableau rows, c-MIR's aggregations), or with
  // sepa_work off estimated as Gomory's: a tableau row (a BTRAN and a pass over the row matrix) for each
  // fractional integer column, at most 500 (neos-960392: a flat round costs ~0.4x the initial solve, mostly
  // Gomory).
  const long long init_iters = iters0 - iters_start + std::max<long long>(0, opt.prior_iterations);
  const double initial_work = static_cast<double>(init_iters) * static_cast<double>(lp.num_rows() + n);
  const long long resolve_cap =
      opt.resolve_work_min > 0.0
          ? static_cast<long long>(std::max(opt.resolve_work_min, opt.resolve_work_rel * std::max(static_cast<double>(lp.work()), initial_work)))
          : 0;
  if (resolve_cap > 0) log.log(2, "  slow root: LP work so far %.3g (initial solve %.3g), re-solve cap %.3g\n", static_cast<double>(lp.work()), initial_work,
                               static_cast<double>(resolve_cap));
  double loop_work = 0.0;
  double plateau_work = 0.0;  // loop work of the rounds since the bound last moved
  double lp_time = 0.0;
  int extra_rounds = 0;
  double obj = lp.objective();
  stats.bound_before = stats.bound_after = obj;
  // Global bound tightenings against the cutoff (CutLoopOptions::fix_fn), after the initial solve and before
  // each round: kept in the LP, which is re-solved. 0: nothing changed, 1: changed (LP optimal again), 2: the LP
  // is not optimal any more (infeasible or cut off: no improving point is left; or a failure): no more rounds.
  std::vector<double> fx_lb, fx_ub;
  std::vector<Index> fx_cols;
  double fix_gain = 0.0;  // bound gained by these re-solves (not by cuts)
  std::vector<char> rc_changed(n, 0);  // reduced-cost fixing below
  std::vector<std::pair<Index, std::pair<double, double>>> rc_saved;  // column, original bounds
  auto global_fix = [&]() -> int {
    if (!opt.fix_fn) return 0;
    fx_lb.resize(n), fx_ub.resize(n);
    for (Index j = 0; j < n; ++j) fx_lb[j] = lp.col_lower(j), fx_ub[j] = lp.col_upper(j);
    fx_cols.clear();
    if (opt.fix_fn(fx_lb, fx_ub, fx_cols) == 0 || fx_cols.empty()) return 0;
    for (Index j : fx_cols) {
      lp.set_col_bounds(j, fx_lb[j], fx_ub[j]);
      if (!rc_changed[j]) continue;
      // Restored at the end only as far as the new bounds allow.
      for (auto& [c, b] : rc_saved) {
        if (c == j) b.first = std::max(b.first, fx_lb[j]), b.second = std::min(b.second, fx_ub[j]);
      }
    }
    stats.global_fixed += static_cast<long long>(fx_cols.size());
    const long long it0 = lp.iterations();
    const double before = lp.objective();
    const LpStatus st = solve();
    log.log(2, "  objective propagation with cliques: %zu columns fixed, LP %s %.10g -> %.10g, %lld iterations\n",
            fx_cols.size(), to_string(st), p.display_sense * (before + p.offset), p.display_sense * (lp.objective() + p.offset),
            lp.iterations() - it0);
    if (st != LpStatus::Optimal) return 2;
    if (opt.on_lp) opt.on_lp(lp);
    fix_gain += lp.objective() - before;
    obj = lp.objective();
    stats.bound_after = obj;
    return 1;
  };
  if (const int g = global_fix(); g == 2 || (g == 1 && opt.gap_closed && opt.gap_closed(obj))) {
    if (g == 1) log.log(2, "  gap closed by the incumbent, no round\n");
    return stats;
  }

  std::vector<std::unique_ptr<Separator>> seps;
  if (opt.gomory) seps.push_back(make_gomory_separator());
  if (opt.cover) seps.push_back(make_cover_separator());
  if (opt.mir) seps.push_back(make_mir_separator());
  if (opt.mir && std::getenv("MILP_ROOT_TMIR")) seps.push_back(make_tableau_mir_separator());
  if (opt.mcf) seps.push_back(make_mcf_separator());
  if (opt.clique) seps.push_back(make_clique_separator());
  if (opt.implied_bound && !std::getenv("MILP_NO_IMPLBD")) seps.push_back(make_implied_bound_separator());
  if (opt.zerohalf && !std::getenv("MILP_NO_ZEROHALF")) seps.push_back(make_zerohalf_separator());
  if (!p.quad_rows.empty()) seps.push_back(make_quad_oa_separator());  // outer approximation of quadratic rows
  if (const char* only = std::getenv("MILP_ONLY_SEPA")) {
    // Diagnostics: keep only the named separators (comma list, e.g. "mir,implbd").
    const std::string list = std::string(",") + only + ",";
    seps.erase(std::remove_if(seps.begin(), seps.end(),
                              [&](const std::unique_ptr<Separator>& s) {
                                return list.find(std::string(",") + s->name() + ",") == std::string::npos;
                              }),
               seps.end());
  }
  if (seps.empty()) return stats;
  CutData local_data;
  if (!shared_data) local_data.build(p);
  CutData& data = shared_data ? *shared_data : local_data;
  if ((opt.clique || opt.implied_bound) && !data.probed) {
    const long long nnz = static_cast<long long>(p.A.index.size());
    static const char* env_work = std::getenv("MILP_PROBE_WORK");
    const long long work = opt.probe_work >= 0 ? opt.probe_work
                           : env_work           ? std::atoll(env_work)
                                                : std::clamp<long long>(2000 * nnz, 1000000LL, 500000000LL);
    const double t0 = timer.elapsed();
    // At the time limit the probing stops (its work budget alone took up to 6.5 s on fiball at 1 thread: a
    // 10 s limit overran by 10 s on a loaded machine).
    data.probe(p, lp.x(), work, [&] { return timer.elapsed() > opt.total_time_limit; });
    loop_work += static_cast<double>(work);  // its budget (counted by the time cap too)
    log.log(2, "  cut probing: %zu implied bounds, %zu literal conflicts, %zu implied variable bounds, %.2fs\n",
            data.probe_imps.size(), data.conflict_lit.size() / 2, data.num_implied_vbs, timer.elapsed() - t0);
  }

  CutPoolOptions popt;
  popt.max_cuts = opt.max_cuts_per_round > 0 ? opt.max_cuts_per_round : std::clamp<int>(n / 10, 50, 500);
  // Density limit: a quarter of the columns, but at least twice the longest problem row.
  Index max_row = 0;
  {
    std::vector<Index> len(m0, 0);
    for (Index i : p.A.index) ++len[i];
    for (Index l : len) max_row = std::max(max_row, l);
  }
  popt.max_nnz = std::max<Index>({50, n / 4 + 20, 2 * max_row});
  popt.set_dense(p, true);
  const Index dense_len_set = popt.dense_len;
  if (const char* v = std::getenv("MILP_SEL_HYBRID")) popt.hybrid = *v && *v != '0';
  if (const char* v = std::getenv("MILP_SEL_PAR")) popt.max_parallelism_weak = std::atof(v);
  if (const char* v = std::getenv("MILP_SEL_PARGOOD")) popt.max_parallelism = std::atof(v);
  if (const char* v = std::getenv("MILP_SEL_GOOD")) popt.good_score = std::atof(v);
  if (const char* v = std::getenv("MILP_SEL_OBJW")) popt.obj_weight = std::atof(v);
  if (const char* v = std::getenv("MILP_SEL_INTW")) popt.int_weight = std::atof(v);
  // Cut rows: up to max(1000, 2 m0); sparse cuts up to twice that while their nonzeros
  // stay below twice those of the problem (degenerate models may need a cut per bin and pattern).
  const Index max_cut_rows = std::max<Index>(1000, 2 * m0);
  const long long max_cut_nnz = 2 * static_cast<long long>(p.A.index.size());
  std::vector<double> history{obj};
  const Index r0 = lp.num_rows();
  // Rows the loop may remove (slack or useless cuts): from the first cut row, or only its own.
  const Index first_removable = opt.keep_rows ? r0 : m0;
  const Basis basis0 = lp.get_basis();  // optimal for the rows before the loop (restored below)
  std::vector<Cut> row_cuts;  // the cut in LP row r0 + k
  std::vector<Cut> dropped;   // cuts removed for slack, candidates for re-adding
  const size_t max_dropped = 20000;
  // MILP_CUT_TRACE=1 (diagnostics, verbosity 2): per round, the cuts re-added from the dropped list,
  // fresh cuts with the coefficients of a cut that was in the LP before (re-separated), the
  // selection's rejections and the removed rows by their age in rounds.
  static const bool trace = [] { const char* v = std::getenv("MILP_CUT_TRACE"); return v && *v && *v != '0'; }();
  std::vector<int> row_round;  // trace: round in which row_cuts[k] entered the LP
  std::unordered_set<uint64_t> seen_rows;
  long long tr_added = 0, tr_readded = 0, tr_resep = 0, tr_removed = 0;
  auto cut_hash = [](const Cut& c) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ULL; };
    for (size_t k = 0; k < c.idx.size(); ++k) {
      uint64_t b;
      std::memcpy(&b, &c.val[k], sizeof b);
      mix(static_cast<uint64_t>(c.idx[k])), mix(b);
    }
    return h;
  };
  // Cut row aging (cut_age = k > 0): a cut row whose logical is basic with slack stays in the LP until
  // k + 1 re-solves in a row found it slack (SCIP / HiGHS keep slack rows for 10 LP solves). Removing
  // it at the first slack re-solve (k = 0) churns: on tr12-30 most removed rows were added in the same
  // round, and a fifth of the added cuts were re-adds or re-separations of removed ones. Aged rows give
  // way when a full round would not fit below max_cut_rows. At the end of the loop (keep_age < 0) every
  // cut row with a basic logical leaves, also at zero slack (the LP optimum stays): aging leaves many
  // degenerate rows behind (mzzv11 with k = 10: 1926 cut rows instead of 579), and the tree's LPs keep
  // the root's cut rows.
  std::vector<int> row_age;  // per row_cuts: consecutive slack re-solves
  auto remove_slack = [&](int keep_age, int round, int* age_hist) {
    std::vector<Index> slack_rows;
    std::vector<char> slack(lp.num_rows(), 0);
    for (Index i = first_removable; i < lp.num_rows(); ++i) {
      if (!lp.row_is_basic(i)) {
        if (i >= r0) row_age[i - r0] = 0;
        continue;
      }
      const double act = -lp.var_value(n + i), ub = -lp.var_lower(n + i);
      const bool is_slack = ub - act > 1e-6 * std::max(1.0, std::abs(ub)) || (keep_age < 0 && i >= r0);
      if (i >= r0) row_age[i - r0] = is_slack ? row_age[i - r0] + 1 : 0;
      if (is_slack && (i < r0 || row_age[i - r0] > keep_age)) slack_rows.push_back(i), slack[i] = 1;
    }
    if (keep_age > 0 &&
        lp.num_rows() - m0 - static_cast<Index>(slack_rows.size()) > max_cut_rows - popt.max_cuts) {
      for (Index i = r0; i < lp.num_rows(); ++i) {  // no room for a full round: all slack rows go
        if (!slack[i] && row_age[i - r0] > 0) slack_rows.push_back(i), slack[i] = 1;
      }
      std::sort(slack_rows.begin(), slack_rows.end());
    }
    lp.remove_rows(slack_rows);
    size_t w = 0;
    for (size_t k = 0; k < row_cuts.size(); ++k) {
      if (slack[r0 + k]) {
        if (trace) {
          const int age = round - row_round[k] + 1;  // removed rows by rounds in the LP: 1, 2, 3-5, 6+
          ++age_hist[age <= 1 ? 0 : age == 2 ? 1 : age <= 5 ? 2 : 3];
        }
        if (dropped.size() < max_dropped) dropped.push_back(std::move(row_cuts[k]));
      } else {
        if (w != k) row_cuts[w] = std::move(row_cuts[k]), row_age[w] = row_age[k];
        if (trace && w != k) row_round[w] = row_round[k];
        ++w;
      }
    }
    row_cuts.resize(w);
    row_age.resize(w);
    if (trace) row_round.resize(w);
    return slack_rows.size();
  };
  const double itol = opt.tol.integrality;
  std::vector<double> debug_sol = load_debug_solution(n);
  drop_if_outside(p, debug_sol);
  int rc_fixed_total = 0;  // reduced-cost fixings in the LP (below)
  double rc_cutoff = kInf;
  auto record = [&](const std::vector<Cut>& cuts) {
    if (all_cuts) all_cuts->insert(all_cuts->end(), cuts.begin(), cuts.end());
    for (const Cut& c : cuts) {
      if (debug_sol.empty()) break;
      // Reduced-cost fixings below the cutoff may exclude the debug solution legitimately.
      if (rc_fixed_total > 0 && p.objective(debug_sol) - p.offset >= rc_cutoff - 1e-9) break;
      const double viol = c.activity(debug_sol.data()) - c.rhs;
      if (viol > 1e-6 * std::max(1.0, std::abs(c.rhs))) {
        log.log(0, "DEBUG-SOL: %s cut violated by the debug solution (%g)\n", c.origin, viol);
      }
    }
  };

  // Reduced-cost fixing in the LP between rounds against the current cutoff (incumbents found
  // while the loop runs, e.g. by the FJ thread): nonbasic integers whose reduced cost exceeds the
  // slack are fixed in the LP for the rest of the loop (as SCIP does at the root), so later
  // rounds separate a tighter LP. Changing the opposite bound of a nonbasic column keeps the basis
  // optimal; the original bounds are restored at the end. Cuts derived from these bounds (Gomory)
  // are valid for solutions better than the cutoff, as the search's own reduced-cost fixings.
  auto rc_fix = [&]() {
    if (!opt.cutoff_fn) return;
    static const char* diag = std::getenv("MILP_CL_CUTOFF");  // diagnostics: a known cutoff (internal sense, no offset)
    const double co = diag ? std::min(std::atof(diag), opt.cutoff_fn()) : opt.cutoff_fn();
    const double z = lp.objective(), slack = co - z;
    if (!is_finite(co) || !(slack >= 0.0)) return;
    bool unsafe = false;  // z is not confirmed by the LP's safe dual bound: z + d bounds nothing
    lp.safe_objective(opt.tol.dual_feas, &unsafe);
    if (unsafe) {
      log.log(2, "  reduced-cost fixing skipped: LP objective %.10g above its safe bound\n", p.display_sense * (z + p.offset));
      return;
    }
    rc_cutoff = std::min(rc_cutoff, co);
    for (Index j = 0; j < n; ++j) {
      if (!p.is_int[j]) continue;
      const double lo = lp.col_lower(j), up = lp.col_upper(j);
      if (lo == up) continue;
      const VarStatus st = lp.status(j);
      const double d = lp.reduced_cost(j);
      double nlo = lo, nup = up;
      if (st == VarStatus::AtLower && d > 1e-7 && is_finite(lo)) {
        nup = std::min(up, lo + std::floor(slack / d + 1e-9));
      } else if (st == VarStatus::AtUpper && d < -1e-7 && is_finite(up)) {
        nlo = std::max(lo, up - std::floor(slack / -d + 1e-9));
      }
      if (nlo > lo + 0.5 || nup < up - 0.5) {
        if (!rc_changed[j]) rc_changed[j] = 1, rc_saved.push_back({j, {lo, up}});
        lp.set_col_bounds(j, nlo, nup);
        ++rc_fixed_total;
      }
    }
  };

  bool lp_optimal = true;     // the LP is at an optimum (the final slack cleanup needs its basis)
  // Fractional integer columns of the current LP point: the measure of progress while the bound is flat
  // (MILP_FLAT_CUTS, see the end of the loop).
  auto count_frac = [&]() {
    const double* xx = lp.x();
    Index k = 0;
    for (Index j = 0; j < n; ++j) k += p.is_int[j] && std::abs(xx[j] - std::round(xx[j])) > itol;
    return k;
  };
  int degenerate_rounds = 0;  // rounds that ended on a dual degenerate LP
  // Dual degeneracy: share of nonbasic structurals with zero reduced cost. On such LPs
  // cuts often only move the solution to an alternative optimum for several rounds
  // (bin packing like models with many identical bins) before the bound moves.
  auto is_dual_degenerate = [&]() {
    Index nonbasic = 0, degenerate = 0;
    for (Index j = 0; j < n; ++j) {
      if (lp.status(j) == VarStatus::Basic || lp.var_lower(j) == lp.var_upper(j)) continue;
      ++nonbasic;
      degenerate += std::abs(lp.reduced_cost(j)) <= 1e-9;
    }
    return nonbasic > 0 && degenerate >= 0.8 * nonbasic;
  };
  bool last_degenerate = is_dual_degenerate();
  const bool initial_degenerate = last_degenerate;
  log.log(2, "  initial LP %s dual degenerate\n", initial_degenerate ? "is" : "is not");
  // Degenerate moves (CutLoopOptions::degen_moves), repeated up to three times while the count of fractional
  // integer columns falls by a tenth. CMS750_4: the bound is 250 through every round and the first vertex has
  // 1391 fractional binaries; with the moves 1100, then 591, 402, ..., 56 after 14 rounds (the reference's log
  // shows 36 and then 6-16); kept (flat_cuts), the 151 binding cuts and that vertex give a tree of 169 nodes
  // instead of 76,000.
  // Budget over the loop: 4 x the initial solve's iterations (at least 1000) pivots and the initial solve's work
  // (the LP's work counter; at least 1e9, about a second); 10 pivots per fractional column (at least 200) per
  // pass. A pass that leaves more fractional columns than it found is undone (the distance to the rounding can
  // fall while the count rises); the moves stop for the rest of the loop after two such calls in a row or a call
  // without any gain.
  long long move_pivots = 0, move_work = 0;
  const long long move_budget = std::max<long long>(1000, 4 * init_iters);
  const long long move_work_budget = std::max<long long>(1000000000LL, lp_work_initial);
  bool moves_off = false;
  int move_fails = 0;  // calls in a row whose last pass was undone
  int move_calls = 0;
  auto move_to_integral = [&](const char* when) {
    Index before = count_frac();
    if (before == 0 || moves_off || move_pivots >= move_budget || move_work >= move_work_budget) return;
    const Index first = before;
    long long pivots = 0;
    const double t0 = timer.elapsed();
    const double z0 = lp.objective();
    const long long w_before = lp.work();
    bool undone = false;
    // The loop's first call is a trial on a fifth of the work budget: unless it cuts the count by a fifth the
    // moves stop (cryptanalysiskb128n5obj14: 3.3 s of moves without a gain; CMS750_4's first call: 1623 -> 596
    // with 1.7e8 units).
    const bool trial = move_calls++ == 0;
    const long long call_work = trial ? std::min(move_work_budget, move_work + move_work_budget / 5) : move_work_budget;
    for (int pass = 0; pass < 3 && move_pivots < move_budget && move_work < call_work; ++pass) {
      const Basis keep = lp.get_basis();
      lpo.time_limit = std::max(0.0, opt.time_limit - timer.elapsed());
      const long long cap = std::min(move_budget - move_pivots, std::max<long long>(200, 10LL * before));
      const long long w0 = lp.work();
      const long long piv = lp.degenerate_moves(p.is_int, lpo, cap, call_work - move_work);
      move_work += lp.work() - w0;
      const LpStatus st = solve();
      pivots += piv;
      move_pivots += piv;
      const Index now = st == LpStatus::Optimal ? count_frac() : before + 1;
      if (st != LpStatus::Optimal || std::abs(lp.objective() - z0) > 1e-7 * std::max(1.0, std::abs(z0)) || now > before) {
        lp.set_basis(keep);  // numerical trouble or a worse point: back to the vertex before this pass
        solve();
        undone = true;
        moves_off = ++move_fails >= 2 || first == before;  // twice in a row, or a call without any gain
        break;
      }
      if (now < before) move_fails = 0;
      const bool go_on = piv > 0 && now * 10 <= before * 9;
      before = now;
      if (!go_on) break;
    }
    if (trial && before * 5 > first * 4) moves_off = true;
    log.log(2, "  degenerate moves %s: %d -> %d fractional integer columns, %lld pivots, %.2fs%s\n", when,
            static_cast<int>(first), static_cast<int>(count_frac()), pivots, timer.elapsed() - t0,
            moves_off ? " (last pass undone, no further moves)" : undone ? " (last pass undone)" : "");
    log.log(3, "  degenerate moves work %.3g (LP work counter), initial solve %.3g\n",
            static_cast<double>(lp.work() - w_before), static_cast<double>(lp_work_initial));
  };
  // The flat-loop rules (degen_moves bit 4, flat_cuts) apply only where at least flat_frac of the free integer
  // columns are fractional at the initial LP: the LP says little about the integers there (CMS750_4: 19%; the
  // instances where the moves bought nothing or cost a second have 1-13%).
  bool flat_lp = true;
  if (opt.flat_frac > 0.0) {
    Index free_ints = 0;
    for (Index j = 0; j < n; ++j) free_ints += p.is_int[j] && lp.col_lower(j) < lp.col_upper(j);
    flat_lp = static_cast<double>(count_frac()) >= opt.flat_frac * static_cast<double>(std::max<Index>(free_ints, 1));
  }
  if ((opt.degen_moves & 1) && initial_degenerate) move_to_integral("after the initial LP");
  const Index frac_initial = count_frac();
  Index frac_now = frac_initial, frac_best = frac_initial;
  int frac_best_round = 0;
  std::vector<double> sep_time(seps.size(), 0.0);
  // MILP_CUT_PROFILE=1 (diagnostics, verbosity 2): per round and separator the separation time, its
  // candidates, those the pool kept, those selected and those binding after the re-solve.
  static const bool profile = [] { const char* v = std::getenv("MILP_CUT_PROFILE"); return v && *v && *v != '0'; }();
  // Separator backoff (opt.backoff): on large models some separators run every round without a cut selected
  // (opm2-z10-s4: clique cuts, 2.4 s per round, none found; snp-02-004-104 and uccase12: c-MIR ~1 s per round
  // after round 3; proteindesign121hz512p9: Gomory cuts all too dense for the pool), over a third of the
  // separation time of the 41 large-LP MIPLIB roots. Per separator: failed calls in a row, next round to call.
  std::vector<int> sepa_fails(seps.size(), 0), sepa_next(seps.size(), 1);
  const double matrix_pass = static_cast<double>(p.A.index.size()) + static_cast<double>(n + m0);
  double work_std = -1.0;  // loop_work after the first max_rounds rounds (cheap_sep_factor)
  auto cheap = [&]() {
    if (opt.cheap_sep_factor > 0.0 && work_std > 0.0 && loop_work - work_std > opt.cheap_sep_factor * work_std) {
      return false;
    }
    return lp.iterations() - iters0 <= std::max<long long>(opt.cheap_work * init_iters, 20000);
  };
  for (int round = 1; round <= std::max(opt.max_rounds, opt.max_rounds_cheap); ++round) {
    if (timer.elapsed() >= opt.time_limit) break;
    if ((opt.stop && opt.stop->load()) || interrupted()) break;
    if (opt.gap_closed && opt.gap_closed(lp.objective())) {
      log.log(2, "  gap closed by the incumbent, no further round\n");
      break;
    }
    if (round > opt.max_rounds && !cheap()) break;
    rc_fix();
    if (const int g = global_fix(); g == 2) {
      lp_optimal = false;
      break;
    } else if (g == 1 && opt.gap_closed && opt.gap_closed(obj)) {
      log.log(2, "  gap closed by the incumbent, no further round\n");
      break;
    }
    const double* x = lp.x();
    bool frac = false;
    for (Index j = 0; j < n && !frac; ++j) {
      if (p.is_int[j]) frac = std::abs(x[j] - std::round(x[j])) > itol;
    }
    if (!frac) break;
    const double round_work0 = loop_work;
    if (opt.flat_work_factor > 0.0 && !opt.sepa_work) {
      Index nfrac = 0;
      for (Index j = 0; j < n; ++j) nfrac += p.is_int[j] && std::abs(x[j] - std::round(x[j])) > itol;
      loop_work += static_cast<double>(std::min<Index>(nfrac, 500)) *
                   static_cast<double>(lp.scaled_row_matrix().nnz() + lp.num_rows());
    }

    // Separate. Cuts dropped earlier for slack compete again when violated (on
    // degenerate LPs the solution otherwise cycles between alternative optima).
    SepaContext ctx{p, lp, x, m0, data, round, popt.max_nnz};
    ctx.threads = opt.threads;
    // On dual degenerate LPs (cuts mostly move the solution between alternative optima, many cheap
    // rounds are typical) separators skip their expensive variants (c-MIR paths through integer
    // columns): neos-3381206-awhea's flat rounds are capped by time, and once its bound moves the
    // extra cuts derail it (root 100% -> 84% closed, solve 10 s -> TL; mzzv11 is not degenerate).
    ctx.expensive = !initial_degenerate &&
                    !(last_degenerate && obj - stats.bound_before <= 1e-9 * std::max(1.0, std::abs(obj)));
    if (dense_len_set > 0 && CutPoolOptions::dense_mode() == 4) {
      // Long cuts compete unhindered while the bound moves: the last round gained at least a twentieth of
      // the loop's gain so far (the first round counts as moving).
      const size_t hh = history.size();
      const double total = obj - stats.bound_before, last = hh >= 2 ? history[hh - 1] - history[hh - 2] : kInf;
      const bool moving = hh < 2 || (total > 1e-9 * std::max(1.0, std::abs(obj)) && last >= 0.05 * total);
      popt.dense_len = moving ? 0 : dense_len_set;
    }
    if (dense_len_set > 0 && CutPoolOptions::dense_mode() == 6) {
      // Modes 3 and 4 together: tableau cuts always, the other families only once the bound has stopped moving.
      const size_t hh = history.size();
      const double total = obj - stats.bound_before, last = hh >= 2 ? history[hh - 1] - history[hh - 2] : kInf;
      popt.dense_gomory_only = hh < 2 || (total > 1e-9 * std::max(1.0, std::abs(obj)) && last >= 0.05 * total);
    }
    CutPool pool(p, popt);
    std::vector<Cut> cand;
    int found[16] = {0}, raw[16] = {0}, from_pool = 0;
    for (size_t k = 0; k < dropped.size(); ++k) {
      const Cut& c = dropped[k];
      if (c.activity(x) - c.rhs <= popt.min_violation * std::max(1.0, std::abs(c.rhs))) continue;
      Cut copy = c;
      copy.pool_id = static_cast<int>(k);
      from_pool += pool.add(std::move(copy), x);
    }
    double round_sep_time[16] = {0.0};
    bool called[16] = {false};
    for (size_t s = 0; s < seps.size(); ++s) {
      if (opt.backoff && round < sepa_next[s]) continue;
      called[s] = true;
      cand.clear();
      const double ts = timer.elapsed();
      seps[s]->separate(ctx, cand);
      round_sep_time[s] = timer.elapsed() - ts;
      sep_time[s] += round_sep_time[s];
      stats.sep_time += round_sep_time[s];
      raw[s] = static_cast<int>(cand.size());
      if (opt.sepa_work) loop_work += static_cast<double>(seps[s]->work());
      for (Cut& c : cand) {
        c.sepa = static_cast<int>(s);
        found[s] += pool.add(std::move(c), x);
      }
    }
    const int pool_candidates = static_cast<int>(pool.size());
    std::vector<Cut> cuts = pool.select();
    {
      Index rows = lp.num_rows() - m0;
      long long nnz = 0;
      for (const Cut& c : row_cuts) nnz += static_cast<long long>(c.idx.size());
      size_t keep = 0;
      for (; keep < cuts.size(); ++keep, ++rows) {
        nnz += static_cast<long long>(cuts[keep].idx.size());
        if (rows >= max_cut_rows && (rows >= 2 * max_cut_rows || nnz > max_cut_nnz)) break;
      }
      cuts.resize(keep);
    }
    int tr_round_readd = 0, tr_round_resep = 0;
    if (trace) {
      for (const Cut& c : cuts) {
        const bool again = !seen_rows.insert(cut_hash(c)).second;
        tr_round_readd += c.pool_id >= 0;
        tr_round_resep += again && c.pool_id < 0;
      }
      tr_added += static_cast<long long>(cuts.size());
      tr_readded += tr_round_readd, tr_resep += tr_round_resep;
    }
    int sel[16] = {0};  // selected fresh cuts per separator
    std::vector<int> cut_sepa;  // profile
    for (const Cut& c : cuts) {
      if (c.pool_id < 0 && c.sepa >= 0) ++sel[c.sepa];
      if (profile) cut_sepa.push_back(c.pool_id < 0 ? c.sepa : -1);
    }
    for (size_t s = 0; s < seps.size(); ++s) {
      if (!called[s]) continue;
      if (sel[s] > 0 || std::max(static_cast<double>(seps[s]->work()), matrix_pass) < opt.backoff_work) {
        sepa_fails[s] = 0, sepa_next[s] = round + 1;
      } else {
        ++sepa_fails[s];
        sepa_next[s] = round + (1 << std::min(sepa_fails[s] - 1, 20));
      }
    }
    {
      // Selected pool cuts leave the pool (they become rows again).
      std::vector<char> taken(dropped.size(), 0);
      bool any = false;
      for (Cut& c : cuts) {
        if (c.pool_id >= 0) taken[c.pool_id] = 1, any = true;
        c.pool_id = -1;
        c.sepa = -1;
      }
      if (any) {
        size_t w = 0;
        for (size_t k = 0; k < dropped.size(); ++k) {
          if (taken[k]) continue;
          if (w != k) dropped[w] = std::move(dropped[k]);
          ++w;
        }
        dropped.resize(w);
      }
    }
    if (cuts.empty()) {
      if (log.verbosity() >= 3) {
        for (size_t s = 0; s < seps.size(); ++s) log.log(3, "  %s: %d candidates, none accepted\n", seps[s]->name(), raw[s]);
      }
      break;
    }

    // Add as rows -inf <= a x <= rhs.
    std::vector<Index> start{0}, idx;
    std::vector<double> val, lo, up;
    for (const Cut& c : cuts) {
      idx.insert(idx.end(), c.idx.begin(), c.idx.end());
      val.insert(val.end(), c.val.begin(), c.val.end());
      start.push_back(static_cast<Index>(idx.size()));
      lo.push_back(-kInf);
      up.push_back(c.rhs);
    }
    const long long round_iters = lp.iterations();
    const Index first_new = lp.num_rows();
    const Basis basis_before = lp.get_basis();
    lp.add_rows(static_cast<Index>(cuts.size()), start, idx, val, lo, up);
    row_cuts.insert(row_cuts.end(), cuts.begin(), cuts.end());
    row_age.resize(row_cuts.size(), 0);
    if (trace) row_round.resize(row_cuts.size(), round);
    const double t_solve = timer.elapsed();
    const long long work_before = lp.work(), outer_limit = lp.work_limit();
    if (resolve_cap > 0) {
      const long long lim = work_before + resolve_cap;
      lp.set_work_limit(outer_limit >= 0 ? std::min(outer_limit, lim) : lim);
    }
    const LpStatus st = solve();
    if (resolve_cap > 0) {
      lp.set_work_limit(outer_limit);
      log.log(2, "  cut round %d: re-solve work %.3g (cap %.3g)%s\n", round, static_cast<double>(lp.work() - work_before),
              static_cast<double>(resolve_cap), lp.work() - work_before >= resolve_cap ? ", cap reached" : "");
    }
    const double round_lp_time = timer.elapsed() - t_solve;
    lp_time += round_lp_time;
    // Expensive re-solves (mzzv11: 300-500 cuts per round took 11-24k iterations each, so only 6 rounds
    // fit in the time share; 250 cuts per round re-solve in ~1k): when a re-solve needs more than a
    // tenth of the initial solve's iterations (and > 5000), halve the cuts per round. Counted in
    // iterations, not time, so that deterministic runs stay reproducible.
    const long long resolve_iters = lp.iterations() - round_iters;
    loop_work += static_cast<double>(resolve_iters) * static_cast<double>(lp.num_rows() + n);
    stats.resolve_iterations += resolve_iters;
    stats.resolve_time += round_lp_time;
    if (resolve_iters > std::max<long long>(5000, init_iters / 10) && popt.max_cuts > 50) {
      popt.max_cuts = std::max(50, popt.max_cuts / 2);
      log.log(2, "  slow LP re-solve (%lld iterations): at most %d cuts per round\n", resolve_iters, popt.max_cuts);
    }
    if (st == LpStatus::Infeasible || st == LpStatus::CutoffReached) {
      lp_optimal = false;
      // Valid cuts: the node is infeasible / cut off; keep the rows, B&B re-detects it.
      stats.cuts_added += static_cast<int>(cuts.size());
      record(cuts);
      stats.rounds = round;
      log.log(2, "  cut round %d: LP %s\n", round, to_string(st));
      break;
    }
    if (st != LpStatus::Optimal) {
      // Numerical trouble or limits: drop this round's cuts and restore the LP.
      std::vector<Index> rows;
      for (Index i = first_new; i < lp.num_rows(); ++i) rows.push_back(i);
      lp.remove_rows(rows);
      row_cuts.resize(row_cuts.size() - rows.size());
      row_age.resize(row_cuts.size());
      if (trace) row_round.resize(row_cuts.size());
      lp.set_basis(basis_before);  // the optimal basis of the previous round (removing nonbasic rows breaks it)
      lp_optimal = solve() == LpStatus::Optimal;
      log.log(2, "  cut round %d: LP %s, round discarded\n", round, to_string(st));
      break;
    }
    stats.rounds = round;
    stats.cuts_added += static_cast<int>(cuts.size());
    record(cuts);
    if (opt.on_lp) opt.on_lp(lp);
    obj = lp.objective();
    stats.bound_after = obj;
    if (obj - history.back() > 1e-9 * std::max(1.0, std::abs(obj))) {
      plateau_work = 0.0;
    } else {
      plateau_work += loop_work - round_work0;
    }

    if (profile && log.verbosity() >= 2) {
      int bind[16] = {0}, sel_pool = 0;
      long long sel_nnz[16] = {0};
      for (size_t k = 0; k < cuts.size(); ++k) {
        const int s = cut_sepa[k];
        if (s < 0) {
          ++sel_pool;
        } else {
          bind[s] += !lp.row_is_basic(first_new + static_cast<Index>(k));
          sel_nnz[s] += static_cast<long long>(cuts[k].idx.size());
        }
      }
      char buf[1000];
      int len = 0;
      double total = 0.0;
      for (size_t s = 0; s < seps.size(); ++s) {
        total += round_sep_time[s];
        if (!called[s]) {
          len += std::snprintf(buf + len, sizeof(buf) - len, " %s -", seps[s]->name());
          continue;
        }
        if (raw[s] == 0 && round_sep_time[s] < 0.005) continue;
        len += std::snprintf(buf + len, sizeof(buf) - len, " %s %.3fs %d/%d/%d/%d w=%.3g", seps[s]->name(),
                             round_sep_time[s], raw[s], found[s], sel[s], bind[s],
                             static_cast<double>(seps[s]->work()));
        if (sel[s] > 0) len += std::snprintf(buf + len, sizeof(buf) - len, " (nnz %lld)", sel_nnz[s] / sel[s]);
      }
      log.log(2, "  profile round %2d: separation %.3fs:%s (raw/kept/selected/binding)%s; re-solve %.3fs %lld iters "
              "w=%.3g; gain %.4g; init w=%.3g\n", round, total, buf, sel_pool ? " other" : "", round_lp_time,
              resolve_iters, static_cast<double>(resolve_iters) * static_cast<double>(lp.num_rows() + n),
              lp.objective() - history.back(), initial_work);
    }
    // Drop cuts that are no longer binding (basic logical with positive slack), after cut_age further slack
    // re-solves (see remove_slack).
    int tr_age[4] = {0, 0, 0, 0};
    const size_t removed = remove_slack(opt.cut_age, round, tr_age);
    if (trace) {
      const CutPool::SelectStats& ss = pool.select_stats();
      tr_removed += static_cast<long long>(removed);
      log.log(2, "  churn round %2d: %d candidates (%d from the dropped list), %d too parallel, %d over the limit; "
              "added %zu (%d re-added, %d re-separated); removed %zu (age 1: %d, 2: %d, 3-5: %d, 6+: %d)\n", round,
              pool_candidates, from_pool, ss.parallel, ss.over_limit, cuts.size(), tr_round_readd, tr_round_resep,
              removed, tr_age[0], tr_age[1], tr_age[2], tr_age[3]);
    }

    const bool dual_degenerate = is_dual_degenerate();
    last_degenerate = dual_degenerate;
    degenerate_rounds += dual_degenerate;
    // Bit 2: after every round of a dual degenerate LP; bit 4: only while the bound has not moved (flat loops).
    if (dual_degenerate && ((opt.degen_moves & 2) ||
                            ((opt.degen_moves & 4) && flat_lp &&
                             obj - stats.bound_before <= 1e-9 * std::max(1.0, std::abs(obj))))) {
      move_to_integral("after the round");
      obj = lp.objective();
    }
    frac_now = count_frac();
    if (frac_now < frac_best) frac_best = frac_now, frac_best_round = round;
    log.log(3, "  round %d: %d fractional integer columns (initial LP %d)\n", round, static_cast<int>(frac_now),
            static_cast<int>(frac_initial));

    if (log.verbosity() >= 2) {
      char buf[400];
      int len = 0;
      for (size_t s = 0; s < seps.size(); ++s) {
        len += std::snprintf(buf + len, sizeof(buf) - len, " %s %d", seps[s]->name(), found[s]);
      }
      if (from_pool > 0) len += std::snprintf(buf + len, sizeof(buf) - len, " pool %d", from_pool);
      log.log(2, "  cut round %2d: bound %.10g, added %zu (found%s), %d cut rows, %lld LP iters (%.2fs), %s%.2fs\n", round,
              p.display_sense * (obj + p.offset), cuts.size(), buf, static_cast<int>(lp.num_rows() - m0),
              lp.iterations() - round_iters, round_lp_time, dual_degenerate ? "degenerate, " : "", timer.elapsed());
    }

    history.push_back(obj);
    if (round == opt.max_rounds) work_std = loop_work;
    const size_t h = history.size();
    if (h > static_cast<size_t>(opt.stall_rounds)) {
      const double old = history[h - 1 - opt.stall_rounds];
      if (obj - old < opt.stall_rel * std::max(1.0, std::abs(obj))) {
        // While the bound has not moved at all on a degenerate LP, keep going for a few
        // extra rounds within an LP work budget (a multiple of the initial solve).
        const bool flat = obj - stats.bound_before <= 1e-9 * std::max(1.0, std::abs(obj));
        const bool flat_time_ok =
            opt.flat_work_factor > 0.0
                ? loop_work <= std::max(opt.flat_work_min, opt.flat_work_factor * initial_work)
                : opt.flat_time_factor <= 0.0 ||
                      timer.elapsed() - t_lp_end <= std::max(1.0, opt.flat_time_factor * initial_lp_time);
        if (opt.flat_work_factor > 0.0 && flat && dual_degenerate) {
          log.log(3, "  flat round %d: loop work %.3g of max(%.3g, %.3g x initial %.3g)\n", round, loop_work,
                  opt.flat_work_min, opt.flat_work_factor, initial_work);
        }
        // flat_cuts bit 2: the flat rounds go on past their work cap while the count of fractional integer columns
        // reached a new low within the last two rounds.
        const bool frac_falls = (opt.flat_cuts & 2) && round - frac_best_round <= 1 && opt.flat_work_factor > 0.0 &&
                                loop_work <= opt.flat_more * std::max(opt.flat_work_min, opt.flat_work_factor * initial_work);
        if (flat && dual_degenerate && !flat_time_ok && !frac_falls) break;
        if (flat && dual_degenerate && extra_rounds < opt.degenerate_rounds &&
            lp.iterations() - iters0 <= 5 * std::max<long long>(init_iters, 1000)) {
          ++extra_rounds;
        } else {
          // Cheap rounds. Mostly degenerate LPs: many weak rounds are typical (awhea), keep going
          // while the bound still moves. Otherwise only while the bound curve has not flattened:
          // the last 10 rounds gave at least cheap_recent of the total gain (gap instances).
          bool more;
          if (2 * degenerate_rounds >= round) {
            more = round < opt.cheap_min_rounds ||
                   obj - history[h > 20 ? h - 21 : 0] >= opt.cheap_rel * std::max(1.0, std::abs(obj));
          } else {
            const double recent = obj - (h > 10 ? history[h - 11] : stats.bound_before);
            more = recent >= opt.cheap_recent * (obj - stats.bound_before) && recent > 0.0;
            // Or while the last 10 rounds still moved the bound by cheap_recent_rel relative to its
            // magnitude (kasai, roll3000, beasleyC3: a long, steady tail that the tree needs).
            if (!more && opt.cheap_recent_rel > 0.0) {
              more = recent >= opt.cheap_recent_rel * std::max(1.0, std::abs(obj));
            }
          }
          if (!(more && cheap())) break;
          // A plateau after the bound moved gets the work budget of the flat rounds at the start
          // (lectsched-5-obj: bound 14 -> 15 in 4 rounds, then 19 rounds / 17 s without gain;
          // neos-3555904-turama: 20 rounds).
          if (opt.plateau_work && !flat && opt.flat_work_factor > 0.0 &&
              plateau_work > std::max(opt.flat_work_min, opt.flat_work_factor * initial_work)) {
            log.log(2, "  no bound change for %.3g work units (initial LP %.3g): no further round\n", plateau_work,
                    initial_work);
            break;
          }
        }
      }
    }
  }
  if (log.verbosity() >= 2 && stats.rounds > 0) {
    char buf[400];
    int len = 0;
    for (size_t s = 0; s < seps.size(); ++s) {
      len += std::snprintf(buf + len, sizeof(buf) - len, " %s %.2fs", seps[s]->name(), sep_time[s]);
    }
    log.log(2, "  separation time:%s; LP re-solves %.2fs\n", buf, lp_time);
    // Cut rows left in the LP by separator (compare with other solvers' "cutting planes" statistics).
    std::vector<std::pair<std::string, int>> by_origin;
    for (const Cut& c : row_cuts) {
      auto it = std::find_if(by_origin.begin(), by_origin.end(), [&](const auto& e) { return e.first == c.origin; });
      if (it == by_origin.end()) by_origin.emplace_back(c.origin, 1);
      else ++it->second;
    }
    std::string s;
    for (const auto& [name, cnt] : by_origin) s += " " + name + " " + std::to_string(cnt);
    log.log(2, "  cut rows by separator:%s\n", s.c_str());
    if (trace) {
      log.log(2, "  churn: %lld cuts added (%lld re-added from the dropped list, %lld re-separated), %lld removed for "
              "slack, %zu distinct cuts in the LP at some point, %zu in the dropped list\n", tr_added, tr_readded,
              tr_resep, tr_removed, seen_rows.size(), dropped.size());
    }
  }
  if (opt.cut_age > 0 && lp_optimal && stats.rounds > 0) {
    // Cut rows with basic logicals leave now: the tree's LPs keep only the cuts the root LP vertex rests on.
    int hist[4] = {0, 0, 0, 0};
    const size_t removed = remove_slack(-1, stats.rounds, hist);
    log.log(2, "  %zu cut rows with basic logicals removed at the end of the loop\n", removed);
  }
  if (!rc_saved.empty()) {
    // Restore the bounds (boxed columns left at the wrong bound are flipped by the next solve).
    for (const auto& [j, b] : rc_saved) lp.set_col_bounds(j, b.first, b.second);
    log.log(2, "  cut loop reduced-cost fixing: %d bound changes on %zu columns\n", rc_fixed_total, rc_saved.size());
  }
  // flat_cuts bit 1: cuts that did not move the bound stay when they made the LP point more integral (at most
  // flat_keep times the fractional integer columns of the initial LP): only the rows with a basic logical leave.
  // Without it the tree's LPs start from the cut-free vertex again (CMS750_4).
  bool keep_flat = false;
  {
    if ((opt.flat_cuts & 1) && flat_lp && lp_optimal && stats.rounds > 0 && lp.num_rows() > first_removable &&
        stats.bound_after - stats.bound_before - fix_gain <= 1e-6 * std::max(1.0, std::abs(stats.bound_before)) &&
        static_cast<double>(frac_now) <= opt.flat_keep * static_cast<double>(frac_initial)) {
      // Counted first: at most flat_rows x the problem's rows may stay (cut rows that never moved the bound make
      // every node LP larger: fhnw-binpack4-4, 122 rows on 493, 65,000 nodes at twice the time).
      Index binding = 0;
      for (Index i = first_removable; i < lp.num_rows(); ++i) binding += !lp.row_is_basic(i) || i < r0;
      if (static_cast<double>(binding) <= opt.flat_rows * static_cast<double>(m0)) {
        int hist[4] = {0, 0, 0, 0};
        const size_t removed = remove_slack(-1, stats.rounds, hist);
        keep_flat = lp.num_rows() > first_removable;
        log.log(2, "  cuts did not improve the bound; %d fractional integer columns (initial LP %d): %d binding cut "
                "rows kept, %zu removed\n", static_cast<int>(frac_now), static_cast<int>(frac_initial),
                static_cast<int>(lp.num_rows() - first_removable), removed);
      }
    }
  }
  // Cuts that did not move the bound only make node LPs larger: drop them all.
  if (!keep_flat && stats.rounds > 0 && lp.num_rows() > first_removable &&
      stats.bound_after - stats.bound_before - fix_gain <= 1e-6 * std::max(1.0, std::abs(stats.bound_before))) {
    std::vector<Index> rows;
    for (Index i = first_removable; i < lp.num_rows(); ++i) rows.push_back(i);
    lp.remove_rows(rows);
    // Back to the rows of the initial LP: its optimal basis is optimal again. remove_rows drops the basis when a
    // removed row's logical is nonbasic, and the re-solve then started from the slack basis (triptim1: the root
    // LP took 36 s by barrier, and that dual simplex the remaining 245 s of a 300 s run).
    if (first_removable == r0) lp.set_basis(basis0);
    log.log(2, "  cuts did not improve the bound; removed\n");
  }
  // Removing basic rows keeps the basis optimal; re-solve cheaply so that the
  // LP state is consistent for the caller.
  if (stats.rounds > 0 || !rc_saved.empty()) solve();
  if (dropped_out) {
    for (Cut& c : dropped) c.pool_id = -1;
    *dropped_out = std::move(dropped);
  }
  return stats;
}

}  // namespace milp
