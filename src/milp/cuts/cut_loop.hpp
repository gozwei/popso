#pragma once
// Root cutting-plane loop: repeatedly solves the LP, separates violated
// valid inequalities and appends them as rows to the LP.

#include <atomic>
#include <functional>
#include <vector>

#include "milp/core/log.hpp"
#include "milp/core/timer.hpp"
#include "milp/cuts/cut.hpp"
#include "milp/lp/simplex.hpp"
#include "milp/search/mip_problem.hpp"
#include "milp/solver/solver.hpp"

namespace milp {

struct CutLoopStats {
  int rounds = 0;
  int cuts_added = 0;
  double bound_before = -kInf;  // LP objective (without offset) before cuts
  double bound_after = -kInf;
  // The rounds' LP re-solves: dual simplex iterations and wall-clock seconds.
  long long resolve_iterations = 0;
  double resolve_time = 0.0;
  double initial_lp_time = 0.0;  // the loop's first LP solve
  double sep_time = 0.0;         // all separators, wall-clock seconds
  long long global_fixed = 0;  // columns fixed by CutLoopOptions::fix_fn
};

// `lp` holds the problem `p` (possibly with rows appended beyond p's rows).
// On return the LP contains the problem rows followed by the kept cuts.
// `cutoff` is the current incumbent value (without offset) or +inf.
// Cuts are disabled when the environment variable MILP_NO_CUTS is set to a
// non-empty value other than "0".
// `data` (optional): shared separator data built for p; probing results are added to it.
// `sub_mip`: the search is a sub-MIP heuristic (no zero-half cuts: they made RENS sub-MIPs on set
// packing slower without helping the heuristics).
// `dropped` (optional): receives the cuts removed from the LP for slack (globally valid; seed of
// the node cut pool).
// `cutoff_fn` (optional): the current cutoff for reduced-cost fixing in the LP between rounds.
// `max_rounds` >= 0: at most this many rounds (no cheap extra rounds; restarted roots).
// `gap_closed` (optional): true when an LP bound closes the gap to the current incumbent; no further
// round then (MILP_CUT_GAP_STOP=0: ignored).
// `lp_checkpoint` (optional): CutLoopOptions::initial_checkpoint, every `checkpoint_every` iterations.
// `threads`: CutLoopOptions::threads.
// `fix_fn` (optional): CutLoopOptions::fix_fn.
CutLoopStats root_cut_loop(const MipProblem& p, DualSimplex& lp, const SolverParams& params, const Logger& log,
                           const Timer& timer, double cutoff, CutData* data = nullptr, bool sub_mip = false,
                           std::vector<Cut>* dropped = nullptr,
                           const std::function<void(const DualSimplex&)>& on_lp = {},
                           const std::atomic<bool>* stop = nullptr,
                           const std::function<double()>& cutoff_fn = {}, int max_rounds = -1,
                           const std::function<bool(double)>& gap_closed = {}, long long prior_iterations = 0,
                           const std::function<bool(long long)>& lp_checkpoint = {}, long long checkpoint_every = 0,
                           int threads = 1,
                           const std::function<Index(std::vector<double>&, std::vector<double>&, std::vector<Index>&)>&
                               fix_fn = {});

struct CutLoopOptions {
  int max_rounds = 30;
  int stall_rounds = 3;          // stop if the bound improved less than stall_rel over this many rounds
  double stall_rel = 1e-3;
  int degenerate_rounds = 12;    // extra rounds while the bound is flat on a dual degenerate LP
  // > 0: the flat extra rounds also stop once the loop (after the initial solve) took this multiple of the
  // initial solve's time (at least 1 s). Wall clock, so the root cut loop sets it in opportunistic mode only.
  double flat_time_factor = 0.0;
  // > 0: the same cap in work units instead of time, in both modes (flat_time_factor is then ignored): the
  // loop's work after the initial solve (LP iterations x (rows + columns) plus an estimate of the
  // separation, see cut_loop.cpp) up to max(flat_work_min, flat_work_factor x the initial solve's work).
  double flat_work_factor = 0.0;
  double flat_work_min = 2e8;
  // The loop's work counts the separators' own work counts (Separator::work()) instead of an estimate.
  bool sepa_work = true;
  // A plateau (rounds without bound change after the bound moved) ends the cheap rounds once its work
  // exceeds the flat rounds' cap (flat_work_factor x the initial solve's work, at least flat_work_min).
  bool plateau_work = true;
  // A separator none of whose cuts was selected skips the next 2^(k-1) - 1 rounds after the k-th such call
  // in a row (called every round again once a cut of it is selected). Only calls with at least backoff_work
  // work count (Separator::work(), at least a pass over the problem's matrix; 1e7 is ~10 ms): cheap calls are
  // not worth skipping, and skipping them changes the cuts of small models where cuts pay off (at 2e5, mc11's
  // ~1 ms network cut calls backed off: tree 10-22 s -> 33-36 s on 2 seeds; rococoC10-001000 lost a late
  // bound jump of its Gomory cuts).
  bool backoff = true;
  double backoff_work = 1e7;
  // Cheap rounds: while the loop's LP iterations stay below max(cheap_work x the initial solve's, 20000),
  // up to max_rounds_cheap rounds as long as the bound moved >= cheap_rel (relative) over the last 20.
  // (and at least cheap_min_rounds: the bound often stays flat for a few rounds first).
  // On LPs that are mostly not dual degenerate, only while the last 10 rounds gave at least cheap_recent of the
  // total bound gain: long flat tails there mostly add rows that slow the tree (mik-250-20-75-4).
  int max_rounds_cheap = 200;
  // > 0: the cheap rounds (those after max_rounds) also stop once their work (LP and separation, the loop's
  // work count) exceeds this multiple of the work of the rounds before them. Their LP iterations may be
  // cheap while the separation is not (app1-1: c-MIR ~60 ms per round, 200 rounds, bound -9.6 -> -8.5 over
  // the last 160).
  double cheap_sep_factor = 0.0;
  int cheap_min_rounds = 10;
  int cheap_work = 20;
  double cheap_rel = 1e-4;
  double cheap_recent = 0.1;
  double cheap_recent_rel = 1e-3;  // or while they moved the bound by this much relative to |bound|
  double time_limit = kInf;      // absolute time (timer) at which the loop stops
  double time_share = 0.0;       // > 0: also allow this share of the time left after the initial LP solve,
  double total_time_limit = kInf;  // up to this absolute time
  int max_cuts_per_round = -1;   // -1: chosen from the problem size
  bool keep_rows = false;        // never remove the rows the LP had on entry (only the loop's own cuts)
  int cut_age = 0;               // a slack cut row stays for this many further slack re-solves (cut_loop.cpp)
  // Dual simplex iterations the LP's optimal basis stands for when it was found before the loop by other
  // means (a barrier root: its warm initial solve takes ~0 iterations); counted as the initial solve's in the
  // budgets relative to it.
  long long prior_iterations = 0;
  long long probe_work = -1;     // probing work limit (matrix entries; -1: from the problem size)
  bool gomory = true, cover = true, mir = true, clique = true, implied_bound = true, zerohalf = true;
  bool mcf = true;  // network cut-sets (MILP_MCF=0: off)
  Tolerances tol;
  // Called with the LP after every optimal solve of the loop (the initial one and each round's):
  // the root search records reduced costs of all these LP optima for reduced-cost fixing.
  std::function<void(const DualSimplex&)> on_lp;
  // Optional: LpOptions::checkpoint of the initial solve (every checkpoint_every iterations).
  std::function<bool(long long)> initial_checkpoint;
  long long checkpoint_every = 0;
  // Optional: no further round once set (a sub-MIP whose caller stops, e.g. the FJ thread's
  // mutation sub-MIPs when the root restarts; its join waited for a whole sub-MIP root before).
  const std::atomic<bool>* stop = nullptr;
  // Optional: the current cutoff (objective without offset, +inf none): reduced-cost fixing in the LP
  // between rounds (see cut_loop.cpp; the fixed bounds are restored at the end).
  std::function<double()> cutoff_fn;
  // Optional: true when the LP bound (without offset) closes the gap to the current incumbent: no further
  // round (cbs-cta: the incumbent 0 = the root bound arrived at 1.8 s, 13 more rounds took 6 s).
  std::function<bool(double)> gap_closed;
  // Optional: bound tightenings valid for every point better than the current cutoff (the search's objective
  // propagation with clique implications, search/obj_cliques.hpp): called with the LP's column bounds before
  // each round, tightens them in place, appends the changed columns and returns their number. The changes stay
  // in the LP (they are global bounds of the search, unlike the reduced-cost fixings above); the LP is re-solved.
  std::function<Index(std::vector<double>&, std::vector<double>&, std::vector<Index>&)> fix_fn;
  int threads = 1;  // SepaContext::threads of the rounds (the cuts do not depend on it)
};

// The loop behind root_cut_loop. If `all_cuts` is given, every cut added to
// the LP is also appended there (for testing).
CutLoopStats cut_loop(const MipProblem& p, DualSimplex& lp, const CutLoopOptions& opt, const Logger& log,
                      const Timer& timer, double cutoff, std::vector<Cut>* all_cuts = nullptr, CutData* data = nullptr,
                      std::vector<Cut>* dropped_out = nullptr);

}  // namespace milp
