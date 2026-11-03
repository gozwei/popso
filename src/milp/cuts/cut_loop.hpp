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
CutLoopStats root_cut_loop(const MipProblem& p, DualSimplex& lp, const SolverParams& params, const Logger& log,
                           const Timer& timer, double cutoff, CutData* data = nullptr, bool sub_mip = false,
                           std::vector<Cut>* dropped = nullptr,
                           const std::function<void(const DualSimplex&)>& on_lp = {},
                           const std::atomic<bool>* stop = nullptr,
                           const std::function<double()>& cutoff_fn = {});

struct CutLoopOptions {
  int max_rounds = 30;
  int stall_rounds = 3;          // stop if the bound improved less than stall_rel over this many rounds
  double stall_rel = 1e-3;
  int degenerate_rounds = 12;    // extra rounds while the bound is flat on a dual degenerate LP
  // > 0: the flat extra rounds also stop once the loop (after the initial solve) took this multiple of the
  // initial solve's time (at least 1 s). Wall clock, so the root cut loop sets it in opportunistic mode only.
  double flat_time_factor = 0.0;
  // Cheap rounds: while the loop's LP iterations stay below max(cheap_work x the initial solve's, 20000),
  // up to max_rounds_cheap rounds as long as the bound moved >= cheap_rel (relative) over the last 20.
  // (and at least cheap_min_rounds: the bound often stays flat for a few rounds first).
  // On LPs that are mostly not dual degenerate, only while the last 10 rounds gave at least cheap_recent of the
  // total bound gain: long flat tails there mostly add rows that slow the tree (mik-250-20-75-4).
  int max_rounds_cheap = 200;
  int cheap_min_rounds = 10;
  int cheap_work = 20;
  double cheap_rel = 1e-4;
  double cheap_recent = 0.1;
  double cheap_recent_rel = 1e-3;  // or while they moved the bound by this much relative to |bound|
  double time_limit = kInf;      // absolute time (timer) at which the loop stops
  double time_share = 0.0;       // > 0: also allow this share of the time left after the initial LP solve,
  double total_time_limit = kInf;  // up to this absolute time
  int max_cuts_per_round = -1;   // -1: chosen from the problem size
  long long probe_work = -1;     // probing work limit (matrix entries; -1: from the problem size)
  bool gomory = true, cover = true, mir = true, clique = true, implied_bound = true, zerohalf = true;
  Tolerances tol;
  // Called with the LP after every optimal solve of the loop (the initial one and each round's):
  // the root search records reduced costs of all these LP optima for reduced-cost fixing.
  std::function<void(const DualSimplex&)> on_lp;
  // Optional: no further round once set (a sub-MIP whose caller stops, e.g. the FJ thread's
  // mutation sub-MIPs when the root restarts; its join waited for a whole sub-MIP root before).
  const std::atomic<bool>* stop = nullptr;
  // Optional: the current cutoff (objective without offset, +inf none): reduced-cost fixing in the LP
  // between rounds (see cut_loop.cpp; the fixed bounds are restored at the end).
  std::function<double()> cutoff_fn;
};

// The loop behind root_cut_loop. If `all_cuts` is given, every cut added to
// the LP is also appended there (for testing).
CutLoopStats cut_loop(const MipProblem& p, DualSimplex& lp, const CutLoopOptions& opt, const Logger& log,
                      const Timer& timer, double cutoff, std::vector<Cut>* all_cuts = nullptr, CutData* data = nullptr,
                      std::vector<Cut>* dropped_out = nullptr);

}  // namespace milp
