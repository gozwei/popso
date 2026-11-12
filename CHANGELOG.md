# Changelog

## v0.10.2 — 2026-11

- Nonconvex models: a first-solution heuristic before the search ("fix-and-solve"). The columns of a vertex cover
  of the products' graph are fixed at a point's values, which leaves a model without nonconvex terms; its solution
  is a solution of the model, and the solve repeats with another cover while the objective improves. Where the
  fixed model is infeasible, the two sides alternate while minimizing the violation of the rows (slack columns),
  from up to four starting points. A fifth of the time limit at most; `MILP_SPATIAL_FIX=0` turns it off. On
  QPLIB's 261 nonconvex instances (120 s) popso now returns a solution on 135 (v0.10.1: 111); the number proven
  optimal is unchanged (23).
- Nonconvex models, opt-in: `MILP_SPATIAL_OBBT=1` tightens the factors' bounds over the relaxation before the
  search (each factor minimized and maximized, with the objective cutoff of the solution in hand). Off by default:
  no gain on QPLIB's nonconvex set.

## v0.10.1 — 2026-11

- **Fix (wrong answer in v0.10.0):** in spatial branch-and-bound a product with an integer factor was judged at
  the factor's LP value. A binary at 1e-8 (integral within the tolerance) times an unbounded factor at 5e8 then
  counted as a product of about 5, and such a point could be returned as a solution although no point with the
  binary at 0 satisfies the row (found by the fuzz test: an incumbent of 2.9e17 on a model whose optimum is 63).
  Integer factors now count at their rounded values, in the search and in the final check of the solution.
- Nonconvex models whose products all have a binary factor once the factors' implied bounds are known (bounds from
  the linear rows, not only the columns' own) are linearized exactly and solved as MILPs. A general integer factor
  with a range of at most 255 counts through its binary expansion (`x = l + sum c_k b_k`, so
  `z x = l z + sum c_k z b_k`). `MILP_SPATIAL_LINEARIZE=0` turns it off.

## v0.10.0 — 2026-11

- **Nonconvex quadratic models** are solved by spatial branch-and-bound instead of ending with the status
  `NONCONVEX`. Every product `x_i x_j` of a quadratic row that is not convex, and of the objective if that is
  not convex, becomes a column with its McCormick rows over the factors' bounds (a square: its secant and
  tangents); convex rows and a convex objective stay as they are (outer approximation). At a point that
  satisfies everything else, the search adds the McCormick rows of the node's own bounds to that node's LP and
  branches on a factor of the most violated product, continuous columns included, until every product holds to
  1e-6. Product bounds follow the factors' bounds at every node, and factor bounds follow the products'.
  - Factors without finite bounds get them from the linear rows, from the square terms of the quadratic rows and
    from the convex part of the model (the column minimized and maximized there) where possible; otherwise a
    product has only the McCormick rows its finite bounds allow, and the search branches until the domain is
    bounded. A model that is itself unbounded may then run to the time limit without a verdict.
  - The solution is checked on the model as given (products recomputed from the factors); a point that violates a
    row by more than 1.5e-6 relative is not returned.
  - `NONCONVEX` remains for nonconvex models with logical constraints (indicators, SOS, semi-continuous columns)
    and with `MILP_SPATIAL=0`.
  - This is a first version with plain McCormick relaxations: exact, and checked against SCIP on random models
    (`benchmarks/tools/nc_fuzz.py`), but slow on dense nonconvex QPs. On QPLIB's 261 nonconvex instances with continuous or general integer
    variables it proves 22 optimal in 120 s (SCIP: 30); see BENCHMARKS.
- Benchmark harness: `benchmarks/tools/qplib.py --nonconvex` converts QPLIB's 261 nonconvex instances with
  continuous or general integer variables (`benchmarks/sets/qplib_nonconvex.txt`); the reference wrapper takes
  the gap and SCIP's feasibility tolerance from `POPSO_REF_GAP` / `POPSO_REF_SCIP_FEASTOL` for comparison runs
  under popso's own settings.

## v0.9.0 — 2026-11

- Mixed-integer models with quadratic constraints get a light presolve before the convexity check and the cone
  detection. Cone members that are only copies of other columns (a continuous column whose one linear row is an
  equality with one other column, as conic benchmark format conversions define every member) are replaced by
  those columns in the quadratic rows (pure scalings only, so a cone stays a cone); and a constraint `sum q x^2 <= c t` whose `x` are
  switched off by one binary `y` (`x <= u y`, `x >= 0`) becomes the rotated cone `sum q x^2 <= c t y` (the
  perspective reformulation: the same integer points, a tighter relaxation; fixed columns in such a row's
  quadratic terms are folded first). The model keeps its columns and rows, and the solution is checked against
  the model as given, with a solve of the model as given if the check fails. `MILP_QUAD_PRESOLVE=0` turns it off.
  Bounds of the columns in quadratic rows can also be tightened from the linear rows
  (`MILP_QUAD_PRESOLVE_BOUNDS=1`; off by default: no gain on CBLIB overall).
- Rotated cones switched by a binary (`x^2 <= c u z`, `z` binary) are enforced as implications (`z = 0` fixes
  `x = 0`; an integral point that violates one is branched on `z`) where no row with a big-M says so: tangent
  cuts reach `x = 0` only in the limit there. `MILP_OA_CONE_IMPL=0` turns it off.
- An integral node point that violates the quadratic constraints after a round of tangent cuts has its continuous
  part solved by the conic interior-point method with the integers fixed: a feasible point becomes an incumbent
  candidate and gives tangent cuts at the subproblem's optimum, an infeasible assignment gives cuts from the
  infeasibility certificate. Cones that the fixings flatten (no interior) are handled as the linear constraints
  they are. `MILP_OA_INT_NLP=0` turns it off.
- `--deterministic` with quadratic constraints: the tangent cuts of a node are rows of that node's LP while it
  is processed (they reach the shared cut pool at the next synchronization point as before), so integral points
  that violate the constraints are cut off at the node, as in the default mode.
- The QP barrier accepts a point by each row's own violation (relative to that row's side), no longer relative
  to the largest side of all rows.
- Probing no longer runs away on continuous columns whose domains have shrunk to rounding errors: two such
  columns could tighten each other by 1e-18 per step without end, and the record of the steps took the memory
  (neos-5104907-jarama: 9 to 15 GB after the root, now 1.8 GB). Such a column counts as fixed, and one probe's
  chain of bound changes is capped.
- The simplex's primal loop switches to Bland's rule after many pivots in one call: with Dantzig pricing it
  could cycle on a degenerate vertex (one node LP of CBLIB's clay0305h ran into the time limit).
- Two opt-in settings for runs that start with an incumbent (issue #4): `MILP_ROOT_LNS_FIRST=1` runs the root's
  neighbourhood searches before the dives, `MILP_ROOT_LNS_ADAPT=f` lets them continue while they improve the
  incumbent, up to the fraction `f` of the time left. Neither is a default: on the warm-start benchmark set
  their effect is inside the run-to-run noise.
- Benchmark harness: `run.py --save-sols DIR` keeps verified feasible solutions as start files, `run.py --starts
  DIR` and `ab.py --starts DIR` run with them (warm-start benchmarks; `benchmarks/sets/warmstart.txt`), and
  `ab.py --primal` also compares the primal gap of the final solutions.

## v0.8.4 — 2026-11

- The root stops at once when an incumbent found before the root LP ended (a user start, feasibility jump, the
  local MIP) already meets the gap tolerance at the root bound; before, the test ran only when a new incumbent
  arrived, and the search went on with the root heuristics, strong branching and the tree. Opportunistic mode.
  Contributed by Clay Cantrell (#3).

## v0.8.3 — 2026-11

- A user start that presolve's dual reductions cut off (its presolved image violates rows although the start is
  feasible) is repaired: its integer values outside the violated rows are kept and the rest is completed by a small
  sub-MIP, which becomes the first incumbent. Before, the start only bounded the search, and the heuristics that
  start from an incumbent had nothing to work with (a 43k-column scheduling model: no solution beyond the start in
  39 s; with the repair optimal in 11 s). The log says when a start is repaired or only bounds the search.
  `MILP_START_REPAIR=0` turns it off. Contributed by Clay Cantrell (#1, #2).
- Solution / start files: a line starting with `#` is a comment unless it reads `<column> <value>` for a column
  whose name starts with `#` (valid in LP files; such a start was ignored entirely).
- CLI: `--abs-gap <abs>` sets the absolute MIP gap (the C API's and Python's `mip_abs_gap`).

## v0.8.2 — 2026-11

The first release on PyPI (`pip install popso`; v0.8.1's wheels were never uploaded). macOS wheels (Apple silicon
and Intel, macOS 11+) build on the current GitHub macOS runners: number parsing and formatting fall back to
`strtod` / `snprintf` where the macOS deployment target predates the system library's floating-point `from_chars`
(macOS 26) or `to_chars` (13.3). No solver changes.

## v0.8.1 — 2026-11

`pip install popso`: wheels for Linux (x86_64), macOS (Apple silicon and Intel, 11+) and Windows (x86_64) with the
`popso` command and the Python bindings, published to PyPI by the release workflow; `pip install .` builds from a
source checkout (CMake, a C++20 compiler). The Python package's metadata moved to the top-level `pyproject.toml`
(scikit-build-core; the version comes from CMakeLists.txt). No solver changes.

## v0.8.0 — 2026-11

Quadratic constraints and cones: convex quadratically constrained models (QCQP / MIQCQP) and second-order cone
programs (SOCP / MISOCP), read from MPS `QCMATRIX`, LP-format brackets in constraints or the conic benchmark format
CBF, solved by outer approximation inside the MILP search with a new conic interior-point method for continuous
models and root relaxations; products of binaries in constraints and objectives are linearized exactly, so
nonconvex models over binaries are solved as MILPs. Continuous QPs go through the conic method first, MIQPs race the
MIQP search against the outer approximation. On QPLIB and CBLIB (1 thread, 120 s) popso solves 186 of 346 models,
SCIP 101 (from its 300 s references). The MILP search is unchanged apart from fixes (core set: no regression).

**Quadratic constraints (convex)**
- Models may have quadratic terms in their constraints: MPS `QCMATRIX` (and CPLEX's `QSECTION <row>`), LP-format
  `[ ... ]` on a constraint's left side; they are written back as `QCMATRIX` and carried by `--write-anon`.
- Solved by outer approximation inside the MILP search, for continuous and mixed-integer models alike: each convex
  quadratic row (positive semidefinite towards a finite upper side, negative semidefinite towards a finite lower side;
  ranged or nonconvex rows are reported as such) is kept out of the LP and represented by gradient cuts, from tangent
  planes at the start, the root cut loop, and at every node while the LP point violates it; every candidate point is
  checked against the rows, and one that only violates them gets its continuous part from the fixed-integer outer
  approximation. A convex quadratic objective goes along as the epigraph row 1/2 x' Q x - eta <= 0. Presolve,
  symmetry handling and restarts are off for these models for now. 240 random convex MIQCQPs (2 to 9 columns,
  continuous / integer / binary, quadratic objectives in a quarter) agree with SCIP 10 within 1e-5, except three
  where SCIP's answer profits from its 1e-6 tolerance on a degenerate row.
- Second-order cones written as quadratic rows are accepted although their quadratic part is indefinite:
  `x'x <= t^2` (and with weights, shifts and a constant: one sheet of a hyperboloid) over `t >= 0`, and the rotated
  `x'x <= c u v` over `u, v >= 0`. Their cuts are tangents of the norm form `||x|| - t <= 0`, and a point is judged
  by that form too: the squared form's tolerance would let a column reach `sqrt(1e-6)` at the cone's apex. The
  benchmark harness's verifier does the same (a SCIP point with `x^2 <= t^2 + 9e-10` at `t = 0`, i.e. `|x| = 3e-5`,
  no longer passes as a better optimum).
- A conic interior-point method (homogeneous self-dual embedding, Nesterov-Todd scaling, Mehrotra predictor-corrector,
  as in ECOS) solves the continuous relaxation in second-order cone form: convex quadratic rows become rotated cones
  through a factor of their matrix, cones stay cones. Continuous models are solved by it outright (the 13 continuous
  convex QCQPs of QPLIB in 0.03 to 4.2 s each, objectives within 1e-6 of QPLIB's; the outer approximation solved one
  of them in 60 s); with integer variables, its optimum gives the first tangents, so the root LP starts at the
  relaxation's bound. `MILP_CONIC=0` turns it off. A convex quadratic objective enters the conic method directly
  (Clarabel's embedding), not through a cone.
- Continuous QPs are solved by the conic method first, with row duals and reduced costs from its duals; anything but
  an optimum or the time limit goes to the QP barrier (it claimed three feasible Maros-Meszaros QPs with objectives of
  1e9-1e12 infeasible). The barrier stalls on some QPs (QPLIB_8559 / 8567: no solution in 60 s; conic 5 s). QPLIB QPs
  (server, 120 s): 20 -> 22 of 25 solved, shifted geomean 0.83; Maros-Meszaros (132, 60 s): 123 -> 125, 0.96, the same
  objectives. `MILP_QP_CONIC=0`: the barrier only.
- MIQP (default mode, 2 threads or more): the MIQP branch-and-bound (1 thread) races the outer approximation in the
  MILP search (the other threads); the first conclusive answer stops the other, else the better incumbent with the
  tighter bound. QPLIB MIQP + generated MIQPs (120 s): the same 19 solved at the same speed, and incumbents where
  neither finishes (QPLIB_5577 / 5924: none before, now within 0.03% / 0.2% of QPLIB's optimum). `MILP_MIQP_RACE=0`
  turns it off.
- Quadratic rows: the fixed-integer repair of a candidate point (its continuous part re-solved with the integers
  fixed, the conic method as the fallback) is remembered by integer assignment, so a repeated assignment costs
  nothing. Deterministic mode, which has no node outer approximation, repaired the same points node after node:
  portqc_hard_1 went from the time limit at 60 s to optimal in 2.2 s (3 repairs, 1021 repeats).
- The LP-guided sub-MIP heuristic's dive re-solves (restoring an LP point after a failed fixing) respect its time
  budget: on CBLIB uflquad-nopsc-30-300 (45300 rows) one ran 30 s past a 1.4 s budget, ending a 30 s run at 49 s.
- Deterministic mode: a round of the parallel search ends at its next node once the time limit has passed. Nodes
  whose work the round budget does not count (the fixed-integer repair of an integral point that violates quadratic
  rows) made one round run past the limit: portqc_hard_1 at 60 s ended after 91.7 s, now 60.3 s.
- Outer approximation at nodes: a fractional LP point is branched on once the quadratic rows' violations are below
  1e-4 (was 1e-7; integral points still need 1e-7): CBLIB (157, 30 s) solved 96 -> 102, shifted geomean 0.85; at 1e-7
  sssd-strong-20-8 spent about 13 rounds of cuts per node (37.9 -> 4.3 s). `MILP_OA_NODE_TOL` sets it.
- Conic method: a stall ends it (the worst residual or gap not halved in 10 iterations on the way to an optimum: its
  best iterate within 1e-6, else a numerical error and the caller's fallback): QPLIB_10038 crept for 45 iterations of
  0.9 s to the time limit, now the QP barrier takes over after 22 s.
- Conic method: optimal also requires the primal and dual objectives to agree, relative to the objective with its
  constant (Clarabel's gap): s'z alone, relative to `c'x + 1/2 x'Px`, let Maros-Meszaros GOULDQP3 (a constant of
  ~29648 cancelling it) end at 2.062811 against the optimum 2.062784.
- Products of binaries are linearized exactly: in a quadratic row whose terms all involve binaries (a product of two
  binaries becomes a column y with y >= x_i + x_j - 1, y <= x_i, y <= x_j; a binary's square the binary; a binary
  times a column with finite bounds a column with the four McCormick rows; a term with a fixed column a linear term),
  convex or not, the row becomes linear; likewise a nonconvex objective whose terms all qualify, so such models (QPLIB
  classes QBL, QBN, LBQ, QBQ) are solved as MILPs instead of reported `NONCONVEX`. On QPLIB's quadratic knapsack rows
  (class LMC) the linear rows propagate and cut much better than tangent planes: QPLIB_10010 optimal in 10 s
  (2.3% gap at 60 s before), 8 of 10 within 60 s (2.5 to 29 s; SCIP 6 of 12 in 300 s). Only the side of y = x_i x_j
  that a row or the objective needs is written (one row per product in a <= row with positive coefficients). `MILP_QBIN_LINEARIZE=0` turns it off (else: at most
  that many new columns, default 200000), `MILP_QBIN_CONT=0` keeps binary-bounded products quadratic,
  `MILP_QBIN_OBJ=2` linearizes convex objectives too (0: none).
- MIQP: when the MIQP search's QP relaxations fail numerically without a solution, the outer approximation in the
  MILP search takes over (QPLIB_10040 / 10057 / 10060, nearly singular Q accepted as convex up to the 1e-6
  perturbation: NUMERIC in 0.1 s, now OPTIMAL in 0.1-0.4 s).
- A cone row's violation is its norm form's distance relative to the cone's constants only (as a linear row's to its
  side), no longer relative to the head: a rotated cone written as `(u - v)^2 + 4 x^2 <= (u + v)^2` (CBLIB's clay
  instances, through affine member columns) let `x` reach 1.5 at `u = 0, v = 6575`, and clay0304h ended "optimal"
  6% below its optimum (37877.8 vs 40262.39). The benchmark harness verifies cones the same way.
- The reported bound of a search ended by an incumbent within the gap of the root LP bound (or the trivial bound) is
  that bound, no longer the incumbent itself (which claimed a zero gap while the optimum could lie up to the gap
  below it).
- QP: a barrier iterate with a NaN part could be kept as the best one and reported optimal with objective `nan`
  (QPLIB_10034: after its first point was retried for violating the rows by 2e-6); now such an iterate is a
  numerical error, and when the retry fails the first point is returned (marked approximate).
- QP: a barrier point that violates the rows by more than 1e-6 is first projected onto the violated and active rows
  (a sparse least-squares correction, the columns clipped to their bounds), before the slower retry without scaling
  (QPLIB_9002: 5e-5 -> 6e-7, 0.14 s instead of 2.0 s; QPLIB_10034: 2.2e-6 -> 9e-10 where the retry failed).
- QP convexity check: a cross term of at most 1e-6 (relative) on a column without a square term is accepted with that
  much added to the two diagonals (QPLIB_8515: `1.5625e-8 x2 x3` with no `x3^2`, reported nonconvex; QPLIB counts it
  convex), within the tolerance already documented for round-off in Q.
- Duals for continuous models with quadratic constraints, from the conic method: row duals (a quadratic row's
  multiplier included) and reduced costs `d = c + Qx - A'y - sum_q y_q grad q(x)` (`--duals`, `popso_get_duals`, Python
  `Result.duals`).
- MPS files are read as a stream (gzip data decompressed in pieces, names kept, the rest of each line dropped): the
  whole text is no longer in memory (QPLIB_10007, a dense `QCMATRIX` of 19 million lines: 1.59 -> 1.06 GB peak).
- MILP: the feasibility-jump helper reports its best point at most every 3e7 effort units (~0.1 s) after the first
  (`MILP_FJ_REPORT`, before 0: every improving point; savsched1's 2335 points cost 8.4 s of checks). Core set, 4 seeds:
  time ratio 0.986 [0.955, 1.019], solved 218 -> 219 of 252.
- CBF input (the conic benchmark format of CBLIB, versions 1-3): `.cbf` / `.cbf.gz` or `--format cbf`; cones F, L+,
  L-, L=, Q and QR on variables and rows become bounds, linear rows and cone rows.
- A fixed-integer subproblem whose outer approximation stops short of the quadratic rows' tolerance (a cone's apex) is
  solved by the conic method instead (CBLIB estein4_A ended NUMERIC: nodes dropped as failed LPs).
- Time limit: the crossover's LU factorization now stops at the deadline (every 64 pivots); a large root basis kept
  highschool1-aigio running 3 minutes past a 120 s limit at 12 threads.
- Mixed-integer conic models: the relaxation's integers rounded and fixed, the continuous rest by the conic method,
  give a first incumbent when feasible.
- C API: `popso_add_qrow_terms`, `popso_num_qrow_terms`, `popso_get_qrow_terms` (monomial coefficients: `(i, j, c)`
  is `c x_i x_j`). Python: comparing a quadratic expression with `<=` / `>=` gives a quadratic constraint
  (`m.add_constr(x * x + y ** 2 <= 2)`), `Model.get_qrow_terms(row)` reads one back; a quadratic `==` is a
  `TypeError` (not convex).
- `MILP_MIQP_OA=1` (opt-in): other convex MIQPs the same way instead of the MIQP branch-and-bound. Correct on the
  generated MIQP families (objectives within 1e-7 of the MIQP search), but much slower on their dense objectives
  (miqprand_medium 11 s against 0.02 s): gradient cuts are a weak model of a dense quadratic, which the MIQP search
  solves exactly at every node.
- Benchmarks: generated MIQCQP families `portqc` (cardinality-constrained portfolio under a risk budget, a dense or
  factor-model quadratic row) and `qcrand` (random convex quadratic rows over general integers), tiny to hard, with
  SCIP references; the harness's MPS reader and verifier take `QCMATRIX` / `QSECTION <row>`.
  `benchmarks/tools/qplib.py` fetches QPLIB's convex instances and converts them to MPS, checking each conversion
  against QPLIB's published solution point.

## v0.7.0 — 2026-11

MILP: objective propagation with clique implications, a root cut loop that backs off on wide models,
faster clique and c-MIR separation, the trivial objective bound (zero-objective models end at the first
feasible point), presolve substitution with bounds implied by other rows and dual aggregation onto
variable upper bounds, a concurrent barrier on large MIP roots, safe dual bounds, and a faster
`--deterministic` mode. MIQP: sums of squares of integer variables are solved as a MILP with exact
tangent rows. New tooling to share a model or its profile without its data (`--write-anon`,
`--profile`). Several correctness fixes, also released as v0.6.2 and v0.6.3.

**Benchmarks**
- 63-instance MIPLIB 2017 subset (120 s, 4 threads), seeds 0 / 1 / 2: 61 / 60 / 63 solved, shifted
  geometric mean 9.4 / 8.7 / 7.9 s (v0.6.0: 59 / 61 / 59, 9.0 / 9.2 / 9.0 s); over all 189 runs 184
  solved and 8.7 s against 179 and 9.0 s.
- Full MIPLIB 2017 benchmark set, 240 instances (AMD EPYC 4465P, Linux, 120 s, 4 threads, references
  measured on the same machine): 89 solved, shifted geometric mean 56.0 s (HiGHS 74 / 66.3 s, SCIP 65 /
  69.1 s); 11 solved by popso only, 7 by HiGHS or SCIP but not popso. (v0.6.0's run was on an Intel
  i7-7700K: 75 solved, HiGHS 63, SCIP 53.)
- 43 hard MIPLIB 2017 instances (600 s, 4 threads): 16 solved, shifted geometric mean 298.5 s (v0.6.0: 18,
  281.5 s; HiGHS 12, SCIP 6). Solved again: neos-1171737, neos-1456979; not solved in this run: app1-2,
  csched007, icir97_tension, supportcase26 (seed-dependent: with seed 1, v0.6.3 and v0.7.0 each solve
  csched007 only).
- Generated models (1 thread, 120 s): 152 of 153 solved (v0.6.0: 146 of 148), including the new
  `sqtrack` family (production targets in squares of integers, 0.02-0.14 s each).

**Root cut loop on wide models**
- Separators back off: one none of whose cuts was selected sits out the next 2^(k-1) - 1 rounds after its
  k-th such call in a row, and runs every round again once a cut of it is selected. Only calls worth at
  least 1e7 work units (~10 ms; the separator's own count, at least one pass over the matrix) count, so
  small models keep their cut loop. On the 41 large-LP MIPLIB roots over a third of the separation time
  went to calls without a kept or selected cut (opm2-z10-s4: clique cuts, 2.4 s per round, none found;
  snp-02-004-104 and uccase12: c-MIR ~1 s per round after round 3). `MILP_SEPA_BACKOFF=0`: off.
- Gomory cuts stop a round after 20 tableau rows in a row too dense for the cut pool, when the previous
  round's rows gave no cut and the round's work has reached 1e7 units. neos-4413714-turia (190k columns)
  computed 500 such rows per round from round 5 on: Gomory 11.4 -> 1.5 s, first root LP and cuts 29.6 ->
  21.8 s with 39 instead of 26 rounds (Mac, 2 threads). `MILP_GOMORY_DENSE_STOP=0`: off.
- The cut loop's work budgets count the work the separators report (Gomory's tableau rows, c-MIR's
  aggregations, implied-bound checks, clique graph scans) instead of a Gomory estimate (`MILP_SEPA_WORK=0`:
  the estimate), and a plateau after the bound moved ends the extra cheap rounds once its work exceeds the
  cap of the flat rounds at the start (lectsched-5-obj: 17 rounds / 14 s without gain, its root 33 -> 10 s;
  `MILP_PLATEAU_WORK=0`: off).
- `MILP_CUT_PROFILE=1` (verbosity 2) logs per round and separator the time, work, candidates, cuts kept,
  selected and binding, and the re-solve's work and bound gain.

**Objective propagation with clique implications**
- Once an incumbent is known, a binary whose value 1 (or 0) forces other binaries off their cheaper value
  through the rows (two items of a row side that cannot both be 1: for a x + C y <= C with C = sum a, y
  and every item) is fixed when the objective of those forced moves alone exceeds the room between the
  bound the column bounds give and the incumbent. Applied between the root cut rounds (the fixings stay in
  the root LP) and with every later incumbent (global bounds; the root is processed again when they cut
  off its LP solution). Like reduced-cost fixing, it keeps every point better than the incumbent.
  hypothyroid-k1 (Mac, 4 threads, 3 runs): 83-85 s -> 3-6 s, solved at the root in 2.8-2.9 s (local
  MIP's incumbent at 1.3 s fixes 74 binaries: root LP -2902.85 -> -2851, the optimum); `--deterministic`
  2 threads 72 -> 52 s (68 fixings at the root node). `MILP_OBJ_CLIQUES=0`: off.

**Trivial objective bound**
- The search starts from the bound the column bounds alone give (offset + sum of min(c_j lb_j, c_j ub_j) over
  the presolved model; the offset for a zero or constant objective), and ends as soon as an incumbent meets
  it: status OPTIMAL with that bound, after postsolve as usual. In the default (opportunistic) mode the stop
  also interrupts whatever still runs for the search (root LP, cut loop, heuristics, sub-MIPs).
  ns1116954 (zero objective): local MIP's point at 3.4 s used to wait for the 131k-row root LP (50 s; 56 s
  in all, or the time limit with bound -inf); now OPTIMAL at 3.4 s. Smaller gains where the optimum is the
  trivial bound (the only core instances where it triggers; 4 threads, seeds 0-2): decomp2 0.34 -> 0.14 s,
  cbs-cta 3.2 -> 3.0 s, neos-3004026-krka 0.29 -> 0.20 s.
  Runs stopped at a limit report the trivial bound instead of -inf where it is better. `--deterministic`
  takes incumbents at deterministic points only and stops there (no interrupt), so deterministic runs stay
  reproducible; they are unchanged where no incumbent meets the bound. `MILP_TRIVIAL_BOUND=0`: off.
- `--deterministic`: the feasibility-jump / local-MIP helper's points also enter during a long initial root
  LP, every 5000 simplex iterations (deterministic points; `MILP_DET_ROOT_SYNC=k`: every k, 0: off). The helper
  used to stop after a few milliseconds of work until the root LP and its cut loop had finished. ns1116954
  (2 threads): OPTIMAL at 22 s, was the time limit without an incumbent (local MIP's point was taken too
  late). Runs whose initial root LP needs fewer than 5000 iterations are unchanged.

**Presolve: substitution with bounds implied by other rows**
- A column substituted out of an equation counts as implied free also when its other rows imply its
  bounds (given the bounds of their other columns), not only the equation itself: the equation is then
  removed instead of kept as the column's bounds. Rows whose coefficient is more than 10x smaller than
  the equation's are not used (postsolve would move their tolerance into the equation).
  `MILP_PRESOLVE_OFF=subst_implied` turns it off.
- Such substitutions, and those that keep the equation, may add up to 8 nonzeros (before: none for a
  kept equation) for continuous columns; a kept equation only after the first probing pass and not for a
  column with a variable bound (`x <= u y`: the cut separators aggregate through those flows; beasleyC3
  and mc11 went from ~15 s to the 120 s limit without this rule). `MILP_PRESOLVE_OFF=subst_fill`.
- Presolved sizes: csched008 294 x 1496 -> 277 x 1466 (rows x columns), csched007 345 x 1749 -> 326 x
  1712, milo-v12-6-r2-40-1 4045 x 1749 -> 3813 x 1560, neos-4954672-berkel 771 x 672 -> 713 x 634,
  uct-subprob 1607 x 1824 -> 1268 x 1335, rocI-4-11 2802 x 1246 -> 2473 x 978, mzzv11 8336 -> 7872 rows,
  mzzv42z 6263 -> 6054.
- miplib_wide_core (120 s, 4 threads, seeds 1-2; the same binary with both knobs off vs on): 0.950x (90%
  CI 0.918-0.983), solved 119 -> 123 of 126; roll3000 34 -> 19 s, 30n20b8 TL -> 79 s, n2seq36q 44 -> 30 s,
  beasleyC3 17 -> 12 s; eil33-2 43 -> 62 s.

**Root cut loop stops when the gap is closed**
- The root cut loop does no further round once the incumbent closes the gap at the LP bound (incumbents
  from the helper threads arrive while it runs). cbs-cta: the incumbent 0, equal to the root bound, came
  at 1.8 s and 13 more flat cut rounds took the root to 8.2 s; now 2.6 s. `--deterministic` compares with
  the incumbent at the loop's start, so deterministic runs are unchanged. `MILP_CUT_GAP_STOP=0`: off.

**Branching on flat objectives (opt-in, off by default)**
- `MILP_FLAT_HYBRID=1`: when every candidate's pseudocost score is at the zero-gain floor, the conflict and
  inference scores decide. On sct2 and csched008 (the plateaus it was made for) the bounds do not move
  either way; verbosity 2 prints how many nodes branched without pseudocost gains.

**Adaptive large-neighbourhood search (opt-in, off by default)**
- `MILP_ALNS=1`: the tree's sub-MIP heuristics are chosen by a bandit that scores each neighbourhood by
  the improvement it finds per unit of LP work, mutation fixing rates adapt after every call, and two
  neighbourhoods join: proximity search (Fischetti and Monaci: minimize the distance to the incumbent
  under an objective cutoff, stop at the first point, recentre) and graph-induced mutation (free the
  integers near a random one in the variable-row graph). A fixing neighbourhood already searched to the
  end without a better point is not solved again. glass4 reaches its optimum through a chain of
  proximity steps in 0.3 s. On 79 MIPLIB instances where v0.6.0 ends more than 1% from the best known
  value, the mean primal gap at 120 s moves 0.316 -> 0.305 (not significant), and the core set is about
  2% slower, so it stays off. `benchmarks/tools/primal_gap.py` compares final primal gaps of two sides.

**Sharing models and profiles**
- `--write-anon FILE` writes the model as MPS with generic names (x1.., c1.., s1.., i1..), rows, columns,
  SOS sets and indicators in a seeded shuffled order (`--anon-seed N`, `--anon-keep-order`) and the numbers
  unchanged, so a model can be shared for debugging without its meaning; `--anon-map FILE` writes the
  name mapping for the user only. The exported model has the same optimum (tested on MILP, QP, MIQP, SOS,
  indicator and semi-continuous models).
- `--profile FILE` writes a JSON profile of the model and the run, with no names and no data: sizes, column
  types, row classes (the MIPLIB 2017 constraint classification), coefficient and bound ranges, row
  lengths, presolved sizes, status / times / nodes / bounds, and the incumbents over time with what found
  them.
- The MPS writer now writes ranged rows so that they read back exactly whenever MPS can express them.

**Presolve: dual aggregation onto variable upper bounds**
- A binary column x with cost <= 0 whose only up-lock is a one-sided row with a binary y, where the row
  holds at y = 1 whatever its other columns are and admits no x = 1 at y = 0 (sum_S a_k x_k <= (sum_S
  a_k) y, or x <= y), equals y in some optimal solution and is substituted by y.
  `MILP_PRESOLVE_OFF=dual_agg` turns it off.
- neos-787933: 61944 aggregations, presolved 1897 x 236376 -> 131 x 1339 (was 1897 x 63708), optimal
  in 1.1 s (was the 120 s limit; HiGHS times out too). The rule fires on 3 core and 20 of the 240
  benchmark instances; on those 21 (120 s, 4 seeds): 0.810x, solved 25 -> 28 of 84.

**Large MIP roots: concurrent barrier**
- MIP roots with at least 200k nonzeros after presolve run the barrier next to the dual simplex once the
  simplex has run 2 s (opportunistic mode, 2 or more threads; `--deterministic` stays on the dual
  simplex), and the first to finish wins. After a barrier root, the root's cut, dive and LNS budgets
  count the iterations the dual simplex would have needed, the vector-length dive runs first with up to
  the barrier's time, and the crossover basis gets exact pricing weights. triptim1: no root LP within 120
  s before; now optimal at the root (50 s on 2 threads). `MILP_CONCURRENT_MIP_NNZ=0`: off.
- An incumbent equal to the root LP bound ends the search at once, also while root heuristics still
  run (opportunistic mode; `MILP_ROOT_GAP_STOP=0`: off).
- A search stopped before the root's children exist reports at least the root LP bound
  (`MILP_ROOT_BOUND_FLOOR=0`: off).
- Large-LP MIPLIB set (41 instances with 150k-1.2M nonzeros, 2 seeds): 0.988x, solved 11 -> 12 of 82;
  rail01 and neos-3988577-wolgan get a root bound within the limit (-84 instead of the trivial -1767, 119
  instead of 0). Core set unchanged in path (0.989x).

**Safe dual bounds**
- Each node LP's bound is checked against the Lagrangian bound of the LP's own duals, computed on the
  unscaled data with rounding error terms (Neumaier and Shcherbina). Where that bound is more than 1e-6
  relative below the LP objective (a dual infeasibility the scaled tolerances hid), it is used as the
  node's bound for pruning, the children and the reported bound, and the node is not closed by an
  integral LP point or used for reduced-cost fixing. The dual simplex's early stop at the cutoff carries
  the same error terms. It fires on a few MIPLIB models by 1e-6 to 5e-5 relative (app1-1, beasleyC3,
  neos17, swath1), costs 10-30 us per LP, and with the LP row-scaling fix undone it keeps var-smallemery-m6j6
  from a wrong OPTIMAL. `MILP_SAFE_BOUND=0`: off.
- The dual simplex's optimality test holds its dual feasibility tolerance (1e-7) for the row duals in the
  units of the model's rows, not of the internally scaled ones: on a row scaled up by 2^k the old test let
  duals of the wrong sign by up to 2^k x 1e-7 through, and the LP stopped with its objective above its
  optimum (the firings above). Such solves now go on with primal simplex pivots, so the node bounds are
  both valid and tight: the safe bound fires 4 times on the core set instead of 119 (deterministic, 60 s),
  at about 1e-6 relative, for under 0.1% more LP iterations. With the row-scaling fix of 0.6.1 undone,
  var-smallemery-m6j6's root LP reaches its optimum -156.06 (was -139.80). `MILP_LP_UNSCALED_DUAL=0`: off.
  Together with the next item, miplib_wide_core (120 s, 2 threads, seeds 1-4, both off vs on): 1.016x (90%
  CI 0.986-1.047), solved 211 -> 209 of 252, within the noise of these runs.
- An LP reported infeasible (node, strong branching probe, dive step, cut round, sub-MIP) now comes with a
  dual ray verified on the unscaled data with the same rounding error terms: the rows' sides weighted by
  the ray must exceed what the ray's combination of columns can reach over the column bounds. A ray that
  fails is retried once from a fresh factorization; if it fails again the LP counts as unsolved (a node is
  then branched on without it) instead of pruning. On the core set about 30k infeasible LPs are checked
  with no failure (5 pass only after the retry). `MILP_FARKAS_CHECK=0`: off.

**Root Gomory cuts keep tiny tableau entries**
- Gomory mixed-integer cuts left tableau entries below 1e-11 out of the row without their bounds, the
  same hole as the node tableau cuts fixed in v0.6.1 (no violated root cut was seen). They are now kept
  and removed later together with their bounds. Core set (120 s, 4 seeds): 0.987x, solved 210 -> 215 of
  252. `MILP_GOMORY_KEEP_TINY=0`: the old drop.

**Objective cutoff propagated on the global bounds**
- Each new incumbent propagates c x <= cutoff over the global bounds; the integer tightenings become
  global like root reduced-cost fixings and count for the restart rules. neos-4413714-turia (min z with
  z >= sum_k d_ik x_ik): the first incumbent fixes 95.7% of its 190k binaries, the root restarts on a
  3660 x 8478 model, and the run ends optimal in 74-92 s (was the 120 s limit). `MILP_GLOBAL_OBJPROP=0`:
  off.

**Zero-objective (feasibility) models**
- Symmetry: SST rows also on orbits of continuous columns, by default only when the objective is zero
  (`MILP_SST_CONT=1`: all models, `0`: off); on models with an objective they slowed the search.
- Branching: the flat-objective hybrid rule (conflict and inference scores) is on by default when the
  objective is zero, with conflict scores recorded; the inference history counts the node's own branching
  only.
- The cut pool's per-node rounds count propagation prunes as evidence on zero objectives (almost every
  leaf closes by propagation there; 90% of the run went into pool rounds).
- fhnw-binpack4-4 (infeasible 2D packing): proved infeasible in 3-6 s (was the 120 s limit; SCIP 14-17
  s). Zero-objective MIPLIB set (7 instances, 4 seeds): solved 12 -> 16 of 28.

**Parallel c-MIR separation at the root (opt-in, off by default)**
- `MILP_MIR_PAR=1`: the root cut loops' c-MIR start rows run on the search's threads (less the
  feasibility-jump helper while it runs), in fixed chunks of 64 with the work budgets checked between
  chunks, so the cuts are the same for any thread count. 4 threads (Mac): app1-1 root 14-15 -> 7-10 s,
  h80x6320d 17-21 -> 12-13 s. Off: unchanged.

**Faster clique separation**
- The clique separator's greedy growth tested each candidate literal against its own neighbour list;
  it now counts, per literal, the clique members it is adjacent to (updated only when a member joins).
  The same cuts (checked round by round on six MIPLIB models), much cheaper on dense conflict graphs:
  qap10's clique separation took 0.2-2.5 s per round; its root now fits 25 cut rounds instead of 11
  in the same time share (bound 335.50 -> 336.13) and the solve takes 37.1 s instead of 42.5 s (4
  threads). `MILP_CLIQUE_COUNT=0`: the old test.

**ZI rounding at the nodes (opt-in, off by default)**
- `MILP_ZIROUND=1`: where simple (lock) rounding fails (root, every 8th node), ZI rounding (Wallace 2010, as SCIP's
  zirounding) shifts each fractional integer variable to an integer within the slack its rows and bounds
  leave at the current point, up to three passes. It stops in a search thread after 1000 calls with fewer
  than 1% feasible points (a call is a pass over the matrix). beasleyC3: its optimum 754 from ZI rounding
  (4 threads, seeds 1-2: 16.6 / 13.1 -> 15.4 / 11.0 s); `2`: at every node. Off by default: no measurable gain
  on the MIPLIB core set.

**Fewer incumbent updates from local MIP**
- Local MIP walks through long chains of slightly better feasible points and reported every one: on
  roll3000 40874 points in 2 s, 25904 of them new incumbents 0.29 apart, each with the incumbent checks,
  root reduced-cost fixing, and an incumbent callback, trace and log entry. After its first point it now
  reports the best point at most every 3e7 effort units (~0.1 s) and at its end; the search itself is
  unchanged (roll3000: the same 40874 points, 112 incumbent updates, local MIP's phase 2.1 -> 0.9 s).
  Default mode only (`--deterministic` takes the points at its sync points as before).
  `MILP_LMIP_REPORT=0`: every point.

**Deterministic mode: fewer waits for sub-MIP jobs**
- `--deterministic` runs sub-MIPs found during the tree beside the tree rounds and lets each result enter at
  the first round barrier where the tree has done enough LP work for it; the barrier waits while a job may
  still finish below that point. The job's credit per round is now the round's LP work / threads (was / the
  threads not running jobs, which credited a running job with several threads' work per round), it enters at
  2x its LP iterations (`MILP_DET_JOB_FACTOR`, was 1x), and sub-MIP jobs report progress after every root LP
  of their cut loop. pk1 at 4 threads: barrier waits 2.5 -> 0.5 s, 7.2 -> 6.0 s. MIPLIB core set
  (deterministic, 4 threads, 2 seeds): time ratio 0.973 [0.955, 0.989], solved 112 -> 111.
  `MILP_DET_JOB_SHARE=0 MILP_DET_JOB_FACTOR=1`: the old credit.

**Squares of integer columns solved as a MILP**
- A quadratic objective that is a sum of squares of integer columns (Q diagonal, every column with a
  nonzero entry integer, or continuous but implied integer by an equality row such as e = sum x - T over
  integer columns; convex in the model's sense; plus any linear terms) is solved as a MILP: on a
  binary x^2 = x; on a general integer, with d = x - c around a center c, d^2 = max over integers k of
  (2k + 1) d - k(k + 1), so a column t >= the tangents for k within 25 of the center (and a coarse grid
  beyond) stands for d^2. The centers come from the continuous relaxation; each MILP is a relaxation of the
  model (its bound is valid), its solution is scored with the exact objective, and where a value falls
  outside its tangents a window is added and the MILP solved again (at most 10 times). The tangents give the
  convex hull of x^2 on the integers, a much tighter relaxation than the continuous x^2 the MIQP search
  branches on. A user's model (141637 rows, 137899 integer columns, 265 squared integer errors): no
  incumbent after 180 s with the MIQP search (its root dive solves a barrier QP per step), optimal at the
  root in 10.9 s now (relaxation 6.8 s, MILP 4.1 s). `MILP_SQ_MILP=0`: the MIQP search; `MILP_SQ_WINDOW`,
  `MILP_SQ_ROUNDS`, `MILP_SQ_CENTER=0` (no relaxation, centers at 0).
- The MIQP search's dives stop after max(2 s, 3x the root relaxation's time) and log their relaxations and
  time: its root dive solved up to 200 barrier QPs before the tree (on that model 5-12 s each, with no log
  line); now 5 relaxations, 40 s, then the tree with its progress lines.

**Root heuristics beside the tree (opt-in, off by default)**
- `MILP_ROOT_HEUR_ASYNC=k`: with at least k threads (default mode), the first root's heuristics (dives,
  RENS, the root LNS rounds) run on a quarter of the threads (`MILP_ROOT_HEUR_SHARE`) while the tree starts on
  the others; the heuristic threads join the tree when done, and an in-tree restart is checked as soon as
  their incumbents have fixed columns. Their time caps (RENS up to 3x the root's time, the LNS rounds at
  least the root's time) did not shrink with more threads: at 12 threads they were 40-50% of n5-3's and
  kasai's solve time. Mac, 4 threads: n5-3 19.2 -> 15.5 s, kasai 32.9 -> 25.0 s, mc11 unchanged.

**Fixes**
- A root that ends past the time limit no longer starts a restart or a tree round (deterministic
  ns1116954: 123.4 -> 120.0 s with a 120 s limit).
- Wrong optimal answer after a restart whose presolve removed the incumbent (also in v0.6.3).
- The final solution no longer profits from the feasibility tolerance: an incumbent accepted within it (LP
  points and sub-MIP solutions may leave rows up to 1e-6 beyond their sides) could carry an objective a little
  better than the optimum (MIPLIB map16715-04: -111.00075 with 257 rows exactly 1e-6 beyond their sides, optimum
  -111; fastxgemm-n2r6s0t2: 229.9997 for 230). The search's final incumbent, if it violates a row or bound by more
  than 1e-9, has its continuous part re-solved with the integers fixed at a 1e-9 tolerance (once, at most 2 s).
  `MILP_FINAL_POLISH=0`: off.
- Memory on large roots: the root's concurrent barrier is not started when the solver already holds 40% of the
  memory limit, and gives up after its symbolic analysis when its factor would not fit in half the limit less
  that (MIPLIB gfd-schedulen180f7d50m30k18, neos-5104907-jarama and neos-5114902-kasavu went past 10 GB with an
  8 GB limit in the barrier).
- `--deterministic`: the root LP's waits for the feasibility-jump / local-MIP helper end at the time limit
  (MIPLIB gfd-schedulen180f7d50m30k18 with 12 runs at a time: killed at 144 s with a 120 s limit).
- Time limit on large roots: when the root's dual simplex stopped at the time limit, the concurrent barrier was
  not told to stop, and its crossover ran on (MIPLIB savsched1, 120 s limit: ended at 180-215 s; now 120.3 s).

## v0.6.3 — 2026-11

A correctness fix on top of v0.6.2. The benchmark numbers of v0.6.0 still apply.

**Fixes**
- Wrong optimal answer after a restart whose presolve removed the incumbent: a restarted search gets the
  incumbent before the restart as its start, mapped through the restart's presolve. A dual reduction of
  that presolve may remove the point (another optimal point is kept); the mapped point was then rejected,
  and its objective, which is not the incumbent's, became the restarted search's cutoff. Below the optimum,
  the search found nothing better and the old incumbent came back as optimal (MIPLIB neos-3083819-nubu,
  `--deterministic --threads 2 --seed 4` on x86: 6315970 reported optimal, optimum 6307996). The cutoff of a
  rejected start is now the start's own objective.

## v0.6.2 — 2026-11

A correctness fix on top of v0.6.1. The benchmark numbers of v0.6.0 still apply.

**Fixes**
- Wrong optimal answer after two root restarts: when the first restarted search found nothing better
  than the incumbent it started from, the second restart was handed that start in the previous
  problem's columns; the second restarted root was pruned against the mismatched point and the old
  incumbent came back as optimal (MIPLIB drayage-25-23 on x86: 101344.27 reported optimal, optimum
  101282.65). The restart now passes the start in the restarted problem's columns, and a start whose
  size does not match the problem is ignored with a warning.

## v0.6.1 — 2026-11

Correctness and time-limit fixes on top of v0.6.0. The benchmark numbers of v0.6.0 still apply.

**Fixes**
- LP row scaling: entries below 1e-9 of a row's largest no longer set its geometric scale. A data-noise
  coefficient (MIPLIB var-smallemery-m6j6: 8.9e-16 next to 106.75) scaled the row by 2^22, so the dual
  simplex's 1e-7 tolerance on its logical accepted an unscaled reduced cost of 0.138: the root LP
  stopped at -139.80 instead of -156.06, and a run claimed OPTIMAL at -141.53 (optimum -149.38).
- Node tableau c-MIR cuts dropped tableau entries below 1e-11 of the row's largest without their
  bounds, which is not a relaxation where a column has a huge range (MIPLIB neos-4413714-turia: a
  budget row with right-hand side 9e8; cuts violated by the optimum, a reported bound 46.47 above the
  optimum 45.37). A tiny entry now leaves only with its bound.
- Conflict analysis could read conflicts already freed by the bounded per-worker conflict pool: adding
  a learned conflict could drop conflicts the node's bound trail still referred to (MIPLIB
  neos-2657525-crna: a crash in 1 of 24 runs at 4 threads; none in 48 runs now).
- Root cut loop: when the root cuts do not move the bound and are removed, the LP continues from the
  initial LP's optimal basis. Removing a cut row whose slack was nonbasic dropped the whole basis and
  the LP was re-solved from scratch (MIPLIB triptim1: that re-solve ran for the rest of a 300 s run).
- Time limits: root dives are capped by the time left when each dive starts, not when it is queued
  (`--deterministic`, 2 threads: ns1116954 ended at 235 s with a 120 s limit, now at 122 s), and
  feasibility jump checks the time limit before publishing each point (savsched1 ran 6 s past it).
- Deterministic runs give different node counts than v0.6.0 where the changed cuts or LP scaling apply.

## v0.6.0 — 2026-11

MILP: presolve sparsify, probing liftings over repeated passes with kept implications, repeated root
restarts, cheaper roots, budgets for the heuristics' LPs, a column-scaled cost perturbation in the dual
simplex, faster Forrest-Tomlin updates and compact open-node storage. Time limits are kept on very large
models.

**Benchmarks**
- Full MIPLIB 2017 benchmark set, 240 instances (Intel i7-7700K, Linux, 120 s, 4 threads), first run:
  75 solved, shifted geometric mean 62.5 s (HiGHS 63 / 76.2 s, SCIP 53 / 77.5 s); 10 solved by popso
  only, 7 by HiGHS or SCIP but not popso.
- 63-instance MIPLIB 2017 subset (120 s, 4 threads), seeds 0 / 1 / 2: 59 / 61 / 59 solved, shifted
  geometric mean 9.0 / 9.2 / 9.0 s (v0.5.0: 58 / 59 / 61, 10.7 / 10.8 / 9.9 s); about 14% faster over
  the three seeds (all 189 runs: 9.0 s against 10.5 s, 179 against 178 solved).
- 43 hard MIPLIB 2017 instances (600 s, 4 threads): 18 solved, shifted geometric mean 281.5 s (v0.5.0:
  15, 324.9 s; HiGHS 12, SCIP 6). Newly solved: app1-2, graphdraw-domain, icir97_tension,
  neos-3216931-puriri, supportcase18; glass4 534 -> 23 s. neos-1171737 and neos-1456979 are now
  unsolved at 600 s.
- Generated models (1 thread, 120 s): 146 of 148 solved, as v0.5.0 (binpack_hard_1 and gap_hard_1).
- Measured with the release candidate; the fixes since then are listed below (Time limits on large
  models, Fixes).

**Node-LP incumbents re-solved warm at a tight tolerance**
- When a node LP point is integral but, rounded, violates the model by more than 1e-9, the node LP is
  re-solved from its basis with the integer columns fixed and primal tolerance 1e-9. Its 1e-6 slack could
  grow past the model check after postsolve (dano3_5: 2.7e-6), and the cold fixed-integer re-solves that
  repaired such points could run out of time on large LPs (dano3_5: 30k iterations), so the solve was
  repeated without presolve and ended without a solution. `MILP_POLISH_LP=0` turns it off.
- miplib_wide_core (120 s, 4 threads, seeds 1-2): neutral, 1.026x (90% CI 0.994-1.060), solved 120 -> 121
  of 126.

**Presolve: sparsify**
- Equations are added to the rows of their shortest column, `r += lambda e`, when that cancels more
  entries of r than it adds (two-row nonzero cancellation, as in HiGHS's presolve). Equations of at
  most 50 entries, |lambda| <= 1e3, a row's coefficient range may not grow beyond 1e6; rows of integer
  columns with integral coefficients stay integral; set partitioning equations are not used and set
  packing / covering / partitioning rows are not changed (their cliques). It runs when the other
  reductions are exhausted, before the clique extension and probing (which find more on the sparser
  rows), up to 3 times; the following rounds remove the singleton and doubleton rows it creates. Rows
  only: no postsolve; LP and fixed-MIP duals are computed on the original rows as before.
- Root gap closed by the presolved LP (presolve only): app1-1 1.2% -> 9.4% (3603 x 1206 / 24.7k ->
  3332 x 1008 / 22.7k), app1-2 0.4% -> 16.5% (39555 x 13265 / 274k -> 38175 x 12342 / 236k; its
  presolve 2 -> 5 s, mostly probing), mzzv42z 39.2% -> 46.0%, mzzv11 34.4% -> 35.8%; fewer nonzeros on
  ran14x18-disj-8 (10.3k -> 8.0k), neos-4954672-berkel, trento1 (947 columns fewer), n5-3,
  neos-3083819-nubu, dano3_3 / dano3_5.
- The hard instances it was made for (300 s, 4 threads, one seed): neos-3216931-puriri 116 -> 70 s
  (202 -> 70 nodes), trento1 gap 0.075% -> 0.028%, neos-4954672-berkel 8.39% -> 8.22%,
  ran14x18-disj-8 56 -> 54 s; app1-2's dual bound reaches the optimum -41 (main: -43; root bound -239
  -> -66), its incumbent is worse (-22 vs -30).
- miplib_wide_core (120 s, 4 threads, seeds 1-2): neutral, 1.010x (90% CI 0.974-1.050), solved 121 ->
  118 of 126 (near-limit instances whose presolved model it does not change). On the 10 instances
  whose presolved model it changes 1.07x: rows combined with equations lose knapsack / MIR structure
  for the cut separators (sp150x300d 0.1 -> 1 s, root bound 69 -> 65; n5-3; neos859080), against
  neos-3083819-nubu (2-5x faster) and dano3_5. `MILP_PRESOLVE_OFF=sparsify` turns it off.
- Equations with a continuous column that has a variable bound (an inequality with continuous columns
  of one sign and integer columns of the other: `x <= u y`, `sum of flows <= 7 y1 + 31 y2`) are not
  used: the cut separators aggregate through those columns (flow balances with variable upper bounds
  on sp150x300d, neos-4954672-berkel and ran14x18-disj-8; arc totals cancelled out of n5-3's capacity
  rows). Their root bounds come back (sp150x300d 65 -> 69, the optimum; n5-3, berkel, ran14x18-disj-8),
  the presolve gains above stay. miplib_wide_core against sparsify without it: 0.965x (90% CI
  0.931-0.996), solved 118 -> 119 of 126; sp150x300d 1.0 -> 0.1 s. `MILP_PRESOLVE_OFF=sparsify_protect`
  turns the rule off.
- Fixed on the way (also on main): two wrong answers of the presolve on small models: GCD tightening of
  a row whose upper side lay just below a multiple of the gcd by round-off (declared infeasible), and a
  round-off coefficient (1.4e-17 on a free integer column) taken as a substitution pivot.

**Heuristic LP budgets**
- Tree dives get a share of the search's LP iterations that follows their success, as SCIP's diving
  heuristics: 5% x (1 + 10 (new incumbents + 1) / (dives + 1)) per thread, at most the previous 10%
  (10% for the first 10 dives, 5.5% after 100 without a new incumbent), and a dive starts only when the
  budget left covers a whole dive (the mean of the thread's finished dives; after a dive that ended at its
  limit, twice its iterations): a dive cut off by the budget finds nothing (pg5_34: 13 of 42 tree dives).
  `MILP_DIVE_ADAPT=0`: the flat 10%.
- Without an incumbent the guided root dive is skipped: it repeated the fractional dive exactly (same LP,
  same choices), in every sub-MIP root and in roots without a start point. `MILP_DIVE_DEDUP=0`: both.
- Sub-MIPs that look for an improvement (RINS, mutation, crossover, local branching, RENS with an
  incumbent) no longer run the feasibility pump: it ignores their cutoff, and its points were nearly
  always worse than the incumbent (6 sub-MIP incumbents from 94 pumps on 13 core instances) while it took
  a quarter to a half of those sub-MIPs' LP iterations (pg5_34, pg, nu25-pr12). `MILP_SUB_FP=1`: the pump.
- A `Heuristics:` summary line: tree dives (calls, points, new incumbents, limit hits, LP iterations) and
  tree sub-MIPs (calls, new incumbents, LP iterations, of them in the sub-MIPs' own dives and pumps).
- miplib_wide_core (120 s, 4 threads, seeds 1-2): solved 121 / 121 of 126, shifted geomean 0.987x (90% CI
  0.962-1.013), first incumbents and final gaps unchanged; more nodes in the same time where both sides
  stop at the limit (rocI-4-11 +18% / +79%, swath3 +46%).

**Probing: liftings, repeated passes, kept implications**
- Probing lifts one-sided rows with what fixing a binary x implies: when the bounds that x = v
  implies leave a row `a y <= b` a slack delta it can never use, x enters the row (x = 1:
  `a_x += delta`; x = 0: `a_x -= delta`, `b -= delta`). At most one lifting per row per pass, never
  equations or ranged rows, new entries capped at 10% of the nonzeros per pass. E.g. `y <= 100 x`
  becomes `y <= 5 x` when x = 1 forces y <= 5 through other rows.
- Probing runs up to 3 passes while a pass fixes, substitutes or lifts something, each with half the
  previous work limit and at most the work the previous one used.
- Implications between binaries found by probing (x = v forces y = w) are kept and propagated
  together with the rows by later probes and passes, so they also work backwards (y = 1 - w forces
  x = 1 - v) where the rows alone do not show it.
- Binaries are no longer substituted out of equations before the first probing pass: probing may
  fix them (neos-3216931-puriri: a binary y = sum of 96 binaries cannot be 0, which turns five such
  rows into partitioning rows).
- Root gap closed by the presolved LP (presolve only): neos-3216931-puriri 1.0% -> 12.3% (LP 58940
  -> 60354), mzzv11 19.2% -> 34.4%, mzzv42z 18.4% -> 39.2% (8769 x 9116 / 126k -> 6250 x 6444 / 74k);
  app1-1, app1-2 and milo-v12-6-r2-40-1 unchanged. At the root (2 threads) the bound after cuts on
  puriri goes 59231 -> 62214 (optimum 71320); mzzv42z is solved at the root (31 -> 19 s), mzzv11 in
  2 nodes. The probing work grows (mzzv42z 179M -> 444M units, presolve about 2x; puriri 21M -> 46M).
- miplib_wide_core (120 s, 4 threads, seeds 1-2, against the same main): solved 119 -> 118 of 126,
  shifted geomean 0.954x (90% CI 0.873-1.034); faster in every run on neos-950242 (8 -> 1-2 s),
  piperout-27 (6 -> 2 s), neos-3381206-awhea and drayage-100-23, slower on cbs-cta (2.4 -> 6.0 s: the
  heuristics find its objective-0 solution later).
- Knobs: `MILP_PRESOLVE_OFF=probing_lift`, `probing_passes` (one pass), `probing_implications`,
  `probing_keep` (all four: the previous presolve); `MILP_PROBING_PASSES=<n>`; opt-in
  `MILP_PRESOLVE_ON=probing_cliques` (the implications x = 1 => y = 0 also extend set-packing rows
  in the clique extension; no better than the liftings alone, and more nonzeros).

**Cheaper roots**
- A root whose restart is already due (enough integer columns fixed by reduced-cost fixing before the
  probes) skips root strong branching; the restarted root probes the smaller problem. The test is the
  restart rule itself, with the root's cost so far (a restart that would be refused keeps the probes), and
  it applies only to roots with some LP work (`MILP_RESTART_SKIP_SB_WORK`, 2e7 iterations x (rows +
  columns)): quick roots keep their probes, which can settle the root (exp-1-500-5-5). h80x6320d (2
  threads): 34.1 / 42.1 / 34.9 s -> 29.9 / 40.4 / 33.4 s; irp (4 threads) 6.5 / 4.6 -> 2.9 / 2.3 s.
  `MILP_RESTART_SKIP_SB=0`: probes in every root.
- With an incumbent known before the root heuristics (from local MIP or FJ), RENS now runs beside the
  dives with that incumbent as its cutoff, as without one, instead of after them with their best point:
  when the dives find the optimum, RENS can only fail and ran to its cap (h80x6320d: 10 s). h80x6320d with
  the above: 29.7 s shifted geomean over 3 seeds (34.3 s without). `MILP_ROOT_MERGE_INC=0`: the old order.
- Restarted roots from the second restart on run their dives, RENS and root LNS only if the search before
  improved the incumbent; an improvement below the gap tolerance (copies of the same point) no longer
  counts (h80x6320d `--deterministic`: 45.2 -> 32.7 s). `MILP_RESTART_IMPROVED_GAP=0`: the old test.
- The root-LP sub-MIP (runs without an incumbent; the root waits for it) gets a budget relative to the
  root: at most max(0.5 s, the root's time) (was max(2 s, ...)), deterministic max(5e7, 3x the root LP
  work) work units (was twice that with a 2e8 floor), and it stops after a first sub-MIP that is
  infeasible, where the LP accepted the fixings and lower fixing rates are guesses. enlight_hard: root 1.15
  -> 0.37 s (`--deterministic`, 2.9 -> 2.1 s). `MILP_LP_RENS_REL=0`: the old caps.
- A root strong-branching pass stops after 4 probes in a row that learn nothing (a child at the iteration
  limit, no fixing): dano3_3 26k probe iterations in 3 passes -> 3-4k (2 threads, 3 seeds: 42.5 -> 35.7 s
  shifted geomean); qap10's root 45 -> 40 s. `MILP_ROOT_SB_ABORT=k`: after k (0: off).
- The root cut loop's cap on flat rounds (the bound not moving on a dual degenerate LP) is now in work units
  instead of wall-clock time, the same in `--deterministic` mode (which had no cap): LP work plus an
  estimate of the separation, at most max(2e8, 3x the initial LP's work). The rounds no longer depend on
  the machine's load (neos-3381206-awhea needs 7 flat rounds, which a loaded machine could cut off), and
  neos-960392's 12 flat rounds in `--deterministic` mode become 1 (124 -> 89 s). `MILP_FLAT_WORK=0`: the
  time cap; `MILP_FLAT_WORK_MIN`: the floor.
- miplib_wide_core (120 s, 4 threads, seeds 1-2, against the code before these changes): solved 120 ->
  122 of 126, shifted geometric mean 10.48 -> 9.25 s (0.892x, 90% CI 0.850-0.935), on a busy machine: part
  of it is neos-3381206-awhea (22.8 / 8.7 -> 3.7 / 3.6 s), whose flat cut rounds the time cap had cut off.
  `MILP_FJ_FIRST=1` stays opt-in (an earlier combined A/B with it: 0.991x; no gain of its own here).

**Dual simplex: column-scaled cost perturbation**
- The dual simplex's cost perturbation, 5e-7 (1 + |c_j|)(1 + random) times a cost scale, now uses the
  column's largest entry (after row scaling) in place of the 1, which is what HiGHS's perturbation of
  its column-scaled LP amounts to. On columns with large entries the old perturbation sat at the level
  of the dual tolerance, where the ratio test's Harris groups tie: csched008's cold root LP crawled at
  its optimal value for ~4900 iterations (HiGHS 1458 in all). The DSE weights now also follow the
  primal simplex pivots that clean up after the perturbation is removed, instead of being reset to 1
  (every warm start from such a basis had unit weights).
- miplib_wide_core (120 s, 4 threads, 4 seeds): 241 -> 245 of 252 solved, shifted geomean 0.977x (90%
  CI 0.946-1.011); root LPs of long degenerate models cheaper (csched008 -24%, dano3 -16%); warm
  re-solves from the same vertex about as before (+2% iterations). `MILP_LP_PERTURB_COLSCALE=0`,
  `MILP_LP_PRIMAL_DSE=0` restore the old behaviour. Measured and left off: perturbing the logicals,
  an objective-stagnation trigger for the cold-solve boost, a signed Harris bound (knobs listed in
  PLAN.md).

**Repeated root restarts (pseudocost carry-over opt-in)**
- The root restarts (presolve again with the root's global fixings and cut rows) now chain: after
  the first one (at 10% of the integer columns newly fixed at the root, as before, at least 10
  columns, was 20) a restarted root restarts again at 10%, up to 4 times (`MILP_MAX_RESTARTS`, was
  1). From the second on a restart needs the presolve after the previous one to have removed at
  least 5% of the integer columns, and the restarted roots so far plus the last one's cost again
  must stay within the first root's cost: its LP work in `--deterministic` mode (so the decision is
  reproducible), its time otherwise (at least 5% of the time limit). Restarted roots from the
  second restart on are cheaper: at most 5 cut rounds (the earlier roots' cuts are carried as
  rows), and the root dives, RENS and root LNS run only if the search before the restart improved
  the incumbent.
- miplib_wide_core + mas74 (120 s, 4 threads, seeds 1-2, both sides on the same main): 122 / 122 of
  128 solved, shifted geomean 1.000x (90% CI 0.968-1.034), measured with the first restart at 5%;
  with it back at 10% the three quick instances that restarted only at 5% (binkar10_1, neos-911970,
  neos17: 0.4-1.4 s slower in both seeds) do not restart, about 0.986x. Chained restarts:
  drayage-25-23 8.8 / 7.6 -> 6.0 / 5.7 s, swath3 solved on one seed (62 s, both TL before),
  h80x6320d 31 / 38 -> 32 / 40 s, swath1 6.8 / 6.2 -> 8.7 / 6.9 s.
- Opt-in `MILP_RESTART_PC`: restarted searches start from the pseudocosts (and inference history)
  of the search before, mapped through the new presolve: `=1` as history with counts capped at the
  reliability threshold (`MILP_RESTART_PC_CAP`), `=2` as estimates only for columns without
  observations of their own. Off by default: on core instances that restart, both needed ~15-25%
  more LP iterations (deterministic runs; mas76 1.60M -> 2.51M / 2.70M, drayage-25-23 46k -> 63k /
  97k).
- Knobs: `MILP_RESTART_FRAC`, `MILP_RESTART_FRAC2`, `MILP_RESTART_MIN`, `MILP_RESTART_MINRED`,
  `MILP_RESTART_WORK`, `MILP_RESTART_ROUNDS`; the old rule is `MILP_MAX_RESTARTS=1
  MILP_RESTART_MIN=20` (identical search). The command line prints the number of restarts;
  `SolveStats` has `restarts` and `tree_restarts`.

**Interrupts and the time limit during the root cut probing**
- The probing that collects implied bounds and literal conflicts before the root cut loop now
  checks the interrupt flag (every 64 probes): an interrupt landing there took up to 1.2-2.2 s to
  stop the solve on cflp_hard_1 (2 threads), now ~0.02 s.
- It also stops at the time limit: its work budget alone could take 6.5 s (fiball, 1 thread), so
  short time limits ran over (fiball at a 4 s limit now ends at 4.04 s).

**Compact open-node storage, gradual memory drain**
- The queue of open nodes keeps a packed 88-byte record per node (was a 112-byte node) with 20-byte
  heap keys that grow by 1.25x (were 24 bytes, doubling), chunked slots that compact under memory
  pressure, warm-start bases stored as single allocations with a 16-byte header (were ~144 bytes for
  a small basis), and the branching bound of an open node inline instead of a bound-path segment of
  its own. gen-ip054 at ~490k open nodes: 493 -> ~265 bytes per open node (334 -> ~220 without
  stored bases); in drain mode at a 300 MB cap 375 -> 200 bytes. Node selection and the applied
  bounds are unchanged: --deterministic runs that stay clear of the memory limit give the same
  nodes, LP iterations and results as before.
- Under --mem-limit: the memory states are decided on the bytes the open nodes use (the hard stop on
  what is allocated); once the limit gets close, queued nodes that an improved incumbent has pruned
  are dropped; drain mode alternates deepest-first pops with best-bound pops (k : 1, k adapted so that
  the tree shrinks slowly; all deepest first close to the stop) and ends below 70% of the limit
  (was 60%), so the bound keeps moving while the tree drains. gen-ip054 at --mem-limit 300 (2
  threads, 300 s): the bound was frozen at 6794.70 from 122 s on; now 6801.41 at the end, still
  rising in drain mode.

**Faster Forrest-Tomlin updates on large sparse bases**
- When the entering column was solved densely, the simplex copied the whole m-row result into the
  update's spike and the update then tested every row. The dense solve now copies only blocks of 8
  rows that hold a nonzero and lists them, and the update reads only those (same entries in the same
  order, so the search is unchanged). Same deterministic work, 2 threads: rmatr100-p10 35.0 ->
  29.8 s, unitcal_7 (160 nodes) 99 -> 91 s, ns1208400 20.0 -> 19.5 s, h80x6320d unchanged. Profiling
  the strong-branching probes found no other per-probe overhead (state restore 0.1-1.2% of probe
  time; a probe iteration costs what an iteration of the root's late cut-round re-solves costs).

**Strong-branching control and score corrections**
- Dynamic strong-branching reliability, on by default (`MILP_SB_DYN=2`): reliability falls from 4
  towards 1 as strong-branching iterations exceed half of the node LP iterations. miplib_wide_core
  (120 s, 4 threads, seeds 1-2, x86-64 Linux machine): solved 109 -> 114 of 126, shifted geomean
  18.45 -> 17.96 s (0.975x); 30n20b8 now solved on both seeds, neos-3381206-awhea 41.7 -> 16.5 s;
  slower neos-3627168-kasai (1.39x). `MILP_SB_DYN=0` turns it off, `=3` uses only the probe
  iteration limit of twice the average node LP, `=1` both.
- Opt-in (off by default): `MILP_SD` (bits), score corrections after Shah and Dey for reliability and root strong
  branching: 1 = child gains capped at the node's gap to the incumbent, 2 = `q_min^0.3 q_max^0.7`
  instead of the product, 4 = an asymmetry factor from the direction of the last branching of the
  infeasible / integral leaves (`MILP_SD_ETA`, default 0.15).
- Opt-in: `MILP_PL` (1 exponential, 2 Pareto fit), probabilistic lookahead after Mexi et al.; once the best
  candidate has not changed for 0.6x the lookahead, probing stops when one more candidate is not
  expected to shrink the abstract tree implied by the gap and the gains seen at the node.
- Opt-in: `MILP_APS` (discount factor), ancestral pseudocosts, used at nodes where every candidate is
  reliable in both its own and its ancestral pseudocosts.
- Deterministic runs on miplib_wide_core (2 threads, node limit 10000, 90 s) against the v0.5.0 defaults:
  `MILP_SD=7` and `MILP_PL=1` change nodes by -1.4% / +1.4% and LP iterations by +6% / +0.6% on the
  instances both solve, with gaps on the unsolved ones better on about as many as worse;
  `MILP_SB_DYN=2` cuts strong branching from 35% to 28% of the LP iterations for 11% more nodes at
  the same total iterations and closes more gap on unsolved instances (7 better, 3 worse);
  `MILP_SB_DYN=1` loses on unitcal_7 and mc11 (longer probes).

**Feasibility jump before local MIP (opt-in, off by default)**
- `MILP_FJ_FIRST=1`: the helper thread runs its first feasibility-jump attempt before the local-MIP
  search instead of after it. Local MIP without a solution restarts until its whole budget is used
  (3e9 effort, 5-10 s, longer on a loaded machine), so on instances where FJ finds points at once and
  local MIP does not, the root ran without an incumbent: neos-3381206-awhea, FJ's first point at
  0.2-0.6 s in v0.4.0, none from local MIP in 30 s on a loaded machine (FJ never ran before the
  thread's deadline). With the knob (2 threads): 4 seeds, shifted geomean 3.86 -> 3.03 s (v0.4.0 3.12 s);
  on a loaded machine 30.4 -> 9.7 s (v0.4.0 15.8 s). Off: the same search as before.

**Time limits on large models**
- Phases that ran past the time limit on large MIPLIB 2017 models (bab2 was stopped by the benchmark runner
  after 144 s at a 120 s limit):
  - Packing-orbitope detection: one work budget for all symmetry components, checked per hashed column (was
    per component, between passes): bab2 several minutes -> 0.7 s.
  - Symmetry-breaking (SST) reductions: the work budget is also checked while the orbits are scored.
  - Sub-MIP presolve stops with the sub-MIP (its stop flag and time limit; the feasibility-jump repair on
    supportcase19, 1.4M columns, ended 4 s late).
  - Probing liftings that add a new entry append it without searching the column: ns1760995's probing 37.6
    -> 2.8 s (~200k new entries per pass), the same presolved model.
- Deterministic runs are unchanged.

**Tree-size-estimation restarts (opt-in, off by default)**
- `MILP_EST_RESTART=1`: the search counts its closed leaves per depth (tree weight: the sum of
  2^-depth, independent of the order in which leaves close, so `--deterministic` reads the same value)
  and estimates the tree size from it with HiGHS's windowed secant. When the estimate stays more than
  20 r times the nodes so far (r from the share of columns fixed) and the gap hardly shrinks, the
  search restarts once (presolve again with what the tree fixed), within the first half of the time
  limit. Verbosity 2 prints the tree weight. miplib_wide_core plus 5 larger instances (120 s, 4 threads,
  seeds 1-2): solved 125 -> 121 of 136, 1.021x (90% CI 0.990-1.055): its restarts are mostly on big
  trees (mas74) that a restart only sets back. Knobs `MILP_EST_FACTOR`, `MILP_EST_GATE`,
  `MILP_EST_MIN_NODES`, `MILP_EST_TRACE`.

**Cut management knobs (opt-in, off by default)**
- Diagnostics: `MILP_CUT_TRACE=1` logs the root cut rounds (candidates, rejections as too parallel or
  over the round limit, re-added and re-separated cuts, rows removed by age); verbosity 2 adds each
  search's node time split (LP solves, cut-round re-solves, separation per separator, pool scans) and
  pool churn; `MILP_POOL_TRACE=1` per progress line.
- Knobs: `MILP_CUT_AGE=k` (root cut rows stay until k + 1 re-solves in a row found them slack),
  `MILP_POOL_DEDUP=1` (pool cuts with the coefficients of a live cut map to it),
  `MILP_POOL_AGE_NEW=k` (pool rows that never bound leave after k slack node LPs), `MILP_POOL_PAR`,
  `MILP_POOL_SPARSE`, `MILP_NODE_SEP_WORK`. miplib_wide_core (120 s, 4 threads, seeds 1-2): dedup +
  aging of new rows 1.002x (90% CI 0.968-1.040), root row aging 1.016x (0.961-1.072); unset, the search
  is unchanged.

**Fixes**
- Probing could prove a feasible model infeasible: a continuous column whose implied bounds crossed
  within the tolerance was fixed at the midpoint, which could lie outside its current domain, and over a
  long row such crossed bounds added up (neos-5104907-jarama: INFEASIBLE after 3 s, optimum 935, in
  the second probing pass). The column is now fixed inside its domain.
- The benchmark verifier (`benchmarks/lib/mps.py`) took indented lines starting with `*` for comments
  and dropped irish-electricity's right-hand sides (its RHS vector is named `*RHS*`), so correct
  solutions were reported infeasible. Only a `*` in column 1 starts a comment now.

## v0.5.0 — 2026-11

MILP: root strong branching, network cut-set cuts, a local-search primal heuristic and earlier
root heuristics, conflict analysis by generalized resolution, presolve of long equations and
implied integers. Also LP and fixed-MIP dual values, interrupts, an incumbent callback and gzipped
input.

**Benchmarks**
- 43 hard MIPLIB 2017 instances (600 s, 4 threads): 15 solved, shifted geometric mean 324.9 s
  (v0.3.0: 9, 468.9 s; HiGHS 12, SCIP 6). Newly solved: nexp-150-20-8-5 (16 s), tr12-30,
  supportcase26, neos-1171737, neos-1456979, fhnw-binpack4-48; all nine solved before still are
  (csched007 326 -> 171 s, gen-ip054 541 -> 217 s; glass4 356 -> 534 s).
- 63-instance MIPLIB 2017 subset (120 s, 4 threads), seeds 0 / 1 / 2: 58 / 59 / 61 solved, shifted
  geometric mean 10.7 / 10.8 / 9.9 s (v0.4.0: 59 / 59 / 58, 10.5 / 10.9 / 11.4 s); about 4% faster
  over the three seeds.
- Generated models (1 thread, 120 s): 146 of 148 solved (v0.4.0: 147); the difference is
  binpack_hard_1, which depends on the seed in both versions.

**Root strong branching rounds**
- After the root cut loop and the root heuristics, the root strong-branches the fractional
  candidates in pseudocost order (dual simplex warm starts from the root basis, at most
  root LP iterations / 10 clamped to [100, 500] iterations per child, stop after 8 candidates
  without a better score, total budget 0.5x the root LP iterations + 5000). A candidate whose one
  child is infeasible or cut off is fixed to the other side globally; the root is then processed
  again (propagation of the fixings, LP, up to 5 cut rounds that keep the earlier cut rows,
  another pass) while passes fix and budget remains. The probes initialise the pseudocosts, the
  root branches on the best probed candidate with its children's exact bounds, and
  max over candidates of the smaller child bound is a global bound. Not in sub-MIP heuristics.
  Knobs: `MILP_ROOT_SB=0` off, `MILP_ROOT_SB_WORK`, `_OFS`, `_ITERS`, `_CANDS`, `_LOOKAHEAD`,
  `_PASSES`, `_CUTS`, `_SUBMIP=1`.
- Root gap (miplib_wide_core, mean share of the root gap closed): 66.8 -> 67.3% (SCIP 64.6%), app1-1
  78 -> 92.5% (SCIP 95%), seymour1 43 -> 48%; root time shifted geomean 3.86 -> 4.06 s.
- End to end (miplib_wide_core, 120 s, 4 threads, A and B back to back): on the 35 instances
  where root strong branching fixes columns or moves the root bound, 0.968x shifted geomean over seeds
  1, 2 (1.010 / 0.928; solved 34 -> 33 / 33 -> 34), swath3 78 -> 31 s, timtab1 14 -> 6 s, ns1830653
  94 -> 49 s; slower h80x6320d, mzzv11. Root time on the other 28: 1.045x. --deterministic
  cflp_medium_1: 7286 -> 3623 nodes.
- `CutLoopOptions::keep_rows`: a cut loop on an LP that already has cut rows only removes its own.

**Interfaces: dual values, interrupts, incumbent callback, gzipped input**
- LP duals: row duals, reduced costs and the optimal basis of LPs solved to optimality. Presolve
  only postsolves primal points, so the duals are recovered in the original space after the solve:
  a crossover (primal push only; `crossover()` now accepts a point without duals) from the
  postsolved optimal point gives a basis of the original LP, and the simplex finishes it (primal
  first). The duals of that optimal basis are complementary to the returned solution (any optimal
  primal-dual pair is), with or without presolve; the primal solution and all search statistics are
  unchanged. Cost: 0-75 cleanup iterations on the generated instances, < 0.01 s up to the medium
  ones, 0.9 s on randlp_large (24 s solve). New `solver/duals.hpp`.
- Fixed-MIP duals: the duals of the LP with the integer columns fixed at the incumbent, after the
  solve, on request (`SolverParams::duals` = 2; 1 = LPs only, 0 = none). Not for models with
  logical constraints (SOS, indicators, semi-continuous) or quadratic objectives.
- Exposed as before for QPs: CLI `--duals <file>` (also sets fixed-MIP duals for MILPs; a comment
  line marks them), C API `popso_get_duals` / `popso_get_reduced_costs` with the new parameter
  `duals` (default 1 in the C API and Python, 0 in the CLI without `--duals`), `popso_has_duals`
  returns 2 for fixed-MIP duals, new `popso_has_basis` / `popso_get_basis` (`POPSO_BASIS_BASIC`,
  `_LOWER`, `_UPPER`, `_ZERO`); Python `Result.duals`, `reduced_costs`, `fixed_mip_duals`,
  `var_basis`, `constr_basis` (`popso.BASIC`, `AT_LOWER`, `AT_UPPER`, `AT_ZERO`), `solve(duals=2)`.
- Interrupts: `SolverParams::interrupt` (an `std::atomic<bool>*`); once set, the solve stops with
  the existing status `Interrupted` (C `POPSO_STATUS_INTERRUPTED`) and the best solution so far. The
  flag is installed as a thread-local pointer (`core/interrupt.hpp`) that every solver thread
  inherits, and polled by the branch-and-bound loops (opportunistic and deterministic), the simplex
  iterations, the barrier iterations, its Cholesky factorization (per supernode and panel) and CG
  refinement, the cut loop, presolve rounds and probing, symmetry detection, feasibility jump,
  diving, the feasibility pump, sub-MIP retries and the MIQP search. Measured stop latency after
  the request: 3-40 ms (cflp_hard root and tree, randlp_large in the barrier, which took up to
  1.5 s before the factorization and CG checks). Separate models in different threads do not
  interrupt each other. C API `popso_interrupt(m)` (one lock-free atomic store: thread-safe and
  async-signal-safe; the request is cleared when `popso_solve` starts). CLI: SIGINT (`std::signal`)
  stops the solve and prints the normal summary; a second Ctrl-C ends the process. Python:
  `Model.interrupt()`; Ctrl-C during `solve()` returns the result with status `"INTERRUPTED"` (the
  native solve runs on a helper thread when called from the main thread).
- Incumbent callback: `SolverParams::on_incumbent(x, objective, bound)` / C
  `popso_set_incumbent_callback(m, fn, user_data)` with `int fn(void* user_data, double objective,
  double bound, const double* x, int num_vars)`, Python `Model.set_incumbent_callback(fn)` with
  `fn(objective, bound, x)`. The point is mapped back through presolve, symmetry reductions,
  restarts and logical reformulations to the model's columns; calls are serialized, strictly
  improving and, in deterministic mode, made at the deterministic merge points (identical sequences
  across runs); the final postsolved solution is reported too if it improves. A nonzero / true
  return (or a Python exception, re-raised by `solve()`) stops the solve like an interrupt; no calls
  follow. Not for quadratic objectives.
- Gzipped input: `.mps.gz` / `.lp.gz` (any file starting with the gzip magic bytes, whatever its
  name) are decompressed while reading by a from-scratch inflate (`io/gzip.hpp`, RFC 1951/1952:
  stored, fixed- and dynamic-Huffman blocks, all header flags, concatenated members, CRC-32 and
  length checks; streaming input and output with a 32 KB window; ~280 MB/s). MIPLIB `.mps.gz` files
  can be read directly (`benchmarks/miplib/download.py` still unpacks them).
- No change to search results: `--deterministic --threads 2` gives the same nodes / LP iterations
  (cflp_medium 7286 / 269906, gap_medium 8551 / 186860, lotsize_medium 7469 / 160109) at the same
  speed (time ratios 0.99-1.004).
- Tests: `lp_duals_kkt_random` (150 random LPs with every bound and row type and presolvable
  structure, minimization and maximization, presolve on and off: stationarity, sign conditions,
  complementarity <= 1e-7, strong duality, square basis), `lp_duals_generated_instances` (randlp,
  transport, 20 lpsanity LPs), `fixed_mip_duals_match_fixed_lp` (80 random MILPs: the fixed-MIP duals
  are optimal for the LP with the integers fixed, against its own solve), interrupt tests (preset
  flag, flag set from a thread during the search, LP / barrier / MIQP paths, scope inheritance),
  incumbent callback order / determinism / stop, gzip test vectors made with Python's zlib (levels
  0-9, Huffman-only strategy, all header flags, two members, 368 KB with window slides, byte-by-byte
  input), corrupt and truncated data, gzipped model files; C API and Python tests for all of it
  (including a simulated Ctrl-C).

**Primal heuristics: earlier incumbents**
- Local-MIP style local search (new `heuristics/lmip.hpp`, after Lin et al., "Local-MIP", 2024/25):
  tight moves, lift moves and breakthrough moves with weighted constraint scores, run first on the
  feasibility-jump thread (default effort 3e9, about 5-10 s; random restarts while no solution is
  known); in opportunistic mode the thread then restarts it from each new incumbent with the
  incumbent as cutoff, at most 5% of the elapsed time. Single-threaded root heuristic study: a first
  feasible point on 58 of 63 core instances (feasibility jump alone: 50) and 36 of 43 hard ones (28).
  Measured end to end (miplib_wide_core, 120 s, 4 threads, seeds 1-2, x86-64 Linux machine): solved 102 ->
  109 of 126, shifted geomean 20.2 -> 18.4 s. `MILP_LMIP=0` turns it off.
- Root-LP sub-MIP (`lp_rens` in `heuristics/submip`), run at the root when no incumbent exists:
  RENS with relaxed, checked fixing. Integer columns that are integral in the root LP are fixed in
  batches (largest |reduced cost| first, then the longest columns), each batch propagated, the LP
  re-solved and the restricted problem presolved (light probing); a batch the LP or presolve rejects
  is undone and split, a single rejected column stays free; with no integral column left the least
  fractional one is rounded. The sub-MIP on the remaining domain is retried with a bisection on the
  fixing rate (exhausted: fix less; limit hit: fix more), and when the two bounds meet the fixings
  between them are left free and the dive starts again from the target (0.7). Work budget in
  deterministic mode (2x the root sub-MIP iteration cap), time otherwise (max(2 s, root time),
  at most 10% of the remaining time). `MILP_NO_LP_RENS=1` turns it off.
- Root heuristics without an incumbent run the sub-MIP heuristics (RENS, feasibility pump,
  root-LP sub-MIP) and the four dives as one parallel batch (sub-MIPs first) instead of dives first;
  in opportunistic mode every point is published as soon as its task ends (blp-ic98: first
  incumbent 8 s instead of 16-21 s). `MILP_NO_ROOT_MERGE`, `MILP_NO_EARLY_PUBLISH`.
- Root LNS rounds get at least 3 s (deterministic: 4x the LP iterations) when the incumbent is more
  than 1% above the root bound (`MILP_ROOT_LNS_POOR`, seconds; 0 = off).
- Tree LNS: on slow nodes the sub-MIPs no longer wait 100 nodes between calls; 10 nodes and 50000
  search LP iterations since the last call suffice (`MILP_LNS_ITER_GAP`, 0 = off). The share of LP
  work they may use is unchanged.
- RINS caps its fixing rate: when the agreement between LP point and incumbent fixes > 60% and the
  sub-MIP is exhausted (often already in presolve: nexp-150-20-8-5 fixed 95%), the next RINS
  leaves a random subset of the agreeing columns free down to 10% below that rate (at least 50%);
  a limit hit raises the cap again (`MILP_NO_RINS_CAP`).
- Measured (60 instances of miplib_wide_core, 120 s, 4 threads, seeds 1-2, A and B back to
  back): solved 110 -> 113 of 120, shifted geomean 13.67 -> 12.17 s (ratio 0.898, 90% CI
  [0.832, 0.969]); blp-ic98 first incumbent 12.6-21 s -> 5.6-8 s. neos-3216931-puriri still has no
  incumbent in 120 s.

**Cutting planes and presolve**
- Network cut-set cuts (new `cuts/mcf.hpp`, separator "mcf", root cut loop): multi-commodity flow
  structure is detected (flow conservation rows, arc or undirected edge capacity rows with integer
  design columns, nodes shared by the commodities), and for single nodes and the node sets formed by
  contracting the non-binding arcs, the flow rows of the set plus the capacity rows of the crossing
  arcs give c-MIR / aggregated cover cuts (as SCIP's mcf separator). nexp-150-20-8-5: root bound
  92.4 -> 230.2 (optimum 231), solved in 22 s (4 threads; before: not solved in 600 s). `MILP_MCF=0`
  turns it off.
- Presolve no longer substitutes continuous columns with an objective coefficient out of an equation
  when the equation does not imply their bounds (the bounds became the row's sides and the columns'
  variable bounds were lost to the c-MIR path aggregation): tr12-30 root bound 127730 -> ~130000
  (optimum 130596), solved in 274 s (4 threads; before: not solved in 600 s).
  `MILP_PRESOLVE_ON=subst_bounded` restores the old behavior.

**Presolve: long equations, implied integers**
- Column singletons are substituted out of equations of any length (the 1000-entry limit now only
  applies to non-singleton pivots), implied free singletons also when their coefficient is small
  relative to the row (objective-defining rows z = c^T x: blp-ic98 z with 1 vs 1973), implied free
  ones first; a singleton that does not qualify no longer keeps the other columns from being tried.
  Presolved nonzeros: blp-ic98 191.7k -> 169.6k, leo1 114.2k -> 107.4k, swath1/3 31.7k -> 25.5k,
  mcsched 7.4k -> 4.5k, germanrr 168.6k -> 157.9k, sct2 20.1k -> 19.0k (HiGHS: 175k / 107k / 25.5k /
  4.6k / 158k / 18.8k).
- Dual bound strengthening keeps an integral continuous bound integral (csched008's makespan bound
  was 190.000000191: a relative 1e-9 margin on top of an exact 190).
- Implied integers (`presolve/implied_int.hpp`): continuous columns integral in every feasible point
  with integral integers (primal rule: an equation whose other columns are (implied) integer with
  integral ratios and side, iterated to a fixed point) or movable to an integral value without loss
  (dual rule: integral bounds and only such rows). Stored as `MipProblem::implied_int` (not
  `is_int`): never branched on, not required for feasibility; their bounds are rounded; used for
  objective integrality (replaces the one-level equation check of the search) and treated as integers
  by the c-MIR and Gomory separators (with integral bounds). csched008 212 of 212 continuous columns,
  csched007 292/292, uct-subprob 1445/1445, neos-3216931-puriri 159/159, trento1 668/683,
  neos-827175 10248/10248; csched007/008 and uct-subprob now have an integral objective.
  `MILP_PRESOLVE_OFF=implint` / `implint_dual` switch the detection / the dual rule off.
- Measured (A and B back to back, 4 threads, 2 seeds, 120 s): 46 of the 63 core instances
  presolve to byte-identical models (unchanged search, presolve time 9.68 vs 9.57 s); on the 17 others
  solved 15 / 13 (seed 1) and 13 / 15 (seed 2), geomean time ratio 0.957, WRONG 0 (irp 0.44, eil33-2
  0.58; swath3 / CMS750_4 lost a seed-1 solve). At 300 s: csched007 solved (288 s; was 2.9% gap),
  uct-subprob optimum 314 found and bound 295.4 -> 298, sct2 gap 0.025% -> 0.017%; csched008 bound still
  171; blp-ic98 35% fewer nodes (denser objective), bound 4454.7 -> 4450.5, incumbent 4508.4 -> 4504.9.
- Tests: long objective-row substitution, singleton blocking, integral dual bound, primal chain and
  dual (makespan) detection, a fractional start for an implied integer, 800 random implied-integer
  models (presolve on / off and an objective-scaled reference without objective integrality), LP
  duals with a substituted 1200-entry objective row.

**Conflict analysis**
- Cut-based conflict analysis by generalized resolution (new `search/genres.hpp`, after Mexi et al.,
  "Cut-based conflict analysis in mixed integer programming", IJOC 2025): an infeasible or cut-off
  node's dual proof is resolved against the reason constraints of the trail's bound changes, with
  c-MIR reduction of the intermediate constraints, and the learned linear constraints are used next
  to the existing conflict clauses. Validity tests by enumeration on random pure-integer and mixed
  models. Measured (miplib_wide_core, 120 s, 4 threads, 3 seeds, A and B back to back): seed 1
  solved 58 -> 60 of 63 at the same shifted geomean; seeds 2-3 solved 104 -> 105 of 126, shifted
  geomean 20.16 -> 19.77 s. `MILP_GENRES=0` turns it off (`2`: conflict clauses only where the
  resolution fails).

**Fixes**
- binkar10_1 (`--gap 1e-6`) sometimes ended OPTIMAL at a worse objective (6742.22-6743.43 for the
  optimum 6742.200024; reproducible with `--deterministic --threads 2 --seed 4`). Presolve
  substitutes its penalty columns (cost 1e6, column singletons of equations) out, so their bound
  becomes a row of the presolved problem; an LP point may violate that row within the feasibility
  tolerance, and postsolve, which clamps the substituted column into its bound, moved the violation
  into the objective (2.5e-8 x 1e6). The search's incumbent and proof were for the presolved value
  (6742.1993); the postsolved point was 0.026 worse. Now a postsolved point whose objective came out
  worse than the search's gets its continuous columns re-optimized (integers fixed, LP at primal
  tolerance 1e-9; 6742.200024 here), and the better point is kept (new `solver/postsolve.hpp`). The
  search is unchanged.
- A search stopped by a limit or an interrupt right after an in-tree restart reported its bound
  as -inf; it now keeps the bound from before the restart.
- Gzip input: the decompressed size is capped (default 16 GiB, `milp::set_gzip_max_output`), so a
  small high-ratio file can no longer exhaust memory; the error names the file and the limit.
- LP and fixed-MIP duals are dropped (with a log line) when the basis they come from does not
  reproduce the solution's objective (relative 1e-6; fixed-MIP: may be better than the incumbent,
  which is logged). Documented what fixed-MIP duals mean after an early stop (the fixed LP's
  optimum, not necessarily complementary to the incumbent's continuous part).
- The dual recovery (crossover and simplex cleanup) now honours interrupts and the time limit
  (at least 1 s) and gives up on the duals when stopped; the crossover used to ignore both.
- C++ API: a stop requested by the incumbent callback set the caller's `SolverParams::interrupt`
  flag, so reusing the params interrupted the next solve at once. The callback now uses an
  internal flag; the caller's flag is only read (`InterruptScope` takes two flags).
- Python: any exception (not only KeyboardInterrupt) escaping the wait in `solve()` now interrupts
  the native solve and waits for the helper thread, which used to keep running on a model that
  could be freed.
- Time limit inside the barrier: checked with the interrupt's fine-grained checks (Cholesky
  supernodes and panels, CG refinement) through a thread-local deadline (`DeadlineScope`,
  `interrupted_or_expired()`); `--time-limit 0.0001` on randlp_large stops after 0.5 s instead of
  7 s.

## v0.4.2 — 2026-11

**Fixes**
- Diving heuristics could loop forever (and grow memory until the limit) on an integer column
  whose LP value lies just outside an integer bound (x = ub + 1e-6, within the LP's tolerance):
  rounding it does not change its domain. Such columns are no longer chosen. MIPLIB
  graphdraw-domain: from the 2 GB memory limit to optimal in 293 s (4 threads, 357 MB).

## v0.4.1 — 2026-11

**Fixes**
- Windows (MSVC): C API calls that reject their arguments deep inside the library (for example
  `popso_add_q_entries` with an out-of-range variable index) aborted the process instead of
  returning `POPSO_ERROR_INVALID_ARGUMENT`. The shared library is now compiled with `/EHs`
  instead of MSVC's default `/EHsc`, which assumes that `extern "C"` functions never throw.

## v0.4.0 — 2026-11

New problem classes: convex quadratic programs (QP) and mixed-integer convex quadratic programs
(MIQP): a quadratic objective `c'x + 1/2 x'Qx` with linear constraints. Also better c-MIR cuts
and symmetry-breaking reductions for MILP.

**Benchmarks**
- 63-instance MIPLIB 2017 subset (120 s, 4 threads): 59 solved, shifted geometric mean 10.5 s
  (v0.3.0: 60, 11.4 s); over three seeds 10.5 / 10.9 / 11.4 s against 11.4 / 11.4 / 11.1 s.
  The instance count differs by two instances just above the time limit.
- Generated models, now 148 including 18 QPs and MIQPs (1 thread): 147 solved; SCIP solves 134,
  HiGHS 125 of the 130 it handles (no SOS, indicator or MIQP).
- Maros–Meszaros convex QP test set: all 97 convex problems of the 98 smallest, 32 of the 40
  larger ones (60 s).

**Quadratic objectives**
- Model: a symmetric Q stored as its lower triangle. Input: MPS `QUADOBJ` (one triangle),
  `QMATRIX` and CPLEX `QSECTION` (whole matrix); LP-format objective terms
  `[ 2 x ^ 2 + 3 x * y ] / 2`. The MPS writer writes `QUADOBJ`. Quadratic constraints (MPS
  `QCMATRIX` or a `QSECTION` of another row, brackets in LP constraints) are rejected with a
  "not supported yet" error.
- Convexity check before solving (sparse Cholesky of the unit-diagonal scaled Q plus a tiny
  shift; a negative pivot means a negative eigenvalue). A nonconvex objective (concave
  objective when maximizing) gets the new status `NONCONVEX` (C API `POPSO_STATUS_NONCONVEX`).
- Continuous QPs: the barrier method with Q in the Newton system (augmented system, or the normal
  equations when Q is diagonal), no crossover; infeasibility and unboundedness are confirmed
  by the simplex on auxiliary LPs. Once the dual regularization's error dominates the primal
  residual, the Newton systems are refined against the unregularized matrix (problems with
  large optimal duals such as YAO stalled otherwise).
- Dual values of continuous QPs: row duals and reduced costs (d = c + Qx - A'y) of the
  barrier's final point, in the model's objective sense. CLI `--duals <file>` (`row <name>
  <value>` / `col <name> <value>` lines), C API `popso_has_duals`, `popso_get_duals`,
  `popso_get_reduced_costs`, Python `Result.duals`, `Result.reduced_costs`, `Result.dual(c)`,
  `Result.reduced_cost(x)`. Not for MIQPs (LPs and MILPs: see "Interfaces" below).
- Presolve for quadratic models (QP and MIQP): fixed and empty columns (their quadratic terms
  folded into the costs and the objective constant), dual fixing of columns without quadratic
  terms, empty, free and singleton rows, redundant row sides and infeasibility from row
  activities, integer bounds from activities (MIQP). No column substitution, so the Hessian
  stays as it is; the duals are carried back (a singleton row whose bound is active takes the
  reduced cost of its column). The convexity test runs on the presolved Hessian (a cross term
  with a fixed column is linear).
- MIQPs: branch-and-bound (best bound with plunging, domain propagation of the rows),
  deterministic. The root relaxation is solved by the barrier; the nodes by a dual active-set
  method (Goldfarb–Idnani on the KKT system of the working set: dense Cholesky of the free
  columns' Hessian plus a Schur complement of the active rows, later working-set changes as
  borders of that factorization), warm-started from the parent's working set and, when it is
  still cached, its factorization; a singular Hessian gets a tiny proximal term. Node bounds
  are safe Lagrangian bounds (valid for any primal point and row multipliers by convexity),
  which also give reduced-cost tightening of integer bounds at the nodes and at the root.
  Reliability branching (strong branching by the active-set method on candidates with fewer
  than 4 pseudocost observations), plunging into the child with the smaller pseudocost
  estimate. Heuristics: rounding, a QP dive and feasibility jump on the objective linearized
  at the root solution. The barrier remains the fallback when the active-set method fails.
  Compared with barrier relaxations at every node: miqprand_large 4.9 s -> 0.3 s,
  miqprand_hard (new) 43 s -> 2.7 s, miqpport_hard (new) 13.4 s -> 0.8 s.
- C API: `popso_set_q`, `popso_add_q_entries`, `popso_num_q_nonzeros`, `popso_get_q`. Python:
  `popso.QuadExpr` (products and squares of variables and linear expressions) in
  `set_objective`, `Model.get_q()`.
- Benchmarks: new families `portfolio`, `lsq` (constrained least squares) and the MIQPs
  `miqpport` (cardinality-constrained portfolio) and `miqprand`; the solution checker
  recomputes quadratic objectives; `benchmarks/tools/qp_fuzz.py` compares random QPs and MIQPs
  with HiGHS and SCIP; `benchmarks/tools/maros_meszaros.py` fetches and converts the
  Maros–Meszaros QP test set. The 16 generated QP/MIQP instances are solved in a shifted
  geometric mean of 0.25 s (2 threads; references: HiGHS 1.5 s on 14, SCIP 5.6 s on 12).
- Maros–Meszaros (qpbenchmark's collection, 2 threads): the 98 problems up to 60 KB are all
  solved (97 convex, 1.8 s in total; VALUES is reported as nonconvex); of the 40 larger ones
  (60 s limit) 32 are solved (BOYD1/2, CONT-300, EXDATA among them); LISWET1, 7-10 and 12 end
  with a numerical error and CVXQP1_L / CVXQP3_L at the time limit.

**Symmetry**
- Symmetry-breaking reductions from a stabilizer chain (Schreier-Sims table / SST): after
  presolve, leaders whose orbit contains columns in conflict with them (a row forbids both at
  1) give x_leader >= x_follower rows and fix the conflicting followers to 0, then the search
  continues in (a subgroup of) the leader's stabilizer. Used when the fixings remove at least
  10% of the columns; the problem is presolved again and the remaining symmetry is handled as
  before (packing orbitopes, orbital branching). cod105: 1024 -> 341 columns, solved at the
  root (12 s -> 1-3 s).
- CLI `--symmetry on|off` (default on) switches all symmetry handling.

**Fixes**
- Barrier: inequality rows with fixed columns had their slack bounds shifted twice (the
  crossover repaired the point, so LP results were right, but the interior solution was not).

**Cutting planes**
- c-MIR path aggregation now eliminates a continuous column through the tight row with the largest
  LP dual (the rows that prove the LP bound) instead of the first tight row found (not for columns
  with a simple variable upper bound x <= u y, where the old order suits fixed-charge paths). On the
  63-instance MIPLIB subset the root node closes 65.2% of the root gap on average (63.8% before;
  SCIP 64.6% with node limit 1); neos17 goes from 18% to 94% of its root gap closed and solves in
  about 1 s instead of 4-6 s. End to end the subset is unchanged (60 s / 120 s A/Bs within noise).

## v0.3.1 — 2026-11

Benchmark update; the solver is the same as in v0.3.0.

- BENCHMARKS.md: new section on 43 hard MIPLIB 2017 instances (those of the 106-instance
  development set that no solver finishes within 120 s), run with a 10-minute limit. popso solves
  9, HiGHS 12, SCIP 6; set file `benchmarks/sets/miplib_hard.txt`.
- HiGHS and SCIP reference results for these instances at 600 s.
- `plot_profile.py`: `--tmin` and `--ymax` to zoom the time and count axes.

## v0.3.0 — 2026-11

New capabilities: an interior-point (barrier) LP solver and the logical modelling features real
models use — semi-continuous variables, SOS1/SOS2 and indicator constraints.

**Benchmarks**
- 63-instance MIPLIB 2017 subset (120 s, 4 threads): 60 solved, shifted geometric mean 11.4 s
  (unchanged from v0.2.1: this release adds features, not MILP speed).
- Generated models, now 130 including three families with logical constraints (1 thread):
  129 solved; SCIP solves 122 of 130, HiGHS 117 of the 122 it can read.
- LP relaxations of the MIPLIB instances: the new default `--lp-method auto` is about 27% faster
  than the dual simplex alone. BENCHMARKS.md has a new plot of popso's progress across versions.

**Barrier (interior-point) LP solver**
- New LP algorithm, written from scratch like the rest: Mehrotra's predictor-corrector
  primal-dual method with Gondzio's centrality correctors, on the presolved and scaled problem
  (geometric scaling; free variables, ranges and two-sided bounds handled directly).
- Linear algebra: an approximate minimum degree ordering (AMD) and a supernodal sparse Cholesky
  factorization, multithreaded for large dense blocks. Each solve picks the cheaper of the normal
  equations and the regularized augmented system (factored as a quasi-definite LDL^T, which suits
  LPs with far more rows than columns and is the form a future QP solver needs). Dense columns in
  the normal equations go through a Sherman–Morrison–Woodbury update; every solve is refined
  iteratively.
- Crossover to an optimal basis: dual and primal push phases, then a short simplex cleanup, so
  the answer is an exact vertex as with the simplex.
- `--lp-method auto|simplex|barrier` (C API parameter `lp_method`: 0, 1, 2; the Python bindings
  also take the names). The method is used for a pure LP and for the root LP of a MIP. `auto`
  (the default) runs the dual simplex and the barrier at the same time on large pure LPs and
  takes the first to finish (not in `--deterministic` mode, where it stays with the simplex);
  MIP root LPs stay with the simplex unless `barrier` is chosen.
- On LP relaxations of 69 MIPLIB instances (4 threads), `auto` takes 27% less time than the
  dual simplex alone in shifted geometric mean and 44% less in total; the barrier alone is about
  as fast as HiGHS' interior-point solver (shifted geometric mean 0.40 s vs 0.43 s, with
  crossover). Examples: chromaticindex512-7 20 s -> 1.6 s, dano3_3 6.2 s -> 2.7 s, qap10
  3.4 s -> 1.1 s, rmatr200-p5 5.9 s -> 2.9 s, brazil3 2.4 s -> 0.8 s.

**Model features**
- Semi-continuous and semi-integer variables (x = 0 or lb <= x <= ub), SOS1 and SOS2
  constraints, and indicator constraints (if a binary variable takes a given value, a linear
  constraint must hold).
- They are read from MPS files (`SC`/`SI` bounds, `SOS` section, `INDICATORS` section) and LP
  files (`Semi-Continuous` section, `SOS` section with `S1::`/`S2::` entries, and indicator
  constraints written `name: b = 1 -> x + y <= 5`). The MPS writer writes them, so models
  round-trip.
- C API: `POPSO_SEMICONTINUOUS` and `POPSO_SEMIINTEGER` variable types, `popso_add_sos`,
  `popso_add_indicator`, `popso_num_sos`, `popso_num_indicators`, and `popso_read_lp` for LP
  files. Python: `SEMICONTINUOUS`/`SEMIINTEGER` variable types, `Model.add_sos`,
  `Model.add_indicator(b, 1, x + y <= 5)` and `popso.read_lp`.
- How they are solved: the solver adds a binary variable for each semi-continuous variable, SOS
  member or SOS2 segment and links it with big-M rows; an indicator constraint becomes a big-M
  row. The big-M values come from the variable bounds, tightened by bound propagation over the
  model rows. Presolve, cuts, heuristics and branching then work as for any MILP, and the
  final solution is checked against the original constraints. When no finite bound exists (or
  it would exceed 1e6), the constraint is not relaxed by a big-M: the solver branches on its
  binary variable and enforces the constraint in that subtree. Presolve is skipped for such
  models.
- Testing: unit tests compare small models with optima found by enumeration, using both the
  big-M form and branching only, and `benchmarks/tools/logical_fuzz.py` compares random small
  models with SCIP. Three new generated benchmark families, `semilot` (minimum
  lot sizes), `pwlnet` (transportation with piecewise-linear concave costs, plant opening and
  single sourcing) and `indsched` (single-machine scheduling with indicator disjunctions and no
  big-M), have reference answers from SCIP (and from HiGHS for `semilot`). The benchmark
  checker verifies the new constraints.

## v0.2.1 — 2026-11

A patch release: faster root cuts on pure integer models and a robustness fix for deterministic
parallel mode under tight memory limits.

**Benchmarks**
- 63-instance MIPLIB 2017 subset (120 s, 4 threads): 60 solved, shifted geometric mean 11.3 s
  (v0.2.0: 12.4 s). See [BENCHMARKS.md](BENCHMARKS.md).

**Solver**
- Deterministic parallel mode (`--deterministic`) now drains the search tree reliably near the
  memory cap (`--mem-limit`). Before, a parallel deterministic run could stop with `MEMORY_LIMIT`
  where a 1-thread run under the same cap proved optimality. The work between two
  synchronization points is now sized so that the tree stays below the cap. Reruns stay
  identical, and runs that stay well below the cap are unchanged.

**Root cuts**
- c-MIR path aggregation continues through general integer columns on pure integer models (root cut loop,
  work-capped, not on dual degenerate root LPs): mzzv11 / mzzv42z are now solved at the root (4 threads:
  51 s -> 17 s and 15 s -> 3.3 s).
- Clique cuts are extended to maximal cliques by zero-valued literals.
- `benchmarks/tools/rootgap.py`: root gap closed per instance vs SCIP / HiGHS (node limit 1) and the known
  optimum; the solver prints a machine-readable `ROOTSTAT` line after each root.

## v0.2.0 — 2026-11

popso becomes usable as a library and on all three major platforms: a C API and Python bindings,
starting solutions, an LP-format reader, CI on Linux, macOS and Windows, a stronger presolve and a
more robust LP factorization.

**Benchmarks**
- 63-instance MIPLIB 2017 subset (120 s, 4 threads): 60 solved (v0.1.0: 59), shifted geometric
  mean 12.4 s (v0.1.0: 13.2 s); HiGHS solves 53, SCIP 50. 118 generated models (1 thread): 116
  solved. See [BENCHMARKS.md](BENCHMARKS.md).

**Solver**
- Stronger presolve:
  - clique extension: set-packing rows grow with conflicting binaries, dominated packing rows are
    dropped, and binaries that conflict with a whole equality row are fixed;
  - dual bound strengthening and GCD-based tightening of integer rows;
  - a set-cover rule for dominated columns;
  - faster probing that also tightens continuous bounds.

  On the MIPLIB subset the presolved models now close more of the root gap on average than
  SCIP's or HiGHS's presolve, and presolve takes about 30% less time than in v0.1.0.
- LP factorization: an update that would make the factorization inaccurate (a very large row
  multiplier in the Forrest–Tomlin update) now triggers a refactorization instead of carrying the
  error forward.

**Library API**
- C API in `include/popso.h`, built as the shared library `popso` (`libpopso.so`,
  `libpopso.dylib`, `popso.dll`). It can build models (variables, sparse rows, objective) or read
  MPS files, and it sets parameters, solves, and queries status, objective, bound, gap, node count,
  time and the solution. It can also write MPS and solution files. Calls return error codes, and
  `popso_last_error` gives a message. Separate models can be solved from different threads.
- Python bindings (`python/popso`), using `ctypes` with no compiled extension and no
  dependencies. They support linear expressions (`m.add_constr(x + 2 * y <= 5)`), a result object
  that can be indexed by variable or name, parameters, and starting solutions. The package can be
  installed with pip.

**Starting solutions**
- `--start <file>` reads a starting solution in the `name value` format that `--sol` writes. A
  complete feasible start becomes the first incumbent. A partial or infeasible start keeps its
  integer values and is completed by a short sub-MIP; if that fails it is ignored, and the log
  says so. Starts are also available in the C API and in Python.

**Input / output**
- CPLEX LP-format reader (`.lp` files, as written by CPLEX, HiGHS, SCIP and modelling
  tools): objective constant, ranged rows (`lo <= expr <= hi`), all bound forms
  (`free`, `-inf`/`+inf`, fixed, double-sided), General/Binary sections, comments and
  case-insensitive keywords. Quadratic terms, indicator constraints, semi-continuous
  variables and SOS are rejected with a clear "not supported yet" error and line number.
- The command line picks the reader by file extension (`.lp` vs `.mps`, case-insensitive);
  `--format mps|lp` overrides it.
- Testing: a pure-Python LP writer in the benchmark modeller and
  `benchmarks/tools/lp_crosscheck.py`, which checks that every benchmark instance read as
  LP gives exactly the same model as the MPS file, and solves LP files written by HiGHS
  and SCIP.

**Platforms**
- Continuous integration on GitHub Actions: every push and pull request is built and tested on
  Linux (gcc and clang), macOS (clang) and Windows (MSVC). Each build runs the unit tests, the
  C API and Python binding tests, and solves the tiny benchmark instances with independently
  verified answers. A separate Linux build runs the unit and C API tests under
  AddressSanitizer and UndefinedBehaviorSanitizer.
- Builds without warnings with gcc 15 and clang. The new CMake option `POPSO_WERROR` turns
  warnings into errors.
- Builds with gcc on macOS (the macOS system headers used for memory measurement did not
  compile with gcc).
- Windows/MSVC: `windows.h` no longer defines `min`/`max` macros, there are no warnings about
  standard C functions such as `getenv`, and the MSVC warning set matches the gcc/clang one.
- Benchmark runner: `benchmarks/run.py` needs only the Python standard library unless it runs
  the reference solvers. It finds `popso.exe` and `build/Release/` on Windows, works there
  without `ps`, and has a `--strict` flag that fails when an instance is not solved.

## v0.1.0 — 2026-11

First public release: a complete, multithreaded LP/MILP solver written from scratch in
C++20, using only the standard library.

**Solves**
- Linear programs (LP) and mixed-integer linear programs (MILP) with continuous, integer
  and binary variables, ranged rows and free variables.
- It proves optimality, infeasibility or unboundedness.
- On a 63-instance MIPLIB 2017 subset it solves 59 within 120 s on 4 threads, against 53
  for HiGHS and 50 for SCIP. On 118 generated models it solves 116 on 1 thread, against
  113 for HiGHS and 111 for SCIP. See [BENCHMARKS.md](BENCHMARKS.md).

**Input / output**
- MPS reader (free and fixed format, including RANGES, OBJSENSE and integer markers).
- Solution file output (`name value` per line), plus a summary of status, objective, bound,
  gap, nodes and time.
- Command-line interface with time, gap, node and memory limits, thread count, seed,
  deterministic mode and racing.

**LP engine**
- Bounded dual simplex.
- LU factorization with Forrest–Tomlin updates and hyper-sparse solves.
- Dual steepest-edge pricing, cost perturbation, and a primal simplex for cleanup.
- Warm starts from stored bases throughout the tree.

**Presolve**
- Removes singleton rows and columns, parallel rows and columns, and dominated columns.
- Tightens bounds and coefficients.
- Aggregates and substitutes variables.
- Probing with implications, and clique merging.
- Full postsolve; restarts after root fixings.

**Cutting planes**
- Gomory mixed-integer cuts and c-MIR cuts on aggregated rows (with variable-bound
  substitution).
- Lifted knapsack covers, also on aggregated rows.
- Clique, implied-bound and zero-half cuts.
- Tableau cuts at tree nodes.
- A dynamic cut pool with aging.

**Primal heuristics**
- Rounding, diving, local search.
- Feasibility pump and feasibility jump.
- Large-neighbourhood sub-MIPs (RINS, RENS, crossover) with a solution pool.

**Branch-and-bound**
- Branching: reliability branching on pseudocosts with strong branching.
- Pruning:
  - node-level domain propagation;
  - conflict analysis;
  - reduced-cost fixing.
- Symmetry: orbital fixing and packing orbitopes.
- Restarts at the root and in the tree.
- Node selection mixes best bound and best estimate.
- Recovery from numerical failures in node LPs.

**Parallelism and resources**
- Multithreaded tree search, opportunistic by default.
- A deterministic mode that gives the same result on every run with the same thread count.
- Search-tree memory stays within a configurable cap, using compact node storage; near the
  cap the solver finishes off open nodes rather than creating new ones.

**Known limitations**
- No quadratic or nonlinear models.
- No SOS or indicator constraints, and no semi-continuous variables.
- No LP-format reader, no library API, and no barrier/interior-point LP method.
