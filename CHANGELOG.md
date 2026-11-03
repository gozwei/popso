# Changelog

## Unreleased

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
