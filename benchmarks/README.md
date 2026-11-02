# Benchmarks

Test instances, reference results and the benchmark runner for the MILP
solver.  Everything here is Python (3.10+); the solver itself never depends
on it.

```
benchmarks/
  requirements.txt        highspy, pyscipopt, numpy (reference solves, generators; run.py needs only the standard library)
  generators/
    lp_sanity.py          hand-written tiny LP/MIP edge cases (MPS text)
    families.py           random instance families (model builders on lib/modeler.py)
    generate.py           driver: size table, deterministic seeds, writes MPS
  instances/              generated .mps files (committed)
  reference.py            HiGHS / SCIP reference solves -> reference/solutions.json
  reference/
    solutions.json        reference status/objective/bound/times per instance (committed)
    miplib_solu.json      MIPLIB subset: known optimal values + HiGHS times (committed)
  miplib/download.py      downloads + decompresses the MIPLIB 2017 subset (dir is gitignored)
  run.py                  benchmark runner for our solver (verifies every answer)
  results/                run.py output: <timestamp>_<git-sha>.json/.csv
  lib/                    shared helpers: MPS reader + verifier, modeler + MPS/LP writers, .sol I/O, solver wrappers
  tools/highs_cli.py      HiGHS-based stand-in implementing our CLI contract (harness tests)
  tools/lp_crosscheck.py  LP-reader check: every instance as LP must read to the same model as its MPS
                          file (--highs N / --scip N: also solve LP files written by HiGHS / SCIP)
```

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r benchmarks/requirements.txt
```

All tooling is open source: the generators build models with a tiny
pure-Python modelling layer (`lib/modeler.py`: variables, linear
expressions, constraints, MPS writer; only numpy is needed), and the
reference solves use HiGHS and SCIP.

**README tables.** `tools/readme_table.py <run.json>` prints the summary and
per-instance comparison tables (popso vs HiGHS / SCIP) used in the
top-level README.

## Memory and parallelism (shared machine!)

The development machine has 24 GB shared with the user's apps and other
agents.  Rules for all tooling here:

* **At most 3 solver processes at a time** in total.  `run.py` runs
  instances sequentially by default (`--jobs 1`); `reference.py` defaults to
  `--jobs 2`.  Never start several runners concurrently.
* **Every solver process is memory-capped** (`--mem-limit MB`, default 2048):
  * `run.py` passes `--mem-limit <MB>` to our solver (which should stop
    with `# Status = MEMORY_LIMIT`) and additionally kills it if its resident
    set size exceeds 125% of the limit (class `memlimit`, counted unsolved);
  * `reference.py` sets RLIMIT_AS/RLIMIT_DATA in the worker (enforced on
    Linux, mostly ignored by macOS), SCIP `limits/memory`,
    and an RSS watchdog in the parent kills the worker at 110% (status
    `memlimit`; `MemoryError` is reported the same way).
  * Use `--no-solver-mem-flag` if a solver binary does not accept
    `--mem-limit` yet (watchdog only).
* The RSS watchdog lives in `lib/proc.py` (`ps`-based, portable to macOS).

## Instances

Names are `<family>_<size>_<seed>.mps` (free MPS, uncompressed), sizes
`tiny < small < medium < large < hard`.  The hand-written edge cases are
`lpsanity_<case>_0.mps`.  Regenerate with

```sh
.venv/bin/python benchmarks/generators/generate.py              # all
.venv/bin/python benchmarks/generators/generate.py --only tsp   # one family
```

Generation is deterministic (seed derived from family/size/seed via SHA-256;
numpy `default_rng`).  Changing a family's code changes its instances, so
re-run `reference.py` for it afterwards.

118 instances: 84 generated (20 families) + 34 hand-written `lpsanity_*`.
Cells: `vars x cons` and the HiGHS 1.15 reference time (1 thread, gap 1e-9,
2-6 solves running concurrently, Apple M-series); non-optimal status noted.

| family | tiny | small | medium | large | hard |
|---|---|---|---|---|---|
| knapsack (0-1, uncorr/weak/strong) | 10x1 0.0s | 50x1 0.0s | 200x1 0.0s | 1000x1 0.1s |  |
| mkp (Chu-Beasley) | 10x2 0.0s | 30x5 0.1s | 60x5 2.3s | 100x5 4.5s | 100x10 43.9s |
| setcover | 20x10 0.0s | 100x50 0.0s | 500x200 1.1s | 1000x400 2.4s | 1000x500 2.7s |
| setpack | 20x10 0.0s | 100x50 0.0s | 200x100 1.1s | 400x200 9.9s | 500x250 161s |
| binpack (assignment form.) | 21x11 0.0s | 52x19 0.0s | 260x44 0.0s | 697x73 0.1s | 874x82 26.8s |
| gap (type D) | 24x11 0.0s | 100x25 0.6s | 200x45 12.1s | 600x70 12.6s | 700x80 >600s |
| cflp (single source) | 36x44 0.0s | 310x340 0.8s | 1220x1280 6.0s | 4040x4140 25.7s | 6050x6170 47.7s |
| netdesign (multicommodity FC) | 56x71 0.0s | 234x298 0.0s | 784x964 0.4s | 4100x4900 5.4s | 6630x7880 30.9s |
| tsp (MTZ) | 24x22 0.0s | 99x92 0.4s | 399x382 6.3s | 1224x1192 68.9s | 2499x2452 106s |
| cvrp (MTZ loads) | 35x31 0.0s | 63x57 0.1s | 99x91 1.2s | 143x133 9.0s |  |
| lotsize (CLSP, setup times) | 28x20 0.0s | 104x72 0.3s | 300x204 6.6s | 465x315 11.9s | 740x500 92.7s |
| jobshop (big-M disjunctive) | 19x27 0.0s | 41x64 0.0s | 127x216 0.3s | 289x512 25.4s | 276x500 17.2s |
| timeidx (P\|r_j\|sum w_jC_j) | 69x25 0.0s | 732x101 0.0s | 2167x148 1.1s | 7917x291 11.9s | 16416x405 18.0s |
| randmilp (B/I/C mixed) | 10x6 0.0s | 40x25 0.0s | 100x60 0.2s | 150x100 25.3s | 200x120 137s |
| unbdmilp (integer ray) |  | 30x20 unbounded | 150x100 unbounded |  |  |
| pigeon (pairwise, infeasible) | 20x45 | 56x204 | 110x561 | 210x1485 (large) |  |
| subsetsum (infeasible) |  | 12x1 0.0s | 22x4 3.6s |  |  |
| transinf (LP-infeasible) |  | 40x13 | 1500x80 |  |  |
| transport (LP) |  | 200x30 0.0s | 5000x150 0.0s | 45000x450 0.4s |  |
| randlp (LP) |  | 50x30 0.0s | 1000x600 0.0s | 8000x5000 39.9s |  |

HiGHS time distribution over all 118: 72 <= 0.1 s, 11 in 0.1-1 s, 15 in
1-10 s, 14 in 10-60 s, 3 in 60-120 s, 2 in 120-600 s, 1 time-out
(gap_hard_1).  Statuses: 100 optimal, 13 infeasible, 5 unbounded.  The
pigeonhole instances are trivial for solvers with clique reasoning but
exponential for plain B&B.

### MPS conventions (what our reader must implement)

Verified against HiGHS 1.15 and SCIP 10 with the `lpsanity_*`
cases (see `generators/lp_sanity.py` for the hand-derived answers):

* Default bounds `[0, +inf)`.  **MARKER integers with no BOUNDS entry at all
  are binary `[0, 1]`**; any BOUNDS entry for such a column (even just `LO`
  or `MI`) restores `ub = +inf`.  (Both solvers agree.)
* `MI` sets `lb = -inf` only; `PL` sets `ub = +inf`; `BV` = integer in [0,1];
  `LI`/`UI` make the column integer; `FR`, `FX` as usual.
* **Negative `UP` on a column without explicit lower bound sets `lb = -inf`**
  (CPLEX convention, which we follow).  HiGHS 1.15 and SCIP keep
  `lb = 0` (-> infeasible, SCIP even fails on the integer case), so for
  `lpsanity_negative_ub_0` and `lpsanity_mip_negative_ub_0` the reference
  is the hand-derived answer only (`HAND_DERIVED_ONLY`).
* RANGES: `L`: `[rhs-|R|, rhs]`, `G`: `[rhs, rhs+|R|]`, `E`: `R>0 ->
  [rhs, rhs+R]`, `R<0 -> [rhs+R, rhs]`.
* RHS on the objective row = **negated** objective constant.
* Only the first `N` row is the objective; further `N` rows are ignored
  (HiGHS 1.15 wrongly applies an RHS given on an extra N row as objective
  offset; our test case avoids that).
* `OBJSENSE` may be a section (`OBJSENSE` / `    MAX`) or inline
  (`OBJSENSE MAX`, as `lib/modeler.py` writes it); `MAXIMIZE`/`MINIMIZE` also occur.
* `lib/modeler.py` writes integer columns with infinite upper bound as
  `LI <col> 0` (to defeat the implicit-binary rule).

## Reference results

```sh
.venv/bin/python benchmarks/reference.py --jobs 3 --time-limit 600
.venv/bin/python benchmarks/reference.py 'benchmarks/instances/tsp_*' --update
```

Each (instance, solver) pair runs in a subprocess (single thread, relative
and absolute MIP gap 1e-9, hard kill at 1.5x the limit).  Every returned
solution is re-verified with `lib/mps.py` (our own reader), which therefore
also cross-checks the reader against two independent MPS parsers.
`reference/solutions.json` contains per instance: `num_vars`, `num_cons`,
`num_nonzeros`, `num_integers`, `num_binaries`, `sense`, the consensus
`status` / `objective` / `bound`, `agree` + `notes` (disagreements,
verification failures) and per solver `status`, `objective`, `bound`, `time`,
`nodes`, `verify`.

Both solvers agree on all 118 instances (after the fixes below) and
every returned solution verifies.  Things found while building the set:

* `negative_ub` semantics differ (see above); HiGHS also applies an RHS on
  a second N row as an objective offset.
* With an integrality tolerance of 1e-5 a solver "solved" the infeasible
  subset-sum instance with x = 7.6e-6 (coefficients now scaled down);
  solutions failing our verification get no vote.
* A random MILP with 4-decimal right-hand sides let a solver find a
  "better" optimum by exploiting the 1e-6 feasibility tolerance (others
  disagreed by 0.2%); right-hand sides are now exact 2-decimal values.
* `gap_hard_1`: HiGHS and SCIP time out at 600 s; the reference is the best
  incumbent and bound (status `timelimit`).
* `reference.py --recompute` re-derives the consensus of the stored entries
  from their per-solver results without solving.
* `--update` re-solves only the given instances and keeps the rest; `meta`
  then describes the last run.

## MIPLIB 2017 subset

```sh
.venv/bin/python benchmarks/miplib/download.py          # SUBSET -> benchmarks/miplib/*.mps
.venv/bin/python benchmarks/run.py --solver build/popso --miplib -t 300
```

48 instances of the MIPLIB 2017 benchmark set (all "easy", <= 150k
nonzeros) that HiGHS 1.15 solves single-threaded within ~2 min (0.1 s to
130 s; 47 optimal + 1 infeasible, neos859080).  They were selected by
running all 128 candidates (`download.py --candidates`, ~260 MB) with HiGHS
(120-300 s limit); 80 timed out.  `reference/miplib_solu.json` holds, per
instance, the official status/objective from `miplib2017-v37.solu`, size
statistics, and the HiGHS status/time from the selection run.  `run.py
--miplib` uses these as reference (only the subset in that file is run).
Note: on `binkar10_1` HiGHS returns 6742.20 although the optimum is
6741.38, i.e. HiGHS itself would be flagged WRONG there.  (SCIP and our
solver also return 6742.20; the MIPLIB solution file for 6741.38 has
integrality/bound violations of 8.2e-7, i.e. it exploits the 1e-6
tolerances; rounding it gives 6742.2001.)

### Wide set (`sets/miplib_wide.txt`, 106 instances)

```sh
.venv/bin/python benchmarks/miplib/download.py --wide        # 106 .mps, ~340 MB
.venv/bin/python benchmarks/run.py --solver build/popso --set miplib_wide -t 120 --threads 4
.venv/bin/python benchmarks/ab.py --a A/milp --b B/milp --set miplib_wide_core -t 120 ...
```

The 48 above plus 58 more instances of the benchmark set (`WIDE_EXTRA` in
`download.py`): all "easy", <= 200k nonzeros; every candidate that HiGHS left
with a small gap (< ~10%) at 120 s single-threaded, a diverse mix of harder
candidates and 8 instances with 150k-200k nonzeros.  HiGHS (4 threads) and
SCIP (single-threaded; it has no parallel B&B) reference runs with 120 s are in
`reference/solutions.json` (entries carry `ref_runs` with limits/threads;
`meta.updates` lists the partial runs).  At 120 s only 62 of the 106 are
solved by any of HiGHS / SCIP / popso; `sets/miplib_wide_core.txt` (63) is
those plus binkar10_1, the useful subset for 120 s A/B runs.

`run.py --set NAME` (repeatable; a file or a name in `sets/`) runs a list
file, `ab.py --set` forwards it.  `--miplib` alone still runs the 48-subset
(`sets.subset` in `miplib_solu.json`).  For MIPLIB instances `run.py` takes
the reference status/objective from the official .solu value when it is
proven (also when a reference solver timed out), and reference times from
`solutions.json` (4 threads) in preference to the 1-thread selection-run
times in `miplib_solu.json`; a reference solver whose optimum contradicts
the .solu value counts as unsolved.

## Running our solver

CLI contract: `milp <model.mps> --sol <out.sol> --time-limit <s> --threads <n> --mem-limit <MB> --gap <rel>`,
writing a `.sol` file in the common "name value" format (`# Status = OPTIMAL|INFEASIBLE|UNBOUNDED|
TIME_LIMIT|MEMORY_LIMIT|...`, `# Objective value = v`, then `name value` lines; only the
header for infeasible/unbounded).

```sh
.venv/bin/python benchmarks/run.py --solver build/popso                      # all generated, 60 s
.venv/bin/python benchmarks/run.py --solver build/popso --max-size small -t 10   # quick smoke test
.venv/bin/python benchmarks/run.py --solver build/popso --families tsp,cvrp -t 120
.venv/bin/python benchmarks/run.py --solver build/popso 'benchmarks/instances/lpsanity_*'
.venv/bin/python benchmarks/run.py --solver ".venv/bin/python benchmarks/tools/highs_cli.py"  # harness self-test
```

For each instance the runner parses the `.sol` file (stdout is ignored),
then **verifies** it independently: bounds, integrality (1e-6), every row
(1e-6, relative to max(1,|rhs|)) and recomputes the objective.  It then
compares with the reference:

| class   | meaning |
|---------|---------|
| `ok`    | proven status matches the reference; for optimal, objective within rel. 1e-6 (`--opt-tol`) |
| `noref` | solved and verified, no proven reference to compare to |
| `unsolved` | time limit / unknown, nothing wrong detected |
| `memlimit` | `# Status = MEMORY_LIMIT`, or killed by the RSS watchdog |
| `error` | crash, non-zero exit, missing or malformed `.sol`, killed after limit + grace |
| `WRONG` | infeasible solution, objective mismatch, claimed optimal but worse than reference, verified solution better than the reference optimum, wrong infeasible/unbounded claim |

`--gap` (default 1e-6) is passed to the solver as `--gap <rel>` and recorded
in the result JSON; the correctness check always uses `--opt-tol` (1e-6), so
running with a looser gap (e.g. the solver's default 1e-4) will flag
suboptimal "optimal" answers as WRONG.  Note: the reference HiGHS/SCIP
solves did *not* use their default gaps (1e-4 / 0) but relative and
absolute gap 1e-9, so the reference objectives are optimal to ~1e-9; the
reference times are therefore slightly pessimistic versus default settings.

Any `WRONG` is printed with a banner and makes the exit code 1.  The `.sol`
and log of failing runs are kept in `results/tmp/`.  Note the 1e-6 optimality
tolerance: the solver must close the gap to ~1e-6 relative (or the absolute
equivalent) before claiming `OPTIMAL`.

Summary: #solved, #wrong, #unsolved, #errors and the shifted geometric mean
of the time (shift 1 s, unsolved/wrong counted at the time limit), next to
the same metric for the HiGHS and SCIP reference times on the same instances
(capped at the run's time limit).  Results are written to
`results/<timestamp>_<git-sha>.json` and `.csv` (the sha gets `-dirty` when
`src/`, `apps/` or `CMakeLists.txt` have uncommitted changes).

`tools/highs_cli.py` implements the CLI on top of HiGHS; `FAKE_BUG=perturb|
objective|infeasible|suboptimal` injects defects to check that the runner
catches them.  It legitimately fails the two negative-UP sanity cases
(HiGHS reads them differently).

## A/B experiments (`ab.py`)

Compare two solver builds over several seeds (runs are interleaved A/B per seed):

```sh
.venv/bin/python benchmarks/ab.py --a /path/to/baseline/milp --b build/popso --seeds 1,2,3 -t 60 $(cat benchmarks/sets/dev.txt)
```

Reports wrong answers (exit code 1 if B has any), solved counts, shifted geometric means, the B/A ratio with a
90% bootstrap CI over instances, per-family ratios and the largest per-instance changes. `benchmarks/sets/dev.txt`
is the standard development set (instances taking ~0.5–60 s). The solver's `--deterministic` flag gives
reproducible parallel runs (identical trees for identical seed/threads) for debugging; speed comparisons use the
default opportunistic mode over several seeds.
