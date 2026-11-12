#!/usr/bin/env python3
"""Benchmark runner for our MILP solver.

Runs  <solver> <model.mps> --sol <out.sol> --time-limit <s> --threads <n>
on each instance, parses the .sol file, independently VERIFIES the solution
(bounds, integrality, constraints, objective recomputation; own Python MPS
reader) and compares status/objective with the reference results.

    .venv/bin/python benchmarks/run.py --solver build/popso
    .venv/bin/python benchmarks/run.py --solver build/popso --sizes tiny,small -t 10
    .venv/bin/python benchmarks/run.py --solver build/popso 'benchmarks/instances/tsp_*'
    .venv/bin/python benchmarks/run.py --solver build/popso --miplib -t 300
    .venv/bin/python benchmarks/run.py --solver build/popso --set miplib_wide -t 120 --threads 4

Wrong answers (infeasible solution, wrong status, objective mismatch vs the
reference) are reported loudly and make the exit code 1.

Writes benchmarks/results/<timestamp>_<git-sha>.json and .csv.  While running,
every finished instance is appended to <timestamp>_<git-sha>.jsonl (removed once
the .json is written), so an interrupted run loses at most the instance in
progress and can be continued with  --resume <that .jsonl>  (same command line).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
sys.path.insert(0, str(BENCH))

from lib import mps  # noqa: E402
from lib.proc import kill_all, run_capped  # noqa: E402
from lib import primal  # noqa: E402
import slots  # noqa: E402
from lib.solfile import (INF_OR_UNBD, INFEASIBLE, OPTIMAL, TIMELIMIT,  # noqa: E402
                         UNBOUNDED, read_sol)

SIZE_ORDER = ["tiny", "small", "medium", "large", "hard", "xhard"]

# result classes
OK = "ok"                  # correct and solved (proven status matches reference)
UNSOLVED = "unsolved"      # time limit / no proof, but nothing wrong detected
WRONG = "WRONG"            # wrong answer -> loud failure
ERROR = "error"            # crash, missing/garbled .sol file, killed after time limit
NOREF = "noref"            # solved, solution verified, but no reference to compare with
MEMLIMIT = "memlimit"      # solver stopped at (MEMORY_LIMIT) or was killed above the memory cap


def sgm(times, shift=1.0):
    if not times:
        return None
    return math.exp(sum(math.log(t + shift) for t in times) / len(times)) - shift


def git_sha():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip() or "nogit"
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "apps", "CMakeLists.txt"],
                               cwd=REPO).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except OSError:
        return "nogit"


def inst_name(path: Path):
    n = path.name
    for ext in (".mps.gz", ".mps"):
        if n.endswith(ext):
            return n[: -len(ext)]
    return path.stem


def expand(patterns):
    files = []
    for p in patterns:
        pp = Path(p)
        if pp.is_dir():
            # x.mps and x.mps.gz in one directory: the instance runs once (from x.mps)
            plain = sorted(pp.glob("*.mps"))
            names = {inst_name(f) for f in plain}
            files += plain + [f for f in sorted(pp.glob("*.mps.gz")) if inst_name(f) not in names]
        else:
            files += [Path(f) for f in sorted(glob.glob(p))]
    seen, out = set(), []
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            out.append(f)
    return out


def repo_rel(path) -> str:
    """Path relative to the repository root (never an absolute local path)."""
    p = Path(os.path.abspath(path))
    for cand in (p, p.resolve()):
        if cand.is_relative_to(REPO):
            return cand.relative_to(REPO).as_posix()
    return os.path.relpath(p, REPO)


def _shown(path) -> str:
    """Path for messages: repository-relative inside the repo, absolute outside."""
    r = repo_rel(path)
    return str(Path(path).resolve()) if r.startswith("..") else r


def _strip_repo(s: str) -> str:
    """Drop the local repository prefix from paths inside a command string."""
    return s.replace(str(REPO) + os.sep, "")


def read_set(spec):
    """Instance list file -> [Path]; `spec` is a file or a name in benchmarks/sets/."""
    p = Path(spec)
    if not p.exists():
        p = BENCH / "sets" / (spec if spec.endswith(".txt") else spec + ".txt")
    out = []
    for ln in p.read_text().splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            q = Path(ln)
            out.append(q if q.is_absolute() else REPO / q)
    return out


def _compatible(solver_status, consensus_status):
    if solver_status in (consensus_status, TIMELIMIT):
        return True
    return solver_status == INF_OR_UNBD and consensus_status in (INFEASIBLE, UNBOUNDED)


def load_references(ref_path, solu_path):
    refs = {}
    d_ref = {"instances": {}}
    if ref_path and Path(ref_path).exists():
        d = d_ref = json.loads(Path(ref_path).read_text())
        for n, e in d["instances"].items():
            refs[n] = {
                "status": e["status"], "objective": e["objective"], "bound": e.get("bound"),
                "opt_tol": e.get("opt_tol"),
                "source": "reference",
                # a reference solver whose status contradicts the consensus counts as unsolved; the MIQP
                # references from a Python branch-and-bound on HiGHS QP relaxations are not HiGHS's times
                "times": {s: (r.get("time"), r.get("status") if _compatible(r.get("status"), e["status"])
                              else "wrong") for s, r in e.get("solvers", {}).items()
                          if r.get("raw_status") != "python-bnb"},
            }
    if solu_path and Path(solu_path).exists():
        d = json.loads(Path(solu_path).read_text())
        for n, e in d["instances"].items():
            if n in refs:
                # MIPLIB instance with reference solves in solutions.json: the official
                # .solu status/value wins (over time limits, and over a reference solver
                # whose "optimum" contradicts it, e.g. HiGHS on binkar10_1, which then
                # counts as unsolved).
                if e["status"] in (OPTIMAL, INFEASIBLE):
                    z = e.get("objective")
                    for s, r in d_ref["instances"][n].get("solvers", {}).items():
                        bad = (r.get("status") in (OPTIMAL, INFEASIBLE) and r.get("status") != e["status"]) or (
                            e["status"] == OPTIMAL and r.get("status") == OPTIMAL and
                            abs(r["objective"] - z) > 1e-6 * max(1.0, abs(z)))
                        if bad and s in refs[n]["times"]:
                            refs[n]["times"][s] = (refs[n]["times"][s][0], "wrong")
                        elif not bad and r.get("status") == e["status"] and s in refs[n]["times"]:
                            # agrees with the official value (e.g. the only solver that proved
                            # it, while the solutions.json consensus is a time limit)
                            refs[n]["times"][s] = (refs[n]["times"][s][0], r["status"])
                    refs[n].update({"status": e["status"], "objective": z, "bound": z, "source": "reference+miplib"})
                elif e.get("objective") is not None and refs[n]["status"] != OPTIMAL:
                    # MIPLIB best known value (not proven): keep the reference status/bound,
                    # but use the better of the reference incumbent and the best known value
                    z, rz = e["objective"], refs[n]["objective"]
                    sense = -1 if d_ref["instances"][n].get("sense") == "max" else 1
                    if rz is None or sense * (rz - z) > 0:
                        refs[n].update({"objective": z, "source": "reference+miplib"})
            else:
                refs[n] = {"status": e["status"], "objective": e.get("objective"),
                           "bound": None, "source": "miplib", "times": {}}
            if e.get("highs") and "highs" not in refs[n]["times"]:
                # selection-run time (1 thread); solutions.json times take precedence
                refs[n]["times"]["highs"] = (e["highs"].get("time"), e["highs"].get("status"))
    return refs


REF_SOLVERS = ("highs", "scip")


def better(sense, a, b, tol):
    """True if objective a is strictly better than b beyond relative tol."""
    return sense * (b - a) > tol * max(1.0, abs(b))


def classify(sense, v, sol, ref, args):
    """Return (class, message, details).  `v` is the mps.Verification of sol.values
    (None when the .sol has no values), `sense` the model's objective sense."""
    details = {}
    st = sol.status
    if sol.errors:
        return ERROR, "malformed .sol: " + "; ".join(sol.errors[:3]), details
    has_values = bool(sol.values)
    if has_values:
        details["verify"] = v.to_dict()
        details["objective_recomputed"] = v.objective
        if not v.feasible:
            return WRONG, f"solution infeasible: {v.worst}", details
        if v.missing:
            details["missing_vars"] = v.missing
        if sol.objective is not None and abs(sol.objective - v.objective) > args.opt_tol * max(1.0, abs(v.objective)):
            return WRONG, (f"reported objective {sol.objective} != recomputed {v.objective}"), details
        obj = v.objective
    else:
        obj = None
    if st == OPTIMAL and not has_values:
        return WRONG, "status OPTIMAL but no solution values", details

    if ref is None:
        if st in (OPTIMAL,):
            return NOREF, "no reference (solution verified)", details
        return UNSOLVED if st == TIMELIMIT else NOREF, f"no reference, status {st}", details

    rst, rz = ref["status"], ref["objective"]
    # A reference may widen the tolerance where the optimum depends on the feasibility tolerance itself (its notes
    # say why: e.g. cones of radius 7e-4, where an absolute 1e-6 is worth 1.7% of the objective).
    tol = max(args.opt_tol, ref.get("opt_tol") or 0.0)
    if rst == OPTIMAL and rz is not None:
        if obj is not None:
            details["gap_to_ref"] = abs(obj - rz) / max(1.0, abs(rz))
            if better(sense, obj, rz, tol):
                return WRONG, (f"verified solution {obj} BETTER than reference optimum {rz} "
                               f"(reference wrong?)"), details
        if st == OPTIMAL:
            if better(sense, rz, obj, tol):
                return WRONG, f"claimed optimal {obj} but reference optimum is {rz}", details
            return OK, "", details
        if st in (INFEASIBLE, UNBOUNDED, INF_OR_UNBD):
            return WRONG, f"claimed {st} but reference optimum is {rz}", details
        return UNSOLVED, f"status {sol.raw_status or st}", details
    if rst == INFEASIBLE:
        if obj is not None:
            return WRONG, (f"verified feasible solution ({obj}) but the reference says infeasible "
                           f"(reference wrong, or the solution exploits the tolerances?)"), details
        if st in (INFEASIBLE, INF_OR_UNBD):
            return OK, "", details
        if st in (OPTIMAL, UNBOUNDED):
            return WRONG, f"claimed {st} but reference says infeasible", details
        return UNSOLVED, f"status {sol.raw_status or st}", details
    if rst == UNBOUNDED:
        if st in (UNBOUNDED, INF_OR_UNBD):
            return OK, "", details
        if st in (OPTIMAL, INFEASIBLE):
            return WRONG, f"claimed {st} but reference says unbounded", details
        return UNSOLVED, f"status {sol.raw_status or st}", details
    # reference has no proven status (time limit / MIPLIB best known only): a verified
    # objective better than the best known one is fine (only a proven dual bound can refute it)
    rb = ref.get("bound")
    if obj is not None and rb is not None and better(sense, obj, rb, tol):
        return WRONG, f"verified objective {obj} beyond reference dual bound {rb}", details
    if obj is not None and rz is not None and better(sense, obj, rz, tol):
        details["improves_best_known"] = True
    if st == OPTIMAL:
        if rz is not None and better(sense, rz, obj, tol):
            return WRONG, f"claimed optimal {obj} but a better solution {rz} is known", details
        return NOREF, "optimal claimed; reference optimum unknown (consistent with known bounds)", details
    if st in (INFEASIBLE, UNBOUNDED) and rz is not None:
        return WRONG, f"claimed {st} but a feasible solution ({rz}) is known", details
    return UNSOLVED, f"status {sol.raw_status or st}", details


def verify_worker(argv):
    """`run.py --verify-worker MPS SOL FEASTOL INTTOL CHECK`: read the model (and verify
    the .sol when CHECK is 1) in a separate, memory-capped process; prints one JSON line."""
    path, sol_path, feastol, inttol, check = argv
    try:
        model = mps.read_mps(path)
    except mps.MPSError as e:
        print(json.dumps({"error": f"cannot read MPS: {e}"}))
        return 0
    v = None
    if check == "1":
        v = mps.verify(model, read_sol(sol_path).values, float(feastol), float(inttol)).to_dict()
    print(json.dumps({"sense": model.sense, "verify": v}))
    return 0


def verify_capped(path, sol_path, check, args, tmpdir):
    """Run verify_worker under the RSS watchdog (--verify-mem-limit): the pure-Python
    model of a large MIPLIB instance takes several GB.  Returns (sense, Verification
    or None, error message or None)."""
    out_path = tmpdir / f"{inst_name(path)}.verify"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--verify-worker", str(path), str(sol_path),
           repr(args.feas_tol), repr(args.int_tol), "1" if check else "0"]
    with open(out_path, "w+") as out:
        rc, reason, wall, peak = run_capped(cmd, args.verify_timeout, args.verify_mem_limit or None,
                                            stdout=out, stderr=subprocess.STDOUT)
        out.seek(0)
        text = out.read()
    out_path.unlink()
    if reason == "memlimit":
        return None, None, f"killed above {args.verify_mem_limit} MB (raise --verify-mem-limit)"
    if reason == "timeout":
        return None, None, f"killed after {wall:.0f} s (raise --verify-timeout)"
    lines = [ln for ln in text.splitlines() if ln.startswith("{")]
    if rc != 0 or not lines:
        return None, None, f"crashed (exit {rc}): {text[-300:]}"
    d = json.loads(lines[-1])
    if "error" in d:
        return None, None, d["error"]
    return d["sense"], (mps.Verification(**d["verify"]) if d["verify"] else None), None


def run_one(path, args, solver_cmd, tmpdir, refs):
    name = inst_name(path)
    tmpdir.mkdir(parents=True, exist_ok=True)  # survive external cleanup of results/tmp
    sol_path = tmpdir / f"{name}.sol"
    if sol_path.exists():
        sol_path.unlink()
    cmd = solver_cmd + [str(path), "--sol", str(sol_path), "--time-limit", f"{args.time_limit:g}",
                        "--threads", str(args.threads)]
    if args.gap is not None:
        cmd += ["--gap", f"{args.gap:g}"]
    if args.mem_limit and not args.no_solver_mem_flag:
        cmd += ["--mem-limit", str(args.mem_limit)]
    start_path = Path(args.starts) / f"{name}.sol" if args.starts else None
    if start_path is not None and start_path.exists():
        cmd += ["--start", str(start_path)]
    else:
        start_path = None
    prof_path = tmpdir / f"{name}.profile.json"
    if args.trace:
        if prof_path.exists():
            prof_path.unlink()
        cmd += ["--profile", str(prof_path)]
    log_path = tmpdir / f"{name}.log"
    grace = max(10.0, 0.2 * args.time_limit)
    # POPSO_SLOTS: wait for the run's threads and memory in the shared pool first (slots.py), outside the timing.
    with slots.slot(args.threads, args.mem_limit or 0), open(log_path, "wb") as log:
        # RSS watchdog: kill at 125% of the limit given to the solver
        rc, reason, wall, peak = run_capped(cmd, args.time_limit + grace,
                                            args.mem_limit * 1.25 if args.mem_limit else None,
                                            stdout=log, stderr=subprocess.STDOUT)
    rec = {"instance": name, "file": repo_rel(path), "time": wall, "returncode": rc,
           "peak_rss_mb": round(peak, 1), "status": None, "objective": None, "class": None,
           "message": ""}
    if args.starts:
        rec["start"] = start_path is not None
    ref = refs.get(name)
    if ref:
        rec["ref_status"], rec["ref_objective"] = ref["status"], ref["objective"]
        ht = ref["times"].get("highs")
        if ht:
            rec["ref_highs_time"], rec["ref_highs_status"] = ht
        stt = ref["times"].get("scip")
        if stt:
            rec["ref_scip_time"], rec["ref_scip_status"] = stt
    if args.trace and prof_path.exists():
        # popso --profile: the incumbents over time, and the primal integral against the reference objective
        try:
            pts = primal.trace_points(json.loads(prof_path.read_text())["run"]["incumbents"])
            rec["incumbents"] = pts
            pi = primal.integral(pts, rec.get("ref_objective"), args.time_limit)
            if pi is not None:
                rec["primal_integral"] = pi
        except (OSError, ValueError, KeyError):
            pass
    if reason == "timeout":
        rec.update({"class": ERROR, "message": f"killed after {wall:.1f}s (time limit not respected)"})
        return rec
    if reason == "memlimit":
        rec.update({"class": MEMLIMIT, "status": MEMLIMIT,
                    "message": f"killed: RSS above {args.mem_limit * 1.25:.0f} MB (memory limit not respected)"})
        return rec
    if not sol_path.exists():
        rec.update({"class": ERROR, "message": f"no .sol file written (exit code {rc})"})
        return rec
    sol = read_sol(sol_path)
    rec["status"], rec["objective"] = sol.status, sol.objective
    sense, v, verr = verify_capped(path, sol_path, bool(sol.values) and not sol.errors, args, tmpdir)
    if verr:
        rec.update({"class": ERROR, "message": f"verifier: {verr}"})
        return rec
    cls, msg, details = classify(sense, v, sol, ref, args)
    if sol.status == MEMLIMIT and cls == UNSOLVED:
        cls = MEMLIMIT
    if rc != 0 and cls != WRONG:
        cls, msg = ERROR, f"exit code {rc}; " + msg
    rec.update({"class": cls, "message": msg, **{k: v for k, v in details.items() if k != "verify"}})
    if "verify" in details:
        v = details["verify"]
        rec["max_viol"] = max(v["max_bound_viol"], v["max_row_viol"], v["max_int_viol"], v.get("max_logic_viol", 0.0))
    try:  # search statistics from the solver log (popso's summary lines)
        text = log_path.read_text(errors="replace")
        m = re.search(r"^Nodes: (\d+)$", text, re.M)
        if m:
            rec["nodes"] = int(m.group(1))
        m = re.search(r"^Restarts: (\d+) root, (\d+) in-tree$", text, re.M)
        if m:
            rec["restarts"], rec["tree_restarts"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"^Safe dual bounds: (\d+) LPs checked, (\d+) below", text, re.M)
        if m:  # printed at verbosity 1 when a safe bound was below its LP objective
            rec["safe_checks"], rec["safe_below"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"^Farkas proofs: (\d+) infeasible LPs checked, (\d+) failed, (\d+) passed", text, re.M)
        if m:  # printed at verbosity 1 when a ray failed its check (or passed only after a refactorization)
            rec["farkas_checks"], rec["farkas_failed"], rec["farkas_retried"] = (int(m.group(k)) for k in (1, 2, 3))
        m = re.search(r"^Unscaled dual feasibility: (\d+) LP solves continued, (\d+) iterations", text, re.M)
        if m:  # LP solves that the scaled dual feasibility test would have stopped earlier, and their iterations
            rec["unscaled_solves"], rec["unscaled_iters"] = int(m.group(1)), int(m.group(2))
    except OSError:
        pass
    if args.keep_sols:
        rec["sol_file"] = str(sol_path)
    if args.save_sols and sol.values and "verify" in details and cls not in (WRONG, ERROR):
        # A verified feasible point (optimal or not): a start file for later runs (--starts).
        Path(args.save_sols).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sol_path, Path(args.save_sols) / f"{name}.sol")
        rec["saved_sol"] = True
    return rec


def size_of(name):
    parts = name.split("_")
    return parts[1] if len(parts) >= 3 else None


def resolve_solver(spec):
    """--solver value -> argv list. A plain existing path is taken as is (Windows paths contain
    backslashes, which shlex would eat); otherwise the string is split like a shell command. If
    the binary does not exist, also try <path>.exe and the multi-config layout
    <dir>/Release/<name>.exe (Visual Studio generators), so `--solver build/popso` works on Windows."""
    cmd = [spec] if os.path.exists(spec) else shlex.split(spec, posix=os.name != "nt")
    exe = cmd[0]
    if not os.path.exists(exe):
        p = Path(exe)
        name = p.name if p.suffix == ".exe" else p.name + ".exe"
        for cand in (p.with_name(name), p.parent / "Release" / name):
            if cand.exists():
                cmd[0] = str(cand)
                break
    return cmd


RESUME_KEYS = ("time_limit", "threads", "mem_limit_mb", "gap", "opt_tol", "feas_tol", "int_tol")


def load_partial(path: Path):
    """Records of an earlier (possibly interrupted) run: a .jsonl written incrementally
    (first line {"meta": ...}, then one {"result": ...} per instance; a torn last line
    from a crash is ignored) or a finished .json.  Returns (meta, [records])."""
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        return d["meta"], d["results"]
    meta, recs = None, []
    for ln in path.read_text().splitlines():
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if "meta" in d:
            meta = d["meta"]
        elif "result" in d:
            recs.append(d["result"])
    if meta is None:
        raise ValueError(f"{path}: no meta line")
    return meta, recs


class _Terminated(Exception):
    pass


def _on_sigterm(signum, frame):
    raise _Terminated()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--verify-worker"]:
        return verify_worker(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", nargs="*", help="files, globs or directories "
                    "(default: benchmarks/instances; with --miplib: benchmarks/miplib)")
    ap.add_argument("--solver", required=True, help="solver binary (or quoted command)")
    ap.add_argument("-t", "--time-limit", type=float, default=60.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("-j", "--jobs", type=int, default=1,
                    help="instances run concurrently (default 1: clean timings and low memory; "
                         "shared machine: keep <= 3)")
    ap.add_argument("--mem-limit", type=int, default=2048,
                    help="memory limit per solver run in MB, passed as --mem-limit to the solver "
                         "and enforced by an RSS watchdog at 125%% (0 = none)")
    ap.add_argument("--gap", type=float, default=1e-6,
                    help="relative MIP gap passed to the solver as --gap (default 1e-6; "
                         "the correctness check uses --opt-tol independently)")
    ap.add_argument("--no-solver-mem-flag", action="store_true",
                    help="do not pass --mem-limit to the solver (watchdog only)")
    ap.add_argument("--miplib", action="store_true", help="run the downloaded MIPLIB subset (48)")
    ap.add_argument("--set", action="append", default=[],
                    help="instance list: a name in benchmarks/sets/ (e.g. miplib_wide) or a file; "
                         "one path per line, relative to the repo root; repeatable")
    ap.add_argument("--sizes", help="comma list, e.g. tiny,small (generated instances only)")
    ap.add_argument("--max-size", help="include sizes up to this one, e.g. medium")
    ap.add_argument("--families", help="comma list of family prefixes")
    ap.add_argument("--exclude", help="regex of instance names to skip")
    ap.add_argument("--reference", default=str(BENCH / "reference" / "solutions.json"))
    ap.add_argument("--miplib-solu", default=str(BENCH / "reference" / "miplib_solu.json"))
    ap.add_argument("--opt-tol", type=float, default=1e-6, help="relative objective tolerance")
    ap.add_argument("--feas-tol", type=float, default=1e-6)
    ap.add_argument("--int-tol", type=float, default=1e-6)
    ap.add_argument("--out-dir", default=str(BENCH / "results"))
    ap.add_argument("--tag", default="", help="label appended to the result file name")
    ap.add_argument("--no-save", action="store_true", help="do not write result files")
    ap.add_argument("--keep-sols", action="store_true", help="keep .sol/.log under results/tmp")
    ap.add_argument("--starts", metavar="DIR",
                    help="warm starts: an instance with a file DIR/<instance>.sol gets it as --start <file> "
                         "(the others run without a start; the record's \"start\" says which)")
    ap.add_argument("--save-sols", metavar="DIR",
                    help="copy every verified feasible solution to DIR/<instance>.sol (start files for --starts, "
                         "e.g. from a short run of this solver or of benchmarks/tools/highs_cli.py)")
    ap.add_argument("--trace", action="store_true",
                    help="popso only: pass --profile and record each run's incumbents over time and its primal "
                         "integral over the time limit (mean primal gap to the reference objective, lib/primal.py)")
    ap.add_argument("--strict", action="store_true",
                    help="exit code 1 also when an instance is not solved (errors, limits; for CI)")
    ap.add_argument("--resume", metavar="FILE",
                    help="continue an interrupted run: FILE is its results/<stamp>_<sha>.jsonl (or a "
                         "finished .json); instances recorded there are skipped, new results are "
                         "appended to it and the final .json/.csv get its name. Use the same command "
                         "line (time limit, threads, gap, memory limit must match)")
    ap.add_argument("--redo", default="",
                    help="with --resume: comma list of classes to run again (e.g. error,memlimit)")
    ap.add_argument("--verify-mem-limit", type=int, default=4096,
                    help="memory cap (MB) of the Python verifier process (reads the MPS file; large "
                         "MIPLIB instances need GBs; 0 = none)")
    ap.add_argument("--verify-timeout", type=float, default=900.0,
                    help="wall-clock limit (s) of the Python verifier process")
    args = ap.parse_args(argv)

    solver_cmd = resolve_solver(args.solver)
    if not os.path.exists(solver_cmd[0]) and len(solver_cmd) == 1:
        print(f"solver binary not found: {solver_cmd[0]}", file=sys.stderr)
        return 2
    set_files = [f for s in args.set for f in read_set(s)]
    missing = [str(f) for f in set_files if not f.exists()]
    if missing:
        print(f"{len(missing)} instance file(s) of the set are missing (run benchmarks/miplib/download.py "
              f"--wide / --benchmark?): {missing[:5]}", file=sys.stderr)
        return 2
    pats = args.instances or ([] if set_files else [str(BENCH / ("miplib" if args.miplib else "instances"))])
    files = expand(pats + [str(f) for f in set_files])
    if args.miplib and not args.instances and not set_files and Path(args.miplib_solu).exists():
        d = json.loads(Path(args.miplib_solu).read_text())
        subset = set(d.get("sets", {}).get("subset", d["instances"]))
        files = [f for f in files if inst_name(f) in subset]
    if args.sizes:
        allowed = set(args.sizes.split(","))
        files = [f for f in files if size_of(inst_name(f)) in allowed or inst_name(f).startswith("lpsanity")]
    if args.max_size:
        lim = SIZE_ORDER.index(args.max_size)
        files = [f for f in files if size_of(inst_name(f)) not in SIZE_ORDER
                 or SIZE_ORDER.index(size_of(inst_name(f))) <= lim]
    if args.families:
        fams = tuple(args.families.split(","))
        files = [f for f in files if inst_name(f).split("_")[0] in fams]
    if args.exclude:
        rx = re.compile(args.exclude)
        files = [f for f in files if not rx.search(inst_name(f))]
    if not files:
        print("no instances selected", file=sys.stderr)
        return 2

    refs = load_references(args.reference, args.miplib_solu)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    sha = git_sha()
    out_dir = Path(args.out_dir)
    base = f"{stamp}_{sha}" + (f"_{args.tag}" if args.tag else "")
    meta = {
        "timestamp": stamp, "git_sha": sha, "solver": _strip_repo(args.solver), "time_limit": args.time_limit,
        "threads": args.threads, "jobs": args.jobs, "mem_limit_mb": args.mem_limit,
        "gap": args.gap,
        "opt_tol": args.opt_tol,
        "feas_tol": args.feas_tol, "int_tol": args.int_tol, "argv": [_strip_repo(a) for a in argv],
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    done, skipped = [], []  # records of the run being resumed; their files
    if args.resume:
        if args.no_save:
            print("--resume and --no-save exclude each other", file=sys.stderr)
            return 2
        rpath = Path(args.resume)
        try:
            old_meta, done = load_partial(rpath)
        except (OSError, ValueError, KeyError) as e:
            print(f"cannot resume from {rpath}: {e}", file=sys.stderr)
            return 2
        diff = {k: (old_meta.get(k), meta[k]) for k in RESUME_KEYS if old_meta.get(k) != meta[k]}
        if diff:
            print(f"cannot resume {rpath}: settings differ (old, new): {diff}", file=sys.stderr)
            return 2
        if old_meta.get("git_sha") != sha:
            print(f"WARNING: resuming a run of {old_meta.get('git_sha')} with {sha}", file=sys.stderr)
        redo = {c.strip() for c in args.redo.split(",") if c.strip()}
        done = [r for r in done if r["class"] not in redo]
        resumed = list(old_meta.get("resumed", [])) + [
            {"timestamp": stamp, "git_sha": sha, "argv": meta["argv"], "kept": len(done)}]
        meta = {**old_meta, "resumed": resumed}
        out_dir = rpath.parent
        base = rpath.name.removesuffix(".jsonl").removesuffix(".json")
        finished = {r["instance"] for r in done}
        skipped = [f for f in files if inst_name(f) in finished]
        files = [f for f in files if inst_name(f) not in finished]
        print(f"resuming {rpath}: {len(done)} instance(s) recorded, {len(skipped)} of the selection "
              f"skipped, {len(files)} to run")
    jsonl = out_dir / f"{base}.jsonl"
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
        # (re)write the header and the kept records; also drops --redo records and a torn last line
        tmp = jsonl.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as fh:
            fh.write(json.dumps({"meta": meta}) + "\n")
            for r in done:
                fh.write(json.dumps({"result": r}) + "\n")
        tmp.replace(jsonl)
    tmpdir = BENCH / "results" / "tmp" / base  # with the tag: concurrent runs (ab.py) keep apart
    tmpdir.mkdir(parents=True, exist_ok=True)

    print(f"running {len(files)} instances, time limit {args.time_limit:g}s, threads {args.threads}, "
          f"gap {args.gap:g}, mem {args.mem_limit} MB, jobs {args.jobs}, solver {args.solver} @ {sha}")
    if not args.no_save:
        print(f"progress: {_shown(jsonl)}  (after an interruption: same command + --resume {_shown(jsonl)})")
    hdr = f"{'instance':<40} {'class':<8} {'status':<11} {'objective':>18} {'ref':>18} {'time':>8} {'highs':>8}"
    print(hdr)
    print("-" * len(hdr))
    recs = list(done)

    def show(r):
        def f(x):
            return f"{x:.10g}" if isinstance(x, (int, float)) and x is not None else "-"
        ht = r.get("ref_highs_time")
        line = (f"{r['instance']:<40} {r['class']:<8} {str(r['status']):<11} {f(r['objective']):>18} "
                f"{f(r.get('ref_objective')) if r.get('ref_status') == OPTIMAL else str(r.get('ref_status', '-')):>18} "
                f"{r['time']:>8.2f} {(f'{ht:.2f}' if ht is not None else '-'):>8}")
        if r["class"] == WRONG:
            line += f"   <<<<<< WRONG: {r['message']}"
        elif r["class"] == ERROR:
            line += f"   << {r['message']}"
        print(line, flush=True)

    def record(r):
        show(r)
        recs.append(r)
        if not args.no_save:
            # one line per finished instance: a crash loses at most the instances still running
            with open(jsonl, "a") as fh:
                fh.write(json.dumps({"result": r}) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    old_term = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        if args.jobs > 1:
            with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                futs = [ex.submit(run_one, p, args, solver_cmd, tmpdir, refs) for p in files]
                try:
                    for fut in as_completed(futs):
                        record(fut.result())
                except BaseException:
                    for fu in futs:
                        fu.cancel()
                    kill_all()
                    raise
        else:
            for p in files:
                record(run_one(p, args, solver_cmd, tmpdir, refs))
    except (KeyboardInterrupt, _Terminated):
        kill_all()
        print(f"\ninterrupted; {len(recs)} instance(s) recorded", file=sys.stderr)
        if not args.no_save:
            print(f"continue with the same command plus  --resume {_shown(jsonl)}", file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGTERM, old_term)
    order = {inst_name(f): i for i, f in enumerate(skipped + files)}
    recs.sort(key=lambda r: order.get(r["instance"], -1))

    # ----------------------------------------------------------- summary
    tl = args.time_limit
    n = len(recs)
    by = {}
    for r in recs:
        by[r["class"]] = by.get(r["class"], 0) + 1
    solved = by.get(OK, 0) + by.get(NOREF, 0)
    wrong = [r for r in recs if r["class"] == WRONG]

    def eff_time(r):
        return min(r["time"], tl) if r["class"] in (OK, NOREF) else tl

    ours = [eff_time(r) for r in recs]

    def ref_eff(r, key):
        t, st = r.get(f"ref_{key}_time"), r.get(f"ref_{key}_status")
        if t is None:
            return None
        return min(t, tl) if st in (OPTIMAL, INFEASIBLE, UNBOUNDED, INF_OR_UNBD) else tl

    summary = {
        "instances": n,
        "solved": solved,
        "wrong": len(wrong),
        "unsolved": by.get(UNSOLVED, 0),
        "memlimit": by.get(MEMLIMIT, 0),
        "errors": by.get(ERROR, 0),
        "classes": by,
        "sgm_time": sgm(ours),
        "sgm_shift": 1.0,
        "time_limit": tl,
    }
    for key in REF_SOLVERS:
        pairs = [(eff_time(r), ref_eff(r, key)) for r in recs]
        pairs = [p for p in pairs if p[1] is not None]
        if pairs:
            summary[f"ref_{key}_instances"] = len(pairs)
            summary[f"ref_{key}_solved"] = sum(1 for p in pairs if p[1] < tl)
            summary[f"ref_{key}_sgm_time"] = sgm([p[1] for p in pairs])
            summary[f"ours_sgm_time_on_{key}_set"] = sgm([p[0] for p in pairs])

    print("-" * len(hdr))
    print(f"instances {n}  solved {solved}  unsolved {by.get(UNSOLVED, 0)}  memlimit {by.get(MEMLIMIT, 0)}  "
          f"errors {by.get(ERROR, 0)}  "
          f"WRONG {len(wrong)}")
    pis = [r["primal_integral"] for r in recs if r.get("primal_integral") is not None]
    if pis:
        summary["primal_integral_mean"] = sum(pis) / len(pis)
        print(f"primal integral / time limit (mean over the {len(pis)} instances with a reference objective): "
              f"{summary['primal_integral_mean']:.4f}")
    print(f"shifted geometric mean time (shift 1s, unsolved = {tl:g}s): popso {summary['sgm_time']:.3f}s", end="")
    for key in REF_SOLVERS:
        if f"ref_{key}_sgm_time" in summary:
            print(f" | {key} {summary[f'ref_{key}_sgm_time']:.3f}s "
                  f"(solved {summary[f'ref_{key}_solved']}/{summary[f'ref_{key}_instances']})", end="")
    print()
    for key in REF_SOLVERS:
        if f"ref_{key}_sgm_time" in summary and summary[f"ref_{key}_instances"] != n:
            print(f"  (on the {summary[f'ref_{key}_instances']} instances with {key} times: "
                  f"popso {summary[f'ours_sgm_time_on_{key}_set']:.3f}s)")
    print("  note: reference times are from reference/solutions.json (measured separately, "
          "possibly on another machine / with a longer time limit)")

    if not args.no_save:
        tmp = out_dir / f"{base}.json.tmp"
        tmp.write_text(json.dumps({"meta": meta, "summary": summary, "results": recs}, indent=1) + "\n")
        tmp.replace(out_dir / f"{base}.json")
        jsonl.unlink(missing_ok=True)  # everything is in the .json now
        cols = ["instance", "class", "status", "objective", "ref_status", "ref_objective", "time", "peak_rss_mb",
                "ref_highs_time", "ref_scip_time", "gap_to_ref", "max_viol", "returncode", "message"]
        with open(out_dir / f"{base}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in recs:
                w.writerow(r)
        print(f"results: {out_dir / (base + '.json')}")
    if not args.keep_sols:
        failing = {r["instance"] for r in recs if r["class"] in (WRONG, ERROR)}
        for f in tmpdir.glob("*"):
            if f.stem not in failing:
                f.unlink()
        try:
            tmpdir.rmdir()
        except OSError:
            print(f"logs of failing runs kept in {tmpdir}")

    if wrong:
        print("\n" + "!" * 78)
        print(f"!!! {len(wrong)} WRONG ANSWER(S):")
        for r in wrong:
            print(f"!!!   {r['instance']}: {r['message']}")
        print("!" * 78)
        return 1
    if args.strict and solved < n:
        print(f"--strict: {n - solved} of {n} instances not solved")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
