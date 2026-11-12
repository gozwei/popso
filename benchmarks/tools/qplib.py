#!/usr/bin/env python3
"""Download the convex instances of QPLIB (qplib.zib.de, CC BY 4.0) and convert them to MPS for the
benchmark harness: QUADOBJ for a quadratic objective, QCMATRIX for quadratic constraints.

    .venv/bin/python benchmarks/tools/qplib.py --solver build/popso [--out benchmarks/qplib]
        [--classes LCD,LMC,LMD] [--max-nz 2000000] [--binary] [NAMES ...]

The instance list and the problem classes (three letters: objective, variables, constraints; see
qplib.zib.de/doc.html) come from the site's instance table; only instances marked convex (continuous
relaxation convex) are taken, or with --binary the nonconvex ones over binary variables only (classes
QBL, QBN, LBQ, QBQ: popso linearizes their products exactly) into benchmarks/sets/qplib_binary.txt. Each LP file is converted by `popso --write-anon --anon-keep-order`
(generic names x1.., c1.., the order and every number kept). The conversion is then checked
independently: QPLIB's published solution point, renamed through the name map, is verified on the
written MPS file by lib/mps.py (feasibility and the objective against the site's SOLOBJVALUE).
Writes <out>/QPLIB_nnnn.mps.gz, <out>/meta.json (class, sizes, published objective, check result)
and benchmarks/sets/qplib_convex.txt (the instances whose check passed).
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
REPO = BENCH.parent
sys.path.insert(0, str(BENCH))

from lib.mps import read_mps, verify  # noqa: E402

SITE = "https://qplib.zib.de/"


def fetch(url, dest=None):
    with urllib.request.urlopen(url, timeout=300) as r:
        data = r.read()
    if dest is not None:
        Path(dest).write_bytes(data)
    return data


def index():
    """[{name, cvx, cls, nvars, nbin, nint, ncons, nquad, nz}] from the site's instance table."""
    t = fetch(SITE + "instances.html").decode("utf-8", "replace")
    out = []
    for r in re.findall(r"<TR bgcolor[^>]*>(.*?)</TR>", t, re.S):
        tds = [html.unescape(re.sub(r"<[^>]+>", "", c)).replace("\xa0", "").strip()
               for c in re.findall(r"<TD[^>]*>(.*?)</TD>", r, re.S)]
        m = re.search(r"QPLIB_\d+", r)
        if len(tds) < 13 or not m:
            continue
        num = lambda s: int(s) if s.isdigit() else 0  # noqa: E731
        out.append(dict(name=m.group(0), cvx=tds[1] not in ("-", ""), cls=tds[2] + tds[5] + tds[9],
                        nvars=num(tds[6]), nbin=num(tds[7]), nint=num(tds[8]), ncons=num(tds[10]),
                        nquad=num(tds[11]), nz=num(tds[12])))
    return out


def published_objective(name):
    t = html.unescape(re.sub(r"<[^>]+>", " ", fetch(f"{SITE}{name}.html").decode("utf-8", "replace")))
    m = re.search(r"solobjvalue\s+(\S+)", re.sub(r"\s+", " ", t))
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def read_qplib_sol(path):
    vals = {}
    for ln in Path(path).read_text().splitlines():
        f = ln.split()
        if len(f) >= 2:
            vals[f[0]] = float(f[1])
    return vals


def convert(solver, name, out, work):
    """LP -> out/name.mps.gz; returns the check result (dict)."""
    lp = work / f"{name}.lp"
    fetch(f"{SITE}lp/{name}.lp", lp)
    mps, amap = work / f"{name}.mps", work / f"{name}.map"
    p = subprocess.run([solver, str(lp), "--write-anon", str(mps), "--anon-keep-order", "--anon-map", str(amap)],
                       capture_output=True, text=True)
    if p.returncode != 0 or not mps.exists():
        return dict(ok=False, error=(p.stderr or p.stdout).strip()[-300:])
    rename = {}
    for ln in amap.read_text().splitlines():
        f = ln.split()
        if len(f) == 2:
            rename[f[1]] = f[0]
    res = dict(ok=True)
    try:
        sol = work / f"{name}.sol"
        fetch(f"{SITE}sol/{name}.sol", sol)
        vals = {rename[k]: v for k, v in read_qplib_sol(sol).items() if k in rename}
        model = read_mps(mps)
        v = verify(model, vals)
        res.update(sol_feasible=v.feasible, sol_objective=v.objective, sol_max_viol=max(v.max_row_viol, v.max_bound_viol, v.max_int_viol),
                   sol_worst=v.worst)
    except Exception as e:  # no published point, or one the reader cannot take
        res.update(sol_error=str(e)[:200])
    with open(mps, "rb") as fi, gzip.open(out / f"{name}.mps.gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    for f in (lp, mps, amap):
        f.unlink(missing_ok=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", default=str(REPO / "build" / "popso"), help="popso binary (does the LP -> MPS conversion)")
    ap.add_argument("--out", default=str(BENCH / "qplib"))
    ap.add_argument("--classes", help="comma list of problem classes to keep (e.g. LCD,LMC,LMD,DML); default all convex")
    ap.add_argument("--max-nz", type=int, default=2_000_000, help="skip instances with more nonzeros")
    ap.add_argument("--binary", action="store_true", help="the nonconvex instances over binaries instead of the convex ones")
    ap.add_argument("--nonconvex", action="store_true",
                    help="the nonconvex instances with continuous or general integer variables (spatial branch-and-bound, v0.10) "
                         "into sets/qplib_nonconvex.txt")
    ap.add_argument("--set", help="set file to write ('' = none; default sets/qplib_convex.txt, qplib_binary.txt or "
                                  "qplib_nonconvex.txt)")
    ap.add_argument("names", nargs="*", help="only these instances (QPLIB_nnnn)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    keep = set(args.classes.split(",")) if args.classes else None
    wanted = (lambda d: not d["cvx"] and d["cls"][1] == "B") if args.binary else (lambda d: d["cvx"])
    if args.nonconvex:
        wanted = lambda d: not d["cvx"] and d["cls"][1] != "B"  # noqa: E731
    if args.set is None:
        args.set = str(BENCH / "sets" / ("qplib_nonconvex.txt" if args.nonconvex else
                                         "qplib_binary.txt" if args.binary else "qplib_convex.txt"))
    todo = [d for d in index() if wanted(d) and (keep is None or d["cls"] in keep) and d["nz"] <= args.max_nz
            and (not args.names or d["name"] in args.names)]
    print(f"{len(todo)} {'nonconvex' if args.nonconvex else 'nonconvex binary' if args.binary else 'convex'} instances selected",
          flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        for d in todo:
            name = d["name"]
            res = convert(args.solver, name, out, Path(tmp))
            pub = published_objective(name)
            rec = dict(d, published_objective=pub, **res)
            if res.get("sol_objective") is not None and pub is not None:
                rec["check"] = bool(res["sol_feasible"]) and abs(res["sol_objective"] - pub) <= 1e-6 * max(1.0, abs(pub))
            else:
                rec["check"] = None
            meta[name] = rec
            meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
            print(f"{name} {d['cls']} nz={d['nz']}: {'converted' if res['ok'] else 'FAILED ' + res.get('error', '')}"
                  f"  published {pub}  check {rec['check']} {res.get('sol_objective')} viol {res.get('sol_max_viol')}"
                  f" {res.get('sol_error', '')}", flush=True)
    if args.set:
        rel = Path(args.out).resolve()
        kind = ("nonconvex instances with continuous or general integer variables" if args.nonconvex else
                "nonconvex instances over binaries" if args.binary else "convex instances")
        lines = [f"# QPLIB {kind} (benchmarks/tools/qplib.py); class: objective / variables / constraints"]
        for name, r in sorted(meta.items()):
            if r.get("ok") and r.get("check") is not False and wanted(r):
                p = rel / f"{name}.mps.gz"
                lines.append(f"{p.relative_to(REPO) if p.is_relative_to(REPO) else p}  # {r['cls']}")
        Path(args.set).write_text("\n".join(lines) + "\n")
        print(f"wrote {args.set}: {len(lines) - 1} instances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
