#!/usr/bin/env python3
"""Markdown tables and the cactus plot of a benchmark run against the HiGHS / SCIP reference times.

  python3 tools/bench_report.py RESULT.json [MORE_SEEDS.json ...] [--limit 120] [--svg docs/x.svg]
                                [--title "..."] [--subtitle "..."]

RESULT.json is a result file of benchmarks/run.py (it carries the reference times of its --reference file).
The table and the plot use the first file; further files (other seeds) only enter the summary line."""
import json, math, sys

def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

files, skip = [], False
for a in sys.argv[1:]:
    if skip: skip = False
    elif a.startswith("--"): skip = True
    else: files.append(a)
lim = float(arg("--limit", "120"))
runs = [json.load(open(f))["results"] for f in files]
SOLVED = ("optimal", "infeasible", "unbounded")

def times(run):
    return {x["instance"]: (x["time"] if x["class"] == "ok" and x["time"] <= lim else None) for x in run}

def ref_times(run, s):
    return {x["instance"]: (x.get(f"ref_{s}_time") if x.get(f"ref_{s}_status") in SOLVED and
                            (x.get(f"ref_{s}_time") or 1e9) <= lim else None) for x in run}

def sgm(t):
    return math.exp(sum(math.log((v if v is not None else lim) + 1.0) for v in t.values()) / len(t)) - 1.0

cols = {"popso": times(runs[0]), "HiGHS": ref_times(runs[0], "highs"), "SCIP": ref_times(runs[0], "scip")}
n = len(cols["popso"])
if len(runs) > 1:
    print("seeds: " + ", ".join(f"{sum(v is not None for v in times(r).values())}/{n} ({sgm(times(r)):.1f} s)" for r in runs))
    print()
print("| solver | solved | shifted geomean (s) |\n|---|---|---|")
for k, t in cols.items():
    print(f"| {k} | {sum(v is not None for v in t.values())}/{n} | {sgm(t):.1f} |")
names = sorted(cols["popso"], key=str.lower)
fastest = {k: 0 for k in cols}
only, missed, none = [], [], 0
for i in names:
    v = {k: cols[k][i] for k in cols}
    solved = [k for k in v if v[k] is not None]
    if not solved: none += 1; continue
    fastest[min(solved, key=lambda k: v[k])] += 1
    if solved == ["popso"]: only.append(i)
    if "popso" not in solved: missed.append(i)
print(f"\nno solver: {none}; fastest: {fastest}\npopso only ({len(only)}): {', '.join(only)}\n"
      f"not popso ({len(missed)}): {', '.join(missed)}\n")
print("| instance | popso | HiGHS | SCIP |\n|---|---|---|---|")
for i in names:
    v = [cols[k][i] for k in cols]
    best = min((x for x in v if x is not None), default=None)
    print(f"| {i} | " + " | ".join("TL" if x is None else (f"**{x:.1f}**" if x == best else f"{x:.1f}") for x in v) + " |")

svg = arg("--svg")
if svg:
    X0, X1, Y0, Y1, tmin = 56.0, 602.0, 354.0, 58.0, 0.01
    fx = lambda t: X0 + (X1 - X0) * (math.log10(max(t, tmin)) - math.log10(tmin)) / (math.log10(lim) - math.log10(tmin))
    fy = lambda c: Y0 + (Y1 - Y0) * c / n
    title = arg("--title", "Benchmark")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 400" width="720" height="400" role="img" '
           f'aria-label="{title}: instances solved versus time">',
           "<style>svg{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}.bg{fill:#fcfcfb}"
           ".t1{fill:#0b0b0b}.t2{fill:#52514e}.grid{stroke:#e4e3df}.axis{stroke:#a3a29c}.s-ours{stroke:#2a78d6}"
           ".f-ours{fill:#2a78d6}.s-highs{stroke:#eb6834}.f-highs{fill:#eb6834}.s-scip{stroke:#1baf7a}.f-scip{fill:#1baf7a}"
           "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.t1{fill:#f0efec}.t2{fill:#c3c2b7}.grid{stroke:#333331}"
           ".axis{stroke:#6b6a64}.s-ours{stroke:#3987e5}.f-ours{fill:#3987e5}.s-highs{stroke:#d95926}.f-highs{fill:#d95926}"
           ".s-scip{stroke:#199e70}.f-scip{fill:#199e70}}</style>",
           '<rect class="bg" width="720" height="400" rx="8"/>',
           f'<text class="t1" x="56" y="24" font-size="15" font-weight="600">{title}</text>',
           f'<text class="t2" x="56" y="42" font-size="12">{arg("--subtitle", "")}</text>']
    step = 10 if n <= 300 else 50
    for c in list(range(0, n, step)) + [n]:
        if c != n and n - c < step / 2: continue
        out.append(f'<line class="grid" x1="56" x2="602" y1="{fy(c):.1f}" y2="{fy(c):.1f}" stroke-width="1"/>')
        out.append(f'<text class="t2" x="48" y="{fy(c) + 4:.1f}" font-size="11" text-anchor="end">{c}</text>')
    for t in [0.01, 0.1, 1, 10, lim]:
        out.append(f'<line class="grid" x1="{fx(t):.1f}" x2="{fx(t):.1f}" y1="58" y2="354" stroke-width="1"/>')
        out.append(f'<text class="t2" x="{fx(t):.1f}" y="372" font-size="11" text-anchor="middle">{t:g}</text>')
    out.append('<line class="axis" x1="56" x2="602" y1="354" y2="354" stroke-width="1"/>')
    out.append('<text class="t2" x="329.0" y="392" font-size="12" text-anchor="middle">time limit per instance (seconds, log scale)</text>')
    out.append('<text class="t2" transform="translate(16 206.0) rotate(-90)" font-size="12" text-anchor="middle">instances solved</text>')
    ends = []
    for k, cls in (("popso", "ours"), ("HiGHS", "highs"), ("SCIP", "scip")):
        ts = sorted(v for v in cols[k].values() if v is not None)
        d = [f"M{X0:.1f} {Y0:.1f}"]
        for c, t in enumerate(ts):
            d += [f"L{fx(t):.1f} {fy(c):.1f}", f"L{fx(t):.1f} {fy(c + 1):.1f}"]
        d.append(f"L{X1:.1f} {fy(len(ts)):.1f}")
        out.append(f'<path class="s-{cls}" d="{" ".join(d)}" fill="none" stroke-width="2" stroke-linejoin="round"/>')
        ends.append([fy(len(ts)), k, cls, len(ts)])
    ends.sort()
    for a, b in zip(ends, ends[1:]):
        if b[0] - a[0] < 15: b[0] = a[0] + 15
    for y, k, cls, c in ends:
        out.append(f'<circle class="f-{cls}" cx="614.0" cy="{y:.1f}" r="4"/>')
        out.append(f'<text class="t1" x="623.0" y="{y + 4:.1f}" font-size="12">{k} <tspan class="t2">{c}/{n}</tspan></text>')
    out.append("</svg>")
    open(svg, "w").write("\n".join(out) + "\n")
