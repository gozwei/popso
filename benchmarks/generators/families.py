"""Random instance families built with the pure-Python modeler (lib/modeler.py).

Every builder has the signature  build_<family>(rng, **params) -> Model
where `rng` is a numpy Generator seeded by the driver (generate.py).  Models
only use plain linear constraints (no addRange / general constraints) so that
the written MPS is a faithful, solver-neutral MILP.  Integer variables always
get explicit finite or infinite bounds (no reliance on MARKER defaults).
"""

from __future__ import annotations

import math

import numpy as np

from lib.modeler import Model
from lib.modeler import lin as _lin
from lib.modeler import quicksum as qsum


def _model(name):
    return Model(name)


# --------------------------------------------------------------- knapsack

def build_knapsack(rng, n, R=1000, kind="strong"):
    """Single 0-1 knapsack. kind: 'uncorr' | 'weak' | 'strong' (Pisinger)."""
    w = rng.integers(1, R + 1, n)
    if kind == "uncorr":
        p = rng.integers(1, R + 1, n)
    elif kind == "weak":
        p = np.maximum(1, w + rng.integers(-R // 10, R // 10 + 1, n))
    else:
        p = w + R // 10
    C = int(w.sum() // 2)
    m = _model("knapsack")
    x = [m.add_var(vtype="B", obj=float(p[j]), name=f"x{j}") for j in range(n)]
    m.add_constr(_lin(w, x) <= C, "cap")
    m.sense = "max"
    return m


def build_mkp(rng, n, k, tight=0.25):
    """Multi-dimensional knapsack (Chu & Beasley 1998)."""
    W = rng.integers(1, 1001, (k, n))
    p = W.sum(axis=0) / k + 500 * rng.random(n)
    p = np.round(p).astype(int)
    m = _model("mkp")
    x = [m.add_var(vtype="B", obj=float(p[j]), name=f"x{j}") for j in range(n)]
    for i in range(k):
        m.add_constr(_lin(W[i], x) <= math.floor(tight * W[i].sum()), f"c{i}")
    m.sense = "max"
    return m


# ------------------------------------------------------ covering / packing

def _random_columns(rng, rows, cols, density, min_per_row):
    """Random 0/1 incidence (list of row sets per column), every row covered
    by at least `min_per_row` columns and every column non-empty."""
    A = rng.random((rows, cols)) < density
    for i in range(rows):
        cnt = A[i].sum()
        if cnt < min_per_row:
            A[i, rng.choice(cols, min_per_row, replace=False)] = True
    for j in range(cols):
        if not A[:, j].any():
            A[rng.integers(rows), j] = True
    return A


def build_setcover(rng, rows, cols, density, cmax=100):
    A = _random_columns(rng, rows, cols, density, 2)
    c = rng.integers(1, cmax + 1, cols)
    m = _model("setcover")
    x = [m.add_var(vtype="B", obj=float(c[j]), name=f"x{j}") for j in range(cols)]
    for i in range(rows):
        idx = np.nonzero(A[i])[0]
        m.add_constr(_lin(np.ones(len(idx)), [x[j] for j in idx]) >= 1, f"r{i}")
    return m


def build_setpack(rng, rows, cols, density, cmax=100):
    A = _random_columns(rng, rows, cols, density, 2)
    # weight correlated with column size makes it less trivial
    c = A.sum(axis=0) * 10 + rng.integers(1, cmax + 1, cols)
    m = _model("setpack")
    x = [m.add_var(vtype="B", obj=float(c[j]), name=f"x{j}") for j in range(cols)]
    for i in range(rows):
        idx = np.nonzero(A[i])[0]
        m.add_constr(_lin(np.ones(len(idx)), [x[j] for j in idx]) <= 1, f"r{i}")
    m.sense = "max"
    return m


# ------------------------------------------------------------ bin packing

def build_binpack(rng, n, C=150, smin=20, smax=100):
    s = rng.integers(smin, smax + 1, n)
    # first-fit decreasing gives the number of bins to model
    bins = []
    for v in sorted(s, reverse=True):
        for b in range(len(bins)):
            if bins[b] + v <= C:
                bins[b] += v
                break
        else:
            bins.append(v)
    B = len(bins)
    m = _model("binpack")
    y = [m.add_var(vtype="B", obj=1.0, name=f"y{b}") for b in range(B)]
    x = [[m.add_var(vtype="B", name=f"x{i}_{b}") for b in range(B)] for i in range(n)]
    for i in range(n):
        m.add_constr(_lin(np.ones(B), x[i]) == 1, f"assign{i}")
    for b in range(B):
        m.add_constr(_lin(s, [x[i][b] for i in range(n)]) - C * y[b] <= 0, f"cap{b}")
    for b in range(B - 1):
        m.add_constr(y[b] - y[b + 1] >= 0, f"sym{b}")
    return m


# ---------------------------------------------------- generalized assignment

def build_gap(rng, agents, jobs, kind="D", ratio=0.8):
    a = rng.integers(1, 101, (agents, jobs))
    if kind == "D":
        c = 111 - a + rng.integers(-10, 11, (agents, jobs))
    else:  # type C
        c = rng.integers(10, 51, (agents, jobs))
    b = np.floor(ratio * a.sum(axis=1) / agents).astype(int)
    m = _model("gap")
    x = [[m.add_var(vtype="B", obj=float(c[i, j]), name=f"x{i}_{j}")
          for j in range(jobs)] for i in range(agents)]
    for j in range(jobs):
        m.add_constr(_lin(np.ones(agents), [x[i][j] for i in range(agents)]) == 1, f"job{j}")
    for i in range(agents):
        m.add_constr(_lin(a[i], x[i]) <= int(b[i]), f"cap{i}")
    return m


# --------------------------------------------------- facility location

def build_cflp(rng, fac, cust, ratio=3.0, single_source=True):
    """Capacitated facility location (Cornuejols et al. style data)."""
    pf = rng.random((fac, 2))
    pc = rng.random((cust, 2))
    d = rng.integers(5, 36, cust)
    s = rng.integers(10, 161, fac).astype(float)
    s = np.ceil(s * ratio * d.sum() / s.sum())
    f = np.round(rng.uniform(0, 90, fac) + rng.uniform(100, 110, fac) * np.sqrt(s))
    dist = np.sqrt(((pf[:, None, :] - pc[None, :, :]) ** 2).sum(axis=2))
    cost = np.round(10 * dist * d[None, :], 2)
    m = _model("cflp")
    vt = "B" if single_source else "C"
    y = [m.add_var(vtype="B", obj=float(f[i]), name=f"y{i}") for i in range(fac)]
    x = [[m.add_var(vtype=vt, ub=1.0, obj=float(cost[i, j]), name=f"x{i}_{j}")
          for j in range(cust)] for i in range(fac)]
    for j in range(cust):
        m.add_constr(_lin(np.ones(fac), [x[i][j] for i in range(fac)]) == 1, f"dem{j}")
    for i in range(fac):
        m.add_constr(_lin(d, x[i]) - s[i] * y[i] <= 0, f"cap{i}")
        for j in range(cust):
            m.add_constr(x[i][j] - y[i] <= 0, f"link{i}_{j}")
    return m


# ------------------------------------------- multicommodity network design

def build_netdesign(rng, nodes, extra_arcs, commodities, cap_frac=0.3, fixed_scale=1000.0):
    """Capacitated multi-commodity fixed-charge network design.

    Arc set = bidirectional ring (guarantees feasibility) + random arcs.
    x[a,k] continuous flow, y[a] binary open; x[a,k] <= d_k y[a] (strong
    linking) and sum_k x[a,k] <= u_a y[a]."""
    pos = rng.random((nodes, 2))
    arcs = set()
    ring = set()
    for i in range(nodes):
        ring.add((i, (i + 1) % nodes))
        ring.add(((i + 1) % nodes, i))
    arcs |= ring
    while len(arcs) < 2 * nodes + extra_arcs:
        i, j = rng.integers(nodes, size=2)
        if i != j:
            arcs.add((int(i), int(j)))
    arcs = sorted(arcs)
    K = []
    while len(K) < commodities:
        o, t = rng.integers(nodes, size=2)
        if o != t:
            K.append((int(o), int(t), int(rng.integers(5, 26))))
    total = sum(k[2] for k in K)
    m = _model("netdesign")
    y, x = {}, {}
    for a in arcs:
        i, j = a
        L = float(np.linalg.norm(pos[i] - pos[j]))
        if a in ring:  # ring arcs can carry everything: guarantees feasibility
            u = float(total)
        else:
            u = float(max(10, round(total * cap_frac * rng.uniform(0.5, 1.5))))
        y[a] = m.add_var(vtype="B", obj=round(fixed_scale * L + rng.uniform(20, 60)), name=f"y{i}_{j}")
        for k, (o, t, dk) in enumerate(K):
            x[a, k] = m.add_var(obj=round(10 * L + 1, 2), name=f"x{i}_{j}_{k}")
        m.add_constr(qsum(x[a, k] for k in range(len(K))) - u * y[a] <= 0, f"cap{i}_{j}")
        for k, (o, t, dk) in enumerate(K):
            m.add_constr(x[a, k] - dk * y[a] <= 0, f"lk{i}_{j}_{k}")
    out_arcs = {v: [a for a in arcs if a[0] == v] for v in range(nodes)}
    in_arcs = {v: [a for a in arcs if a[1] == v] for v in range(nodes)}
    for k, (o, t, dk) in enumerate(K):
        for v in range(nodes):
            rhs = dk if v == o else (-dk if v == t else 0)
            m.add_constr(qsum(x[a, k] for a in out_arcs[v]) - qsum(x[a, k] for a in in_arcs[v]) == rhs,
                        f"fl{v}_{k}")
    return m


# ------------------------------------------------------------- routing

def _dist_matrix(rng, n, scale=100):
    p = rng.random((n, 2)) * scale
    return np.round(np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=2))).astype(int)


def build_tsp(rng, n):
    """Asymmetric-form TSP with Miller-Tucker-Zemlin subtour elimination."""
    D = _dist_matrix(rng, n)
    m = _model("tsp")
    x = {(i, j): m.add_var(vtype="B", obj=float(D[i, j]), name=f"x{i}_{j}")
         for i in range(n) for j in range(n) if i != j}
    u = {i: m.add_var(lb=1, ub=n - 1, name=f"u{i}") for i in range(1, n)}
    for i in range(n):
        m.add_constr(qsum(x[i, j] for j in range(n) if j != i) == 1, f"out{i}")
        m.add_constr(qsum(x[j, i] for j in range(n) if j != i) == 1, f"in{i}")
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                m.add_constr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2, f"mtz{i}_{j}")
    return m


def build_cvrp(rng, n, Q=None, qmax=10):
    """CVRP with MTZ load variables; depot 0, customers 1..n-1, free fleet."""
    D = _dist_matrix(rng, n)
    q = np.concatenate([[0], rng.integers(1, qmax + 1, n - 1)])
    if Q is None:
        Q = int(max(qmax, math.ceil(q.sum() / max(2, n // 5))))
    m = _model("cvrp")
    x = {(i, j): m.add_var(vtype="B", obj=float(D[i, j]), name=f"x{i}_{j}")
         for i in range(n) for j in range(n) if i != j}
    u = {i: m.add_var(lb=float(q[i]), ub=float(Q), name=f"u{i}") for i in range(1, n)}
    for i in range(1, n):
        m.add_constr(qsum(x[i, j] for j in range(n) if j != i) == 1, f"out{i}")
        m.add_constr(qsum(x[j, i] for j in range(n) if j != i) == 1, f"in{i}")
    m.add_constr(qsum(x[0, j] for j in range(1, n)) - qsum(x[j, 0] for j in range(1, n)) == 0, "depot")
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                # u_j >= u_i + q_j - Q (1 - x_ij)
                m.add_constr(u[i] - u[j] + Q * x[i, j] <= Q - int(q[j]), f"load{i}_{j}")
    return m


# ------------------------------------------------------------ lot sizing

def build_lotsize(rng, items, periods, util=0.85):
    """Multi-item capacitated lot sizing with setup times/costs and overtime."""
    d = rng.integers(0, 201, (items, periods))
    d[rng.random((items, periods)) < 0.2] = 0
    st = rng.integers(10, 51, items)
    sc = rng.integers(50, 1001, items)
    h = rng.integers(1, 6, items)
    cap = np.full(periods, math.ceil(d.sum() / periods / util))
    m = _model("lotsize")
    x, s, y = {}, {}, {}
    o = [m.add_var(obj=1000.0, name=f"ot{t}") for t in range(periods)]
    for i in range(items):
        cum = np.cumsum(d[i][::-1])[::-1]  # remaining demand from t
        for t in range(periods):
            x[i, t] = m.add_var(name=f"x{i}_{t}")
            s[i, t] = m.add_var(obj=float(h[i]), name=f"s{i}_{t}")
            y[i, t] = m.add_var(vtype="B", obj=float(sc[i]), name=f"y{i}_{t}")
            M = float(min(cum[t], cap[t] + 1000))
            m.add_constr(x[i, t] - M * y[i, t] <= 0, f"su{i}_{t}")
            prev = s[i, t - 1] if t > 0 else 0
            m.add_constr(prev + x[i, t] - s[i, t] == int(d[i, t]), f"bal{i}_{t}")
    for t in range(periods):
        m.add_constr(qsum(x[i, t] + int(st[i]) * y[i, t] for i in range(items)) - o[t] <= float(cap[t]),
                    f"cap{t}")
    return m


# ------------------------------------------------------------ scheduling

def build_jobshop(rng, jobs, machines, pmax=99):
    """Job shop makespan minimisation, disjunctive big-M formulation."""
    p = rng.integers(1, pmax + 1, (jobs, machines))
    route = [rng.permutation(machines) for _ in range(jobs)]
    M = int(p.sum())
    m = _model("jobshop")
    S = [[m.add_var(ub=M, name=f"s{j}_{k}") for k in range(machines)] for j in range(jobs)]
    Cmax = m.add_var(obj=1.0, ub=M, name="cmax")
    for j in range(jobs):
        r = route[j]
        for o in range(machines - 1):
            a, b = r[o], r[o + 1]
            m.add_constr(S[j][b] - S[j][a] >= int(p[j, a]), f"prec{j}_{o}")
        last = r[-1]
        m.add_constr(Cmax - S[j][last] >= int(p[j, last]), f"mk{j}")
    for k in range(machines):
        for j1 in range(jobs):
            for j2 in range(j1 + 1, jobs):
                z = m.add_var(vtype="B", name=f"z{j1}_{j2}_{k}")
                # z = 1: j1 before j2 on machine k
                m.add_constr(S[j1][k] + int(p[j1, k]) - S[j2][k] - M * (1 - z) <= 0, f"d{j1}_{j2}_{k}a")
                m.add_constr(S[j2][k] + int(p[j2, k]) - S[j1][k] - M * z <= 0, f"d{j1}_{j2}_{k}b")
    return m


def build_timeidx(rng, jobs, machines=1, pmax=10, wmax=10, release=True):
    """Time-indexed formulation for P|r_j|sum w_j C_j (identical machines,
    aggregated capacity constraint, which is exact for identical machines)."""
    p = rng.integers(1, pmax + 1, jobs)
    w = rng.integers(1, wmax + 1, jobs)
    T = int(math.ceil(p.sum() / machines)) + int(p.max())
    r = rng.integers(0, max(1, T // 2), jobs) if release else np.zeros(jobs, int)
    T += int(r.max())
    m = _model("timeidx")
    x = {}
    for j in range(jobs):
        for t in range(int(r[j]), T - int(p[j]) + 1):
            x[j, t] = m.add_var(vtype="B", obj=float(w[j] * (t + p[j])), name=f"x{j}_{t}")
    for j in range(jobs):
        m.add_constr(qsum(x[j, t] for t in range(T) if (j, t) in x) == 1, f"job{j}")
    for t in range(T):
        terms = [x[j, s] for j in range(jobs) for s in range(t - int(p[j]) + 1, t + 1) if (j, s) in x]
        if terms:
            m.add_constr(qsum(terms) <= machines, f"t{t}")
    return m


# ------------------------------------------------------ general MILP

def _random_sparse_rows(rng, rows, cols, density, coef=10):
    """Rows as (idx, vals) with integer coefficients in [-coef, coef] \\ {0}."""
    out = []
    for i in range(rows):
        k = max(2, rng.binomial(cols, density))
        idx = np.sort(rng.choice(cols, min(k, cols), replace=False))
        vals = rng.integers(1, coef + 1, len(idx)) * rng.choice([-1, 1], len(idx))
        out.append((idx, vals))
    # make sure each column appears somewhere
    used = np.zeros(cols, bool)
    for idx, _ in out:
        used[idx] = True
    for j in np.nonzero(~used)[0]:
        i = rng.integers(rows)
        idx, vals = out[i]
        out[i] = (np.append(idx, j), np.append(vals, rng.integers(1, coef + 1)))
    return out


def build_randmilp(rng, cols, rows, density, frac_int=0.5, frac_bin=0.3, frac_eq=0.1,
                   ray=False):
    """Random feasible sparse MILP with binaries, general integers (some with
    negative lower bounds) and continuous variables.  A hidden integer point
    x0 is generated first; rhs = A x0 + slack so x0 is feasible.

    ray=True makes the instance unbounded: a subset of integer columns gets
    infinite upper bounds and an integer direction d with A d <= 0 (= 0 on
    equality rows) and c.d < 0 is embedded."""
    nb = int(cols * frac_bin)
    ni = int(cols * frac_int)
    vt = np.array(["B"] * nb + ["I"] * ni + ["C"] * (cols - nb - ni))
    rng.shuffle(vt)
    lb = np.zeros(cols)
    ub = np.zeros(cols)
    for j in range(cols):
        if vt[j] == "B":
            ub[j] = 1
        elif vt[j] == "I":
            lb[j] = -5 if rng.random() < 0.2 else 0
            ub[j] = rng.integers(5, 21)
        else:
            lb[j] = -10 if rng.random() < 0.1 else 0
            ub[j] = rng.choice([50.0, 100.0, np.inf])
    x0 = np.empty(cols)
    for j in range(cols):
        hi = ub[j] if np.isfinite(ub[j]) else lb[j] + 20
        x0[j] = rng.uniform(lb[j], hi)
        # integers exact; continuous values with 2 decimals so that row
        # activities (integer coefficients) are exact short decimals
        x0[j] = np.round(x0[j]) if vt[j] != "C" else np.round(x0[j], 2)
    R = _random_sparse_rows(rng, rows, cols, density)
    u = rng.random(rows)
    kind = np.where(u < frac_eq, "E", np.where(rng.random(rows) < 0.5, "L", "G"))
    c = rng.integers(-20, 21, cols).astype(float)
    # bounded objective in the minimisation direction for unbounded cols
    for j in range(cols):
        if not np.isfinite(ub[j]):
            c[j] = abs(c[j]) + 1
    S = []
    if ray:
        ints = [j for j in range(cols) if vt[j] == "I"]
        S = list(rng.choice(ints, min(6, len(ints)), replace=False))
        d = {j: int(rng.integers(1, 3)) for j in S}
        for j in S:
            ub[j] = np.inf
        newR = []
        for i, (idx, vals) in enumerate(R):
            idx, vals = list(idx), list(vals)
            if kind[i] == "E":
                # remove ray columns from equality rows
                keep = [k for k in range(len(idx)) if idx[k] not in d]
                if len(keep) < 2:
                    kind[i] = "L"
                else:
                    idx = [idx[k] for k in keep]
                    vals = [vals[k] for k in keep]
            if kind[i] != "E":
                # need A_i d <= 0 for <= rows and A_i d >= 0 for >= rows
                sgn = 1 if kind[i] == "L" else -1
                r = sgn * sum(v * d[j] for j, v in zip(idx, vals) if j in d)
                if r > 0:
                    jstar = S[i % len(S)]
                    delta = sgn * (math.ceil(r / d[jstar]) + 1)
                    if jstar in idx:
                        k = idx.index(jstar)
                        vals[k] -= delta
                        if vals[k] == 0:
                            vals[k] = -sgn
                    else:
                        idx.append(jstar)
                        vals.append(-delta)
            newR.append((np.array(idx), np.array(vals)))
        R = newR
        for j in S:
            c[j] = -abs(c[j]) - 1
    m = _model("randmilp")
    x = []
    for j in range(cols):
        x.append(m.add_var(lb=lb[j], ub=ub[j] if np.isfinite(ub[j]) else float("inf"),
                          vtype=str(vt[j]), obj=float(c[j]), name=f"x{j}"))
    for i, (idx, vals) in enumerate(R):
        act = round(float(np.dot(vals, x0[idx])), 2)
        expr = _lin(vals, [x[j] for j in idx])
        if kind[i] == "E":
            m.add_constr(expr == act, f"e{i}")
        else:
            slack = float(rng.integers(0, 10))
            if kind[i] == "L":
                m.add_constr(expr <= round(act + slack, 2), f"l{i}")
            else:
                m.add_constr(expr >= round(act - slack, 2), f"g{i}")
    return m


# ------------------------------------------------------- infeasible MILPs

def build_pigeon(rng, holes):
    """Pigeonhole with pairwise conflicts: holes+1 pigeons, LP-feasible,
    integer-infeasible; exponential for B&B without clique reasoning."""
    P = holes + 1
    m = _model("pigeon")
    x = [[m.add_var(vtype="B", obj=float(rng.integers(1, 10)), name=f"x{p}_{h}")
          for h in range(holes)] for p in range(P)]
    for p in range(P):
        m.add_constr(qsum(x[p]) == 1, f"p{p}")
    for h in range(holes):
        for p in range(P):
            for q in range(p + 1, P):
                m.add_constr(x[p][h] + x[q][h] <= 1, f"c{h}_{p}_{q}")
    return m


def build_subsetsum(rng, n, lo, hi, side_rows=0):
    """Infeasible equality subset-sum: sum a_j x_j = b with b unreachable
    (checked by DP) but strictly inside the LP-feasible range."""
    a = rng.integers(lo, hi + 1, n)
    total = int(a.sum())
    reach = np.zeros(total + 1, bool)
    reach[0] = True
    for v in a:
        reach[v:] |= reach[:-v].copy()
    target = total // 2
    cands = np.nonzero(~reach)[0]
    b = int(cands[np.argmin(np.abs(cands - target))])
    m = _model("subsetsum")
    x = [m.add_var(vtype="B", obj=float(rng.integers(1, 10)), name=f"x{j}") for j in range(n)]
    m.add_constr(_lin(a, x) == b, "sum")
    for i in range(side_rows):
        idx = rng.choice(n, max(2, n // 3), replace=False)
        m.add_constr(qsum(x[j] for j in idx) <= len(idx) - 1, f"side{i}")
    return m


def build_transinf(rng, sources, sinks, frac_int=0.5):
    """Transportation problem with integer flows where total supply falls
    short of total demand by a tiny amount: LP-infeasible, needs aggregated
    reasoning (presolve or phase 1) to prove."""
    supply = rng.integers(20, 100, sources)
    demand = rng.integers(10, 60, sinks)
    # scale so total demand = total supply + 1
    diff = int(supply.sum() - demand.sum()) + 1
    demand[0] += diff if diff > 0 else 0
    if demand.sum() <= supply.sum():
        demand[0] += int(supply.sum() - demand.sum()) + 1
    cost = rng.integers(1, 50, (sources, sinks))
    m = _model("transinf")
    x = [[m.add_var(vtype="I" if rng.random() < frac_int else "C", ub=200.0,
                   obj=float(cost[i, j]), name=f"x{i}_{j}")
          for j in range(sinks)] for i in range(sources)]
    for i in range(sources):
        m.add_constr(qsum(x[i]) <= int(supply[i]), f"s{i}")
    for j in range(sinks):
        m.add_constr(qsum(x[i][j] for i in range(sources)) >= int(demand[j]), f"d{j}")
    return m


# ------------------------------------------------------------- pure LPs

def build_transport(rng, sources, sinks):
    """Balanced transportation LP (continuous)."""
    supply = rng.integers(50, 500, sources)
    demand = rng.multinomial(int(supply.sum()), np.ones(sinks) / sinks)
    ps = rng.random((sources, 2))
    pt = rng.random((sinks, 2))
    cost = np.round(100 * np.sqrt(((ps[:, None] - pt[None]) ** 2).sum(axis=2)), 2) + 1
    m = _model("transport")
    x = [[m.add_var(obj=float(cost[i, j]), name=f"x{i}_{j}") for j in range(sinks)]
         for i in range(sources)]
    for i in range(sources):
        m.add_constr(qsum(x[i]) <= int(supply[i]), f"s{i}")
    for j in range(sinks):
        m.add_constr(qsum(x[i][j] for i in range(sources)) == int(demand[j]), f"d{j}")
    return m


def build_randlp(rng, cols, rows, density):
    """Random sparse LP with equality/inequality/ranged-like rows, free and
    bounded variables; feasible and bounded by construction (box on all
    variables that have a negative objective direction)."""
    x0 = rng.uniform(-5, 10, cols)
    lb = np.where(rng.random(cols) < 0.1, -np.inf, np.minimum(0, np.floor(x0)))
    ub = np.where(rng.random(cols) < 0.5, np.ceil(x0) + rng.integers(0, 20, cols), np.inf)
    R = _random_sparse_rows(rng, rows, cols, density)
    m = _model("randlp")
    x = [m.add_var(lb=float(lb[j]) if np.isfinite(lb[j]) else -float("inf"),
                  ub=float(ub[j]) if np.isfinite(ub[j]) else float("inf"),
                  obj=0.0, name=f"x{j}") for j in range(cols)]
    kinds = rng.choice(["E", "L", "G"], rows, p=[0.3, 0.35, 0.35])
    for i, (idx, vals) in enumerate(R):
        act = float(np.dot(vals, x0[idx]))
        expr = _lin(vals, [x[j] for j in idx])
        if kinds[i] == "E":
            m.add_constr(expr == round(act, 6), f"r{i}")
        elif kinds[i] == "L":
            m.add_constr(expr <= round(act + rng.uniform(0, 5), 6), f"r{i}")
        else:
            m.add_constr(expr >= round(act - rng.uniform(0, 5), 6), f"r{i}")
    # objective: random but force boundedness by boxing everything that is
    # unbounded in the improving direction
    cobj = np.round(rng.uniform(-10, 10, cols), 3)
    for j in range(cols):
        if cobj[j] < 0 and not np.isfinite(ub[j]):
            x[j].ub = float(np.ceil(x0[j]) + 50)
        if cobj[j] > 0 and not np.isfinite(lb[j]):
            x[j].lb = float(np.floor(x0[j]) - 50)
        x[j].obj = float(cobj[j])
    return m


# ------------------------------------------------- logical constraints
# Semi-continuous / semi-integer variables, SOS sets and indicator constraints.
# HiGHS reads semi-continuous variables (MPS SC bounds) but not SOS or
# indicators; SCIP reads all of them.

def build_semilot(rng, n, m, frac_int=0.3, tight=0.35):
    """Multi-resource production with minimum lot sizes: product j is either
    not made or made in [l_j, u_j] (semi-continuous; semi-integer for a
    fraction of the products, e.g. batches). Maximize profit."""
    lo = rng.integers(5, 25, n)
    up = lo + rng.integers(10, 60, n)
    A = rng.integers(1, 10, (m, n)) * (rng.random((m, n)) < 0.7)
    profit = A.sum(axis=0) * rng.uniform(0.6, 1.4, n) + rng.integers(1, 10, n)
    md = _model("semilot")
    x = []
    for j in range(n):
        vt = "SI" if rng.random() < frac_int else "SC"
        x.append(md.add_var(lb=float(lo[j]), ub=float(up[j]), vtype=vt, obj=round(float(profit[j]), 2),
                            name=f"x{j}"))
    for i in range(m):
        idx = np.nonzero(A[i])[0]
        if len(idx) == 0:
            continue
        cap = math.floor(tight * float((A[i] * up).sum()))
        md.add_constr(_lin(A[i, idx], [x[j] for j in idx]) <= cap, f"res{i}")
    md.sense = "max"
    return md


def build_pwlnet(rng, sources, sinks, K=4, frac_single=0.3, frac_minship=0.3):
    """Transportation with plant opening and economies of scale.

    Plant i: binary open y_i with fixed cost; indicator y_i = 0 -> no shipments
    from i; production cost piecewise-linear concave in the output (lambda
    formulation with an SOS2 set over K+1 breakpoints). Arcs with a minimum
    shipment are semi-continuous; single-sourced customers have an SOS1 set
    over their incoming arcs. Minimize total cost."""
    D = rng.integers(10, 40, sinks)
    S = rng.integers(int(D.sum() * 1.5 / sources) + 1, int(D.sum() * 2.5 / sources) + 2, sources)
    S = np.maximum(S, D.max())
    cost = rng.integers(2, 20, (sources, sinks))
    fixed = rng.integers(50, 200, sources)
    md = _model("pwlnet")
    y = [md.add_var(vtype="B", obj=float(fixed[i]), name=f"y{i}") for i in range(sources)]
    x = [[None] * sinks for _ in range(sources)]
    for i in range(sources):
        for j in range(sinks):
            ub = float(min(S[i], D[j]))
            if rng.random() < frac_minship:
                x[i][j] = md.add_var(lb=float(max(2, D[j] // 3)), ub=ub, vtype="SC", obj=float(cost[i, j]),
                                     name=f"x{i}_{j}")
            else:
                x[i][j] = md.add_var(ub=ub, obj=float(cost[i, j]), name=f"x{i}_{j}")
    for i in range(sources):
        # concave cost: decreasing marginal cost per segment
        bp = np.linspace(0, float(S[i]), K + 1)
        marg = np.sort(rng.uniform(2.0, 12.0, K))[::-1]
        cval = np.concatenate([[0.0], np.cumsum(marg * np.diff(bp))])
        lam = [md.add_var(ub=1.0, obj=round(float(cval[k]), 3), name=f"l{i}_{k}") for k in range(K + 1)]
        md.add_constr(_lin(np.ones(K + 1), lam) == 1, f"conv{i}")
        md.add_constr(_lin(np.ones(sinks), x[i]) - _lin(np.round(bp, 3), lam) == 0, f"out{i}")
        md.add_sos(lam, [float(k + 1) for k in range(K + 1)], sos_type=2, name=f"pwl{i}")
        md.add_indicator(y[i], 0, _lin(np.ones(sinks), x[i]) <= 0, name=f"closed{i}")
    for j in range(sinks):
        col = [x[i][j] for i in range(sources)]
        md.add_constr(_lin(np.ones(sources), col) >= float(D[j]), f"dem{j}")
        if rng.random() < frac_single:
            md.add_sos(col, [float(i + 1) for i in range(sources)], sos_type=1, name=f"single{j}")
    return md


def build_indsched(rng, n, pmax=20):
    """Single machine, release dates, total weighted completion time; the job
    order is modelled with indicator constraints (y_ij = 1 -> i before j,
    y_ij = 0 -> j before i). Start times have no upper bound, so no big-M
    is implied by the model."""
    p = rng.integers(1, pmax + 1, n)
    r = rng.integers(0, int(p.sum() * 0.6) + 1, n)
    w = rng.integers(1, 10, n)
    md = _model("indsched")
    s = [md.add_var(lb=float(r[j]), obj=float(w[j]), name=f"s{j}") for j in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            yij = md.add_var(vtype="B", name=f"y{i}_{j}")
            md.add_indicator(yij, 1, s[i] - s[j] <= -float(p[i]), name=f"b{i}_{j}")
            md.add_indicator(yij, 0, s[j] - s[i] <= -float(p[j]), name=f"a{i}_{j}")
    return md


BUILDERS = {
    "knapsack": build_knapsack,
    "mkp": build_mkp,
    "setcover": build_setcover,
    "setpack": build_setpack,
    "binpack": build_binpack,
    "gap": build_gap,
    "cflp": build_cflp,
    "netdesign": build_netdesign,
    "tsp": build_tsp,
    "cvrp": build_cvrp,
    "lotsize": build_lotsize,
    "jobshop": build_jobshop,
    "timeidx": build_timeidx,
    "randmilp": build_randmilp,
    "unbdmilp": lambda rng, **kw: build_randmilp(rng, ray=True, **kw),
    "pigeon": build_pigeon,
    "subsetsum": build_subsetsum,
    "transinf": build_transinf,
    "transport": build_transport,
    "randlp": build_randlp,
    "semilot": build_semilot,
    "pwlnet": build_pwlnet,
    "indsched": build_indsched,
}
