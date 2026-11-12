"""Tests of the popso Python bindings (standard library unittest).

Run from the repository root after building:  python -m unittest discover -s python/tests
(the package is imported from python/; the library is found in build/ or via POPSO_LIBRARY).
"""

import math
import os
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

import popso  # noqa: E402
INSTANCES = os.path.join(ROOT, "benchmarks", "instances")


def small_model():
    """max 5x + 4y + 3z s.t. 2x + 3y + z <= 5, 4x + y + 2z <= 11, 3x + 4y + 2z <= 8; optimum 13 at (2, 0, 1)."""
    m = popso.Model("small")
    x, y, z = (m.add_var(vtype=popso.INTEGER, name=n) for n in "xyz")
    m.add_constr(2 * x + 3 * y + z <= 5, name="c0")
    m.add_constr({x: 4, y: 1, z: 2}, ub=11, name="c1")
    m.add_constr(([0, 1, 2], [3, 4, 2]), ub=8)
    m.set_objective(5 * x + 4 * y + 3 * z, sense=popso.MAXIMIZE)
    m.params.threads = 1
    return m, x, y, z


class TestExpressions(unittest.TestCase):
    def test_operators(self):
        m = popso.Model()
        x, y = m.add_var(name="x"), m.add_var(name="y")
        e = 2 * x + y - 3 + x / 2 - (-y)
        self.assertEqual(e.terms, {x.index: 2.5, y.index: 2.0})
        self.assertEqual(e.constant, -3.0)
        self.assertEqual(sum([x, y, 1]).terms, {0: 1.0, 1: 1.0})
        c = x + 2 * y <= 5 + x
        self.assertEqual((c.lb, c.ub), (-popso.INF, 5.0))
        self.assertEqual(c.expr.terms, {0: 0.0, 1: 2.0})
        c = 3 >= x  # reflected: x <= 3
        self.assertEqual((c.lb, c.ub), (-popso.INF, 3.0))
        c = x - y == 1
        self.assertEqual((c.lb, c.ub), (1.0, 1.0))

    def test_identity_semantics(self):
        m = popso.Model()
        x, y = m.add_var(), m.add_var()
        self.assertTrue(x in [y, x])
        self.assertFalse(x in [y])
        self.assertEqual({x: 1, y: 2}[m.var(0)], 1)
        self.assertNotEqual(x, y)
        with self.assertRaises(TypeError):
            bool(x + y <= 1)
        self.assertIsInstance(x * y, popso.QuadExpr)
        with self.assertRaises(TypeError):
            x * y * x  # degree 3

    def test_mixing_models_is_rejected(self):
        a, b = popso.Model(), popso.Model()
        x, y = a.add_var(), b.add_var()
        with self.assertRaises(ValueError):
            x + y


class TestModel(unittest.TestCase):
    def test_version(self):
        self.assertTrue(popso.version())
        self.assertEqual(popso.__version__, popso.version())

    def test_solve_mip(self):
        m, x, y, z = small_model()
        self.assertEqual((m.num_vars, m.num_constrs, m.num_nonzeros), (3, 3, 9))
        r = m.solve()
        self.assertEqual(r.status, "OPTIMAL")
        self.assertTrue(r.optimal and r.has_solution)
        self.assertAlmostEqual(r.objective, 13.0)
        self.assertAlmostEqual(r.bound, 13.0)
        self.assertLessEqual(r.gap, 1e-4)
        self.assertAlmostEqual(r[x], 2.0)
        self.assertAlmostEqual(r["y"], 0.0)
        self.assertAlmostEqual(r[2], 1.0)
        self.assertAlmostEqual(r[5 * x + 3 * z], 13.0)
        self.assertEqual(set(r.as_dict()), {"x", "y", "z"})
        self.assertGreaterEqual(r.nodes, 0)
        self.assertGreaterEqual(r.time, 0.0)

    def test_lp_and_attributes(self):
        m = popso.Model()
        x = m.add_var(lb=-popso.INF, name="x", obj=1)
        y = m.add_var(name="y", obj=1)
        m.add_constr(x + 2 * y, lb=2)
        c = m.add_constr(3 * x + y >= 3, name="second")
        self.assertEqual(c.name, "second")
        self.assertEqual((c.lb, c.ub), (3.0, popso.INF))
        self.assertEqual(x.lb, -popso.INF)
        x.lb = 0
        self.assertEqual(x.lb, 0.0)
        self.assertEqual(m.sense, popso.MINIMIZE)
        r = m.solve()
        self.assertEqual(r.status, "OPTIMAL")
        self.assertAlmostEqual(r.objective, 1.4)
        self.assertAlmostEqual(r[x], 0.8)
        self.assertAlmostEqual(r[y], 0.6)
        for method in ("barrier", "simplex", "auto"):
            r = m.solve(lp_method=method)
            self.assertEqual(r.status, "OPTIMAL")
            self.assertAlmostEqual(r.objective, 1.4)
            self.assertAlmostEqual(r[x], 0.8)
        self.assertEqual(m.get_param("lp_method"), 0)
        with self.assertRaises(ValueError):
            m.set_param("lp_method", "interior")
        y.obj = 10
        x.vtype = "I"
        self.assertEqual(x.vtype, popso.INTEGER)
        r = m.solve()
        self.assertAlmostEqual(r.objective, 2.0)  # x = 2, y = 0

    def test_add_vars_bulk(self):
        m = popso.Model()
        xs = m.add_vars(4, ub=[1, 2, 3, 4], obj=-1.0, vtype="I", names=["a", "b", "c", "d"])
        m.add_constr(sum(xs) <= 7)
        r = m.solve()
        self.assertAlmostEqual(r.objective, -7.0)
        self.assertEqual(m.var("c").index, 2)
        self.assertEqual(xs[3].ub, 4.0)

    def test_infeasible(self):
        m = popso.Model()
        x = m.add_var(vtype=popso.BINARY)
        m.add_constr(x >= 2)
        r = m.solve()
        self.assertEqual(r.status, "INFEASIBLE")
        self.assertIsNone(r.objective)
        self.assertFalse(r.has_solution)
        with self.assertRaises(ValueError):
            r[x]

    def test_errors(self):
        m = popso.Model()
        with self.assertRaises(popso.PopsoError) as cm:
            m.result()
        self.assertEqual(cm.exception.code, 5)
        with self.assertRaises(ValueError):
            m.add_var(vtype="Q")
        with self.assertRaises(popso.PopsoError):
            m.add_constr(([3], [1.0]), ub=1)
        with self.assertRaises(AttributeError):
            m.params.no_such_param = 1
        with self.assertRaises(popso.PopsoError):
            m.set_param("threads", -1)
        with self.assertRaises(popso.PopsoError) as cm:
            m.read_mps("/nonexistent/model.mps")
        self.assertEqual(cm.exception.code, 3)
        with self.assertRaises(KeyError):
            m.var("missing")

    def test_params(self):
        m = popso.Model()
        m.params.time_limit = 5
        m.params.deterministic = True
        m.params.mip_gap = 1e-6
        self.assertEqual(m.params.time_limit, 5.0)
        self.assertIs(m.params.deterministic, True)
        self.assertEqual(m.params.verbosity, 0)
        self.assertEqual(m.params.node_limit, -1)
        self.assertEqual(m.get_param("mip_gap"), 1e-6)
        self.assertIn("racing", repr(m.params))

    def test_files_and_start(self):
        m, x, y, z = small_model()
        with tempfile.TemporaryDirectory() as d:
            mps, sol = os.path.join(d, "m.mps"), os.path.join(d, "m.sol")
            m.write_mps(mps)
            m.solve()
            m.write_solution(sol)
            with open(sol) as f:
                self.assertIn("# Status = OPTIMAL", f.read())
            r = popso.read_mps(mps)
            self.assertEqual(r.num_vars, 3)
            self.assertEqual(r.sense, popso.MAXIMIZE)
            r.read_start(sol)
            self.assertAlmostEqual(r.solve(threads=1).objective, 13.0)
            r.set_start({"x": 2})
            self.assertAlmostEqual(r.solve().objective, 13.0)
            r.set_start([2, None, math.nan])
            self.assertAlmostEqual(r.solve().objective, 13.0)
            r.clear_start()
            with self.assertRaises(popso.PopsoError):
                r.set_start([1, 2])  # wrong length

    def test_benchmark_instance(self):
        path = os.path.join(INSTANCES, "knapsack_small_1.mps")
        if not os.path.exists(path):
            self.skipTest("benchmark instances not available")
        m = popso.read_mps(path)
        r = m.solve(threads=1)
        self.assertEqual(r.status, "OPTIMAL")
        # The optimum as a start, and a search stopped after one node, keeps it.
        m.set_start(r.x)
        r2 = m.solve(node_limit=1)
        self.assertAlmostEqual(r2.objective, r.objective, delta=1e-6 * abs(r.objective))

    def test_threads_separate_models(self):
        results = [None] * 4

        def work(k):
            m, *_ = small_model()
            results[k] = m.solve().objective

        threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual([round(v, 6) for v in results], [13.0] * 4)

    def test_context_manager(self):
        with popso.Model("ctx") as m:
            m.add_var(obj=1)
            self.assertEqual(m.name, "ctx")
        self.assertIsNone(m._h)


class TestLogical(unittest.TestCase):
    def test_semicontinuous_sos_indicator(self):
        # Facility: open costs 5; closed -> no production (indicator); production semi-continuous in {0} u [3, 8];
        # demand 4 from production (cost 1) or purchases (cost 3). Open: 5 + 4 = 9 < 12.
        m = popso.Model("logic")
        y = m.add_var(vtype=popso.BINARY, obj=5, name="y")
        p = m.add_var(lb=3, ub=8, vtype=popso.SEMICONTINUOUS, obj=1, name="p")
        q = m.add_var(obj=3, name="q")
        m.add_constr(p + q >= 4, name="dem")
        self.assertEqual(m.add_indicator(y, 0, p <= 0, name="closed"), 0)
        self.assertEqual(m.num_indicators, 1)
        self.assertEqual(p.vtype, popso.SEMICONTINUOUS)
        r = m.solve(threads=1)
        self.assertEqual(r.status, "OPTIMAL")
        self.assertAlmostEqual(r.objective, 9.0, places=6)
        self.assertAlmostEqual(r[p], 4.0, places=6)
        # Demand 2: production must be 0 or >= 3: buy 2 (6) or open and make 3 (8) -> 6.
        m.constrs[0].set_bounds(2, popso.INF)
        r = m.solve()
        self.assertAlmostEqual(r.objective, 6.0, places=6)
        self.assertAlmostEqual(r[p], 0.0, places=6)

    def test_sos_and_lp_file(self):
        m = popso.Model()
        xs = m.add_vars(3, ub=2, obj=[3, 2, 1], names=["a", "b", "c"])
        m.add_constr(xs[0] + xs[1] + xs[2] <= 3)
        m.sense = popso.MAXIMIZE
        self.assertEqual(m.add_sos(xs, weights=[1, 2, 3], type=popso.SOS2), 0)
        self.assertEqual(m.num_sos, 1)
        r = m.solve(threads=1)
        self.assertAlmostEqual(r.objective, 3 * 2 + 2 * 1, places=6)  # a = 2, b = 1 (neighbours)
        with self.assertRaises(popso.PopsoError):
            m.add_sos(xs, type=3)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.lp")
            with open(path, "w") as f:
                f.write("max\n obj: 3 a + 2 b + c\nst\n c1: a + b + c <= 3\n i: z = 1 -> a <= 0.5\nbounds\n"
                        " a <= 2\n b <= 2\n c <= 2\nsos\n s: S1:: a:1 b:2 c:3\nend\n")
            m2 = popso.read_lp(path)
            self.assertEqual((m2.num_sos, m2.num_indicators), (1, 1))
            self.assertEqual(m2.var("z").vtype, popso.BINARY)
            r2 = m2.solve(threads=1)
            self.assertAlmostEqual(r2.objective, 6.0, places=6)  # a = 2 with z = 0



class TestQuadratic(unittest.TestCase):
    def test_quadexpr_operators(self):
        m = popso.Model()
        x, y = m.add_var(name="x"), m.add_var(name="y")
        q = 2 * x * x + 3 * x * y - y + 1
        self.assertIsInstance(q, popso.QuadExpr)
        self.assertEqual(q.quad, {(0, 0): 2.0, (1, 0): 3.0})
        self.assertEqual(q.linear.terms, {1: -1.0})
        self.assertEqual(q.constant, 1.0)
        p = (x + 1) * (y - 2)  # x y - 2 x + y - 2
        self.assertEqual(p.quad, {(1, 0): 1.0})
        self.assertEqual(p.linear.terms, {0: -2.0, 1: 1.0})
        self.assertEqual(p.constant, -2.0)
        self.assertEqual((x ** 2 - q).quad, {(0, 0): -1.0, (1, 0): -3.0})
        self.assertEqual((q / 2).quad[(0, 0)], 1.0)
        with self.assertRaises(TypeError):
            x * y == 1  # noqa: B015 (a quadratic equality)
        with self.assertRaises(TypeError):
            x * y * x  # noqa: B018 (degree 3)

    def test_quadratic_constraints(self):
        m = popso.Model()
        x = m.add_var(lb=-10, ub=10, name="x")
        y = m.add_var(lb=-10, ub=10, name="y")
        disk = m.add_constr(x * x + y ** 2 <= 2, name="disk")
        self.assertEqual(m.get_qrow_terms(disk), [(0, 0, 1.0), (1, 1, 1.0)])
        self.assertEqual(m.get_qrow_terms(0), [(0, 0, 1.0), (1, 1, 1.0)])
        m.set_objective(x + y, sense=popso.MAXIMIZE)
        r = m.solve(threads=1)
        self.assertEqual(r.status, "OPTIMAL")
        self.assertAlmostEqual(r.objective, 2.0, places=5)
        self.assertLessEqual(r[x * x + y ** 2], 2.0 + 1e-6)
        self.assertAlmostEqual(r.dual(disk), 0.5, places=5)  # maximizing: (1, 1) = 0.5 grad(x^2 + y^2)
        # An epigraph with the quadratic on the right: t >= (z - 2.6)^2, z integer -> z = 3, t = 0.16.
        e = popso.Model()
        z = e.add_var(lb=0, ub=10, vtype=popso.INTEGER, name="z")
        t = e.add_var(lb=-100, ub=100, name="t")
        row = e.add_constr(t >= (z - 2.6) * (z - 2.6))  # reflected: (z - 2.6)^2 - t <= 0
        self.assertEqual(e.get_qrow_terms(row), [(0, 0, 1.0)])
        e.set_objective(t)
        r = e.solve(threads=1)
        self.assertEqual(r.status, "OPTIMAL")
        self.assertAlmostEqual(r[z], 3.0, places=6)
        self.assertAlmostEqual(r.objective, 0.16, places=6)
        # Nonconvex (a convex quadratic bounded from below): spatial branch-and-bound (v0.10); a = b = -1.
        n = popso.Model()
        a, b = n.add_var(lb=-1, ub=1), n.add_var(lb=-1, ub=1)
        n.add_constr(a * a + b * b >= 1)
        n.set_objective(a + b)
        rn = n.solve(threads=1)
        self.assertEqual(rn.status, "OPTIMAL")
        self.assertAlmostEqual(rn.objective, -2.0, places=5)
        with self.assertRaises(IndexError):
            n.get_qrow_terms(5)

    def test_qp_and_miqp(self):
        m = popso.Model()
        x = m.add_var(lb=-10, ub=10, name="x")
        y = m.add_var(lb=-10, ub=10, name="y")
        m.add_constr(x + y >= 1)
        # min (x - 1)^2 + (y - 2)^2 + x y  ->  x^2 + y^2 + x y - 2 x - 4 y + 5
        obj = (x - 1) ** 2 + (y - 2) ** 2 + x * y
        m.set_objective(obj)
        self.assertEqual(sorted(m.get_q()), [(0, 0, 2.0), (1, 0, 1.0), (1, 1, 2.0)])
        r = m.solve(threads=1)
        self.assertEqual(r.status, "OPTIMAL")
        # Unconstrained optimum: 2x + y = 2, x + 2y = 4 -> x = 0, y = 2 (feasible).
        self.assertAlmostEqual(r[x], 0.0, places=5)
        self.assertAlmostEqual(r[y], 2.0, places=5)
        self.assertAlmostEqual(r.objective, r[obj], places=7)
        self.assertAlmostEqual(r.objective, 1.0, places=6)
        # Duals: the row is inactive; x - y <= -2.5 binds (x = -0.25, y = 2.25, dual -0.25).
        self.assertEqual(len(r.duals), 1)
        self.assertAlmostEqual(r.duals[0], 0.0, places=6)
        self.assertAlmostEqual(r.reduced_cost(x), 0.0, places=6)
        cap = m.add_constr(x - y <= -2.5, name="cap")
        r = m.solve(threads=1)
        self.assertAlmostEqual(r.objective, 1.0625, places=6)
        self.assertAlmostEqual(r.dual(cap), -0.25, places=6)
        self.assertAlmostEqual(r.duals[0], 0.0, places=6)
        self.assertAlmostEqual(r.reduced_cost("y"), 0.0, places=6)
        m.sense = popso.MAXIMIZE
        m.set_objective(-1 * obj)
        r = m.solve(threads=1)
        self.assertAlmostEqual(r.objective, -1.0625, places=6)
        self.assertAlmostEqual(r.dual(cap), 0.25, places=6)
        mi = popso.Model()
        a = mi.add_var(lb=-5, ub=5, vtype=popso.INTEGER, name="a")
        b = mi.add_var(lb=-5, ub=5, name="b")
        mi.set_objective((a - 0.6) ** 2 + (b - a - 0.3) ** 2)
        r2 = mi.solve(threads=1)
        self.assertEqual(r2.status, "OPTIMAL")
        self.assertIsNone(r2.duals)
        with self.assertRaises(ValueError):
            r2.reduced_cost(a)
        self.assertAlmostEqual(r2[a], 1.0, places=6)
        self.assertAlmostEqual(r2.objective, 0.16, places=6)
        # Replacing the objective by a linear one removes Q.
        mi.set_objective(a + b)
        self.assertEqual(mi.get_q(), [])

    def test_nonconvex_and_files(self):
        m = popso.Model()
        x = m.add_var(lb=-1, ub=1, name="x")
        m.set_objective(-1 * x * x)
        # Not convex: spatial branch-and-bound (v0.10); the minimum of -x^2 on [-1, 1] is -1.
        r0 = m.solve(threads=1)
        self.assertEqual(r0.status, "OPTIMAL")
        self.assertAlmostEqual(r0.objective, -1.0, places=5)
        m.sense = popso.MAXIMIZE
        r = m.solve(threads=1)
        self.assertEqual(r.status, "OPTIMAL")
        self.assertAlmostEqual(r.objective, 0.0, places=6)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "q.mps")
            m.write_mps(path)
            m2 = popso.read_mps(path)
            self.assertEqual(m2.get_q(), [(0, 0, -2.0)])
            lp = os.path.join(d, "q.lp")
            with open(lp, "w") as f:
                f.write("min\n obj: x + [ x ^ 2 + 4 x * y + 6 y ^ 2 ] / 2\nst\n c: x + y >= 1\nend\n")
            m3 = popso.read_lp(lp)
            self.assertEqual(sorted(m3.get_q()), [(0, 0, 1.0), (1, 0, 2.0), (1, 1, 6.0)])
            self.assertEqual(m3.solve(threads=1).status, "OPTIMAL")


def hard_model():
    path = os.path.join(INSTANCES, "cflp_hard_1.mps")
    if not os.path.exists(path):
        return None
    m = popso.read_mps(path)
    m.params.threads = 2
    return m


class TestDualsAndControl(unittest.TestCase):
    def lp(self):
        """min -x - 2y s.t. x + y <= 4 (c0), x - y >= -2 (c1), 0 <= x <= 3, y >= 0: optimum -7 at (1, 3)."""
        m = popso.Model("lp")
        x, y = m.add_var(ub=3, name="x"), m.add_var(name="y")
        c0 = m.add_constr(x + y <= 4, name="c0")
        c1 = m.add_constr(x - y >= -2, name="c1")
        m.set_objective(-x - 2 * y)
        return m, x, y, c0, c1

    def test_lp_duals(self):
        m, x, y, c0, c1 = self.lp()
        for presolve in (True, False):
            r = m.solve(presolve=presolve, threads=1)
            self.assertEqual(r.status, "OPTIMAL")
            self.assertAlmostEqual(r.objective, -7.0)
            # Stationarity d = c - A^T y; c0 is active at its upper bound (dual <= 0 when minimizing),
            # c1 at its lower bound (dual >= 0).
            self.assertAlmostEqual(r.dual(c0), -1.5)
            self.assertAlmostEqual(r.dual(c1), 0.5)
            self.assertAlmostEqual(r.reduced_cost(x), 0.0)
            self.assertAlmostEqual(r.reduced_cost("y"), 0.0)
            self.assertFalse(r.fixed_mip_duals)
            self.assertEqual(sorted(r.var_basis + r.constr_basis).count(popso.BASIC), 2)
            self.assertEqual(r.constr_basis, [popso.AT_UPPER, popso.AT_LOWER])
        # Maximization flips the signs.
        m.set_objective(x + 2 * y, sense=popso.MAXIMIZE)
        r = m.solve()
        self.assertAlmostEqual(r.dual(c0), 1.5)
        self.assertAlmostEqual(r.dual(c1), -0.5)
        # duals=0: none.
        self.assertIsNone(m.solve(duals=0).duals)

    def test_fixed_mip_duals(self):
        m, x, y, z = small_model()
        self.assertIsNone(m.solve().duals)  # MILP: only with duals=2
        r = m.solve(duals=2)
        self.assertEqual(r.status, "OPTIMAL")
        self.assertTrue(r.fixed_mip_duals)
        # All variables are integers: every row dual is 0, the reduced costs are the objective.
        self.assertEqual(r.duals, [0.0, 0.0, 0.0])
        self.assertEqual(r.reduced_costs, [5.0, 4.0, 3.0])
        # A continuous variable w: max ... + 2w, w <= 1 + x (with x fixed at 2: w <= 3).
        w = m.add_var(name="w")
        c = m.add_constr(w - x <= 1)
        m.set_objective(5 * x + 4 * y + 3 * z + 2 * w, sense=popso.MAXIMIZE)
        r = m.solve(duals=2)
        self.assertTrue(r.fixed_mip_duals)
        self.assertAlmostEqual(r.dual(c), 2.0)
        self.assertAlmostEqual(r.reduced_cost(w), 0.0)
        self.assertAlmostEqual(r.reduced_cost(x), 5.0 + 2.0)  # d = c - A^T y = 5 - (-1)(2)

    def test_interrupt_from_thread(self):
        m = hard_model()
        if m is None:
            self.skipTest("benchmark instances not available")
        threading.Timer(0.5, m.interrupt).start()
        r = m.solve()
        self.assertEqual(r.status, "INTERRUPTED")
        self.assertLess(r.time, 3.0)
        # An interrupt while no solve runs does not affect the next solve.
        m.interrupt()
        r = m.solve(node_limit=1)
        self.assertEqual(r.status, "NODE_LIMIT")

    def test_keyboard_interrupt(self):
        m = hard_model()
        if m is None:
            self.skipTest("benchmark instances not available")
        import _thread
        threading.Timer(0.5, _thread.interrupt_main).start()
        r = m.solve()  # Ctrl-C stops the solve; no KeyboardInterrupt escapes
        self.assertEqual(r.status, "INTERRUPTED")
        self.assertLess(r.time, 3.0)

    def test_error_while_waiting_stops_the_helper_thread(self):
        # Something other than KeyboardInterrupt escaping the main thread's wait: the native solve
        # is interrupted and its helper thread finished before solve() raises (else the model could
        # be freed under the running solve).
        m = hard_model()
        if m is None:
            self.skipTest("benchmark instances not available")
        import popso.model as pm
        events = []

        class FailingEvent(threading.Event):
            def __init__(self):
                super().__init__()
                events.append(self)
                self.calls = 0

            def wait(self, timeout=None):
                done = super().wait(timeout)
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError("injected")
                return done

        class Shim:
            Event = FailingEvent

            def __getattr__(self, name):
                return getattr(threading, name)

        saved = pm.threading
        pm.threading = Shim()
        try:
            with self.assertRaises(RuntimeError):
                m.solve(time_limit=60)
        finally:
            pm.threading = saved
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].is_set())  # the helper thread's solve has returned
        r = m.solve(node_limit=1)  # the model stays usable
        self.assertEqual(r.status, "NODE_LIMIT")

    def test_incumbent_callback(self):
        m = hard_model()
        if m is None:
            self.skipTest("benchmark instances not available")
        seen = []
        m.set_incumbent_callback(lambda obj, bound, x: seen.append((obj, bound, list(x))) or len(seen) >= 2)
        r = m.solve(deterministic=True)
        self.assertEqual(r.status, "INTERRUPTED")
        self.assertEqual(len(seen), 2)
        self.assertLess(seen[1][0], seen[0][0])  # minimization: strictly improving
        for obj, bound, x in seen:
            self.assertLessEqual(bound, obj + 1e-6)
            self.assertEqual(len(x), m.num_vars)
        # The search may still improve the incumbent before it stops (not reported after the stop request).
        self.assertLessEqual(r.objective, seen[-1][0] + 1e-6 * abs(seen[-1][0]))
        # Deterministic mode: the same sequence again.
        seen2 = list(seen)
        seen.clear()
        m.solve(deterministic=True)
        self.assertEqual([s[0] for s in seen], [s[0] for s in seen2[:len(seen)]])

        # An exception stops the solve and is re-raised; the model stays usable.
        def boom(obj, bound, x):
            raise RuntimeError("boom")

        m.set_incumbent_callback(boom)
        with self.assertRaises(RuntimeError):
            m.solve()
        m.set_incumbent_callback(None)
        small, *_ = small_model()
        calls = []
        small.set_incumbent_callback(lambda obj, bound, x: calls.append(obj))
        self.assertAlmostEqual(small.solve().objective, 13.0)
        self.assertGreaterEqual(len(calls), 1)
        self.assertAlmostEqual(calls[-1], 13.0)
        self.assertEqual(calls, sorted(calls))  # maximization: increasing

    def test_incumbent_callback_removed_during_solve(self):
        # Removing (or replacing) the callback from inside it must not free the ctypes object the
        # running solve still calls (it crashed with a segmentation fault).
        import gc
        path = os.path.join(INSTANCES, "setcover_medium_1.mps")
        if not os.path.exists(path):
            self.skipTest("benchmark instances not available")
        m = popso.read_mps(path)
        calls = []

        def cb(obj, bound, x):
            calls.append(obj)
            if len(calls) == 1:
                m.set_incumbent_callback(None)
                gc.collect()
            return False

        m.set_incumbent_callback(cb)
        r = m.solve(threads=2, deterministic=True, time_limit=20)
        self.assertEqual(r.status, "OPTIMAL")
        self.assertGreater(len(calls), 1)  # the running solve keeps its callback
        calls.clear()
        m.solve(threads=2, deterministic=True, time_limit=20)
        self.assertEqual(calls, [])  # removed for the next solve

    def test_gzip_input(self):
        import gzip
        path = os.path.join(INSTANCES, "knapsack_small_1.mps")
        if not os.path.exists(path):
            self.skipTest("benchmark instances not available")
        with tempfile.TemporaryDirectory() as d:
            gz = os.path.join(d, "k.mps.gz")
            with open(path, "rb") as f, gzip.open(gz, "wb") as g:
                g.write(f.read())
            a, b = popso.read_mps(path), popso.read_mps(gz)
            self.assertEqual((a.num_vars, a.num_constrs), (b.num_vars, b.num_constrs))
            self.assertAlmostEqual(a.solve(threads=1).objective, b.solve(threads=1).objective)


if __name__ == "__main__":
    unittest.main()
