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
        with self.assertRaises(TypeError):
            x * y  # nonlinear

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


if __name__ == "__main__":
    unittest.main()
