/* Tests of the C API (include/popso.h), compiled as C99. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "popso.h"

static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                     \
  do {                                                                  \
    ++checks;                                                           \
    if (!(cond)) {                                                      \
      ++failures;                                                       \
      fprintf(stderr, "%s:%d: FAILED: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                   \
  } while (0)

#define CHECK_OK(call) CHECK((call) == POPSO_OK)
#define CHECK_NEAR(a, b) CHECK(fabs((a) - (b)) <= 1e-6 * (1.0 + fabs(b)))

/* max 5x + 4y + 3z  s.t.  2x + 3y + z <= 5,  4x + y + 2z <= 11,  3x + 4y + 2z <= 8,  x, y, z >= 0 integer.
 * Optimum 13 at (2, 0, 1). */
static popso_model* build_small(void) {
  popso_model* m = popso_create();
  int x, y, z;
  CHECK(m != NULL);
  CHECK_OK(popso_set_model_name(m, "small"));
  CHECK_OK(popso_add_var(m, 0.0, POPSO_INFINITY, 5.0, POPSO_INTEGER, "x", &x));
  CHECK_OK(popso_add_var(m, 0.0, POPSO_INFINITY, 4.0, POPSO_INTEGER, "y", &y));
  CHECK_OK(popso_add_var(m, 0.0, POPSO_INFINITY, 3.0, POPSO_INTEGER, NULL, &z));
  CHECK(x == 0 && y == 1 && z == 2);
  {
    const int starts[4] = {0, 3, 6, 9};
    const int cols[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    const double vals[9] = {2, 3, 1, 4, 1, 2, 3, 4, 2};
    const double ub[3] = {5, 11, 8};
    CHECK_OK(popso_add_rows(m, 3, starts, cols, vals, NULL, ub, NULL));
  }
  CHECK_OK(popso_set_objective_sense(m, POPSO_MAXIMIZE));
  return m;
}

static void test_basics(void) {
  const char* v = popso_version();
  CHECK(v != NULL && strlen(v) > 0);
  CHECK(strcmp(popso_status_string(POPSO_STATUS_OPTIMAL), "OPTIMAL") == 0);
  CHECK(strcmp(popso_status_string(12345), "INVALID") == 0);
  /* Null handles are rejected, with a per-thread error message. */
  CHECK(popso_solve(NULL) == POPSO_ERROR_INVALID_ARGUMENT);
  CHECK(strlen(popso_last_error(NULL)) > 0);
  popso_free(NULL);
}

static void test_build_and_solve(void) {
  popso_model* m = build_small();
  double obj, bound, gap, t, x[3], lb, ub, c;
  long long nodes;
  int vt;
  CHECK(popso_num_vars(m) == 3);
  CHECK(popso_num_rows(m) == 3);
  CHECK(popso_num_nonzeros(m) == 9);
  CHECK(strcmp(popso_get_var_name(m, 2), "C2") == 0);
  CHECK(strcmp(popso_get_row_name(m, 0), "R0") == 0);
  CHECK(popso_get_var_name(m, 3) == NULL);
  CHECK(popso_find_var(m, "y") == 1);
  CHECK(popso_find_var(m, "nope") == -1);
  CHECK_OK(popso_get_var(m, 0, &lb, &ub, &c, &vt));
  CHECK(lb == 0.0 && isinf(ub) && c == 5.0 && vt == POPSO_INTEGER);
  CHECK_OK(popso_get_row(m, 1, &lb, &ub));
  CHECK(isinf(lb) && lb < 0 && ub == 11.0);
  CHECK(popso_get_objective_sense(m) == POPSO_MAXIMIZE);

  CHECK(popso_get_status(m) == POPSO_STATUS_NOT_SOLVED);
  CHECK(popso_get_objective(m, &obj) == POPSO_ERROR_NOT_SOLVED);
  CHECK(strlen(popso_last_error(m)) > 0);

  CHECK_OK(popso_solve(m));
  CHECK(strlen(popso_last_error(m)) == 0);
  CHECK(popso_get_status(m) == POPSO_STATUS_OPTIMAL);
  CHECK(popso_has_solution(m) == 1);
  CHECK_OK(popso_get_objective(m, &obj));
  CHECK_NEAR(obj, 13.0);
  CHECK_OK(popso_get_bound(m, &bound));
  CHECK_NEAR(bound, 13.0);
  CHECK_OK(popso_get_gap(m, &gap));
  CHECK(gap <= 1e-4);
  CHECK_OK(popso_get_node_count(m, &nodes));
  CHECK(nodes >= 0);
  CHECK_OK(popso_get_solve_time(m, &t));
  CHECK(t >= 0.0);
  CHECK(popso_get_solution(m, x, 2) == POPSO_ERROR_INVALID_ARGUMENT);
  CHECK_OK(popso_get_solution(m, x, 3));
  CHECK_NEAR(x[0], 2.0);
  CHECK_NEAR(x[1], 0.0);
  CHECK_NEAR(x[2], 1.0);

  /* A change discards the result; a tighter bound changes the optimum. */
  CHECK_OK(popso_set_var_bounds(m, 0, 0.0, 1.0));
  CHECK(popso_get_status(m) == POPSO_STATUS_NOT_SOLVED);
  CHECK_OK(popso_solve(m));
  CHECK_OK(popso_get_objective(m, &obj));
  CHECK(obj < 13.0 - 1e-6);

  /* Objective constant and coefficients. */
  CHECK_OK(popso_set_var_bounds(m, 0, 0.0, POPSO_INFINITY));
  CHECK_OK(popso_set_objective_constant(m, 100.0));
  CHECK_OK(popso_solve(m));
  CHECK_OK(popso_get_objective(m, &obj));
  CHECK_NEAR(obj, 113.0);
  {
    const int vars[1] = {0};
    const double vals[1] = {1.0};
    CHECK_OK(popso_set_objective(m, 1, vars, vals)); /* max x + 100 */
  }
  CHECK_OK(popso_solve(m));
  CHECK_OK(popso_get_objective(m, &obj));
  CHECK_NEAR(obj, 102.0); /* x <= 2 from 3x + 4y + 2z <= 8 */
  popso_free(m);
}

static void test_errors_and_params(void) {
  popso_model* m = popso_create();
  double v;
  const int bad_col[1] = {5};
  const double one[1] = {1.0};
  CHECK(popso_add_var(m, 0.0, 1.0, 0.0, 7, "x", NULL) == POPSO_ERROR_INVALID_ARGUMENT);
  CHECK(popso_num_vars(m) == 0);
  CHECK(popso_add_var(m, NAN, 1.0, 0.0, POPSO_CONTINUOUS, "x", NULL) == POPSO_ERROR_INVALID_ARGUMENT);
  CHECK_OK(popso_add_var(m, -3.0, 9.0, 1.0, POPSO_BINARY, "b", NULL));
  CHECK_OK(popso_get_var(m, 0, &v, NULL, NULL, NULL));
  CHECK(v == 0.0); /* binary bounds are clamped to [0, 1] */
  CHECK(popso_add_row(m, 1, bad_col, one, 0.0, 1.0, "r", NULL) == POPSO_ERROR_INVALID_ARGUMENT);
  CHECK(popso_num_rows(m) == 0);
  CHECK(popso_set_var_bounds(m, 3, 0.0, 1.0) == POPSO_ERROR_INVALID_ARGUMENT);
  CHECK(strstr(popso_last_error(m), "out of range") != NULL);

  CHECK(popso_set_param(m, "no_such_param", 1.0) == POPSO_ERROR_INVALID_ARGUMENT);
  CHECK(popso_set_param(m, "threads", 1.5) == POPSO_ERROR_INVALID_ARGUMENT);
  CHECK(popso_set_param(m, "deterministic", 2.0) == POPSO_ERROR_INVALID_ARGUMENT);
  CHECK_OK(popso_set_param(m, "threads", 2));
  CHECK_OK(popso_set_param(m, "time_limit", 30));
  CHECK_OK(popso_set_param(m, "mip_gap", 1e-6));
  CHECK_OK(popso_set_param(m, "seed", 7));
  CHECK_OK(popso_set_param(m, "deterministic", 1));
  CHECK_OK(popso_set_param(m, "node_limit", 1000));
  CHECK_OK(popso_set_param(m, "mem_limit", 512));
  CHECK_OK(popso_set_param(m, "presolve", 0));
  CHECK_OK(popso_get_param(m, "threads", &v));
  CHECK(v == 2.0);
  CHECK_OK(popso_get_param(m, "verbosity", &v));
  CHECK(v == 0.0); /* the library is silent by default */
  CHECK_OK(popso_get_param(m, "mip_gap", &v));
  CHECK(v == 1e-6);
  CHECK_OK(popso_get_param(m, "presolve", &v));
  CHECK(v == 0.0);
  CHECK(popso_read_mps(m, "/nonexistent/dir/model.mps") == POPSO_ERROR_IO);
  CHECK(popso_num_vars(m) == 1); /* unchanged */
  CHECK(popso_write_solution(m, "x.sol") == POPSO_ERROR_NOT_SOLVED);
  popso_free(m);
}

static void test_infeasible_and_lp(void) {
  popso_model* m = popso_create();
  const int cols[2] = {0, 1};
  const double ones[2] = {1.0, 1.0};
  double obj, x[2];
  CHECK_OK(popso_add_vars(m, 2, NULL, NULL, NULL, NULL, NULL));
  CHECK_OK(popso_add_row(m, 2, cols, ones, 3.0, POPSO_INFINITY, "lo", NULL));
  CHECK_OK(popso_add_row(m, 2, cols, ones, -POPSO_INFINITY, 2.0, "hi", NULL));
  CHECK_OK(popso_solve(m));
  CHECK(popso_get_status(m) == POPSO_STATUS_INFEASIBLE);
  CHECK(popso_has_solution(m) == 0);
  CHECK(popso_get_objective(m, &obj) == POPSO_ERROR_NO_SOLUTION);
  popso_free(m);

  /* LP: min x + y  s.t.  x + 2y >= 2,  3x + y >= 3  ->  (0.8, 0.6), objective 1.4. */
  m = popso_create();
  {
    const double c[2] = {1.0, 1.0};
    const char* names[2] = {"x", "y"};
    const double a1[2] = {1.0, 2.0}, a2[2] = {3.0, 1.0};
    CHECK_OK(popso_add_vars(m, 2, NULL, NULL, c, NULL, names));
    CHECK_OK(popso_add_row(m, 2, cols, a1, 2.0, POPSO_INFINITY, NULL, NULL));
    CHECK_OK(popso_add_row(m, 2, cols, a2, 3.0, POPSO_INFINITY, NULL, NULL));
  }
  CHECK_OK(popso_solve(m));
  CHECK(popso_get_status(m) == POPSO_STATUS_OPTIMAL);
  CHECK_OK(popso_get_objective(m, &obj));
  CHECK_NEAR(obj, 1.4);
  CHECK_OK(popso_get_solution(m, x, 2));
  CHECK_NEAR(x[0], 0.8);
  CHECK_NEAR(x[1], 0.6);
  popso_free(m);
}

static void test_files_and_start(void) {
  const char* mps = POPSO_BINARY_DIR "/c_api_test_small.mps";
  const char* sol = POPSO_BINARY_DIR "/c_api_test_small.sol";
  popso_model* m = build_small();
  popso_model* r;
  double obj;
  CHECK_OK(popso_write_mps(m, mps));
  CHECK_OK(popso_solve(m));
  CHECK_OK(popso_write_solution(m, sol));
  popso_free(m);

  r = popso_create();
  CHECK_OK(popso_read_mps(r, mps));
  CHECK(popso_num_vars(r) == 3 && popso_num_rows(r) == 3);
  CHECK(popso_get_objective_sense(r) == POPSO_MAXIMIZE);
  CHECK(popso_find_var(r, "x") == 0);
  /* A complete start read from the solution file. */
  CHECK_OK(popso_read_start(r, sol));
  CHECK_OK(popso_solve(r));
  CHECK(popso_get_status(r) == POPSO_STATUS_OPTIMAL);
  CHECK_OK(popso_get_objective(r, &obj));
  CHECK_NEAR(obj, 13.0);
  /* A partial start (x = 2), and a start that cannot be completed (x = 2, y = 2 violates row 0). */
  {
    const int vars[2] = {0, 1};
    const double vals[2] = {2.0, 2.0};
    CHECK_OK(popso_set_start(r, 1, vars, vals));
    CHECK_OK(popso_solve(r));
    CHECK_OK(popso_get_objective(r, &obj));
    CHECK_NEAR(obj, 13.0);
    CHECK_OK(popso_set_start(r, 2, vars, vals));
    CHECK_OK(popso_solve(r));
    CHECK_OK(popso_get_objective(r, &obj));
    CHECK_NEAR(obj, 13.0);
  }
  CHECK(popso_set_start(r, 2, NULL, NULL) == POPSO_ERROR_INVALID_ARGUMENT);
  CHECK_OK(popso_clear_start(r));
  CHECK(popso_read_start(r, POPSO_BINARY_DIR "/c_api_test_missing.sol") == POPSO_ERROR_IO);
  popso_free(r);

  /* A generated benchmark instance. */
  r = popso_create();
  CHECK_OK(popso_read_mps(r, POPSO_SOURCE_DIR "/benchmarks/instances/knapsack_tiny_1.mps"));
  CHECK_OK(popso_set_param(r, "threads", 1));
  CHECK_OK(popso_solve(r));
  CHECK(popso_get_status(r) == POPSO_STATUS_OPTIMAL);
  popso_free(r);
  remove(mps);
  remove(sol);
}

int main(void) {
  test_basics();
  test_build_and_solve();
  test_errors_and_params();
  test_infeasible_and_lp();
  test_files_and_start();
  if (failures) {
    fprintf(stderr, "c_api: %d of %d checks failed\n", failures, checks);
    return 1;
  }
  printf("c_api: all %d checks passed\n", checks);
  return 0;
}
