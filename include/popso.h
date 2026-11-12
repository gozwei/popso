/*
 * popso C API: build or read an LP, MILP, QP or MIQP model, solve it and query the result.
 *
 *   popso_model* m = popso_create();
 *   int x, y;
 *   popso_add_var(m, 0.0, 10.0, 1.0, POPSO_INTEGER, "x", &x);
 *   popso_add_var(m, 0.0, POPSO_INFINITY, 2.0, POPSO_CONTINUOUS, "y", &y);
 *   int cols[2] = {x, y};
 *   double vals[2] = {1.0, 1.0};
 *   popso_add_row(m, 2, cols, vals, 3.5, POPSO_INFINITY, "c1", NULL);
 *   popso_solve(m);
 *   double obj;
 *   if (popso_get_status(m) == POPSO_STATUS_OPTIMAL) popso_get_objective(m, &obj);
 *   popso_free(m);
 *
 * Conventions
 *  - Functions that can fail return an error code (POPSO_OK = 0 on success);
 *    popso_last_error() then describes the failure. Simple model queries
 *    return their value directly.
 *  - Variables and rows are numbered from 0 in the order they were added (or
 *    read). Infinite bounds are given as +-POPSO_INFINITY or IEEE infinity
 *    (any |value| >= 1e30 counts as infinite); queries return IEEE infinity.
 *  - A handle may be used by one thread at a time; separate handles may be
 *    used from different threads concurrently. The exception is
 *    popso_interrupt, which may be called from any thread (or a signal
 *    handler) while another thread solves. Note that the memory limit is
 *    also checked against the resident size of the whole process.
 *  - Changing the model discards the last result.
 *  - The objective may have a quadratic part 1/2 x^T Q x (popso_set_q).
 *    Convex (Q positive semidefinite when minimizing, negative semidefinite
 *    when maximizing): continuous QPs are solved by the barrier method (no
 *    basis); with integer variables by branch-and-bound on QP relaxations.
 *  - Rows may have quadratic terms (popso_add_qrow_terms). Convex rows (one
 *    finite side, the quadratic part positive semidefinite when the finite
 *    side is an upper bound, negative semidefinite for a lower bound) are
 *    solved by outer approximation (gradient cuts) inside the
 *    branch-and-bound; such models have no duals and no basis.
 *  - A quadratic objective or row that is not convex is solved by spatial
 *    branch-and-bound since v0.10 (McCormick relaxations of the products,
 *    branching on their factors). The status POPSO_STATUS_NONCONVEX remains
 *    for nonconvex models with logical constraints (indicators, SOS,
 *    semi-continuous variables) and when the environment variable
 *    MILP_SPATIAL=0 turns spatial branch-and-bound off.
 *  - Besides linear rows, a model may have semi-continuous and semi-integer
 *    variables, SOS1/SOS2 constraints and indicator constraints. They are
 *    reformulated with auxiliary binaries internally; solutions and queries
 *    refer to the variables of the model only.
 *  - The library is silent by default (verbosity 0); the log goes to stdout.
 *
 *  - Model files may be gzip-compressed (detected by their content, e.g.
 *    model.mps.gz).
 *
 * The header is C99 and C++ compatible. Link against the shared library
 * `popso` (libpopso.so, libpopso.dylib or popso.dll). Define POPSO_STATIC
 * when linking a static build on Windows.
 */
#ifndef POPSO_H
#define POPSO_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(POPSO_BUILDING_LIBRARY)
#    define POPSO_API __declspec(dllexport)
#  elif defined(POPSO_STATIC)
#    define POPSO_API
#  else
#    define POPSO_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#  define POPSO_API __attribute__((visibility("default")))
#else
#  define POPSO_API
#endif

/* Infinite bound. */
#define POPSO_INFINITY 1e30

/* Error codes. */
#define POPSO_OK 0
#define POPSO_ERROR_INVALID_ARGUMENT 1 /* bad handle, index, value or parameter name */
#define POPSO_ERROR_OUT_OF_MEMORY 2
#define POPSO_ERROR_IO 3              /* a file cannot be opened, read or written */
#define POPSO_ERROR_PARSE 4           /* malformed MPS or solution file */
#define POPSO_ERROR_NOT_SOLVED 5      /* no result: popso_solve() not called since the last change */
#define POPSO_ERROR_NO_SOLUTION 6     /* solved, but no feasible solution is known */
#define POPSO_ERROR_INTERNAL 7

/* Variable types. */
#define POPSO_CONTINUOUS 0
#define POPSO_INTEGER 1
#define POPSO_BINARY 2 /* integer with bounds clamped to [0, 1] */
#define POPSO_SEMICONTINUOUS 3 /* x = 0 or lb <= x <= ub */
#define POPSO_SEMIINTEGER 4    /* x = 0 or lb <= x <= ub, x integer */

/* SOS types. */
#define POPSO_SOS1 1 /* at most one member nonzero */
#define POPSO_SOS2 2 /* at most two members nonzero, adjacent in the order of their weights */

/* Basis status of a variable or row (popso_get_basis). A row is "at lower"
 * when its activity is at its lower bound. */
#define POPSO_BASIS_BASIC 0
#define POPSO_BASIS_LOWER 1
#define POPSO_BASIS_UPPER 2
#define POPSO_BASIS_ZERO 3 /* nonbasic free variable (or free row) at zero */

/* Objective sense. */
#define POPSO_MINIMIZE 1
#define POPSO_MAXIMIZE -1

/* Solve status. */
#define POPSO_STATUS_NOT_SOLVED 0
#define POPSO_STATUS_OPTIMAL 1
#define POPSO_STATUS_INFEASIBLE 2
#define POPSO_STATUS_UNBOUNDED 3
#define POPSO_STATUS_INF_OR_UNBD 4
#define POPSO_STATUS_TIME_LIMIT 5
#define POPSO_STATUS_NODE_LIMIT 6
#define POPSO_STATUS_MEMORY_LIMIT 7
#define POPSO_STATUS_ITERATION_LIMIT 8
#define POPSO_STATUS_INTERRUPTED 9 /* popso_interrupt or the incumbent callback stopped the solve */
#define POPSO_STATUS_NUMERICAL_ERROR 10
#define POPSO_STATUS_UNKNOWN 11
#define POPSO_STATUS_NONCONVEX 12 /* the quadratic objective or a quadratic row is not convex */

typedef struct popso_model popso_model;

/* ---- Library ---------------------------------------------------------- */

/* Version string, e.g. "0.2.0". */
POPSO_API const char* popso_version(void);

/* Name of a status code ("OPTIMAL", "TIME_LIMIT", ...); "INVALID" if unknown. */
POPSO_API const char* popso_status_string(int status);

/* Description of the last error on this handle (empty string if none). With
 * m == NULL: the last error of a call that had no valid handle, per thread. */
POPSO_API const char* popso_last_error(const popso_model* m);

/* ---- Model lifetime --------------------------------------------------- */

/* Creates an empty minimization model; NULL if out of memory. */
POPSO_API popso_model* popso_create(void);
POPSO_API void popso_free(popso_model* m); /* NULL is ignored */

/* Replaces the model by the contents of an MPS file (free or fixed format,
 * optionally gzip-compressed). Parameters and the start solution are kept;
 * the start is cleared if the number of variables changes. */
POPSO_API int popso_read_mps(popso_model* m, const char* filename);
/* Replaces the model by the contents of a CPLEX LP-format file (same rules
 * as popso_read_mps). */
POPSO_API int popso_read_lp(popso_model* m, const char* filename);
/* Writes the model as free MPS (with SC bounds, SOS and INDICATORS sections
 * for the logical constraints). */
POPSO_API int popso_write_mps(popso_model* m, const char* filename);

/* ---- Building a model ------------------------------------------------- */

POPSO_API int popso_set_model_name(popso_model* m, const char* name);
POPSO_API const char* popso_get_model_name(const popso_model* m);

/* Adds a variable. name may be NULL or "" (then it is named C<index>).
 * index (may be NULL) receives the new variable's index. */
POPSO_API int popso_add_var(popso_model* m, double lb, double ub, double obj, int vtype, const char* name,
                            int* index);
/* Adds count variables. Each array may be NULL for its default: lb 0,
 * ub +inf, obj 0, continuous, automatic names. */
POPSO_API int popso_add_vars(popso_model* m, int count, const double* lb, const double* ub, const double* obj,
                             const int* vtype, const char* const* names);

/* Adds the row lb <= sum_k vals[k] * x[cols[k]] <= ub. Use lb == ub for an
 * equality and +-POPSO_INFINITY for a one-sided row. Repeated columns are
 * summed. name may be NULL. index (may be NULL) receives the row's index. */
POPSO_API int popso_add_row(popso_model* m, int nnz, const int* cols, const double* vals, double lb, double ub,
                            const char* name, int* index);
/* Adds count rows in compressed sparse row form: the entries of row r are
 * cols/vals[starts[r] .. starts[r+1]-1] (starts has count+1 entries).
 * lb, ub and names may be NULL (defaults -inf, +inf, automatic names). */
POPSO_API int popso_add_rows(popso_model* m, int count, const int* starts, const int* cols, const double* vals,
                             const double* lb, const double* ub, const char* const* names);

/* Adds an SOS constraint of type POPSO_SOS1 or POPSO_SOS2 over nnz distinct
 * variables. weights (may be NULL: 1, 2, ..., nnz) order the members; they
 * must be distinct. index (may be NULL) receives the SOS index. */
POPSO_API int popso_add_sos(popso_model* m, int type, int nnz, const int* vars, const double* weights,
                            const char* name, int* index);
/* Adds an indicator constraint: if variable binvar (which must be binary)
 * equals binval (0 or 1), then lb <= sum_k vals[k] * x[cols[k]] <= ub must
 * hold; otherwise the row is ignored. index (may be NULL) receives the
 * indicator's index. */
POPSO_API int popso_add_indicator(popso_model* m, int binvar, int binval, int nnz, const int* cols,
                                  const double* vals, double lb, double ub, const char* name, int* index);

POPSO_API int popso_set_var_bounds(popso_model* m, int var, double lb, double ub);
POPSO_API int popso_set_var_type(popso_model* m, int var, int vtype);
POPSO_API int popso_set_row_bounds(popso_model* m, int row, double lb, double ub);

/* Objective: sense, constant term, single coefficients, or all coefficients
 * at once (popso_set_objective zeroes every coefficient not listed). */
POPSO_API int popso_set_objective_sense(popso_model* m, int sense);
POPSO_API int popso_set_objective_constant(popso_model* m, double constant);
POPSO_API int popso_set_obj_coef(popso_model* m, int var, double value);
POPSO_API int popso_set_objective(popso_model* m, int nnz, const int* vars, const double* vals);

/* Quadratic objective term 1/2 x^T Q x, Q symmetric. Entries are given as
 * (rows[k], cols[k], vals[k]) from either triangle: an off-diagonal entry
 * (i, j) sets both Q_ij and Q_ji (give each pair once), a diagonal entry Q_jj
 * contributes 1/2 Q_jj x_j^2. Example: x^2 + 3 x y is Q_xx = 2, Q_xy = 3.
 * Repeated entries are summed. popso_set_q replaces Q (nnz = 0 removes it),
 * popso_add_q_entries adds to it. */
POPSO_API int popso_set_q(popso_model* m, int nnz, const int* rows, const int* cols, const double* vals);
POPSO_API int popso_add_q_entries(popso_model* m, int nnz, const int* rows, const int* cols, const double* vals);
/* Number of nonzeros of Q's lower triangle (-1 on a bad handle). */
POPSO_API int popso_num_q_nonzeros(popso_model* m);
/* Copies Q's lower triangle (rows[k] >= cols[k], sorted by column, then row)
 * into the arrays, which must hold popso_num_q_nonzeros() entries (size). */
POPSO_API int popso_get_q(popso_model* m, int size, int* rows, int* cols, double* vals);

/* Quadratic constraint terms: adds vals[k] * x_{vars1[k]} * x_{vars2[k]} to the
 * activity of `row` for each k (plain monomial coefficients, no 1/2: the pair
 * (i, j) with i != j is one term vals * x_i * x_j, (j, j) is vals * x_j^2).
 * Repeated entries are summed. Example: row 0 as x^2 + 3 x y + 2 x <= 4 is
 * popso_add_row with x: 2, upper bound 4, then terms (x, x, 1), (x, y, 3). */
POPSO_API int popso_add_qrow_terms(popso_model* m, int row, int nnz, const int* vars1, const int* vars2,
                                   const double* vals);
/* Number of quadratic terms of a row, merged (-1 on a bad handle or row). */
POPSO_API int popso_num_qrow_terms(const popso_model* m, int row);
/* Copies a row's quadratic terms (vars1[k] >= vars2[k], sorted, duplicates
 * merged) into the arrays, which must hold popso_num_qrow_terms() entries. */
POPSO_API int popso_get_qrow_terms(const popso_model* m, int row, int size, int* vars1, int* vars2, double* vals);

/* ---- Model queries (return -1 / NULL / NaN on a bad handle or index) ---- */

POPSO_API int popso_num_vars(const popso_model* m);
POPSO_API int popso_num_rows(const popso_model* m);
POPSO_API int popso_num_sos(const popso_model* m);
POPSO_API int popso_num_indicators(const popso_model* m);
POPSO_API long long popso_num_nonzeros(popso_model* m);
POPSO_API int popso_get_objective_sense(const popso_model* m);
POPSO_API double popso_get_objective_constant(const popso_model* m);
/* Any output pointer may be NULL. */
POPSO_API int popso_get_var(const popso_model* m, int var, double* lb, double* ub, double* obj, int* vtype);
POPSO_API int popso_get_row(const popso_model* m, int row, double* lb, double* ub);
/* Names stay valid until the model is changed or freed. */
POPSO_API const char* popso_get_var_name(const popso_model* m, int var);
POPSO_API const char* popso_get_row_name(const popso_model* m, int row);
/* Index of the variable with this name, or -1. */
POPSO_API int popso_find_var(popso_model* m, const char* name);

/* ---- Parameters ------------------------------------------------------- */

/* Parameters by name (all values are doubles; integer parameters must be
 * integral, flags 0 or 1):
 *   time_limit     wall-clock seconds (default: +inf)
 *   threads        0 = all cores (default 0)
 *   mip_gap        relative gap at which to stop (default 1e-4)
 *   mip_abs_gap    absolute gap at which to stop (default 1e-10)
 *   seed           random seed (default 0)
 *   deterministic  1 = reproducible parallel search (default 0)
 *   verbosity      0 silent (default here), 1 summary, 2 progress, 3 debug
 *   mem_limit      approximate memory cap in MB (default 4096)
 *   node_limit     branch-and-bound node limit, -1 = none (default)
 *   presolve       1 = on (default), 0 = off
 *   racing         number of concurrently racing searches (default 1)
 *   lp_method      LP algorithm for pure LPs and the MIP root LP: 0 auto (default),
 *                  1 dual simplex, 2 barrier (interior point with crossover)
 *   duals          dual values (popso_get_duals): 0 none, 1 for LPs (default),
 *                  2 also fixed-MIP duals for MILPs (see popso_has_duals);
 *                  continuous QPs always have them */
POPSO_API int popso_set_param(popso_model* m, const char* name, double value);
POPSO_API int popso_get_param(const popso_model* m, const char* name, double* value);

/* ---- Starting solution ------------------------------------------------ */

/* Sets a (possibly partial) starting solution: x[vars[k]] = vals[k]. With
 * vars == NULL, vals holds a value for every variable (nnz must equal the
 * number of variables) and NaN entries mean "not given". A complete feasible
 * start becomes the first incumbent; for a partial or infeasible one the
 * given integer values are fixed and the rest completed by a short sub-MIP
 * (the start is ignored if that fails). Replaces any previous start. */
POPSO_API int popso_set_start(popso_model* m, int nnz, const int* vars, const double* vals);
/* Reads a start from a "name value" solution file (the format popso writes). */
POPSO_API int popso_read_start(popso_model* m, const char* filename);
POPSO_API int popso_clear_start(popso_model* m);

/* ---- Solving and results ---------------------------------------------- */

/* Solves the model. Returns POPSO_OK whenever the solver ran (whatever the
 * status); an error code only if it could not run. */
POPSO_API int popso_solve(popso_model* m);

/* Asks a running popso_solve on m to stop soon (typically well within 0.1 s)
 * with status POPSO_STATUS_INTERRUPTED and the best solution found so far.
 * Thread-safe and async-signal-safe (a single atomic store): call it from
 * another thread or from a signal handler. The request is cleared when
 * popso_solve starts, so a call while no solve runs has no effect on later
 * solves. Returns POPSO_ERROR_INVALID_ARGUMENT for a NULL handle (without
 * setting the error message). */
POPSO_API int popso_interrupt(popso_model* m);

/* Incumbent callback: called for each new best solution during popso_solve
 * with its objective, the current best bound (both in the model's sense) and
 * the solution x (num_vars values, valid during the call only). Calls are
 * serialized (one at a time, though not always from the same thread), each
 * one with a strictly better objective; with the deterministic parameter they
 * come at deterministic points of the search. Returning nonzero stops the
 * solve (status POPSO_STATUS_INTERRUPTED). The callback must not call popso
 * functions on m other than popso_interrupt. Not called for quadratic
 * objectives. */
typedef int (*popso_incumbent_callback)(void* user_data, double objective, double bound, const double* x,
                                        int num_vars);
/* Sets (or with cb == NULL removes) the incumbent callback. */
POPSO_API int popso_set_incumbent_callback(popso_model* m, popso_incumbent_callback cb, void* user_data);

POPSO_API int popso_get_status(const popso_model* m); /* POPSO_STATUS_*, NOT_SOLVED before a solve */
POPSO_API int popso_has_solution(const popso_model* m); /* 1 if a feasible solution is known */
/* Objective value of the best solution (POPSO_ERROR_NO_SOLUTION if none). */
POPSO_API int popso_get_objective(const popso_model* m, double* value);
/* Best proven bound on the optimal objective (+-inf if none). */
POPSO_API int popso_get_bound(const popso_model* m, double* value);
/* Relative gap |objective - bound| / max(|objective|, 1e-10); +inf without a
 * solution or a finite bound. */
POPSO_API int popso_get_gap(const popso_model* m, double* value);
POPSO_API int popso_get_node_count(const popso_model* m, long long* value);
POPSO_API int popso_get_lp_iterations(const popso_model* m, long long* value);
POPSO_API int popso_get_solve_time(const popso_model* m, double* seconds);
/* Copies the best solution into x (size >= number of variables). */
POPSO_API int popso_get_solution(const popso_model* m, double* x, int size);
/* Writes the result as a "name value" solution file (status and objective in
 * '#' comment lines; values only if a solution is known). */
POPSO_API int popso_write_solution(const popso_model* m, const char* filename);
/* Dual values: row duals y and reduced costs d = c + Q x - A^T y.
 * Minimization: y_i >= 0 at an active row lower bound, <= 0 at an upper one,
 * d_j >= 0 at a variable's lower bound, <= 0 at its upper one; maximization:
 * the opposite signs.
 *  - LPs solved to optimality (parameter duals >= 1, the default): the duals of
 *    an optimal basis of the original (unpresolved) LP, found from the optimal
 *    solution after the solve; complementary to the returned solution.
 *  - MILPs with a solution (parameter duals = 2): "fixed-MIP duals", the duals
 *    of an optimal basis of the LP with every integer variable fixed at its
 *    value in the solution. popso_has_duals returns 2 for these. If the solve
 *    stopped early (limit, interrupt), the solution's continuous part need not
 *    be optimal for that fixed LP: the duals belong to the fixed LP's optimum
 *    and need not be complementary to the returned continuous values (the
 *    solution itself is not changed).
 *  - Continuous QPs solved to optimality: from the barrier's final iterate (no
 *    basis), satisfying the KKT conditions to the solver's tolerance.
 * Recovered LP / fixed-MIP duals whose basis does not reproduce the solution's
 * objective (relative tolerance 1e-6) are dropped, as are duals whose recovery
 * was interrupted or ran out of time (at least 1 s is allowed).
 * Not for MIQPs and models with SOS, indicator or semi-continuous variables.
 * popso_has_duals returns 0 if none are available; the getters then fail with
 * POPSO_ERROR_NO_SOLUTION. */
POPSO_API int popso_has_duals(const popso_model* m);
POPSO_API int popso_get_duals(const popso_model* m, double* y, int size);          /* size >= rows */
POPSO_API int popso_get_reduced_costs(const popso_model* m, double* d, int size);  /* size >= vars */
/* The optimal basis the LP or fixed-MIP duals come from (POPSO_BASIS_*), one
 * entry per variable and per row; either array may be NULL. popso_has_basis
 * returns 1 if there is one (not for QPs). */
POPSO_API int popso_has_basis(const popso_model* m);
POPSO_API int popso_get_basis(const popso_model* m, int* var_status, int* row_status);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* POPSO_H */
