/*
 * popso C API: build or read an LP/MILP model, solve it and query the result.
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
 *    used from different threads concurrently. Note that the memory limit is
 *    also checked against the resident size of the whole process.
 *  - Changing the model discards the last result.
 *  - The library is silent by default (verbosity 0); the log goes to stdout.
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
#define POPSO_STATUS_INTERRUPTED 9
#define POPSO_STATUS_NUMERICAL_ERROR 10
#define POPSO_STATUS_UNKNOWN 11

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

/* Replaces the model by the contents of an MPS file (free or fixed format).
 * Parameters and the start solution are kept; the start is cleared if the
 * number of variables changes. */
POPSO_API int popso_read_mps(popso_model* m, const char* filename);
/* Writes the model as free MPS. */
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

POPSO_API int popso_set_var_bounds(popso_model* m, int var, double lb, double ub);
POPSO_API int popso_set_var_type(popso_model* m, int var, int vtype);
POPSO_API int popso_set_row_bounds(popso_model* m, int row, double lb, double ub);

/* Objective: sense, constant term, single coefficients, or all coefficients
 * at once (popso_set_objective zeroes every coefficient not listed). */
POPSO_API int popso_set_objective_sense(popso_model* m, int sense);
POPSO_API int popso_set_objective_constant(popso_model* m, double constant);
POPSO_API int popso_set_obj_coef(popso_model* m, int var, double value);
POPSO_API int popso_set_objective(popso_model* m, int nnz, const int* vars, const double* vals);

/* ---- Model queries (return -1 / NULL / NaN on a bad handle or index) ---- */

POPSO_API int popso_num_vars(const popso_model* m);
POPSO_API int popso_num_rows(const popso_model* m);
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
 *   racing         number of concurrently racing searches (default 1) */
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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* POPSO_H */
