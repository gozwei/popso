// popso C API (include/popso.h) on top of the internal milp:: solver.
//
// The handle keeps a milp::Model whose constraint matrix is assembled lazily:
// rows added through the API are collected as triplets and merged into the
// CSC matrix when it is needed (solve, MPS output, nonzero count).

#include "popso.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "milp/core/model.hpp"
#include "milp/io/lp_reader.hpp"
#include "milp/io/mps_reader.hpp"
#include "milp/io/solution.hpp"
#include "milp/solver/solver.hpp"

#ifndef POPSO_VERSION
#define POPSO_VERSION "unknown"
#endif

using milp::Index;

static_assert(std::atomic<bool>::is_always_lock_free, "popso_interrupt needs a lock-free atomic flag");

struct popso_model {
  milp::Model model;
  // Pending matrix entries (row, column, value) not yet in model.A.
  std::vector<Index> pend_row, pend_col;
  std::vector<double> pend_val;
  milp::SolverParams params;
  std::vector<double> start;  // empty: none; else NaN = not given (resized to the model on solve)
  milp::SolveResult result;
  bool solved = false;
  std::atomic<bool> interrupt{false};  // popso_interrupt
  popso_incumbent_callback callback = nullptr;
  void* callback_data = nullptr;
  std::unordered_map<std::string, Index> name_index;  // lazily built by popso_find_var
  bool name_index_valid = false;
  std::string error;
};

namespace {

thread_local std::string g_error;  // errors without a valid handle

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct ApiError {
  int code;
  std::string message;
};

[[noreturn]] void fail(int code, std::string message) { throw ApiError{code, std::move(message)}; }

// Runs f, translating exceptions into error codes and the handle's (or the thread's) last error.
template <class F>
int guarded(popso_model* m, F&& f) {
  std::string& err = m ? m->error : g_error;
  if (!m) {
    g_error = "null model handle";
    return POPSO_ERROR_INVALID_ARGUMENT;
  }
  try {
    f();
    err.clear();
    return POPSO_OK;
  } catch (const ApiError& e) {
    err = e.message;
    return e.code;
  } catch (const std::bad_alloc&) {
    err = "out of memory";
    return POPSO_ERROR_OUT_OF_MEMORY;
  } catch (const std::exception& e) {
    err = e.what();
    return POPSO_ERROR_INTERNAL;
  } catch (...) {
    err = "unknown error";
    return POPSO_ERROR_INTERNAL;
  }
}

// Const queries record errors too (the error string is not part of the logical state).
template <class F>
int guarded(const popso_model* m, F&& f) {
  return guarded(const_cast<popso_model*>(m), std::forward<F>(f));
}

double to_internal_bound(double v) {
  if (v >= POPSO_INFINITY) return milp::kInf;
  if (v <= -POPSO_INFINITY) return -milp::kInf;
  return v;
}

void check_bounds(double lb, double ub, const char* what) {
  if (std::isnan(lb) || std::isnan(ub)) fail(POPSO_ERROR_INVALID_ARGUMENT, std::string(what) + ": NaN bound");
}

milp::VarType to_var_type(int t) {
  switch (t) {
    case POPSO_CONTINUOUS: return milp::VarType::Continuous;
    case POPSO_INTEGER: return milp::VarType::Integer;
    case POPSO_BINARY: return milp::VarType::Binary;
    case POPSO_SEMICONTINUOUS: return milp::VarType::SemiContinuous;
    case POPSO_SEMIINTEGER: return milp::VarType::SemiInteger;
    default: fail(POPSO_ERROR_INVALID_ARGUMENT, "invalid variable type " + std::to_string(t));
  }
}

int from_var_type(milp::VarType t) {
  switch (t) {
    case milp::VarType::Continuous: return POPSO_CONTINUOUS;
    case milp::VarType::Integer: return POPSO_INTEGER;
    case milp::VarType::Binary: return POPSO_BINARY;
    case milp::VarType::SemiContinuous: return POPSO_SEMICONTINUOUS;
    case milp::VarType::SemiInteger: return POPSO_SEMIINTEGER;
  }
  return POPSO_CONTINUOUS;
}

int from_status(milp::Status s) {
  switch (s) {
    case milp::Status::Unknown: return POPSO_STATUS_UNKNOWN;
    case milp::Status::Optimal: return POPSO_STATUS_OPTIMAL;
    case milp::Status::Infeasible: return POPSO_STATUS_INFEASIBLE;
    case milp::Status::Unbounded: return POPSO_STATUS_UNBOUNDED;
    case milp::Status::InfeasibleOrUnbounded: return POPSO_STATUS_INF_OR_UNBD;
    case milp::Status::TimeLimit: return POPSO_STATUS_TIME_LIMIT;
    case milp::Status::NodeLimit: return POPSO_STATUS_NODE_LIMIT;
    case milp::Status::MemoryLimit: return POPSO_STATUS_MEMORY_LIMIT;
    case milp::Status::IterationLimit: return POPSO_STATUS_ITERATION_LIMIT;
    case milp::Status::Interrupted: return POPSO_STATUS_INTERRUPTED;
    case milp::Status::NumericalError: return POPSO_STATUS_NUMERICAL_ERROR;
    case milp::Status::NonConvex: return POPSO_STATUS_NONCONVEX;
  }
  return POPSO_STATUS_UNKNOWN;
}

// The model changed: the result and the name index are stale.
void touched(popso_model* m) {
  m->solved = false;
  m->result = milp::SolveResult{};
  m->name_index_valid = false;
}

void check_var(const popso_model* m, int j) {
  if (j < 0 || j >= m->model.num_cols())
    fail(POPSO_ERROR_INVALID_ARGUMENT, "variable index " + std::to_string(j) + " out of range");
}

void check_row(const popso_model* m, int i) {
  if (i < 0 || i >= m->model.num_rows())
    fail(POPSO_ERROR_INVALID_ARGUMENT, "row index " + std::to_string(i) + " out of range");
}

void check_solved(const popso_model* m) {
  if (!m->solved) fail(POPSO_ERROR_NOT_SOLVED, "the model has not been solved since it was last changed");
}

// Merges pending row entries into the CSC matrix (also after columns were added).
void flush_matrix(popso_model* m) {
  milp::Model& md = m->model;
  if (m->pend_val.empty() && md.A.num_major == md.num_cols() && md.A.num_minor == md.num_rows()) return;
  std::vector<Index> col, row;
  std::vector<double> val;
  const size_t nnz = static_cast<size_t>(md.A.nnz()) + m->pend_val.size();
  col.reserve(nnz);
  row.reserve(nnz);
  val.reserve(nnz);
  for (Index j = 0; j < md.A.num_major; ++j) {
    for (Index k = md.A.begin(j); k < md.A.end(j); ++k) {
      col.push_back(j);
      row.push_back(md.A.index[k]);
      val.push_back(md.A.value[k]);
    }
  }
  col.insert(col.end(), m->pend_col.begin(), m->pend_col.end());
  row.insert(row.end(), m->pend_row.begin(), m->pend_row.end());
  val.insert(val.end(), m->pend_val.begin(), m->pend_val.end());
  md.A = milp::SparseMatrix::from_triplets(md.num_cols(), md.num_rows(), col, row, val);
  m->pend_row.clear();
  m->pend_col.clear();
  m->pend_val.clear();
}

void check_file_readable(const char* filename) {
  if (!filename) fail(POPSO_ERROR_INVALID_ARGUMENT, "null file name");
  std::ifstream f(filename, std::ios::binary);
  if (!f) fail(POPSO_ERROR_IO, std::string(filename) + ": cannot open file");
}

void add_var_impl(popso_model* m, double lb, double ub, double obj, int vtype, const char* name) {
  check_bounds(lb, ub, "add_var");
  if (!std::isfinite(obj)) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_var: objective coefficient must be finite");
  const milp::VarType t = to_var_type(vtype);
  lb = to_internal_bound(lb);
  ub = to_internal_bound(ub);
  if (t == milp::VarType::Binary) {
    lb = std::max(lb, 0.0);
    ub = std::min(ub, 1.0);
  }
  milp::Model& md = m->model;
  const Index j = md.num_cols();
  md.col_names.push_back(name && *name ? std::string(name) : "C" + std::to_string(j));
  md.obj.push_back(obj);
  md.col_lb.push_back(lb);
  md.col_ub.push_back(ub);
  md.col_type.push_back(t);
}

void add_row_impl(popso_model* m, int nnz, const int* cols, const double* vals, double lb, double ub,
                  const char* name) {
  if (nnz < 0 || (nnz > 0 && (!cols || !vals))) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_row: invalid entries");
  check_bounds(lb, ub, "add_row");
  milp::Model& md = m->model;
  const Index i = md.num_rows();
  for (int k = 0; k < nnz; ++k) {
    if (cols[k] < 0 || cols[k] >= md.num_cols())
      fail(POPSO_ERROR_INVALID_ARGUMENT, "add_row: variable index " + std::to_string(cols[k]) + " out of range");
    if (!std::isfinite(vals[k])) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_row: coefficient must be finite");
  }
  for (int k = 0; k < nnz; ++k) {
    if (vals[k] == 0.0) continue;
    m->pend_row.push_back(i);
    m->pend_col.push_back(cols[k]);
    m->pend_val.push_back(vals[k]);
  }
  md.row_names.push_back(name && *name ? std::string(name) : "R" + std::to_string(i));
  md.row_lb.push_back(to_internal_bound(lb));
  md.row_ub.push_back(to_internal_bound(ub));
}

bool is_integral_value(double v) { return std::isfinite(v) && v == std::floor(v); }

double gap_of(const milp::SolveResult& r) {
  if (!r.has_solution || !std::isfinite(r.bound)) return milp::kInf;
  const double diff = std::abs(r.objective - r.bound);
  return diff == 0.0 ? 0.0 : diff / std::max(std::abs(r.objective), 1e-10);
}

}  // namespace

extern "C" {

const char* popso_version(void) { return POPSO_VERSION; }

const char* popso_status_string(int status) {
  switch (status) {
    case POPSO_STATUS_NOT_SOLVED: return "NOT_SOLVED";
    case POPSO_STATUS_OPTIMAL: return "OPTIMAL";
    case POPSO_STATUS_INFEASIBLE: return "INFEASIBLE";
    case POPSO_STATUS_UNBOUNDED: return "UNBOUNDED";
    case POPSO_STATUS_INF_OR_UNBD: return "INF_OR_UNBD";
    case POPSO_STATUS_TIME_LIMIT: return "TIME_LIMIT";
    case POPSO_STATUS_NODE_LIMIT: return "NODE_LIMIT";
    case POPSO_STATUS_MEMORY_LIMIT: return "MEMORY_LIMIT";
    case POPSO_STATUS_ITERATION_LIMIT: return "ITERATION_LIMIT";
    case POPSO_STATUS_INTERRUPTED: return "INTERRUPTED";
    case POPSO_STATUS_NUMERICAL_ERROR: return "NUMERICAL_ERROR";
    case POPSO_STATUS_UNKNOWN: return "UNKNOWN";
    case POPSO_STATUS_NONCONVEX: return "NONCONVEX";
    default: return "INVALID";
  }
}

const char* popso_last_error(const popso_model* m) { return m ? m->error.c_str() : g_error.c_str(); }

popso_model* popso_create(void) {
  try {
    popso_model* m = new popso_model();
    m->params.verbosity = 0;
    m->params.duals = 1;
    return m;
  } catch (...) {
    g_error = "out of memory";
    return nullptr;
  }
}

void popso_free(popso_model* m) { delete m; }

int popso_read_mps(popso_model* m, const char* filename) {
  return guarded(m, [&] {
    check_file_readable(filename);
    std::vector<std::string> warnings;
    milp::Model md;
    try {
      md = milp::read_mps(filename, &warnings);
    } catch (const milp::MpsError& e) {
      fail(POPSO_ERROR_PARSE, e.what());
    }
    if (m->params.verbosity >= 1) {
      for (const std::string& w : warnings) std::printf("MPS warning: %s\n", w.c_str());
    }
    if (md.num_cols() != m->model.num_cols()) m->start.clear();
    m->model = std::move(md);
    m->pend_row.clear();
    m->pend_col.clear();
    m->pend_val.clear();
    touched(m);
  });
}

int popso_read_lp(popso_model* m, const char* filename) {
  return guarded(m, [&] {
    check_file_readable(filename);
    std::vector<std::string> warnings;
    milp::Model md;
    try {
      md = milp::read_lp(filename, &warnings);
    } catch (const milp::LpError& e) {
      fail(POPSO_ERROR_PARSE, e.what());
    }
    if (m->params.verbosity >= 1) {
      for (const std::string& w : warnings) std::printf("LP warning: %s\n", w.c_str());
    }
    if (md.num_cols() != m->model.num_cols()) m->start.clear();
    m->model = std::move(md);
    m->pend_row.clear();
    m->pend_col.clear();
    m->pend_val.clear();
    touched(m);
  });
}

int popso_write_mps(popso_model* m, const char* filename) {
  return guarded(m, [&] {
    if (!filename) fail(POPSO_ERROR_INVALID_ARGUMENT, "null file name");
    flush_matrix(m);
    try {
      milp::write_mps(m->model, filename);
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const std::exception& e) {
      fail(POPSO_ERROR_IO, e.what());
    }
  });
}

int popso_set_model_name(popso_model* m, const char* name) {
  return guarded(m, [&] { m->model.name = name ? name : ""; });
}

const char* popso_get_model_name(const popso_model* m) { return m ? m->model.name.c_str() : nullptr; }

int popso_add_var(popso_model* m, double lb, double ub, double obj, int vtype, const char* name, int* index) {
  return guarded(m, [&] {
    add_var_impl(m, lb, ub, obj, vtype, name);
    touched(m);
    if (index) *index = m->model.num_cols() - 1;
  });
}

int popso_add_vars(popso_model* m, int count, const double* lb, const double* ub, const double* obj,
                   const int* vtype, const char* const* names) {
  return guarded(m, [&] {
    if (count < 0) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_vars: negative count");
    // Validate everything first so that a failure adds nothing.
    for (int k = 0; k < count; ++k) {
      check_bounds(lb ? lb[k] : 0.0, ub ? ub[k] : 0.0, "add_vars");
      if (obj && !std::isfinite(obj[k])) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_vars: objective must be finite");
      if (vtype) to_var_type(vtype[k]);
    }
    for (int k = 0; k < count; ++k) {
      add_var_impl(m, lb ? lb[k] : 0.0, ub ? ub[k] : milp::kInf, obj ? obj[k] : 0.0,
                   vtype ? vtype[k] : POPSO_CONTINUOUS, names ? names[k] : nullptr);
    }
    touched(m);
  });
}

int popso_add_row(popso_model* m, int nnz, const int* cols, const double* vals, double lb, double ub,
                  const char* name, int* index) {
  return guarded(m, [&] {
    add_row_impl(m, nnz, cols, vals, lb, ub, name);
    touched(m);
    if (index) *index = m->model.num_rows() - 1;
  });
}

int popso_add_rows(popso_model* m, int count, const int* starts, const int* cols, const double* vals,
                   const double* lb, const double* ub, const char* const* names) {
  return guarded(m, [&] {
    if (count < 0 || (count > 0 && !starts)) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_rows: invalid arguments");
    for (int r = 0; r < count; ++r) {
      if (starts[r] < 0 || starts[r + 1] < starts[r]) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_rows: invalid starts");
      check_bounds(lb ? lb[r] : 0.0, ub ? ub[r] : 0.0, "add_rows");
      for (int k = starts[r]; k < starts[r + 1]; ++k) {
        if (!cols || !vals) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_rows: null entries");
        if (cols[k] < 0 || cols[k] >= m->model.num_cols())
          fail(POPSO_ERROR_INVALID_ARGUMENT, "add_rows: variable index " + std::to_string(cols[k]) + " out of range");
        if (!std::isfinite(vals[k])) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_rows: coefficient must be finite");
      }
    }
    for (int r = 0; r < count; ++r) {
      const int b = starts[r], n = starts[r + 1] - starts[r];
      add_row_impl(m, n, n ? cols + b : nullptr, n ? vals + b : nullptr, lb ? lb[r] : -milp::kInf,
                   ub ? ub[r] : milp::kInf, names ? names[r] : nullptr);
    }
    touched(m);
  });
}

int popso_add_sos(popso_model* m, int type, int nnz, const int* vars, const double* weights, const char* name,
                  int* index) {
  return guarded(m, [&] {
    if (type != POPSO_SOS1 && type != POPSO_SOS2) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_sos: type must be 1 or 2");
    if (nnz < 0 || (nnz > 0 && !vars)) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_sos: invalid members");
    milp::SosConstraint s;
    s.type = type;
    s.name = name && *name ? std::string(name) : "SOS" + std::to_string(m->model.sos.size());
    std::vector<char> seen(static_cast<size_t>(m->model.num_cols()), 0);
    for (int k = 0; k < nnz; ++k) {
      check_var(m, vars[k]);
      if (seen[vars[k]]) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_sos: variable " + std::to_string(vars[k]) + " repeated");
      seen[vars[k]] = 1;
      const double w = weights ? weights[k] : static_cast<double>(k + 1);
      if (!std::isfinite(w)) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_sos: weights must be finite");
      s.cols.push_back(vars[k]);
      s.weights.push_back(w);
    }
    std::vector<double> sorted = s.weights;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
      fail(POPSO_ERROR_INVALID_ARGUMENT, "add_sos: weights must be distinct");
    m->model.sos.push_back(std::move(s));
    touched(m);
    if (index) *index = static_cast<int>(m->model.sos.size()) - 1;
  });
}

int popso_add_indicator(popso_model* m, int binvar, int binval, int nnz, const int* cols, const double* vals, double lb,
                        double ub, const char* name, int* index) {
  return guarded(m, [&] {
    check_var(m, binvar);
    if (binval != 0 && binval != 1) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_indicator: binval must be 0 or 1");
    const milp::Model& md = m->model;
    if (!milp::is_integral_type(md.col_type[binvar]) || milp::is_semi_type(md.col_type[binvar]) ||
        md.col_lb[binvar] < 0.0 || md.col_ub[binvar] > 1.0)
      fail(POPSO_ERROR_INVALID_ARGUMENT, "add_indicator: variable " + std::to_string(binvar) + " is not binary");
    if (nnz < 0 || (nnz > 0 && (!cols || !vals))) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_indicator: invalid entries");
    check_bounds(lb, ub, "add_indicator");
    milp::IndicatorConstraint c;
    c.binary = binvar;
    c.value = binval;
    c.name = name && *name ? std::string(name) : "IND" + std::to_string(md.indicators.size());
    for (int k = 0; k < nnz; ++k) {
      check_var(m, cols[k]);
      if (!std::isfinite(vals[k])) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_indicator: coefficient must be finite");
      if (vals[k] == 0.0) continue;
      c.cols.push_back(cols[k]);
      c.coefs.push_back(vals[k]);
    }
    c.lb = to_internal_bound(lb);
    c.ub = to_internal_bound(ub);
    m->model.indicators.push_back(std::move(c));
    touched(m);
    if (index) *index = static_cast<int>(m->model.indicators.size()) - 1;
  });
}

int popso_set_var_bounds(popso_model* m, int var, double lb, double ub) {
  return guarded(m, [&] {
    check_var(m, var);
    check_bounds(lb, ub, "set_var_bounds");
    m->model.col_lb[var] = to_internal_bound(lb);
    m->model.col_ub[var] = to_internal_bound(ub);
    touched(m);
  });
}

int popso_set_var_type(popso_model* m, int var, int vtype) {
  return guarded(m, [&] {
    check_var(m, var);
    const milp::VarType t = to_var_type(vtype);
    m->model.col_type[var] = t;
    if (t == milp::VarType::Binary) {
      m->model.col_lb[var] = std::max(m->model.col_lb[var], 0.0);
      m->model.col_ub[var] = std::min(m->model.col_ub[var], 1.0);
    }
    touched(m);
  });
}

int popso_set_row_bounds(popso_model* m, int row, double lb, double ub) {
  return guarded(m, [&] {
    check_row(m, row);
    check_bounds(lb, ub, "set_row_bounds");
    m->model.row_lb[row] = to_internal_bound(lb);
    m->model.row_ub[row] = to_internal_bound(ub);
    touched(m);
  });
}

int popso_set_objective_sense(popso_model* m, int sense) {
  return guarded(m, [&] {
    if (sense != POPSO_MINIMIZE && sense != POPSO_MAXIMIZE)
      fail(POPSO_ERROR_INVALID_ARGUMENT, "sense must be POPSO_MINIMIZE or POPSO_MAXIMIZE");
    m->model.sense = sense == POPSO_MAXIMIZE ? milp::ObjSense::Maximize : milp::ObjSense::Minimize;
    touched(m);
  });
}

int popso_set_objective_constant(popso_model* m, double constant) {
  return guarded(m, [&] {
    if (!std::isfinite(constant)) fail(POPSO_ERROR_INVALID_ARGUMENT, "objective constant must be finite");
    m->model.obj_offset = constant;
    touched(m);
  });
}

int popso_set_obj_coef(popso_model* m, int var, double value) {
  return guarded(m, [&] {
    check_var(m, var);
    if (!std::isfinite(value)) fail(POPSO_ERROR_INVALID_ARGUMENT, "objective coefficient must be finite");
    m->model.obj[var] = value;
    touched(m);
  });
}

int popso_set_objective(popso_model* m, int nnz, const int* vars, const double* vals) {
  return guarded(m, [&] {
    if (nnz < 0 || (nnz > 0 && (!vars || !vals))) fail(POPSO_ERROR_INVALID_ARGUMENT, "set_objective: invalid entries");
    std::vector<double> obj(m->model.num_cols(), 0.0);
    for (int k = 0; k < nnz; ++k) {
      check_var(m, vars[k]);
      if (!std::isfinite(vals[k])) fail(POPSO_ERROR_INVALID_ARGUMENT, "objective coefficient must be finite");
      obj[vars[k]] += vals[k];
    }
    m->model.obj = std::move(obj);
    touched(m);
  });
}

int popso_set_objective_attributes(popso_model* m, int priority, double weight, double abs_tol, double rel_tol) {
  return guarded(m, [&] {
    if (!std::isfinite(weight)) fail(POPSO_ERROR_INVALID_ARGUMENT, "objective weight must be finite");
    if (!(abs_tol >= 0.0) || !(rel_tol >= 0.0) || !std::isfinite(abs_tol) || !std::isfinite(rel_tol))
      fail(POPSO_ERROR_INVALID_ARGUMENT, "objective tolerances must be finite and not negative");
    m->model.obj_priority = priority;
    m->model.obj_weight = weight;
    m->model.obj_abs_tol = abs_tol;
    m->model.obj_rel_tol = rel_tol;
    touched(m);
  });
}

int popso_add_objective(popso_model* m, int nnz, const int* vars, const double* vals, double constant, int priority,
                        double weight, double abs_tol, double rel_tol, const char* name, int* index) {
  return guarded(m, [&] {
    if (nnz < 0 || (nnz > 0 && (!vars || !vals))) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_objective: invalid entries");
    if (!std::isfinite(constant) || !std::isfinite(weight))
      fail(POPSO_ERROR_INVALID_ARGUMENT, "objective constant and weight must be finite");
    if (!(abs_tol >= 0.0) || !(rel_tol >= 0.0) || !std::isfinite(abs_tol) || !std::isfinite(rel_tol))
      fail(POPSO_ERROR_INVALID_ARGUMENT, "objective tolerances must be finite and not negative");
    milp::Objective o;
    std::vector<std::pair<int, double>> e;
    for (int k = 0; k < nnz; ++k) {
      check_var(m, vars[k]);
      if (!std::isfinite(vals[k])) fail(POPSO_ERROR_INVALID_ARGUMENT, "objective coefficient must be finite");
      e.emplace_back(vars[k], vals[k]);
    }
    std::stable_sort(e.begin(), e.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [j, v] : e) {
      if (!o.cols.empty() && o.cols.back() == j) o.coefs.back() += v;
      else o.cols.push_back(j), o.coefs.push_back(v);
    }
    o.offset = constant;
    o.priority = priority;
    o.weight = weight;
    o.abs_tol = abs_tol;
    o.rel_tol = rel_tol;
    o.name = name ? name : "";
    m->model.objectives.push_back(std::move(o));
    if (index) *index = static_cast<int>(m->model.objectives.size());
    touched(m);
  });
}

int popso_clear_objectives(popso_model* m) {
  return guarded(m, [&] {
    m->model.objectives.clear();
    touched(m);
  });
}

int popso_num_objectives(const popso_model* m) { return m ? m->model.num_objectives() : -1; }

int popso_get_objective_value(const popso_model* m, int index, double* value) {
  return guarded(m, [&] {
    if (!value) fail(POPSO_ERROR_INVALID_ARGUMENT, "null output pointer");
    if (index < 0 || index >= m->model.num_objectives())
      fail(POPSO_ERROR_INVALID_ARGUMENT, "objective index " + std::to_string(index) + " out of range");
    check_solved(m);
    if (!m->result.has_solution) fail(POPSO_ERROR_NO_SOLUTION, "no feasible solution is known");
    *value = m->model.objective_value(index, m->result.x);
  });
}

namespace {

void add_q_impl(popso_model* m, int nnz, const int* rows, const int* cols, const double* vals, const char* what) {
  if (nnz < 0 || (nnz > 0 && (!rows || !cols || !vals))) fail(POPSO_ERROR_INVALID_ARGUMENT, std::string(what) + ": invalid entries");
  for (int k = 0; k < nnz; ++k) {
    check_var(m, rows[k]);
    check_var(m, cols[k]);
    if (!std::isfinite(vals[k])) fail(POPSO_ERROR_INVALID_ARGUMENT, std::string(what) + ": coefficient must be finite");
  }
  for (int k = 0; k < nnz; ++k) m->model.add_q(rows[k], cols[k], vals[k]);
}

// The Hessian in canonical form (lower triangle, duplicates merged, column-major order).
void compact_q(popso_model* m) {
  milp::Model& md = m->model;
  const milp::SparseMatrix Q = md.q_lower();
  md.q_row.clear();
  md.q_col.clear();
  md.q_val.clear();
  for (Index j = 0; j < Q.num_major; ++j) {
    for (Index k = Q.begin(j); k < Q.end(j); ++k) {
      md.q_row.push_back(Q.index[k]);
      md.q_col.push_back(j);
      md.q_val.push_back(Q.value[k]);
    }
  }
}

}  // namespace

int popso_set_q(popso_model* m, int nnz, const int* rows, const int* cols, const double* vals) {
  return guarded(m, [&] {
    milp::Model backup_q = milp::Model{};
    backup_q.q_row.swap(m->model.q_row);
    backup_q.q_col.swap(m->model.q_col);
    backup_q.q_val.swap(m->model.q_val);
    try {
      add_q_impl(m, nnz, rows, cols, vals, "set_q");
    } catch (...) {
      m->model.q_row.swap(backup_q.q_row);
      m->model.q_col.swap(backup_q.q_col);
      m->model.q_val.swap(backup_q.q_val);
      throw;
    }
    compact_q(m);
    touched(m);
  });
}

int popso_add_q_entries(popso_model* m, int nnz, const int* rows, const int* cols, const double* vals) {
  return guarded(m, [&] {
    add_q_impl(m, nnz, rows, cols, vals, "add_q_entries");
    compact_q(m);
    touched(m);
  });
}

int popso_num_q_nonzeros(popso_model* m) {
  if (!m) return -1;
  int n = -1;
  if (guarded(m, [&] {
        compact_q(m);
        n = static_cast<int>(m->model.q_val.size());
      }) != POPSO_OK)
    return -1;
  return n;
}

int popso_get_q(popso_model* m, int size, int* rows, int* cols, double* vals) {
  return guarded(m, [&] {
    compact_q(m);
    const int n = static_cast<int>(m->model.q_val.size());
    if (size < n) fail(POPSO_ERROR_INVALID_ARGUMENT, "get_q: the arrays need " + std::to_string(n) + " entries");
    if (n > 0 && (!rows || !cols || !vals)) fail(POPSO_ERROR_INVALID_ARGUMENT, "get_q: null array");
    for (int k = 0; k < n; ++k) {
      rows[k] = m->model.q_row[k];
      cols[k] = m->model.q_col[k];
      vals[k] = m->model.q_val[k];
    }
  });
}

namespace {

// The quadratic terms of one row, duplicates merged and zeros dropped, sorted by (i, j) with i >= j.
using QrowTerms = std::vector<std::pair<std::pair<Index, Index>, double>>;
void qrow_terms(const popso_model* m, int row, QrowTerms& t) {
  t.clear();
  const milp::Model& md = m->model;
  for (size_t k = 0; k < md.qr_val.size(); ++k) {
    if (md.qr_row[k] == row) t.push_back({{md.qr_i[k], md.qr_j[k]}, md.qr_val[k]});
  }
  std::sort(t.begin(), t.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  size_t w = 0;
  for (size_t k = 0; k < t.size(); ++k) {
    if (w > 0 && t[w - 1].first == t[k].first) {
      t[w - 1].second += t[k].second;
    } else {
      t[w++] = t[k];
    }
  }
  t.resize(w);
  t.erase(std::remove_if(t.begin(), t.end(), [](const auto& e) { return e.second == 0.0; }), t.end());
}

}  // namespace

int popso_add_qrow_terms(popso_model* m, int row, int nnz, const int* vars1, const int* vars2, const double* vals) {
  return guarded(m, [&] {
    check_row(m, row);
    if (nnz < 0 || (nnz > 0 && (!vars1 || !vars2 || !vals))) {
      fail(POPSO_ERROR_INVALID_ARGUMENT, "add_qrow_terms: invalid entries");
    }
    for (int k = 0; k < nnz; ++k) {
      check_var(m, vars1[k]);
      check_var(m, vars2[k]);
      if (!std::isfinite(vals[k])) fail(POPSO_ERROR_INVALID_ARGUMENT, "add_qrow_terms: coefficient must be finite");
    }
    for (int k = 0; k < nnz; ++k) {
      if (vals[k] != 0.0) m->model.add_qrow(row, vars1[k], vars2[k], vals[k]);
    }
    touched(m);
  });
}

int popso_num_qrow_terms(const popso_model* m, int row) {
  if (!m || row < 0 || row >= m->model.num_rows()) return -1;
  QrowTerms t;
  qrow_terms(m, row, t);
  return static_cast<int>(t.size());
}

int popso_get_qrow_terms(const popso_model* m, int row, int size, int* vars1, int* vars2, double* vals) {
  return guarded(m, [&] {
    check_row(m, row);
    QrowTerms t;
    qrow_terms(m, row, t);
    const int n = static_cast<int>(t.size());
    if (size < n) fail(POPSO_ERROR_INVALID_ARGUMENT, "get_qrow_terms: the arrays need " + std::to_string(n) + " entries");
    if (n > 0 && (!vars1 || !vars2 || !vals)) fail(POPSO_ERROR_INVALID_ARGUMENT, "get_qrow_terms: null array");
    for (int k = 0; k < n; ++k) {
      vars1[k] = t[k].first.first;
      vars2[k] = t[k].first.second;
      vals[k] = t[k].second;
    }
  });
}

int popso_num_vars(const popso_model* m) { return m ? m->model.num_cols() : -1; }
int popso_num_rows(const popso_model* m) { return m ? m->model.num_rows() : -1; }
int popso_num_sos(const popso_model* m) { return m ? static_cast<int>(m->model.sos.size()) : -1; }
int popso_num_indicators(const popso_model* m) { return m ? static_cast<int>(m->model.indicators.size()) : -1; }

long long popso_num_nonzeros(popso_model* m) {
  if (!m) return -1;
  long long n = -1;
  if (guarded(m, [&] {
        flush_matrix(m);
        n = m->model.num_nonzeros();
      }) != POPSO_OK)
    return -1;
  return n;
}

int popso_get_objective_sense(const popso_model* m) {
  if (!m) return 0;
  return m->model.sense == milp::ObjSense::Maximize ? POPSO_MAXIMIZE : POPSO_MINIMIZE;
}

double popso_get_objective_constant(const popso_model* m) { return m ? m->model.obj_offset : kNaN; }

int popso_get_var(const popso_model* m, int var, double* lb, double* ub, double* obj, int* vtype) {
  return guarded(m, [&] {
    check_var(m, var);
    if (lb) *lb = m->model.col_lb[var];
    if (ub) *ub = m->model.col_ub[var];
    if (obj) *obj = m->model.obj[var];
    if (vtype) *vtype = from_var_type(m->model.col_type[var]);
  });
}

int popso_get_row(const popso_model* m, int row, double* lb, double* ub) {
  return guarded(m, [&] {
    check_row(m, row);
    if (lb) *lb = m->model.row_lb[row];
    if (ub) *ub = m->model.row_ub[row];
  });
}

const char* popso_get_var_name(const popso_model* m, int var) {
  if (!m || var < 0 || var >= m->model.num_cols()) return nullptr;
  return m->model.col_names[var].c_str();
}

const char* popso_get_row_name(const popso_model* m, int row) {
  if (!m || row < 0 || row >= m->model.num_rows()) return nullptr;
  return m->model.row_names[row].c_str();
}

int popso_find_var(popso_model* m, const char* name) {
  if (!m || !name) return -1;
  int found = -1;
  guarded(m, [&] {
    if (!m->name_index_valid) {
      m->name_index.clear();
      m->name_index.reserve(m->model.col_names.size());
      // First occurrence wins for duplicate names.
      for (Index j = 0; j < m->model.num_cols(); ++j) m->name_index.emplace(m->model.col_names[j], j);
      m->name_index_valid = true;
    }
    auto it = m->name_index.find(name);
    if (it != m->name_index.end()) found = it->second;
  });
  return found;
}

int popso_set_param(popso_model* m, const char* name, double value) {
  return guarded(m, [&] {
    if (!name) fail(POPSO_ERROR_INVALID_ARGUMENT, "null parameter name");
    const std::string n = name;
    milp::SolverParams& p = m->params;
    auto bad = [&](const char* why) { fail(POPSO_ERROR_INVALID_ARGUMENT, "parameter " + n + ": " + why); };
    if (std::isnan(value)) bad("NaN value");
    auto as_flag = [&]() -> bool {
      if (value != 0.0 && value != 1.0) bad("must be 0 or 1");
      return value == 1.0;
    };
    auto as_int = [&](double lo, double hi) -> long long {
      if (!is_integral_value(value) || value < lo || value > hi) bad("must be an integer in range");
      return static_cast<long long>(value);
    };
    if (n == "time_limit") {
      if (value < 0) bad("must be nonnegative");
      p.time_limit = value >= POPSO_INFINITY ? milp::kInf : value;
    } else if (n == "threads") {
      p.threads = static_cast<int>(as_int(0, 4096));
    } else if (n == "mip_gap") {
      if (value < 0 || !std::isfinite(value)) bad("must be a nonnegative number");
      p.mip_rel_gap = value;
    } else if (n == "mip_abs_gap") {
      if (value < 0 || !std::isfinite(value)) bad("must be a nonnegative number");
      p.mip_abs_gap = value;
    } else if (n == "seed") {
      p.seed = static_cast<unsigned>(as_int(0, 4294967295.0));
    } else if (n == "deterministic") {
      p.deterministic = as_flag();
    } else if (n == "verbosity") {
      p.verbosity = static_cast<int>(as_int(0, 3));
    } else if (n == "mem_limit") {
      p.mem_limit_mb = as_int(1, 1e15);
    } else if (n == "node_limit") {
      p.node_limit = as_int(-1, 9e18);
    } else if (n == "presolve") {
      p.presolve = as_flag();
    } else if (n == "racing") {
      p.racing = static_cast<int>(as_int(1, 4096));
    } else if (n == "lp_method") {
      p.lp_method = static_cast<milp::LpMethod>(as_int(0, 2));
    } else if (n == "duals") {
      p.duals = static_cast<int>(as_int(0, 2));
    } else {
      fail(POPSO_ERROR_INVALID_ARGUMENT,
           "unknown parameter '" + n +
               "' (known: time_limit, threads, mip_gap, mip_abs_gap, seed, deterministic, verbosity, mem_limit, "
               "node_limit, presolve, racing, lp_method, duals)");
    }
  });
}

int popso_get_param(const popso_model* m, const char* name, double* value) {
  return guarded(m, [&] {
    if (!name || !value) fail(POPSO_ERROR_INVALID_ARGUMENT, "null argument");
    const std::string n = name;
    const milp::SolverParams& p = m->params;
    if (n == "time_limit") *value = p.time_limit;
    else if (n == "threads") *value = p.threads;
    else if (n == "mip_gap") *value = p.mip_rel_gap;
    else if (n == "mip_abs_gap") *value = p.mip_abs_gap;
    else if (n == "seed") *value = p.seed;
    else if (n == "deterministic") *value = p.deterministic ? 1.0 : 0.0;
    else if (n == "verbosity") *value = p.verbosity;
    else if (n == "mem_limit") *value = static_cast<double>(p.mem_limit_mb);
    else if (n == "node_limit") *value = static_cast<double>(p.node_limit);
    else if (n == "presolve") *value = p.presolve ? 1.0 : 0.0;
    else if (n == "racing") *value = p.racing;
    else if (n == "lp_method") *value = static_cast<double>(static_cast<int>(p.lp_method));
    else if (n == "duals") *value = p.duals;
    else fail(POPSO_ERROR_INVALID_ARGUMENT, "unknown parameter '" + n + "'");
  });
}

int popso_set_start(popso_model* m, int nnz, const int* vars, const double* vals) {
  return guarded(m, [&] {
    const Index n = m->model.num_cols();
    if (nnz < 0 || (nnz > 0 && !vals)) fail(POPSO_ERROR_INVALID_ARGUMENT, "set_start: invalid entries");
    std::vector<double> x(n, kNaN);
    if (!vars) {
      if (nnz != n) fail(POPSO_ERROR_INVALID_ARGUMENT, "set_start: a dense start needs one value per variable");
      for (Index j = 0; j < n; ++j) x[j] = vals[j];
    } else {
      for (int k = 0; k < nnz; ++k) {
        check_var(m, vars[k]);
        x[vars[k]] = vals[k];
      }
    }
    for (double v : x) {
      if (std::isinf(v)) fail(POPSO_ERROR_INVALID_ARGUMENT, "set_start: infinite value");
    }
    m->start = std::move(x);
  });
}

int popso_read_start(popso_model* m, const char* filename) {
  return guarded(m, [&] {
    check_file_readable(filename);
    std::vector<double> x;
    try {
      x = milp::read_solution(filename, m->model);
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const std::exception& e) {
      fail(POPSO_ERROR_PARSE, e.what());
    }
    for (double v : x) {
      if (std::isinf(v)) fail(POPSO_ERROR_PARSE, std::string(filename) + ": infinite value in start");
    }
    m->start = std::move(x);
  });
}

int popso_clear_start(popso_model* m) {
  return guarded(m, [&] { m->start.clear(); });
}

int popso_solve(popso_model* m) {
  return guarded(m, [&] {
    touched(m);
    flush_matrix(m);
    m->model.validate();
    std::vector<double> start = m->start;
    if (!start.empty()) start.resize(m->model.num_cols(), kNaN);  // variables added after set_start
    milp::SolverParams params = m->params;
    m->interrupt.store(false);
    params.interrupt = &m->interrupt;
    if (m->callback) {
      const popso_incumbent_callback cb = m->callback;
      void* data = m->callback_data;
      params.on_incumbent = [cb, data](const std::vector<double>& x, double objective, double bound) {
        return cb(data, objective, bound, x.data(), static_cast<int>(x.size())) != 0;
      };
    }
    m->result = milp::solve(m->model, params, start);
    m->solved = true;
  });
}

int popso_interrupt(popso_model* m) {
  // Async-signal-safe: no error message, no allocation, one lock-free atomic store.
  if (!m) return POPSO_ERROR_INVALID_ARGUMENT;
  m->interrupt.store(true);
  return POPSO_OK;
}

int popso_set_incumbent_callback(popso_model* m, popso_incumbent_callback cb, void* user_data) {
  return guarded(m, [&] {
    m->callback = cb;
    m->callback_data = cb ? user_data : nullptr;
  });
}

int popso_get_status(const popso_model* m) {
  if (!m || !m->solved) return POPSO_STATUS_NOT_SOLVED;
  return from_status(m->result.status);
}

int popso_has_solution(const popso_model* m) { return m && m->solved && m->result.has_solution ? 1 : 0; }

int popso_get_objective(const popso_model* m, double* value) {
  return guarded(m, [&] {
    if (!value) fail(POPSO_ERROR_INVALID_ARGUMENT, "null output pointer");
    check_solved(m);
    if (!m->result.has_solution) fail(POPSO_ERROR_NO_SOLUTION, "no feasible solution is known");
    *value = m->result.objective;
  });
}

int popso_get_bound(const popso_model* m, double* value) {
  return guarded(m, [&] {
    if (!value) fail(POPSO_ERROR_INVALID_ARGUMENT, "null output pointer");
    check_solved(m);
    *value = m->result.bound;
  });
}

int popso_get_gap(const popso_model* m, double* value) {
  return guarded(m, [&] {
    if (!value) fail(POPSO_ERROR_INVALID_ARGUMENT, "null output pointer");
    check_solved(m);
    *value = gap_of(m->result);
  });
}

int popso_get_node_count(const popso_model* m, long long* value) {
  return guarded(m, [&] {
    if (!value) fail(POPSO_ERROR_INVALID_ARGUMENT, "null output pointer");
    check_solved(m);
    *value = m->result.stats.nodes;
  });
}

int popso_get_lp_iterations(const popso_model* m, long long* value) {
  return guarded(m, [&] {
    if (!value) fail(POPSO_ERROR_INVALID_ARGUMENT, "null output pointer");
    check_solved(m);
    *value = m->result.stats.lp_iterations;
  });
}

int popso_get_solve_time(const popso_model* m, double* seconds) {
  return guarded(m, [&] {
    if (!seconds) fail(POPSO_ERROR_INVALID_ARGUMENT, "null output pointer");
    check_solved(m);
    *seconds = m->result.stats.time;
  });
}

int popso_get_solution(const popso_model* m, double* x, int size) {
  return guarded(m, [&] {
    check_solved(m);
    if (!m->result.has_solution) fail(POPSO_ERROR_NO_SOLUTION, "no feasible solution is known");
    const Index n = m->model.num_cols();
    if (!x || size < n) fail(POPSO_ERROR_INVALID_ARGUMENT, "get_solution: buffer smaller than the number of variables");
    for (Index j = 0; j < n; ++j) x[j] = m->result.x[j];
  });
}

int popso_has_duals(const popso_model* m) {
  if (!m || !m->solved || !m->result.has_duals) return 0;
  return m->result.fixed_mip_duals ? 2 : 1;
}

int popso_has_basis(const popso_model* m) {
  if (!m || !m->solved || !m->result.has_duals) return 0;
  const milp::SolveResult& r = m->result;
  const bool sized = r.col_basis.size() == r.reduced_cost.size() && r.row_basis.size() == r.row_dual.size();
  return sized && !(r.col_basis.empty() && r.row_basis.empty()) ? 1 : 0;
}

int popso_get_basis(const popso_model* m, int* var_status, int* row_status) {
  return guarded(m, [&] {
    check_solved(m);
    if (!popso_has_basis(m)) fail(POPSO_ERROR_NO_SOLUTION, "no basis (LPs and fixed-MIP duals only, see popso_has_duals)");
    if (var_status) {
      for (size_t j = 0; j < m->result.col_basis.size(); ++j) var_status[j] = m->result.col_basis[j];
    }
    if (row_status) {
      for (size_t i = 0; i < m->result.row_basis.size(); ++i) row_status[i] = m->result.row_basis[i];
    }
  });
}

int popso_get_duals(const popso_model* m, double* y, int size) {
  return guarded(m, [&] {
    check_solved(m);
    if (!m->result.has_duals)
      fail(POPSO_ERROR_NO_SOLUTION, "no dual values (LPs and continuous QPs solved to optimality; MILPs with the duals parameter 2)");
    const Index rows = m->model.num_rows();
    if (!y || size < rows) fail(POPSO_ERROR_INVALID_ARGUMENT, "get_duals: buffer smaller than the number of rows");
    for (Index i = 0; i < rows; ++i) y[i] = m->result.row_dual[i];
  });
}

int popso_get_reduced_costs(const popso_model* m, double* d, int size) {
  return guarded(m, [&] {
    check_solved(m);
    if (!m->result.has_duals)
      fail(POPSO_ERROR_NO_SOLUTION, "no reduced costs (LPs and continuous QPs solved to optimality; MILPs with the duals parameter 2)");
    const Index n = m->model.num_cols();
    if (!d || size < n) fail(POPSO_ERROR_INVALID_ARGUMENT, "get_reduced_costs: buffer smaller than the number of variables");
    for (Index j = 0; j < n; ++j) d[j] = m->result.reduced_cost[j];
  });
}

int popso_write_solution(const popso_model* m, const char* filename) {
  return guarded(m, [&] {
    if (!filename) fail(POPSO_ERROR_INVALID_ARGUMENT, "null file name");
    check_solved(m);
    try {
      milp::write_solution(filename, m->model, m->result);
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const std::exception& e) {
      fail(POPSO_ERROR_IO, e.what());
    }
  });
}

}  // extern "C"
