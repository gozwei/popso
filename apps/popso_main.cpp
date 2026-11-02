// popso command-line interface: read an MPS or LP model, solve it, report, and
// optionally write a .sol file (common "name value" format).

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "milp/core/model.hpp"
#include "milp/core/timer.hpp"
#include "milp/io/lp_reader.hpp"
#include "milp/io/mps_reader.hpp"
#include "milp/io/solution.hpp"
#include "milp/solver/solver.hpp"

namespace {

void print_usage(std::FILE* out) {
  std::fprintf(out,
               "usage: popso <model.mps|model.lp> [options]\n"
               "options:\n"
               "  --sol <file>          write solution file (\"name value\" .sol format)\n"
               "  --start <file>        starting solution (same format; may be partial)\n"
               "  --format <mps|lp>     input format (default: from the file extension, .lp = LP, else MPS)\n"
               "  --time-limit <sec>    wall-clock time limit (default: none)\n"
               "  --threads <n>         number of threads (0 = all cores, default)\n"
               "  --verbosity <n>       0 silent, 1 summary (default), 2 progress, 3 debug\n"
               "  --gap <rel>           relative MIP gap tolerance (default 1e-4)\n"
               "  --no-presolve         disable presolve\n"
               "  --mem-limit <MB>      approximate search-tree memory cap (default 4096)\n"
               "  --seed <n>            random seed (default 0)\n"
               "  --node-limit <n>      stop after n branch-and-bound nodes (default: none)\n"
               "  --deterministic       reproducible parallel search (slower; default: opportunistic)\n"
               "  --racing <k>          run k differently seeded searches concurrently (default 1)\n"
               "  --version             print the version\n"
               "  --help                show this message\n");
}

[[noreturn]] void usage_error(const std::string& msg) {
  std::fprintf(stderr, "popso: %s\n", msg.c_str());
  print_usage(stderr);
  std::exit(2);
}

double parse_double_arg(const char* opt, const char* s) {
  char* end = nullptr;
  const double v = std::strtod(s, &end);
  if (end == s || *end != '\0' || std::isnan(v)) usage_error(std::string("invalid value for ") + opt + ": '" + s + "'");
  return v;
}

// Input format from --format, else from the extension (.lp, case-insensitive).
bool is_lp_format(const std::string& format, const std::string& file) {
  if (!format.empty()) return format == "lp";
  if (file.size() < 3) return false;
  std::string ext = file.substr(file.size() - 3);
  for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return ext == ".lp";
}

long long parse_int_arg(const char* opt, const char* s) {
  char* end = nullptr;
  const long long v = std::strtoll(s, &end, 10);
  if (end == s || *end != '\0') usage_error(std::string("invalid value for ") + opt + ": '" + s + "'");
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, sol_path, start_path, format;
  milp::SolverParams params;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    auto value = [&]() -> const char* {
      if (i + 1 >= argc) usage_error(std::string("missing value for ") + a);
      return argv[++i];
    };
    if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) {
      print_usage(stdout);
      return 0;
    } else if (!std::strcmp(a, "--version")) {
      std::printf("popso %s\n", POPSO_VERSION);
      return 0;
    } else if (!std::strcmp(a, "--sol")) {
      sol_path = value();
    } else if (!std::strcmp(a, "--start")) {
      start_path = value();
    } else if (!std::strcmp(a, "--format")) {
      format = value();
      if (format != "mps" && format != "lp") usage_error("format must be 'mps' or 'lp'");
    } else if (!std::strcmp(a, "--time-limit")) {
      params.time_limit = parse_double_arg(a, value());
      if (params.time_limit < 0) usage_error("time limit must be nonnegative");
    } else if (!std::strcmp(a, "--threads")) {
      const long long t = parse_int_arg(a, value());
      if (t < 0 || t > 4096) usage_error("threads must be in [0, 4096]");
      params.threads = static_cast<int>(t);
    } else if (!std::strcmp(a, "--verbosity")) {
      params.verbosity = static_cast<int>(parse_int_arg(a, value()));
    } else if (!std::strcmp(a, "--gap")) {
      params.mip_rel_gap = parse_double_arg(a, value());
      if (params.mip_rel_gap < 0) usage_error("gap must be nonnegative");
    } else if (!std::strcmp(a, "--no-presolve")) {
      params.presolve = false;
    } else if (!std::strcmp(a, "--opportunistic")) {
      params.deterministic = false;
    } else if (!std::strcmp(a, "--racing")) {
      params.racing = static_cast<int>(parse_int_arg(a, value()));
      if (params.racing < 1) usage_error("racing must be >= 1");
    } else if (!std::strcmp(a, "--deterministic")) {
      params.deterministic = true;
    } else if (!std::strcmp(a, "--mem-limit")) {
      params.mem_limit_mb = parse_int_arg(a, value());
      if (params.mem_limit_mb <= 0) usage_error("memory limit must be positive");
    } else if (!std::strcmp(a, "--node-limit")) {
      params.node_limit = parse_int_arg(a, value());
    } else if (!std::strcmp(a, "--seed")) {
      const long long s = parse_int_arg(a, value());
      if (s < 0) usage_error("seed must be nonnegative");
      params.seed = static_cast<unsigned>(s);
    } else if (a[0] == '-' && a[1] == '-') {
      usage_error(std::string("unknown option ") + a);
    } else if (model_path.empty()) {
      model_path = a;
    } else {
      usage_error(std::string("unexpected argument ") + a);
    }
  }
  if (model_path.empty()) usage_error("no model file given");

  milp::Timer timer;
  milp::Model model;
  try {
    model = is_lp_format(format, model_path) ? milp::read_lp(model_path) : milp::read_mps(model_path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "milp: error reading model: %s\n", e.what());
    return 1;
  }
  const double read_time = timer.elapsed();
  const bool quiet = params.verbosity <= 0;
  if (!quiet) {
    std::printf("Model %s: %d rows, %d columns (%d integer), %d nonzeros, read in %.3fs\n",
                model.name.empty() ? model_path.c_str() : model.name.c_str(), model.num_rows(), model.num_cols(),
                model.num_integers(), model.num_nonzeros(), read_time);
  }

  std::vector<double> start;
  if (!start_path.empty()) {
    try {
      start = milp::read_solution(start_path, model);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "popso: error reading start solution: %s\n", e.what());
      return 1;
    }
  }

  milp::SolveResult result;
  try {
    result = milp::solve(model, params, start);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "milp: solver error: %s\n", e.what());
    return 1;
  }

  if (!quiet) {
    std::printf("Status: %s\n", milp::to_string(result.status));
    if (result.has_solution) std::printf("Objective: %.10g\n", result.objective);
    else std::printf("Objective: -\n");
    std::printf("Bound: %.10g\n", result.bound);
    double gap = milp::kInf;
    if (result.has_solution && std::isfinite(result.bound)) {
      const double diff = std::abs(result.objective - result.bound);
      gap = diff == 0.0 ? 0.0 : diff / std::max(std::abs(result.objective), 1e-10);
    }
    if (std::isfinite(gap)) std::printf("Gap: %.4f%%\n", 100.0 * gap);
    else std::printf("Gap: inf\n");
    std::printf("Nodes: %lld\n", result.stats.nodes);
    std::printf("LP iterations: %lld\n", result.stats.lp_iterations);
    std::printf("Time: %.3fs (total %.3fs)\n", result.stats.time, timer.elapsed());
  }

  if (!sol_path.empty()) {
    try {
      milp::write_solution(sol_path, model, result);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "milp: %s\n", e.what());
      return 1;
    }
  }
  return 0;
}
