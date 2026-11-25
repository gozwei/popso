// popso command-line interface: read an MPS or LP model, solve it, report, and
// optionally write a .sol file (common "name value" format), a JSON profile of the
// model and the run (--profile), or an anonymized copy of the model (--write-anon).

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "milp/core/model.hpp"
#include "milp/core/timer.hpp"
#include "milp/io/anonymize.hpp"
#include "milp/io/cbf_reader.hpp"
#include "milp/io/lp_reader.hpp"
#include "milp/io/mps_reader.hpp"
#include "milp/io/solution.hpp"
#include "milp/search/node_store.hpp"
#include "milp/solver/profile.hpp"
#include "milp/solver/solver.hpp"

namespace {

// Ctrl-C (SIGINT) stops the solve with the best solution so far; a second one ends the process.
std::atomic<bool> g_interrupt{false};

void on_sigint(int) {
  g_interrupt.store(true);
  std::signal(SIGINT, SIG_DFL);
}

void print_usage(std::FILE* out) {
  std::fprintf(out,
               "usage: popso <model.mps|model.lp|model.cbf>[.gz] [options]\n"
               "options:\n"
               "  --sol <file>          write solution file (\"name value\" .sol format)\n"
               "  --start <file>        starting solution (same format; may be partial)\n"
               "  --duals <file>        write row duals and reduced costs (\"row|col name value\"): LPs and QPs;\n"
               "                        for a MILP those of the LP with the integers fixed at the solution\n"
               "  --format <mps|lp|cbf> input format (default: from the file extension, .lp = LP, .cbf = CBF\n"
               "                        (conic benchmark format: linear, quadratic and rotated cones), else MPS;\n"
               "                        gzipped files (.mps.gz, .lp.gz) are detected and decompressed)\n"
               "  --time-limit <sec>    wall-clock time limit (default: none)\n"
               "  --threads <n>         number of threads (0 = all cores, default)\n"
               "  --verbosity <n>       0 silent, 1 summary (default), 2 progress, 3 debug\n"
               "  --gap <rel>           relative MIP gap tolerance (default 1e-4)\n"
               "  --abs-gap <abs>       absolute MIP gap tolerance (default 1e-10)\n"
               "  --no-presolve         disable presolve\n"
               "  --symmetry <on|off>   symmetry handling (default on)\n"
               "  --lp-method <m>       LP algorithm for pure LPs and the MIP root LP: auto (default),\n"
               "                        simplex (dual simplex) or barrier (interior point + crossover)\n"
               "  --mem-limit <MB>      approximate search-tree memory cap (default 4096)\n"
               "  --seed <n>            random seed (default 0)\n"
               "  --node-limit <n>      stop after n branch-and-bound nodes (default: none)\n"
               "  --deterministic       reproducible parallel search (slower; default: opportunistic)\n"
               "  --racing <k>          run k differently seeded searches concurrently (default 1)\n"
               "  --profile <file>      after the solve (also LPs, QPs and stops at a limit), write a JSON profile:\n"
               "                        sizes, column types, row classes (MIPLIB 2017 classification), coefficient\n"
               "                        ranges, presolve and run statistics, incumbent trace (time, objective,\n"
               "                        source); aggregated only: no names, no per-row or per-column data, no paths\n"
               "  --write-anon <file>   write the model as free MPS with generic names (columns x1.., rows c1..,\n"
               "                        SOS sets s1.., indicators i1.., model \"anon\"), rows and columns in a\n"
               "                        shuffled order and every number unchanged (a warning counts the ranged\n"
               "                        rows MPS cannot express exactly: off by an ulp), then exit without solving\n"
               "  --anon-map <file>     with --write-anon: also write the private name mapping, one line\n"
               "                        \"<new name> <original name>\" per column, row, SOS set and indicator\n"
               "                        (keep it: it maps a solution of the copy back; never written by default)\n"
               "  --anon-seed <n>       with --write-anon: seed of the shuffle (default 1; same seed, same file)\n"
               "  --anon-keep-order     with --write-anon: keep the row and column order (only the names change)\n"
               "  --solve               with --write-anon: solve the original model as well (needed for --sol,\n"
               "                        --duals, --start and --profile together with --write-anon)\n"
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

// Input format from --format, else from the extension (.lp / .cbf, also .gz, case-insensitive; anything else MPS).
std::string input_format(const std::string& format, const std::string& file) {
  if (!format.empty()) return format;
  std::string name = file;
  for (char& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (name.size() >= 3 && name.compare(name.size() - 3, 3, ".gz") == 0) name.resize(name.size() - 3);
  if (name.size() >= 3 && name.compare(name.size() - 3, 3, ".lp") == 0) return "lp";
  if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".cbf") == 0) return "cbf";
  return "mps";
}

long long parse_int_arg(const char* opt, const char* s) {
  char* end = nullptr;
  const long long v = std::strtoll(s, &end, 10);
  if (end == s || *end != '\0') usage_error(std::string("invalid value for ") + opt + ": '" + s + "'");
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, sol_path, start_path, duals_path, format, profile_path, anon_path, anon_map_path;
  milp::AnonOptions anon;
  bool anon_option = false, solve_too = false;
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
    } else if (!std::strcmp(a, "--duals")) {
      duals_path = value();
      params.duals = 2;
    } else if (!std::strcmp(a, "--start")) {
      start_path = value();
    } else if (!std::strcmp(a, "--format")) {
      format = value();
      if (format != "mps" && format != "lp" && format != "cbf") usage_error("format must be 'mps', 'lp' or 'cbf'");
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
    } else if (!std::strcmp(a, "--abs-gap")) {
      params.mip_abs_gap = parse_double_arg(a, value());
      if (params.mip_abs_gap < 0) usage_error("abs-gap must be nonnegative");
    } else if (!std::strcmp(a, "--no-presolve")) {
      params.presolve = false;
    } else if (!std::strcmp(a, "--symmetry")) {
      const std::string m = value();
      if (m == "on") params.symmetry = true;
      else if (m == "off") params.symmetry = false;
      else usage_error("symmetry must be 'on' or 'off'");
    } else if (!std::strcmp(a, "--lp-method")) {
      const std::string m = value();
      if (m == "auto") params.lp_method = milp::LpMethod::Auto;
      else if (m == "simplex") params.lp_method = milp::LpMethod::Simplex;
      else if (m == "barrier") params.lp_method = milp::LpMethod::Barrier;
      else usage_error("lp-method must be 'auto', 'simplex' or 'barrier'");
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
    } else if (!std::strcmp(a, "--profile")) {
      profile_path = value();
      params.trace_incumbents = true;
    } else if (!std::strcmp(a, "--write-anon")) {
      anon_path = value();
    } else if (!std::strcmp(a, "--anon-map")) {
      anon_map_path = value();
      anon_option = true;
    } else if (!std::strcmp(a, "--anon-seed")) {
      const long long s = parse_int_arg(a, value());
      if (s < 0) usage_error("anon-seed must be nonnegative");
      anon.seed = static_cast<uint64_t>(s);
      anon_option = true;
    } else if (!std::strcmp(a, "--anon-keep-order")) {
      anon.keep_order = true;
      anon_option = true;
    } else if (!std::strcmp(a, "--solve")) {
      solve_too = true;
    } else if (a[0] == '-' && a[1] == '-') {
      usage_error(std::string("unknown option ") + a);
    } else if (model_path.empty()) {
      model_path = a;
    } else {
      usage_error(std::string("unexpected argument ") + a);
    }
  }
  if (model_path.empty()) usage_error("no model file given");
  if (anon_path.empty() && (anon_option || solve_too))
    usage_error("--anon-map, --anon-seed, --anon-keep-order and --solve go with --write-anon");
  // --write-anon alone does not solve: options that need a solve must say so with --solve.
  if (!anon_path.empty() && !solve_too && (!sol_path.empty() || !duals_path.empty() || !start_path.empty() ||
                                           !profile_path.empty()))
    usage_error("--write-anon exits without solving; add --solve for --sol, --duals, --start or --profile");

  milp::Timer timer;
  milp::Model model;
  try {
    const std::string fmt = input_format(format, model_path);
    model = fmt == "lp" ? milp::read_lp(model_path) : fmt == "cbf" ? milp::read_cbf(model_path) : milp::read_mps(model_path);
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

  if (!anon_path.empty()) {
    try {
      milp::AnonMap map;
      const milp::Model copy = milp::anonymize(model, anon, &map);
      milp::Index inexact = 0;
      milp::write_mps(copy, anon_path, &inexact);
      if (!anon_map_path.empty()) milp::write_anon_map(anon_map_path, model, map);
      if (inexact > 0) {
        std::fprintf(stderr, "popso: warning: %d ranged rows are off by up to an ulp in the MPS file (no MPS range "
                     "gives both sides exactly)\n", inexact);
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "popso: error writing the anonymized model: %s\n", e.what());
      return 1;
    }
    if (!quiet) {
      std::printf("Anonymized model written to %s%s%s\n", anon_path.c_str(), anon_map_path.empty() ? "" : ", map to ",
                  anon_map_path.c_str());
    }
    if (!solve_too) return 0;
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
  params.interrupt = &g_interrupt;
  std::signal(SIGINT, on_sigint);
  // Past the time limit the process only reports and exits: the open nodes are not freed one by one first.
  if (std::isfinite(params.time_limit) && params.time_limit < 1e9) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(params.time_limit);
    milp::bnb::g_leak_pools_after_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count();
  }
  try {
    result = milp::solve(model, params, start);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "milp: solver error: %s\n", e.what());
    return 1;
  }
  std::signal(SIGINT, SIG_DFL);

  if (!quiet) {
    std::printf("Status: %s\n", milp::to_string(result.status));
    if (result.has_solution) std::printf("Objective: %.10g\n", result.objective);
    else std::printf("Objective: -\n");
    // Several objectives: the value of each (the line above repeats the first), in the file's order.
    if (result.has_solution && !model.objectives.empty()) {
      for (int k = 0; k < model.num_objectives(); ++k) {
        const std::string& nm = k == 0 ? model.obj_name : model.objectives[static_cast<size_t>(k - 1)].name;
        const int prio = k == 0 ? model.obj_priority : model.objectives[static_cast<size_t>(k - 1)].priority;
        std::printf("Objective %d (%s, priority %d): %.10g\n", k, nm.empty() ? "-" : nm.c_str(), prio,
                    result.objective_values[static_cast<size_t>(k)]);
      }
    }
    std::printf("Bound: %.10g\n", result.bound);
    double gap = milp::kInf;
    if (result.has_solution && std::isfinite(result.bound)) {
      const double diff = std::abs(result.objective - result.bound);
      gap = diff == 0.0 ? 0.0 : diff / std::max(std::abs(result.objective), 1e-10);
    }
    // Several objectives: the first objective's distance to its bound is the room the later stages used.
    if (!model.objectives.empty()) std::printf("Gap: -\n");
    else if (std::isfinite(gap)) std::printf("Gap: %.4f%%\n", 100.0 * gap);
    else std::printf("Gap: inf\n");
    std::printf("Nodes: %lld\n", result.stats.nodes);
    std::printf("LP iterations: %lld\n", result.stats.lp_iterations);
    if (result.stats.restarts + result.stats.tree_restarts > 0)
      std::printf("Restarts: %d root, %d in-tree\n", result.stats.restarts, result.stats.tree_restarts);
    std::printf("Time: %.3fs (total %.3fs)\n", result.stats.time, timer.elapsed());
  }

  if (!duals_path.empty()) {
    try {
      milp::write_duals(duals_path, model, result);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "milp: %s\n", e.what());
      return 1;
    }
  }
  if (!sol_path.empty()) {
    try {
      milp::write_solution(sol_path, model, result);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "milp: %s\n", e.what());
      return 1;
    }
  }
  if (!profile_path.empty()) {
    try {
      milp::write_profile(profile_path, model, params, result, POPSO_VERSION);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "milp: %s\n", e.what());
      return 1;
    }
  }
  return 0;
}
