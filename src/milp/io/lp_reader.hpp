#pragma once
// Reader for the CPLEX LP file format, as written by CPLEX, HiGHS, SCIP and
// most modelling tools.
//
// Supported:
//  - Objective section: Minimize/Minimise/Minimum/Min or Maximize/Maximise/
//    Maximum/Max, an optional "name:" label, linear terms and a constant
//    term (-> obj_offset). The section may be empty.
//  - Constraints: Subject To / Such That / st / s.t. Each constraint is
//    "[name:] expr op rhs" with op one of <=, =<, <, >=, =>, >, =, == and a
//    constant right-hand side. Constants in expr are moved to the right-hand
//    side. Ranged rows are written "[name:] lo <= expr <= hi" (or with >=).
//    Unnamed rows get the names R1, R2, ... (made unique).
//  - Bounds: "x <= u", "x >= l", "l <= x", "l <= x <= u", "x = v", "x free";
//    inf/infinity (any case, optional sign) and |value| >= 1e30 are
//    infinite. A negative upper bound does not change the default lower
//    bound 0 (as in HiGHS and SCIP; a warning is issued).
//  - General/Generals/Gen/Integer/Integers: integer variables (default bounds
//    [0, +inf)); Binary/Binaries/Bin: binary variables (bounds intersected
//    with [0, 1]).
//  - Semi-Continuous/Semis/Semi: semi-continuous variables (x = 0 or
//    lb <= x <= ub, bounds from the Bounds section); semi-integer if also
//    General or Binary.
//  - SOS (or SOS1/SOS2): entries "[name:] S1:: x1:w1 x2:w2 ..." (S2:: for
//    SOS2; in a SOS1/SOS2 section the type may be omitted).
//  - Indicator constraints in the constraints section:
//    "[name:] b = 1 -> expr op rhs" (or "b = 0 -> ..."). An indicator
//    variable not declared integer is made binary.
//  - End. Anything after End is ignored.
//  - Comments start with a backslash and run to the end of the line. A first
//    comment "\Problem name: <name>" sets the model name.
//  - Keywords are case-insensitive and recognised only at the start of a
//    line (and not when followed by ':'); names are case-sensitive.
//    Expressions may span lines.
//  - Names: letters, digits and !"#$%&()/,.;?@_`'{}|~ (and non-ASCII
//    bytes); a name must not start with a digit or a period.
//  - Numbers: decimal with optional exponent. A coefficient may be written
//    directly before its variable ("3x" = "3 x"; "2e1x" = 20 x).
//  - Duplicate variables in one row or in the objective are summed.
//  - Variables are numbered in order of first appearance (objective,
//    constraints, bounds, then integer sections); variables that appear only
//    in Bounds or General/Binary sections are created there.
//
// Not supported (rejected with a "not supported" error, never silently
// misparsed): quadratic terms ('[' ... ']', '^'), equivalence constraints
// ('<->'), lazy constraints / user cuts, general constraints and multiple
// objectives.
//
// Errors are thrown as LpError with "<source>:<line>: <message>". Warnings
// go to `warnings` if given, otherwise to stderr (capped).

#include <stdexcept>
#include <string>
#include <vector>

#include "milp/core/model.hpp"

namespace milp {

struct LpError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

Model read_lp(const std::string& path, std::vector<std::string>* warnings = nullptr);
Model read_lp_string(const std::string& text, std::vector<std::string>* warnings = nullptr);

}  // namespace milp
