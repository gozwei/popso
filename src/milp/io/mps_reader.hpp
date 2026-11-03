#pragma once
// MPS reader (free and fixed format) and free-format MPS writer.
//
// Format handling: every line is split on whitespace, which covers free MPS
// and fixed MPS whose names contain no spaces (the common case). Optional set
// names in RHS/RANGES/BOUNDS are detected from the field count, so fixed files
// with blank set-name fields parse correctly. Limitation: fixed-format names
// containing spaces are not supported. Gzipped files are not supported (no
// zlib); decompress first.
//
// Semantics (common conventions, as in CPLEX / HiGHS / SCIP):
//  - The first N row is the objective (or the one named by OBJNAME); other N
//    rows are dropped together with their coefficients.
//  - OBJSENSE MAX/MAXIMIZE/MIN/MINIMIZE, on the same line or the next line.
//  - RHS on the objective row sets obj_offset = -value.
//  - RANGES R: G row [rhs, rhs+|R|], L row [rhs-|R|, rhs], E row [rhs, rhs+R]
//    if R > 0 and [rhs+R, rhs] if R < 0. Ranges on N rows are ignored.
//  - Columns default to [0, +inf). Integer columns from MARKER INTORG/INTEND
//    blocks that have no entry in BOUNDS default to [0, 1] (the convention of
//    HiGHS, SCIP and MIPLIB); any BOUNDS entry switches to the [0, +inf) default.
//  - BOUNDS: UP, LO, FX, FR, MI (lb = -inf), PL (ub = +inf), BV (binary
//    [0, 1]), LI/UI (integer + bound), SC (semi-continuous, semi-integer on
//    an integer column; the value, if given, is the upper bound), SI
//    (semi-integer). A semi column is x = 0 or lb <= x <= ub.
//  - SOS: set headers " S1 SOS <name> <priority>" (CPLEX) or " S1 <name>"
//    (SCIP), then member lines "<column> <weight>" or "<column>:<weight>".
//  - INDICATORS: " IF <row> <column> <value>"; the row becomes an indicator
//    constraint (enforced when the column equals the value) instead of an
//    ordinary row. A non-integer indicator column is made binary (warning).
//    UP with a negative value on a column whose lower bound is 0 and was not
//    set explicitly sets lb = -inf (with a warning).
//  - |value| >= 1e30 in RHS/RANGES/BOUNDS is treated as infinite.
//  - A column appearing in BOUNDS but not in COLUMNS is an error.
//  - Only the first RHS/RANGES/BOUNDS set is used; entries for other set
//    names are ignored with a warning.
//  - Duplicate (row, column) entries are summed.
//
// Errors are thrown as MpsError with "<source>:<line>: <message>". Warnings
// go to `warnings` if given, otherwise to stderr (capped).

#include <stdexcept>
#include <string>
#include <vector>

#include "milp/core/model.hpp"

namespace milp {

struct MpsError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

Model read_mps(const std::string& path, std::vector<std::string>* warnings = nullptr);
Model read_mps_string(const std::string& text, std::vector<std::string>* warnings = nullptr);

// Writes free MPS with full round-trip precision. Names that are empty or
// contain whitespace are replaced (C<j> / R<i>, or whitespace -> '_').
// Free rows (both bounds infinite) are written as extra N rows, which a
// reader drops; they carry no constraint anyway. Semi-continuous columns get
// SC bounds, SOS sets an SOS section, and indicator constraints are written
// as rows (after the ordinary ones) listed in an INDICATORS section.
void write_mps(const Model& model, const std::string& path);
std::string write_mps_string(const Model& model);

}  // namespace milp
