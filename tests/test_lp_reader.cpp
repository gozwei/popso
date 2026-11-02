// Tests for the CPLEX LP format reader.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "milp/io/lp_reader.hpp"
#include "milp/io/mps_reader.hpp"
#include "test_framework.hpp"

using namespace milp;

namespace {

Index col(const Model& m, const std::string& name) {
  for (Index j = 0; j < m.num_cols(); ++j)
    if (m.col_names[j] == name) return j;
  return -1;
}

Index row(const Model& m, const std::string& name) {
  for (Index i = 0; i < m.num_rows(); ++i)
    if (m.row_names[i] == name) return i;
  return -1;
}

double coef(const Model& m, const std::string& r, const std::string& c) {
  const Index i = row(m, r), j = col(m, c);
  if (i < 0 || j < 0) return std::nan("");
  for (Index p = m.A.begin(j); p < m.A.end(j); ++p)
    if (m.A.index[p] == i) return m.A.value[p];
  return 0.0;
}

// Expects read_lp_string(text) to fail with a message containing `needle`.
bool error_contains(const std::string& text, const std::string& needle) {
  try {
    read_lp_string(text);
  } catch (const LpError& e) {
    const std::string msg = e.what();
    if (msg.find(needle) != std::string::npos) return true;
    std::fprintf(stderr, "  error message was: %s\n", msg.c_str());
    return false;
  }
  std::fprintf(stderr, "  no error for: %s\n", text.c_str());
  return false;
}

bool is_binary(const Model& m, Index j) {
  return m.col_type[j] == VarType::Binary ||
         (m.col_type[j] == VarType::Integer && m.col_lb[j] == 0.0 && m.col_ub[j] == 1.0);
}

// Exact comparison (same order of rows and columns, bit-identical numbers).
// Integer [0, 1] and Binary columns count as equal. Returns "" if equal.
std::string model_diff(const Model& a, const Model& b, bool compare_name) {
  auto num = [](double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
  };
  if (compare_name && a.name != b.name) return "name '" + a.name + "' vs '" + b.name + "'";
  if (a.sense != b.sense) return "objective sense";
  if (a.obj_offset != b.obj_offset) return "objective offset " + num(a.obj_offset) + " vs " + num(b.obj_offset);
  if (a.num_cols() != b.num_cols()) return "column count";
  if (a.num_rows() != b.num_rows()) return "row count";
  for (Index j = 0; j < a.num_cols(); ++j) {
    const std::string c = "column " + a.col_names[j] + ": ";
    if (a.col_names[j] != b.col_names[j]) return c + "name vs " + b.col_names[j];
    if (a.obj[j] != b.obj[j]) return c + "objective " + num(a.obj[j]) + " vs " + num(b.obj[j]);
    if (a.col_lb[j] != b.col_lb[j]) return c + "lower bound " + num(a.col_lb[j]) + " vs " + num(b.col_lb[j]);
    if (a.col_ub[j] != b.col_ub[j]) return c + "upper bound " + num(a.col_ub[j]) + " vs " + num(b.col_ub[j]);
    if (a.col_type[j] != b.col_type[j] && !(is_binary(a, j) && is_binary(b, j))) return c + "type";
    if (a.A.end(j) - a.A.begin(j) != b.A.end(j) - b.A.begin(j)) return c + "nonzero count";
    // Entries sorted by row index in both (from_triplets sorts).
    for (Index p = a.A.begin(j), q = b.A.begin(j); p < a.A.end(j); ++p, ++q) {
      if (a.A.index[p] != b.A.index[q]) return c + "row pattern";
      if (a.A.value[p] != b.A.value[q]) return c + "coefficient " + num(a.A.value[p]) + " vs " + num(b.A.value[q]);
    }
  }
  for (Index i = 0; i < a.num_rows(); ++i) {
    const std::string r = "row " + a.row_names[i] + ": ";
    if (a.row_names[i] != b.row_names[i]) return r + "name vs " + b.row_names[i];
    if (a.row_lb[i] != b.row_lb[i]) return r + "lower " + num(a.row_lb[i]) + " vs " + num(b.row_lb[i]);
    if (a.row_ub[i] != b.row_ub[i]) return r + "upper " + num(a.row_ub[i]) + " vs " + num(b.row_ub[i]);
  }
  return "";
}

const char* kExample =
    "\\Problem name: example\n"
    "\\ A comment line\n"
    "Maximize\n"
    " obj: x1 + 2 x2 + 3 x3 + x4 + 1.5\n"
    "Subject To\n"
    " c1: - x1 + x2 + x3 + 10 x4 <= 20\n"
    " c2: x1 - 3 x2 + x3 <= 30\n"
    " c3: x2 - 3.5 x4 = 0\n"
    "Bounds\n"
    " 0 <= x1 <= 40\n"
    " 2 <= x4 <= 3\n"
    "General\n"
    " x4\n"
    "End\n";

}  // namespace

TEST(lp_reader_example) {
  std::vector<std::string> warnings;
  const Model m = read_lp_string(kExample, &warnings);
  CHECK(warnings.empty());
  CHECK_EQ(m.name, std::string("example"));
  CHECK(m.sense == ObjSense::Maximize);
  CHECK_EQ(m.obj_offset, 1.5);
  REQUIRE(m.num_cols() == 4);
  REQUIRE(m.num_rows() == 3);
  CHECK_EQ(m.col_names[0], std::string("x1"));
  CHECK_EQ(m.col_names[3], std::string("x4"));
  CHECK_EQ(m.obj[2], 3.0);
  CHECK_EQ(m.row_names[1], std::string("c2"));
  CHECK_EQ(m.row_lb[0], -kInf);
  CHECK_EQ(m.row_ub[0], 20.0);
  CHECK_EQ(m.row_lb[2], 0.0);
  CHECK_EQ(m.row_ub[2], 0.0);
  CHECK_EQ(coef(m, "c1", "x1"), -1.0);
  CHECK_EQ(coef(m, "c1", "x4"), 10.0);
  CHECK_EQ(coef(m, "c2", "x2"), -3.0);
  CHECK_EQ(coef(m, "c3", "x4"), -3.5);
  CHECK_EQ(coef(m, "c3", "x1"), 0.0);
  CHECK_EQ(m.col_ub[0], 40.0);
  CHECK_EQ(m.col_lb[3], 2.0);
  CHECK_EQ(m.col_ub[3], 3.0);
  CHECK(m.col_type[3] == VarType::Integer);
  CHECK(m.col_type[0] == VarType::Continuous);
  CHECK_EQ(m.col_ub[1], kInf);
  CHECK_EQ(m.num_nonzeros(), 9);
}

TEST(lp_reader_keyword_variants) {
  // Case-insensitive keywords, all spellings of sections and operators.
  const char* variants[][5] = {
      {"MINIMIZE", "SUBJECT TO", "BOUNDS", "GENERALS", "BINARIES"},
      {"min", "st", "bound", "gen", "bin"},
      {"Minimise", "s.t.", "Bounds", "Integers", "Binary"},
      {"minimum", "such that", "bounds", "general", "binary"},
      {"Min", "S.T.", "BOUNDS", "integer", "BIN"},
  };
  for (const auto& v : variants) {
    const std::string text = std::string(v[0]) + "\n x + y + z\n" + v[1] + "\n c1: x + y >= 1\n c2: x =< 3\n c3: y => 1\n" +
                              " c4: z < 4\n c5: z > -1\n c6: x == 2\n" + v[2] + "\n x <= 5\n" + v[3] + "\n y\n" + v[4] +
                              "\n z\nEND\n";
    const Model m = read_lp_string(text);
    REQUIRE(m.num_rows() == 6);
    CHECK(m.sense == ObjSense::Minimize);
    CHECK_EQ(m.row_ub[1], 3.0);
    CHECK_EQ(m.row_lb[2], 1.0);
    CHECK_EQ(m.row_ub[3], 4.0);
    CHECK_EQ(m.row_lb[4], -1.0);
    CHECK_EQ(m.row_lb[5], 2.0);
    CHECK_EQ(m.row_ub[5], 2.0);
    CHECK_EQ(m.col_ub[0], 5.0);
    CHECK(m.col_type[1] == VarType::Integer);
    CHECK(m.col_type[2] == VarType::Binary);
    CHECK_EQ(m.col_ub[2], 1.0);
  }
  for (const char* s : {"MAX", "maximize", "Maximise", "maximum"})
    CHECK(read_lp_string(std::string(s) + "\n x\nst\n x <= 1\nend\n").sense == ObjSense::Maximize);
}

TEST(lp_reader_empty_objective_and_sections) {
  Model m = read_lp_string("Minimize\nSubject To\n c: x + y >= 1\nEnd\n");
  CHECK_EQ(m.num_cols(), 2);
  CHECK_EQ(m.obj[0], 0.0);
  m = read_lp_string("minimize\n obj:\nsubject to\n x >= 1\nend");  // no newline at end
  CHECK_EQ(m.num_cols(), 1);
  CHECK_EQ(m.row_names[0], std::string("R1"));
  // Only an objective; no End.
  m = read_lp_string("max obj: 2 x + 3\n");
  CHECK_EQ(m.num_rows(), 0);
  CHECK_EQ(m.obj[0], 2.0);
  CHECK_EQ(m.obj_offset, 3.0);
  // Constraints without an objective section.
  m = read_lp_string("subject to\n x + y <= 1\nend\n");
  CHECK_EQ(m.num_rows(), 1);
  // Empty semi-continuous / SOS sections (HiGHS writes "semi") are fine.
  m = read_lp_string("min\n x\nst\n x >= 1\nsemi\nsos\nsemi-continuous\nend\n");
  CHECK_EQ(m.num_rows(), 1);
  // Anything after End is ignored.
  m = read_lp_string("min\n x\nst\n x >= 1\nend\n garbage [ ^ \n");
  CHECK_EQ(m.num_rows(), 1);
}

TEST(lp_reader_bounds) {
  std::vector<std::string> warnings;
  const Model m = read_lp_string(
      "min\n a + b + c + d + e + f + g + h + i + k\n"
      "st\n a + b + c + d + e + f + g + h + i + k >= -100\n"
      "bounds\n"
      " a free\n"
      " b >= -5\n"
      " -inf <= c <= 7\n"
      " d = 3.5\n"
      " -1e30 <= e <= +1E+30\n"
      " f >= -infinity\n"
      " -3 <= g\n"
      " 10 >= h >= -2\n"
      " i <= -4\n"
      " k <= 1e300\n"
      " new >= 2\n"
      "end\n",
      &warnings);
  CHECK_EQ(warnings.size(), 1u);  // negative upper bound on i
  REQUIRE(m.num_cols() == 11);
  CHECK_EQ(m.col_lb[col(m, "a")], -kInf);
  CHECK_EQ(m.col_ub[col(m, "a")], kInf);
  CHECK_EQ(m.col_lb[col(m, "b")], -5.0);
  CHECK_EQ(m.col_ub[col(m, "b")], kInf);
  CHECK_EQ(m.col_lb[col(m, "c")], -kInf);
  CHECK_EQ(m.col_ub[col(m, "c")], 7.0);
  CHECK_EQ(m.col_lb[col(m, "d")], 3.5);
  CHECK_EQ(m.col_ub[col(m, "d")], 3.5);
  CHECK_EQ(m.col_lb[col(m, "e")], -kInf);
  CHECK_EQ(m.col_ub[col(m, "e")], kInf);
  CHECK_EQ(m.col_lb[col(m, "f")], -kInf);
  CHECK_EQ(m.col_lb[col(m, "g")], -3.0);
  CHECK_EQ(m.col_lb[col(m, "h")], -2.0);
  CHECK_EQ(m.col_ub[col(m, "h")], 10.0);
  // Negative upper bound keeps the default lower bound 0 (HiGHS / SCIP).
  CHECK_EQ(m.col_lb[col(m, "i")], 0.0);
  CHECK_EQ(m.col_ub[col(m, "i")], -4.0);
  CHECK_EQ(m.col_ub[col(m, "k")], kInf);
  // A variable that appears only in Bounds is created there.
  REQUIRE(col(m, "new") == 10);
  CHECK_EQ(m.col_lb[10], 2.0);
  CHECK_EQ(m.obj[10], 0.0);

  warnings.clear();
  read_lp_string("min\n x\nst\n x >= -10\nbounds\n x <= -1\nend\n", &warnings);
  CHECK_EQ(warnings.size(), 1u);
  warnings.clear();
  read_lp_string("min\n x\nst\n x >= -10\nbounds\n x <= -1\n x >= -2\nend\n", &warnings);
  CHECK(warnings.empty());
}

TEST(lp_reader_integer_sections) {
  const Model m = read_lp_string(
      "max\n x + y + z + w\nst\n x + y + z + w <= 10\n"
      "bounds\n -2 <= z <= 5\n w <= 0.5\n"
      "generals\n x\n"
      "binaries\n y z w\n u\n"
      "generals\n v\n"
      "end\n");
  REQUIRE(m.num_cols() == 6);
  CHECK(m.col_type[col(m, "x")] == VarType::Integer);
  CHECK_EQ(m.col_lb[col(m, "x")], 0.0);
  CHECK_EQ(m.col_ub[col(m, "x")], kInf);  // no binary default as in MPS
  CHECK(m.col_type[col(m, "y")] == VarType::Binary);
  CHECK_EQ(m.col_ub[col(m, "y")], 1.0);
  // Binary bounds are intersected with [0, 1].
  CHECK_EQ(m.col_lb[col(m, "z")], 0.0);
  CHECK_EQ(m.col_ub[col(m, "z")], 1.0);
  CHECK_EQ(m.col_ub[col(m, "w")], 0.5);
  // Variables appearing only in integer sections.
  CHECK(m.col_type[col(m, "u")] == VarType::Binary);
  CHECK(m.col_type[col(m, "v")] == VarType::Integer);
  CHECK_EQ(col(m, "u"), 4);
  CHECK_EQ(col(m, "v"), 5);
}

TEST(lp_reader_ranged_and_constant_terms) {
  const Model m = read_lp_string(
      "min\n x + y\n"
      "st\n"
      " r1: -3 <= x + y <= 4\n"
      " r2: 10 >= x - y >= -2.5\n"
      " r3: -inf <= x <= 8\n"
      " r4: 2 <= 3 x + y\n"
      " r5: 5 >= y\n"
      " r6: x + 2 + y - 1 >= 4\n"
      " r7: 1 <= x + 1 <= 3\n"
      "end\n");
  REQUIRE(m.num_rows() == 7);
  CHECK_EQ(m.row_lb[0], -3.0);
  CHECK_EQ(m.row_ub[0], 4.0);
  CHECK_EQ(m.row_lb[1], -2.5);
  CHECK_EQ(m.row_ub[1], 10.0);
  CHECK_EQ(m.row_lb[2], -kInf);
  CHECK_EQ(m.row_ub[2], 8.0);
  CHECK_EQ(m.row_lb[3], 2.0);
  CHECK_EQ(m.row_ub[3], kInf);
  CHECK_EQ(coef(m, "r4", "x"), 3.0);
  CHECK_EQ(m.row_lb[4], -kInf);
  CHECK_EQ(m.row_ub[4], 5.0);
  CHECK_EQ(m.row_lb[5], 3.0);
  CHECK_EQ(m.row_lb[6], 0.0);
  CHECK_EQ(m.row_ub[6], 2.0);
}

TEST(lp_reader_terms_and_numbers) {
  const Model m = read_lp_string(
      "minimize\n obj: 3x + 2e1y - -z + .5 w + 1.5E-1 v - 2 x + e1 + 0 q\n"
      "subject to\n"
      " c1: x + x + 2 y - y - 0.5 w - 0.5 w + 3 z - 3 z >= 1e-3\n"
      " c2: + - x <= 5\n"
      "end\n");
  REQUIRE(m.num_cols() == 7);
  CHECK_EQ(m.obj[col(m, "x")], 1.0);   // 3x - 2 x
  CHECK_EQ(m.obj[col(m, "y")], 20.0);  // 2e1y
  CHECK_EQ(m.obj[col(m, "z")], 1.0);   // - -z
  CHECK_EQ(m.obj[col(m, "w")], 0.5);
  CHECK_EQ(m.obj[col(m, "v")], 0.15);
  CHECK_EQ(m.obj[col(m, "e1")], 1.0);  // a name, not a number
  CHECK_EQ(col(m, "q"), 6);            // zero coefficient still declares the column
  // Duplicates summed; entries that cancel are dropped.
  CHECK_EQ(coef(m, "c1", "x"), 2.0);
  CHECK_EQ(coef(m, "c1", "y"), 1.0);
  CHECK_EQ(coef(m, "c1", "w"), -1.0);
  CHECK_EQ(m.A.end(col(m, "z")) - m.A.begin(col(m, "z")), 0);
  CHECK_EQ(m.row_lb[0], 1e-3);
  CHECK_EQ(coef(m, "c2", "x"), -1.0);
}

TEST(lp_reader_names_and_whitespace) {
  const std::string text =
      "\\* block style comment *\\\n"
      "  MAXIMIZE   \\ trailing comment\r\n"
      "\tprofit:\t x_1.a + y#2 + z{3}(a,b) + ~w! + \"q\" \r\n"
      "   + 'p'\n"
      "\n\n"
      "SUBJECT    TO\r\n"
      "c_1: x_1.a + y#2\n"
      "   + z{3}(a,b)\n"
      "<=\n"
      "   10 c_2:~w!>=1 c_3 : \"q\" + 'p' = 2\n"
      " x_1.a<=4\n"
      " -y#2>=-5\n"
      "Bounds\n"
      " x_1.a<=3 y#2>=-1\n"
      "End\n";
  const Model m = read_lp_string(text);
  REQUIRE(m.num_cols() == 6);
  REQUIRE(m.num_rows() == 5);
  CHECK(m.sense == ObjSense::Maximize);
  CHECK_EQ(m.col_names[2], std::string("z{3}(a,b)"));
  CHECK_EQ(m.row_names[0], std::string("c_1"));
  CHECK_EQ(m.row_ub[0], 10.0);
  CHECK_EQ(m.row_names[1], std::string("c_2"));
  CHECK_EQ(m.row_lb[1], 1.0);
  CHECK_EQ(m.row_names[2], std::string("c_3"));
  CHECK_EQ(m.row_lb[2], 2.0);
  CHECK_EQ(m.row_names[3], std::string("R4"));
  CHECK_EQ(m.row_names[4], std::string("R5"));
  CHECK_EQ(coef(m, "R5", "y#2"), -1.0);
  CHECK_EQ(m.col_ub[col(m, "x_1.a")], 3.0);
  CHECK_EQ(m.col_lb[col(m, "y#2")], -1.0);
  CHECK_EQ(m.num_nonzeros(), 8);
}

TEST(lp_reader_default_row_names_unique) {
  const Model m = read_lp_string("min\n x\nst\n x >= 1\n R1: x <= 5\n R2_1: x <= 6\n x <= 7\nend\n");
  REQUIRE(m.num_rows() == 4);
  CHECK_EQ(m.row_names[0], std::string("R1_1"));
  CHECK_EQ(m.row_names[1], std::string("R1"));
  CHECK_EQ(m.row_names[3], std::string("R4"));
}

TEST(lp_reader_keywords_only_at_line_start) {
  // "free", "end", "bin" and friends are names unless they start a line.
  const Model m = read_lp_string("min\n x + bin + end\nst\n c: x + bin + end >= 1\nend\n");
  CHECK_EQ(m.num_cols(), 3);
  CHECK_EQ(m.num_rows(), 1);
  // A keyword followed by ':' is a label.
  const Model m2 = read_lp_string("min\n x\nst\n bounds: x >= 1\nend\n");
  CHECK_EQ(m2.row_names[0], std::string("bounds"));
}

TEST(lp_reader_errors) {
  CHECK(error_contains("min\n x + [ x ^ 2 ] / 2\nst\n x >= 1\nend\n", "<string>:2: quadratic terms"));
  CHECK(error_contains("min\n x\nst\n c: x ^ 2 <= 1\nend\n", "quadratic terms"));
  CHECK(error_contains("min\n x\nst\n x[1] >= 1\nend\n", "brackets are not allowed"));
  CHECK(error_contains("min\n x\nst\n c: b = 1 -> x >= 1\nend\n", "indicator constraints"));
  CHECK(error_contains("min\n x\nst\n x >= 1\nsemi-continuous\n x\nend\n", "<string>:6: semi-continuous"));
  CHECK(error_contains("min\n x\nst\n x >= 1\nsos\n s1: S1:: x:1 y:2\nend\n", "SOS constraints are not supported yet"));
  CHECK(error_contains("min\n x\nst\n x >= 1\nlazy constraints\n x <= 2\nend\n", "lazy constraints are not supported yet"));
  CHECK(error_contains("min\n x\nst\n x >= 1\nuser cuts\n x <= 2\nend\n", "user cuts are not supported yet"));
  CHECK(error_contains("min\n x\nst\n x >= 1\ngeneral constraints\nend\n", "general constraints are not supported"));
  CHECK(error_contains("minimize multi-objectives\n obj1: x\nst\n x >= 1\nend\n", "multiple objectives"));
  CHECK(error_contains("min\n x\nst\n x >= 1\nmax\n y\nend\n", "multiple objective sections"));
  CHECK(error_contains("min\n x\nst\n c1: x + y\n c2: x >= 1\nend\n", "<string>:5: expected '<=', '>=' or '=' in constraint 'c1'"));
  CHECK(error_contains("min\n x\nst\n c1: x + y >= z\nend\n", "<string>:4: expected a number for the right-hand side"));
  CHECK(error_contains("min\n x\nst\n c1: x + y >= 1 <= 2\nend\n", "ranged constraints are written"));
  CHECK(error_contains("min\n x\nst\n c1: 1 <= x + y >= 2\nend\n", "two '<=' or two '>='"));
  CHECK(error_contains("min\n x\nst\n c1: x >= 1\n c1: x <= 2\nend\n", "<string>:5: duplicate constraint name 'c1'"));
  CHECK(error_contains("min\n x y\nst\n x >= 1\nend\n", "<string>:2: expected '+' or '-' before 'y'"));
  CHECK(error_contains("min\n x\nst\n c: x + 3 * y >= 1\nend\n", "unexpected character '*'"));
  CHECK(error_contains("min\n x\nst\n x >= 1\nbounds\n x <= y\nend\n", "<string>:6: expected a number for the bound"));
  CHECK(error_contains("min\n x\nst\n x >= 1\nbounds\n x >= inf\nend\n", "lower bound +inf"));
  CHECK(error_contains("min\n x\nst\n x >= 1\nbounds\n 2 x <= 5\nend\n", "expected '<=', '>=' or '=' in bound"));
  CHECK(error_contains("min\n x\nst\n x >= 1\nbounds\n x\nend\n", "expected '<=', '>=', '=' or 'free'"));
  CHECK(error_contains("min\n x\nst\n x >= 1\ngeneral\n 3\nend\n", "expected a variable name in the General section"));
  CHECK(error_contains("min\n x\nst\n x = inf\nend\n", "infinite right-hand side"));
  CHECK(error_contains("min\n inf x\nst\n x >= 1\nend\n", "infinite coefficient"));
  CHECK(error_contains("min\n x\n\n + y <= 1\nend\n", "<string>:4: unexpected '<=' in objective (missing 'Subject To'?)"));
  CHECK(error_contains("min\n x\n\n x <= 1\nend\n", "<string>:4: expected '+' or '-' before 'x'"));
  CHECK(error_contains("x + y\nst\n x >= 1\nend\n", "<string>:1: expected 'Minimize', 'Maximize' or 'Subject To'"));
  CHECK(error_contains("max: 3x + 2y;\nc1: x + y <= 4;\n", "lp_solve"));
  CHECK(error_contains("", "no objective or constraint section"));
  CHECK(error_contains("\\ only a comment\n", "no objective or constraint section"));
  CHECK_THROWS(read_lp_string("min\n 1e999 x\nend\n"), LpError);  // out of range (or infinite)
  CHECK(error_contains("st\n x <= 1\n st\n x >= 0\nmin\n x\nend\n", "objective section must come first"));
  CHECK_THROWS(read_lp("/nonexistent/file.lp"), LpError);
}

TEST(lp_reader_highs_style_file) {
  // As written by HiGHS: '+1 x' style terms, objective constant, split ranged
  // rows, "bin"/"gen"/empty "semi" sections.
  const char* text =
      "\\ File written by HiGHS .lp file handler\n"
      "max\n"
      " obj: +1 x -2 y +3.5 w  +7\n"
      "st\n"
      " c1: +1 x +3 y <= +4\n"
      " c2: +2 x +5 w = +1\n"
      " c3lo: +4 z +6 b >= -2\n"
      " c3up: +4 z +6 b <= +5\n"
      "bounds\n"
      " y free\n"
      " -3 <= z <= 7\n"
      " w = 2\n"
      " b <= 1\n"
      "bin\n"
      " b\n"
      "gen\n"
      " y\n"
      " z\n"
      "semi\n"
      "end\n";
  const Model m = read_lp_string(text);
  REQUIRE(m.num_cols() == 5);
  REQUIRE(m.num_rows() == 4);
  CHECK_EQ(m.obj_offset, 7.0);
  CHECK(m.sense == ObjSense::Maximize);
  CHECK_EQ(m.col_names[2], std::string("w"));  // order of appearance
  CHECK_EQ(m.col_names[3], std::string("z"));
  CHECK(m.col_type[col(m, "b")] == VarType::Binary);
  CHECK(m.col_type[col(m, "y")] == VarType::Integer);
  CHECK_EQ(m.col_lb[col(m, "y")], -kInf);
  CHECK_EQ(m.col_lb[col(m, "z")], -3.0);
  CHECK_EQ(m.col_lb[col(m, "w")], 2.0);
}

TEST(lp_reader_scip_and_cplex_style_files) {
  // As written by SCIP: statistics comments, "Obj:", "Subject to", fixed
  // variables as "3 <= v <= 3", integer names on one line.
  const Model m = read_lp_string(
      "\\ SCIP STATISTICS\n"
      "\\   Problem name     : mip_mixed_bounds\n"
      "Minimize\n Obj: -1 y +1 x +2 w +100\n"
      "Subject to\n c1: +1 x +1 y >= -3.5\n c2: +1 x -1 y +1 w >= +0.5\n c4: +1 x +1 v <= +6\n"
      "Bounds\n -5 <= y <= 5\n x free\n 3 <= v <= 3\n"
      "Generals\n y x v\nEnd\n");
  REQUIRE(m.num_cols() == 4);
  CHECK_EQ(m.col_names[3], std::string("v"));
  CHECK_EQ(m.obj_offset, 100.0);
  CHECK_EQ(m.col_lb[3], 3.0);
  CHECK_EQ(m.col_ub[3], 3.0);
  CHECK(m.col_type[1] == VarType::Integer);
  CHECK(m.col_type[2] == VarType::Continuous);
  // As written by CPLEX: encoding comment before the problem name.
  const Model c = read_lp_string(
      "\\ENCODING=ISO-8859-1\n\\Problem name: diet\n\nMinimize\n obj: 2 x1 + 3 x2\nSubject To\n"
      " c1: x1 + x2 >= 1\nBounds\n x2 <= 4\nEnd\n");
  CHECK_EQ(c.name, std::string("diet"));
}

TEST(lp_reader_matches_mps) {
  // The same model in both formats reads to the same Model.
  const char* mps =
      "NAME          both\n"
      "OBJSENSE MAX\n"
      "ROWS\n N  obj\n L  c1\n G  c2\n E  c3\n G  c4\n"
      "COLUMNS\n"
      "    x  obj  1  c1  1\n    x  c2  1\n"
      "    M  'MARKER'  'INTORG'\n"
      "    y  obj  2  c1  1\n    y  c3  -1\n"
      "    M  'MARKER'  'INTEND'\n"
      "    z  obj  -1  c2  1\n    z  c3  1\n    z  c4  2.5\n"
      "RHS\n    RHS  c1  4  c2  1\n    RHS  c3  7  c4 -1\n    RHS  obj  -3\n"
      "RANGES\n    RNG  c4  6\n"
      "BOUNDS\n UP BND x 4\n LO BND y -1\n UP BND y 1\n FR BND z\n"
      "ENDATA\n";
  const char* lp =
      "\\Problem name: both\n"
      "Maximize\n obj: x + 2 y - z + 3\n"
      "Subject To\n c1: x + y <= 4\n c2: x + z >= 1\n c3: - y + z = 7\n c4: -1 <= 2.5 z <= 5\n"
      "Bounds\n x <= 4\n -1 <= y <= 1\n z free\n"
      "Generals\n y\nEnd\n";
  const std::string diff = model_diff(read_mps_string(mps), read_lp_string(lp), true);
  CHECK_EQ(diff, std::string(""));
  if (!diff.empty()) std::fprintf(stderr, "  %s\n", diff.c_str());
}

// Cross-check hook: if POPSO_LP_CROSSCHECK_DIR is set, every <stem>.lp in it
// must read to exactly the same model as benchmarks/instances/<stem>.mps.
// benchmarks/tools/lp_crosscheck.py writes the LP files and runs this test.
TEST(lp_reader_crosscheck_dir) {
  const char* dir = std::getenv("POPSO_LP_CROSSCHECK_DIR");
  if (!dir || !*dir) return;
  const std::string mps_dir = std::string(MILP_SOURCE_DIR) + "/benchmarks/instances";
  int compared = 0;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path().extension() != ".lp") continue;
    const std::string stem = entry.path().stem().string();
    const std::string mps_file = mps_dir + "/" + stem + ".mps";
    if (!std::filesystem::exists(mps_file)) {
      CHECK(!"LP file without matching MPS instance");
      std::fprintf(stderr, "  no %s\n", mps_file.c_str());
      continue;
    }
    std::vector<std::string> w1, w2;
    const Model a = read_mps(mps_file, &w1);
    const Model b = read_lp(entry.path().string(), &w2);
    const std::string diff = model_diff(a, b, true);
    if (!CHECK(diff.empty())) std::fprintf(stderr, "  %s: %s\n", stem.c_str(), diff.c_str());
    ++compared;
  }
  std::printf("lp_reader_crosscheck_dir: compared %d LP files with their MPS originals\n", compared);
  CHECK(compared > 0);
}
