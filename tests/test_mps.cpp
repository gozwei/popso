// Tests for the MPS reader and writer.

#include <cstdio>
#include <filesystem>
#include <string>

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

bool error_contains(const std::string& text, const std::string& needle) {
  try {
    read_mps_string(text);
  } catch (const MpsError& e) {
    const std::string msg = e.what();
    if (msg.find(needle) != std::string::npos) return true;
    std::fprintf(stderr, "  error message was: %s\n", msg.c_str());
    return false;
  }
  return false;
}

// Classic example (fixed format, from the lp_solve / IBM documentation).
const char* kTestProb =
    "NAME          TESTPROB\n"
    "ROWS\n"
    " N  COST\n"
    " L  LIM1\n"
    " G  LIM2\n"
    " E  MYEQN\n"
    "COLUMNS\n"
    "    XONE      COST                 1   LIM1                 1\n"
    "    XONE      LIM2                 1\n"
    "    YTWO      COST                 2   LIM1                 1\n"
    "    YTWO      MYEQN               -1\n"
    "    ZTHREE    COST                -1   LIM2                 1\n"
    "    ZTHREE    MYEQN                1\n"
    "RHS\n"
    "    RHS       LIM1                 4   LIM2                 1\n"
    "    RHS       MYEQN                7\n"
    "BOUNDS\n"
    " UP BND       XONE                 4\n"
    " LO BND       YTWO                -1\n"
    " UP BND       YTWO                 1\n"
    "ENDATA\n";

}  // namespace

TEST(mps_fixed_testprob) {
  std::vector<std::string> warnings;
  Model m = read_mps_string(kTestProb, &warnings);
  CHECK(warnings.empty());
  CHECK_EQ(m.name, "TESTPROB");
  CHECK(m.sense == ObjSense::Minimize);
  REQUIRE(m.num_rows() == 3);
  REQUIRE(m.num_cols() == 3);
  CHECK_EQ(m.num_nonzeros(), 6);
  CHECK_EQ(m.num_integers(), 0);
  CHECK((m.obj == std::vector<double>{1, 2, -1}));
  CHECK_EQ(m.row_lb[row(m, "LIM1")], -kInf);
  CHECK_EQ(m.row_ub[row(m, "LIM1")], 4.0);
  CHECK_EQ(m.row_lb[row(m, "LIM2")], 1.0);
  CHECK_EQ(m.row_ub[row(m, "LIM2")], kInf);
  CHECK_EQ(m.row_lb[row(m, "MYEQN")], 7.0);
  CHECK_EQ(m.row_ub[row(m, "MYEQN")], 7.0);
  CHECK_EQ(coef(m, "MYEQN", "YTWO"), -1.0);
  CHECK_EQ(coef(m, "LIM2", "YTWO"), 0.0);
  CHECK_EQ(m.col_lb[col(m, "XONE")], 0.0);
  CHECK_EQ(m.col_ub[col(m, "XONE")], 4.0);
  CHECK_EQ(m.col_lb[col(m, "YTWO")], -1.0);
  CHECK_EQ(m.col_ub[col(m, "YTWO")], 1.0);
  CHECK_EQ(m.col_lb[col(m, "ZTHREE")], 0.0);
  CHECK_EQ(m.col_ub[col(m, "ZTHREE")], kInf);
  CHECK_EQ(m.obj_offset, 0.0);
}

TEST(mps_free_format_equivalent) {
  const char* text =
      "NAME TESTPROB\n"
      "ROWS\n N COST\n L LIM1\n G LIM2\n E MYEQN\n"
      "COLUMNS\n"
      " XONE COST 1 LIM1 1\n XONE LIM2 1\n"
      "\tYTWO COST 2 LIM1 1\n YTWO MYEQN -1\n"
      " ZTHREE COST -1 LIM2 1\n ZTHREE MYEQN 1\n"
      "RHS\n RHS LIM1 4 LIM2 1\n RHS MYEQN 7\n"
      "BOUNDS\n UP BND XONE 4\n LO BND YTWO -1\n UP BND YTWO 1\n"
      "ENDATA\n";
  Model a = read_mps_string(kTestProb);
  Model b = read_mps_string(text);
  CHECK(a.obj == b.obj);
  CHECK(a.col_lb == b.col_lb);
  CHECK(a.col_ub == b.col_ub);
  CHECK(a.row_lb == b.row_lb);
  CHECK(a.row_ub == b.row_ub);
  CHECK(a.A.index == b.A.index);
  CHECK(a.A.value == b.A.value);
  CHECK(a.col_names == b.col_names);
  CHECK(a.row_names == b.row_names);
}

TEST(mps_fixed_blank_set_names) {
  // Fixed-format files may leave the RHS/RANGES/BOUNDS set-name field blank.
  const char* text =
      "NAME\n"
      "ROWS\n N  obj\n L  c1\n E  c2\n"
      "COLUMNS\n"
      "    x         obj             1.0   c1              1.0\n"
      "    x         c2              1.0\n"
      "    y         c1              1.0\n"
      "RHS\n"
      "              c1              5.0   c2              2.0\n"
      "RANGES\n"
      "              c1              3.0\n"
      "BOUNDS\n"
      " UP           x               4.0\n"
      " MI           y\n"
      " BV           y\n"
      "ENDATA\n";
  Model m = read_mps_string(text);
  CHECK(m.name.empty());
  CHECK_EQ(m.row_lb[row(m, "c1")], 2.0);
  CHECK_EQ(m.row_ub[row(m, "c1")], 5.0);
  CHECK_EQ(m.row_lb[row(m, "c2")], 2.0);
  CHECK_EQ(m.col_ub[col(m, "x")], 4.0);
  CHECK(m.col_type[col(m, "y")] == VarType::Binary);
  CHECK_EQ(m.col_lb[col(m, "y")], 0.0);
  CHECK_EQ(m.col_ub[col(m, "y")], 1.0);
}

TEST(mps_comments_blank_lines_crlf) {
  const char* text =
      "* leading comment\r\n"
      "NAME   commented\r\n"
      "\r\n"
      "ROWS\r\n"
      "* comment in rows\r\n"
      " N obj\r\n"
      "   \r\n"
      " G c\r\n"
      "COLUMNS\r\n"
      " x obj 1 c 1\r\n"
      "RHS\r\n"
      " RHS c 2\r\n"
      "ENDATA";  // no trailing newline
  std::vector<std::string> w;
  Model m = read_mps_string(text, &w);
  CHECK(w.empty());
  CHECK_EQ(m.name, "commented");
  CHECK_EQ(m.num_rows(), 1);
  CHECK_EQ(m.row_lb[0], 2.0);
}

TEST(mps_objsense_next_line) {
  Model m = read_mps_string(
      "NAME t\nOBJSENSE\n    MAX\nROWS\n N obj\nCOLUMNS\n x obj 3\nBOUNDS\n UP BND x 1\nENDATA\n");
  CHECK(m.sense == ObjSense::Maximize);
  CHECK_EQ(m.obj[0], 3.0);  // coefficients stay in the original sense
  Model m2 = read_mps_string("NAME t\nOBJSENSE\n  MAXIMIZE\nROWS\n N obj\nCOLUMNS\n x obj 3\nENDATA\n");
  CHECK(m2.sense == ObjSense::Maximize);
  Model m3 = read_mps_string("NAME t\nOBJSENSE\n  MIN\nROWS\n N obj\nCOLUMNS\n x obj 3\nENDATA\n");
  CHECK(m3.sense == ObjSense::Minimize);
}

TEST(mps_objsense_same_line) {
  Model m = read_mps_string("NAME t\nOBJSENSE MAX\nROWS\n N obj\nCOLUMNS\n x obj 3\nENDATA\n");
  CHECK(m.sense == ObjSense::Maximize);
  Model m2 = read_mps_string("NAME t\nOBJSENSE    MAXIMIZE\nROWS\n N obj\nCOLUMNS\n x obj 3\nENDATA\n");
  CHECK(m2.sense == ObjSense::Maximize);
  Model m3 = read_mps_string("NAME t\nOBJSENSE minimize\nROWS\n N obj\nCOLUMNS\n x obj 3\nENDATA\n");
  CHECK(m3.sense == ObjSense::Minimize);
  // Free MPS writers sometimes put the value unindented on the next line.
  Model m4 = read_mps_string("NAME t\nOBJSENSE\nMAX\nROWS\n N obj\nCOLUMNS\n x obj 3\nENDATA\n");
  CHECK(m4.sense == ObjSense::Maximize);
  CHECK(error_contains("NAME t\nOBJSENSE UP\nROWS\n N obj\nENDATA\n", "invalid OBJSENSE"));
}

TEST(mps_objective_rhs_offset) {
  Model m = read_mps_string(
      "NAME t\nROWS\n N obj\n L c\nCOLUMNS\n x obj 1 c 1\nRHS\n RHS obj 12.5\n RHS c 3\nENDATA\n");
  CHECK_EQ(m.obj_offset, -12.5);
  CHECK_EQ(m.row_ub[0], 3.0);
  CHECK_NEAR(m.objective_value({2.0}), 2.0 - 12.5, 1e-12);
}

TEST(mps_extra_n_rows_dropped) {
  std::vector<std::string> w;
  Model m = read_mps_string(
      "NAME t\nROWS\n N obj\n N other\n L c\n N third\nCOLUMNS\n x obj 1 other 5\n x c 1 third 7\n"
      " y other 1\nRHS\n RHS other 9\nRANGES\n RNG other 1\nENDATA\n",
      &w);
  CHECK_EQ(w.size(), 1u);
  CHECK_EQ(m.num_rows(), 1);
  CHECK_EQ(m.num_cols(), 2);
  CHECK_EQ(m.num_nonzeros(), 1);
  CHECK_EQ(m.obj[0], 1.0);
  CHECK_EQ(m.obj[1], 0.0);
  CHECK_EQ(m.obj_offset, 0.0);
}

TEST(mps_objname_selects_objective) {
  Model m = read_mps_string(
      "NAME t\nOBJNAME second\nROWS\n N first\n N second\nCOLUMNS\n x first 1 second 2\nENDATA\n");
  CHECK_EQ(m.obj[0], 2.0);
  CHECK(error_contains("NAME t\nOBJNAME nope\nROWS\n N first\nCOLUMNS\n x first 1\nENDATA\n", "OBJNAME"));
}

TEST(mps_ranges_semantics) {
  const char* text =
      "NAME ranges\n"
      "ROWS\n N obj\n G g\n L l\n E ep\n E en\n E ez\n G gneg\n L lneg\n"
      "COLUMNS\n x obj 1 g 1\n x l 1 ep 1\n x en 1 ez 1\n x gneg 1 lneg 1\n"
      "RHS\n RHS g 1 l 10\n RHS ep 5 en 5\n RHS ez 5 gneg 1\n RHS lneg 10\n"
      "RANGES\n RNG g 4 l 4\n RNG ep 3 en -3\n RNG ez 0 gneg -4\n RNG lneg -4\n"
      "ENDATA\n";
  Model m = read_mps_string(text);
  auto lb = [&](const char* r) { return m.row_lb[row(m, r)]; };
  auto ub = [&](const char* r) { return m.row_ub[row(m, r)]; };
  CHECK_EQ(lb("g"), 1.0);
  CHECK_EQ(ub("g"), 5.0);
  CHECK_EQ(lb("l"), 6.0);
  CHECK_EQ(ub("l"), 10.0);
  CHECK_EQ(lb("ep"), 5.0);  // E, R > 0: [rhs, rhs + R]
  CHECK_EQ(ub("ep"), 8.0);
  CHECK_EQ(lb("en"), 2.0);  // E, R < 0: [rhs + R, rhs]
  CHECK_EQ(ub("en"), 5.0);
  CHECK_EQ(lb("ez"), 5.0);  // E, R = 0: equality
  CHECK_EQ(ub("ez"), 5.0);
  CHECK_EQ(lb("gneg"), 1.0);  // G/L use |R|
  CHECK_EQ(ub("gneg"), 5.0);
  CHECK_EQ(lb("lneg"), 6.0);
  CHECK_EQ(ub("lneg"), 10.0);
}

TEST(mps_ranges_on_objective_ignored) {
  std::vector<std::string> w;
  Model m = read_mps_string("NAME t\nROWS\n N obj\n G c\nCOLUMNS\n x obj 1 c 1\nRANGES\n RNG obj 5\nENDATA\n", &w);
  CHECK_EQ(w.size(), 1u);
  CHECK_EQ(m.row_lb[0], 0.0);
  CHECK_EQ(m.row_ub[0], kInf);
}

TEST(mps_marker_integer_blocks) {
  const char* text =
      "NAME ints\n"
      "ROWS\n N obj\n L c\n"
      "COLUMNS\n"
      " a obj 1 c 1\n"
      " MARKER 'MARKER' 'INTORG'\n"
      " b obj 1 c 1\n"
      " c obj 1 c 1\n"
      " MARKER 'MARKER' 'INTEND'\n"
      " d obj 1 c 1\n"
      "    MARKER                 'MARKER'                 'INTORG'\n"
      " e obj 1 c 1\n"
      "    MARKER                 'MARKER'                 'INTEND'\n"
      "RHS\n RHS c 10\n"
      "BOUNDS\n UP BND c 5\nENDATA\n";
  Model m = read_mps_string(text);
  REQUIRE(m.num_cols() == 5);
  CHECK(m.col_type[col(m, "a")] == VarType::Continuous);
  CHECK(m.col_type[col(m, "b")] == VarType::Integer);
  CHECK(m.col_type[col(m, "c")] == VarType::Integer);
  CHECK(m.col_type[col(m, "d")] == VarType::Continuous);
  CHECK(m.col_type[col(m, "e")] == VarType::Integer);
  CHECK_EQ(m.num_integers(), 3);
  // Integer columns without any BOUNDS entry are binary (HiGHS/SCIP convention).
  CHECK_EQ(m.col_lb[col(m, "b")], 0.0);
  CHECK_EQ(m.col_ub[col(m, "b")], 1.0);
  CHECK_EQ(m.col_ub[col(m, "c")], 5.0);
  CHECK(error_contains("NAME t\nROWS\n N obj\nCOLUMNS\n M 'MARKER' 'FOO'\nENDATA\n", "marker"));
}

TEST(mps_bound_types) {
  const char* text =
      "NAME b\nROWS\n N obj\nCOLUMNS\n"
      " up obj 1\n lo obj 1\n fx obj 1\n fr obj 1\n mi obj 1\n pl obj 1\n bv obj 1\n li obj 1\n ui obj 1\n"
      " both obj 1\n inf obj 1\n"
      "BOUNDS\n"
      " UP BND up 7.5\n LO BND lo -3\n FX BND fx 2.25\n FR BND fr\n MI BND mi\n UP BND pl 3\n PL BND pl\n"
      " BV BND bv\n LI BND li -2\n UI BND ui 9\n MI BND both\n UP BND both 4\n"
      " LO BND inf -1e30\n UP BND inf 1e31\n"
      "ENDATA\n";
  Model m = read_mps_string(text);
  auto lb = [&](const char* c) { return m.col_lb[col(m, c)]; };
  auto ub = [&](const char* c) { return m.col_ub[col(m, c)]; };
  auto ty = [&](const char* c) { return m.col_type[col(m, c)]; };
  CHECK_EQ(lb("up"), 0.0);
  CHECK_EQ(ub("up"), 7.5);
  CHECK_EQ(lb("lo"), -3.0);
  CHECK_EQ(ub("lo"), kInf);
  CHECK_EQ(lb("fx"), 2.25);
  CHECK_EQ(ub("fx"), 2.25);
  CHECK_EQ(lb("fr"), -kInf);
  CHECK_EQ(ub("fr"), kInf);
  CHECK_EQ(lb("mi"), -kInf);
  CHECK_EQ(ub("mi"), kInf);
  CHECK_EQ(lb("pl"), 0.0);
  CHECK_EQ(ub("pl"), kInf);
  CHECK(ty("bv") == VarType::Binary);
  CHECK_EQ(lb("bv"), 0.0);
  CHECK_EQ(ub("bv"), 1.0);
  CHECK(ty("li") == VarType::Integer);
  CHECK_EQ(lb("li"), -2.0);
  CHECK(ty("ui") == VarType::Integer);
  CHECK_EQ(ub("ui"), 9.0);
  CHECK_EQ(lb("both"), -kInf);
  CHECK_EQ(ub("both"), 4.0);
  CHECK_EQ(lb("inf"), -kInf);
  CHECK_EQ(ub("inf"), kInf);
  CHECK(ty("up") == VarType::Continuous);
  CHECK_EQ(m.num_integers(), 3);
}

TEST(mps_bv_with_value_and_no_set) {
  Model m = read_mps_string("NAME t\nROWS\n N obj\nCOLUMNS\n x obj 1\n y obj 1\nBOUNDS\n BV x 1\n BV BND y 1\nENDATA\n");
  CHECK(m.col_type[0] == VarType::Binary);
  CHECK(m.col_type[1] == VarType::Binary);
}

TEST(mps_negative_upper_bound) {
  std::vector<std::string> w;
  Model m = read_mps_string(
      "NAME t\nROWS\n N obj\nCOLUMNS\n x obj 1\n y obj 1\n z obj 1\n"
      "BOUNDS\n UP BND x -5\n LO BND y 0\n UP BND y -5\n LO BND z -10\n UP BND z -5\nENDATA\n",
      &w);
  // x: default lb 0 + negative UP => lb = -inf with a warning.
  CHECK_EQ(m.col_lb[0], -kInf);
  CHECK_EQ(m.col_ub[0], -5.0);
  // y: explicit LO 0 is kept (infeasible bounds, solver's business).
  CHECK_EQ(m.col_lb[1], 0.0);
  CHECK_EQ(m.col_ub[1], -5.0);
  // z: explicit lower bound untouched.
  CHECK_EQ(m.col_lb[2], -10.0);
  CHECK_EQ(w.size(), 1u);
  CHECK(w.size() == 1 && w[0].find("negative upper bound") != std::string::npos);
  // LO after a negative UP overrides the implied -inf.
  Model m2 = read_mps_string("NAME t\nROWS\n N obj\nCOLUMNS\n x obj 1\nBOUNDS\n UP BND x -5\n LO BND x -7\nENDATA\n", &w);
  CHECK_EQ(m2.col_lb[0], -7.0);
}

TEST(mps_missing_rhs_and_bounds) {
  Model m = read_mps_string("NAME t\nROWS\n N obj\n E e\n L l\nCOLUMNS\n x obj 1 e 1\n x l 2\nENDATA\n");
  CHECK_EQ(m.row_lb[0], 0.0);
  CHECK_EQ(m.row_ub[0], 0.0);
  CHECK_EQ(m.row_lb[1], -kInf);
  CHECK_EQ(m.row_ub[1], 0.0);
}

TEST(mps_no_objective_row) {
  Model m = read_mps_string("NAME t\nROWS\n L c\nCOLUMNS\n x c 1\nRHS\n RHS c 1\nENDATA\n");
  CHECK_EQ(m.num_cols(), 1);
  CHECK_EQ(m.obj[0], 0.0);
}

TEST(mps_numbers) {
  Model m = read_mps_string("NAME t\nROWS\n N obj\n L c\nCOLUMNS\n x obj +1.5 c 1e-3\n y obj -.25 c 2.E2\nENDATA\n");
  CHECK_EQ(m.obj[0], 1.5);
  CHECK_EQ(m.obj[1], -0.25);
  CHECK_EQ(coef(m, "c", "x"), 1e-3);
  CHECK_EQ(coef(m, "c", "y"), 200.0);
  // Exact round trip of 17-digit values.
  Model m2 = read_mps_string("NAME t\nROWS\n N obj\nCOLUMNS\n x obj 0.10000000000000001\n y obj 3.141592653589793\nENDATA\n");
  CHECK_EQ(m2.obj[0], 0.1);
  CHECK_EQ(m2.obj[1], 3.141592653589793);
}

TEST(mps_duplicates_summed_and_noncontiguous) {
  Model m = read_mps_string(
      "NAME t\nROWS\n N obj\n L c\n L d\nCOLUMNS\n x obj 1 c 1\n y c 1\n x c 2 d 3\nENDATA\n");
  CHECK_EQ(m.num_cols(), 2);
  CHECK_EQ(coef(m, "c", "x"), 3.0);
  CHECK_EQ(coef(m, "d", "x"), 3.0);
  CHECK_EQ(m.num_nonzeros(), 3);
}

TEST(mps_explicit_zero_coefficients) {
  Model m = read_mps_string("NAME t\nROWS\n N obj\n L c\nCOLUMNS\n x obj 0 c 0\n y c 1\nENDATA\n");
  CHECK_EQ(m.num_cols(), 2);
  CHECK_EQ(m.num_nonzeros(), 1);
}

TEST(mps_multiple_rhs_sets) {
  std::vector<std::string> w;
  Model m = read_mps_string("NAME t\nROWS\n N obj\n L c\nCOLUMNS\n x c 1\nRHS\n RHS1 c 4\n RHS2 c 8\nENDATA\n", &w);
  CHECK_EQ(m.row_ub[0], 4.0);
  CHECK_EQ(w.size(), 1u);
}

TEST(mps_missing_endata_warns) {
  std::vector<std::string> w;
  Model m = read_mps_string("NAME t\nROWS\n N obj\nCOLUMNS\n x obj 1\n", &w);
  CHECK_EQ(m.num_cols(), 1);
  CHECK_EQ(w.size(), 1u);
}

TEST(mps_errors) {
  // Line numbers are reported.
  CHECK(error_contains("NAME t\nROWS\n N obj\nCOLUMNS\n x nope 1\nENDATA\n", ":5: unknown row 'nope'"));
  CHECK(error_contains("NAME t\nROWS\n N obj\n X bad\nENDATA\n", ":4: ROWS: invalid row type"));
  CHECK(error_contains("NAME t\nROWS\n N obj\n L c\n G c\nENDATA\n", "duplicate row 'c'"));
  CHECK(error_contains("NAME t\nROWS\n N obj\nCOLUMNS\n x obj abc\nENDATA\n", "invalid number 'abc'"));
  CHECK(error_contains("NAME t\nROWS\n N obj\nCOLUMNS\n x obj 1.0D0\nENDATA\n", "invalid number"));
  CHECK(error_contains("NAME t\nROWS\n N obj\nCOLUMNS\n x obj\nENDATA\n", "COLUMNS: expected"));
  CHECK(error_contains("NAME t\nROWS\n N obj\nCOLUMNS\n x obj 1\nBOUNDS\n UP BND y 1\nENDATA\n",
                       "column 'y' in BOUNDS does not appear in COLUMNS"));
  CHECK(error_contains("NAME t\nROWS\n N obj\nCOLUMNS\n x obj 1\nSOS\n x 1\nENDATA\n", "member line before a set header"));
  CHECK(error_contains("NAME t\nROWS\n N obj\nCOLUMNS\n x obj 1\nSOS\n S1 SOS s\n y 1\nENDATA\n",
                       "column 'y' in SOS does not appear"));
  CHECK(error_contains("NAME t\nROWS\n N obj\n L c\nCOLUMNS\n x c 1\nINDICATORS\n IF c x 2\nENDATA\n",
                       "the value must be 0 or 1"));
  CHECK(error_contains("NAME t\nROWS\n N obj\n L c\nCOLUMNS\n x c 1\nINDICATORS\n IF obj x 1\nENDATA\n",
                       "objective (N) row"));
  CHECK(error_contains("NAME t\nROWS\n N obj\nCOLUMNS\n x obj 1\nBOUNDS\n XX BND x 1\nENDATA\n", "invalid bound type"));
  CHECK(error_contains("NAME t\nROWS\n N obj\nCOLUMNS\n x obj 1\nBOUNDS\n UP BND x\nENDATA\n", "BOUNDS: expected"));
  CHECK(error_contains("NAME t\nROWS\n N obj\nCOLUMNS\n x obj 1\nQUADOBJ\n x x 1\nENDATA\n", "unsupported MPS section"));
  CHECK(error_contains("NAME t\nFOO\nENDATA\n", "unknown MPS section 'FOO'"));
  CHECK(error_contains("  x obj 1\n", "data line outside of a section"));
  CHECK(error_contains("NAME t\nROWS\n N obj\n L c\nCOLUMNS\n x c 1\nRHS\n RHS c 1 d 2\nENDATA\n", "unknown row 'd'"));
  CHECK(error_contains("NAME t\nROWS\n N obj\nCOLUMNS\n x obj 1 obj 1 obj 1 obj 1 obj 1\nENDATA\n", "too many"));
  CHECK_THROWS(read_mps("/nonexistent/dir/model.mps"), MpsError);
  CHECK_THROWS(read_mps("model.mps.gz"), MpsError);
}

TEST(mps_read_file) {
  const auto path = std::filesystem::temp_directory_path() / "milp_test_read.mps";
  std::FILE* f = std::fopen(path.string().c_str(), "wb");
  REQUIRE(f != nullptr);
  std::fputs(kTestProb, f);
  std::fclose(f);
  Model m = read_mps(path.string());
  CHECK_EQ(m.num_rows(), 3);
  CHECK_EQ(m.num_cols(), 3);
  try {
    read_mps_string("NAME t\nROWS\n N obj\nCOLUMNS\n x zz 1\n");
  } catch (const MpsError& e) {
    CHECK(std::string(e.what()).rfind("<string>:5:", 0) == 0);
  }
  std::filesystem::remove(path);
}

namespace {

void check_same_model(const Model& a, const Model& b) {
  CHECK_EQ(a.name, b.name);
  CHECK(a.sense == b.sense);
  CHECK_EQ(a.obj_offset, b.obj_offset);
  CHECK(a.col_names == b.col_names);
  CHECK(a.row_names == b.row_names);
  CHECK(a.obj == b.obj);
  CHECK(a.col_lb == b.col_lb);
  CHECK(a.col_ub == b.col_ub);
  CHECK(a.col_type == b.col_type);
  CHECK(a.row_lb == b.row_lb);
  CHECK(a.row_ub == b.row_ub);
  CHECK(a.A.start == b.A.start);
  CHECK(a.A.index == b.A.index);
  CHECK(a.A.value == b.A.value);
}

}  // namespace

TEST(mps_write_read_roundtrip) {
  ModelBuilder bld;
  bld.model().name = "round";
  bld.model().sense = ObjSense::Maximize;
  bld.model().obj_offset = 0.1;
  Index x = bld.add_col("x", 1.0 / 3.0, 0.0, kInf);
  Index y = bld.add_col("y", -2.0, -kInf, 5.0, VarType::Integer);
  Index z = bld.add_col("z", 0.0, 0.0, 1.0, VarType::Binary);
  Index w = bld.add_col("w", 1e-17, -kInf, kInf);
  Index v = bld.add_col("v", 3.0, 0.0, -2.0);  // infeasible bounds, ub < 0
  Index u = bld.add_col("u", 0.0, 2.0, 2.0, VarType::Integer);
  Index t = bld.add_col("t", 0.0, 0.0, kInf, VarType::Integer);
  Index s = bld.add_col("s", 1.0, 0.0, kInf);  // empty column
  (void)s;
  Index e = bld.add_row("eq", 1.5, 1.5);
  Index l = bld.add_row("le", -kInf, 7.0);
  Index g = bld.add_row("ge", -3.0, kInf);
  Index r = bld.add_row("rng", -1.0, 2.0);
  bld.add_coef(e, x, 0.7);
  bld.add_coef(e, y, -1.25);
  bld.add_coef(l, z, 2.0);
  bld.add_coef(l, w, 1e-9);
  bld.add_coef(g, v, 3.0);
  bld.add_coef(r, u, 1.0);
  bld.add_coef(r, t, 1.0);
  bld.add_coef(g, x, 123456789.123);
  Model a = bld.build();
  const std::string text = write_mps_string(a);
  std::vector<std::string> warnings;
  Model b = read_mps_string(text, &warnings);
  CHECK(warnings.empty());
  check_same_model(a, b);

  // And through a file.
  const auto path = std::filesystem::temp_directory_path() / "milp_test_roundtrip.mps";
  write_mps(a, path.string());
  Model c = read_mps(path.string());
  check_same_model(a, c);
  std::filesystem::remove(path);

  // Read -> write -> read of the reference problem is stable as well.
  Model p = read_mps_string(kTestProb);
  check_same_model(p, read_mps_string(write_mps_string(p)));
}

TEST(mps_write_sanitizes_names) {
  ModelBuilder bld;
  Index a = bld.add_col("", 1.0, 0.0, 1.0);
  Index b2 = bld.add_col("has space", 1.0, 0.0, 1.0);
  Index r = bld.add_row("OBJ", -kInf, 1.0);  // collides with the default objective name
  bld.add_coef(r, a, 1.0);
  bld.add_coef(r, b2, 1.0);
  Model m = bld.build();
  Model back = read_mps_string(write_mps_string(m));
  CHECK_EQ(back.col_names[0], "C0");
  CHECK_EQ(back.col_names[1], "has_space");
  CHECK_EQ(back.row_names[0], "OBJ");
  CHECK_EQ(back.num_rows(), 1);
  CHECK(back.obj == m.obj);
}
