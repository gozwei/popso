#include "milp/io/mps_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "milp/io/charconv_compat.hpp"

namespace milp {
namespace {

using std::string_view;

constexpr double kMpsInf = 1e30;  // |value| >= this is treated as infinite
constexpr int kMaxTokens = 8;
constexpr int kMaxPrintedWarnings = 10;

bool iequals(string_view a, string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    char x = a[i], y = b[i];
    if (x >= 'a' && x <= 'z') x = static_cast<char>(x - 'a' + 'A');
    if (y >= 'a' && y <= 'z') y = static_cast<char>(y - 'a' + 'A');
    if (x != y) return false;
  }
  return true;
}

inline bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v'; }

double to_inf(double v) {
  if (v >= kMpsInf) return kInf;
  if (v <= -kMpsInf) return -kInf;
  return v;
}

// Open-addressing hash table from names (views into the input buffer, which
// outlives the table) to indices. Much faster than std::unordered_map here.
class NameTable {
 public:
  explicit NameTable(size_t expected = 64) {
    size_t cap = 16;
    while (cap < 2 * expected) cap *= 2;
    slots_.resize(cap);
  }

  const Index* find(string_view key) const {
    const uint64_t h = hash(key);
    const size_t mask = slots_.size() - 1;
    for (size_t i = h & mask;; i = (i + 1) & mask) {
      const Slot& s = slots_[i];
      if (!s.used) return nullptr;
      if (s.hash == h && s.key == key) return &s.value;
    }
  }

  // Inserts key -> value; returns false (and leaves the table unchanged) if
  // the key is already present.
  bool insert(string_view key, Index value) {
    if (2 * (size_ + 1) > slots_.size()) grow();
    const uint64_t h = hash(key);
    const size_t mask = slots_.size() - 1;
    for (size_t i = h & mask;; i = (i + 1) & mask) {
      Slot& s = slots_[i];
      if (!s.used) {
        s = Slot{key, h, value, true};
        ++size_;
        return true;
      }
      if (s.hash == h && s.key == key) return false;
    }
  }

 private:
  struct Slot {
    string_view key;
    uint64_t hash = 0;
    Index value = 0;
    bool used = false;
  };

  static uint64_t hash(string_view s) {
    uint64_t h = 1469598103934665603ULL;  // FNV-1a
    for (char c : s) h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ULL;
    return h ^ (h >> 29);
  }

  void grow() {
    std::vector<Slot> old(slots_.size() * 2);
    old.swap(slots_);
    const size_t mask = slots_.size() - 1;
    for (const Slot& s : old) {
      if (!s.used) continue;
      size_t i = s.hash & mask;
      while (slots_[i].used) i = (i + 1) & mask;
      slots_[i] = s;
    }
  }

  std::vector<Slot> slots_;
  size_t size_ = 0;
};

enum class Section { None, Name, ObjSense, ObjName, Rows, Columns, Rhs, Ranges, Bounds, Sos, Indicators, End };

constexpr Index kObjRow = -1;
constexpr Index kDroppedRow = -2;

struct RowInfo {
  char type = 'E';
  double rhs = 0.0;
  double range = 0.0;
  bool has_range = false;
};

class MpsParser {
 public:
  MpsParser(string_view text, std::string source, std::vector<std::string>* warnings)
      : text_(text),
        source_(std::move(source)),
        warnings_(warnings),
        rows_(text.size() / 400 + 16),
        cols_(text.size() / 200 + 16) {}

  Model parse();

 private:
  [[noreturn]] void error(const std::string& msg) const {
    throw MpsError(source_ + ":" + std::to_string(line_no_) + ": " + msg);
  }
  void warn(const std::string& msg) {
    std::string full = source_ + ":" + std::to_string(line_no_) + ": warning: " + msg;
    if (warnings_) {
      warnings_->push_back(std::move(full));
    } else if (num_warnings_ < kMaxPrintedWarnings) {
      std::fprintf(stderr, "%s\n", full.c_str());
    }
    ++num_warnings_;
  }

  void process_line(const char* begin, const char* end);
  bool handle_header(string_view line);
  void parse_rows();
  void parse_columns();
  void parse_rhs();
  void parse_ranges();
  void parse_bounds();
  void parse_sos();
  void parse_indicators();
  void parse_sense(string_view tok);

  double number(string_view tok) const {
    double v;
    if (!io_detail::parse_double(tok, v)) error("invalid number '" + std::string(tok) + "'");
    return v;
  }
  Index row_index(string_view name) const {
    const Index* r = rows_.find(name);
    if (!r) error("unknown row '" + std::string(name) + "'");
    return *r;
  }
  Index col_index(string_view name, const char* section = "BOUNDS") const {
    const Index* c = cols_.find(name);
    if (!c) error("column '" + std::string(name) + "' in " + section + " does not appear in COLUMNS");
    return *c;
  }
  // Checks the optional set name of RHS/RANGES/BOUNDS entries. Only the first
  // named set is used; an empty set name (fixed format, blank field) matches.
  bool accept_set(string_view set, string_view& chosen, bool& warned, const char* what) {
    if (set.empty()) return true;
    if (chosen.empty()) {
      chosen = set;
      return true;
    }
    if (set == chosen) return true;
    if (!warned) {
      warn(std::string("multiple ") + what + " sets; ignoring entries of set '" + std::string(set) + "'");
      warned = true;
    }
    return false;
  }

  string_view text_;
  std::string source_;
  std::vector<std::string>* warnings_;
  size_t line_no_ = 0;
  int num_warnings_ = 0;

  string_view tok_[kMaxTokens];
  int ntok_ = 0;
  Section section_ = Section::None;
  bool seen_end_ = false;

  std::string name_;
  ObjSense sense_ = ObjSense::Minimize;
  string_view obj_name_;  // from OBJNAME
  bool have_obj_ = false;
  double obj_offset_ = 0.0;

  NameTable rows_;
  std::vector<string_view> row_names_;
  std::vector<RowInfo> row_info_;

  NameTable cols_;
  std::vector<string_view> col_names_;
  std::vector<double> obj_, lb_, ub_;
  std::vector<VarType> type_;
  std::vector<char> lb_set_;
  std::vector<char> in_bounds_;  // column appears in the BOUNDS section
  std::vector<char> semi_;       // SC / SI bound: semi-continuous (semi-integer if the column is integer)
  std::vector<SosConstraint> sos_;
  struct IndicatorSpec {
    Index row, col;
    int value;
    size_t line;
  };
  std::vector<IndicatorSpec> indicators_;
  std::vector<Index> trip_col_, trip_row_;
  std::vector<double> trip_val_;
  bool in_int_block_ = false;
  Index cur_col_ = -1;
  string_view cur_col_name_;

  string_view rhs_set_, ranges_set_, bounds_set_;
  bool rhs_warned_ = false, ranges_warned_ = false, bounds_warned_ = false;
};

Model MpsParser::parse() {
  trip_col_.reserve(text_.size() / 24);
  trip_row_.reserve(text_.size() / 24);
  trip_val_.reserve(text_.size() / 24);
  const char* p = text_.data();
  const char* end = p + text_.size();
  while (p < end && !seen_end_) {
    const char* nl = static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
    const char* le = nl ? nl : end;
    ++line_no_;
    process_line(p, le);
    p = nl ? nl + 1 : end;
  }
  if (!seen_end_) warn("missing ENDATA");
  if (!obj_name_.empty() && !have_obj_) error("OBJNAME row '" + std::string(obj_name_) + "' not found in ROWS");
  if (!warnings_ && num_warnings_ > kMaxPrintedWarnings)
    std::fprintf(stderr, "%s: %d more warnings suppressed\n", source_.c_str(), num_warnings_ - kMaxPrintedWarnings);

  Model m;
  m.name = name_;
  m.sense = sense_;
  m.obj_offset = obj_offset_;
  const size_t n = col_names_.size(), nr = row_info_.size();
  m.col_names.reserve(n);
  for (string_view s : col_names_) m.col_names.emplace_back(s);
  m.obj = std::move(obj_);
  // Integer columns without any BOUNDS entry are binary (HiGHS/SCIP/MIPLIB convention).
  for (size_t j = 0; j < n; ++j) {
    if (type_[j] == VarType::Integer && !in_bounds_[j]) ub_[j] = 1.0;
    if (semi_[j]) type_[j] = is_integral_type(type_[j]) ? VarType::SemiInteger : VarType::SemiContinuous;
  }
  m.col_lb = std::move(lb_);
  m.col_ub = std::move(ub_);
  m.col_type = std::move(type_);
  m.row_names.reserve(nr);
  m.row_lb.resize(nr);
  m.row_ub.resize(nr);
  for (size_t i = 0; i < nr; ++i) {
    m.row_names.emplace_back(row_names_[i]);
    const RowInfo& r = row_info_[i];
    double lo = r.rhs, hi = r.rhs;
    if (r.type == 'L') lo = -kInf;
    if (r.type == 'G') hi = kInf;
    if (r.has_range) {
      const double R = r.range;
      if (r.type == 'L') lo = r.rhs - std::abs(R);
      else if (r.type == 'G') hi = r.rhs + std::abs(R);
      else if (R > 0) hi = r.rhs + R;
      else lo = r.rhs + R;
    }
    m.row_lb[i] = lo;
    m.row_ub[i] = hi;
  }
  m.A = SparseMatrix::from_triplets(static_cast<Index>(n), static_cast<Index>(nr), trip_col_, trip_row_, trip_val_);
  m.sos = std::move(sos_);
  if (!indicators_.empty()) {
    // Rows named in INDICATORS are indicator constraints, not ordinary rows.
    const SparseMatrix R = m.A.transpose();
    std::vector<char> del(nr, 0);
    for (const IndicatorSpec& spec : indicators_) {
      line_no_ = spec.line;
      if (del[spec.row]) error("row '" + m.row_names[spec.row] + "' appears twice in INDICATORS");
      del[spec.row] = 1;
      IndicatorConstraint c;
      c.name = m.row_names[spec.row];
      c.binary = spec.col;
      c.value = spec.value;
      c.lb = m.row_lb[spec.row];
      c.ub = m.row_ub[spec.row];
      for (Index k = R.begin(spec.row); k < R.end(spec.row); ++k) {
        c.cols.push_back(R.index[k]);
        c.coefs.push_back(R.value[k]);
      }
      VarType& t = m.col_type[spec.col];
      if (!is_integral_type(t) || is_semi_type(t)) {
        warn("indicator variable '" + m.col_names[spec.col] + "' is not integer; made binary");
        t = VarType::Binary;
      }
      if (m.col_lb[spec.col] < 0.0 || m.col_ub[spec.col] > 1.0) {
        if (t != VarType::Binary) warn("indicator variable '" + m.col_names[spec.col] + "' bounds clamped to [0, 1]");
        m.col_lb[spec.col] = std::max(m.col_lb[spec.col], 0.0);
        m.col_ub[spec.col] = std::min(m.col_ub[spec.col], 1.0);
      }
      m.indicators.push_back(std::move(c));
    }
    m.remove_rows(del);
  }
  try {
    m.validate();
  } catch (const std::exception& e) {
    throw MpsError(source_ + ": " + e.what());
  }
  return m;
}

void MpsParser::process_line(const char* begin, const char* end) {
  while (end > begin && is_space(end[-1])) --end;
  if (begin == end || *begin == '*') return;
  ntok_ = 0;
  int total = 0;
  for (const char* p = begin; p < end;) {
    while (p < end && is_space(*p)) ++p;
    if (p == end) break;
    const char* s = p;
    while (p < end && !is_space(*p)) ++p;
    if (total < kMaxTokens) tok_[total] = string_view(s, static_cast<size_t>(p - s));
    ++total;
  }
  if (total == 0) return;
  if (total > kMaxTokens) error("too many fields on line");
  ntok_ = total;

  const bool indented = is_space(*begin);
  if (!indented && handle_header(string_view(begin, static_cast<size_t>(end - begin)))) return;

  switch (section_) {
    case Section::ObjSense:
      parse_sense(tok_[0]);
      break;
    case Section::ObjName:
      obj_name_ = tok_[0];
      break;
    case Section::Rows: parse_rows(); break;
    case Section::Columns: parse_columns(); break;
    case Section::Rhs: parse_rhs(); break;
    case Section::Ranges: parse_ranges(); break;
    case Section::Bounds: parse_bounds(); break;
    case Section::Sos: parse_sos(); break;
    case Section::Indicators: parse_indicators(); break;
    default: error("data line outside of a section");
  }
}

// Returns true if the (non-indented) line is a section header.
bool MpsParser::handle_header(string_view line) {
  const string_view kw = tok_[0];
  auto rest = [&]() {
    string_view r = line.substr(kw.size());
    while (!r.empty() && is_space(r.front())) r.remove_prefix(1);
    return r;
  };
  if (iequals(kw, "NAME")) {
    name_ = std::string(rest());
    section_ = Section::Name;
    return true;
  }
  if (iequals(kw, "OBJSENSE") || iequals(kw, "OBJSENSE:")) {
    section_ = Section::ObjSense;
    if (ntok_ >= 2) parse_sense(tok_[1]);
    return true;
  }
  if (iequals(kw, "OBJNAME")) {
    section_ = Section::ObjName;
    if (ntok_ >= 2) obj_name_ = tok_[1];
    return true;
  }
  if (iequals(kw, "ENDATA")) {
    section_ = Section::End;
    seen_end_ = true;
    return true;
  }
  struct DataSection {
    const char* name;
    Section sec;
  };
  static const DataSection kData[] = {{"ROWS", Section::Rows},
                                      {"COLUMNS", Section::Columns},
                                      {"RHS", Section::Rhs},
                                      {"RANGES", Section::Ranges},
                                      {"BOUNDS", Section::Bounds},
                                      {"SOS", Section::Sos},
                                      {"INDICATORS", Section::Indicators}};
  const bool in_data = section_ == Section::ObjSense || section_ == Section::ObjName || section_ == Section::Rows ||
                       section_ == Section::Columns || section_ == Section::Rhs || section_ == Section::Ranges ||
                       section_ == Section::Bounds || section_ == Section::Sos || section_ == Section::Indicators;
  for (const auto& d : kData) {
    if (!iequals(kw, d.name)) continue;
    if (ntok_ == 1) {
      section_ = d.sec;
      if (d.sec == Section::Columns || d.sec == Section::Rhs) in_int_block_ = false;
      return true;
    }
    if (in_data) return false;  // e.g. an unindented RHS entry with set name "RHS"
    error("unexpected fields after section " + std::string(kw));
  }
  static const char* const kUnsupported[] = {"QUADOBJ",  "QSECTION", "QMATRIX", "QCMATRIX", "LAZYCONS", "USERCUTS",
                                             "GENCONS",  "PWLOBJ",   "CSECTION", "BRANCH",   "SETS"};
  for (const char* u : kUnsupported) {
    if (iequals(kw, u)) error("unsupported MPS section '" + std::string(kw) + "'");
  }
  if (in_data) return false;  // free MPS data line starting in column 1
  error("unknown MPS section '" + std::string(kw) + "'");
}

void MpsParser::parse_sense(string_view tok) {
  if (iequals(tok, "MAX") || iequals(tok, "MAXIMIZE") || iequals(tok, "MAXIMISE")) {
    sense_ = ObjSense::Maximize;
  } else if (iequals(tok, "MIN") || iequals(tok, "MINIMIZE") || iequals(tok, "MINIMISE")) {
    sense_ = ObjSense::Minimize;
  } else {
    error("invalid OBJSENSE '" + std::string(tok) + "'");
  }
}

void MpsParser::parse_rows() {
  if (ntok_ != 2) error("ROWS: expected '<type> <name>'");
  const string_view t = tok_[0];
  char type = t.size() == 1 ? t[0] : '?';
  if (type >= 'a' && type <= 'z') type = static_cast<char>(type - 'a' + 'A');
  const string_view name = tok_[1];
  Index value;
  if (type == 'N') {
    const bool is_obj = obj_name_.empty() ? !have_obj_ : name == obj_name_;
    value = is_obj ? kObjRow : kDroppedRow;
    if (is_obj) have_obj_ = true;
  } else if (type == 'E' || type == 'L' || type == 'G') {
    value = static_cast<Index>(row_info_.size());
  } else {
    error("ROWS: invalid row type '" + std::string(t) + "'");
  }
  if (!rows_.insert(name, value)) error("duplicate row '" + std::string(name) + "'");
  if (value >= 0) {
    row_names_.push_back(name);
    RowInfo r;
    r.type = type;
    row_info_.push_back(r);
  }
}

void MpsParser::parse_columns() {
  if (ntok_ >= 3 && tok_[1] == "'MARKER'") {
    if (tok_[2] == "'INTORG'") in_int_block_ = true;
    else if (tok_[2] == "'INTEND'") in_int_block_ = false;
    else error("COLUMNS: invalid marker '" + std::string(tok_[2]) + "'");
    return;
  }
  if (ntok_ != 3 && ntok_ != 5) error("COLUMNS: expected '<column> <row> <value> [<row> <value>]'");
  const string_view name = tok_[0];
  if (cur_col_ < 0 || name != cur_col_name_) {
    const Index next = static_cast<Index>(col_names_.size());
    if (cols_.insert(name, next)) {
      col_names_.push_back(name);
      obj_.push_back(0.0);
      lb_.push_back(0.0);
      ub_.push_back(kInf);
      type_.push_back(in_int_block_ ? VarType::Integer : VarType::Continuous);
      lb_set_.push_back(0);
      in_bounds_.push_back(0);
      semi_.push_back(0);
      cur_col_ = next;
    } else {
      cur_col_ = *cols_.find(name);  // non-contiguous column: entries are merged
    }
    cur_col_name_ = name;
  }
  for (int k = 1; k + 1 < ntok_; k += 2) {
    const Index r = row_index(tok_[k]);
    const double v = number(tok_[k + 1]);
    if (r == kObjRow) {
      obj_[cur_col_] += v;
    } else if (r >= 0 && v != 0.0) {
      trip_col_.push_back(cur_col_);
      trip_row_.push_back(r);
      trip_val_.push_back(v);
    }
  }
}

void MpsParser::parse_rhs() {
  if (ntok_ < 2 || ntok_ > 5) error("RHS: expected '[<set>] <row> <value> [<row> <value>]'");
  const int first = ntok_ % 2;  // odd field count => leading set name
  if (!accept_set(first ? tok_[0] : string_view(), rhs_set_, rhs_warned_, "RHS")) return;
  for (int k = first; k + 1 < ntok_; k += 2) {
    const Index r = row_index(tok_[k]);
    const double v = number(tok_[k + 1]);
    if (r == kObjRow) obj_offset_ = -v;
    else if (r >= 0) row_info_[r].rhs = to_inf(v);
  }
}

void MpsParser::parse_ranges() {
  if (ntok_ < 2 || ntok_ > 5) error("RANGES: expected '[<set>] <row> <value> [<row> <value>]'");
  const int first = ntok_ % 2;
  if (!accept_set(first ? tok_[0] : string_view(), ranges_set_, ranges_warned_, "RANGES")) return;
  for (int k = first; k + 1 < ntok_; k += 2) {
    const Index r = row_index(tok_[k]);
    const double v = number(tok_[k + 1]);
    if (r < 0) {
      warn("RANGES entry on N row '" + std::string(tok_[k]) + "' ignored");
      continue;
    }
    row_info_[r].range = to_inf(v);
    row_info_[r].has_range = true;
  }
}

void MpsParser::parse_bounds() {
  const string_view type = tok_[0];
  enum { kNeedsValue, kNoValue, kOptionalValue } kind;
  if (iequals(type, "UP") || iequals(type, "LO") || iequals(type, "FX") || iequals(type, "LI") ||
      iequals(type, "UI")) {
    kind = kNeedsValue;
  } else if (iequals(type, "FR") || iequals(type, "MI") || iequals(type, "PL")) {
    kind = kNoValue;
  } else if (iequals(type, "BV") || iequals(type, "SC") || iequals(type, "SI")) {
    kind = kOptionalValue;
  } else {
    error("BOUNDS: invalid bound type '" + std::string(type) + "'");
  }

  string_view set, col, val;
  if (kind == kNeedsValue) {
    if (ntok_ == 4) set = tok_[1], col = tok_[2], val = tok_[3];
    else if (ntok_ == 3) col = tok_[1], val = tok_[2];
    else error("BOUNDS: expected '<type> [<set>] <column> <value>'");
    double dummy;
    if (!io_detail::parse_double(val, dummy) && ntok_ == 3)
      error("BOUNDS: expected '<type> [<set>] <column> <value>' (missing value?)");
  } else {
    double dummy;
    if (ntok_ == 2) col = tok_[1];
    else if (ntok_ == 4) set = tok_[1], col = tok_[2];
    else if (ntok_ == 3 && kind == kOptionalValue && cols_.find(tok_[1]) && io_detail::parse_double(tok_[2], dummy))
      col = tok_[1];
    else if (ntok_ == 3) set = tok_[1], col = tok_[2];
    else error("BOUNDS: expected '<type> [<set>] <column>'");
  }
  if (!accept_set(set, bounds_set_, bounds_warned_, "BOUNDS")) return;
  const Index j = col_index(col);
  in_bounds_[j] = 1;
  const double v = kind == kNeedsValue ? to_inf(number(val)) : 0.0;
  if ((type[0] | 0x20) == 's') {
    // SC / SI: semi-continuous (semi-integer for SI or an integer column) with upper bound `value`
    // (none given: the upper bound is kept).
    semi_[j] = 1;
    if ((type[1] | 0x20) == 'i') type_[j] = VarType::Integer;
    if (ntok_ == 4 || (ntok_ == 3 && col == tok_[1])) ub_[j] = to_inf(number(tok_[ntok_ - 1]));
    return;
  }

  auto set_ub = [&](double u) {
    ub_[j] = u;
    if (u < 0 && lb_[j] == 0.0 && !lb_set_[j]) {
      lb_[j] = -kInf;
      warn("negative upper bound on column '" + std::string(col) + "' with default lower bound 0; lower bound set to -inf");
    }
  };
  auto set_lb = [&](double l) {
    lb_[j] = l;
    lb_set_[j] = 1;
  };

  switch (type[0] | 0x20) {  // lower-case first letter
    case 'u':
      if ((type[1] | 0x20) == 'i') type_[j] = VarType::Integer;
      set_ub(v);
      break;
    case 'l':
      if ((type[1] | 0x20) == 'i') type_[j] = VarType::Integer;
      set_lb(v);
      break;
    case 'f':
      if ((type[1] | 0x20) == 'x') {
        set_lb(v);
        ub_[j] = v;
      } else {
        set_lb(-kInf);
        ub_[j] = kInf;
      }
      break;
    case 'm': set_lb(-kInf); break;
    case 'p': ub_[j] = kInf; break;
    case 'b':
      type_[j] = VarType::Binary;
      set_lb(0.0);
      ub_[j] = 1.0;
      break;
    default: break;
  }
}

// SOS section, as written by CPLEX (" S1 SOS <name> <priority>") or SCIP (" S1 <name>"), followed by
// member lines "<column> <weight>" (or "<column>:<weight>"; without a weight: its position).
void MpsParser::parse_sos() {
  const string_view t0 = tok_[0];
  const bool type_tok = t0.size() == 2 && (t0[0] | 0x20) == 's' && (t0[1] == '1' || t0[1] == '2');
  double dummy;
  const bool header = type_tok && (ntok_ == 1 || iequals(tok_[1], "SOS") || !io_detail::parse_double(tok_[1], dummy));
  if (header) {
    SosConstraint s;
    s.type = t0[1] - '0';
    int k = 1;
    if (ntok_ > k && iequals(tok_[k], "SOS")) ++k;
    s.name = ntok_ > k ? std::string(tok_[k]) : "SOS" + std::to_string(sos_.size() + 1);
    sos_.push_back(std::move(s));
    return;
  }
  if (sos_.empty()) error("SOS: member line before a set header ('S1' or 'S2')");
  SosConstraint& s = sos_.back();
  string_view name = tok_[0];
  string_view wtok;
  if (ntok_ == 1) {
    const size_t c = name.find(':');
    if (c != string_view::npos) wtok = name.substr(c + 1), name = name.substr(0, c);
  } else if (ntok_ == 2) {
    wtok = tok_[1];
  } else {
    error("SOS: expected '<column> <weight>'");
  }
  const Index j = col_index(name, "SOS");
  s.cols.push_back(j);
  s.weights.push_back(wtok.empty() ? static_cast<double>(s.cols.size()) : number(wtok));
}

// INDICATORS section: " IF <row> <binary column> [<value>]" (value 0 or 1, default 1).
void MpsParser::parse_indicators() {
  if (ntok_ < 3 || ntok_ > 4 || !iequals(tok_[0], "IF")) error("INDICATORS: expected 'IF <row> <column> <value>'");
  const Index r = row_index(tok_[1]);
  if (r < 0) error("INDICATORS: row '" + std::string(tok_[1]) + "' is an objective (N) row");
  const Index j = col_index(tok_[2], "INDICATORS");
  int value = 1;
  if (ntok_ == 4) {
    const double v = number(tok_[3]);
    if (v != 0.0 && v != 1.0) error("INDICATORS: the value must be 0 or 1");
    value = static_cast<int>(v);
  }
  indicators_.push_back({r, j, value, line_no_});
}

std::string read_file(const std::string& path) {
  if (path.size() >= 3 && path.compare(path.size() - 3, 3, ".gz") == 0)
    throw MpsError(path + ": gzipped MPS files are not supported; decompress first");
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw MpsError(path + ": cannot open file");
  std::string data;
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (!ec) data.reserve(static_cast<size_t>(size));
  char buf[1 << 16];
  size_t got;
  while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) data.append(buf, got);
  const bool bad = std::ferror(f) != 0;
  std::fclose(f);
  if (bad) throw MpsError(path + ": read error");
  return data;
}

// ---------------------------------------------------------------------------
// Writer

std::string sanitize(const std::string& name, char prefix, size_t idx) {
  if (name.empty()) return prefix + std::to_string(idx);
  std::string s = name;
  for (char& c : s)
    if (is_space(c) || c == '\n') c = '_';
  return s;
}

void append_num(std::string& out, double v) {
  char buf[40];
  if (v == kInf) {
    out += "1e+30";
  } else if (v == -kInf) {
    out += "-1e+30";
  } else {
    out.append(buf, static_cast<size_t>(io_detail::format_double(buf, v)));
  }
}

void append_entry(std::string& out, const std::string& a, const std::string& b, double v) {
  out += "    ";
  out += a;
  out += "  ";
  out += b;
  out += "  ";
  append_num(out, v);
  out += '\n';
}

}  // namespace

Model read_mps_string(const std::string& text, std::vector<std::string>* warnings) {
  MpsParser parser(text, "<string>", warnings);
  return parser.parse();
}

Model read_mps(const std::string& path, std::vector<std::string>* warnings) {
  const std::string data = read_file(path);
  MpsParser parser(data, path, warnings);
  return parser.parse();
}

std::string write_mps_string(const Model& model_in) {
  // Indicator constraints are written as ordinary rows (appended) named in the INDICATORS section.
  Model with_ind;
  std::vector<std::pair<size_t, Index>> ind_rows;  // (indicator, row); vacuous ones (no finite side) are left out
  if (!model_in.indicators.empty()) {
    with_ind = model_in;
    with_ind.indicators.clear();
    std::vector<Index> J, I;
    std::vector<double> V;
    for (Index j = 0; j < model_in.num_cols(); ++j) {
      for (Index p = model_in.A.begin(j); p < model_in.A.end(j); ++p) {
        J.push_back(j);
        I.push_back(model_in.A.index[p]);
        V.push_back(model_in.A.value[p]);
      }
    }
    for (size_t q = 0; q < model_in.indicators.size(); ++q) {
      const IndicatorConstraint& c = model_in.indicators[q];
      if (c.lb == -kInf && c.ub == kInf) continue;
      const Index i = with_ind.num_rows();
      ind_rows.emplace_back(q, i);
      with_ind.row_names.push_back(c.name.empty() ? "IND" + std::to_string(i) : c.name);
      with_ind.row_lb.push_back(c.lb);
      with_ind.row_ub.push_back(c.ub);
      for (size_t k = 0; k < c.cols.size(); ++k) J.push_back(c.cols[k]), I.push_back(i), V.push_back(c.coefs[k]);
    }
    with_ind.A = SparseMatrix::from_triplets(with_ind.num_cols(), with_ind.num_rows(), J, I, V);
  }
  const Model& model = model_in.indicators.empty() ? model_in : with_ind;
  const Index n = model.num_cols(), m = model.num_rows();
  std::vector<std::string> cn(n), rn(m);
  for (Index j = 0; j < n; ++j) cn[j] = sanitize(model.col_names[j], 'C', j);
  for (Index i = 0; i < m; ++i) rn[i] = sanitize(model.row_names[i], 'R', i);
  {
    // Unique row names (indicator names may repeat row names).
    std::unordered_set<std::string> seen;
    for (Index i = 0; i < m; ++i) {
      std::string nm = rn[i];
      for (int k = 1; !seen.insert(nm).second; ++k) nm = rn[i] + "_" + std::to_string(k);
      rn[i] = nm;
    }
  }
  std::unordered_set<std::string_view> row_set(rn.begin(), rn.end());
  std::string objname = "OBJ";
  for (int k = 1; row_set.count(objname); ++k) objname = "OBJ_" + std::to_string(k);

  std::string out;
  out.reserve(64 + 48 * static_cast<size_t>(model.num_nonzeros() + n + m));
  out += "NAME";
  if (!model.name.empty()) out += " " + sanitize(model.name, 'M', 0);
  out += "\n";
  if (model.sense == ObjSense::Maximize) out += "OBJSENSE\n    MAX\n";
  out += "ROWS\n N  " + objname + "\n";

  std::vector<double> rhs(m, 0.0), range(m, 0.0);
  std::vector<char> has_range(m, 0);
  for (Index i = 0; i < m; ++i) {
    const double lo = model.row_lb[i], hi = model.row_ub[i];
    char t;
    if (lo == hi) {
      t = 'E', rhs[i] = lo;
    } else if (lo == -kInf && hi == kInf) {
      t = 'N';
    } else if (hi == kInf) {
      t = 'G', rhs[i] = lo;
    } else if (lo == -kInf) {
      t = 'L', rhs[i] = hi;
    } else {
      t = 'G', rhs[i] = lo, range[i] = hi - lo, has_range[i] = 1;
    }
    out += ' ';
    out += t;
    out += "  ";
    out += rn[i];
    out += '\n';
  }

  out += "COLUMNS\n";
  bool in_int = false;
  int marker = 0;
  for (Index j = 0; j < n; ++j) {
    const bool is_int = is_integral_type(model.col_type[j]);
    if (is_int != in_int) {
      out += "    MARKER" + std::to_string(marker++) + "  'MARKER'  " + (is_int ? "'INTORG'\n" : "'INTEND'\n");
      in_int = is_int;
    }
    const bool empty = model.A.begin(j) == model.A.end(j);
    if (model.obj[j] != 0.0 || empty) append_entry(out, cn[j], objname, model.obj[j]);
    for (Index p = model.A.begin(j); p < model.A.end(j); ++p) append_entry(out, cn[j], rn[model.A.index[p]], model.A.value[p]);
  }
  if (in_int) out += "    MARKER" + std::to_string(marker) + "  'MARKER'  'INTEND'\n";

  out += "RHS\n";
  if (model.obj_offset != 0.0) append_entry(out, "RHS", objname, -model.obj_offset);
  for (Index i = 0; i < m; ++i)
    if (rhs[i] != 0.0) append_entry(out, "RHS", rn[i], rhs[i]);

  bool any_range = false;
  for (Index i = 0; i < m; ++i) {
    if (!has_range[i]) continue;
    if (!any_range) out += "RANGES\n";
    any_range = true;
    append_entry(out, "RNG", rn[i], range[i]);
  }

  out += "BOUNDS\n";
  auto bound = [&](const char* type, Index j) {
    out += ' ';
    out += type;
    out += " BND  ";
    out += cn[j];
    out += '\n';
  };
  auto bound_val = [&](const char* type, Index j, double v) {
    out += ' ';
    out += type;
    out += " BND  ";
    out += cn[j];
    out += "  ";
    append_num(out, v);
    out += '\n';
  };
  for (Index j = 0; j < n; ++j) {
    const double lo = model.col_lb[j], hi = model.col_ub[j];
    const bool is_int = is_integral_type(model.col_type[j]);
    if (is_semi_type(model.col_type[j])) {
      // Semi-continuous: LO, then SC with the upper bound; semi-integer: SI inside the integer markers
      // (HiGHS reads SC on an integer column as semi-continuous; SCIP and our reader take SI too).
      if (lo == -kInf) bound("MI", j);
      else if (lo != 0.0) bound_val("LO", j, lo);
      bound_val(model.col_type[j] == VarType::SemiInteger ? "SI" : "SC", j, hi);
      continue;
    }
    if (model.col_type[j] == VarType::Binary && lo == 0.0 && hi == 1.0) {
      bound("BV", j);
    } else if (lo == hi) {
      bound_val("FX", j, lo);
    } else if (lo == -kInf && hi == kInf) {
      bound("FR", j);
    } else {
      // Lower bound first: an explicit LO stops a negative UP from implying lb = -inf.
      if (lo == -kInf) bound("MI", j);
      else if (lo != 0.0 || hi < 0.0) bound_val("LO", j, lo);
      if (hi != kInf) bound_val("UP", j, hi);
      else if (is_int) bound("PL", j);  // explicit for readers using the legacy [0,1] default
    }
  }
  if (!model.sos.empty()) {
    out += "SOS\n";
    for (size_t s = 0; s < model.sos.size(); ++s) {
      const SosConstraint& set = model.sos[s];
      out += set.type == 2 ? " S2 SOS  " : " S1 SOS  ";
      out += sanitize(set.name, 'S', s + 1);
      out += "  1\n";
      for (size_t k = 0; k < set.cols.size(); ++k) {
        out += "    ";
        out += cn[set.cols[k]];
        out += "  ";
        append_num(out, set.weights[k]);
        out += '\n';
      }
    }
  }
  if (!ind_rows.empty()) {
    out += "INDICATORS\n";
    for (const auto& [q, i] : ind_rows) {
      const IndicatorConstraint& c = model_in.indicators[q];
      out += " IF  " + rn[i] + "  " + cn[c.binary] + "  " + std::to_string(c.value) + "\n";
    }
  }
  out += "ENDATA\n";
  return out;
}

void write_mps(const Model& model, const std::string& path) {
  const std::string text = write_mps_string(model);
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error(path + ": cannot open file for writing");
  const bool wrote = std::fwrite(text.data(), 1, text.size(), f) == text.size();
  const bool closed = std::fclose(f) == 0;
  if (!wrote || !closed) throw std::runtime_error(path + ": write error");
}

}  // namespace milp
