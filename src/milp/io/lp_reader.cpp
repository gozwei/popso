#include "milp/io/lp_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "milp/io/charconv_compat.hpp"

namespace milp {
namespace {

using std::string_view;

constexpr double kLpInf = 1e30;  // |value| >= this is treated as infinite
constexpr int kMaxPrintedWarnings = 10;

inline char lower(char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; }

bool iequals(string_view a, string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (lower(a[i]) != lower(b[i])) return false;
  return true;
}

bool iequals_any(string_view a, std::initializer_list<const char*> words) {
  for (const char* w : words)
    if (iequals(a, w)) return true;
  return false;
}

inline bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v' || c == '\0'; }
inline bool is_digit(char c) { return c >= '0' && c <= '9'; }

// First character of a name: a letter, one of the special characters allowed
// by the format, or a non-ASCII byte. Digits and '.' are allowed later on.
inline bool is_name_start(char c) {
  if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
  if (static_cast<unsigned char>(c) >= 0x80) return true;
  return std::strchr("!\"#$%&()/,;?@_`'{}|~", c) != nullptr && c != '\0';
}
inline bool is_name_char(char c) { return is_name_start(c) || is_digit(c) || c == '.'; }

double to_inf(double v) {
  if (v >= kLpInf) return kInf;
  if (v <= -kLpInf) return -kInf;
  return v;
}

enum class T : uint8_t { Name, Number, Le, Ge, Eq, Plus, Minus, Colon, Arrow, Eof };

struct Token {
  T type;
  bool line_start;  // first token on its line
  uint32_t line;
  string_view text;
  double value;  // for Number
};

enum class Sec { None, Objective, Constraints, Bounds, Generals, Binaries, Semi, Sos, End, Unsupported };

class LpParser {
 public:
  LpParser(string_view text, std::string source, std::vector<std::string>* warnings)
      : text_(text), source_(std::move(source)), warnings_(warnings) {}

  Model parse();

 private:
  [[noreturn]] void error_at(uint32_t line, const std::string& msg) const {
    throw LpError(source_ + ":" + std::to_string(line) + ": " + msg);
  }
  [[noreturn]] void error(const std::string& msg) const { error_at(cur().line, msg); }
  void warn(const std::string& msg) {
    std::string full = source_ + ": warning: " + msg;
    if (warnings_) {
      warnings_->push_back(std::move(full));
    } else if (num_warnings_ < kMaxPrintedWarnings) {
      std::fprintf(stderr, "%s\n", full.c_str());
    }
    ++num_warnings_;
  }

  void tokenize();
  const Token& cur() const { return toks_[pos_]; }
  const Token& at(size_t i) const { return toks_[std::min(i, toks_.size() - 1)]; }
  static std::string describe(const Token& t);

  // Section keyword at token i: returns the number of tokens it spans (0 if
  // none) and the section in `sec`.
  int keyword(size_t i, Sec& sec, std::string& what) const;
  bool at_keyword() const {
    Sec s;
    std::string w;
    return keyword(pos_, s, w) > 0;
  }
  bool is_label(size_t i) const { return at(i).type == T::Name && at(i + 1).type == T::Colon; }
  // Name token usable as a variable at position i (not a label, not a keyword).
  bool is_var_name(size_t i) const {
    if (at(i).type != T::Name || is_label(i)) return false;
    Sec s;
    std::string w;
    return keyword(i, s, w) == 0;
  }
  static bool is_op(T t) { return t == T::Le || t == T::Ge || t == T::Eq; }

  Index col(string_view name);
  void parse_objective();
  // Parses a linear expression; stops at an operator, a label, a keyword or
  // the end. Returns the number of terms; `nvars` counts variable terms.
  int parse_expr(double& constant, int& nvars);
  double parse_signed_number(const char* what);
  void parse_constraint();
  void parse_bound();
  void parse_sos(int default_type);
  void add_row(string_view name, uint32_t line, double lo, double hi);

  string_view text_;
  std::string source_;
  std::vector<std::string>* warnings_;
  int num_warnings_ = 0;

  std::vector<Token> toks_;
  size_t pos_ = 0;
  std::string name_;

  ObjSense sense_ = ObjSense::Minimize;
  double obj_offset_ = 0.0;

  std::unordered_map<string_view, Index> col_index_;
  std::vector<string_view> col_names_;
  std::vector<double> obj_, lb_, ub_;
  std::vector<VarType> type_;
  std::vector<char> lb_set_;
  std::vector<char> semi_;  // listed in the semi-continuous section

  // Indicator constraints; the one being parsed (ind_col_ >= 0) goes to indicators_ instead of the rows.
  Index ind_col_ = -1;
  int ind_value_ = 1;
  std::vector<IndicatorConstraint> indicators_;
  std::vector<SosConstraint> sos_;

  std::unordered_set<string_view> row_name_set_;
  std::vector<string_view> row_names_;
  std::vector<double> row_lb_, row_ub_;
  std::vector<Index> trip_col_, trip_row_;
  std::vector<double> trip_val_;

  // Current expression terms (column, coefficient), in order of appearance.
  std::vector<std::pair<Index, double>> terms_;
  std::vector<double> acc_;     // dense accumulator indexed by column
  std::vector<char> touched_;   // column already in the current row
};

std::string LpParser::describe(const Token& t) {
  switch (t.type) {
    case T::Name:
    case T::Number: return "'" + std::string(t.text) + "'";
    case T::Le: return "'<='";
    case T::Ge: return "'>='";
    case T::Eq: return "'='";
    case T::Plus: return "'+'";
    case T::Minus: return "'-'";
    case T::Colon: return "':'";
    case T::Arrow: return "'->'";
    case T::Eof: return "end of file";
  }
  return "?";
}

void LpParser::tokenize() {
  const char* p = text_.data();
  const char* const end = p + text_.size();
  uint32_t line = 1;
  bool line_start = true;
  toks_.reserve(text_.size() / 4 + 16);
  auto fail = [&](const std::string& msg) -> void { error_at(line, msg); };
  while (p < end) {
    const char c = *p;
    if (c == '\n') {
      ++line;
      line_start = true;
      ++p;
      continue;
    }
    if (is_space(c)) {
      ++p;
      continue;
    }
    if (c == '\\') {  // comment to end of line
      const char* nl = static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
      const char* le = nl ? nl : end;
      if (toks_.empty() && name_.empty()) {
        string_view body(p + 1, static_cast<size_t>(le - p - 1));
        while (!body.empty() && is_space(body.front())) body.remove_prefix(1);
        constexpr string_view kProb = "problem name:";
        if (body.size() >= kProb.size() && iequals(body.substr(0, kProb.size()), kProb)) {
          body.remove_prefix(kProb.size());
          while (!body.empty() && is_space(body.front())) body.remove_prefix(1);
          while (!body.empty() && is_space(body.back())) body.remove_suffix(1);
          name_ = std::string(body);
        }
      }
      p = le;
      continue;
    }
    Token t{T::Eof, line_start, line, {}, 0.0};
    line_start = false;
    const char* s = p;
    if (is_digit(c) || (c == '.' && p + 1 < end && is_digit(p[1]))) {
      while (p < end && is_digit(*p)) ++p;
      if (p < end && *p == '.') {
        ++p;
        while (p < end && is_digit(*p)) ++p;
      }
      if (p < end && (*p == 'e' || *p == 'E')) {
        const char* q = p + 1;
        if (q < end && (*q == '+' || *q == '-')) ++q;
        if (q < end && is_digit(*q)) {
          p = q;
          while (p < end && is_digit(*p)) ++p;
        }
      }
      t.type = T::Number;
      t.text = string_view(s, static_cast<size_t>(p - s));
      // Copy into a NUL-terminated buffer: the strtod fallback of
      // parse_double must not read past the token.
      char buf[128];
      if (t.text.size() >= sizeof(buf)) fail("number too long: '" + std::string(t.text.substr(0, 40)) + "...'");
      std::memcpy(buf, t.text.data(), t.text.size());
      buf[t.text.size()] = '\0';
      if (!io_detail::parse_double(string_view(buf, t.text.size()), t.value))
        fail("invalid number '" + std::string(t.text) + "'");
    } else if (is_name_start(c)) {
      while (p < end && is_name_char(*p)) ++p;
      t.text = string_view(s, static_cast<size_t>(p - s));
      if (iequals(t.text, "inf") || iequals(t.text, "infinity")) {
        t.type = T::Number;
        t.value = kInf;
      } else {
        t.type = T::Name;
      }
      if (p < end && *p == '[')
        fail("invalid character '[' in name '" + std::string(t.text) + "' (brackets are not allowed in LP names)");
      if (t.line_start && t.type == T::Name && iequals(t.text, "end")) {
        // End keyword (unless it is a label): ignore the rest of the file.
        const char* q = p;
        while (q < end && (is_space(*q) || *q == '\n')) ++q;
        if (q == end || *q != ':') {
          toks_.push_back(t);
          break;
        }
      }
    } else {
      ++p;
      const char nx = p < end ? *p : '\0';
      switch (c) {
        case '<':
          if (nx == '=') ++p;
          else if (nx == '-' && p + 1 < end && p[1] == '>') fail("equivalence constraints ('<->') are not supported");
          t.type = T::Le;
          break;
        case '>':
          if (nx == '=') ++p;
          t.type = T::Ge;
          break;
        case '=':
          if (nx == '<') ++p, t.type = T::Le;
          else if (nx == '>') ++p, t.type = T::Ge;
          else {
            if (nx == '=') ++p;
            t.type = T::Eq;
          }
          break;
        case '+': t.type = T::Plus; break;
        case '-':
          if (nx == '>') ++p, t.type = T::Arrow;
          else t.type = T::Minus;
          break;
        case ':': t.type = T::Colon; break;
        case '[':
        case ']':
        case '^':
          fail(std::string("quadratic terms ('") + c + "') are not supported yet");
          break;  // (fail throws)
        default: {
          char msg[64];
          if (static_cast<unsigned char>(c) >= 32 && c != 127) std::snprintf(msg, sizeof(msg), "unexpected character '%c'", c);
          else std::snprintf(msg, sizeof(msg), "unexpected character 0x%02x", static_cast<unsigned char>(c));
          fail(msg);
        }
      }
      t.text = string_view(s, static_cast<size_t>(p - s));
    }
    toks_.push_back(t);
  }
  toks_.push_back(Token{T::Eof, true, line, {}, 0.0});
}

int LpParser::keyword(size_t i, Sec& sec, std::string& what) const {
  const Token& t = at(i);
  if (t.type != T::Name || !t.line_start || at(i + 1).type == T::Colon) return 0;
  const string_view w = t.text;
  // Second word of a two-word keyword: must follow on the same line.
  auto next_is = [&](const char* word) {
    const Token& n = at(i + 1);
    return n.type == T::Name && !n.line_start && iequals(n.text, word) && at(i + 2).type != T::Colon;
  };
  if (iequals_any(w, {"minimize", "minimise", "minimum", "min", "maximize", "maximise", "maximum", "max"})) {
    sec = Sec::Objective;
    return 1;
  }
  if (iequals_any(w, {"st", "s.t.", "st.", "s.t"})) {
    sec = Sec::Constraints;
    return 1;
  }
  if (iequals(w, "subject") && next_is("to")) {
    sec = Sec::Constraints;
    return 2;
  }
  if (iequals(w, "such") && next_is("that")) {
    sec = Sec::Constraints;
    return 2;
  }
  if (iequals_any(w, {"bounds", "bound"})) {
    sec = Sec::Bounds;
    return 1;
  }
  if (iequals_any(w, {"general", "generals", "gen", "integer", "integers"})) {
    if (next_is("constraints") || next_is("constraint")) {
      sec = Sec::Unsupported;
      what = "general constraints";
      return 2;
    }
    sec = Sec::Generals;
    return 1;
  }
  if (iequals_any(w, {"binary", "binaries", "bin"})) {
    sec = Sec::Binaries;
    return 1;
  }
  if (iequals_any(w, {"semi", "semis", "semicontinuous"})) {
    sec = Sec::Semi;
    // "semi-continuous" is tokenised as semi - continuous.
    const Token& n1 = at(i + 1);
    const Token& n2 = at(i + 2);
    if (iequals(w, "semi") && n1.type == T::Minus && !n1.line_start && n2.type == T::Name && !n2.line_start &&
        iequals(n2.text, "continuous"))
      return 3;
    return 1;
  }
  if (iequals_any(w, {"sos", "sos1", "sos2"})) {
    sec = Sec::Sos;
    what = std::string(w);
    return 1;
  }
  if (iequals(w, "end")) {
    sec = Sec::End;
    return 1;
  }
  if (iequals(w, "lazy") && next_is("constraints")) {
    sec = Sec::Unsupported;
    what = "lazy constraints";
    return 2;
  }
  if (iequals(w, "user") && next_is("cuts")) {
    sec = Sec::Unsupported;
    what = "user cuts";
    return 2;
  }
  if (iequals_any(w, {"pwlobj", "pwl"})) {
    sec = Sec::Unsupported;
    what = "piecewise-linear objectives";
    return 1;
  }
  return 0;
}

Index LpParser::col(string_view name) {
  auto [it, inserted] = col_index_.try_emplace(name, static_cast<Index>(col_names_.size()));
  if (inserted) {
    col_names_.push_back(name);
    obj_.push_back(0.0);
    lb_.push_back(0.0);
    ub_.push_back(kInf);
    type_.push_back(VarType::Continuous);
    lb_set_.push_back(0);
    semi_.push_back(0);
  }
  return it->second;
}

int LpParser::parse_expr(double& constant, int& nvars) {
  int count = 0;
  nvars = 0;
  for (;;) {
    const Token& t = cur();
    if (t.type == T::Eof || is_op(t.type) || is_label(pos_) || at_keyword()) break;
    double sign = 1.0;
    bool had_sign = false;
    while (cur().type == T::Plus || cur().type == T::Minus) {
      if (cur().type == T::Minus) sign = -sign;
      had_sign = true;
      ++pos_;
    }
    if (count > 0 && !had_sign) error("expected '+' or '-' before " + describe(t));
    const Token& u = cur();
    if (u.type == T::Number) {
      const double v = sign * u.value;
      ++pos_;
      if (is_var_name(pos_)) {
        if (!std::isfinite(v)) error("infinite coefficient for variable '" + std::string(cur().text) + "'");
        terms_.emplace_back(col(cur().text), v);
        ++pos_;
        ++nvars;
      } else {
        constant += v;  // may be infinite; checked by the caller
      }
    } else if (is_var_name(pos_)) {
      terms_.emplace_back(col(u.text), sign);
      ++pos_;
      ++nvars;
    } else {
      error("expected a term after sign, found " + describe(u));
    }
    ++count;
  }
  return count;
}

void LpParser::parse_objective() {
  // Multi-objective files ("Minimize multi-objectives") are not supported.
  if (cur().type == T::Name && !cur().line_start && iequals(cur().text, "multi") && at(pos_ + 1).type == T::Minus)
    error("multiple objectives are not supported yet");
  if (is_label(pos_)) pos_ += 2;
  terms_.clear();
  double constant = 0.0;
  int nvars = 0;
  parse_expr(constant, nvars);
  for (const auto& [j, v] : terms_) obj_[j] += v;
  terms_.clear();
  if (!std::isfinite(constant)) error("infinite constant term in objective");
  obj_offset_ += constant;
  if (cur().type != T::Eof && !at_keyword()) {
    if (is_op(cur().type)) error("unexpected " + describe(cur()) + " in objective (missing 'Subject To'?)");
    error("unexpected " + describe(cur()) + " in objective");
  }
}

double LpParser::parse_signed_number(const char* what) {
  double sign = 1.0;
  if (cur().type == T::Plus || cur().type == T::Minus) {
    if (cur().type == T::Minus) sign = -1.0;
    ++pos_;
  }
  if (cur().type != T::Number) error(std::string("expected a number for the ") + what + ", found " + describe(cur()));
  const double v = sign * cur().value;
  ++pos_;
  return to_inf(v);
}

void LpParser::add_row(string_view name, uint32_t line, double lo, double hi) {
  if (lo == kInf || hi == -kInf)
    error_at(line, "invalid right-hand side in constraint" + (name.empty() ? std::string() : " '" + std::string(name) + "'"));
  if (!name.empty() && !row_name_set_.insert(name).second)
    error_at(line, "duplicate constraint name '" + std::string(name) + "'");
  IndicatorConstraint* ind = nullptr;
  Index r = -1;
  if (ind_col_ >= 0) {
    indicators_.emplace_back();
    ind = &indicators_.back();
    ind->name = name.empty() ? "IND" + std::to_string(indicators_.size()) : std::string(name);
    ind->binary = ind_col_;
    ind->value = ind_value_;
    ind->lb = lo;
    ind->ub = hi;
    ind_col_ = -1;
  } else {
    r = static_cast<Index>(row_lb_.size());
    row_names_.push_back(name);
    row_lb_.push_back(lo);
    row_ub_.push_back(hi);
  }
  // Sum duplicate terms; keep the order of first appearance.
  if (acc_.size() < col_names_.size()) {
    acc_.resize(col_names_.size(), 0.0);
    touched_.resize(col_names_.size(), 0);
  }
  std::vector<Index> order;
  order.reserve(terms_.size());
  for (const auto& [j, v] : terms_) {
    if (!touched_[j]) {
      touched_[j] = 1;
      order.push_back(j);
    }
    acc_[j] += v;
  }
  for (Index j : order) {
    if (acc_[j] != 0.0 && ind) {
      ind->cols.push_back(j);
      ind->coefs.push_back(acc_[j]);
    } else if (acc_[j] != 0.0) {
      trip_col_.push_back(j);
      trip_row_.push_back(r);
      trip_val_.push_back(acc_[j]);
    }
    acc_[j] = 0.0;
    touched_[j] = 0;
  }
  terms_.clear();
}

void LpParser::parse_constraint() {
  const uint32_t line = cur().line;
  string_view name;
  if (is_label(pos_)) {
    name = cur().text;
    pos_ += 2;
  }
  auto where = [&]() { return name.empty() ? std::string(" in constraint") : " in constraint '" + std::string(name) + "'"; };
  // Indicator constraint: "[name:] b = 0|1 -> expr op rhs".
  ind_col_ = -1;
  if (is_var_name(pos_) && at(pos_ + 1).type == T::Eq && at(pos_ + 3).type == T::Arrow) {
    if (at(pos_ + 2).type != T::Number || (at(pos_ + 2).value != 0.0 && at(pos_ + 2).value != 1.0))
      error("the indicator value must be 0 or 1" + where());
    ind_col_ = col(cur().text);
    ind_value_ = static_cast<int>(at(pos_ + 2).value);
    pos_ += 4;
  }
  terms_.clear();
  double c1 = 0.0;
  int nvars1 = 0;
  const int n1 = parse_expr(c1, nvars1);
  if (!is_op(cur().type)) {
    if (n1 == 0 && name.empty()) error("unexpected " + describe(cur()) + " in constraints section");
    error("expected '<=', '>=' or '='" + where() + ", found " + describe(cur()));
  }
  if (std::isnan(c1)) error("invalid constant term" + where());
  const T op1 = cur().type;
  ++pos_;

  // "expr op rhs": the right-hand side is a single (signed) constant. Also
  // used for a constant-only left side followed by a lone constant.
  const bool rhs_is_constant = [&] {
    size_t i = pos_;
    if (at(i).type == T::Plus || at(i).type == T::Minus) ++i;
    return at(i).type == T::Number && !is_var_name(i + 1);
  }();
  if (n1 == 0 && !rhs_is_constant) error("missing left-hand side" + where());
  if (nvars1 > 0 || rhs_is_constant) {
    if (!std::isfinite(c1)) error("infinite constant term" + where());
    const double rhs = parse_signed_number("right-hand side");
    if (is_op(cur().type))
      error("unexpected " + describe(cur()) + where() + " (ranged constraints are written 'lo <= expr <= hi')");
    const double b = std::isfinite(rhs) ? rhs - c1 : rhs;
    double lo = -kInf, hi = kInf;
    if (op1 == T::Le) hi = b;
    else if (op1 == T::Ge) lo = b;
    else lo = hi = b;
    if (op1 == T::Eq && !std::isfinite(b)) error_at(line, "infinite right-hand side in equality" + where());
    add_row(name, line, lo, hi);
    return;
  }

  // "c1 op1 expr [op2 c2]": constant on the left (ranged constraints).
  double k = 0.0;
  int nvars2 = 0;
  const int n2 = parse_expr(k, nvars2);
  if (n2 == 0 || nvars2 == 0) error("expected a linear expression" + where() + ", found " + describe(cur()));
  if (!std::isfinite(k)) error("infinite constant term" + where());
  const double left = std::isfinite(c1) ? c1 - k : c1;
  double lo = -kInf, hi = kInf;
  if (op1 == T::Le) lo = left;  // c1 <= expr
  else if (op1 == T::Ge) hi = left;
  else lo = hi = left;
  if (is_op(cur().type)) {
    const T op2 = cur().type;
    ++pos_;
    const double c2 = parse_signed_number("right-hand side");
    const double right = std::isfinite(c2) ? c2 - k : c2;
    if (op2 == T::Eq || op1 == T::Eq) error_at(line, "'=' is not allowed in a ranged constraint" + where());
    if (op1 != op2) error_at(line, "a ranged constraint needs two '<=' or two '>='" + where());
    if (op2 == T::Le) hi = right;
    else lo = right;
    if (is_op(cur().type)) error("unexpected " + describe(cur()) + where());
  }
  add_row(name, line, lo, hi);
}

void LpParser::parse_bound() {
  auto apply = [&](Index j, T op, double v, bool var_on_left) {
    // Normalise to "x op v".
    if (!var_on_left) op = op == T::Le ? T::Ge : (op == T::Ge ? T::Le : T::Eq);
    const std::string name(col_names_[j]);
    if (op == T::Le) {
      if (v == -kInf) error("upper bound -inf for variable '" + name + "'");
      ub_[j] = v;
    } else if (op == T::Ge) {
      if (v == kInf) error("lower bound +inf for variable '" + name + "'");
      lb_[j] = v;
      lb_set_[j] = 1;
    } else {
      if (!std::isfinite(v)) error("variable '" + name + "' fixed to an infinite value");
      lb_[j] = ub_[j] = v;
      lb_set_[j] = 1;
    }
  };
  const Token& t = cur();
  if (t.type == T::Number || t.type == T::Plus || t.type == T::Minus) {
    const double v1 = parse_signed_number("bound");
    if (!is_op(cur().type)) error("expected '<=', '>=' or '=' in bound, found " + describe(cur()));
    const T op1 = cur().type;
    ++pos_;
    if (!is_var_name(pos_)) error("expected a variable name in bound, found " + describe(cur()));
    const Index j = col(cur().text);
    ++pos_;
    apply(j, op1, v1, false);
    if (is_op(cur().type)) {
      const T op2 = cur().type;
      ++pos_;
      const double v2 = parse_signed_number("bound");
      if (op1 == T::Eq || op2 == T::Eq || op1 != op2) error("invalid double bound on variable '" + std::string(col_names_[j]) + "'");
      apply(j, op2, v2, true);
    }
    return;
  }
  if (!is_var_name(pos_)) error("expected a bound, found " + describe(t));
  const Index j = col(t.text);
  ++pos_;
  if (cur().type == T::Name && iequals(cur().text, "free") && !is_label(pos_)) {
    ++pos_;
    lb_[j] = -kInf;
    ub_[j] = kInf;
    lb_set_[j] = 1;
    return;
  }
  if (!is_op(cur().type)) error("expected '<=', '>=', '=' or 'free' after '" + std::string(t.text) + "' in bounds");
  const T op = cur().type;
  ++pos_;
  apply(j, op, parse_signed_number("bound"), true);
  if (is_op(cur().type)) error("unexpected " + describe(cur()) + " in bounds");
}

Model LpParser::parse() {
  tokenize();
  if (toks_.front().type == T::Name && toks_.size() > 2 && toks_[1].type == T::Colon &&
      iequals_any(toks_.front().text, {"max", "min", "maximize", "minimize", "maximise", "minimise"}))
    error_at(toks_.front().line, "this looks like an lp_solve-format file; only the CPLEX LP format is supported");

  Sec sec = Sec::None;
  int sos_type = 0;  // a "SOS1" / "SOS2" section header gives the type of entries without "S1::"
  bool seen_objective = false, seen_section = false;
  while (cur().type != T::Eof) {
    Sec s = Sec::None;
    std::string what;
    const int k = keyword(pos_, s, what);
    if (k > 0) {
      const Token kw = cur();
      pos_ += static_cast<size_t>(k);
      if (s == Sec::End) {
        sec = Sec::End;
        break;
      }
      if (s == Sec::Unsupported) error_at(kw.line, what + " are not supported yet");
      if (s == Sec::Sos) sos_type = iequals(what, "sos1") ? 1 : iequals(what, "sos2") ? 2 : 0;
      seen_section = true;
      if (s == Sec::Objective) {
        if (seen_objective) error_at(kw.line, "multiple objective sections are not supported");
        if (sec != Sec::None) error_at(kw.line, "the objective section must come first");
        seen_objective = true;
        sense_ = lower(kw.text[1]) == 'a' ? ObjSense::Maximize : ObjSense::Minimize;
        parse_objective();
      }
      sec = s;
      continue;
    }
    switch (sec) {
      case Sec::None:
        error("expected 'Minimize', 'Maximize' or 'Subject To' at the start of an LP file, found " + describe(cur()));
      case Sec::Objective:  // parse_objective() stops only at a keyword
        error("unexpected " + describe(cur()));
      case Sec::Constraints: parse_constraint(); break;
      case Sec::Bounds: parse_bound(); break;
      case Sec::Generals:
      case Sec::Binaries: {
        if (!is_var_name(pos_))
          error(std::string("expected a variable name in the ") + (sec == Sec::Generals ? "General" : "Binary") +
                " section, found " + describe(cur()));
        const Index j = col(cur().text);
        type_[j] = sec == Sec::Generals ? VarType::Integer : VarType::Binary;
        ++pos_;
        break;
      }
      case Sec::Semi: {
        if (!is_var_name(pos_)) error("expected a variable name in the semi-continuous section, found " + describe(cur()));
        semi_[col(cur().text)] = 1;
        ++pos_;
        break;
      }
      case Sec::Sos: parse_sos(sos_type); break;
      default: error("unexpected " + describe(cur()));
    }
  }
  if (!seen_section) error_at(cur().line, "no objective or constraint section found (not an LP file?)");

  // Indicator variables must be binary: undeclared (continuous) ones are made binary.
  for (const IndicatorConstraint& c : indicators_) {
    if (type_[c.binary] == VarType::Continuous && !semi_[c.binary]) type_[c.binary] = VarType::Binary;
  }
  const size_t n = col_names_.size(), nr = row_lb_.size();
  for (size_t j = 0; j < n; ++j) {
    if (semi_[j]) {
      // Semi-continuous; semi-integer if also General or Binary.
      type_[j] = type_[j] == VarType::Continuous ? VarType::SemiContinuous : VarType::SemiInteger;
      continue;
    }
    if (type_[j] == VarType::Binary) {
      lb_[j] = std::max(lb_[j], 0.0);
      ub_[j] = std::min(ub_[j], 1.0);
    } else if (ub_[j] < 0.0 && lb_[j] == 0.0 && !lb_set_[j]) {
      warn("negative upper bound on variable '" + std::string(col_names_[j]) +
           "' with default lower bound 0 (the lower bound stays 0)");
    }
  }
  if (!warnings_ && num_warnings_ > kMaxPrintedWarnings)
    std::fprintf(stderr, "%s: %d more warnings suppressed\n", source_.c_str(), num_warnings_ - kMaxPrintedWarnings);

  Model m;
  m.name = name_;
  m.sense = sense_;
  m.obj_offset = obj_offset_;
  m.col_names.reserve(n);
  for (string_view s : col_names_) m.col_names.emplace_back(s);
  m.obj = std::move(obj_);
  m.col_lb = std::move(lb_);
  m.col_ub = std::move(ub_);
  m.col_type = std::move(type_);
  m.row_names.reserve(nr);
  std::unordered_set<std::string> taken;
  for (string_view s : row_names_)
    if (!s.empty()) taken.emplace(s);
  for (size_t i = 0; i < nr; ++i) {
    if (!row_names_[i].empty()) {
      m.row_names.emplace_back(row_names_[i]);
      continue;
    }
    std::string base = "R" + std::to_string(i + 1), nm = base;
    for (int k = 1; taken.count(nm); ++k) nm = base + "_" + std::to_string(k);
    taken.insert(nm);
    m.row_names.push_back(std::move(nm));
  }
  m.row_lb = std::move(row_lb_);
  m.row_ub = std::move(row_ub_);
  m.A = SparseMatrix::from_triplets(static_cast<Index>(n), static_cast<Index>(nr), trip_col_, trip_row_, trip_val_);
  m.sos = std::move(sos_);
  m.indicators = std::move(indicators_);
  try {
    m.validate();
  } catch (const std::exception& e) {
    throw LpError(source_ + ": " + e.what());
  }
  return m;
}

// One SOS entry: "[name:] S1:: x1:w1 x2:w2 ..." (or "S2::"). In a "SOS1" / "SOS2" section the
// type may be omitted. Each member is "variable:weight".
void LpParser::parse_sos(int default_type) {
  SosConstraint s;
  auto type_spec = [&](size_t i) {
    const Token& t = at(i);
    return t.type == T::Name && t.text.size() == 2 && lower(t.text[0]) == 's' && (t.text[1] == '1' || t.text[1] == '2') &&
           at(i + 1).type == T::Colon && at(i + 2).type == T::Colon;
  };
  const uint32_t line = cur().line;
  if (!type_spec(pos_) && cur().type == T::Name && at(pos_ + 1).type == T::Colon && at(pos_ + 2).type != T::Number) {
    s.name = std::string(cur().text);
    pos_ += 2;
  }
  if (type_spec(pos_)) {
    s.type = cur().text[1] - '0';
    pos_ += 3;
  } else if (default_type > 0) {
    s.type = default_type;
  } else {
    error("expected 'S1::' or 'S2::' in SOS section, found " + describe(cur()));
  }
  if (s.name.empty()) s.name = "SOS" + std::to_string(sos_.size() + 1);
  // Members: name ':' number, until the next entry (a name followed by ':' and not a number) or keyword.
  while (cur().type == T::Name && at(pos_ + 1).type == T::Colon && at(pos_ + 2).type == T::Number &&
         !at_keyword() && !type_spec(pos_)) {
    s.cols.push_back(col(cur().text));
    s.weights.push_back(at(pos_ + 2).value);
    pos_ += 3;
  }
  if (s.cols.empty()) error_at(line, "SOS '" + s.name + "' has no members (expected 'variable:weight')");
  if (cur().type != T::Eof && !at_keyword() && !(cur().type == T::Name && at(pos_ + 1).type == T::Colon))
    error("unexpected " + describe(cur()) + " in SOS section");
  sos_.push_back(std::move(s));
}

std::string read_file(const std::string& path) {
  if (path.size() >= 3 && path.compare(path.size() - 3, 3, ".gz") == 0)
    throw LpError(path + ": gzipped LP files are not supported; decompress first");
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw LpError(path + ": cannot open file");
  std::string data;
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (!ec) data.reserve(static_cast<size_t>(size));
  char buf[1 << 16];
  size_t got;
  while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) data.append(buf, got);
  const bool bad = std::ferror(f) != 0;
  std::fclose(f);
  if (bad) throw LpError(path + ": read error");
  return data;
}

}  // namespace

Model read_lp_string(const std::string& text, std::vector<std::string>* warnings) {
  LpParser parser(text, "<string>", warnings);
  return parser.parse();
}

Model read_lp(const std::string& path, std::vector<std::string>* warnings) {
  const std::string data = read_file(path);
  LpParser parser(data, path, warnings);
  return parser.parse();
}

}  // namespace milp
