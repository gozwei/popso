#include "milp/io/anonymize.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace milp {
namespace {

uint64_t splitmix64(uint64_t& state) {
  uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// new -> original. Fisher-Yates on our own generator: std::shuffle and the standard distributions
// differ between standard libraries, and a seed must give the same file everywhere.
std::vector<Index> permutation(size_t n, bool shuffle, uint64_t& state) {
  std::vector<Index> p(n);
  std::iota(p.begin(), p.end(), Index{0});
  if (!shuffle) return p;
  for (size_t i = n; i > 1; --i) {
    const size_t j = static_cast<size_t>(splitmix64(state) % i);  // modulo bias is irrelevant here
    std::swap(p[i - 1], p[j]);
  }
  return p;
}

std::vector<Index> inverse(const std::vector<Index>& p) {
  std::vector<Index> inv(p.size());
  for (size_t k = 0; k < p.size(); ++k) inv[static_cast<size_t>(p[k])] = static_cast<Index>(k);
  return inv;
}

std::string map_name(const std::string& name, char prefix, size_t idx) {
  if (name.empty()) return prefix + std::to_string(idx);
  std::string s = name;
  for (char& c : s)
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v') c = '_';
  return s;
}

}  // namespace

Model anonymize(const Model& model, const AnonOptions& options, AnonMap* map_out) {
  model.validate();
  const Index n = model.num_cols(), m = model.num_rows();
  uint64_t state = options.seed;
  const bool shuffle = !options.keep_order;
  AnonMap mp;
  mp.col = permutation(static_cast<size_t>(n), shuffle, state);
  mp.row = permutation(static_cast<size_t>(m), shuffle, state);
  mp.sos = permutation(model.sos.size(), shuffle, state);
  mp.indicator = permutation(model.indicators.size(), shuffle, state);
  const std::vector<Index> new_col = inverse(mp.col), new_row = inverse(mp.row);

  Model a;
  a.name = "anon";
  a.sense = model.sense;
  a.obj_offset = model.obj_offset;
  a.col_names.resize(static_cast<size_t>(n));
  a.obj.resize(static_cast<size_t>(n));
  a.col_lb.resize(static_cast<size_t>(n));
  a.col_ub.resize(static_cast<size_t>(n));
  a.col_type.resize(static_cast<size_t>(n));
  for (Index j = 0; j < n; ++j) {
    const Index o = mp.col[j];
    a.col_names[j] = "x" + std::to_string(j + 1);
    a.obj[j] = model.obj[o];
    a.col_lb[j] = model.col_lb[o];
    a.col_ub[j] = model.col_ub[o];
    a.col_type[j] = model.col_type[o];
  }
  a.row_names.resize(static_cast<size_t>(m));
  a.row_lb.resize(static_cast<size_t>(m));
  a.row_ub.resize(static_cast<size_t>(m));
  for (Index i = 0; i < m; ++i) {
    const Index o = mp.row[i];
    a.row_names[i] = "c" + std::to_string(i + 1);
    a.row_lb[i] = model.row_lb[o];
    a.row_ub[i] = model.row_ub[o];
  }

  // Matrix: column by column, the entries moved as they are (no summing or zero dropping) and
  // sorted by their new rows.
  a.A.num_major = n;
  a.A.num_minor = m;
  a.A.start.assign(1, 0);
  a.A.index.reserve(model.A.index.size());
  a.A.value.reserve(model.A.value.size());
  std::vector<std::pair<Index, double>> entries;
  for (Index j = 0; j < n; ++j) {
    const Index o = mp.col[j];
    entries.clear();
    for (Index p = model.A.begin(o); p < model.A.end(o); ++p)
      entries.emplace_back(new_row[model.A.index[p]], model.A.value[p]);
    std::stable_sort(entries.begin(), entries.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    for (const auto& [i, v] : entries) {
      a.A.index.push_back(i);
      a.A.value.push_back(v);
    }
    a.A.start.push_back(static_cast<Index>(a.A.index.size()));
  }

  // Further objectives: the same order (priorities, weights and tolerances kept), columns renumbered and
  // sorted, neutral names.
  a.obj_priority = model.obj_priority;
  a.obj_weight = model.obj_weight;
  a.obj_abs_tol = model.obj_abs_tol;
  a.obj_rel_tol = model.obj_rel_tol;
  if (!model.objectives.empty()) a.obj_name = "obj0";
  for (size_t k = 0; k < model.objectives.size(); ++k) {
    Objective o = model.objectives[k];
    o.name = "obj" + std::to_string(k + 1);
    std::vector<std::pair<Index, double>> e;
    for (size_t t = 0; t < o.cols.size(); ++t) e.emplace_back(new_col[o.cols[t]], o.coefs[t]);
    std::stable_sort(e.begin(), e.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    for (size_t t = 0; t < e.size(); ++t) o.cols[t] = e[t].first, o.coefs[t] = e[t].second;
    a.objectives.push_back(std::move(o));
  }

  // Quadratic objective: (row, col) with row >= col in the new indices, sorted by (col, row);
  // the stable sort keeps the order of repeated entries (they are summed in that order).
  {
    const size_t nq = model.q_val.size();
    std::vector<size_t> ord(nq);
    std::vector<Index> qr(nq), qc(nq);
    for (size_t k = 0; k < nq; ++k) {
      const Index r = new_col[model.q_row[k]], c = new_col[model.q_col[k]];
      qr[k] = std::max(r, c);
      qc[k] = std::min(r, c);
      ord[k] = k;
    }
    std::stable_sort(ord.begin(), ord.end(),
                     [&](size_t x, size_t y) { return qc[x] != qc[y] ? qc[x] < qc[y] : qr[x] < qr[y]; });
    a.q_row.reserve(nq);
    a.q_col.reserve(nq);
    a.q_val.reserve(nq);
    for (size_t k : ord) {
      a.q_row.push_back(qr[k]);
      a.q_col.push_back(qc[k]);
      a.q_val.push_back(model.q_val[k]);
    }
  }

  // Quadratic constraint terms: new row and columns (i >= j), sorted by (row, i, j); the stable sort keeps repeated
  // entries in their order.
  {
    const size_t nq = model.qr_val.size();
    std::vector<size_t> ord(nq);
    std::vector<Index> rr(nq), ri(nq), rj(nq);
    for (size_t k = 0; k < nq; ++k) {
      const Index i = new_col[model.qr_i[k]], j = new_col[model.qr_j[k]];
      rr[k] = new_row[model.qr_row[k]];
      ri[k] = std::max(i, j);
      rj[k] = std::min(i, j);
      ord[k] = k;
    }
    std::stable_sort(ord.begin(), ord.end(), [&](size_t x, size_t y) {
      return rr[x] != rr[y] ? rr[x] < rr[y] : ri[x] != ri[y] ? ri[x] < ri[y] : rj[x] < rj[y];
    });
    for (size_t k : ord) a.add_qrow(rr[k], ri[k], rj[k], model.qr_val[k]);
  }

  // SOS sets: members in their order (with the weights, it is the set's order), renamed.
  a.sos.resize(model.sos.size());
  for (size_t s = 0; s < model.sos.size(); ++s) {
    const SosConstraint& o = model.sos[static_cast<size_t>(mp.sos[s])];
    SosConstraint& t = a.sos[s];
    t.name = "s" + std::to_string(s + 1);
    t.type = o.type;
    t.weights = o.weights;
    t.cols.reserve(o.cols.size());
    for (Index j : o.cols) t.cols.push_back(new_col[j]);
  }

  // Indicator constraints: entries sorted by the new columns.
  a.indicators.resize(model.indicators.size());
  for (size_t q = 0; q < model.indicators.size(); ++q) {
    const IndicatorConstraint& o = model.indicators[static_cast<size_t>(mp.indicator[q])];
    IndicatorConstraint& t = a.indicators[q];
    t.name = "i" + std::to_string(q + 1);
    t.binary = new_col[o.binary];
    t.value = o.value;
    t.lb = o.lb;
    t.ub = o.ub;
    entries.clear();
    for (size_t k = 0; k < o.cols.size(); ++k) entries.emplace_back(new_col[o.cols[k]], o.coefs[k]);
    std::stable_sort(entries.begin(), entries.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    for (const auto& [j, v] : entries) {
      t.cols.push_back(j);
      t.coefs.push_back(v);
    }
  }
  if (map_out) *map_out = std::move(mp);
  return a;
}

std::string anon_map_string(const Model& original, const AnonMap& map) {
  std::string out = "# popso anonymization map: <new name> <original name>\n";
  auto line = [&out](const std::string& a, const std::string& b) {
    out += a;
    out += ' ';
    out += b;
    out += '\n';
  };
  for (size_t k = 0; k < map.col.size(); ++k) {
    const size_t o = static_cast<size_t>(map.col[k]);
    line("x" + std::to_string(k + 1), map_name(o < original.col_names.size() ? original.col_names[o] : "", 'C', o));
  }
  for (size_t k = 0; k < map.row.size(); ++k) {
    const size_t o = static_cast<size_t>(map.row[k]);
    line("c" + std::to_string(k + 1), map_name(o < original.row_names.size() ? original.row_names[o] : "", 'R', o));
  }
  for (size_t k = 0; k < map.sos.size(); ++k) {
    const size_t o = static_cast<size_t>(map.sos[k]);
    line("s" + std::to_string(k + 1), map_name(original.sos[o].name, 'S', o));
  }
  for (size_t k = 0; k < map.indicator.size(); ++k) {
    const size_t o = static_cast<size_t>(map.indicator[k]);
    line("i" + std::to_string(k + 1), map_name(original.indicators[o].name, 'I', o));
  }
  return out;
}

void write_anon_map(const std::string& path, const Model& original, const AnonMap& map) {
  const std::string text = anon_map_string(original, map);
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error(path + ": cannot open file for writing");
  const bool wrote = std::fwrite(text.data(), 1, text.size(), f) == text.size();
  const bool closed = std::fclose(f) == 0;
  if (!wrote || !closed) throw std::runtime_error(path + ": write error");
}

}  // namespace milp
