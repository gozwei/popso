#include "milp/core/sparse.hpp"

#include <algorithm>
#include <numeric>
#include <utility>

namespace milp {

SparseMatrix SparseMatrix::from_triplets(Index num_major, Index num_minor, const std::vector<Index>& major,
                                         const std::vector<Index>& minor, const std::vector<double>& val) {
  SparseMatrix m;
  m.num_major = num_major;
  m.num_minor = num_minor;
  const size_t n = major.size();
  std::vector<Index> count(num_major + 1, 0);
  for (size_t k = 0; k < n; ++k) ++count[major[k] + 1];
  std::partial_sum(count.begin(), count.end(), count.begin());
  std::vector<Index> idx(n);
  std::vector<double> v(n);
  std::vector<Index> pos(count.begin(), count.end() - 1);
  for (size_t k = 0; k < n; ++k) {
    Index p = pos[major[k]]++;
    idx[p] = minor[k];
    v[p] = val[k];
  }
  // Sort each major vector by minor index, merge duplicates, drop zeros.
  m.start.assign(num_major + 1, 0);
  m.index.reserve(n);
  m.value.reserve(n);
  std::vector<std::pair<Index, double>> buf;
  for (Index j = 0; j < num_major; ++j) {
    buf.clear();
    for (Index p = count[j]; p < count[j + 1]; ++p) buf.emplace_back(idx[p], v[p]);
    std::sort(buf.begin(), buf.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (size_t k = 0; k < buf.size();) {
      Index i = buf[k].first;
      double s = 0.0;
      while (k < buf.size() && buf[k].first == i) s += buf[k++].second;
      if (s != 0.0) {
        m.index.push_back(i);
        m.value.push_back(s);
      }
    }
    m.start[j + 1] = static_cast<Index>(m.index.size());
  }
  return m;
}

SparseMatrix SparseMatrix::transpose() const {
  SparseMatrix t;
  t.num_major = num_minor;
  t.num_minor = num_major;
  t.start.assign(num_minor + 1, 0);
  for (Index i : index) ++t.start[i + 1];
  std::partial_sum(t.start.begin(), t.start.end(), t.start.begin());
  t.index.resize(index.size());
  t.value.resize(value.size());
  std::vector<Index> pos(t.start.begin(), t.start.end() - 1);
  for (Index j = 0; j < num_major; ++j) {
    for (Index p = start[j]; p < start[j + 1]; ++p) {
      Index q = pos[index[p]]++;
      t.index[q] = j;
      t.value[q] = value[p];
    }
  }
  return t;
}

void SparseMatrix::append(const Index* idx, const double* val, Index len) {
  index.insert(index.end(), idx, idx + len);
  value.insert(value.end(), val, val + len);
  start.push_back(static_cast<Index>(index.size()));
  ++num_major;
}

}  // namespace milp
