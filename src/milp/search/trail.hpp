#pragma once
// Bound trail of a node for conflict analysis of propagation failures: every
// bound change made at the node (branching decisions first, then propagated
// bounds) with its reason (a model row or a conflict). When propagation proves
// the node infeasible, the failure is recorded too, so the implication graph
// can be walked back to the decisions (see analyze_propagation_conflict).

#include <cstdint>
#include <memory>
#include <vector>

#include "milp/core/types.hpp"

namespace milp {

struct Conflict;

struct BoundTrail {
  enum Kind : uint8_t { kDecision, kRow, kConflict };
  enum Fail : uint8_t { kNone, kFailRow, kFailConflict, kFailCrossing };
  struct Entry {
    Index col;
    uint8_t upper;  // 1: upper bound x <= val, 0: lower bound x >= val
    uint8_t kind;
    Index row;               // kRow: the row that implied it
    double val;
    const Conflict* conf;    // kConflict: the conflict that implied it
    int prev;                // previous entry of the same bound (-1: the global bound)
    int level;               // decision level: of a decision its path position, of an implied
                             // bound the highest level among the bounds of its reason
  };

  void init(Index num_cols, size_t capacity) {
    last.assign(2 * static_cast<size_t>(num_cols), -1);
    cap = capacity;
    e.clear();
    e.reserve(64);
    held.clear();
  }
  void clear() {
    for (const Entry& x : e) last[2 * static_cast<size_t>(x.col) + x.upper] = -1;
    e.clear();
    overflow = false;
    level = 0;
    fail = kNone;
    fail_conf = nullptr;
    held.clear();
  }
  // Records a bound; lvl < 0: the current level. Levels never decrease along
  // the entries of one bound (so the reasons of an entry have lower keys).
  void push(Index col, bool up, double val, Kind kind, Index row = -1, const Conflict* c = nullptr, int lvl = -1) {
    if (e.size() >= cap) {
      overflow = true;  // incomplete trail: no analysis for this node
      return;
    }
    int& l = last[2 * static_cast<size_t>(col) + (up ? 1 : 0)];
    if (lvl < 0 || chrono) lvl = level;
    if (l >= 0 && e[l].level > lvl) lvl = e[l].level;
    e.push_back({col, static_cast<uint8_t>(up ? 1 : 0), static_cast<uint8_t>(kind), row, val, c, l, lvl});
    l = static_cast<int>(e.size()) - 1;
  }
  // Decision level of the current bound (0: global).
  int bound_level(Index col, bool up) const {
    const int k = last[2 * static_cast<size_t>(col) + (up ? 1 : 0)];
    return k >= 0 ? e[k].level : 0;
  }
  void fail_row_at(Index row, bool upper_side) { fail = kFailRow, fail_row = row, fail_upper = upper_side; }
  void fail_conflict(const Conflict* c) { fail = kFailConflict, fail_conf = c; }
  void fail_crossing(Index col) { fail = kFailCrossing, fail_col = col; }
  // Keeps a conflict alive while entries (or the failure) point to it: the worker's bounded conflict pool may drop
  // it before the analysis (neos-2657525-crna: a conflict added by the generalized-resolution analysis shrank the
  // pool, and the propagation analysis that follows read a freed conflict; SIGSEGV in 1 of 24 runs, 4 threads).
  void hold(std::shared_ptr<const Conflict> c) { held.push_back(std::move(c)); }
  bool usable() const { return !overflow && fail != kNone; }

  std::vector<Entry> e;
  std::vector<int> last;  // per bound (2 * col + upper): latest entry or -1
  size_t cap = 0;
  bool overflow = false;
  int level = 0;  // level of new decisions (and the highest level in use)
  // Every entry gets the level current when it is pushed (a search that backtracks on this trail: the entries of
  // a level are then a contiguous block). Default: an implied bound gets the highest level among its reasons.
  bool chrono = false;
  // Why propagation failed: a row side (fail_upper: row_ub side violated),
  // a conflict, or a column whose propagated bounds crossed.
  Fail fail = kNone;
  Index fail_row = -1, fail_col = -1;
  bool fail_upper = false;
  const Conflict* fail_conf = nullptr;
  std::vector<std::shared_ptr<const Conflict>> held;  // conflicts the entries and fail_conf refer to (hold())
};

}  // namespace milp
