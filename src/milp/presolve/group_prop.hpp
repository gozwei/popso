#pragma once
// Bound propagation that knows the model's one-of groups (presolve).
//
// Plain activity propagation treats the binaries of a row as independent. Where a set-partitioning row says that
// exactly one of them is 1 (a job's start options, an assignment), a row's activity over that group is the
// coefficient of the chosen member, not the sum of the positive ones: with  S = sum_t t x_t  and  sum_t x_t = 1
// the row gives S its window [min t, max t], and a bound on S removes the options outside it. Time-indexed
// scheduling models are written this way (start and completion rows over the job's options, precedence rows on
// the start and completion columns), and the plain rule sees none of it.
//
// Groups: rows  sum x = 1  (exact) and  sum x <= 1  (at most one) over binaries with coefficient 1; a binary
// belongs to one group (exact rows first, larger first; a row whose members are partly taken becomes an
// at-most-one group of the rest). A row's activity bounds take, per group, the smallest / largest coefficient
// over the group's free members, with 0 as a candidate when a free member is absent from the row or the group
// is at-most-one. From these:
//   - integer columns outside the groups get the usual bounds from the tighter activities;
//   - a member is fixed at 0 when choosing it violates the row; the members absent from a row are fixed at 0
//     when "none of the row's members" violates it.
// Shaving (optional): an integer column with a cost is tested at its cheapest value; if the propagation proves
// that infeasible the bound moves by one, repeatedly (30n20b8: both resource levels 0 -> 2, root bound 123 ->
// 302, the optimum).

#include <vector>

#include "milp/search/mip_problem.hpp"

namespace milp {

struct GroupPropOptions {
  long long work_limit = 0;   // nonzeros touched (0: 50 * nnz + 100000)
  bool shave = true;          // shaving of integer cost columns
  int max_shave_cols = 64;    // no shaving when more columns than this have a cost
  int max_shave_steps = 16;   // per column
};

struct GroupPropResult {
  bool infeasible = false;
  std::vector<double> lb, ub;  // the tightened bounds (all columns of the problem)
  long long work = 0;
  Index groups = 0;            // groups found (0: nothing was done)
  Index fixed = 0;             // group members fixed
  Index tightened = 0;         // bounds of other integer columns tightened
  Index shaved = 0;            // bound steps by shaving
  Index shave_probes = 0;
};

// The rows of p must be feasible for its bounds as far as the caller knows; quadratic rows, implications and
// products of p are not read (the caller passes the linear part: every reduction holds for a relaxation).
GroupPropResult group_propagate(const MipProblem& p, const GroupPropOptions& opt = {});

}  // namespace milp
