#pragma once
// Clique merging: replaces set-packing rows (sum of binaries <= 1) by fewer,
// larger clique rows covering them, e.g. the pairwise conflicts
// x_a + x_b <= 1 of a pigeonhole model become one row per hole. Only rows
// change, so no postsolve is needed.
//
// Clique extension: a conflict graph on the binaries (x_u = x_v = 1 infeasible) is collected from every row
// (for a side sum a_j x_j <= b: the binaries with the largest positive coefficients whose pairs exceed the
// slack b - minact form a clique). Each set-packing row is extended greedily by binaries adjacent to all of
// its members; packing rows contained in another (extended) one are dropped. A binary adjacent to every
// member of a set-partitioning row (sum = 1) is fixed to 0. Rows and bounds only, no postsolve.

#include <utility>
#include <vector>

#include "milp/search/mip_problem.hpp"

namespace milp {

// Returns the number of set-packing rows removed (dominated by new cliques).
Index merge_cliques(MipProblem& p, Index max_new_rows = 10000);

struct CliqueExtendStats {
  Index cliques = 0;        // cliques collected from the rows
  Index extended = 0;       // packing rows that gained columns
  long long added = 0;      // entries added to them
  Index dominated = 0;      // packing rows dropped (contained in another clique row)
  Index fixed = 0;          // binaries fixed to 0 (adjacent to a whole partitioning row)
  long long work = 0;
};

struct CliqueExtension {
  std::vector<Index> fixed;                                 // binaries to fix to 0
  std::vector<std::pair<Index, std::vector<Index>>> ext;   // packing row -> columns to add (coefficient 1)
  std::vector<Index> removed;                               // dominated packing rows
  CliqueExtendStats stats;
};

// Finds the reductions without applying them.
CliqueExtension find_clique_extensions(const MipProblem& p, long long work_limit);

// Applies find_clique_extensions to p. Returns false if nothing changed. work_limit: rough number of elementary operations.
bool extend_cliques(MipProblem& p, long long work_limit, CliqueExtendStats* stats);

}  // namespace milp
