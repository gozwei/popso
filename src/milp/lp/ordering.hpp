#pragma once
// Fill-reducing orderings for sparse Cholesky factorization.
//
// amd_order computes an approximate minimum degree ordering (Amestoy, Davis and Duff, "An
// approximate minimum degree ordering algorithm", SIAM J. Matrix Anal. Appl. 17, 1996) of a
// symmetric sparsity pattern. It works on the quotient graph (variables and elements) with
// supervariables (indistinguishable nodes found by hashing), approximate external degrees,
// element absorption and aggressive absorption. Nodes of very high degree ("dense" rows) are
// ordered last.

#include <atomic>
#include <vector>

#include "milp/core/types.hpp"

namespace milp {

// Symmetric pattern of an n x n matrix in compressed form: the neighbours of node j are
// index[start[j] .. start[j+1]). Either triangle or both may be given (each edge is used in both
// directions); diagonal entries and duplicates are ignored. Returns perm with perm[k] = the node
// eliminated k-th. stop: when set during the run, an empty vector is returned.
std::vector<Index> amd_order(Index n, const std::vector<Index>& start, const std::vector<Index>& index,
                             const std::atomic<bool>* stop = nullptr);

}  // namespace milp
