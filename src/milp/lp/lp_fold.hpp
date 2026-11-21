#pragma once
// LP folding: dimension reduction of an LP by an equitable partition of its rows and columns (Grohe, Kersting,
// Mladenov, Selman, "Dimension reduction via colour refinement", ESA 2014).
//
// Colour refinement on the bipartite graph of the LP (columns coloured by cost and bounds, rows by their sides, edges
// by coefficient) gives the coarsest partition in which every row of a row class has the same coefficient sum on
// every column class, and every column of a column class the same sum on every row class. Averaging a solution
// over the classes then keeps it feasible with the same objective, so the LP has an optimal solution that is
// constant on the column classes: the folded LP has one column z_C per column class (cost |C| c, the class's
// bounds) and one row per row class (a representative row with its coefficients summed per column class).
// From an optimal (z, w) of the folded LP: x_j = z_C(j), and y_i = w_R(i) / |R| are optimal row duals of the
// original (its reduced costs are the folded ones divided by |C|). The point is in general not a vertex: a
// crossover turns it into a basis.

#include <vector>

#include "milp/core/sparse.hpp"
#include "milp/core/types.hpp"

namespace milp {

struct LpFold {
  bool ok = false;  // false: no (sufficient) reduction, or the partition failed its exact check
  Index num_col_classes = 0, num_row_classes = 0;
  std::vector<Index> col_class, row_class;  // class of each original column / row
  std::vector<Index> col_size, row_size;    // class sizes
  SparseMatrix A;                           // folded matrix, CSC: columns = column classes, rows = row classes
  std::vector<double> cost, col_lb, col_ub, row_lb, row_ub;
  int rounds = 0;  // splitter classes processed
};

// A: CSC (columns x rows). The fold is used only if it keeps at most max_share of the rows + columns.
LpFold fold_lp(const SparseMatrix& A, const std::vector<double>& cost, const std::vector<double>& col_lb,
               const std::vector<double>& col_ub, const std::vector<double>& row_lb, const std::vector<double>& row_ub,
               double max_share = 0.67);

}  // namespace milp
