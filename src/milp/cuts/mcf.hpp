#pragma once
// Multi-commodity flow (network) structure of a MIP, for the cut-set separator (cuts/mir.cpp, "mcf"):
// flow conservation rows (purely continuous rows with unit coefficients, one per commodity and node),
// arc flow columns (in at most two flow rows of one commodity, with opposite orientation) and arc
// capacity rows (flow columns of the arc's commodities against integer design columns,
// sum_k f_ak <= sum_m c_m y_am; undirected edges have both directions of each commodity in the row).
// Nodes: flow rows of different commodities with the same incident capacity rows, and the rows at the
// tail (head) of a directed capacitated arc. As SCIP's mcf separator (Achterberg & Raack 2010),
// simplified (no inconsistency-tolerant node identification, no commodity-wise arc capacities).
// Detection errors never make cuts invalid: any aggregation of rows is a valid base for the c-MIR.

#include <vector>

#include "milp/core/types.hpp"
#include "milp/search/mip_problem.hpp"

namespace milp {

struct McfNetwork {
  struct FlowRow {
    Index row, node, comm;
    double mult;  // mult * (a_row x) = outflow - inflow of the commodity at the node (unit coefficients)
  };
  struct Arc {
    Index tail = -1, head = -1;  // nodes; -1: outside the network (a source or sink column)
    Index cap_row = -1;          // capacity row (-1: uncapacitated)
  };
  Index num_nodes = 0, num_comms = 0;
  std::vector<FlowRow> flow_rows;  // sorted by node
  std::vector<Index> node_start;   // flow rows of node v: [node_start[v], node_start[v + 1])
  std::vector<Arc> arcs;
  std::vector<Index> col_arc;      // per column: its arc (-1: not a flow column)
  std::vector<Index> col_comm;     // per column: its commodity (-1: not a flow column)
  std::vector<double> col_cap;     // per column: its coefficient in the arc capacity row (0: none)
  bool empty() const { return arcs.empty() || num_nodes < 2; }
};

// Detects the network; returns an empty network if there is none (fewer than 2 nodes or no
// capacitated arc with an integer design column).
McfNetwork detect_mcf_network(const MipProblem& p);

// Test hook for MILP_MCF_LONG (-1: the environment decides; 0 / 1: off / on). With it on, a column in more
// than two candidate equality flow rows drops only the longest of them instead of all (see mcf.cpp).
extern int g_mcf_drop_long;

}  // namespace milp
