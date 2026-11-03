#pragma once
// Fundamental types and constants shared by every component.

#include <cmath>
#include <cstdint>
#include <limits>

namespace milp {

using Index = int32_t;  // row/column/nonzero index type

inline constexpr double kInf = std::numeric_limits<double>::infinity();

// SemiContinuous / SemiInteger: x = 0 or lb <= x <= ub (continuous / integer
// on [lb, ub]); see Model.
enum class VarType : uint8_t { Continuous, Integer, Binary, SemiContinuous, SemiInteger };

enum class ObjSense : int8_t { Minimize = 1, Maximize = -1 };

inline bool is_integral_type(VarType t) { return t == VarType::Integer || t == VarType::Binary || t == VarType::SemiInteger; }
inline bool is_semi_type(VarType t) { return t == VarType::SemiContinuous || t == VarType::SemiInteger; }

inline bool is_finite(double v) { return std::isfinite(v); }

// Numerical tolerances. Kept in one place so every component agrees.
struct Tolerances {
  double primal_feas = 1e-6;   // bound / row activity violation
  double dual_feas = 1e-7;     // reduced cost sign violation
  double integrality = 1e-6;   // |x - round(x)|
  double pivot = 1e-7;         // smallest acceptable simplex pivot
  double zero = 1e-12;         // values below are treated as exact zero
};

}  // namespace milp
