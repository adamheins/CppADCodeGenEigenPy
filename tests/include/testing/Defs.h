#pragma once

#include <Eigen/Eigen>

namespace CppADCodeGenEigenPy {

template <typename Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

} // namespace CppADCodeGenEigenPy
