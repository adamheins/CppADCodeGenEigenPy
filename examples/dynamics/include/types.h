#pragma once

#include <Eigen/Eigen>

const size_t STATE_DIM = 7 + 6;
const size_t INPUT_DIM = 6;

template <typename Scalar>
using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

template <typename Scalar>
using Vec6 = Eigen::Matrix<Scalar, 6, 1>;

template <typename Scalar>
using StateVec = Eigen::Matrix<Scalar, STATE_DIM, 1>;

template <typename Scalar>
using StateErrorVec = Eigen::Matrix<Scalar, 2 * INPUT_DIM, 1>;

template <typename Scalar>
using InputVec = Eigen::Matrix<Scalar, INPUT_DIM, 1>;

// Note that we need to specify row-major here to match the
// expected-convention for numpy, when passing in inertia parameters.
template <typename Scalar>
using Mat3 = Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>;
