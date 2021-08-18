#pragma once

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADModel.h>
#include <CppADCodeGenEigenPy/Util.h>

#include "testing/Defs.h"

namespace CppADCodeGenEigenPy {
namespace BasicModelTest {

using Scalar = double;

const std::string MODEL_NAME = "BasicTestModel";
const std::string DIRECTORY_PATH = "/tmp/CppADCodeGenEigenPy";
const std::string LIB_GENERIC_PATH =
    get_library_generic_path(MODEL_NAME, DIRECTORY_PATH);
const std::string LIB_REAL_PATH =
    get_library_real_path(MODEL_NAME, DIRECTORY_PATH);

const int NUM_INPUT = 3;
const int NUM_OUTPUT = NUM_INPUT;

// Just multiply input vector by 2.
template <typename Scalar>
static Vector<Scalar> evaluate(const Vector<Scalar>& input) {
    return input * Scalar(2.);
}

template <typename Scalar>
struct BasicTestModel : public ADModel<Scalar> {
    using typename ADModel<Scalar>::ADScalar;
    using typename ADModel<Scalar>::ADVector;
    using typename ADModel<Scalar>::ADMatrix;

    // Generate the input to the function
    ADVector input() const override { return ADVector::Ones(NUM_INPUT); }

    // Evaluate the function
    ADVector function(const ADVector& input) const override {
        return evaluate<ADScalar>(input);
    }
};

}  // namespace BasicModelTest
}  // namespace CppADCodeGenEigenPy
