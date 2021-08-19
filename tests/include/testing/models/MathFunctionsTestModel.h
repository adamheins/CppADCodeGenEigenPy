#pragma once

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADModel.h>
#include <CppADCodeGenEigenPy/Util.h>

#include "testing/Defs.h"

namespace CppADCodeGenEigenPy {
namespace MathFunctionsModelTest {

using Scalar = double;

static const std::string MODEL_NAME = "MathFunctionsTestModel";
const std::string DIRECTORY_PATH = "/tmp/CppADCodeGenEigenPy";
const std::string LIB_GENERIC_PATH =
    get_library_generic_path(MODEL_NAME, DIRECTORY_PATH);
const std::string LIB_REAL_PATH =
    get_library_real_path(MODEL_NAME, DIRECTORY_PATH);

static const int NUM_INPUT = 3;
static const int NUM_OUTPUT = NUM_INPUT;

// Use various math routines: trigonometry, square root, vector operations
template <typename Scalar>
static Vector<Scalar> evaluate(const Vector<Scalar>& input) {
    Vector<Scalar> output(NUM_OUTPUT);
    output << sin(input(0)) * cos(input(1)), sqrt(input(2)),
        input.transpose() * input;
    return output;
}

template <typename Scalar>
struct MathFunctionsTestModel : public ADModel<Scalar> {
    using typename ADModel<Scalar>::ADScalar;
    using typename ADModel<Scalar>::ADVector;

    // Generate the input to the function
    ADVector input() const override { return ADVector::Ones(NUM_INPUT); }

    // Evaluate the function
    ADVector function(const ADVector& input) const override {
        return evaluate<ADScalar>(input);
    }
};

}  // namespace MathFunctionsModelTest
}  // namespace CppADCodeGenEigenPy
