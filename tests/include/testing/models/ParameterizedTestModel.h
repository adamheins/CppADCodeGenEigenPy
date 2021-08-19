#pragma once

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADModel.h>
#include <CppADCodeGenEigenPy/Util.h>

#include "testing/Defs.h"

namespace CppADCodeGenEigenPy {
namespace ParameterizedModelTest {

using Scalar = double;

const std::string MODEL_NAME = "ParameterizedTestModel";
const std::string DIRECTORY_PATH = "/tmp/CppADCodeGenEigenPy";
const std::string LIB_GENERIC_PATH =
    get_library_generic_path(MODEL_NAME, DIRECTORY_PATH);
const std::string LIB_REAL_PATH =
    get_library_real_path(MODEL_NAME, DIRECTORY_PATH);

const int NUM_INPUT = 3;
const int NUM_OUTPUT = 1;
const int NUM_PARAM = NUM_INPUT;

// Parameterized sum of squares.
template <typename Scalar>
static Vector<Scalar> evaluate(const Vector<Scalar>& input,
                               const Vector<Scalar>& parameters) {
    Vector<Scalar> output(NUM_OUTPUT);
    for (int i = 0; i < NUM_INPUT; ++i) {
        output(0) += 0.5 * parameters(i) * input(i) * input(i);
    }
    return output;
}

// Parameterized weighted sum model.
template <typename Scalar>
struct ParameterizedTestModel : public ADModel<Scalar> {
    using typename ADModel<Scalar>::ADScalar;
    using typename ADModel<Scalar>::ADVector;

    // Generate the input to the function
    ADVector input() const override { return ADVector::Ones(NUM_INPUT); }

    ADVector parameters() const override { return ADVector::Ones(NUM_PARAM); }

    // Evaluate the function
    ADVector function(const ADVector& input,
                      const ADVector& parameters) const override {
        return evaluate<ADScalar>(input, parameters);
    }
};

}  // namespace ParameterizedModelTest
}  // namespace CppADCodeGenEigenPy
