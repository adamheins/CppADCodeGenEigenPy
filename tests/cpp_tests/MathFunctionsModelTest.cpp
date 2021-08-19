#include <gtest/gtest.h>

#include <Eigen/Eigen>
#include <boost/filesystem.hpp>

#include <CppADCodeGenEigenPy/ADModel.h>
#include <CppADCodeGenEigenPy/CompiledModel.h>
#include <CppADCodeGenEigenPy/Util.h>

#include "testing/models/MathFunctionsTestModel.h"

namespace CppADCodeGenEigenPy {
namespace MathFunctionsModelTest {

class MathFunctionsTestModelFixture : public ::testing::Test {
   protected:
    using Vector = CompiledModel<Scalar>::Vector;
    using Matrix = CompiledModel<Scalar>::Matrix;

    static void SetUpTestSuite() {
        // Compile and load our model
        boost::filesystem::create_directories(DIRECTORY_PATH);
        ad_model_ptr_.reset(new MathFunctionsTestModel<Scalar>());
        ad_model_ptr_->compile(MODEL_NAME, DIRECTORY_PATH,
                               DerivativeOrder::Second);
        compiled_model_ptr_.reset(
            new CompiledModel<Scalar>(MODEL_NAME, LIB_GENERIC_PATH));
    }

    static void TearDownTestSuite() {
        // Delete the compiled shared object.
        boost::filesystem::remove_all(DIRECTORY_PATH);
    }

    static std::unique_ptr<ADModel<Scalar>> ad_model_ptr_;
    static std::unique_ptr<CompiledModel<Scalar>> compiled_model_ptr_;
};

std::unique_ptr<ADModel<Scalar>> MathFunctionsTestModelFixture::ad_model_ptr_ =
    nullptr;
std::unique_ptr<CompiledModel<Scalar>>
    MathFunctionsTestModelFixture::compiled_model_ptr_ = nullptr;

TEST_F(MathFunctionsTestModelFixture, Evaluation) {
    Vector input = Vector::Ones(NUM_INPUT);

    Vector output_expected = evaluate<Scalar>(input);
    Vector output_actual = compiled_model_ptr_->evaluate(input);

    EXPECT_TRUE(output_actual.isApprox(output_expected))
        << "Function evaluation is incorrect.";
}

TEST_F(MathFunctionsTestModelFixture, Jacobian) {
    Vector input = Vector::Ones(NUM_INPUT);

    // clang-format off
    Matrix J_expected(NUM_OUTPUT, NUM_INPUT);
    J_expected << cos(input(0)) * cos(input(1)), -sin(input(0)) * sin(input(1)), 0,
                  0, 0, 0.5 / sqrt(input(2)),
                  2 * input.transpose();
    // clang-format on
    Matrix J_actual = compiled_model_ptr_->jacobian(input);

    EXPECT_TRUE(J_actual.isApprox(J_expected)) << "Jacobian is incorrect.";
}

TEST_F(MathFunctionsTestModelFixture, Hessian) {
    Vector input = Vector::Ones(NUM_INPUT);

    // clang-format off
    Matrix H0_expected(NUM_INPUT, NUM_INPUT);
    H0_expected <<
        -sin(input(0)) * cos(input(1)), -cos(input(0)) * sin(input(1)), 0,
        -cos(input(0)) * sin(input(1)), -sin(input(0)) * cos(input(1)), 0,
        0, 0, 0;
    // clang-format on

    Matrix H1_expected = Matrix::Zero(NUM_INPUT, NUM_INPUT);
    H1_expected(2, 2) = -0.25 * pow(input(2), -1.5);

    Matrix H2_expected = 2 * Matrix::Identity(NUM_INPUT, NUM_INPUT);

    Matrix H0_actual = compiled_model_ptr_->hessian(input, 0);
    Matrix H1_actual = compiled_model_ptr_->hessian(input, 1);
    Matrix H2_actual = compiled_model_ptr_->hessian(input, 2);

    EXPECT_TRUE(H0_actual.isApprox(H0_expected))
        << "Hessian for dim 0 is incorrect.";
    EXPECT_TRUE(H1_actual.isApprox(H1_expected))
        << "Hessian for dim 1 is incorrect.";
    EXPECT_TRUE(H2_actual.isApprox(H2_expected))
        << "Hessian for dim 2 is incorrect.";
}

}  // namespace MathFunctionsModelTest
}  // namespace CppADCodeGenEigenPy
