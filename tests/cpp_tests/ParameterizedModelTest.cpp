#include <gtest/gtest.h>

#include <Eigen/Eigen>
#include <boost/filesystem.hpp>

#include <CppADCodeGenEigenPy/ADModel.h>
#include <CppADCodeGenEigenPy/Model.h>
#include <CppADCodeGenEigenPy/Util.h>

#include "testing/models/ParameterizedTestModel.h"

namespace CppADCodeGenEigenPy {
namespace ParameterizedModelTest {

class ParameterizedTestModelFixture : public ::testing::Test {
   protected:
    using Vector = CompiledModel<Scalar>::Vector;
    using Matrix = CompiledModel<Scalar>::Matrix;

    static void SetUpTestSuite() {
        // Compile and load our model
        boost::filesystem::create_directories(DIRECTORY_PATH);
        ad_model_ptr_.reset(new ParameterizedTestModel<Scalar>());
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

std::unique_ptr<ADModel<Scalar>>
    ParameterizedTestModelFixture::ad_model_ptr_ = nullptr;
std::unique_ptr<CompiledModel<Scalar>>
    ParameterizedTestModelFixture::compiled_model_ptr_ = nullptr;

TEST_F(ParameterizedTestModelFixture, Evaluation) {
    Vector input = Vector::Ones(NUM_INPUT);
    Vector parameters = Vector::Ones(NUM_INPUT);

    Vector output_expected = evaluate<Scalar>(input, parameters);
    Vector output_actual = compiled_model_ptr_->evaluate(input, parameters);
    EXPECT_TRUE(output_actual.isApprox(output_expected))
        << "Function evaluation is incorrect.";

    // if I forget to pass the parameters, I should get a runtime error
    EXPECT_THROW(compiled_model_ptr_->evaluate(input), std::runtime_error)
        << "Missing parameters did not throw error.";
}

TEST_F(ParameterizedTestModelFixture, Jacobian) {
    Vector input = Vector::Ones(NUM_INPUT);
    Vector parameters = Vector::Ones(NUM_INPUT);

    // note Jacobian of scalar function is a row vector, hence the transpose
    Matrix P = Matrix::Zero(NUM_INPUT, NUM_INPUT);
    P.diagonal() << parameters;
    Matrix J_expected = input.transpose() * P;
    Matrix J_actual = compiled_model_ptr_->jacobian(input, parameters);
    EXPECT_TRUE(J_actual.isApprox(J_expected)) << "Jacobian is incorrect.";

    EXPECT_THROW(compiled_model_ptr_->jacobian(input), std::runtime_error)
        << "Missing parameters did not throw error.";
}

TEST_F(ParameterizedTestModelFixture, Hessian) {
    Vector input = Vector::Ones(NUM_INPUT);
    Vector parameters = Vector::Ones(NUM_INPUT);

    Matrix H_expected = Matrix::Zero(NUM_INPUT, NUM_INPUT);
    H_expected.diagonal() << parameters;
    Matrix H_actual = compiled_model_ptr_->hessian(input, parameters, 0);
    EXPECT_TRUE(H_actual.isApprox(H_expected)) << "Hessian is incorrect.";
    EXPECT_THROW(compiled_model_ptr_->hessian(input, 0), std::runtime_error)
        << "Missing parameters did not throw error.";
}

}  // namespace ParameterizedModelTest
}  // namespace CppADCodeGenEigenPy
