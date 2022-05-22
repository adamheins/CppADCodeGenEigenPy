#include <gtest/gtest.h>

#include <Eigen/Eigen>
#include <boost/filesystem.hpp>

#include <CppADCodeGenEigenPy/ADModel.h>
#include <CppADCodeGenEigenPy/CompiledModel.h>

#include "testing/models/BasicTestModel.h"

namespace CppADCodeGenEigenPy {
namespace LowOrderModelTest {

using namespace BasicModelTest;

class LowOrderTestModelFixture : public ::testing::Test {
   protected:
    using Vector = CompiledModel<Scalar>::Vector;
    using Matrix = CompiledModel<Scalar>::Matrix;

    static void SetUpTestSuite() {
        // Here we just reuse the basic model, but only compile with
        // zero-order.
        boost::filesystem::create_directories(DIRECTORY_PATH);
        ad_model_ptr_.reset(new BasicTestModel<Scalar>());
        ad_model_ptr_->compile(MODEL_NAME, DIRECTORY_PATH,
                               DerivativeOrder::Zero);
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

std::unique_ptr<ADModel<Scalar>> LowOrderTestModelFixture::ad_model_ptr_ =
    nullptr;
std::unique_ptr<CompiledModel<Scalar>>
    LowOrderTestModelFixture::compiled_model_ptr_ = nullptr;

TEST_F(LowOrderTestModelFixture, ThrowsOnJacobianHessian) {
    Vector input = Vector::Ones(NUM_INPUT);

    EXPECT_NO_THROW(compiled_model_ptr_->evaluate(input)) << "Evaluate threw.";
    EXPECT_THROW(compiled_model_ptr_->jacobian(input), std::runtime_error)
        << "Jacobian did not throw.";
    EXPECT_THROW(compiled_model_ptr_->hessian(input, 0), std::runtime_error)
        << "Hessian did not throw.";
}

}  // namespace LowOrderModelTest
}  // namespace CppADCodeGenEigenPy
