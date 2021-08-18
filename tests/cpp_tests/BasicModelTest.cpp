#include <gtest/gtest.h>

#include <Eigen/Eigen>
#include <boost/filesystem.hpp>

#include <CppADCodeGenEigenPy/ADModel.h>
#include <CppADCodeGenEigenPy/CompiledModel.h>
#include <CppADCodeGenEigenPy/Util.h>

#include "testing/models/BasicTestModel.h"

namespace CppADCodeGenEigenPy {
namespace BasicModelTest {

class BasicTestModelFixture : public ::testing::Test {
   protected:
    using Vector = CompiledModel<Scalar>::Vector;
    using Matrix = CompiledModel<Scalar>::Matrix;

    static void SetUpTestSuite() {
        // Compile and load our model
        boost::filesystem::create_directories(DIRECTORY_PATH);
        ad_model_ptr_.reset(new BasicTestModel<Scalar>());
        ad_model_ptr_->compile(MODEL_NAME, DIRECTORY_PATH,
                               DerivativeOrder::Second);
        compiled_model_ptr_.reset(
            new CompiledModel<Scalar>(MODEL_NAME, LIB_GENERIC_PATH));
    }

    static void TearDownTestSuite() {
        // Delete the compiled shared object.
        boost::filesystem::remove_all(DIRECTORY_PATH);
    }

    static std::unique_ptr<BasicTestModel<Scalar>> ad_model_ptr_;
    static std::unique_ptr<CompiledModel<Scalar>> compiled_model_ptr_;
};

std::unique_ptr<BasicTestModel<Scalar>> BasicTestModelFixture::ad_model_ptr_ =
    nullptr;
std::unique_ptr<CompiledModel<Scalar>>
    BasicTestModelFixture::compiled_model_ptr_ = nullptr;

TEST_F(BasicTestModelFixture, CreatesSharedLib) {
    EXPECT_TRUE(boost::filesystem::exists(LIB_REAL_PATH))
        << "Shared library file " << LIB_REAL_PATH << " not found.";
}

TEST_F(BasicTestModelFixture, Dimensions) {
    // test that the model shape is correct
    EXPECT_EQ(compiled_model_ptr_->get_input_size(), NUM_INPUT)
        << "Input size is incorrect.";
    EXPECT_EQ(compiled_model_ptr_->get_output_size(), NUM_OUTPUT)
        << "Output size is incorrect.";

    // generate an input with a size too large
    Vector input = Vector::Ones(NUM_INPUT + 1);

    EXPECT_THROW(compiled_model_ptr_->evaluate(input), std::runtime_error)
        << "Evaluate with input of wrong size did not throw.";
    EXPECT_THROW(compiled_model_ptr_->jacobian(input), std::runtime_error)
        << "Jacobian with input of wrong size did not throw.";
    EXPECT_THROW(compiled_model_ptr_->hessian(input, 0), std::runtime_error)
        << "Hessian with input of wrong size did not throw.";
}

TEST_F(BasicTestModelFixture, Evaluation) {
    Vector input = Vector::Ones(NUM_INPUT);

    Vector output_expected = evaluate<Scalar>(input);
    Vector output_actual = compiled_model_ptr_->evaluate(input);
    EXPECT_TRUE(output_actual.isApprox(output_expected))
        << "Function evaluation is incorrect.";
}

TEST_F(BasicTestModelFixture, Jacobian) {
    Vector input = Vector::Ones(NUM_INPUT);

    Matrix J_expected(NUM_OUTPUT, NUM_INPUT);
    J_expected.setZero();
    J_expected.diagonal() << 2, 2, 2;
    Matrix J_actual = compiled_model_ptr_->jacobian(input);

    EXPECT_TRUE(J_actual.isApprox(J_expected)) << "Jacobian is incorrect.";
}

TEST_F(BasicTestModelFixture, Hessian) {
    Vector input = Vector::Ones(NUM_INPUT);

    // Hessian of all output dimensions should be zero
    Matrix H_expected = Matrix::Zero(NUM_INPUT, NUM_INPUT);
    Matrix H0_actual = compiled_model_ptr_->hessian(input, 0);
    Matrix H1_actual = compiled_model_ptr_->hessian(input, 1);
    Matrix H2_actual = compiled_model_ptr_->hessian(input, 2);

    EXPECT_TRUE(H0_actual.isApprox(H_expected))
        << "Hessian for dim 0 is incorrect.";
    EXPECT_TRUE(H1_actual.isApprox(H_expected))
        << "Hessian for dim 1 is incorrect.";
    EXPECT_TRUE(H2_actual.isApprox(H_expected))
        << "Hessian for dim 2 is incorrect.";

    EXPECT_THROW(compiled_model_ptr_->hessian(input, NUM_OUTPUT),
                 std::runtime_error)
        << "Hessian with too-large output_dim did not throw.";
}

}  // namespace BasicModelTest
}  // namespace CppADCodeGenEigenPy
