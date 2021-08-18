#include <gtest/gtest.h>

#include <Eigen/Eigen>
#include <boost/filesystem.hpp>

#include <CppADCodeGenEigenPy/ADModel.h>
#include <CppADCodeGenEigenPy/CompiledModel.h>

using namespace CppADCodeGenEigenPy;

using Scalar = double;

static const std::string MODEL_NAME = "BasicTestModel";
static const std::string FOLDER_NAME = "/tmp/CppADCodeGenEigenPy";
static const std::string LIB_GENERIC_NAME =
    get_library_generic_path(MODEL_NAME, FOLDER_NAME);

static const int NUM_INPUT = 3;
static const int NUM_OUTPUT = NUM_INPUT;

template <typename Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// Just multiply input vector by 2.
template <typename Scalar>
static Eigen::Matrix<Scalar, NUM_OUTPUT, 1> evaluate(
    const Eigen::Matrix<Scalar, NUM_INPUT, 1>& input) {
    return input * Scalar(2.);
}

template <typename Scalar>
struct BasicTestModel : public ADModel<Scalar, NUM_INPUT, NUM_OUTPUT> {
    // TODO I have to do more unpleasant work to get the typenames now that it
    // is parameterized
    using Base = ADModel<Scalar, NUM_INPUT, NUM_OUTPUT>;
    using typename Base::ADScalar;
    using typename Base::ADInput;
    using typename Base::ADOutput;

    // Generate the input to the function
    ADInput input() const override { return ADInput::Ones(); }

    // Evaluate the function
    ADOutput function(const ADInput& input) const override {
        return evaluate<ADScalar>(input);
    }
};

class BasicTestModelFixture : public ::testing::Test {
   protected:
    using Vector = CompiledModel<Scalar>::Vector;
    using Matrix = CompiledModel<Scalar>::Matrix;

    void SetUp() override {
        ad_model_ptr_.reset(new BasicTestModel<Scalar>());
        ad_model_ptr_->compile(MODEL_NAME, FOLDER_NAME,
                               DerivativeOrder::Second);
        // model_ptr_.reset(
        //     new CompiledModel<Scalar>(MODEL_NAME, LIB_GENERIC_NAME));
    }

    // TODO we could delete the .so afterward
    // void TearDown() override {}

    std::unique_ptr<BasicTestModel<Scalar>> ad_model_ptr_;
    // std::unique_ptr<CompiledModel<Scalar>> model_ptr_;
};

TEST_F(BasicTestModelFixture, CreatesSharedLib) {
    EXPECT_TRUE(boost::filesystem::exists(
        get_library_real_path(MODEL_NAME, FOLDER_NAME)))
        << "Shared library file not found.";
    // EXPECT_TRUE(ad_model_ptr_->library_exists())
    //     << "Shared library file not found.";
}

// TEST_F(BasicTestModelFixture, Dimensions) {
//     // test that the model shape is correct
//     EXPECT_EQ(model_ptr_->get_input_size(), NUM_INPUT)
//         << "Input size is incorrect.";
//     EXPECT_EQ(model_ptr_->get_output_size(), NUM_OUTPUT)
//         << "Output size is incorrect.";
//
//     // generate an input with a size too large
//     Vector input = Vector::Ones(NUM_INPUT + 1);
//
//     EXPECT_THROW(model_ptr_->evaluate(input), std::runtime_error)
//         << "Evaluate with input of wrong size did not throw.";
//     EXPECT_THROW(model_ptr_->jacobian(input), std::runtime_error)
//         << "Jacobian with input of wrong size did not throw.";
//     EXPECT_THROW(model_ptr_->hessian(input, 0), std::runtime_error)
//         << "Hessian with input of wrong size did not throw.";
// }
//
// TEST_F(BasicTestModelFixture, Evaluation) {
//     Vector input = Vector::Ones(NUM_INPUT);
//
//     Vector output_expected = evaluate<Scalar>(input);
//     Vector output_actual = model_ptr_->evaluate(input);
//     EXPECT_TRUE(output_actual.isApprox(output_expected))
//         << "Function evaluation is incorrect.";
// }
//
// TEST_F(BasicTestModelFixture, Jacobian) {
//     Vector input = Vector::Ones(NUM_INPUT);
//
//     Matrix J_expected(NUM_OUTPUT, NUM_INPUT);
//     J_expected.setZero();
//     J_expected.diagonal() << 2, 2, 2;
//     Matrix J_actual = model_ptr_->jacobian(input);
//
//     EXPECT_TRUE(J_actual.isApprox(J_expected)) << "Jacobian is incorrect.";
// }
//
// TEST_F(BasicTestModelFixture, Hessian) {
//     Vector input = Vector::Ones(NUM_INPUT);
//
//     // Hessian of all output dimensions should be zero
//     Matrix H_expected = Matrix::Zero(NUM_INPUT, NUM_INPUT);
//     Matrix H0_actual = model_ptr_->hessian(input, 0);
//     Matrix H1_actual = model_ptr_->hessian(input, 1);
//     Matrix H2_actual = model_ptr_->hessian(input, 2);
//
//     EXPECT_TRUE(H0_actual.isApprox(H_expected))
//         << "Hessian for dim 0 is incorrect.";
//     EXPECT_TRUE(H1_actual.isApprox(H_expected))
//         << "Hessian for dim 1 is incorrect.";
//     EXPECT_TRUE(H2_actual.isApprox(H_expected))
//         << "Hessian for dim 2 is incorrect.";
//
//     EXPECT_THROW(model_ptr_->hessian(input, NUM_OUTPUT), std::runtime_error)
//         << "Hessian with too-large output_dim did not throw.";
// }
