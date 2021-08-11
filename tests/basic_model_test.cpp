#include <gtest/gtest.h>

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADFunction.h>
#include <CppADCodeGenEigenPy/ADModel.h>

using Scalar = double;

static const std::string MODEL_NAME = "BasicTestModel";
static const std::string FOLDER_NAME = "/tmp/CppADCodeGenEigenPy";

static const int NUM_INPUT = 3;
static const int NUM_OUTPUT = NUM_INPUT;

template <typename Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// Just multiply input vector by 2.
template <typename Scalar>
static Vector<Scalar> evaluate(const Vector<Scalar>& input) {
    return input * Scalar(2.);
}

template <typename Scalar>
struct BasicTestModel : public ADModel<Scalar> {
    using typename ADModel<Scalar>::ADScalar;
    using typename ADModel<Scalar>::ADVector;

    BasicTestModel(const std::string& model_name,
                   const std::string& folder_name, ADOrder order)
        : ADModel<Scalar>(model_name, folder_name, order){};

    // Generate the input to the function
    ADVector input() const override { return ADVector::Ones(NUM_INPUT); }

    // Evaluate the function
    ADVector function(const ADVector& input) const override {
        return evaluate<ADScalar>(input);
    }
};

class BasicTestModelFixture : public ::testing::Test {
   protected:
    using Vector = ADFunction<Scalar>::Vector;
    using Matrix = ADFunction<Scalar>::Matrix;

    void SetUp() override {
        model_ptr_.reset(new BasicTestModel<Scalar>(MODEL_NAME, FOLDER_NAME,
                                                    ADOrder::Second));
        model_ptr_->compile();
        function_ptr_.reset(
            new ADFunction<Scalar>(model_ptr_->get_model_name(),
                                   model_ptr_->get_library_generic_path()));
    }

    // TODO we could delete the .so afterward
    // void TearDown() override {}

    std::unique_ptr<ADModel<Scalar>> model_ptr_;
    std::unique_ptr<ADFunction<Scalar>> function_ptr_;
};

TEST_F(BasicTestModelFixture, CreatesSharedLib) {
    ASSERT_TRUE(model_ptr_->library_exists())
        << "Shared library file not found.";
}

TEST_F(BasicTestModelFixture, Dimensions) {
    // test that the model shape is correct
    ASSERT_EQ(function_ptr_->get_input_size(), NUM_INPUT)
        << "Input size is incorrect.";
    ASSERT_EQ(function_ptr_->get_output_size(), NUM_OUTPUT)
        << "Output size is incorrect.";

    // generate an input with a size too large
    Vector x = Vector::Ones(NUM_INPUT + 1);

    ASSERT_THROW(function_ptr_->evaluate(x), std::runtime_error)
        << "Evaluate with input of wrong size did not throw.";
    ASSERT_THROW(function_ptr_->jacobian(x), std::runtime_error)
        << "Jacobian with input of wrong size did not throw.";
    ASSERT_THROW(function_ptr_->hessian(x, 0), std::runtime_error)
        << "Hessian with input of wrong size did not throw.";
}

TEST_F(BasicTestModelFixture, Evaluation) {
    Vector input = Vector::Ones(NUM_INPUT);

    Vector output_expected = evaluate<Scalar>(input);
    Vector output_actual = function_ptr_->evaluate(input);
    ASSERT_TRUE(output_actual.isApprox(output_expected))
        << "Function evaluation is incorrect.";
}

TEST_F(BasicTestModelFixture, Jacobian) {
    Vector input = Vector::Ones(NUM_INPUT);

    Matrix J_expected(NUM_OUTPUT, NUM_INPUT);
    J_expected.setZero();
    J_expected.diagonal() << 2, 2, 2;
    Matrix J_actual = function_ptr_->jacobian(input);

    ASSERT_TRUE(J_actual.isApprox(J_expected)) << "Jacobian is incorrect.";
}

TEST_F(BasicTestModelFixture, Hessian) {
    Vector input = Vector::Ones(NUM_INPUT);

    // Hessian of all output dimensions should be zero
    Matrix H_expected = Matrix::Zero(NUM_INPUT, NUM_INPUT);
    Matrix H0_actual = function_ptr_->hessian(input, 0);
    Matrix H1_actual = function_ptr_->hessian(input, 1);
    Matrix H2_actual = function_ptr_->hessian(input, 2);

    ASSERT_TRUE(H0_actual.isApprox(H_expected))
        << "Hessian for dim 0 is incorrect.";
    ASSERT_TRUE(H1_actual.isApprox(H_expected))
        << "Hessian for dim 1 is incorrect.";
    ASSERT_TRUE(H2_actual.isApprox(H_expected))
        << "Hessian for dim 2 is incorrect.";

    ASSERT_THROW(function_ptr_->hessian(input, NUM_OUTPUT), std::runtime_error)
        << "Hessian with too-large output_dim did not throw.";
}
