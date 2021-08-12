#include <gtest/gtest.h>

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADFunction.h>
#include <CppADCodeGenEigenPy/ADModel.h>

using namespace CppADCodeGenEigenPy;

using Scalar = double;

static const std::string MODEL_NAME = "MathFunctionsTestModel";
static const std::string FOLDER_NAME = "/tmp/CppADCodeGenEigenPy";

static const int NUM_INPUT = 3;
static const int NUM_OUTPUT = NUM_INPUT;

template <typename Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

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

    MathFunctionsTestModel(const std::string& model_name,
                           const std::string& folder_name, ADOrder order)
        : ADModel<Scalar>(model_name, folder_name, order){};

    // Generate the input to the function
    ADVector input() const override { return ADVector::Ones(NUM_INPUT); }

    // Evaluate the function
    ADVector function(const ADVector& input) const override {
        return evaluate<ADScalar>(input);
    }
};

class MathFunctionsTestModelFixture : public ::testing::Test {
   protected:
    using Vector = ADFunction<Scalar>::Vector;
    using Matrix = ADFunction<Scalar>::Matrix;

    void SetUp() override {
        model_ptr_.reset(new MathFunctionsTestModel<Scalar>(
            MODEL_NAME, FOLDER_NAME, ADOrder::Second));
        model_ptr_->compile();
        function_ptr_.reset(
            new ADFunction<Scalar>(model_ptr_->get_model_name(),
                                   model_ptr_->get_library_generic_path()));
    }

    std::unique_ptr<ADModel<Scalar>> model_ptr_;
    std::unique_ptr<ADFunction<Scalar>> function_ptr_;
};

TEST_F(MathFunctionsTestModelFixture, Evaluation) {
    Vector input = Vector::Ones(NUM_INPUT);

    Vector output_expected = evaluate<Scalar>(input);
    Vector output_actual = function_ptr_->evaluate(input);

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
    Matrix J_actual = function_ptr_->jacobian(input);

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

    Matrix H0_actual = function_ptr_->hessian(input, 0);
    Matrix H1_actual = function_ptr_->hessian(input, 1);
    Matrix H2_actual = function_ptr_->hessian(input, 2);

    EXPECT_TRUE(H0_actual.isApprox(H0_expected))
        << "Hessian for dim 0 is incorrect.";
    EXPECT_TRUE(H1_actual.isApprox(H1_expected))
        << "Hessian for dim 1 is incorrect.";
    EXPECT_TRUE(H2_actual.isApprox(H2_expected))
        << "Hessian for dim 2 is incorrect.";
}
