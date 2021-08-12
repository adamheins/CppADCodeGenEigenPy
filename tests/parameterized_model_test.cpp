#include <gtest/gtest.h>

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADFunction.h>
#include <CppADCodeGenEigenPy/ADModel.h>

using namespace CppADCodeGenEigenPy;

using Scalar = double;

static const std::string MODEL_NAME = "ParameterizedTestModel";
static const std::string FOLDER_NAME = "/tmp/CppADCodeGenEigenPy";

static const int NUM_INPUT = 3;
static const int NUM_PARAM = NUM_INPUT;
static const int NUM_OUTPUT = 1;

template <typename Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

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

    ParameterizedTestModel(const std::string& model_name,
                           const std::string& folder_name, ADOrder order)
        : ADModel<Scalar>(model_name, folder_name, order){};

    // Generate the input to the function
    ADVector input() const override { return ADVector::Ones(NUM_INPUT); }

    ADVector parameters() const override { return ADVector::Ones(NUM_PARAM); }

    // Evaluate the function
    ADVector function(const ADVector& input,
                      const ADVector& parameters) const override {
        return evaluate<ADScalar>(input, parameters);
    }
};

class ParameterizedTestModelFixture : public ::testing::Test {
   protected:
    using Vector = ADFunction<Scalar>::Vector;
    using Matrix = ADFunction<Scalar>::Matrix;

    void SetUp() override {
        model_ptr_.reset(new ParameterizedTestModel<Scalar>(
            MODEL_NAME, FOLDER_NAME, ADOrder::Second));
        model_ptr_->compile();
        function_ptr_.reset(
            new ADFunction<Scalar>(model_ptr_->get_model_name(),
                                   model_ptr_->get_library_generic_path()));
    }

    std::unique_ptr<ADModel<Scalar>> model_ptr_;
    std::unique_ptr<ADFunction<Scalar>> function_ptr_;
};

TEST_F(ParameterizedTestModelFixture, Evaluation) {
    Vector input = Vector::Ones(NUM_INPUT);
    Vector parameters = Vector::Ones(NUM_INPUT);

    Vector output_expected = evaluate<Scalar>(input, parameters);
    Vector output_actual = function_ptr_->evaluate(input, parameters);
    EXPECT_TRUE(output_actual.isApprox(output_expected))
        << "Function evaluation is incorrect.";

    // if I forget to pass the parameters, I should get a runtime error
    EXPECT_THROW(function_ptr_->evaluate(input), std::runtime_error)
        << "Missing parameters did not throw error.";
}

TEST_F(ParameterizedTestModelFixture, Jacobian) {
    Vector input = Vector::Ones(NUM_INPUT);
    Vector parameters = Vector::Ones(NUM_INPUT);

    // note Jacobian of scalar function is a row vector, hence the transpose
    Matrix P = Matrix::Zero(NUM_INPUT, NUM_INPUT);
    P.diagonal() << parameters;
    Matrix J_expected = input.transpose() * P;
    Matrix J_actual = function_ptr_->jacobian(input, parameters);
    EXPECT_TRUE(J_actual.isApprox(J_expected)) << "Jacobian is incorrect.";

    EXPECT_THROW(function_ptr_->jacobian(input), std::runtime_error)
        << "Missing parameters did not throw error.";
}

TEST_F(ParameterizedTestModelFixture, Hessian) {
    Vector input = Vector::Ones(NUM_INPUT);
    Vector parameters = Vector::Ones(NUM_INPUT);

    Matrix H_expected = Matrix::Zero(NUM_INPUT, NUM_INPUT);
    H_expected.diagonal() << parameters;
    Matrix H_actual = function_ptr_->hessian(input, parameters, 0);
    EXPECT_TRUE(H_actual.isApprox(H_expected)) << "Hessian is incorrect.";
    EXPECT_THROW(function_ptr_->hessian(input, 0), std::runtime_error)
        << "Missing parameters did not throw error.";
}
