#include <gtest/gtest.h>

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADFunction.h>
#include <CppADCodeGenEigenPy/ADModel.h>

using Scalar = double;
using Vector = ADFunction<Scalar>::Vector;
using Matrix = ADFunction<Scalar>::Matrix;

static const std::string MODEL_NAME = "ParameterizedTestModel";
static const std::string FOLDER_NAME = "/tmp/CppADCodeGenEigenPy";

static const int NUM_INPUT = 3;
static const int NUM_PARAM = NUM_INPUT;
static const int NUM_OUTPUT = 1;

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
        ADVector output(1);
        for (int i = 0; i < NUM_INPUT; ++i) {
            output(0) += parameters(i) * input(i);
        }
        return output;
    }
};

class ParameterizedTestModelFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        model_ptr_.reset(new ParameterizedTestModel<Scalar>(
            MODEL_NAME, FOLDER_NAME, ADOrder::Second));
        model_ptr_->compile();
    }

    // TODO we could delete the .so afterward
    // void TearDown() override {}

    std::unique_ptr<ADModel<Scalar>> model_ptr_;
};

TEST_F(ParameterizedTestModelFixture, ADFunctionEvaluation) {
    ADFunction<Scalar> f(model_ptr_->get_model_name(),
                         model_ptr_->get_library_generic_path());

    Vector x = Vector::Ones(NUM_INPUT);
    Vector p = Vector::Ones(NUM_INPUT);

    // test that the function output is correct
    Vector y_expected = Vector::Zero(NUM_OUTPUT);
    for (int i = 0; i < NUM_INPUT; ++i) {
        y_expected(0) += p(i) * x(i);
    }
    Vector y_actual = f.evaluate(x, p);
    ASSERT_TRUE(y_actual.isApprox(y_expected))
        << "Function evaluation is incorrect.";

    // if I forget to pass the parameters, I should get a runtime error
    ASSERT_THROW(f.evaluate(x), std::runtime_error);

    // // test Jacobian
    // Matrix J_expected(NUM_INPUT, NUM_INPUT);
    // J_expected.setZero();
    // J_expected.diagonal() << 2, 2, 2;
    //
    // Matrix J_actual = f.jacobian(x);
    //
    // ASSERT_TRUE(J_actual.isApprox(J_expected)) << "Jacobian is incorrect.";
    //
    // // test Hessian
    // Matrix H_expected = Matrix::Zero(NUM_INPUT, NUM_INPUT);
    // Matrix H_actual = f.hessian(x, 0);  // all should be zero
    // ASSERT_TRUE(H_actual.isApprox(H_expected)) << "Hessian is incorrect.";
}

