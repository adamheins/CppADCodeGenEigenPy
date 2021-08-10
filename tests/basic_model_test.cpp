#include <gtest/gtest.h>

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADFunction.h>
#include <CppADCodeGenEigenPy/ADModel.h>

using Scalar = double;
using Vector = ADFunction<Scalar>::Vector;
using Matrix = ADFunction<Scalar>::Matrix;

static const std::string MODEL_NAME = "BasicTestModel";
static const std::string FOLDER_NAME = "/tmp/CppADCodeGenEigenPy";

static const int NUM_INPUT = 3;
static const int NUM_OUTPUT = NUM_INPUT;


// Create a basic model that takes in a vector of length 3 and multiples it by
// 2.
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
        ADVector output = input * ADScalar(2.);
        return output;
    }
};

class BasicTestModelFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        model_ptr_.reset(new BasicTestModel<Scalar>(MODEL_NAME, FOLDER_NAME,
                                                    ADOrder::Second));
        model_ptr_->compile();
    }

    // TODO we could delete the .so afterward
    // void TearDown() override {}

    std::unique_ptr<ADModel<Scalar>> model_ptr_;
};

TEST_F(BasicTestModelFixture, CreatesSharedLib) {
    ASSERT_TRUE(model_ptr_->library_exists())
        << "Shared library file not found.";
}

TEST_F(BasicTestModelFixture, ADFunctionDimensions) {
    ADFunction<Scalar> f(model_ptr_->get_model_name(),
                         model_ptr_->get_library_generic_path());

    // test that the model shape is correct
    ASSERT_EQ(f.get_input_size(), NUM_INPUT) << "Input size is incorrect.";
    ASSERT_EQ(f.get_output_size(), NUM_OUTPUT) << "Output size is incorrect.";

    // generate an input with a size too large
    Vector x(NUM_INPUT + 1);
    x.setOnes();

    ASSERT_THROW(f.evaluate(x), std::runtime_error);
    ASSERT_THROW(f.jacobian(x), std::runtime_error);
    ASSERT_THROW(f.hessian(x, 0), std::runtime_error);
}

TEST_F(BasicTestModelFixture, ADFunctionEvaluation) {
    ADFunction<Scalar> f(model_ptr_->get_model_name(),
                         model_ptr_->get_library_generic_path());

    Vector x(NUM_INPUT);
    x.setOnes();

    // test that the function output is correct
    Vector y_expected = 2 * x;
    Vector y_actual = f.evaluate(x);
    ASSERT_TRUE(y_actual.isApprox(y_expected))
        << "Function evaluation is incorrect.";

    // test Jacobian
    Matrix J_expected(NUM_OUTPUT, NUM_INPUT);
    J_expected.setZero();
    J_expected.diagonal() << 2, 2, 2;

    Matrix J_actual = f.jacobian(x);

    ASSERT_TRUE(J_actual.isApprox(J_expected)) << "Jacobian is incorrect.";

    // test Hessian
    Matrix H_expected = Matrix::Zero(NUM_INPUT, NUM_INPUT);
    Matrix H_actual = f.hessian(x, 0);  // all should be zero
    ASSERT_TRUE(H_actual.isApprox(H_expected)) << "Hessian is incorrect.";
}

