#include <gtest/gtest.h>

#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADFunction.h>
#include <CppADCodeGenEigenPy/ADModel.h>

using namespace CppADCodeGenEigenPy;

using Scalar = double;

static const std::string MODEL_NAME = "LowOrderTestModel";
static const std::string FOLDER_NAME = "/tmp/CppADCodeGenEigenPy";

static const int NUM_INPUT = 3;
static const int NUM_OUTPUT = NUM_INPUT;

template <typename Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
struct LowOrderTestModel : public ADModel<Scalar> {
    using typename ADModel<Scalar>::ADScalar;
    using typename ADModel<Scalar>::ADVector;

    LowOrderTestModel(const std::string& model_name,
                      const std::string& folder_name, ADOrder order)
        : ADModel<Scalar>(model_name, folder_name, order){};

    ADVector input() const override { return ADVector::Ones(NUM_INPUT); }

    ADVector function(const ADVector& input) const override { return input; }
};

class LowOrderTestModelFixture : public ::testing::Test {
   protected:
    using Vector = ADFunction<Scalar>::Vector;
    using Matrix = ADFunction<Scalar>::Matrix;

    void SetUp() override {
        model_ptr_.reset(new LowOrderTestModel<Scalar>(MODEL_NAME, FOLDER_NAME,
                                                       ADOrder::Zero));
        model_ptr_->compile();
        function_ptr_.reset(
            new ADFunction<Scalar>(model_ptr_->get_model_name(),
                                   model_ptr_->get_library_generic_path()));
    }

    std::unique_ptr<ADModel<Scalar>> model_ptr_;
    std::unique_ptr<ADFunction<Scalar>> function_ptr_;
};

TEST_F(LowOrderTestModelFixture, ThrowsOnJacobianHessian) {
    Vector input = Vector::Ones(NUM_INPUT);

    EXPECT_NO_THROW(function_ptr_->evaluate(input)) << "Evaluate threw.";
    EXPECT_THROW(function_ptr_->jacobian(input), std::runtime_error)
        << "Jacobian did not throw.";
    EXPECT_THROW(function_ptr_->hessian(input, 0), std::runtime_error)
        << "Hessian did not throw.";
}
