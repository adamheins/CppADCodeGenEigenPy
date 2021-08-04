#include <gtest/gtest.h>

#include <CppADCodeGenEigenPy/ADModel.h>
#include <CppADCodeGenEigenPy/ADFunction.h>


using Scalar = double;
using Vector = ADFunction<Scalar>::Vector;
using Matrix = ADFunction<Scalar>::Matrix;

static const std::string MODEL_NAME = "TestModel";
static const std::string FOLDER_NAME = "/tmp/CppADCodeGenEigenPy";

// TODO would probably be better if this was just the path
static const std::string LIB_NAME = FOLDER_NAME + "/lib" + MODEL_NAME;

static const int NUM_INPUT = 3;


template <typename Scalar>
struct MyADModel : public ADModel<Scalar> {
    using typename ADModel<Scalar>::ADScalar;
    using typename ADModel<Scalar>::ADVector;

    MyADModel(const std::string& model_name, const std::string& folder_name,
              ADOrder order)
        : ADModel<Scalar>(model_name, folder_name, order){};

    // Generate the input to the function
    ADVector input() override { return ADVector::Ones(NUM_INPUT); }

    // Evaluate the function
    ADVector function(const ADVector& input) override {
        ADVector output = input * ADScalar(2.);
        output(0) *= input(1);
        return output;
    }
};

class ADModelTest : public ::testing::Test {
   protected:
    void SetUp() override {
        model_ptr_.reset(
            new MyADModel<Scalar>(MODEL_NAME, FOLDER_NAME, ADOrder::First));
        model_ptr_->compile();
    }

    // void TearDown() override {}

    std::unique_ptr<MyADModel<Scalar>> model_ptr_;
};

// TODO what do we want to test:
// * .so is actually generated
// * forward, Jacobian, Hessian are correct
//      - orders are correct
// * parameterized vs nonparameterized
// * dimensions

TEST_F(ADModelTest, CreatesSharedLib) {
    ASSERT_TRUE(model_ptr_->library_exists()) << "Shared library file not found!";
}

TEST_F(ADModelTest, LoadADFunction) {
    ADFunction<Scalar> f(MODEL_NAME, LIB_NAME);

    ASSERT_TRUE(model_ptr_->library_exists()) << "Shared library file not found!";
}
