#include <Eigen/Eigen>

#include <CppADCodeGenEigenPy/ADModel.h>

namespace ad = CppADCodeGenEigenPy;

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
struct BasicTestModel : public ad::ADModel<Scalar> {
    using typename ad::ADModel<Scalar>::ADScalar;
    using typename ad::ADModel<Scalar>::ADVector;

    BasicTestModel(const std::string& model_name,
                   const std::string& folder_name, ad::ADOrder order)
        : ad::ADModel<Scalar>(model_name, folder_name, order){};

    // Generate the input to the function
    ADVector input() const override { return ADVector::Ones(NUM_INPUT); }

    // Evaluate the function
    ADVector function(const ADVector& input) const override {
        return evaluate<ADScalar>(input);
    }
};
