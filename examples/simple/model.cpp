#include <CppADCodeGenEigenPy/ADModel.h>
#include <Eigen/Eigen>

namespace ad = CppADCodeGenEigenPy;

// Our custom model extends the ad::ADModel class
template <typename Scalar>
struct ExampleModel : public ad::ADModel<Scalar> {
    using typename ad::ADModel<Scalar>::ADVector;

    // Generate the input used when differentiating the function
    ADVector input() const override { return ADVector::Ones(3); }

    // Generate parameters used when differentiating the function
    // This can be skipped if the function takes no parameters
    ADVector parameters() const override { return ADVector::Ones(3); }

    // Implement a weighted sum-of-squares function.
    ADVector function(const ADVector& input,
                      const ADVector& parameters) const override {
        ADVector output(1);
        for (int i = 0; i < 3; ++i) {
            output(0) += 0.5 * parameters(i) * input(i) * input(i);
        }
        return output;
    }
};

int main() {
    // Compile model named ExampleModel and save in the current directory; the
    // model is saved as a shared object file named libExampleModel.so
    ExampleModel<double>().compile("ExampleModel", ".",
                                   ad::DerivativeOrder::First);
}
