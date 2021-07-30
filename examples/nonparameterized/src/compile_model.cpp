#include <iostream>

#include <CppADCodeGenEigenPy/ADModel.h>

#include "example/Defs.h"


template <typename Scalar>
struct MyADModel : public ADModel<Scalar> {
    using typename ADModel<Scalar>::ADScalar;
    using typename ADModel<Scalar>::ADVector;

    MyADModel(const std::string& model_name, const std::string& folder_name)
        : ADModel<Scalar>(model_name, folder_name){};

    // Generate the input to the function
    ADVector input() override { return ADVector::Ones(NUM_INPUT); }

    // Evaluate the function
    ADVector function(const ADVector& input) override {
        ADVector output = input * ADScalar(2.);
        output(0) += input(1);
        return output;
    }
};

int main() {
    MyADModel<double> model(MODEL_NAME, FOLDER_NAME);
    model.compile();
}
