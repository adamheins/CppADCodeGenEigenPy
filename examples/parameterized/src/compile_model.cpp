#include <iostream>

#include <CppADCodeGenEigenPy/ADModel.h>

#include "example/Defs.h"

struct MyParameterizedADModel : ADModel<double> {
    MyParameterizedADModel() : ADModel(MODEL_NAME, FOLDER_NAME){};

    // Generate the input to the function used when recording autodiff
    // operations.
    ADVector input() override { return ADVector::Ones(NUM_INPUT); }

    // Also define the vector of paramters
    ADVector parameters() override { return ADVector::Ones(NUM_PARAM); }

    // Evaluate the function
    ADVector function(const ADVector& input,
                      const ADVector& parameters) override {
        ADVector output = input * parameters(0);
        output(0) = output(0) + input(1);
        return output;
    }
};

int main() {
    MyParameterizedADModel model;
    model.compile();
}
