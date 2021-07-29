#include <iostream>

#include <CppADCodeGenEigenPy/ADModel.h>

#include "example/Defs.h"

struct MyADModel : ADModel<double> {
    MyADModel() : ADModel(MODEL_NAME, FOLDER_NAME){};

    // Generate the input to the function
    ADVector input() override { return ADVector::Ones(NUM_INPUT); }

    // Evaluate the function
    ADVector function(const ADVector& input) override {
        ADVector output = input * ADScalar(2);
        output(0) += input(1);
        return output;
    }
};

int main() {
    MyADModel model;
    model.compile();
}
