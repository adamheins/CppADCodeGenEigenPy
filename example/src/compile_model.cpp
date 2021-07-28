#include <iostream>

#include <CppADCodeGenEigenPy/ADModel.h>

#include "example/Defs.h"

struct MyADModel : ADModel<double> {
    MyADModel() : ADModel(MODEL_NAME, FOLDER_NAME){};

    ADVector parameters() override { return ADVector::Ones(1); }

    // Generate the input to the function
    ADVector input() override { return ADVector::Ones(NUM_INPUT); }

    // Evaluate the function
    ADVector function(const ADVector& input,
                      const ADVector& parameters) override {
        ADVector output = input * parameters(0);
        output(0) = output(0) + input(1);
        return output;
    }
};

int main() {
    MyADModel model;
    model.compile();
}
