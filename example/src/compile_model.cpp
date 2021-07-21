#include <iostream>

#include <CppADCodeGenEigenPy/ADModel.h>

#include "example/Defs.h"


struct MyADModel : ADModel<double> {
    MyADModel() : ADModel(MODEL_NAME, FOLDER_NAME){};

    // Generate the input to the function
    ADVector input() override { return ADVector::Ones(NUM_INPUT); }

    // Evaluate the function
    ADVector function(const ADVector& x) override {
        ADVector y = x * ADScalar(2);
        y(0) += x(1);
        return y;
    }
};


int main() {
    MyADModel model;
    model.compile();
}
