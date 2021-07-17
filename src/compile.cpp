#include <CppADCodeGenEigenPy/ADModel.h>


struct MyADModel : ADModel<double> {
    MyADModel(size_t input_dim, size_t output_dim)
        : ADModel("my_model", input_dim, output_dim, "autogen"){};

    // Generate the input to the function
    ADVector input() { return ADVector::Ones(input_dim); }

    // Evaluate the function
    ADVector function(const ADVector& x) override {
        ADVector y = x * ADScalar(2);
        y(0) += x(1);
        return y;
    }
};


int main() {
    MyADModel model(3, 3);
    model.compile();
}
