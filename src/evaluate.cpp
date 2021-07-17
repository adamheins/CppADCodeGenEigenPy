#include <iostream>

#include <CppADCodeGenEigenPy/ADFunction.h>


int main() {
    ADFunction<double> f("my_model", "autogen/libmy_model", 3, 3);

    using Vector = ADFunction<double>::Vector;
    using Matrix = ADFunction<double>::Matrix;

    Vector x = Vector::Ones(3);
    Vector y = f.evaluate(x);
    Matrix J = f.jacobian(x);

    std::cout << y << std::endl;
    std::cout << J << std::endl;
}
