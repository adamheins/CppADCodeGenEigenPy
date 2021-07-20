#include <iostream>

#include <CppADCodeGenEigenPy/ADFunction.h>


int main() {
    using Scalar = double;

    ADFunction<Scalar> f("my_model", "autogen/libmy_model");

    using Vector = ADFunction<Scalar>::Vector;
    using Matrix = ADFunction<Scalar>::Matrix;

    Vector x = Vector::Ones(3);
    Vector y = f.evaluate(x);
    Matrix J = f.jacobian(x);

    std::cout << y << std::endl;
    std::cout << J << std::endl;
}
