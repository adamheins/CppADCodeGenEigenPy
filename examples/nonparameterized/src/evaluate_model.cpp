#include <iostream>

#include <CppADCodeGenEigenPy/ADFunction.h>

#include "example/Defs.h"


int main() {
    using Scalar = double;

    ADFunction<Scalar> f(MODEL_NAME, LIB_NAME);

    using Vector = ADFunction<Scalar>::Vector;
    using Matrix = ADFunction<Scalar>::Matrix;

    Vector x = Vector::Ones(NUM_INPUT);
    Vector y = f.evaluate(x);
    Matrix J = f.jacobian(x);

    std::cout << y << std::endl;
    std::cout << J << std::endl;
}
