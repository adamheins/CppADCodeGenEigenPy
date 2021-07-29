#include <iostream>

#include <CppADCodeGenEigenPy/ADFunction.h>

#include "example/Defs.h"


int main() {
    using Scalar = double;

    ADFunction<Scalar> f(MODEL_NAME, LIB_NAME);

    using Vector = ADFunction<Scalar>::Vector;
    using Matrix = ADFunction<Scalar>::Matrix;

    Vector x = Vector::Ones(NUM_INPUT);
    Vector p = Vector::Ones(NUM_PARAM);
    Vector y = f.evaluate(x, p);
    Matrix J = f.jacobian(x, p);

    std::cout << y << std::endl;
    std::cout << J << std::endl;
}
