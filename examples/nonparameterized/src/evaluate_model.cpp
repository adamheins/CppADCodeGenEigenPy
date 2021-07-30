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
    Matrix H0 = f.hessian(x, 0);

    std::cout << "y = " << y << std::endl;
    std::cout << "J = " << J << std::endl;
    std::cout << "H0 = " << H0 << std::endl;
}
