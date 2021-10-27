# CppADCodeGenEigenPy

CppADCodeGen with an easy Eigen interface and Python bindings.

This project has been tested on Ubuntu 16.04. It should work on more recent
Ubuntu versions, and likely works on other *nix systems as well.

## Motivation

I want to be able to prototype code in Python while making use of fast,
compiled auto-differentiated code. Autodiff tools in Python, such as
[JAX](https://github.com/google/jax), often just-in-time compile the code rather
than ahead-of-time compile it. I don't want to wait for JIT every time I run
the script if nothing has changed.

This project was heavily inspired by larger frameworks that incorporate
auto-diff functionality via CppADCodeGen for convenience; in particular, the
[automatic_differentiation](https://github.com/leggedrobotics/ocs2/tree/main/ocs2_core/include/ocs2_core/automatic_differentiation)
module from [OCS2](https://github.com/leggedrobotics/ocs2) was a big influence.

## How it works

This project provides a simple interface to CppADCodeGen to define the required
function in C++, which is then compiled (along with its automatically-computed
derivatives) into a dynamic library. This dynamic library can then be easily used
from C++ or imported into Python via bindings based on
[pybind11](https://github.com/pybind/pybind11).

CppADCodeGenEigenPy is built around two classes. The first is the `ADModel`,
which defines the function to be auto-differentiated. It is compiled into a
dynamic library which can then be loaded by the `CompiledModel` class. The
`CompiledModel` can evaluate the function itself as well as its first and
second derivatives.

## Install

First ensure you have the required dependencies:
* a compiler with C++11 support
* [cmake](https://cmake.org/) version 3.12+
* [Eigen](https://eigen.tuxfamily.org/) version 3.3+
* [CppADCodeGen](https://github.com/joaoleal/CppADCodeGen)

Clone CppADCodeGenEigenPy:
```
git clone https://github.com/adamheins/CppADCodeGenEigenPy
cd CppADCodeGenEigenPy
```

The main C++ part of CppADCodeGenEigenPy uses cmake. The library itself is
header-only, so you need only do:
```
mkdir build
cmake -S . -B build
sudo cmake --install build
```

To build and install the Python bindings, `pip` is used. Note that `pip`
version 10+ is required in order to parse `pyproject.toml`. Run
```
python -m pip install .
```

## Tests

The C++ tests require [Boost](https://www.boost.org/) (version 1.58+).

To build the tests, run
```
cmake --build build
```

To run the C++ tests:
```
cd build
ctest
```

To run the Python tests:
```
python -m pytest
```
Note that the tests assume that the name of the build directory is `build` in
order to find the required dynamic libraries. If this is not the case, you can
pass `--builddir NAME` to pytest to change it.


## Example

A minimal example of the C++ code to define and generate an auto-diff model
would look something like:
```c++
#include <CppADCodeGenEigenPy/ADModel.h>
#include <Eigen/Eigen>

namespace ad = CppADCodeGenEigenPy;

// Our custom model extends the ad::ADModel class
template <typename Scalar>
struct ExampleModel : public ad::ADModel<Scalar> {
    using typename ad::ADModel<Scalar>::ADVector;

    // Generate the input used when differentiating the function
    ADVector input() const override { return ADVector::Ones(3); }

    // Generate parameters used when differentiating the function
    // This can be skipped if the function takes no parameters
    ADVector parameters() const override { return ADVector::Ones(3); }

    // Implement a weighted sum-of-squares function.
    ADVector function(const ADVector& input,
                      const ADVector& parameters) const override {
        ADVector output(1);
        for (int i = 0; i < 3; ++i) {
            output(0) += 0.5 * parameters(i) * input(i) * input(i);
        }
        return output;
    }
};

int main() {
    // Compile model named ExampleModel and save in the current directory; the
    // model is saved as a shared object file named libExampleModel.so
    ExampleModel<double>().compile("ExampleModel", ".",
                                   ad::DerivativeOrder::First);
}
```

Compile the code and generate the model:
```
g++ -std=c++11 -I/usr/local/include/eigen3 model.cpp -ldl -o make_model
./make_model
```

This code can then be called from Python using:
```python
import numpy as np
from CppADCodeGenEigenPy import CompiledModel

# note that the .so file extension not included on the second argument
model = CompiledModel("ExampleModel", "libExampleModel")

inputs = np.array([1, 2, 3])
params = np.ones(3)

# compute model output and first derivative
output = model.evaluate(inputs, params)
jacobian = model.jacobian(inputs, params)

print(f"output = {output}")
print(f"jacobian = {jacobian}")
```
This C++ and Python code can be found in the
[example](https://github.com/adamheins/CppADCodeGenEigenPy/tree/main/example)
directory.

A fully worked example that differentiates functions related to rigid body
dynamics (which could be used for something like optimal control) can be found
[here](https://github.com/adamheins/CppADCodeGenEigenPy-dynamics-example).

## License

MIT
