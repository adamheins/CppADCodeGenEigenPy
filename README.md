# CppADCodeGenEigenPy

CppADCodeGen with an easy Eigen interface and Python bindings.

This project has been tested on Ubuntu 16.04. It should work on more recent
Ubuntu versions, and likely works on other *nix systems as well.

## Motivation

I want to be able to prototype code in Python while making use of fast,
compiled auto-differentiated code. Autodiff tools in Python often just-in-time
compile the code rather than ahead-of-time compile it. I don't want to wait for
autodiff's JIT every time I run the script if nothing has changed.

This project was heavily inspired by larger frameworks that incorporate
auto-diff functionality via CppADCodeGen for convenience; in particular, the
[automatic_differentiation](https://github.com/leggedrobotics/ocs2/tree/main/ocs2_core/include/ocs2_core/automatic_differentiation)
module from [OCS2](https://github.com/leggedrobotics/ocs2) was a big influence.

## How it works

This project provides a simple interface to CppADCodeGen to define the required
function in C++, which is then compiled (along with its automatically-computed
derivates) into a dynamic library. This dynamic library can then be easily used
from C++ or imported into Python via pybind11-based bindings.

CppADCodeGenEigenPy is built around two classes. The first is the `ADModel`,
which defines the function to be auto-differentiated. It is compiled into a
dynamic library which can then be loaded by the second class, the regular
`Model`. The `Model` can evaluate the function itself as well as its first and
second-order derivatives.

## Install

First ensure you have the required dependencies:
* a compiler with C++11 support
* [cmake](https://cmake.org/) version 3.12+
* [Boost](https://www.boost.org/) (tested with version 1.58)
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

The C++ tests require [Boost](https://www.boost.org/) (tested with version 1.58).

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

A basic example differentiating functions related to rigid body dynamics can be
found [here](https://github.com/adamheins/CppADCodeGenEigenPy-dynamics-example).

## License

MIT
