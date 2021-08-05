# CppADCodeGenEigenPy

CppADCodeGen meets pybind11 with an Eigen interface.

## Motivation

I want to be able to prototype code in Python while making use of fast,
compiled auto-differentiated code. Autodiff tools in Python often just-in-time
compile the code rather than ahead-of-time compile it. I don't want to wait for
autodiff's JIT every time I run the script if nothing has changed.

## How it works

This project provides a simple interface to CppADCodeGen to define the required
function in C++, which is then compiled (along with its automatically-computed
derivates) into a dynamic library. This dynamic library can then be easily used
from C++ or imported into Python.

## Install
```
git clone ...
cd CppADCodeGenEigenPy
mkdir build
cmake -S . -B build
cmake --build build

# headers are installed to the usual system-wide include location
# the python module is installed to whichever Python is active
sudo make install
# or
sudo cmake --install build

# run tests
cd build && ctest
```


