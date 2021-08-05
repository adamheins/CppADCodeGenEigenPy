# CppADCodeGenEigenPy

CppADCodeGen meets pybind11 with an Eigen interface.

This project has been tested on Ubuntu 16.04. It should work on more recent
Ubuntu versions, and likely works on other *nix systems as well.

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

First ensure you have the required dependencies:
* a compiler with C++11 support
* [cmake](https://cmake.org/) version 3.12+
* [Boost](https://www.boost.org/) (tested with version 1.58)
* [CppADCodeGen](https://github.com/joaoleal/CppADCodeGen)

Then clone and built CppADCodeGenEigenPy:
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

## Example

This is basic example that you can also find implemented in the `examples`
directory, along with some others.

TODO

## License

TODO
