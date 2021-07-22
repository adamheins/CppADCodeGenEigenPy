cmake_minimum_required(VERSION 3.12)

project(CppADCodeGenEigenPy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Note that cmake 3.12+ is required to find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# find pybind11
add_subdirectory(deps/pybind11)

# generate the Python bindings
pybind11_add_module(${PROJECT_NAME} src/bindings.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include)

# install the module containing the bindings
install(TARGETS ${PROJECT_NAME} DESTINATION ${Python3_SITELIB})

# install header files
install(DIRECTORY include/ DESTINATION include)
