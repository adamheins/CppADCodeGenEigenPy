cmake_minimum_required(VERSION 3.12)

project(CppADCodeGenEigenPy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Note that cmake 3.12+ is required to find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)

include(FetchContent)

# get pybind11 from github
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.7.1
)
FetchContent_MakeAvailable(pybind11)

# get googletest from github
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# avoid installing googletest along with this project
option(INSTALL_GTEST "Enable installation of googletest." OFF)
FetchContent_MakeAvailable(googletest)

# generate the Python bindings
pybind11_add_module(${PROJECT_NAME} src/bindings.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include/)

# install the module containing the bindings
# install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# install project header files
install(DIRECTORY include/ DESTINATION include)

# testing
enable_testing()

add_executable(
  model_tests
  tests/cpp/basic_model_test.cpp
  tests/cpp/parameterized_model_test.cpp
  tests/cpp/math_functions_model_test.cpp
  tests/cpp/low_order_model_test.cpp
)
target_include_directories(model_tests PUBLIC include)
target_link_libraries(
  model_tests
  gtest_main
  dl
  ${Boost_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(model_tests)
