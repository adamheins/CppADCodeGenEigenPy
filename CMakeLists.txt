cmake_minimum_required(VERSION 3.12)

project(CppADCodeGenEigenPy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Note that cmake 3.12+ is required to find Python
# find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)

# install project header files
install(DIRECTORY include/ DESTINATION include)

include(FetchContent)

# get googletest from github
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# avoid installing googletest along with this project
option(INSTALL_GTEST "Enable installation of googletest." OFF)
FetchContent_MakeAvailable(googletest)

# build compiler for models used for the Python tests, then run it
add_executable(
  python_test_model_compiler
  tests/python_tests/compile_models.cpp
)
target_include_directories(python_test_model_compiler PUBLIC include tests/include)
target_link_libraries(
  python_test_model_compiler
  dl
)
add_custom_target(
  compile_python_test_models ALL
  COMMAND python_test_model_compiler ${CMAKE_CURRENT_SOURCE_DIR}/tests/python_tests
  COMMENT "Compiling models for Python tests"
  VERBATIM
)

# testing
enable_testing()

add_executable(
  model_tests
  tests/cpp_tests/MiscTest.cpp
  tests/cpp_tests/BasicModelTest.cpp
  tests/cpp_tests/ParameterizedModelTest.cpp
  tests/cpp_tests/MathFunctionsModelTest.cpp
  tests/cpp_tests/LowOrderModelTest.cpp
)
target_include_directories(model_tests PUBLIC include tests/include)
target_link_libraries(
  model_tests
  gtest_main
  dl
  ${Boost_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(model_tests)
