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

# testing
enable_testing()

add_executable(
  model_tests
  tests/cpp_tests/BasicModelTest.cpp
  # tests/cpp_tests/parameterized_model_test.cpp
  # tests/cpp_tests/math_functions_model_test.cpp
  # tests/cpp_tests/low_order_model_test.cpp
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
