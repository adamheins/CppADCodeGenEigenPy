cmake_minimum_required(VERSION 3.12)

project(CppADCodeGenEigenPy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Note that cmake 3.12+ is required to find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)

# find pybind11
add_subdirectory(deps/pybind11)

# Get googletest from github
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
FetchContent_MakeAvailable(googletest)

# generate the Python bindings
pybind11_add_module(${PROJECT_NAME} src/bindings.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include)

# install the module containing the bindings
install(TARGETS ${PROJECT_NAME} DESTINATION ${Python3_SITELIB})

# install header files
install(DIRECTORY include/ DESTINATION include)

# testing
enable_testing()

add_executable(
  model_test
  test/model_test.cpp
)
target_include_directories(model_test PUBLIC include)
target_link_libraries(
  model_test
  gtest_main
  dl
  ${Boost_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(model_test)
