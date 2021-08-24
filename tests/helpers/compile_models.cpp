#include <iostream>
#include <string>

#include "testing/models/BasicTestModel.h"
#include "testing/models/ParameterizedTestModel.h"
#include "testing/models/MathFunctionsTestModel.h"

using namespace CppADCodeGenEigenPy;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Directory path is required." << std::endl;
        return 1;
    }
    std::string directory_path = argv[1];
    BasicModelTest::BasicTestModel<double>().compile(
        BasicModelTest::MODEL_NAME, directory_path, DerivativeOrder::Second,
        /* verbose = */ true);
    BasicModelTest::BasicTestModel<double>().compile(
        "LowOrderTestModel", directory_path, DerivativeOrder::Zero,
        /* verbose = */ true);
    ParameterizedModelTest::ParameterizedTestModel<double>().compile(
        ParameterizedModelTest::MODEL_NAME, directory_path,
        DerivativeOrder::Second,
        /* verbose = */ true);
    MathFunctionsModelTest::MathFunctionsTestModel<double>().compile(
        MathFunctionsModelTest::MODEL_NAME, directory_path,
        DerivativeOrder::Second,
        /* verbose = */ true);
}
