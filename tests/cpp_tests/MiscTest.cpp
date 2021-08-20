#include <gtest/gtest.h>

#include <CppADCodeGenEigenPy/CompiledModel.h>

namespace CppADCodeGenEigenPy {
namespace MiscModelTest {

TEST(MiscModelTest, NonexistentLib) {
    EXPECT_THROW(CompiledModel<double> model("FakeModel", "nonexistent_path"),
                 std::runtime_error)
        << "No throw when trying to load non-existent library.";
}

}  // namespace MiscModelTest
}  // namespace CppADCodeGenEigenPy
