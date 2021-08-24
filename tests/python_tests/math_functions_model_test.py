import pytest
import numpy as np

from CppADCodeGenEigenPy import CompiledModel

BUILD_DIR_NAME = "build"
MODEL_NAME = "MathFunctionsTestModel"
MODEL_LIB_NAME = "lib" + MODEL_NAME

NUM_INPUT = 3
NUM_OUTPUT = NUM_INPUT


@pytest.fixture
def model(pytestconfig):
    lib_path = str(
        pytestconfig.rootdir / pytestconfig.getoption("builddir") / MODEL_LIB_NAME
    )
    return CompiledModel(MODEL_NAME, lib_path)


def test_model_evaluate(model):
    x = np.ones(NUM_INPUT)

    y_expected = np.array([np.sin(x[0]) * np.cos(x[1]), np.sqrt(x[2]), x.dot(x)])
    y_actual = model.evaluate(x)
    assert np.allclose(y_actual, y_expected)


def test_model_jacobian(model):
    x = np.ones(NUM_INPUT)

    # fmt: off
    J_expected = np.array([
        [np.cos(x[0]) * np.cos(x[1]), -np.sin(x[0]) * np.sin(x[1]), 0],
        [0, 0, 0.5 / np.sqrt(x[2])],
        [2 * x[0], 2 * x[1], 2 * x[2]],
    ])
    # fmt: on

    J_actual = model.jacobian(x)
    assert np.allclose(J_actual, J_expected)


def test_model_hessian(model):
    x = np.ones(NUM_INPUT)

    # fmt: off
    H0_expected = np.array([
        [-np.sin(x[0]) * np.cos(x[1]), -np.cos(x[0]) * np.sin(x[1]), 0],
        [-np.cos(x[0]) * np.sin(x[1]), -np.sin(x[0]) * np.cos(x[1]), 0],
        [0, 0, 0]
    ])
    # fmt: on
    H1_expected = np.zeros((NUM_INPUT, NUM_INPUT))
    H1_expected[2, 2] = -0.25 * x[2] ** -1.5

    H2_expected = 2 * np.eye(NUM_INPUT)

    H0_actual = model.hessian(x, 0)
    H1_actual = model.hessian(x, 1)
    H2_actual = model.hessian(x, 2)

    assert np.allclose(H0_actual, H0_expected)
    assert np.allclose(H1_actual, H1_expected)
    assert np.allclose(H2_actual, H2_expected)
