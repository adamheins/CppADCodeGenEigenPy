#!/usr/bin/env python

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


with open("README.md") as f:
    long_description = f.read()


ext_modules = [
    Pybind11Extension(
        "CppADCodeGenEigenPy",
        ["src/bindings.cpp"],
        include_dirs=["include"],
    ),
]

setup(
    name="CppADCodeGenEigenPy",
    version="0.1",
    description="CppADCodeGen with an Eigen interface and Python bindings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Adam Heins",
    author_email="mail@adamheins.com",
    install_requires=["numpy"],
    extras_require={"test": "pytest"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    license="MIT",
    zip_safe=False,
)
