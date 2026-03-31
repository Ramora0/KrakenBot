"""Build the C++ extension module.

Usage:
    python setup.py build_ext --inplace
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

common_args = ["-O3", "-march=native", "-DNDEBUG"]
cpp_dir = ["cpp"]

setup(
    name="hex_cpp_bot",
    ext_modules=[
        Pybind11Extension("ai_cpp", ["cpp/ai_cpp.cpp"],
                          cxx_std=17, extra_compile_args=common_args,
                          include_dirs=cpp_dir),
    ],
    cmdclass={"build_ext": build_ext},
)
