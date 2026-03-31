"""Build Cython extensions for MCTS.

Usage: python setup_puct.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "mcts._puct_cy",
        ["mcts/_puct_cy.pyx"],
    ),
    Extension(
        "mcts._mcts_cy",
        ["mcts/_mcts_cy.pyx"],
    ),
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
    }),
)
