#!/bin/bash
# Rebuild all C++ extensions from scratch.
# Run from any directory — it finds the project root automatically.

cd "$(dirname "$0")/.."
rm -rf build/
python setup.py build_ext --inplace
