#!/bin/bash
set -e

echo "Adding nanobind"
uv add --dev nanobind scikit-build-core

echo "Building..."
uv pip install --no-build-isolation -e .

# Generate a separate compile_commands.json with stable paths using the venv python
PYTHON=$(uv run which python)
NANOBIND_INCLUDE=$(uv run python -c "import nanobind; print(nanobind.include_dir())")
NUMPY_INCLUDE=$(uv run python -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE=$(uv run python -c "import sysconfig; print(sysconfig.get_path('include'))")

echo "Compiling commands..."
cmake -S . -B build/ide \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPython_EXECUTABLE="$PYTHON"

echo "Generating stubs"
SO_DIR=$(find build/ -name "_lineremovernn_ext*.so" -exec dirname {} \; | head -n 1)

if [ -z "$SO_DIR" ]; then
    echo "Error: Could not find compiled _lineremovernn_ext binary in build directory."
    exit 1
fi

PYTHONPATH="$SO_DIR" uv run python -m nanobind.stubgen \
  -m _lineremovernn_ext \
  -o lineremovernn/_lineremovernn_ext.pyi

ln -sf build/ide/compile_commands.json compile_commands.json

echo "Done. Restart clangd/your IDE."