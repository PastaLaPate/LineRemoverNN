#!/bin/bash
set -e

# Install nanobind into the venv so paths are stable
uv add --dev nanobind scikit-build-core

# Build and install the extension
uv pip install --no-build-isolation -e .

# Generate a separate compile_commands.json with stable paths using the venv python
PYTHON=$(uv run which python)
NANOBIND_INCLUDE=$(uv run python -c "import nanobind; print(nanobind.include_dir())")
NUMPY_INCLUDE=$(uv run python -c "import numpy; print(numpy.get_include())")
PYTHON_INCLUDE=$(uv run python -c "import sysconfig; print(sysconfig.get_path('include'))")

cmake -S . -B build/ide \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPython_EXECUTABLE="$PYTHON"

ln -sf build/ide/compile_commands.json compile_commands.json

echo "Done. Restart clangd/your IDE."