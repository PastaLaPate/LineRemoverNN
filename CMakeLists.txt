cmake_minimum_required(VERSION 3.15...3.27)
project(lineremovernn_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT SKBUILD)
  message(WARNING "Run via: pip install --no-build-isolation -ve .")
endif()

# --- Python + nanobind ---
find_package(Python 3.12
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# --- OpenCV (system) ---
find_package(OpenCV REQUIRED COMPONENTS core imgproc photo)

# --- Cairo ---
find_package(PkgConfig REQUIRED)
pkg_check_modules(CAIRO REQUIRED IMPORTED_TARGET cairo)

# --- NumPy ---
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

# --- Extension ---
nanobind_add_module(
  _lineremovernn_ext
  src/lineremovernn_ext/bindings.cpp
  src/lineremovernn_ext/page_gen.cpp
  src/lineremovernn_ext/datasets/iam.cpp
  src/lineremovernn_ext/datasets/factory.cpp
)

# Insert the stub generator right after creating the module
if (NOT CMAKE_CROSSCOMPILING)
  nanobind_add_stub(
    _lineremovernn_ext_stub
    MODULE _lineremovernn_ext  
    OUTPUT src/lineremovernn_ext/_lineremovernn_ext.pyi
    DEPENDS _lineremovernn_ext
  )
endif()

target_include_directories(_lineremovernn_ext PRIVATE
  src/lineremovernn_ext
  ${NUMPY_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(_lineremovernn_ext PRIVATE
  -Wno-deprecated-enum-enum-conversion
)

target_link_libraries(_lineremovernn_ext PRIVATE
  ${OpenCV_LIBS}
  PkgConfig::CAIRO
)

install(TARGETS _lineremovernn_ext LIBRARY DESTINATION lineremovernn)