cmake_minimum_required(VERSION 3.22)
project(lineremovernn_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT SKBUILD)
  message(WARNING "Huh weird build ?")
endif()

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS core imgproc photo imgcodecs)

find_package(PkgConfig REQUIRED)
pkg_check_modules(CAIRO REQUIRED IMPORTED_TARGET cairo)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

nanobind_add_module(
  _lineremovernn_ext
  NOMINSIZE
  src/lineremovernn_ext/bindings.cpp
  src/lineremovernn_ext/page_gen.cpp
  src/lineremovernn_ext/datasets/iam.cpp
  src/lineremovernn_ext/datasets/mathwriting.cpp
  src/lineremovernn_ext/datasets/ai2d.cpp
  src/lineremovernn_ext/datasets/factory.cpp
  src/lineremovernn_ext/utils.hpp
  src/lineremovernn_ext/pugixml/pugiconfig.hpp
  src/lineremovernn_ext/pugixml/pugixml.hpp
  src/lineremovernn_ext/pugixml/pugixml.cpp
)

target_include_directories(_lineremovernn_ext PRIVATE
  src/lineremovernn_ext
  ${NUMPY_INCLUDE_DIR}
)

target_compile_options(_lineremovernn_ext PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-deprecated-enum-enum-conversion>
)

target_link_libraries(_lineremovernn_ext PRIVATE
  opencv_core
  opencv_imgproc
  opencv_photo
  opencv_imgcodecs
  PkgConfig::CAIRO
)

if(NOT CMAKE_CROSSCOMPILING)
  nanobind_add_stub(
    _lineremovernn_ext_stub
    ALL
    MODULE _lineremovernn_ext
    OUTPUT ${CMAKE_SOURCE_DIR}/lineremovernn/_lineremovernn_ext.pyi
    DEPENDS _lineremovernn_ext
  )
endif()

install(TARGETS _lineremovernn_ext DESTINATION lineremovernn)
