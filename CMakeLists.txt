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

file(GLOB_RECURSE EXT_SOURCES CONFIGURE_DEPENDS
  "src/lineremovernn_ext/*.cpp"
)

nanobind_add_module(
  _lineremovernn_ext
  NOMINSIZE
  ${EXT_SOURCES}
)

target_include_directories(_lineremovernn_ext PRIVATE
  src/lineremovernn_ext
  ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(_lineremovernn_ext PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-deprecated-enum-enum-conversion>
)

target_link_libraries(_lineremovernn_ext PRIVATE
  ${OpenCV_LIBS}
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
