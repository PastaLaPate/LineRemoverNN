#include "page_gen.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_lineremovernn_ext, m) {
  m.doc() =
      "High-performance page generation and line removal (OpenCV + Cairo)";

  m.def("generate_pages", &generate_pages);
}