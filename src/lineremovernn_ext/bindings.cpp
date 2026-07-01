#include "page_gen.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_lineremovernn_ext, m) {
  m.doc() =
      "High-performance page generation and line removal (OpenCV + Cairo)";

  nb::class_<DatasetS>(m, "Dataset")
      .def(
          "__init__",
          [](DatasetS *d, std::string id, std::string path, float proportion) {
            new (d) DatasetS{std::move(id), std::move(path), proportion};
          },
          "id"_a, "path"_a, "proportion"_a = 1.0f)
      .def_rw("id", &DatasetS::id)
      .def_rw("path", &DatasetS::path)
      .def_rw("proportion", &DatasetS::proportion);

  m.def("generate_pages", &generate_pages, "target"_a, "datasets"_a, "n"_a = 5,
        "preload"_a = false, "use_arc"_a = true, "document"_a = true,
        "max_warp"_a = .1, "imperfect_lines"_a = true, "save_xml"_a = false,
        nb::call_guard<nb::gil_scoped_release>());
}