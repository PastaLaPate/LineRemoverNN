#include "page_gen.h"
#include "logging/python_logger.h"
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
          [](DatasetS *d, std::string id, std::string path, float proportion,
             bool preload, bool index) {
            new (d) DatasetS{std::move(id), std::move(path), proportion,
                             preload, index};
          },
          "id"_a, "path"_a, "proportion"_a = 1.0f, "preload"_a = false,
          "index"_a = false)
      .def_rw("id", &DatasetS::id)
      .def_rw("path", &DatasetS::path)
      .def_rw("proportion", &DatasetS::proportion);

  nb::enum_<BlockType>(m, "BLOCK_TYPES")
      .value("TITLE", BlockType::Title)
      .value("CAT_TITLE", BlockType::CatTitle)
      .value("PARAGRAPH", BlockType::Paragraph)
      .value("SCHEMA", BlockType::Schema)
      .value("SKIP_LINE", BlockType::SkipLine)
      .export_values();

  nb::class_<PageSettings>(m, "PageSettings")
      .def(nb::init<bool, bool, int, int, int, int, float, bool, bool>(),
           nb::arg("document"), nb::arg("save_labels"), nb::arg("w"),
           nb::arg("h"), nb::arg("line_height"), nb::arg("brightness"),
           nb::arg("max_warp"), nb::arg("imperfect_lines"), nb::arg("arc"))
      .def_ro("document", &PageSettings::document)
      .def_ro("save_labels", &PageSettings::save_labels)
      .def_ro("w", &PageSettings::w)
      .def_ro("h", &PageSettings::h)
      .def_ro("line_height", &PageSettings::line_height)
      .def_ro("brightness", &PageSettings::brightness)
      .def_ro("max_warp", &PageSettings::max_warp)
      .def_ro("imperfect_lines", &PageSettings::imperfect_lines)
      .def_ro("arc", &PageSettings::arc);

  nb::class_<PageAsset>(m, "PageAsset")
      .def(nb::init<std::string, int, int, int, int, int, int, float,
                    std::string>(),
           nb::arg("dataset_id"), nb::arg("idx"), nb::arg("page_idx"),
           nb::arg("w"), nb::arg("h"), nb::arg("x"), nb::arg("y"),
           nb::arg("scale"), nb::arg("transcript"))
      .def_ro("dataset_id", &PageAsset::dataset_id)
      .def_ro("idx", &PageAsset::idx)
      .def_ro("page_idx", &PageAsset::page_idx)
      .def_ro("w", &PageAsset::w)
      .def_ro("h", &PageAsset::h)
      .def_ro("x", &PageAsset::x)
      .def_ro("y", &PageAsset::y)
      .def_ro("scale", &PageAsset::scale)
      .def_ro("transcript", &PageAsset::transcript);

  nb::class_<LayoutBlock>(m, "LayoutBlock")
      .def(nb::init<BlockType, int, int, int, float, int,
                    std::vector<std::vector<PageAsset>>>(),
           nb::arg("type"), nb::arg("y_start"), nb::arg("height"),
           nb::arg("n_lines"), nb::arg("schema_x_offset"),
           nb::arg("line_skipped"), nb::arg("assets"))
      .def_ro("type", &LayoutBlock::type)
      .def_ro("y_start", &LayoutBlock::y_start)
      .def_ro("height", &LayoutBlock::height)
      .def_ro("n_lines", &LayoutBlock::n_lines)
      .def_ro("schema_x_offset", &LayoutBlock::schema_x_offset)
      .def_ro("line_skipped", &LayoutBlock::line_skipped)
      .def_ro("assets", &LayoutBlock::assets);

  m.def(
      "generate_pages",
      [](std::filesystem::path target, std::vector<DatasetS> datasets, int n,
         bool use_arc, bool document, float max_warp, bool imperfect_lines,
         bool save_xml, bool debug, int max_workers, nb::object logger) {
        PythonLoggerBridge logger_bridge(std::move(logger));
        {
          nb::gil_scoped_release release;
          generate_pages(std::move(target), std::move(datasets), n, use_arc,
                         document, max_warp, imperfect_lines, save_xml, debug,
                         max_workers, logger_bridge);
        }
      },
      "target"_a, "datasets"_a, "n"_a = 5, "use_arc"_a = true,
      "document"_a = true, "max_warp"_a = .1, "imperfect_lines"_a = true,
      "save_xml"_a = false, "debug"_a = false, "max_workers"_a = 0,
      "logger"_a = nb::none());
}