#include "page_gen.h"
#include "barkeep/barkeep.h"
#include "datasets/factory.h"
#include <cairo/cairo.h>
#include <cmath>
#include <filesystem>
#include <format>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>

namespace fs = std::filesystem;
namespace bk = barkeep;
using namespace std::chrono_literals;

void generate_pages(std::filesystem::path target,
                    std::vector<DatasetS> datasets, int n, bool preload,
                    bool use_arc, float max_warp, bool imperfect_lines,
                    bool save_json) {
  std::cout << std::format("Generating {} pages", n) << std::endl;
  std::cout << std::format("Creating dirs...") << std::endl;
  fs::path ruled_dir = target / "ruled-pages";
  fs::path clean_dir = target / "clean-pages";
  fs::path labels_dir = target / "labels";
  fs::create_directories(ruled_dir);
  fs::create_directories(clean_dir);
  if (save_json) {
    fs::create_directories(labels_dir);
  }

  if (datasets.empty()) {
    throw std::invalid_argument("Datasets cannot be empty");
  }

  std::cout << std::format("Constructing datasets...") << std::endl;

  float total_weight = 0.0f;
  for (const auto &dataset : datasets) {
    total_weight += dataset.proportion;
  }
  // Normalize weights
  for (auto &dataset : datasets) {
    dataset.proportion = dataset.proportion / total_weight;
  }

  std::vector<std::unique_ptr<Dataset>> loaded;
  for (const auto &d : datasets) {
    auto dataset = make_dataset(d); // throws if unknown id
    if (!dataset->valid())
      throw std::invalid_argument("Invalid dataset path: " + d.path.string());
    loaded.push_back(std::move(dataset));
  }
  std::cout << std::format("Loading datasets...") << std::endl;
  for (const auto &d : loaded) {
    if (d->valid()) {
      d->load();
    } else {
      throw std::invalid_argument(std::format(
          "Dataset ID {}, path {} couldnt be loaded", d->id, d->path.string()));
    }
  }

  int work{0};
  auto bar = bk::ProgressBar(&work, {
                                        .total = n,
                                        .message = "Generating pages",
                                        .speed = 1.,
                                        .speed_unit = "page/s",
                                    });
  for (int i = 0; i < n; i++) {
    work++;
  }
  bar->done();
}