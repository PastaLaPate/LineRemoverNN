#include "page_gen.h"
#include "barkeep/barkeep.h"
#include "datasets/factory.h"
#include <cairo/cairo.h>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <format>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
namespace bk = barkeep;
using namespace std::chrono_literals;
using namespace cv;

Dataset *select_dataset(const std::vector<std::unique_ptr<Dataset>> &datasets) {
  float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

  for (const auto &d : datasets) {
    if (r < d->proportion) {
      return d.get();
    }
    r -= d->proportion;
  }
  return datasets[0].get();
}

void generate_pages(std::filesystem::path target,
                    std::vector<DatasetS> datasets, int n, bool preload,
                    bool use_arc, float max_warp, bool imperfect_lines,
                    bool save_json) {

  std::cout << std::format("Generating {} pages", n) << std::endl;
  std::cout << std::format("Creating dirs...") << std::endl;
  srand(time(0));
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
  std::map<std::string, int> offsets;
  for (auto const &d : loaded) {
    offsets[d->id] = rand() % d->len() + 1;
  }
  for (int i = 0; i < n; i++) {
    int w = rand() % 2000;
    int h = rand() % 2000;
    Mat clean = Mat::ones(w, h, CV_8UC1) * 255; // White page
    Point cursor = {0, 0};
    while (cursor.y + 100 < h) {   // 100 pixels margin down
      while (cursor.x + 100 < w) { // 100 pixels margin right
        auto d = select_dataset(loaded);
        int offset = offsets[d->id] % d->len();
        Mat img = d->get_image(offset);
        offsets[d->id] += 1;
        Size s = img.size();
      }
    }
    work++;
  }
  bar->done();
}