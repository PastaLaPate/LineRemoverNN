#include "page_gen.h"
#include <cairo/cairo.h>
#include <cmath>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <stdexcept>

namespace fs = std::filesystem;

void generate_pages(std::filesystem::path target, std::vector<Dataset> datasets,
                    int n, bool preload, bool use_arc, float max_warp,
                    bool imperfect_lines, bool save_json) {
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

  float total_weight = 0.0f;
  for (const auto &dataset : datasets) {
    total_weight += dataset.proportion;
  }
  // Normalize weights
  for (auto &dataset : datasets) {
    dataset.proportion = dataset.proportion / total_weight;
  }
}