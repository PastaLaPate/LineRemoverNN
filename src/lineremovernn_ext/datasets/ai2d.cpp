#include "ai2d.h"
#include "datasets/datasets.h"
#include <filesystem>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/mat.hpp>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"

using namespace cv;

void AI2D::load() {
  std::filesystem::path images_path = this->path / "ai2d" / "images";
  std::filesystem::directory_iterator dir_iter =
      std::filesystem::directory_iterator(images_path);
  for (const auto &entry : dir_iter) {
    this->images.push_back(entry.path());
  }
}

bool AI2D::valid() {
  std::filesystem::path main_path = this->path / "ai2d";
  std::filesystem::path images_path = main_path / "images";
  return std::filesystem::exists(main_path) &&
         std::filesystem::is_directory(main_path) &&
         std::filesystem::exists(images_path) &&
         std::filesystem::is_directory(images_path);
}

long AI2D::len() { return this->images.size(); }

cv::Mat AI2D::get_image(int idx) {
  std::filesystem::path path = this->images[idx];
  cv::Mat img = cv::imread(path, IMREAD_GRAYSCALE);
  if (img.empty()) {
    return img;
  }
  return img;
}

AssetRow AI2D::get_asset(int idx) {
  cv::Mat img = this->get_image(idx);
  return {.idx = idx, .dataset = "ai2d", .image = img, .transcript = ""};
}

std::array<int, 2> AI2D::get_size(int idx) {
  const char *filename = this->images[idx].c_str();
  int width = 0;
  int height = 0;
  int channels = 0;

  if (stbi_info(filename, &width, &height, &channels)) {
    return {width, height};
  } else {
    std::cout
        << "Failed to parse image header. Formatter unsupported or corrupt.\n";
    return {0, 0};
  }
}