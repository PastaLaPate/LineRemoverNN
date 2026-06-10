#include "mathwriting.h"
#include "../utils.hpp"
#include <filesystem>
#include <format>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/mat.hpp>
#include <string>
#include <vector>

using namespace cv;

void MathWriting::load() {
  // std::ifstream WordsIndex(this->path / "words.txt");
  std::string line;

  auto start_time = std::chrono::high_resolution_clock::now();
  size_t parsed_count = 0;
  std::vector<std::filesystem::path> target_dirs = {this->path / "train",
                                                    this->path / "synthetic"};
  for (const auto &dir : target_dirs) {
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".inkml") {
        this->assets.push_back(entry.path());
        parsed_count++;
      }
    }
  }
  // WordsIndex.close();
  auto end_time = std::chrono::high_resolution_clock::now();

  // 3. Calculate durations
  std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;

  // 4. Print the statistics
  if (parsed_count > 0) {
    double avg_time = duration_ms.count() / parsed_count;
    std::cout << std::format(
        "[MathWriting::load] Loaded {} assets in {:.2f} ms ({:.4f} ms/asset)\n",
        parsed_count, duration_ms.count(), avg_time);
  } else {
    std::cout << "[MathWriting::load] No assets were loaded.\n";
  }
}

bool MathWriting::valid() {
  std::filesystem::path train_path = this->path / "train";
  std::filesystem::path synth_path = this->path / "synthetic";
  return std::filesystem::exists(train_path) &&
         std::filesystem::is_directory(train_path) &&
         std::filesystem::exists(synth_path) &&
         std::filesystem::is_directory(synth_path);
}

long MathWriting::len() { return this->assets.size(); }

cv::Mat MathWriting::get_image(int idx) {
  std::filesystem::path asset_path = this->assets[idx];
}