#include "iam.h"
#include "../utils.hpp"
#include "datasets/datasets.h"
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/core/mat.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace cv;

void IAM::load() {
  std::ifstream WordsIndex(this->path / "words.txt");
  std::string line;

  auto start_time = std::chrono::high_resolution_clock::now();

  while (std::getline(WordsIndex, line)) {
    if (line.starts_with("#"))
      continue;
    std::vector<std::string> tokens = split_ws(line);
    if (tokens.size() != 9)
      continue;
    if (tokens[1] == "err")
      continue;

    std::vector<std::string_view> parts = split(tokens[0], '-');
    std::span<std::string_view> second_path = {parts.begin(), 2};
    std::string joined_parts = std::accumulate(
        std::next(second_path.begin()), second_path.end(),
        std::string(second_path[0]), // Explicitly start with a std::string
        [](std::string a, std::string_view b) {
          return std::move(a) + "-" + std::string(b);
        });
    std::filesystem::path img_path =
        this->path / "words" / parts[0] / joined_parts / (tokens[0] + ".png");
    this->words.push_back(
        {.path{img_path},
         .bbox{{parse_int(tokens[3]), parse_int(tokens[4]),
                parse_int(tokens[5]), parse_int(tokens[6])}},
         .transcript{tokens[8]},
         .gray_scale = static_cast<uint8_t>(parse_int(tokens[2]))});
  }

  WordsIndex.close();
  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;

  int parsed_count = this->words.size();
  if (parsed_count > 0) {
    double avg_time = duration_ms.count() / parsed_count;
    std::cout << std::format(
        "[IAM::load] Loaded {} words in {:.2f} ms ({:.4f} ms/word)\n",
        parsed_count, duration_ms.count(), avg_time);
  } else {
    std::cout << "[IAM::load] No words were loaded.\n";
  }
}

bool IAM::valid() {
  std::filesystem::path words_index_path = this->path / "words.txt";
  std::filesystem::path words_path = this->path / "words";
  return std::filesystem::exists(words_index_path) &&
         std::filesystem::is_regular_file(words_index_path) &&
         std::filesystem::exists(words_path) &&
         std::filesystem::is_directory(words_path);
}

long IAM::len() { return this->words.size(); }

cv::Mat IAM::load_and_process_disk(int idx) {
  if (idx < 0 || static_cast<size_t>(idx) >= this->words.size()) {
    return cv::Mat();
  }

  IAMWordEntry word = this->words[idx];
  cv::Mat img = cv::imread(word.path, IMREAD_GRAYSCALE);
  if (!img.empty()) {
    img.setTo(255, img > 160);
  }
  return img;
}

void IAM::evict_unlocked() {}

cv::Mat IAM::get_image(int idx) {
  IAMWordEntry word = this->words[idx];
  if (img.empty()) {
    return img;
  }
  return img;
}

AssetRow IAM::get_asset(int idx) {
  IAMWordEntry word = this->words[idx];
  std::ifstream f(word.path, std::ios::binary);
  std::vector<uchar> buf((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
  cv::Mat img = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
  if (img.empty()) {
    return {.idx = idx,
            .dataset = "iam",
            .image = img,
            .transcript = word.transcript};
  }
  img.setTo(255, img > 160);
  return {.idx = idx,
          .dataset = "iam",
          .image = img,
          .transcript = word.transcript};
}

std::array<int, 2> IAM::get_size(int idx) {
  IAMWordEntry word = this->words[idx];
  return {word.bbox[2], word.bbox[3]};
}