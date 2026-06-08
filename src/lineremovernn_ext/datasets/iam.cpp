#include "iam.h"
#include <charconv>
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
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace cv;

std::vector<std::string> split_ws(const std::string &s) {
  std::istringstream ss(s);
  return {std::istream_iterator<std::string>(ss), {}};
}

std::vector<std::string_view> split(std::string_view s, char delim) {
  std::vector<std::string_view> tokens;
  size_t start = 0, pos;
  while ((pos = s.find(delim, start)) != std::string_view::npos) {
    tokens.emplace_back(s.substr(start, pos - start));
    start = pos + 1;
  }
  tokens.emplace_back(s.substr(start));
  return tokens;
}

// Parse int — from_chars is the modern, fast, no-exception way
int parse_int(std::string_view s) {
  int result;
  auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), result);
  if (ec != std::errc{})
    throw std::invalid_argument("Invalid int: \"" + std::string(s) + "\"");
  return result;
}

void IAM::load() {
  std::ifstream WordsIndex(this->path / "words.txt");
  std::string line;

  auto start_time = std::chrono::high_resolution_clock::now();
  size_t parsed_count = 0;

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
    parsed_count++;
  }

  WordsIndex.close();
  auto end_time = std::chrono::high_resolution_clock::now();

  // 3. Calculate durations
  std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;

  // 4. Print the statistics
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

cv::Mat IAM::get_image(int idx) {
  IAMWordEntry word = this->words[idx];
  cv::Mat img = cv::imread(word.path);

  cv::Rect roi(word.bbox[0], word.bbox[1], word.bbox[2], word.bbox[3]);

  return img(roi).clone(); // Clone to apply
}