#include "mathwriting.h"
#include "../pugixml/pugixml.hpp"
#include "../utils.hpp"
#include <algorithm>
#include <cairo.h>
#include <filesystem>
#include <format>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/mat.hpp>
#include <random>
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
  static thread_local std::mt19937 rng(std::random_device{}());
  auto rand_int = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };
  std::filesystem::path asset_path = this->assets[idx];
  pugi::xml_document doc;
  pugi::xml_parse_result result = doc.load_file(asset_path.c_str());
  if (!result) {
    std::cerr << "Failed to load asset: " << asset_path
              << "\nError description: " << result.description() << "\n";
    return cv::Mat();
  }
  // Parse the XML and extract strokes data

  std::vector<std::vector<cv::Point>> strokes;
  for (pugi::xml_node node : doc.child("ink").children("trace")) {
    std::string mutable_text = node.text().get();
    std::vector<std::string_view> points =
        split(strstrip(mutable_text.data()), ',');
    std::vector<cv::Point> sub_strokes;
    for (auto &point : points) {
      std::vector<std::string_view> coords = split(point, ' ');
      if (coords.size() != 3) {
        std::cout << std::format(
            "Warning: Skipping malformed point \"{}\" in asset {}\n", point,
            asset_path.filename().string());
        continue;
      }
      int x = parse_int(coords[0]);
      int y = parse_int(coords[1]);
      sub_strokes.emplace_back(x, y);
    }
    if (!sub_strokes.empty()) {
      strokes.push_back(std::move(sub_strokes));
    }
  }

  // Compute extreme stroke limits for spatial box tracking
  int xmin = std::numeric_limits<int>::max();
  int ymin = std::numeric_limits<int>::max();
  int xmax = std::numeric_limits<int>::min();
  int ymax = std::numeric_limits<int>::min();

  for (const auto &stroke : strokes) {
    for (const auto &pt : stroke) {
      xmin = std::min(xmin, pt.x);
      ymin = std::min(ymin, pt.y);
      xmax = std::max(xmax, pt.x);
      ymax = std::max(ymax, pt.y);
    }
  }

  // Fallback case if for some reason no valid points were found
  if (strokes.empty() || xmin > xmax || ymin > ymax) {
    xmin = ymin = xmax = ymax = 0;
  }

  int margin = 6;
  int stroke_width = rand_int(3, 7);
  int pen_darkness = rand_int(190, 240);

  int w = std::max(1, xmax - xmin + 2 * margin);
  int h = std::max(1, ymax - ymin + 2 * margin);

  int shift_x = margin - xmin;
  int shift_y = margin - ymin;

  auto surface = cairo_image_surface_create(CAIRO_FORMAT_A8, w, h);
  auto context = cairo_create(surface);
  cairo_set_source_rgba(context, 0, 0, 0, 1);
  cairo_set_operator(context, CAIRO_OPERATOR_SOURCE);
  cairo_paint(context);

  cairo_set_source_rgba(context, 0, 0, 0, 1.0 - pen_darkness / 255.0);
  cairo_set_line_width(context, stroke_width);
  cairo_set_line_cap(context, CAIRO_LINE_CAP_ROUND);
  cairo_set_line_join(context, CAIRO_LINE_JOIN_ROUND);
  cairo_set_operator(context, CAIRO_OPERATOR_SOURCE);

  for (const auto &stroke : strokes) {
    int n_points = stroke.size();
    if (n_points == 1) {
      cairo_arc(context, stroke[0].x + shift_x, stroke[0].y + shift_y,
                stroke_width / 2.0, 0, 2 * CV_PI);
      cairo_fill(context);
    } else {
      cairo_move_to(context, stroke[0].x + shift_x, stroke[0].y + shift_y);
      for (int i = 1; i < n_points; ++i) {
        cairo_line_to(context, stroke[i].x + shift_x, stroke[i].y + shift_y);
      }
      cairo_stroke(context);
    }
  }
  auto data = cairo_image_surface_get_data(surface);
  int stride = cairo_image_surface_get_stride(surface);
  cv::Mat img(h, w, CV_8UC1, data, stride);
  cv::Mat final_img = img.clone();
  cairo_destroy(context);
  cairo_surface_destroy(surface);

  /*std::cout << std::format(
      "[MathWriting::get_image] Generated image from asset {} with "
      "original bbox [({}, {}), ({}, {})], final size ({}, {})\n",
      asset_path.filename().string(), xmin, ymin, xmax, ymax, w, h);*/
  return final_img;
}