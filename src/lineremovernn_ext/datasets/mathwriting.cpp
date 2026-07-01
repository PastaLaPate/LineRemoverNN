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
  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<std::filesystem::path> target_dirs = {this->path / "train",
                                                    this->path / "synthetic"};
  for (const auto &dir : target_dirs) {
    if (!std::filesystem::exists(dir))
      continue;
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".inkml") {
        this->assets.push_back(entry.path());
      }
    }
  }

  this->parsed_cache.resize(this->assets.size());

  // parse all XML data
  for (size_t i = 0; i < this->assets.size(); ++i) {
    const auto &asset_path = this->assets[i];
    ParsedAsset &parsed = this->parsed_cache[i];

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(asset_path.c_str());
    if (!result) {
      std::cerr << "Failed to load asset: " << asset_path
                << "\nError description: " << result.description() << "\n";
      continue;
    }

    // cache transcript
    for (pugi::xml_node node : doc.child("ink").children("annotation")) {
      if (strcmp(node.attribute("type").value(), "normalizedLabel") == 0) {
        parsed.transcript = node.text().get();
        break;
      }
    }

    // cache xmin, ymin, xmax, ymax and strokes
    parsed.xmin = std::numeric_limits<int>::max();
    parsed.ymin = std::numeric_limits<int>::max();
    parsed.xmax = std::numeric_limits<int>::min();
    parsed.ymax = std::numeric_limits<int>::min();

    for (pugi::xml_node node : doc.child("ink").children("trace")) {
      std::string mutable_text = node.text().get();
      std::vector<std::string_view> points =
          split(strstrip(mutable_text.data()), ',');
      std::vector<cv::Point> sub_strokes;

      for (const auto &point : points) {
        std::vector<std::string_view> coords = split(point, ' ');
        if (coords.size() != 3) {
          continue;
        }
        int x = parse_int(coords[0]);
        int y = parse_int(coords[1]);
        sub_strokes.emplace_back(x, y);

        parsed.xmin = std::min(parsed.xmin, x);
        parsed.ymin = std::min(parsed.ymin, y);
        parsed.xmax = std::max(parsed.xmax, x);
        parsed.ymax = std::max(parsed.ymax, y);
      }
      if (!sub_strokes.empty()) {
        parsed.strokes.push_back(std::move(sub_strokes));
      }
    }

    // Handle edge fallback layout edge cases cleanly
    if (parsed.strokes.empty() || parsed.xmin > parsed.xmax ||
        parsed.ymin > parsed.ymax) {
      parsed.xmin = parsed.ymin = parsed.xmax = parsed.ymax = 0;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;

  if (!this->assets.empty()) {
    double avg_time = duration_ms.count() / this->assets.size();
    std::cout << std::format("[MathWriting::load] Loaded and parsed {} assets "
                             "in {:.2f} ms ({:.4f} ms/asset)\n",
                             this->assets.size(), duration_ms.count(),
                             avg_time);
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

cv::Mat MathWriting::get_image(int idx) { return this->get_asset(idx).image; }
AssetRow MathWriting::get_asset(int idx) {
  static thread_local std::mt19937 rng(std::random_device{}());
  auto rand_int = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };

  // get directly from cache
  const auto &parsed = this->parsed_cache[idx];

  int margin = 6;
  int w = std::max(1, parsed.xmax - parsed.xmin + 2 * margin);
  int h = std::max(1, parsed.ymax - parsed.ymin + 2 * margin);
  int target = 256;
  int min_h = 128;
  if (h / w > 2) {
    target = 512;
  }

  float scale = static_cast<float>(target) / std::max(w, h);
  if (h * scale < min_h) {
    scale = static_cast<float>(min_h) / h;
  }
  float target_stroke_min = 4.0f, target_stroke_max = 7.0f;
  int stroke_width =
      rand_int(std::max(1, static_cast<int>(target_stroke_min / scale)),
               std::max(2, static_cast<int>(target_stroke_max / scale)));

  int pen_darkness = rand_int(230, 255);

  int shift_x = margin - parsed.xmin;
  int shift_y = margin - parsed.ymin;

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

  for (const auto &stroke : parsed.strokes) {
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
  int new_w = static_cast<int>(w * scale);
  int new_h = static_cast<int>(h * scale);

  cv::Mat scaled;
  cv::resize(final_img, scaled, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);

  int canvas_w = std::max(target, new_w);
  int canvas_h = std::max(target, new_h);

  cv::Mat padded(canvas_h, canvas_w, CV_8UC1, cv::Scalar(255));
  int x_offset = (canvas_w - new_w) / 2;
  int y_offset = (canvas_h - new_h) / 2;
  scaled.copyTo(padded(cv::Rect(x_offset, y_offset, new_w, new_h)));

  return {.idx = idx,
          .dataset = "mathwriting",
          .image = padded,
          .transcript = parsed.transcript};
}

std::array<int, 2> MathWriting::get_size(int idx) {
  // also cached
  const auto &parsed = this->parsed_cache[idx];
  constexpr int margin = 6;
  int w = std::max(1, parsed.xmax - parsed.xmin + 2 * margin);
  int h = std::max(1, parsed.ymax - parsed.ymin + 2 * margin);
  return {w, h};
}