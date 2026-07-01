#include "page_gen.h"
#include "barkeep/barkeep.h"
#include "datasets/datasets.h"
#include "datasets/factory.h"
#include "pugixml/pugixml.hpp"
#include <algorithm>
#include <atomic>
#include <cairo/cairo.h>
#include <cassert>
#include <cmath>
#include <csignal>
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
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
namespace bk = barkeep;
using namespace std::chrono_literals;
using namespace cv;

enum class BlockType { Title, CatTitle, Paragraph, Schema, SkipLine };

struct PageSettings {
  bool document;
  bool save_labels;
  int w;
  int h;
  int line_height;
  int brightness; // per page

  float max_warp;

  // Lines params
  bool imperfect_lines;
  bool arc;
};

struct PageAsset {
  int idx;
  int page_idx; // global asset index, from left to right, top to bottom
  std::string dataset_id;
  int w, h, x, y;
  float scale;

  // metadata
  std::string transcript;
};

struct LayoutBlock {
  BlockType type;
  int y_start, height;   // Common Params in pixels
  int n_lines;           // Paragraph Param
  float schema_x_offset; // Schema 0.0 = left, .5 = center, 1.0 = right
  int line_skipped;      // SkipLine
  std::vector<std::vector<PageAsset>> assets;
};

std::atomic<bool> shutdown_requested(false);

void signal_handler(int signal) {
  if (signal == SIGINT) {
    shutdown_requested = true;
  }
}

Dataset *get_random_dataset(
    const std::map<DatasetType, std::vector<std::unique_ptr<Dataset>>>
        &datasets_by_type,
    std::initializer_list<DatasetType> types, std::mt19937 &rng) {
  std::vector<Dataset *> candidates;
  for (auto type : types) {
    for (const auto &d : datasets_by_type.at(type)) {
      candidates.push_back(d.get());
    }
  }
  if (candidates.empty())
    throw std::runtime_error("No datasets for requested types");
  std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
  return candidates[dist(rng)];
}

void add_random_perspective(const Mat &img, Mat &transformed, float max_warp,
                            int target_height, std::mt19937 &rng) {
  if (max_warp <= 0.0f) {
    float scale = static_cast<float>(target_height) / img.rows;
    int target_width = std::max(1, static_cast<int>(img.cols * scale));
    resize(img, transformed, Size(target_width, target_height), 0, 0,
           INTER_LINEAR);
    return;
  }

  int width = img.cols;
  int height = img.rows;

  std::vector<Point2f> src_points = {
      Point2f(0.0f, 0.0f), Point2f(static_cast<float>(width), 0.0f),
      Point2f(static_cast<float>(width), static_cast<float>(height)),
      Point2f(0.0f, static_cast<float>(height))};

  float max_dx = width * max_warp;
  float max_dy = height * max_warp;

  std::uniform_real_distribution<float> dist_x(-max_dx, max_dx);
  std::uniform_real_distribution<float> dist_y(-max_dy, max_dy);

  std::vector<Point2f> dst_points = {
      Point2f(dist_x(rng), dist_y(rng)),
      Point2f(width + dist_x(rng), dist_y(rng)),
      Point2f(width + dist_x(rng), height + dist_y(rng)),
      Point2f(dist_x(rng), height + dist_y(rng))};

  // Compute the perspective transformation matrix (returns a 3x3 CV_64F Mat)
  Mat matrix = getPerspectiveTransform(src_points, dst_points);

  float min_x = std::min(
      {dst_points[0].x, dst_points[1].x, dst_points[2].x, dst_points[3].x});
  float max_x = std::max(
      {dst_points[0].x, dst_points[1].x, dst_points[2].x, dst_points[3].x});
  float min_y = std::min(
      {dst_points[0].y, dst_points[1].y, dst_points[2].y, dst_points[3].y});
  float max_y = std::max(
      {dst_points[0].y, dst_points[1].y, dst_points[2].y, dst_points[3].y});

  int new_width = std::max(1, static_cast<int>(max_x - min_x));
  int new_height = std::max(1, static_cast<int>(max_y - min_y));

  // Shift the matrix to (0,0)
  matrix.row(0) -= min_x * matrix.row(2);
  matrix.row(1) -= min_y * matrix.row(2);

  float scale;
  if (target_height > 0) {
    scale = static_cast<float>(target_height) / new_height;
    matrix.row(0) *= scale;
    matrix.row(1) *= scale;
  }

  int target_width = target_height > 0
                         ? std::max(1, static_cast<int>(new_width * scale))
                         : new_width;

  Scalar border_val;
  if (img.channels() == 1) {
    border_val = Scalar(255);
  } else {
    border_val = Scalar(255, 255, 255, 255);
  }

  warpPerspective(img, transformed, matrix, Size(target_width, target_height),
                  INTER_LINEAR, BORDER_CONSTANT, border_val);
}

void draw_lines(Mat &img, bool use_arc, bool imperfect_lines,
                std::mt19937 &rng) {
  int W = img.cols;
  int H = img.rows;

  std::uniform_int_distribution<int> line_spacing_dist(45, 100);
  std::uniform_int_distribution<int> sub_line_spacing_dist(2, 5);
  std::uniform_int_distribution<int> rand_color(100, 180);
  std::uniform_int_distribution<int> rand_sub_color(160, 190);
  std::uniform_int_distribution<int> rand_lw(1, 3);
  std::uniform_int_distribution<int> rand_sub_lw(1, 2);
  std::uniform_int_distribution<int> rand_jitter(-3, 3);
  std::uniform_real_distribution<float> rand_amp(-15.0f, 15.0f);

  int line_spacing = line_spacing_dist(rng);
  int sub = sub_line_spacing_dist(rng);
  int margin_top = line_spacing;
  int margin_left = line_spacing * 2;
  int n_lines = (H - margin_top) / line_spacing;

  std::vector<float> pi_t_vals_h;
  std::vector<Point> pts_buffer_h;

  if (use_arc) {
    int step_h = std::max(1, W / 120);
    for (int x = 0; x < W; x += step_h) {
      pi_t_vals_h.push_back((static_cast<float>(x) / static_cast<float>(W)) *
                            CV_PI);
      pts_buffer_h.push_back(Point(x, 0));
    }
    if (pts_buffer_h.empty() || pts_buffer_h.back().x != W - 1) {
      pi_t_vals_h.push_back(CV_PI);
      pts_buffer_h.push_back(Point(W - 1, 0));
    }
  }

  std::vector<float> pi_t_vals_v;
  std::vector<Point> pts_buffer_v;

  if (use_arc) {
    int step_v = std::max(1, H / 120);
    for (int y = 0; y < H; y += step_v) {
      pi_t_vals_v.push_back((static_cast<float>(y) / static_cast<float>(H)) *
                            CV_PI);
      pts_buffer_v.push_back(Point(0, y));
    }
    if (pts_buffer_v.empty() || pts_buffer_v.back().y != H - 1) {
      pi_t_vals_v.push_back(CV_PI);
      pts_buffer_v.push_back(Point(0, H - 1));
    }
  }

  for (int x_v = 0; x_v < W; x_v += line_spacing) {
    int darkness = rand_color(rng);
    int lw = std::max(1, rand_lw(rng) - 1);

    if (use_arc) {
      float amplitude = rand_amp(rng);
      for (size_t p = 0; p < pts_buffer_v.size(); ++p) {
        float x_calculated =
            static_cast<float>(x_v) + amplitude * std::sin(pi_t_vals_v[p]);
        pts_buffer_v[p].x = cvRound(x_calculated);
      }
      if (pts_buffer_v.size() >= 2) {
        polylines(img, pts_buffer_v, false, Scalar(darkness), lw, LINE_AA);
      }
    } else {
      line(img, Point(x_v, 0), Point(x_v, H), Scalar(darkness), lw);
    }
  }

  for (int i = 0; i <= n_lines; ++i) {
    int y_group = margin_top + i * line_spacing;
    float sub_step = static_cast<float>(line_spacing) / sub;

    for (int j = 0; j < sub; ++j) {
      float y_base = y_group + j * sub_step;
      bool is_main = (j == 0 || j == sub - 1);

      int darkness = is_main ? rand_color(rng) : rand_sub_color(rng);
      int lw = is_main ? rand_lw(rng) : rand_sub_lw(rng);
      int y_off = rand_jitter(rng);

      if (use_arc) {
        float amplitude = rand_amp(rng);
        for (size_t p = 0; p < pts_buffer_h.size(); ++p) {
          float y_calculated =
              y_base + y_off + amplitude * std::sin(pi_t_vals_h[p]);
          pts_buffer_h[p].y = cvRound(y_calculated);
        }
        if (pts_buffer_h.size() >= 2) {
          polylines(img, pts_buffer_h, false, Scalar(darkness), lw, LINE_AA);
        }
      } else {
        int final_y = static_cast<int>(y_base) + y_off;
        line(img, Point(0, final_y), Point(W, final_y), Scalar(darkness), lw);
      }
    }
  }

  int margin_darkness = rand_color(rng) + 20;
  int margin_lw = rand_lw(rng) + 1;
  if (use_arc) {
    float amplitude = rand_amp(rng);
    for (size_t p = 0; p < pts_buffer_v.size(); ++p) {
      float x_calculated = static_cast<float>(margin_left) +
                           amplitude * std::sin(pi_t_vals_v[p]);
      pts_buffer_v[p].x = cvRound(x_calculated);
    }
    if (pts_buffer_v.size() >= 2) {
      polylines(img, pts_buffer_v, false, Scalar(margin_darkness), margin_lw,
                LINE_AA);
    }
  } else {
    line(img, Point(margin_left, 0), Point(margin_left, H),
         Scalar(margin_darkness), margin_lw);
  }

  if (imperfect_lines) {
    std::uniform_int_distribution<int> rand_dots(40, 120);
    std::uniform_int_distribution<int> rand_x(0, W - 1);
    std::uniform_int_distribution<int> rand_y(0, H - 1);
    std::uniform_int_distribution<int> rand_r(1, 4);
    int num_spots = rand_dots(rng);
    for (int k = 0; k < num_spots; ++k) {
      circle(img, Point(rand_x(rng), rand_y(rng)), rand_r(rng), Scalar(255),
             FILLED);
    }
  }
}

std::vector<LayoutBlock> generate_document_layout(PageSettings settings,
                                                  std::mt19937 &rng) {
  std::vector<LayoutBlock> blocks;

  auto rand_int = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };

  auto rand_float = [&](float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(rng);
  };

  int top_margin = rand_int(30, std::min(settings.h, 100));
  int y = top_margin;

  if (rand_float(0.f, 1.f) < 0.66f) {
    float title_scale = rand_float(1.2f, 1.5f);
    int title_h = static_cast<int>(settings.line_height * title_scale);

    blocks.push_back({.type = BlockType::Title,
                      .y_start = y,
                      .height = title_h,
                      .n_lines = 1});
    y += title_h;

    if (y + settings.line_height * 2 <= settings.h) {
      blocks.push_back(
          {.type = BlockType::SkipLine, .y_start = y, .line_skipped = 2});
      y += settings.line_height * 2;
    }
  }

  BlockType last_type = BlockType::SkipLine;

  while (y + settings.line_height <= settings.h) {

    BlockType current_type;
    while (true) {
      float choice = rand_float(0.f, 1.f);
      if (choice < 0.1f) {
        if (last_type != BlockType::CatTitle) {
          current_type = BlockType::CatTitle;
          break;
        }
      } else if (choice < 0.9f) { // 0.1 + 0.8 = 0.8
        current_type = BlockType::Paragraph;
        break;
      } else { // Remaining 0.1
        current_type = BlockType::Schema;
        break;
      }
    }

    if (current_type == BlockType::CatTitle) {
      float cat_scale = rand_float(1.3f, 2.0f);
      int cat_h = static_cast<int>(settings.line_height * cat_scale);

      if (y + cat_h > settings.h)
        break;

      blocks.push_back({.type = BlockType::CatTitle,
                        .y_start = y,
                        .height = cat_h,
                        .n_lines = 1});
      y += cat_h;
      last_type = BlockType::CatTitle;

    } else if (current_type == BlockType::Paragraph) {
      int remaining_lines = (settings.h - y) / settings.line_height;
      if (remaining_lines < 1)
        break;

      int min_lines = std::min(3, remaining_lines);
      int max_lines = std::min(8, remaining_lines);
      int n_lines = rand_int(min_lines, max_lines);

      blocks.push_back({.type = BlockType::Paragraph,
                        .y_start = y,
                        .height = settings.line_height * n_lines,
                        .n_lines = n_lines});
      y += settings.line_height * n_lines;
      last_type = BlockType::Paragraph;

    } else if (current_type == BlockType::Schema) {
      float schema_ratio = rand_float(0.6f, 0.9f);
      float x_offset = rand_float(0, 1);
      int height = rand_int(300, 500);

      blocks.push_back({.type = BlockType::Schema,
                        .y_start = y,
                        .height = height,
                        .schema_x_offset = x_offset});
      y += height;
      last_type = BlockType::Schema;
    }

    if (y + settings.line_height <= settings.h) {
      blocks.push_back(
          {.type = BlockType::SkipLine, .y_start = y, .line_skipped = 1});
      y += settings.line_height;
    }
  }

  return blocks;
}

std::vector<LayoutBlock> generate_page_layout(PageSettings settings,
                                              std::mt19937 &rng) {
  std::vector<LayoutBlock> blocks;
  auto rand_int = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };
  auto rand_float = [&]() {
    return std::uniform_real_distribution<float>(0.f, 1.f)(rng);
  };

  int top_margin = rand_int(30, std::min(settings.h, 100));
  int y = top_margin;
  while (y + settings.line_height <= settings.h) {

    int remaining = (settings.h - y) / settings.line_height;
    assert(remaining >= 1 && "rand_int would receive lo > hi");
    int n_lines = rand_int(1, std::min(4, remaining));
    blocks.push_back({.type = BlockType::Paragraph,
                      .y_start = y,
                      .height = settings.line_height * n_lines,
                      .n_lines = n_lines});
    y += settings.line_height * n_lines + rand_int(20, 40);
    if (rand_float() > 0.5 && y + settings.line_height < settings.h) {
      int remaining = (settings.h - y) / settings.line_height;
      assert(remaining >= 1 && "rand_int would receive lo > hi");
      int n_lines = rand_int(1, std::min(4, remaining));
      blocks.push_back({.type = BlockType::SkipLine,
                        .y_start = y,
                        .height = settings.line_height * n_lines,
                        .line_skipped = n_lines});
      y += settings.line_height * n_lines;
    }
  }

  return blocks;
}

std::vector<LayoutBlock> generate_layout(PageSettings settings,
                                         std::mt19937 &rng) {
  return settings.document ? generate_document_layout(settings, rng)
                           : generate_page_layout(settings, rng);
}

void select_assets(
    PageSettings settings, std::vector<LayoutBlock> &layout,
    std::map<DatasetType, std::vector<std::unique_ptr<Dataset>>> const
        &datasets,
    std::mt19937 &rng) {
  auto rand_int = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };
  auto rand_float = [&]() {
    return std::uniform_real_distribution<float>(0.f, 1.f)(rng);
  };

  std::unordered_map<std::string, int> offsets;
  for (auto const &[k, sub_datasets] : datasets) {
    for (auto const &d : sub_datasets) {
      offsets[d->id] = rand_int(0, d->len() - 1);
    }
  }

  int page_asset_idx = 0; // Global asset index for the entire page

  for (auto &block : layout) {
    switch (block.type) {
    case BlockType::Title:
    case BlockType::CatTitle: {
      Dataset *dataset =
          get_random_dataset(datasets, {DatasetType::HandwrittenWords}, rng);
      int x = rand_int(50, 80);
      int max_x = settings.w - rand_int(.1 * settings.w, .5 * settings.w);
      int retry_n = 0;

      block.assets.push_back({});
      auto &line = block.assets.back();

      while (x < max_x) {
        offsets[dataset->id]++;
        std::array<int, 2> s =
            dataset->get_size((offsets[dataset->id] - 1) % dataset->len());
        float scale = static_cast<float>(block.height) / s[1];

        int scaled_w = static_cast<int>(s[0] * scale);

        if (scaled_w + x > max_x) {
          if (retry_n + 1 > 3) {
            retry_n = 0;
            break;
          }
          retry_n++;
          continue;
        }

        line.push_back(
            {.idx = (int)((offsets[dataset->id] - 1) % dataset->len()),
             .page_idx = page_asset_idx,
             .dataset_id = dataset->id,
             .w = scaled_w,
             .h = block.height,
             .x = x,
             .y = block.y_start});
        page_asset_idx++;
        x += scaled_w + rand_int(10, 20);
      }
      break;
    }
    case BlockType::Paragraph: {
      int x = rand_int(30, 50);
      block.assets.reserve(block.n_lines);
      for (int i = 0; i < block.n_lines; i++) {
        int retry_n = 0;
        block.assets.push_back({});
        auto &line = block.assets.back();

        while (x < settings.w) {
          Dataset *dataset = get_random_dataset(
              datasets, {DatasetType::HandwrittenWords, DatasetType::MathExpr},
              rng);
          offsets[dataset->id]++;
          std::array<int, 2> s =
              dataset->get_size((offsets[dataset->id] - 1) % dataset->len());
          float scale = static_cast<float>(settings.line_height) / s[1];

          int scaled_w = static_cast<int>(s[0] * scale);

          if (scaled_w + x > settings.w) {
            if (retry_n + 1 > 3) {
              retry_n = 0;
              break;
            }
            retry_n++;
            continue;
          }

          line.push_back(
              {.idx = (int)((offsets[dataset->id] - 1) % dataset->len()),
               .page_idx = page_asset_idx,
               .dataset_id = dataset->id,
               .w = scaled_w,
               .h = settings.line_height,
               .x = x,
               .y = block.y_start + i * settings.line_height});
          x += scaled_w + rand_int(10, 20);
        }
        x = rand_int(30, 50);
      }
      break;
    }
    case BlockType::Schema: {
      Dataset *dataset =
          get_random_dataset(datasets, {DatasetType::Diagram}, rng);
      offsets[dataset->id]++;
      std::array<int, 2> s =
          dataset->get_size((offsets[dataset->id] - 1) % dataset->len());
      float ratio = static_cast<float>(block.height) / static_cast<float>(s[1]);
      if (ratio <= 0.0f)
        ratio = 0.1f;
      int w = ratio * s[0];
      int h = ratio * s[1];
      int offset = (settings.w - w) * block.schema_x_offset;
      block.assets.push_back({});
      auto &line = block.assets.back();

      line.push_back({
          .idx = (int)((offsets[dataset->id] - 1) % dataset->len()),
          .page_idx = page_asset_idx,
          .dataset_id = dataset->id,
          .w = w,
          .h = h,
          .x = offset,
          .y = block.y_start,
      });
      break;
    };
    case BlockType::SkipLine: {
      break;
    }
    }
  }
}

Mat render_clean_page(PageSettings settings, std::vector<LayoutBlock> &layout,
                      std::unordered_map<std::string, Dataset *>
                          &datasets, // map<dataset_id, dataset*>
                      std::mt19937 &rng) {
  auto rand_int = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };
  auto rand_float = [&]() {
    return std::uniform_real_distribution<float>(0.f, 1.f)(rng);
  };
  int w = settings.w;
  int h = settings.h;
  int brightness = rand_int(220, 255);
  Mat clean = Mat::ones(h, w, CV_8UC1) * brightness; // White page
  Mat warped; // Temp container for warped imgs
  for (auto &block : layout) {
    for (auto &lines : block.assets) {
      for (auto &asset : lines) {
        Dataset *d = datasets[asset.dataset_id];
        AssetRow row = d->get_asset(asset.idx);
        Mat img = row.image;
        if (img.empty() || img.cols == 0 || img.rows == 0) {
          std::cerr << std::format("[Warning] Dataset '{}' returned an empty "
                                   "image at offset {}. Skipping token.",
                                   d->id, asset.idx)
                    << std::endl;
          continue;
        }
        asset.transcript = row.transcript;
        if (d->type == DatasetType::HandwrittenWords) {
          add_random_perspective(img, warped, settings.max_warp, asset.h, rng);
        } else {
          // It will just scale it without adding perspective
          add_random_perspective(img, warped, 0, asset.h, rng);
        }
        Rect target_roi({asset.x, asset.y}, warped.size());
        target_roi &= Rect(0, 0, w, h); // clamp to page bounds
        if (target_roi.empty())
          continue;
        Mat warped_cropped =
            warped(Rect(0, 0, target_roi.width, target_roi.height));
        Mat roi = clean(target_roi);
        cv::min(roi, warped_cropped, roi);
      }
    }
  }
  return clean;
}

std::unordered_map<BlockType, std::string> block_type_to_string = {
    {BlockType::Title, "title"},
    {BlockType::CatTitle, "category_title"},
    {BlockType::Paragraph, "paragraph"},
    {BlockType::Schema, "schema"},
    {BlockType::SkipLine, "line_skip"}};

pugi::xml_document serialize_xml(int page_idx, PageSettings settings,
                                 std::vector<LayoutBlock> &layout) {
  pugi::xml_document doc;
  pugi::xml_node page = doc.append_child("page");
  page.append_attribute("idx") = page_idx;
  page.append_attribute("w") = settings.w;
  page.append_attribute("h") = settings.h;
  page.append_attribute("line_height") = settings.line_height;
  page.append_attribute("brightness") = settings.brightness;

  for (const auto &block : layout) {
    pugi::xml_node block_node = page.append_child("block");
    block_node.append_attribute("type") = block_type_to_string[block.type];
    block_node.append_attribute("y_start") = block.y_start;
    block_node.append_attribute("height") = block.height;
    switch (block.type) {
    case BlockType::Title:
    case BlockType::CatTitle: {
      pugi::xml_node line_node = block_node.append_child("line");
      line_node.append_attribute("y") = block.y_start;
      for (const auto &asset : block.assets[0]) {
        pugi::xml_node word_node = line_node.append_child("word");
        word_node.append_attribute("idx") = asset.page_idx;
        word_node.append_attribute("dataset_idx") = asset.idx;
        word_node.append_attribute("dataset_id") = asset.dataset_id.c_str();
        word_node.append_attribute("x") = asset.x;
        word_node.append_attribute("y") = asset.y;
        word_node.append_attribute("w") = asset.w;
        word_node.append_attribute("h") = asset.h;
        word_node.text() = asset.transcript.c_str();
      }
      break;
    }
    case BlockType::Paragraph: {
      for (const auto &line : block.assets) {
        pugi::xml_node line_node = block_node.append_child("line");
        line_node.append_attribute("y") = line[0].y;
        for (const auto &asset : line) {
          pugi::xml_node word_node = line_node.append_child("word");
          word_node.append_attribute("idx") = asset.page_idx;
          word_node.append_attribute("dataset_idx") = asset.idx;
          word_node.append_attribute("dataset_id") = asset.dataset_id.c_str();
          word_node.append_attribute("x") = asset.x;
          word_node.append_attribute("y") = asset.y;
          word_node.append_attribute("w") = asset.w;
          word_node.append_attribute("h") = asset.h;
          word_node.text() = asset.transcript.c_str();
        }
      }
      break;
    }
    case BlockType::Schema: {
      pugi::xml_node schema_node = block_node.append_child("schema");
      auto &asset = block.assets[0][0];
      schema_node.append_attribute("idx") = asset.page_idx;
      schema_node.append_attribute("dataset_idx") = asset.idx;
      schema_node.append_attribute("dataset_id") = asset.dataset_id.c_str();
      schema_node.append_attribute("x") = asset.x;
      schema_node.append_attribute("y") = asset.y;
      schema_node.append_attribute("w") = asset.w;
      schema_node.append_attribute("h") = asset.h;
      break;
    }
    case BlockType::SkipLine: {
      block_node.append_attribute("line_skipped") = block.line_skipped;
      break;
    }
    }
  }
  return doc;
}

void generate_page(int idx, PageSettings settings,
                   std::map<DatasetType, std::vector<std::unique_ptr<Dataset>>>
                       &datasets_by_type,
                   const fs::path &clean_dir, const fs::path &ruled_dir,
                   const fs::path &labels_dir, std::mt19937 &rng) {
  std::vector<LayoutBlock> layout = generate_layout(settings, rng);
  select_assets(settings, layout, datasets_by_type, rng);
  std::unordered_map<std::string, Dataset *> dataset_by_id;
  for (const auto &[type, datasets] : datasets_by_type) {
    for (auto &dataset : datasets) {
      dataset_by_id[dataset->id] = dataset.get();
    }
  }
  Mat clean = render_clean_page(settings, layout, dataset_by_id, rng);

  std::vector<int> compression_params;
  compression_params.push_back(IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);
  fs::path clean_path = clean_dir / std::format("{}.jpg", idx);
  imwrite(clean_path, clean, compression_params);
  Mat ruled = Mat::ones(settings.h, settings.w, CV_8UC1) *
              255; // Start with a white page for ruled version
  draw_lines(ruled, settings.arc, settings.imperfect_lines, rng);
  cv::min(ruled, clean, ruled);
  fs::path ruled_path = ruled_dir / std::format("{}.jpg", idx);
  imwrite(ruled_path, ruled, compression_params);
  if (settings.save_labels) {
    pugi::xml_document doc = serialize_xml(idx, settings, layout);
    fs::path xml_path = labels_dir / std::format("{}.xml", idx);
    doc.save_file(xml_path.c_str());
  }
}

void generate_pages(fs::path target, std::vector<DatasetS> datasets, int n,
                    bool preload, bool use_arc, bool document, float max_warp,
                    bool imperfect_lines, bool save_xml) {
  cv::setNumThreads(1);
  std::signal(SIGINT, signal_handler);

  shutdown_requested = false;
  std::mt19937 rng(std::random_device{}());
  auto rand_int = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };
  auto rand_float = [&]() {
    return std::uniform_real_distribution<float>(0.f, 1.f)(rng);
  };

  std::cout << std::format("Generating {} pages", n) << std::endl;
  std::cout << std::format("Creating dirs...") << std::endl;
  fs::path ruled_dir = target / "ruled-pages";
  fs::path clean_dir = target / "clean-pages";
  fs::path labels_dir = target / "labels";
  fs::create_directories(ruled_dir);
  fs::create_directories(clean_dir);
  if (save_xml) {
    fs::create_directories(labels_dir);
  }

  if (datasets.empty()) {
    throw std::invalid_argument("Datasets cannot be empty");
  }

  std::cout << std::format("Constructing datasets...") << std::endl;

  std::map<DatasetType, std::vector<std::unique_ptr<Dataset>>> datasets_by_type;
  std::map<DatasetType, float> total_weight_by_type;

  for (auto type : {DatasetType::HandwrittenWords, DatasetType::MathExpr,
                    DatasetType::Diagram}) {
    datasets_by_type.emplace(type, std::vector<std::unique_ptr<Dataset>>{});
    total_weight_by_type.emplace(type, 0.0f);
  }

  for (const auto &dataset : datasets) {
    total_weight_by_type[get_dataset_type(dataset.id)] += dataset.proportion;
  }
  // Normalize weights
  for (auto &dataset : datasets) {
    dataset.proportion =
        dataset.proportion / total_weight_by_type[get_dataset_type(dataset.id)];
  }

  for (const auto &d : datasets) {
    auto dataset = make_dataset(d); // throws if unknown id
    if (!dataset->valid())
      throw std::invalid_argument("Invalid dataset path: " + d.path.string());
    datasets_by_type[dataset->type].push_back(std::move(dataset));
    std::cout << std::format("Found dataset {} at {}", d.id, d.path.string())
              << std::endl;
  }

  std::cout << std::format("Loading datasets...") << std::endl;
  for (const auto &[type, vec] : datasets_by_type) {
    for (const auto &d : vec) {
      if (d->valid()) {
        d->load();
      } else {
        throw std::invalid_argument(
            std::format("Dataset ID {}, path {} couldn't be loaded", d->id,
                        d->path.string()));
      }
    }
  }

  int work{0};
  std::mutex progress_mutex;
  auto bar = bk::ProgressBar(&work, {
                                        .total = n,
                                        .message = "Generating pages",
                                        .speed = 1.,
                                        .speed_unit = "page/s",
                                    });
  std::atomic<int> next_page_idx{0};
  unsigned int num_threads = std::thread::hardware_concurrency();
  std::vector<std::jthread> workers;

  std::cout << std::format("Spawning {} worker threads... Starting generation",
                           num_threads)
            << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();
  for (unsigned int t = 0; t < num_threads; ++t) {
    workers.emplace_back([&]() {
      std::mt19937 local_rng(std::random_device{}());
      auto rand_int = [&](int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(local_rng);
      };
      auto rand_float = [&]() {
        return std::uniform_real_distribution<float>(0.f, 1.f)(local_rng);
      };
      while (true) {
        if (shutdown_requested) {
          break;
        }

        int i = next_page_idx.fetch_add(1);
        if (i >= n) {
          break; // No more pages left to generate
        }
        try {

          int w;
          int h;
          int line_height;
          if (document) {
            float aspect_rand = rand_float();
            float aspect;
            h = rand_int(2500, 3000);

            // Either sqrt(2) aspect (A serie), 17:22 (American letter), or
            // 17:28 (American legal)
            if (aspect_rand < 1.0f / 3.0f) {
              aspect = 1.0f / std::sqrt(2.0f);
            } else if (aspect_rand < 2.0f / 3.0f && aspect_rand >= 1.0f / 3) {
              aspect = 17.f / 22.f;
            } else if (aspect_rand >= 2.0f / 3.0f) {
              aspect = 17.f / 28.f;
            }
            w = static_cast<int>(std::round(h * aspect));
            line_height = rand_int(45, 65);
          } else {
            w = rand_int(500, 1600);
            h = rand_int(800, 2000);
            line_height = rand_int(50, 190);
          }

          PageSettings settings = {.document = document,
                                   .w = w,
                                   .h = h,
                                   .line_height = line_height,
                                   .max_warp = max_warp,
                                   .imperfect_lines = imperfect_lines,
                                   .arc = use_arc};
          generate_page(i, settings, datasets_by_type, clean_dir, ruled_dir,
                        labels_dir, local_rng);
          {
            std::lock_guard<std::mutex> lock(progress_mutex);
            work++;
          }
        } catch (const std::exception &e) {
          std::cerr << std::format("[Worker] Exception on page {}: {}\n", i,
                                   e.what());
        } catch (...) {
          std::cerr << std::format("[Worker] Unknown exception on page {}\n",
                                   i);
        }
      }
    });
  }

  // Join all jthreads before clean up.
  workers.clear();

  if (shutdown_requested) {
    std::cout << "\nGeneration interrupted by user. Exiting cleanly..."
              << std::endl;
  }

  bar->done();
  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;

  double avg_time = duration_ms.count() / n;
  std::cout << std::format("Generated {} pages in {:.2f} ms ({:.4f} ms/page)\n",
                           n, duration_ms.count(), avg_time);
}