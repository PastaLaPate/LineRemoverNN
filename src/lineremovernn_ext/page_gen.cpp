#include "page_gen.h"
#include "barkeep/barkeep.h"
#include "datasets/datasets.h"
#include "datasets/factory.h"
#include "datasets/utils.h"
#include "generation/layout.h"
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
  std::vector<float> weights;
  for (auto type : types) {
    for (const auto &d : datasets_by_type.at(type)) {
      candidates.push_back(d.get());
      weights.push_back(d->proportion);
    }
  }
  if (candidates.empty())
    throw std::runtime_error("No datasets for requested types");

  std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
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

Mat render_clean_page(PageSettings settings, std::vector<LayoutBlock> &layout,
                      std::unordered_map<std::string, Dataset *>
                          &datasets, // map<dataset_id, dataset*>
                      std::mt19937 &rng, bool debug) {
  auto rand_int = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };
  auto rand_float = [&]() {
    return std::uniform_real_distribution<float>(0.f, 1.f)(rng);
  };
  int w = settings.w;
  int h = settings.h;
  int brightness = settings.brightness;
  Mat clean = Mat::ones(h, w, CV_8UC1) * brightness; // White page
  Mat warped; // Temp container for warped imgs

  double load_total_ms = 0.0, warp_total_ms = 0.0, copy_total_ms = 0.0;
  int asset_count = 0;

  struct DatasetStat {
    double total_ms = 0.0;
    int count = 0;
  };
  std::unordered_map<std::string, DatasetStat> load_stats_by_dataset;

  for (auto &block : layout) {
    for (auto &lines : block.assets) {
      for (auto &asset : lines) {
        Dataset *d = datasets[asset.dataset_id];

        auto t0 = std::chrono::high_resolution_clock::now();
        AssetRow row = d->get_asset(asset.idx);
        auto t1 = std::chrono::high_resolution_clock::now();

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
        auto t2 = std::chrono::high_resolution_clock::now();

        Rect target_roi({asset.x, asset.y}, warped.size());
        target_roi &= Rect(0, 0, w, h); // clamp to page bounds
        if (target_roi.empty())
          continue;
        Mat warped_cropped =
            warped(Rect(0, 0, target_roi.width, target_roi.height));
        Mat roi = clean(target_roi);
        cv::min(roi, warped_cropped, roi);
        auto t3 = std::chrono::high_resolution_clock::now();

        double load_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        load_total_ms += load_ms;
        warp_total_ms +=
            std::chrono::duration<double, std::milli>(t2 - t1).count();
        copy_total_ms +=
            std::chrono::duration<double, std::milli>(t3 - t2).count();
        asset_count++;

        auto &stat = load_stats_by_dataset[asset.dataset_id];
        stat.total_ms += load_ms;
        stat.count++;
      }
    }
  }

  if (asset_count > 0 && debug) {
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lk(log_mutex);

    std::cout << std::format(
        "[render_clean_page] {} assets — avg load: {:.4f} ms, avg warp: {:.4f} "
        "ms, avg copy: {:.4f} ms "
        "(totals: load {:.2f} ms, warp {:.2f} ms, copy {:.2f} ms)\n",
        asset_count, load_total_ms / asset_count, warp_total_ms / asset_count,
        copy_total_ms / asset_count, load_total_ms, warp_total_ms,
        copy_total_ms);

    for (const auto &[dataset_id, stat] : load_stats_by_dataset) {
      std::cout << std::format(
          "    dataset '{}': {} loads, avg {:.4f} ms, total {:.2f} ms\n",
          dataset_id, stat.count, stat.total_ms / stat.count, stat.total_ms);
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
                   const fs::path &labels_dir, std::mt19937 &rng, bool debug) {
  Layout layout = generate_layout(settings);

  select_assets(settings, layout, datasets_by_type, rng);
  auto start_time = std::chrono::high_resolution_clock::now();

  DatasetLookup lookup = make_dataset_lookup(datasets_by_type);

  Mat clean = render_clean_page(settings, layout, lookup, rng, debug);

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
  if (debug) {
    std::cout << "Render in " << duration_ms.count() << " ms" << std::endl;
  }

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
                    bool use_arc, bool document, float max_warp,
                    bool imperfect_lines, bool save_xml, bool debug,
                    int max_workers) {
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

  DatasetGroups datasets_groups = construct_datasets(datasets);

  std::cout << std::format("Loading datasets...") << std::endl;
  for (const auto &[type, vec] : datasets_groups) {
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
  unsigned int num_threads =
      std::min(max_workers == 0 ? std::thread::hardware_concurrency()
                                : static_cast<unsigned int>(max_workers),
               static_cast<unsigned int>(n));
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
                                   .save_labels = save_xml,

                                   .w = w,
                                   .h = h,
                                   .line_height = line_height,
                                   .brightness = rand_int(220, 255),
                                   .max_warp = max_warp,
                                   .imperfect_lines = imperfect_lines,
                                   .arc = use_arc};
          generate_page(i, settings, datasets_groups, clean_dir, ruled_dir,
                        labels_dir, local_rng, debug);
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