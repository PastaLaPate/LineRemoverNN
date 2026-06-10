#include "page_gen.h"
#include "barkeep/barkeep.h"
#include "datasets/factory.h"
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
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

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

Dataset *select_dataset(const std::vector<std::unique_ptr<Dataset>> &datasets,
                        std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(rng);

  for (const auto &d : datasets) {
    if (r < d->proportion) {
      return d.get();
    }
    r -= d->proportion;
  }
  return datasets.back().get();
}

void add_random_perspective(const Mat &img, Mat &transformed, float max_warp,
                            std::mt19937 &rng) {
  if (max_warp <= 0.0f) {
    transformed = img.clone();
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

  matrix.row(0) -= min_x * matrix.row(2);
  matrix.row(1) -= min_y * matrix.row(2);

  Scalar border_val;
  if (img.channels() == 1) {
    border_val = Scalar(255);
  } else {
    border_val = Scalar(255, 255, 255, 255);
  }

  warpPerspective(img, transformed, matrix, Size(new_width, new_height),
                  INTER_LINEAR, BORDER_CONSTANT, border_val);
}

void draw_lines(Mat &img, bool use_arc, bool imperfect_lines,
                std::mt19937 &rng) {
  int W = img.cols;
  int H = img.rows;

  std::uniform_int_distribution<int> line_spacing_dist(35, 100);
  std::uniform_int_distribution<int> sub_line_spacing_dist(3, 6);
  std::uniform_int_distribution<int> rand_color(100, 180);
  std::uniform_int_distribution<int> rand_sub_color(200, 240);
  std::uniform_int_distribution<int> rand_lw(1, 3);
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
      int lw = is_main ? rand_lw(rng) : 1;
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

void generate_single_page(int idx, int max_warp, bool use_arc,
                          bool imperfect_lines,
                          const std::vector<std::unique_ptr<Dataset>> &loaded,
                          unsigned int seed, const fs::path &clean_dir,
                          const fs::path &ruled_dir) {
  std::mt19937 rng(std::random_device{}());
  auto rand_int = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };
  auto rand_float = [&]() {
    return std::uniform_real_distribution<float>(0.f, 1.f)(rng);
  };

  std::map<std::string, int> offsets;
  for (auto const &d : loaded) {
    offsets[d->id] = rand_int(0, d->len() - 1);
  }

  int w = rand_int(400, 1600);
  int h = rand_int(400, 2000);
  int margin_left = 50 + rand_int(0, 200);
  Mat clean = Mat::ones(h, w, CV_8UC1) * rand_int(220, 255); // White page
  Mat warped_text;

  Point cursor = {margin_left, 50};
  while (cursor.y < h) {
    int max_h = 0;
    while (cursor.x < w) {
      auto d = select_dataset(loaded, rng);
      int offset = offsets[d->id] % d->len();
      Mat img = d->get_image(offset);
      offsets[d->id] += 1;
      if (img.empty() || img.cols == 0 || img.rows == 0) {
        std::cerr << std::format("[Warning] Dataset '{}' returned an empty "
                                 "image at offset {}. Skipping token.",
                                 d->id, offset)
                  << std::endl;
        continue; // Skip this iteration safely without crashing!
      }
      add_random_perspective(img, warped_text, max_warp, rng);
      if (cursor.x + warped_text.cols >= w) {
        break;
      }
      if (cursor.y + warped_text.rows >= h) {
        cursor.y = h;
        break;
      }
      if (max_h < warped_text.rows) {
        max_h = warped_text.rows;
      }
      Rect target_roi(cursor, warped_text.size());
      clean(target_roi) = cv::min(clean(target_roi), warped_text);

      cursor.x += warped_text.cols + rand_int(-5, 25);
    }
    cursor.x = margin_left + rand_int(0, 20);
    cursor.y += max_h + rand_int(20, 30);
  }

  std::vector<int> compression_params;
  compression_params.push_back(IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);
  std::filesystem::path clean_path = clean_dir / std::format("{}.jpg", idx);
  imwrite(clean_path, clean, compression_params);
  draw_lines(clean, use_arc, imperfect_lines, rng);
  std::filesystem::path ruled_path = ruled_dir / std::format("{}.jpg", idx);
  imwrite(ruled_path, clean, compression_params);
}

void generate_pages(std::filesystem::path target,
                    std::vector<DatasetS> datasets, int n, bool preload,
                    bool use_arc, float max_warp, bool imperfect_lines,
                    bool save_json) {
  std::signal(SIGINT, signal_handler);

  shutdown_requested = false;
  std::mt19937 rng(std::random_device{}());
  auto rand_int = [&](int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
  };

  std::cout << std::format("Generating {} pages", n) << std::endl;
  std::cout << std::format("Creating dirs...") << std::endl;
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

  std::cout << std::format("Spawning {} worker threads...", num_threads)
            << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();
  for (unsigned int t = 0; t < num_threads; ++t) {
    workers.emplace_back([&]() {
      while (true) {
        if (shutdown_requested) {
          break;
        }

        int i = next_page_idx.fetch_add(1);
        if (i >= n) {
          break; // No more pages left to generate
        }

        generate_single_page(i, max_warp, use_arc, imperfect_lines, loaded, 0,
                             clean_dir, ruled_dir);

        {
          std::lock_guard<std::mutex> lock(progress_mutex);
          work++;
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