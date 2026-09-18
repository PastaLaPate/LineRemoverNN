#include "rendering.h"
#include "generation/asset_selection.h"
#include "utils/random.h"
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

void add_random_perspective(const Mat &img, Mat &transformed, float max_warp,
                            int target_height) {
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

  auto dist_x = [&] { return ThreadRandom::rand_int(-max_dx, max_dx); };
  auto dist_y = [&] { return ThreadRandom::rand_int(-max_dy, max_dy); };

  std::vector<Point2f> dst_points = {
      Point2f(dist_x(), dist_y()), Point2f(width + dist_x(), dist_y()),
      Point2f(width + dist_x(), height + dist_y()),
      Point2f(dist_x(), height + dist_y())};

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

void draw_lines(Mat &img, bool use_arc, bool imperfect_lines) {
  int W = img.cols;
  int H = img.rows;

  int line_spacing = ThreadRandom::rand_int(45, 100);
  int sub = ThreadRandom::rand_int(2, 5);

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
    int darkness = ThreadRandom::rand_int(100, 180);
    int lw = std::max(1, ThreadRandom::rand_int(1, 3) - 1);

    if (use_arc) {
      float amplitude = ThreadRandom::rand_float(-15.0f, 15.0f);
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

      int darkness = is_main ? ThreadRandom::rand_int(100, 180)
                             : ThreadRandom::rand_int(160, 190);
      int lw =
          is_main ? ThreadRandom::rand_int(1, 3) : ThreadRandom::rand_int(1, 2);
      int y_off = ThreadRandom::rand_int(-3, 3);

      if (use_arc) {
        float amplitude = ThreadRandom::rand_float(-15.0f, 15.0f);
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

  int margin_darkness = ThreadRandom::rand_int(100, 180) + 20;
  int margin_lw = ThreadRandom::rand_int(1, 3) + 1;
  if (use_arc) {
    float amplitude = ThreadRandom::rand_float(-15.0f, 15.0f);
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
    int num_spots = ThreadRandom::rand_int(40, 120);
    for (int k = 0; k < num_spots; ++k) {
      circle(img,
             Point(ThreadRandom::rand_int(0, W - 1),
                   ThreadRandom::rand_int(0, H - 1)),
             ThreadRandom::rand_int(1, 4), Scalar(255), FILLED);
    }
  }
}

Mat render_clean_page(const PageSettings settings, Layout &layout,
                      const DatasetLookup &datasets, bool debug) {
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
        auto it = datasets.find(asset.dataset_id);
        if (it == datasets.end() || it->second == nullptr) {
          throw std::runtime_error("Unknown dataset id: " + asset.dataset_id);
        }
        Dataset &d = *it->second;

        auto t0 = std::chrono::high_resolution_clock::now();
        AssetRow row = d.get_asset(asset.idx);
        auto t1 = std::chrono::high_resolution_clock::now();

        Mat img = row.image;
        if (img.empty() || img.cols == 0 || img.rows == 0) {
          std::cerr << std::format("[Warning] Dataset '{}' returned an empty "
                                   "image at offset {}. Skipping token.",
                                   d.id, asset.idx)
                    << std::endl;
          continue;
        }
        asset.transcript = row.transcript;

        if (d.type == DatasetType::HandwrittenWords) {
          add_random_perspective(img, warped, settings.max_warp, asset.h);
        } else {
          // It will just scale it without adding perspective
          add_random_perspective(img, warped, 0, asset.h);
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