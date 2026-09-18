#include "page_gen.h"
#include "barkeep/barkeep.h"
#include "datasets/datasets.h"
#include "datasets/utils.h"
#include "generation/asset_selection.h"
#include "generation/layout.h"
#include "generation/rendering.h"
#include "generation/serializer.h"
#include "logging/python_logger.h"
#include "pugixml/pugixml.hpp"
#include "utils/random.h"
#include <algorithm>
#include <atomic>
#include <cairo/cairo.h>
#include <cassert>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <stdexcept>
#include <thread>
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

void generate_page(int idx, PageSettings settings, DatasetGroups &groups,
                   const fs::path &clean_dir, const fs::path &ruled_dir,
                   const fs::path &labels_dir, PythonLoggerBridge &logger,
                   bool debug) {
  Layout layout = generate_layout(settings);

  select_assets(settings, layout, groups);
  auto start_time = std::chrono::high_resolution_clock::now();

  DatasetLookup lookup = make_dataset_lookup(groups);

  Mat clean = render_clean_page(settings, layout, lookup, logger, debug);

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
  if (debug) {
    logger.debug(std::format("Render in {:.2f} ms", duration_ms.count()));
  }

  std::vector<int> compression_params;
  compression_params.push_back(IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);

  fs::path clean_path = clean_dir / std::format("{}.jpg", idx);
  imwrite(clean_path, clean, compression_params);

  Mat ruled = Mat::ones(settings.h, settings.w, CV_8UC1) *
              255; // Start with a white page for ruled version
  draw_lines(ruled, settings.arc, settings.imperfect_lines);

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
                    int max_workers, PythonLoggerBridge &logger) {
  cv::setNumThreads(1);
  std::signal(SIGINT, signal_handler);

  shutdown_requested = false;

  logger.info(std::format("Generating {} pages", n));
  logger.info("Creating output directories");
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

  logger.info("Constructing datasets");

  DatasetGroups datasets_groups = construct_datasets(datasets, logger);

  logger.info("Loading datasets");
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

  logger.info(std::format(
      "Spawning {} worker threads... Starting generation", num_threads));

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
        try {

          int w;
          int h;
          int line_height;
          if (document) {
            float aspect_rand = ThreadRandom::rand_float();
            float aspect;
            h = ThreadRandom::rand_int(2500, 3000);

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
            line_height = ThreadRandom::rand_int(45, 65);
          } else {
            w = ThreadRandom::rand_int(500, 1600);
            h = ThreadRandom::rand_int(800, 2000);
            line_height = ThreadRandom::rand_int(50, 190);
          }

          PageSettings settings = {.document = document,
                                   .save_labels = save_xml,

                                   .w = w,
                                   .h = h,

                                   .line_height = line_height,
                                   .brightness =
                                       ThreadRandom::rand_int(220, 255),
                                   .max_warp = max_warp,
                                   .imperfect_lines = imperfect_lines,
                                   .arc = use_arc};
          generate_page(i, settings, datasets_groups, clean_dir, ruled_dir,
                        labels_dir, logger, debug);
          {
            std::lock_guard<std::mutex> lock(progress_mutex);
            work++;
          }
        } catch (const std::exception &e) {
          logger.error(
              std::format("[Worker] Exception on page {}: {}", i, e.what()));
        } catch (...) {
          logger.error(std::format("[Worker] Unknown exception on page {}", i));
        }
      }
    });
  }

  // Join all jthreads before clean up.
  workers.clear();

  if (shutdown_requested) {
    logger.warning("Generation interrupted by user. Exiting cleanly...");
  }

  bar->done();
  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;

  double avg_time = duration_ms.count() / n;
  logger.info(std::format("Generated {} pages in {:.2f} ms ({:.4f} ms/page)",
                          n, duration_ms.count(), avg_time));
}