#pragma once
#include "datasets/datasets.h"
#include <filesystem>
#include <vector>

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
  std::string dataset_id;
  int idx;

  int page_idx; // global asset index, from left to right, top to bottom
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

void generate_pages(std::filesystem::path target,
                    std::vector<DatasetS> datasets, int n, bool use_arc,
                    bool document, float max_warp, bool imperfect_lines,
                    bool save_xml, bool debug, int max_workers);