#pragma once
#include <filesystem>
#include <string>
#include <vector>

struct Dataset {
  std::string id;
  std::filesystem::path path;
  float proportion;
};

void generate_pages(std::filesystem::path target, std::vector<Dataset> datasets,
                    int n, bool preload, bool use_arc, float max_warp,
                    bool imperfect_lines, bool save_json);