#pragma once
#include "datasets/datasets.h"
#include "datasets/factory.h"
#include <filesystem>
#include <vector>

void generate_pages(std::filesystem::path target,
                    std::vector<DatasetS> datasets, int n, bool preload,
                    bool use_arc, bool document, float max_warp,
                    bool imperfect_lines, bool save_json);