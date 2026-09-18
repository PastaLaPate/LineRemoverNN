#pragma once
#include "generation/asset_selection.h"
#include "generation/layout.h"
#include "page_gen.h"
#include <opencv2/core/mat.hpp>

cv::Mat render_clean_page(const PageSettings settings, Layout &layout,
                          const DatasetLookup &datasets, bool debug);
void draw_lines(cv::Mat &img, bool use_arc, bool imperfect_lines);