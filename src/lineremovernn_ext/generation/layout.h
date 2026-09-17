
#include "page_gen.h"
#include <random>
#include <vector>

using Layout = std::vector<LayoutBlock>;

Layout generate_layout(const PageSettings &settings, std::mt19937 &rng);
