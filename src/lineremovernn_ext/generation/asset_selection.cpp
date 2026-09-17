#include "asset_selection.h"
#include "utils/random.h"

Dataset *get_random_dataset(const DatasetGroups &datasets_by_type,
                            std::initializer_list<DatasetType> types) {
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

  return candidates[ThreadRandom::sample_weighted(weights.begin(),
                                                  weights.end())];
}

struct SelectionState {
  std::unordered_map<std::string, int> offsets;
  std::unordered_map<std::string, float> typical_scales;
  int next_page_asset_idx = 0;
};

float get_typical_scale(Dataset &dataset, int target_height,
                        SelectionState &state) {
  std::string key = dataset.id + "_" + std::to_string(target_height);
  if (state.typical_scales.find(key) ==
      state.typical_scales.end()) { // No cached
    std::vector<int> heights;
    for (int k = 0; k < 50; k++) {
      int peek_idx = ThreadRandom::rand_int(0, dataset.len() - 1);
      heights.push_back(dataset.get_size(peek_idx)[1]);
    }
    std::sort(heights.begin(), heights.end());
    int median_h = heights[heights.size() / 2];
    state.typical_scales[key] =
        static_cast<float>(target_height) / std::max(1, median_h); // prevent /0
  }
  return state.typical_scales[key];
};

void select_assets(const PageSettings &settings, Layout &layout,
                   const DatasetGroups &datasets) {
  SelectionState state;

  for (auto const &[k, sub_datasets] : datasets) {
    for (auto const &d : sub_datasets) {
      state.offsets[d->id] = ThreadRandom::rand_int(0, d->len() - 1);
    }
  }

  for (auto &block : layout) {
    switch (block.type) {
    case BlockType::Title:
    case BlockType::CatTitle: {
      Dataset *dataset =
          get_random_dataset(datasets, {DatasetType::HandwrittenWords});
      int x = ThreadRandom::rand_int(50, 80);
      int max_x =
          settings.w - ThreadRandom::rand_int(.1 * settings.w, .5 * settings.w);
      int retry_n = 0;

      block.assets.push_back({});
      auto &line = block.assets.back();

      float typical_scale = get_typical_scale(*dataset, block.height, state);

      while (x < max_x) {
        state.offsets[dataset->id]++;
        std::array<int, 2> s = dataset->get_size(
            (state.offsets[dataset->id] - 1) % dataset->len());
        float raw_scale = static_cast<float>(block.height) / s[1];
        float scale = raw_scale;

        if (raw_scale > typical_scale * 2.0f) { // > x2 scalling
          scale = typical_scale;
        }

        int scaled_w = static_cast<int>(s[0] * scale);
        int scaled_h = static_cast<int>(s[1] * scale);
        int y_offset = block.height - scaled_h; // Anchor to bottom

        if (scaled_w + x > max_x) {
          if (retry_n + 1 > 3) {
            retry_n = 0;
            break;
          }
          retry_n++;
          continue;
        }

        line.push_back(
            {.dataset_id = dataset->id,
             .idx = (int)((state.offsets[dataset->id] - 1) % dataset->len()),
             .page_idx = state.next_page_asset_idx,
             .w = scaled_w,
             .h = scaled_h,
             .x = x,
             .y = block.y_start + y_offset});
        state.next_page_asset_idx++;
        x += scaled_w + ThreadRandom::rand_int(10, 20);
      }
      break;
    }
    case BlockType::Paragraph: {
      int x = ThreadRandom::rand_int(30, 50);
      block.assets.reserve(block.n_lines);
      for (int i = 0; i < block.n_lines; i++) {
        int retry_n = 0;
        block.assets.push_back({});
        auto &line = block.assets.back();

        while (x < settings.w) {
          Dataset *dataset = get_random_dataset(
              datasets, {DatasetType::HandwrittenWords, DatasetType::MathExpr});
          state.offsets[dataset->id]++;
          std::array<int, 2> s = dataset->get_size(
              (state.offsets[dataset->id] - 1) % dataset->len());

          float typical_scale =
              get_typical_scale(*dataset, settings.line_height, state);
          float raw_scale = static_cast<float>(settings.line_height) / s[1];
          float scale = raw_scale;

          if (raw_scale > typical_scale * 2.0f) { // > x2 scalling
            scale = typical_scale;
          }

          int scaled_w = static_cast<int>(s[0] * scale);
          int scaled_h = static_cast<int>(s[1] * scale);
          int y_offset = settings.line_height - scaled_h; // Anchor to bottom

          if (scaled_w + x > settings.w) {
            if (retry_n + 1 > 3) {
              retry_n = 0;
              break;
            }
            retry_n++;
            continue;
          }

          line.push_back(
              {.dataset_id = dataset->id,
               .idx = (int)((state.offsets[dataset->id] - 1) % dataset->len()),
               .page_idx = state.next_page_asset_idx,
               .w = scaled_w,
               .h = scaled_h,
               .x = x,
               .y = block.y_start + y_offset + i * settings.line_height});
          state.next_page_asset_idx++;
          x += scaled_w + ThreadRandom::rand_int(10, 20);
        }
        x = ThreadRandom::rand_int(30, 50);
      }
      break;
    }
    case BlockType::Schema: {
      Dataset *dataset = get_random_dataset(datasets, {DatasetType::Diagram});
      state.offsets[dataset->id]++;
      std::array<int, 2> s =
          dataset->get_size((state.offsets[dataset->id] - 1) % dataset->len());
      float ratio = static_cast<float>(block.height) / static_cast<float>(s[1]);
      if (ratio <= 0.0f)
        ratio = 0.1f;
      int w = ratio * s[0];
      int h = ratio * s[1];
      int offset = (settings.w - w) * block.schema_x_offset;
      block.assets.push_back({});
      auto &line = block.assets.back();

      line.push_back({
          .dataset_id = dataset->id,
          .idx = (int)((state.offsets[dataset->id] - 1) % dataset->len()),
          .page_idx = state.next_page_asset_idx,
          .w = w,
          .h = h,
          .x = offset,
          .y = block.y_start,
      });
      state.next_page_asset_idx++;
      break;
    };
    case BlockType::SkipLine: {
      break;
    }
    }
  }
};
