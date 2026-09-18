#include "layout.h"
#include "utils/random.h"

Layout generate_document_layout(const PageSettings &settings) {
  Layout blocks;

  int top_margin = ThreadRandom::rand_int(30, std::min(settings.h, 100));
  int y = top_margin;

  if (ThreadRandom::rand_float(0.f, 1.f) < 0.66f) {
    float title_scale = ThreadRandom::rand_float(1.2f, 1.5f);
    int title_h = static_cast<int>(settings.line_height * title_scale);

    blocks.push_back({.type = BlockType::Title,
                      .y_start = y,
                      .height = title_h,
                      .n_lines = 1});
    y += title_h;

    if (y + settings.line_height * 2 <= settings.h) {
      blocks.push_back(
          {.type = BlockType::SkipLine, .y_start = y, .line_skipped = 2});
      y += settings.line_height * 2;
    }
  }

  BlockType last_type = BlockType::SkipLine;

  while (y + settings.line_height <= settings.h) {

    BlockType current_type;
    while (true) {
      float choice = ThreadRandom::rand_float(0.f, 1.f);
      if (choice < 0.1f) {
        if (last_type != BlockType::CatTitle) {
          current_type = BlockType::CatTitle;
          break;
        }
      } else if (choice < 0.9f) { // 0.1 + 0.8 = 0.9
        current_type = BlockType::Paragraph;
        break;
      } else { // Remaining 0.1
        current_type = BlockType::Schema;
        break;
      }
    }

    if (current_type == BlockType::CatTitle) {
      float cat_scale = ThreadRandom::rand_float(1.3f, 2.0f);
      int cat_h = static_cast<int>(settings.line_height * cat_scale);

      if (y + cat_h > settings.h)
        break;

      blocks.push_back({.type = BlockType::CatTitle,
                        .y_start = y,
                        .height = cat_h,
                        .n_lines = 1});
      y += cat_h;
      last_type = BlockType::CatTitle;

    } else if (current_type == BlockType::Paragraph) {
      int remaining_lines = (settings.h - y) / settings.line_height;
      if (remaining_lines < 1)
        break;

      int min_lines = std::min(3, remaining_lines);
      int max_lines = std::min(8, remaining_lines);
      int n_lines = ThreadRandom::rand_int(min_lines, max_lines);

      blocks.push_back({.type = BlockType::Paragraph,
                        .y_start = y,
                        .height = settings.line_height * n_lines,
                        .n_lines = n_lines});
      y += settings.line_height * n_lines;
      last_type = BlockType::Paragraph;

    } else if (current_type == BlockType::Schema) {
      float schema_ratio = ThreadRandom::rand_float(0.6f, 0.9f);
      float x_offset = ThreadRandom::rand_float(0, 1);
      int height = ThreadRandom::rand_int(300, 500);

      blocks.push_back({.type = BlockType::Schema,
                        .y_start = y,
                        .height = height,
                        .schema_x_offset = x_offset});
      y += height;
      last_type = BlockType::Schema;
    }

    if (y + settings.line_height <= settings.h) {
      blocks.push_back(
          {.type = BlockType::SkipLine, .y_start = y, .line_skipped = 1});
      y += settings.line_height;
    }
  }

  return blocks;
}

Layout generate_page_layout(const PageSettings &settings) {

  Layout blocks;

  int top_margin = ThreadRandom::rand_int(30, std::min(settings.h, 100));
  int y = top_margin;

  while (y + settings.line_height <= settings.h) {
    int remaining = (settings.h - y) / settings.line_height;
    assert(remaining >= 1 && "rand_int would receive lo < hi");

    int n_lines = ThreadRandom::rand_int(1, std::min(4, remaining));
    blocks.push_back({.type = BlockType::Paragraph,
                      .y_start = y,
                      .height = settings.line_height * n_lines,
                      .n_lines = n_lines});

    y += settings.line_height * n_lines + ThreadRandom::rand_int(20, 40);

    if (ThreadRandom::rand_float() > 0.5 &&
        y + settings.line_height < settings.h) {
      int remaining = (settings.h - y) / settings.line_height;
      assert(remaining >= 1 && "rand_int would receive lo < hi");

      int n_lines = ThreadRandom::rand_int(1, std::min(4, remaining));
      blocks.push_back({.type = BlockType::SkipLine,
                        .y_start = y,
                        .height = settings.line_height * n_lines,
                        .line_skipped = n_lines});

      y += settings.line_height * n_lines;
    }
  }

  return blocks;
}

Layout generate_layout(const PageSettings &settings) {
  return settings.document ? generate_document_layout(settings)
                           : generate_page_layout(settings);
}
