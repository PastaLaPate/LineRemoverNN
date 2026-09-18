#include "generation/serializer.h"
#include <unordered_map>

std::unordered_map<BlockType, std::string> block_type_to_string = {
    {BlockType::Title, "title"},
    {BlockType::CatTitle, "category_title"},
    {BlockType::Paragraph, "paragraph"},
    {BlockType::Schema, "schema"},
    {BlockType::SkipLine, "line_skip"}};

pugi::xml_document serialize_xml(int page_idx, const PageSettings &settings,
                                 const Layout &layout) {

  pugi::xml_document doc;
  pugi::xml_node page = doc.append_child("page");
  page.append_attribute("idx") = page_idx;
  page.append_attribute("w") = settings.w;
  page.append_attribute("h") = settings.h;
  page.append_attribute("line_height") = settings.line_height;
  page.append_attribute("brightness") = settings.brightness;

  for (const auto &block : layout) {
    pugi::xml_node block_node = page.append_child("block");
    block_node.append_attribute("type") = block_type_to_string[block.type];
    block_node.append_attribute("y_start") = block.y_start;
    block_node.append_attribute("height") = block.height;
    switch (block.type) {
    case BlockType::Title:
    case BlockType::CatTitle: {
      pugi::xml_node line_node = block_node.append_child("line");
      line_node.append_attribute("y") = block.y_start;
      for (const auto &asset : block.assets[0]) {
        pugi::xml_node word_node = line_node.append_child("word");
        word_node.append_attribute("idx") = asset.page_idx;
        word_node.append_attribute("dataset_idx") = asset.idx;
        word_node.append_attribute("dataset_id") = asset.dataset_id.c_str();
        word_node.append_attribute("x") = asset.x;
        word_node.append_attribute("y") = asset.y;
        word_node.append_attribute("w") = asset.w;
        word_node.append_attribute("h") = asset.h;
        word_node.text() = asset.transcript.c_str();
      }
      break;
    }
    case BlockType::Paragraph: {
      for (const auto &line : block.assets) {
        pugi::xml_node line_node = block_node.append_child("line");
        line_node.append_attribute("y") = line[0].y;
        for (const auto &asset : line) {
          pugi::xml_node word_node = line_node.append_child("word");
          word_node.append_attribute("idx") = asset.page_idx;
          word_node.append_attribute("dataset_idx") = asset.idx;
          word_node.append_attribute("dataset_id") = asset.dataset_id.c_str();
          word_node.append_attribute("x") = asset.x;
          word_node.append_attribute("y") = asset.y;
          word_node.append_attribute("w") = asset.w;
          word_node.append_attribute("h") = asset.h;
          word_node.text() = asset.transcript.c_str();
        }
      }
      break;
    }
    case BlockType::Schema: {
      pugi::xml_node schema_node = block_node.append_child("schema");
      auto &asset = block.assets[0][0];
      schema_node.append_attribute("idx") = asset.page_idx;
      schema_node.append_attribute("dataset_idx") = asset.idx;
      schema_node.append_attribute("dataset_id") = asset.dataset_id.c_str();
      schema_node.append_attribute("x") = asset.x;
      schema_node.append_attribute("y") = asset.y;
      schema_node.append_attribute("w") = asset.w;
      schema_node.append_attribute("h") = asset.h;
      break;
    }
    case BlockType::SkipLine: {
      block_node.append_attribute("line_skipped") = block.line_skipped;
      break;
    }
    }
  }
  return doc;
};