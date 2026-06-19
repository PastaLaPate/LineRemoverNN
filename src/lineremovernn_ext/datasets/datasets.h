#pragma once
#include <filesystem>
#include <opencv2/core/mat.hpp>
#include <string>

struct AssetRow {
  int idx;
  std::string dataset;
  cv::Mat image;
  std::string transcript;
};

struct DatasetS {
  std::string id;
  std::filesystem::path path;
  float proportion;
};

class Dataset {
public:
  std::string id;
  std::filesystem::path path;
  float proportion;

  Dataset(std::string p_id, std::filesystem::path p_path)
      : id(std::move(p_id)), path(std::move(p_path)), proportion(1.0f) {}

  Dataset(std::string p_id, std::filesystem::path p_path, float p_proportion)
      : id(std::move(p_id)), path(std::move(p_path)), proportion(p_proportion) {
  }

  virtual cv::Mat get_image(int idx) = 0;
  virtual AssetRow get_asset(int idx) = 0;
  virtual void load() = 0; // Create internal structure mappings
  virtual bool valid() = 0;
  virtual long len() = 0;

  virtual ~Dataset() = default;
};