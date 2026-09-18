#pragma once
#include <filesystem>
#include <opencv2/core/mat.hpp>
#include <string>

class PythonLoggerBridge;

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
  bool preload;
  bool index;
};

enum class DatasetType {
  HandwrittenWords,
  MathExpr,
  Diagram,
};

class Dataset {
public:
  std::string id;
  enum DatasetType type;
  std::filesystem::path path;
  float proportion;
  bool preload = false;
  bool index = false;
  PythonLoggerBridge *logger = nullptr;

  Dataset(std::string p_id, enum DatasetType type, std::filesystem::path p_path)
      : id(std::move(p_id)), type(type), path(std::move(p_path)),
        proportion(1.0f) {}

  Dataset(std::string p_id, enum DatasetType type, std::filesystem::path p_path,
          float p_proportion, bool index, bool preload)
      : id(std::move(p_id)), type(type), path(std::move(p_path)),
        proportion(p_proportion), index(index), preload(preload) {}

  virtual cv::Mat get_image(int idx) = 0;
  virtual AssetRow get_asset(int idx) = 0;
  virtual std::array<int, 2> get_size(int idx) = 0;
  virtual void load() = 0; // Create internal structure mappings
  virtual bool valid() = 0;
  virtual long len() = 0;

  void set_logger(PythonLoggerBridge *p_logger) { logger = p_logger; }
  void log_debug(const std::string &message) const;
  void log_info(const std::string &message) const;
  void log_warning(const std::string &message) const;
  void log_error(const std::string &message) const;

  virtual ~Dataset() = default;
};