#include "datasets.h"
#include <array>
#include <filesystem>
#include <opencv2/core/mat.hpp>
#include <vector>

// IAMWordEntry, path, bbox, transcript, gray_scale
struct IAMWordEntry {
  std::filesystem::path path;
  std::array<int, 4> bbox; // [x, y, w, h]
  std::string transcript;
  uint8_t gray_scale;
  uint64_t blob_offset = 0;
  uint32_t blob_length = 0;
};

class IAM : public Dataset {
public:
  IAM(std::filesystem::path p_path, float p_proportion = 1.0f,
      size_t max_cache_mb = 1024)
      : Dataset("iam", DatasetType::HandwrittenWords, std::move(p_path),
                p_proportion) {}

  cv::Mat get_image(int idx) override;
  AssetRow get_asset(int idx) override;
  std::array<int, 2> get_size(int idx) override;
  void load() override;
  bool valid() override;
  long len() override;

private:
  std::vector<IAMWordEntry> words;

  int blob_fd = -1;
  std::vector<cv::Mat> preloaded_images;

  cv::Mat read_image(int idx, bool preloading = false);

  void generate_blob();
  void read_blob();
};