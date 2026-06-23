#include "datasets.h"
#include <filesystem>
#include <vector>

// IAMWordEntry, path, bbox, transcript, gray_scale
struct IAMWordEntry {
  std::filesystem::path path;
  std::array<int, 4> bbox; // [x, y, w, h]
  std::string transcript;
  uint8_t gray_scale;
};

class IAM : public Dataset {
public:
  IAM(std::filesystem::path p_path, float p_proportion = 1.0f)
      : Dataset("iam", DatasetType::HandwrittenWords, std::move(p_path),
                p_proportion) {}

  cv::Mat get_image(int idx) override;
  AssetRow get_asset(int idx) override;
  void load() override;
  bool valid() override;
  long len() override;

private:
  std::vector<IAMWordEntry> words;
};