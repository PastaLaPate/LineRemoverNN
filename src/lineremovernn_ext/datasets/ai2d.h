#include "datasets.h"
#include <filesystem>
#include <vector>

class AI2D : public Dataset {
public:
  AI2D(std::filesystem::path p_path, float p_proportion = 1.0f)
      : Dataset("ai2d", DatasetType::Diagram, std::move(p_path), p_proportion) {
  }

  cv::Mat get_image(int idx) override;
  AssetRow get_asset(int idx) override;
  void load() override;
  bool valid() override;
  long len() override;

private:
  std::vector<std::filesystem::path> images;
};