#include "datasets.h"
#include <filesystem>
#include <vector>

class MathWriting : public Dataset {
public:
  MathWriting(std::filesystem::path p_path, float p_proportion = 1.0f)
      : Dataset("mathwriting", std::move(p_path), p_proportion) {}

  cv::Mat get_image(int idx) override;
  void load() override;
  bool valid() override;
  long len() override;

private:
  std::vector<std::filesystem::path> assets;
};