#include "datasets.h"
#include <array>
#include <filesystem>
#include <list>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <unordered_map>
#include <unordered_set>
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
  IAM(std::filesystem::path p_path, float p_proportion = 1.0f,
      size_t max_cache_mb = 1024)
      : Dataset("iam", DatasetType::HandwrittenWords, std::move(p_path),
                p_proportion),
        max_cache_bytes_(max_cache_mb * 1024 * 1024) {}

  cv::Mat get_image(int idx) override;
  AssetRow get_asset(int idx) override;
  std::array<int, 2> get_size(int idx) override;
  void load() override;
  bool valid() override;
  long len() override;

private:
  std::vector<IAMWordEntry> words;

  enum class QueueType { IN_FIFO, MAIN_LRU };

  struct CacheNode {
    cv::Mat img;
    QueueType type;
    std::list<int>::iterator it;
    size_t bytes;
  };

  mutable std::mutex cache_mutex_;
  size_t max_cache_bytes_;
  size_t current_cache_bytes_ = 0;
  static constexpr int PREFETCH_WINDOW = 5;

  std::list<int> q_in_;
  std::list<int> q_main_;
  std::unordered_map<int, CacheNode> cache_;
  std::unordered_set<int> in_flight_;

  cv::Mat load_and_process_disk(int idx);
  void trigger_prefetch(int current_idx);
  void evict_unlocked();
};