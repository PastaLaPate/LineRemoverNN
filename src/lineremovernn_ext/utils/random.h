#include <random>

class ThreadRandom {
public:
  static int rand_int(int lo, int hi);
  static float rand_float(float lo = 0.0f, float hi = 0.0f);

  template <typename InputIt>
  static size_t sample_weighted(InputIt begin, InputIt end) {
    std::discrete_distribution<size_t> dist(begin, end);
    return dist(get_engine());
  }

  static std::mt19937 &get_engine();
};