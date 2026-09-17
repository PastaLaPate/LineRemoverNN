#include <random>

class ThreadRandom {
public:
  static int rand_int(int lo, int hi);
  static float rand_float(float lo = 0.0f, float hi = 0.0f);

  static std::mt19937 &get_engine();
};