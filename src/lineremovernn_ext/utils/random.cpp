#include "random.h"
#include <random>

int ThreadRandom::rand_int(int lo, int hi) {
  return std::uniform_int_distribution<int>(lo, hi)(get_engine());
};

float ThreadRandom::rand_float(float lo, float hi) {
  return std::uniform_real_distribution<float>(lo, hi)(get_engine());
};

std::mt19937 &ThreadRandom::get_engine() {
  thread_local std::mt19937 rng(std::random_device{}());
  return rng;
};