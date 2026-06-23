#include "factory.h"

std::unique_ptr<Dataset> make_dataset(const DatasetS &d) {
  if (d.id == "iam")
    return std::make_unique<IAM>(d.path, d.proportion);
  if (d.id == "mathwriting")
    return std::make_unique<MathWriting>(d.path, d.proportion);
  if (d.id == "ai2d")
    return std::make_unique<AI2D>(d.path, d.proportion);
  throw std::invalid_argument("Unknown dataset id: \"" + d.id + "\"");
}