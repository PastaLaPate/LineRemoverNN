#pragma once
#include "datasets.h"
#include "iam.h"
#include <memory>

std::unique_ptr<Dataset> make_dataset(const DatasetS &d);