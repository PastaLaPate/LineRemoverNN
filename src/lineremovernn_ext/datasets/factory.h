#pragma once
#include "ai2d.h"
#include "datasets.h"
#include "iam.h"
#include "mathwriting.h"
#include <memory>

std::unique_ptr<Dataset> make_dataset(const DatasetS &d);
DatasetType get_dataset_type(const std::string &id);