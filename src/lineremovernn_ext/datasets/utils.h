#include "datasets/datasets.h"
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/// DatasetType to list of datasets of this type
using DatasetGroups =
    std::map<DatasetType, std::vector<std::unique_ptr<Dataset>>>;

/// Dataset ID -> Dataset pointer.
using DatasetLookup = std::unordered_map<std::string, Dataset *>;

DatasetLookup make_dataset_lookup(DatasetGroups &groups);