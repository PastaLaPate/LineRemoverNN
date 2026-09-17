#include "datasets/datasets.h"
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/// DatasetType to list of datasets of this type
using DatasetGroups =
    std::map<DatasetType, std::vector<std::unique_ptr<Dataset>>>;

/// Dataset ID -> Dataset pointer. For rendering pass.
using DatasetLookup = std::unordered_map<std::string, Dataset *>;

DatasetGroups construct_datasets(std::vector<DatasetS> &datasets);

DatasetLookup make_dataset_lookup(const DatasetGroups &groups);