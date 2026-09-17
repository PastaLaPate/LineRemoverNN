#include "utils.h"
#include "datasets/factory.h"
#include <format>
#include <iostream>

DatasetGroups construct_datasets(std::vector<DatasetS> &datasets) {
  DatasetGroups groups;
  std::map<DatasetType, float> total_weight_by_type;

  for (auto type : {DatasetType::HandwrittenWords, DatasetType::MathExpr,
                    DatasetType::Diagram}) {
    groups.emplace(type, std::vector<std::unique_ptr<Dataset>>{});
    total_weight_by_type.emplace(type, 0.0f);
  }

  for (const auto &dataset : datasets) {
    total_weight_by_type[get_dataset_type(dataset.id)] += dataset.proportion;
  }

  for (auto &dataset : datasets) {
    if (total_weight_by_type[get_dataset_type(dataset.id)] > 0.0) {
      dataset.proportion = dataset.proportion /
                           total_weight_by_type[get_dataset_type(dataset.id)];
    }
  }

  for (const auto &d : datasets) {
    auto dataset = make_dataset(d); // throws if unknown id
    if (!dataset->valid())
      throw std::invalid_argument("Invalid dataset path: " + d.path.string());
    groups.at(dataset->type).push_back(std::move(dataset));
    std::cout << std::format("Found dataset {} at {}", d.id, d.path.string())
              << std::endl;
  }

  return groups;
};

DatasetLookup make_dataset_lookup(const DatasetGroups &groups) {
  DatasetLookup lookup;
  for (auto &[type, datasets] : groups) {
    for (auto &dataset : datasets) {
      if (lookup.contains(dataset->id)) {
        std::cout << "[DatasetLookupMaker] WARNING: Duplicated dataset id, "
                     "ignoring it."
                  << std::endl;
        continue;
      }
      lookup[dataset->id] = dataset.get();
    }
  }
  return lookup;
}