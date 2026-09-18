#include "utils.h"
#include "datasets/factory.h"
#include "logging/python_logger.h"
#include <format>
#include <iostream>

void Dataset::log_debug(const std::string &message) const {
  if (logger)
    logger->debug(message);
  else
    std::clog << message << '\n';
}

void Dataset::log_info(const std::string &message) const {
  if (logger)
    logger->info(message);
  else
    std::clog << message << '\n';
}

void Dataset::log_warning(const std::string &message) const {
  if (logger)
    logger->warning(message);
  else
    std::clog << message << '\n';
}

void Dataset::log_error(const std::string &message) const {
  if (logger)
    logger->error(message);
  else
    std::clog << message << '\n';
}

DatasetGroups construct_datasets(std::vector<DatasetS> &datasets,
                                 PythonLoggerBridge &logger) {
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
    dataset->set_logger(&logger);
    groups.at(dataset->type).push_back(std::move(dataset));
    logger.info(std::format("Found dataset {} at {}", d.id, d.path.string()));
  }

  return groups;
};

DatasetLookup make_dataset_lookup(const DatasetGroups &groups) {
  DatasetLookup lookup;
  for (auto &[type, datasets] : groups) {
    for (auto &dataset : datasets) {
      if (lookup.contains(dataset->id)) {
        throw std::invalid_argument("Duplicated dataset id: " + dataset->id);
      }
      lookup[dataset->id] = dataset.get();
    }
  }
  return lookup;
}