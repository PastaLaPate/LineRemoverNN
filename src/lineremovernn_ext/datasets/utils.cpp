#include "utils.h"
#include <iostream>

DatasetLookup make_dataset_lookup(DatasetGroups &groups) {
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