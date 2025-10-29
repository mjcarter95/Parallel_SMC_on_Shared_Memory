#ifndef BINARY_SEARCH_H
#define BINARY_SEARCH_H

#include <cstddef>

// ============================================================================
// Device-callable binary searches (OpenMP target offload).
// All pointers must already be present on the device.
// You can call these from within any #pragma omp target/teams/parallel region.
// ============================================================================

// Naive redistribution case (resized samples):
//   pivot is the smallest index 'middle' such that
//     csum[middle] > limit
//   and also
//     csum[middle] - ncopies[middle] <= limit
//   and ncopies[middle] != 0
//
// Parameters:
//   ncopies : int[loc_n]   (device pointer)
//   csum    : int[loc_n]   (device pointer) — typically the cumulative sum of ncopies
//   loc_n   : length
//   limit   : search threshold
//
// Returns: pivot index (>=0); defaults to 0 if no pivot is found (to match CPU code).
int d_binary_search_ncopies_csum(const int* ncopies,
                                 const int* csum,
                                 int loc_n,
                                 int limit);

// Optimal redistribution case (product array1*array2):
//   Same logic as above but using product array1[middle]*array2[middle] instead of ncopies[middle].
//
// Parameters:
//   array1 : int[loc_n]    (device pointer) — e.g., ncopies
//   array2 : int[loc_n]    (device pointer) — e.g., M
//   csum   : int[loc_n]    (device pointer) — typically cumulative dot product (cdot)
//   loc_n  : length
//   limit  : search threshold
//
// Returns: pivot index (>=0); defaults to 0 if no pivot is found (to match CPU code).
int d_binary_search_prod_csum(const int* array1,
                              const int* array2,
                              const int* csum,
                              int loc_n,
                              int limit);

#endif // BINARY_SEARCH_H
