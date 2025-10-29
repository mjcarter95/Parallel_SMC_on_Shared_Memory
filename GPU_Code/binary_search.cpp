#include "binary_search.h"

// Make these functions available to device code
#pragma omp declare target

// ---------------------------------------------------------------------------
// Naive redistribution (resized samples) binary search
// Mirrors your CPU code exactly, but uses device pointers and no std::vector.
// ---------------------------------------------------------------------------
int d_binary_search_ncopies_csum(const int* ncopies,
                                 const int* csum,
                                 int loc_n,
                                 int limit)
{
    int pivot  = 0;
    int first  = 0;
    int last   = loc_n - 1;
    int middle = (first + last) / 2;

    while (first <= last) {
        if (csum[middle] <= limit) {
            first = middle + 1;
        } else {
            // Check left boundary via csum - ncopies
            if (csum[middle] - ncopies[middle] > limit) {
                last = middle - 1;
            } else {
                // Ensure the "width" at middle is nonzero
                if (ncopies[middle] == 0) {
                    last = middle - 1;
                } else {
                    pivot = middle;
                    break;
                }
            }
        }
        middle = (first + last) / 2;
    }

    return pivot;
}

// ---------------------------------------------------------------------------
// Optimal redistribution binary search using product array1[m]*array2[m]
// (e.g., ncopies * M) and cumulative dot-product 'csum' (i.e., 'cdot').
// ---------------------------------------------------------------------------
int d_binary_search_prod_csum(const int* array1,
                              const int* array2,
                              const int* csum,
                              int loc_n,
                              int limit)
{
    int pivot  = 0;
    int first  = 0;
    int last   = loc_n - 1;
    int middle = (first + last) / 2;

    while (first <= last) {
        if (csum[middle] <= limit) {
            first = middle + 1;
        } else {
            const int width = array1[middle] * array2[middle];
            if (csum[middle] - width > limit) {
                last = middle - 1;
            } else {
                if (width == 0) {
                    last = middle - 1;
                } else {
                    pivot = middle;
                    break;
                }
            }
        }
        middle = (first + last) / 2;
    }

    return pivot;
}

#pragma omp end declare target
