#ifndef REDISTRIBUTION_H
#define REDISTRIBUTION_H

#include <cstddef>
#include "prefix_reduction_operations.h"
#include "reduction_operations.h"
#include "binary_search.h"

// Packed layout: X length = csumM[loc_n-1]
// Padded layout (naive only): width = maxM, length = loc_n*maxM
// M: int[loc_n], csumM: int[loc_n], ncopies: int[loc_n]

// ---- Naive (packed → padded → fixed → packed) ----
void pad(int*& X, int& x_len,
         const int* M, const int* csumM,
         int loc_n,
         int& maxM);

void fixed_size_redistribution(int* X, const int* ncopies, int loc_n, int maxM);

void restore(int*& X, int& x_len,
             const int* M, int* csumM,
             int loc_n, int maxM);

void naive_variable_size_redistribution(int*& X, int& x_len,
                                        int* M, int* csumM,
                                        const int* ncopies,
                                        int loc_n);

// ---- Optimal (packed → packed, cdot + binary search) ----
// Order: cdot → copy X by k (no external csum) → redistribute M via fixed-size (maxM=1) → recompute csumM
void optimal_variable_size_redistribution(int*& X, int& x_len,
                                          int* M, int* csumM,
                                          const int* ncopies,
                                          int loc_n);

void sequential_redistribution(int*& X, int& x_len,
                                        int* M, int* csumM,
                                        const int* ncopies,
                                        int loc_n);

#endif // REDISTRIBUTION_H

