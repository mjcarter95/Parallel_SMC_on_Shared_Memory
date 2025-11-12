#ifndef PREFIX_REDUCTION_OPERATIONS_H
#define PREFIX_REDUCTION_OPERATIONS_H

#include <cstddef>

// ============================================================================
// GPU (OpenMP target) inclusive prefix operations on device-resident arrays.
// All pointers must already be present on the device.
// No host<->device mappings happen inside these functions.
// ============================================================================

// Inclusive prefix sum (scan) computed entirely on the GPU (in-place).
void prefix_sum_int_inplace_device(int* data, int n, int tile = 1024);
void prefix_sum_double_inplace_device(double* data, int n, int tile = 1024);

// Inclusive prefix dot-product of two integer arrays on the GPU.
// out[i] := sum_{k=0..i} (a1[k] * a2[k])
// Implementation: out[i] = a1[i]*a2[i] (parallel), then in-place prefix sum on 'out'.
void prefix_dot_product_inplace_device(const int* a1, const int* a2,
                                       int* out, int n, int tile = 1024);

bool d_validate_csumM_inclusive(const int* M,
                                       const int* csumM,
                                       int loc_n,
                                       int device);

#endif // PREFIX_REDUCTION_OPERATIONS_H
