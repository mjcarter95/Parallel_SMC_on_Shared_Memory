#ifndef REDUCTION_OPERATIONS_H
#define REDUCTION_OPERATIONS_H

#include <cstddef>

// ======================= Device reductions / transforms =======================
// These operate on arrays already resident on the device (OpenMP target offload).
// Use is_device_ptr(...) in callers when invoking them inside target regions.

// Max over int array -> device scalar output
void d_max_int(const int* array, int n, int* d_out_max);

// Max over double array -> device scalar output
void d_max_double(const double* array, int n, double* d_out_max);

// Log-sum-exp of a log-weight vector (returns host double)
double d_log_sum_exp(const double* logw, int n);

// In-place normalisation of log-weights on device: logw[i] -= logsumexp(logw)
void d_normalise_logw(double* logw, int n);

// Effective Sample Size from (possibly unnormalised) log-weights:
// ESS = 1 / sum_i exp(2*(logw[i]-logsumexp(logw)))
double d_ESS_from_logw(const double* logw, int n);

// Packed layout estimate for the "sum of bits" statistic:
// Returns sum_i [ exp(logw[i]) * sum_{j=0..M[i]-1} X[ base_i + j ] ],
// where base_i = (i==0 ? 0 : csumM[i-1]).
double d_weighted_estimate_sum_bits(const int* X,
                                    const int* M,
                                    const int* csumM,
                                    const double* logw,
                                    int loc_n);

#endif // REDUCTION_OPERATIONS_H

