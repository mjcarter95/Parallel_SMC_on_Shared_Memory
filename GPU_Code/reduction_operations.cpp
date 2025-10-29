#include "reduction_operations.h"
#include <omp.h>
#include <cmath>
#include <limits>

// --------------------------------- d_max_int ---------------------------------
void d_max_int(const int* array, int n, int* d_out_max)
{
    if (n <= 0) {
        // write a sentinel value on device
        #pragma omp target is_device_ptr(d_out_max)
        { d_out_max[0] = std::numeric_limits<int>::min(); }
        return;
    }

    int h_max = std::numeric_limits<int>::min();
    #pragma omp target teams distribute parallel for is_device_ptr(array) reduction(max:h_max)
    for (int i = 0; i < n; ++i) {
        if (array[i] > h_max) h_max = array[i];
    }

    // store back to device scalar
    #pragma omp target is_device_ptr(d_out_max)
    { d_out_max[0] = h_max; }
}

// ------------------------------- d_max_double --------------------------------
void d_max_double(const double* array, int n, double* d_out_max)
{
    if (n <= 0) {
        #pragma omp target is_device_ptr(d_out_max)
        { d_out_max[0] = -std::numeric_limits<double>::infinity(); }
        return;
    }

    double h_max = -std::numeric_limits<double>::infinity();
    #pragma omp target teams distribute parallel for is_device_ptr(array) reduction(max:h_max)
    for (int i = 0; i < n; ++i) {
        if (array[i] > h_max) h_max = array[i];
    }

    #pragma omp target is_device_ptr(d_out_max)
    { d_out_max[0] = h_max; }
}

// ------------------------------- d_log_sum_exp -------------------------------
double d_log_sum_exp(const double* logw, int n)
{
    if (n <= 0) return -std::numeric_limits<double>::infinity();

    // 1) max
    double maxv = -std::numeric_limits<double>::infinity();
    #pragma omp target teams distribute parallel for is_device_ptr(logw) reduction(max:maxv)
    for (int i = 0; i < n; ++i) {
        if (logw[i] > maxv) maxv = logw[i];
    }

    // 2) sum exp(logw - max)
    double sum = 0.0;
    #pragma omp target teams distribute parallel for is_device_ptr(logw) firstprivate(maxv) reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += std::exp(logw[i] - maxv);
    }

    return std::log(sum) + maxv;
}

// ----------------------------- d_normalise_logw ------------------------------
void d_normalise_logw(double* logw, int n)
{
    if (n <= 0) return;
    const double lse = d_log_sum_exp(logw, n);

    #pragma omp target teams distribute parallel for is_device_ptr(logw) firstprivate(lse)
    for (int i = 0; i < n; ++i) {
        logw[i] -= lse;
    }
}

// ------------------------------- d_ESS_from_logw -----------------------------
double d_ESS_from_logw(const double* logw, int n)
{
    if (n <= 0) return 0.0;

    const double lse = d_log_sum_exp(logw, n);

    double sum_sq = 0.0;
    #pragma omp target teams distribute parallel for is_device_ptr(logw) firstprivate(lse) reduction(+:sum_sq)
    for (int i = 0; i < n; ++i) {
        const double wi = std::exp(logw[i] - lse); // normalised weight
        sum_sq += wi * wi;
    }

    return (sum_sq > 0.0) ? (1.0 / sum_sq) : 0.0;
}

// ---------------------- d_weighted_estimate_sum_bits -------------------------
double d_weighted_estimate_sum_bits(const int* X,
                                    const int* M,
                                    const int* csumM,
                                    const double* logw,
                                    int loc_n)
{
    if (loc_n <= 0) return 0.0;

    // If logw aren't normalised, result is proportional; normalise outside if needed.
    double total = 0.0;

    #pragma omp target teams distribute parallel for reduction(+:total) \
        is_device_ptr(X, M, csumM, logw)
    for (int i = 0; i < loc_n; ++i) {
        const int Mi   = M[i];
        const int base = (i == 0) ? 0 : csumM[i - 1];

        int s_i = 0;
        for (int j = 0; j < Mi; ++j) {
            s_i += X[base + j];
        }

        const double wi = std::exp(logw[i]); // assumes logw are normalised elsewhere
        total += wi * (double)s_i;
    }

    return total;
}
