#include "resampling.h"
#include "prefix_reduction_operations.h"
#include "redistribution.h"
#include "random.h"

#include <omp.h>
#include <cmath>
#include <stdio.h>

// Build ncopies on device using the standard "systematic-like" formula on logw.
static void d_choice_from_logw(const double* logw,
                               int* ncopies,
                               int loc_n,
                               unsigned long long* rng_s0,
                               unsigned long long* rng_s1)
{
    if (loc_n <= 0) return;

    // 1) w_scaled = exp(logw[i]) * N
    double* w_scaled = static_cast<double*>(
        omp_target_alloc(sizeof(double) * (size_t)loc_n, omp_get_default_device())
    );

    #pragma omp target teams distribute parallel for \
        is_device_ptr(w_scaled, logw)
    for (int i = 0; i < loc_n; ++i) {
        w_scaled[i] = std::exp(logw[i]) * (double)loc_n;
    }

    // 2) cdf = prefix sum of w_scaled (inclusive)
    double* cdf = static_cast<double*>(
        omp_target_alloc(sizeof(double) * (size_t)loc_n, omp_get_default_device())
    );
    #pragma omp target teams distribute parallel for is_device_ptr(cdf, w_scaled)
    for (int i = 0; i < loc_n; ++i) cdf[i] = w_scaled[i];
    prefix_sum_double_inplace_device(cdf, loc_n);

    // 3) One device-wide uniform u0 ~ U(0,1)
    double* d_u0 = static_cast<double*>(
        omp_target_alloc(sizeof(double), omp_get_default_device())
    );
    #pragma omp target is_device_ptr(d_u0, rng_s0, rng_s1)
    {
        d_u0[0] = rng_uniform01(rng_s0[0], rng_s1[0]);
    }

    // 4) Compute ncopies
    #pragma omp target teams distribute parallel for \
        is_device_ptr(ncopies, cdf, w_scaled, d_u0)
    for (int i = 0; i < loc_n; ++i) {
        const double u0 = d_u0[0];
        const double a  = std::ceil(cdf[i] - u0);
        const double b  = std::ceil((cdf[i] - w_scaled[i]) - u0);
        ncopies[i] = (int)(a - b);
    }

    omp_target_free(w_scaled, omp_get_default_device());
    omp_target_free(cdf,      omp_get_default_device());
    omp_target_free(d_u0,     omp_get_default_device());
}

// Device helper: reset logw to uniform in log-space
static void d_reset_logw(double* logw, int loc_n)
{
    if (loc_n <= 0) return;
    #pragma omp target teams distribute parallel for is_device_ptr(logw)
    for (int i = 0; i < loc_n; ++i) {
        logw[i] = std::log(1.0 / (double)loc_n);
    }
}

void resampling(int*& X, int& x_len,
                int* M, int* csumM,
                double* logw,
                int loc_n,
                int redistribution, // 0=naive, 1=optimal
                unsigned long long* rng_s0,
                unsigned long long* rng_s1,
                double* time_sec)
{
    if (loc_n <= 0) return;

    const int dev = omp_get_default_device();

    // Allocate ncopies on device
    int* ncopies = static_cast<int*>(
        omp_target_alloc(sizeof(int) * (size_t)loc_n, dev)
    );

    // Build ncopies from logw on device
    d_choice_from_logw(logw, ncopies, loc_n, rng_s0, rng_s1);

    // --- Check if the total number of copies is equal to loc_n ---
    /*int ncopies_sum = 0;
    #pragma omp target teams distribute parallel for reduction(+:ncopies_sum) map(to: ncopies[0:loc_n]) firstprivate(loc_n)
    for (int i = 0; i < loc_n; i++) ncopies_sum += ncopies[i];

    if (ncopies_sum != loc_n) {
        printf("[FATAL] Sum(ncopies) = %d, expected %d\n", ncopies_sum, loc_n);
        fflush(stdout);
        exit(EXIT_FAILURE);
    }
    else {
        printf("[INFO] Sum(ncopies) OK: %d\n", ncopies_sum);
        fflush(stdout);
    }*/
    // ----------------------------


    // Time redistribution (optional)
    double t0 = 0.0, t1 = 0.0;
    if (time_sec) t0 = omp_get_wtime();

    if (redistribution == 0) {
        // Naive: pad → fixed → restore (reallocates X and updates M, csumM)
        naive_variable_size_redistribution(X, x_len, M, csumM, ncopies, loc_n);
    } else if (redistribution == 1)  {
        // Optimal: cdot + binary search (reallocates X and updates M, csumM)
        optimal_variable_size_redistribution(X, x_len, M, csumM, ncopies, loc_n);
    }

    else {
        sequential_redistribution(X, x_len, M, csumM, ncopies, loc_n);
    }

    if (time_sec) { t1 = omp_get_wtime(); *time_sec += (t1 - t0); }

    // Reset log-weights to uniform
    d_reset_logw(logw, loc_n);

    // Free
    omp_target_free(ncopies, dev);
}

