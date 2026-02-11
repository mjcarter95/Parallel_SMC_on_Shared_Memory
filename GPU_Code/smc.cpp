#include "smc.h"

#include <omp.h>
#include <cmath>
#include <cstddef>
#include <stdio.h>
// Project headers (GPU implementations)
#include "importance_sampling.h"
#include "resampling.h"
#include "reduction_operations.h"
#include "model.h"
#include "random.h"
#include "prefix_reduction_operations.h"
#include "redistribution.h"

// Helper: free device pointer safely
static inline void d_free(void* p) {
    if (p) omp_target_free(p, omp_get_default_device());
}

double smc_sampler(int N,
                   int K,
                   int /*T*/,
                   int redistribution,
                   double* time_total,
                   double* red_percentage,
                   unsigned long long seed)
{
    const int dev = omp_get_default_device();
    if (time_total) *time_total = 0.0;
    if (red_percentage) *red_percentage = 0.0;

    // ----------------------------
    // Device allocations
    // ----------------------------
    // Packed samples
    int* X        = nullptr;  // device pointer; allocated in initialise
    int  x_len    = 0;

    // Per-particle sizes and prefix
    int* M        = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)N, dev));
    int* csumM    = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)N, dev));

    // Initial copies (for q and weights)
    int* initM      = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)N, dev));
    int* initcsumM  = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)N, dev));
    int* initX      = nullptr;  // device pointer; allocated in initialise
    int  init_x_len = 0;

    // Log-weights (device)
    double* logw  = static_cast<double*>(omp_target_alloc(sizeof(double) * (size_t)N, dev));

    // RNG states (device): two 64-bit streams per particle
    unsigned long long* rng_s0 = static_cast<unsigned long long*>(
        omp_target_alloc(sizeof(unsigned long long) * (size_t)N, dev)
    );
    unsigned long long* rng_s1 = static_cast<unsigned long long*>(
        omp_target_alloc(sizeof(unsigned long long) * (size_t)N, dev)
    );

    // ----------------------------
    // Initialise population (on GPU)
    // - seeds RNGs
    // - builds X (packed), M, csumM
    // - saves initX/initM/initcsumM on device
    // - computes initial logw on device
    // ----------------------------
    initialise(/*X*/ X, /*x_len*/ x_len,
               /*M*/ M, /*csumM*/ csumM,
               /*initM*/ initM,
               /*initX*/ initX, /*init_x_len*/ init_x_len,
               /*initcsumM*/ initcsumM,
               /*logw*/ logw,
               /*loc_n*/ N,
               /*rng_s0*/ rng_s0, /*rng_s1*/ rng_s1,
               /*seed*/ seed);

    // ----------------------------
    // SMC loop
    // ----------------------------
    const double Nt = static_cast<double>(N);
    double time_resampling = 0.0;

    const double loop_t0 = omp_get_wtime();
    for (int k = 0; k < K; ++k) {
        // Normalise log-weights (device)
        d_normalise_logw(logw, N);

        // Effective sample size (host scalar; computed from device arrays)
        const double neff = d_ESS_from_logw(logw, N);

        // Resample if ESS < N (same logic as your CPU driver)
        if (neff < Nt) {
            resampling( X,  x_len, M, csumM, logw, N, redistribution, rng_s0, rng_s1, &time_resampling);
        }

        // One step of the proposal / IS weight update (device)
        importance_sampling(X, x_len, M, csumM, initM, initX, init_x_len, initcsumM, logw, N, rng_s0, rng_s1);

        // (Optional) progress
        printf("Iteration %d\n", k);
    }
    const double loop_t1 = omp_get_wtime();

    if (time_total) *time_total = (loop_t1 - loop_t0);
    if (red_percentage && *time_total > 0.0) {
        *red_percentage = time_resampling;
    }

    return 0.0;

    // ----------------------------
    // Final estimate
    // Ensure logw are normalised before estimating
    // ----------------------------
    d_normalise_logw(logw, N);
    const double estimate = d_weighted_estimate_sum_bits(/*X*/ X,
                                                         /*M*/ M,
                                                         /*csumM*/ csumM,
                                                         /*logw*/ logw,
                                                         /*loc_n*/ N);

    // ----------------------------
    // Cleanup
    // ----------------------------
    d_free(X);
    d_free(initX);
    d_free(M);
    d_free(csumM);
    d_free(initM);
    d_free(initcsumM);
    d_free(logw);
    d_free(rng_s0);
    d_free(rng_s1);

    return estimate;
}
