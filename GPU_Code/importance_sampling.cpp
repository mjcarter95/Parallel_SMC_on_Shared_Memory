#include "importance_sampling.h"
#include "model.h"
#include "random.h"
#include <omp.h>

// ------------------------------
// initialise (GPU)  -- now computes logw here
// ------------------------------
void initialise(int*& X, int& x_len,
                int* M, int* csumM,
                int* initM,
                int*& initX, int& init_x_len,
                int* initcsumM,
                double* logw,                   // still passed to initialise
                int loc_n,
                unsigned long long* rng_s0,
                unsigned long long* rng_s1,
                unsigned long long seed)
{
    // Seed per-particle RNG states on the device
    rng_seed_all(rng_s0, rng_s1, loc_n, seed);

    // Build initial packed population and initial copies (NO logw here)
    sampling_from_q0(X, x_len,
                     M, csumM,
                     initM,
                     initX, init_x_len,
                     initcsumM,
                     loc_n,
                     rng_s0, rng_s1);

    // Compute initial logw on device:
    // logw[i] = intervals_lpmf_M(M[i]) - intervals_lpmf_M(M[i])  (typically 0)
    #pragma omp target teams distribute parallel for \
        is_device_ptr(logw, M)
    for (int i = 0; i < loc_n; ++i) {
        const double lp  = intervals_lpmf_M(M[i]);
        const double lq0 = intervals_lpmf_M(M[i]); // q0 by size
        logw[i] = lp - lq0;
    }
}

// ------------------------------
// importance_sampling (GPU) — unchanged
// ------------------------------
void importance_sampling(int*& X, int& x_len,
                         int* M, int* csumM,
                         const int* initM,
                         const int* /*initX*/, int /*init_x_len*/,
                         const int* /*initcsumM*/,
                         double* logw,
                         int loc_n,
                         unsigned long long* rng_s0,
                         unsigned long long* rng_s1)
{
    if (loc_n <= 0) return;

    // Pre-weight update on device
    #pragma omp target teams distribute parallel for \
        is_device_ptr(logw, M, initM)
    for (int i = 0; i < loc_n; ++i) {
        const double prior = intervals_lpmf_M(M[i]);
        const double qv    = q_by_sizes(M[i], initM[i]);
        logw[i] += -prior + qv;
    }

    // Proposal step
    sampling_from_q(X, x_len, M, csumM, loc_n, rng_s0, rng_s1);

    // Post-weight update on device
    #pragma omp target teams distribute parallel for \
        is_device_ptr(logw, M, initM)
    for (int i = 0; i < loc_n; ++i) {
        const double prior = intervals_lpmf_M(M[i]);
        const double qv    = q_by_sizes(M[i], initM[i]);
        logw[i] += prior - qv;
    }
}
