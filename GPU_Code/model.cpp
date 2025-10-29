#include "model.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include "prefix_reduction_operations.h"
#include "random.h"  // rng_uniform01, rng_uniform_int

#pragma omp declare target
double prior_threshold = 0.99;
int    min_dim_range[2] = {1, 3};
int    max_dim_range[2] = {698, 700};
double cum_probs[3]     = {0.4, 0.8, 1.0};

static inline double dlog(double x) { return std::log(x); }

double intervals_lpmf_M(int M)
{
    double logprob = 0.0;
    if (M >= min_dim_range[0] && M <= min_dim_range[1]) {
        logprob += dlog(prior_threshold);
        logprob -= dlog((double)min_dim_range[1] - (double)min_dim_range[0] + 1.0);
    } else {
        logprob += dlog(1.0 - prior_threshold);
        logprob -= dlog((double)max_dim_range[1] - (double)min_dim_range[0] + 1.0);
    }
    logprob -= (double)M * dlog(2.0);
    return logprob;
}

double q_by_sizes(int M, int initM)
{
    if (M <= initM) return dlog(0.5) + dlog(cum_probs[0]);
    return dlog(cum_probs[1]); // M > initM
}
#pragma omp end declare target

// Host helper: read last int from a device array (len > 0)
static inline int device_read_last_int(const int* d_arr, int len) {
    int out = 0;
    omp_target_memcpy(&out, d_arr + (len - 1), sizeof(int),
                      0, 0, omp_get_initial_device(), omp_get_default_device());
    return out;
}

// ------------------------------
// sampling_from_q0 (GPU)  -- NO logw here anymore
// ------------------------------
void sampling_from_q0(int*& X, int& x_len,
                      int* M, int* csumM,
                      int* initM,
                      int*& initX, int& init_x_len,
                      int* initcsumM,
                      int loc_n,
                      unsigned long long* rng_s0,
                      unsigned long long* rng_s1)
{
    if (loc_n <= 0) { X = nullptr; x_len = 0; return; }

    const int dev = omp_get_default_device();

    // 1) Draw M[i] from the prior on device
    #pragma omp target teams distribute parallel for \
        is_device_ptr(M, rng_s0, rng_s1)
    for (int i = 0; i < loc_n; ++i) {
        const double u = rng_uniform01(rng_s0[i], rng_s1[i]);
        if (u < prior_threshold) {
            M[i] = rng_uniform_int(rng_s0[i], rng_s1[i], min_dim_range[0], min_dim_range[1]);
        } else {
            M[i] = rng_uniform_int(rng_s0[i], rng_s1[i], max_dim_range[0], max_dim_range[1]);
        }
    }

    // 2) csumM = prefix sum of M
    #pragma omp target teams distribute parallel for is_device_ptr(csumM, M)
    for (int i = 0; i < loc_n; ++i) csumM[i] = M[i];
    prefix_sum_int_inplace_device(csumM, loc_n);
    const int total = device_read_last_int(csumM, loc_n);

    // 3) Allocate packed X and fill with random bits
    if (X) omp_target_free(X, dev);
    X = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)total, dev));
    x_len = total;

    #pragma omp target teams distribute parallel for \
        is_device_ptr(X, M, csumM, rng_s0, rng_s1)
    for (int i = 0; i < loc_n; ++i) {
        const int Mi   = M[i];
        const int base = (i == 0) ? 0 : csumM[i - 1];
        #pragma omp parallel for
        for (int j = 0; j < Mi; ++j) {
            X[base + j] = (rng_uniform01(rng_s0[i], rng_s1[i]) > 0.5) ? 1 : 0;
        }
    }

    // 4) Save initial copies on device: initM, initcsumM, initX
    #pragma omp target teams distribute parallel for is_device_ptr(initM, M)
    for (int i = 0; i < loc_n; ++i) initM[i] = M[i];

    #pragma omp target teams distribute parallel for is_device_ptr(initcsumM, csumM)
    for (int i = 0; i < loc_n; ++i) initcsumM[i] = csumM[i];

    if (initX) omp_target_free(initX, dev);
    initX = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)total, dev));
    init_x_len = total;

    #pragma omp target teams distribute parallel for is_device_ptr(initX, X)
    for (int k = 0; k < total; ++k) initX[k] = X[k];

    // Note: logw is intentionally NOT computed here anymore.
}

// ------------------------------
// sampling_from_q (GPU) — unchanged
// ------------------------------
void sampling_from_q(int*& X, int& x_len,
                     int* M, int* csumM,
                     int loc_n,
                     unsigned long long* rng_s0,
                     unsigned long long* rng_s1)
{
    if (loc_n <= 0) return;

    const int dev = omp_get_default_device();

    // Snapshot old M and csumM
    int* M_old = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)loc_n, dev));
    int* csumM_old = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)loc_n, dev));

    #pragma omp target teams distribute parallel for is_device_ptr(M_old, M)
    for (int i = 0; i < loc_n; ++i) M_old[i] = M[i];

    #pragma omp target teams distribute parallel for is_device_ptr(csumM_old, csumM)
    for (int i = 0; i < loc_n; ++i) csumM_old[i] = csumM[i];

    // Draw r and compute new sizes
    #pragma omp target teams distribute parallel for \
        is_device_ptr(M, M_old, rng_s0, rng_s1)
    for (int i = 0; i < loc_n; ++i) {
        const double r = rng_uniform01(rng_s0[i], rng_s1[i]);
        const int Mi_old = M_old[i];
        if (r <= cum_probs[0]) {
            M[i] = Mi_old + 1;
        } else if (r < cum_probs[1]) {
            M[i] = (Mi_old > 1) ? (Mi_old - 1) : Mi_old;
        } else {
            M[i] = Mi_old;
        }
    }

    // Build new csumM and get new total length
    #pragma omp target teams distribute parallel for is_device_ptr(csumM, M)
    for (int i = 0; i < loc_n; ++i) csumM[i] = M[i];
    prefix_sum_int_inplace_device(csumM, loc_n);
    const int new_total = device_read_last_int(csumM, loc_n);

    // Allocate temp_x and fill new samples
    int* temp_x = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)new_total, dev));

    #pragma omp target teams distribute parallel for \
        is_device_ptr(X, temp_x, M, M_old, csumM_old, csumM, rng_s0, rng_s1)
    for (int i = 0; i < loc_n; ++i) {
        const int Mi_old   = M_old[i];
        const int Mi_new   = M[i];
        const int old_base = (i == 0) ? 0 : csumM_old[i - 1];
        const int new_base = (i == 0) ? 0 : csumM[i - 1];

        const double r2 = rng_uniform01(rng_s0[i], rng_s1[i]);

        if (r2 <= cum_probs[0]) {
            // grow
            #pragma omp parallel for
            for (int j = 0; j < Mi_old; ++j) temp_x[new_base + j] = X[old_base + j];
            const int bit = (rng_uniform01(rng_s0[i], rng_s1[i]) > 0.5) ? 1 : 0;
            temp_x[new_base + Mi_old] = bit;

        } else if (r2 < cum_probs[1]) {
            // shrink
            const int copy_len = (Mi_old > 1) ? (Mi_old - 1) : Mi_old;
            #pragma omp parallel for
            for (int j = 0; j < copy_len; ++j) temp_x[new_base + j] = X[old_base + j];

        } else {
            // flip
            #pragma omp parallel for
            for (int j = 0; j < Mi_old; ++j) temp_x[new_base + j] = X[old_base + j];
            if (Mi_old > 0) {
                const int jflip = rng_uniform_int(rng_s0[i], rng_s1[i], 0, Mi_old - 1);
                temp_x[new_base + jflip] = 1 - temp_x[new_base + jflip];
            }
        }
        (void)Mi_new;
    }

    // Replace X with temp_x
    int* newX = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)new_total, dev));
    #pragma omp target teams distribute parallel for is_device_ptr(newX, temp_x)
    for (int k = 0; k < new_total; ++k) newX[k] = temp_x[k];

    if (X) omp_target_free(X, dev);
    X = newX;
    x_len = new_total;

    // Free temporaries
    omp_target_free(temp_x,    dev);
    omp_target_free(M_old,     dev);
    omp_target_free(csumM_old, dev);
}
