#ifndef IMPORTANCE_SAMPLING_H
#define IMPORTANCE_SAMPLING_H

#include <cstddef>

// Initialise on GPU (packed):
// - Seeds RNG states on device (outside here; pass already-seeded states, or seed before call).
// - Calls sampling_from_q0(...) to build X/M/csumM and initial copies, and set logw.
void initialise(int*& X, int& x_len,
                int* M, int* csumM,
                int* initM,
                int*& initX, int& init_x_len,
                int* initcsumM,
                double* logw,
                int loc_n,
                unsigned long long* rng_s0,
                unsigned long long* rng_s1,
                unsigned long long seed);

// One IS iteration on GPU (packed):
// - Pre-weight update: logw += -prior(M) + q_by_sizes(M, initM)
// - sampling_from_q(...) updates X, x_len, M, csumM
// - Post-weight update: logw += +prior(M) - q_by_sizes(M, initM)
void importance_sampling(int*& X, int& x_len,
                         int* M, int* csumM,
                         const int* initM,
                         const int* initX, int init_x_len, // currently unused (kept for compatibility)
                         const int* initcsumM,             // currently unused (kept for compatibility)
                         double* logw,
                         int loc_n,
                         unsigned long long* rng_s0,
                         unsigned long long* rng_s1);

#endif // IMPORTANCE_SAMPLING_H
