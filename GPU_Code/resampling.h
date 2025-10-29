#ifndef RESAMPLING_H
#define RESAMPLING_H

#include <cstddef>

// Resampling on GPU, packed storage.
// - X (device) is reallocated inside (pass by reference) and x_len updated.
// - M, csumM (device) are updated by redistribution.
// - logw (device) is used to form ncopies and then reset to log(1/N).
// - redistribution: 0 = naive (pad → fixed → restore), 1 = optimal (cdot + binary search).
// - rng_s0/rng_s1: device RNG states (already seeded by caller).
// - time_sec (host ptr, optional): adds elapsed wall time of redistribution section.
void resampling(int*& X, int& x_len,
                int* M, int* csumM,
                double* logw,
                int loc_n,
                int redistribution, // 0=naive, 1=optimal
                unsigned long long* rng_s0,
                unsigned long long* rng_s1,
                double* time_sec = nullptr);

#endif // RESAMPLING_H
