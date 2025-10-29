#ifndef RANDOM_H
#define RANDOM_H

#include <cstddef>

// ================================================================
// Device RNG API (xoroshiro128** family + splitmix64 seeding)
// ================================================================
//
// You can create as many independent RNG state arrays as you like
// (e.g., one for IS, another for resampling) by allocating separate
// s0/s1 buffers and seeding them with different seeds.
//
// All functions are device-callable with OpenMP target offload.
// ================================================================

// Seed per-particle RNG states on the device.
void rng_seed_all(unsigned long long* s0, unsigned long long* s1,
                  int count, unsigned long long seed);

// Draw U ~ Uniform[0,1)
double rng_uniform01(unsigned long long& s0, unsigned long long& s1);

// Draw integer in [lo, hi] (inclusive)
int    rng_uniform_int(unsigned long long& s0, unsigned long long& s1,
                       int lo, int hi);

#endif // RANDOM_H
