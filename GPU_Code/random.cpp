#include "random.h"
#include <omp.h>

// ======================= Device-side RNG =======================
#pragma omp declare target

// ---- helpers ----
static inline unsigned long long rotl64(unsigned long long x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline unsigned long long splitmix64_next(unsigned long long& x) {
    unsigned long long z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline unsigned long long rng_next_u64(unsigned long long& s0,
                                              unsigned long long& s1) {
    // xoroshiro128** style step
    unsigned long long s00 = s0;
    unsigned long long s01 = s1;

    unsigned long long result = rotl64(s00 * 5ULL, 7) * 9ULL;
    s01 ^= s00;
    s0 = rotl64(s00, 24) ^ s01 ^ (s01 << 16);
    s1 = rotl64(s01, 37);

    return result;
}

// ---- public API ----
double rng_uniform01(unsigned long long& s0, unsigned long long& s1) {
    unsigned long long r = rng_next_u64(s0, s1);
    // Top 53 bits to double in [0,1)
    return (double)((r >> 11) * (1.0 / 9007199254740992.0));
}

int rng_uniform_int(unsigned long long& s0, unsigned long long& s1,
                    int lo, int hi) {
    // Normalize bounds defensively
    if (hi < lo) { int t = lo; lo = hi; hi = t; }

    // Width in 32-bit domain (your ranges are small, so this is safe)
    unsigned int width = (unsigned int)((hi - lo) + 1);

    // Use multiply-high scaling with a 32-bit slice from the 64-bit RNG
    // This avoids slow/quirky 64-bit modulus in device code paths.
    unsigned int x = (unsigned int)(rng_next_u64(s0, s1) >> 32);
    unsigned int scaled = (unsigned int)(
        ((unsigned long long)x * (unsigned long long)width) >> 32
    );

    return lo + (int)scaled;
}

#pragma omp end declare target
// ==================== end device-side RNG ======================


// ================ Host-side seeding on the GPU =================
// s0/s1 must be device pointers allocated via omp_target_alloc
void rng_seed_all(unsigned long long* s0, unsigned long long* s1,
                  int count, unsigned long long seed) {
    #pragma omp target teams distribute parallel for \
        is_device_ptr(s0, s1) firstprivate(seed, count)
    for (int i = 0; i < count; ++i) {
        unsigned long long x = seed ^
            (0x9E3779B97F4A7C15ULL * (unsigned long long)(i + 1));
        unsigned long long a = splitmix64_next(x);
        unsigned long long b = splitmix64_next(x);
        if (a == 0ULL && b == 0ULL) b = 1ULL; // avoid all-zero state
        s0[i] = a;
        s1[i] = b;
    }
}
