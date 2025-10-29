#ifndef SMC_H
#define SMC_H

// Run K SMC iterations with N particles on the GPU.
// - redistribution: 0 = naive (pad → fixed → restore), 1 = optimal (cdot + binary search)
// - time (host out): total wall time (s) for the SMC loop (not including initialisation)
// - red_percentage (host out): fraction of time spent in redistribution
// - seed: RNG seed used to initialise device RNG streams
//
// Returns: final weighted estimate of the "sum of bits" statistic.
double smc_sampler(int N,
                   int K,
                   int T,                 // kept for signature compatibility; not used by GPU path
                   int redistribution,
                   double* time,
                   double* red_percentage,
                   unsigned long long seed = 0ULL);

#endif // SMC_H
