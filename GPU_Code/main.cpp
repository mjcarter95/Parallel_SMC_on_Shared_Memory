// main.cpp (GPU-offload build)
// Usage: ./app <T> <log2N> <K> <redistribution> [num_MC_runs]
//   T: kept for signature compatibility (threads); GPU path ignores it
//   log2N: N = 1 << log2N
//   K: number of SMC iterations
//   redistribution: 0 = naive, 1 = optimal
//   num_MC_runs: optional (default 1)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <omp.h>

#include "smc.h"

using namespace std;

static double median(vector<double> a) {
    const size_t n = a.size();
    if (n == 0) return 0.0;
    if (n % 2 == 1) {
        nth_element(a.begin(), a.begin() + n/2, a.end());
        return a[n/2];
    } else {
        nth_element(a.begin(), a.begin() + n/2,     a.end());
        double hi = a[n/2];
        nth_element(a.begin(), a.begin() + n/2 - 1, a.end());
        double lo = a[n/2 - 1];
        return 0.5 * (lo + hi);
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <T> <log2N> <K> <redistribution> [num_MC_runs]\n";
        return 1;
    }

    const int T              = atoi(argv[1]);           // kept for signature compat
    const int log2N          = atoi(argv[2]);
    const int K              = atoi(argv[3]);
    const int redistribution = atoi(argv[4]);           // 0 = naive, 1 = optimal
    const int num_MC_runs    = (argc >= 6) ? atoi(argv[5]) : 1;

    const int N = 1 << log2N;

    // Optional: bind OpenMP default device (if multiple GPUs present)
    // omp_set_default_device(0);

    // Keep API parity with the old code; not used by GPU path
    omp_set_num_threads(T);

    vector<double> times(num_MC_runs, 0.0);
    vector<double> red_pct(num_MC_runs, 0.0);

    for (int run = 0; run < num_MC_runs; ++run) {
        double time_loop = 0.0;
        double red_percentage = 0.0;

        // You can vary the seed per run to decorrelate timings
        const unsigned long long seed = (unsigned long long)run;

        double est = smc_sampler(/*N*/ N,
                                 /*K*/ K,
                                 /*T*/ T, // kept for signature compat
                                 /*redistribution*/ redistribution,
                                 /*time*/ &time_loop,
                                 /*red_percentage*/ &red_percentage,
                                 /*seed*/ seed);

        (void)est; // estimation value available if you want to print it

        times[run]   = time_loop;
        red_pct[run] = red_percentage;
        // printf("Run %d: time = %.6f s, redistribution = %.2f%%\n",
        //        run, time_loop, 100.0 * red_percentage);
    }

    const double med_time = median(times);

    double avg_red = 0.0;
    for (double p : red_pct) avg_red += p;
    if (num_MC_runs > 0) avg_red /= (double)num_MC_runs;

    if (redistribution == 0)
        printf("Naive parallel redistribution results:\n");
    else if (redistribution == 1)
        printf("Optimal parallel redistribution results:\n");
    else 
        printf("Sequential redistribution results:\n");
    printf("\nMedian time over %d run(s): %.6f s\n", num_MC_runs, med_time);
    printf("Average redistribution share: %.2f%%\n", 100.0 * avg_red);

    return 0;
}
