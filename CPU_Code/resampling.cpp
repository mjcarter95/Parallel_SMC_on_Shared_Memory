#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <random>
#include "random.h"
#include "prefix_reduction_operations.h"
#include "redistribution.h"


std::vector<int> choice(std::vector<double>& logw, unsigned int loc_n, int T, std::mt19937& mt) {
    omp_set_num_threads(T);
    std::vector<double> cdf(loc_n);
    std::vector<int> ncopies(loc_n);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < loc_n; i++) {
        logw[i] = exp(logw[i]) * loc_n;
    }

    cdf = prefix_sum(logw, loc_n, T);    

    //u0 = rand()/double(N);
    double u0 = thread_safe_rand(0.0, 1.0, mt); // / double(loc_n);

    #pragma omp parallel for firstprivate(u0) schedule(static)
    for (int i = 0; i < loc_n; i++) {
        ncopies[i] = (int)(ceil(cdf[i] - u0) - ceil(cdf[i] - logw[i] - u0));
    }

    return ncopies;
}

void reset(std::vector<double>& logw, unsigned int loc_n, int T) {
    omp_set_num_threads(T);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < loc_n; i++) {
        logw[i] = log(1 / double(loc_n));
    }
    return;
}


void resampling(std::vector<std::vector<int>>& x, std::vector<double>& logw, unsigned int loc_n, int T, std::mt19937& mt, int redistribution, double *time) {
    std::vector<int> ncopies(loc_n);

    ncopies = choice(logw, loc_n, T, mt);
    
    double start = omp_get_wtime();
    if (redistribution == 0) {
        naive_redistribution(x, ncopies, loc_n, T);
    }
    else {
        dynamic_redistribution(x, ncopies, loc_n, T);
    }
    double end = omp_get_wtime();

    *time = *time + (end - start);

    reset(logw, loc_n, T);
}