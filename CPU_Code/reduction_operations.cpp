#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "model.h"


int max(std::vector<int> array, unsigned int loc_n, int T) {
    omp_set_num_threads(T);
    int glob_max = array[0];

    #pragma omp parallel for reduction (max:glob_max)
    for (int i = 0; i < loc_n; i++) {
        if (array[i] > glob_max) glob_max = array[i];
    }

    return glob_max;
}

double max(std::vector<double> array, unsigned int loc_n, int T) {
    omp_set_num_threads(T);
    double glob_max = array[0];

    #pragma omp parallel for reduction (max:glob_max)
    for (int i = 0; i < loc_n; i++) {
        if (array[i] > glob_max) glob_max = array[i];
    }

    return glob_max;
}

double log_sum_exp(std::vector<double> array, unsigned int loc_n, int T) {
    omp_set_num_threads(T);
    double total = 0;
    double maximum = max(array, loc_n, T);

    
    #pragma omp parallel for firstprivate(maximum) reduction(+:total)
    for(int i = 0; i < loc_n; i++){
        total += exp(array[i] - maximum);
    }
    return log(total) + maximum;
}

void normalise(std::vector<double>& logw, unsigned int loc_n, int T) {
    omp_set_num_threads(T);
    double log_wsum = log_sum_exp(logw, loc_n, T);

    //printf("log sum exp = %lf\n", log_wsum);

    #pragma omp parallel for firstprivate(log_wsum) schedule(static)
    for(int i = 0; i < loc_n; i++){
        logw[i] = logw[i] - log_wsum;
    }
}

double ESS(std::vector<double> logw, unsigned int loc_n, int T) {
    omp_set_num_threads(T);
    double sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < loc_n; i++) {
        sum += exp(2 * (logw[i]));
    }
    return 1 / sum;
}

double estimate(std::vector<std::vector<int>> x, std::vector<double> logw, int loc_n, int T, int (*f)(std::vector<int> particle)) {
    omp_set_num_threads(T);
    double sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < loc_n; i++) {
        sum += logw[i] * (double)f(x[i]);
    }

    return sum;
}