#ifndef SMC_H
#define SMC_H

double calculateStandardDeviation(std::vector<double>& data);
double smc_sampler(int N, int K, int T, int redistribution, double *time, double *red_percentage, int seed = 0);

#endif