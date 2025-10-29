#ifndef REDUCTION_OPERATIONS_H
#define REDUCTION_OPERATIONS_H

#include <iostream>
#include <vector>

int max(std::vector<int> array, unsigned int loc_n, int T);
double max(std::vector<double> array, unsigned int loc_n, int T);
double log_sum_exp(std::vector<double> logw, unsigned int loc_n, int T);
void normalise(std::vector<double>& logw, unsigned int loc_n, int T);
double ESS(std::vector<double> logw, unsigned int loc_n, int T);
double estimate(std::vector<std::vector<int>> x, std::vector<double> logw, int loc_n, int T, int (*f)(std::vector<int> particle));

#endif // !REDUCTION_OPERATIONS
