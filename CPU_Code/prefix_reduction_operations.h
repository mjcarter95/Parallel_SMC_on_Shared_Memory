#ifndef PREFIX_REDUCTION_OPERATIONS_H
#define PREFIX_REDUCTION_OPERATIONS_H

#include <vector>

std::vector<double> prefix_sum(std::vector<double> array, unsigned int loc_n, int T);
std::vector<int> prefix_sum(std::vector<int> array, unsigned int loc_n, int T);
std::vector<int> prefix_dot_product(std::vector<int> array1, std::vector<int> array2, unsigned int loc_n, int T);

#endif