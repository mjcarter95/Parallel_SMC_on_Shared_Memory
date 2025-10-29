#ifndef PREFIX_SUM_LOAD_BALANCING_H
#define PREFIX_SUM_LOAD_BALANCING_H

#include <vector>

int binary_search(std::vector<int> ncopies, std::vector<int> csum, int loc_n, int limit);
void static_prefix_sum_load_balancing_redistribution(std::vector<std::vector<int>>& x, std::vector<int> ncopies, int loc_n, int T);
void dynamic_prefix_sum_load_balancing_redistribution(std::vector<std::vector<int>>& x,
    std::vector<int> ncopies, std::vector<int> M, int loc_n, int T);

#endif