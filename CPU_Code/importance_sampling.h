#ifndef IMPORTANCE_SAMPLING_H
#define IMPORTANCE_SAMPLING_H

#include <vector>
#include <random>

void initialise(std::vector<std::vector<int>>& x, std::vector<double>& logw, std::vector<std::vector<int>>& initx,
	std::vector<int>& initM, int loc_n, int T, int seed, std::vector<std::mt19937>& mts);
void importance_sampling(std::vector<std::vector<int>>& x, std::vector<double>& logw, std::vector<std::vector<int>>& initx,
	std::vector<int>& initM, int loc_n, int T, std::vector<std::mt19937>& mts);

#endif