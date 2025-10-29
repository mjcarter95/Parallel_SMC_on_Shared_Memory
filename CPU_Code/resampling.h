#ifndef RESAMPLING_H
#define RESAMPLING_H

#include <vector>

std::vector<int> choice(std::vector<double>& logw, unsigned int loc_n, int T, std::mt19937& mt);
void reset(std::vector<double>& logw, unsigned int loc_n, int T);
void resampling(std::vector<std::vector<int>>& x, std::vector<double>& logw, unsigned int loc_n, int T, std::mt19937& mt, int redistribution, double *time);

#endif