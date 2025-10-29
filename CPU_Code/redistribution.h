#ifndef REDISTRIBUTION_H
#define REDISTRIBUTION_H

#include <vector>

void pad(std::vector<std::vector<int>>& x, int loc_n, int maxM, int T);
void restore(std::vector<std::vector<int>>& x, int loc_n, int T);
void naive_redistribution(std::vector<std::vector<int>>& x, std::vector<int> ncopies, int loc_n, int T);
void dynamic_redistribution(std::vector<std::vector<int>>& x, std::vector<int> ncopies, int loc_n, int T);

#endif
