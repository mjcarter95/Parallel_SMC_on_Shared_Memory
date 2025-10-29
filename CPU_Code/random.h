#ifndef RANDOM_H
#define RANDOM_H

#include <vector>
#include <random>

double thread_safe_rand(double lb, double ub);
double thread_safe_rand(double lb, double ub, std::mt19937 & mt);

#endif
