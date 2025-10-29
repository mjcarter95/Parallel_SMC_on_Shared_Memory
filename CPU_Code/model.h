#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <random>

int binomial_coefficient(int n, int k);
double poisson_lpmf(int x, double lambda);
double binomial_lpmf(int x, int n, double p);
double log_pdf(std::vector<int> coins);
int f(std::vector<int> x);
double q0(std::vector<int> x);
std::vector<int> sampling_from_q0(std::mt19937& mt);
double q(std::vector<int> x, std::vector<int> initx, int initM);
void sampling_from_q(std::vector<int>& x, std::mt19937& mt);
int get_mid_size();

#endif