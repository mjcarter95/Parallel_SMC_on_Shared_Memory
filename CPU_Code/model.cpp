#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>
#include "random.h"
using namespace std;

double mu = 5.0;
double sigma = 1.0;
double lambda = 2.0;
int r = 2;
int rMax = 40;
double prior_threshold = 0.99; // Probability of having small particles, i.e., with dimension in min_dim_range. 
int min_dim_range[2] = { 1, 3 };
int max_dim_range[2] = { 698, 700}; //Increase this will make the STD between small and big particles grow

int mid_size = (max_dim_range[0] - min_dim_range[1]) / 2;

double cum_probs[3] = { 0.4, 0.8, 1.0 };

int get_mid_size() {
    return mid_size;
}

int binomial_coefficient(int n, int k) {
    if(k == 0)
        return 1;

    int temp1 = n - k + 1, temp2;
    for(int i = 1; i < k; i++) {
        temp1 = (temp2 = temp1) * (n - k + 1 + i) / (i + 1);
    }

    return temp1;
}

double binomial_lpmf(int x, int n, double p) {
    //double binom = ((double)binomial_coefficient(x, n));
    //return log(binom * pow(p, (double)x) * pow(1 - p, (double)(n - x)));

    return lgamma((double)n + 1.0) - lgamma((double)x + 1.0) - lgamma((double)x + (double)n + 1.0) + (double)x * log(p) + ((double)n - (double)x) * log(1.0 - p);
}

double poisson_lpmf(int x, double lambda) {
    return (double)x * log(lambda) - lambda - lgamma((double)x + 1.0);
}

double intervals_lpmf(int n) {
    double logprob = 0.0;

    if (n >= min_dim_range[0] && n <= min_dim_range[1]) {
        logprob += log(prior_threshold); // probability of being small
        logprob -= log((double)min_dim_range[1] - (double)min_dim_range[0] + 1.0); // normalised by range size, because it is a mass function
        //p(being small) * p(M | being small)
    }
    else {
        logprob += log(1.0 - prior_threshold); // probability of being big
        logprob -= log((double)max_dim_range[1] - (double)min_dim_range[0] + 1.0); // normalised by range size, because it is a mass function
    }

    logprob -= n * log(2); // p(combination of heads and tails)

    return logprob;
}

double log_pdf(std::vector<int> coins) {
    double lpdf = 1.0;
    int n = coins.size();  //int(mu * mu / (sigma * sigma - mu));
    double p = 0.5; //mu / (sigma * sigma);
    int num_heads = 0;
    for(int i = 1; i < coins.size(); i++) {
        num_heads += coins[i];
    }
    //printf("x=%d, n=%d, p=%lf\n", num_heads, n, p);
    double prior = intervals_lpmf(n); //poisson_lpmf(n, lambda); 
    //double likelihood = binomial_lpmf(num_heads, n, p);
    //printf("prior = %lf, likelihood = %lf\n", prior, likelihood);
    return prior;
}

int f(std::vector<int> x) {
    return accumulate(x.begin(), x.end(), 0);
}

double q0(std::vector<int> x) {
    double logprob = 0.0;
    int M = x.size();

    if (M >= min_dim_range[0] && M <= min_dim_range[1]) {
        logprob += log(prior_threshold); // probability of being small
        logprob -= log((double)min_dim_range[1] - (double)min_dim_range[0] + 1.0); // normalised by range size, because it is a mass function
        //p(being small) * p(M | being small)
    }
    else {
        logprob += log(1.0 - prior_threshold); // probability of being big
        logprob -= log((double)max_dim_range[1] - (double)min_dim_range[0] + 1.0); // normalised by range size, because it is a mass function
    }

    logprob -= M * log(2); // p(combination of heads and tails)

    return logprob;
}

std::vector<int> sampling_from_q0(std::mt19937& mt) {
    int num_coins;
    double r = thread_safe_rand(0.0, 1.0, mt);

    if (r < prior_threshold) {
        num_coins = (int)thread_safe_rand((double)min_dim_range[0], (double)min_dim_range[1], mt);
    }
    else {
        num_coins = (int)thread_safe_rand((double)max_dim_range[0], (double)max_dim_range[1], mt);
    }

    std::vector<int> x(num_coins, 0);
    for (int j = 0; j < num_coins; j++) {
        x[j] = (int)round(thread_safe_rand(0.0, 1.0, mt));
    }

    return x;
}

double q(std::vector<int> x, std::vector<int> initx, int initM) {
    double logprob = 0.0;
    int M = x.size();

    if (M <= initM) {
        return log(0.5) + log(cum_probs[0]);
    }
    else if (M > initM) {
        return log(cum_probs[1]);
    }
    else if (x != initx) {
        return log(cum_probs[2]);
    }
    else {
        return -std::numeric_limits<double>::infinity();
    }
}

void sampling_from_q(std::vector<int>& x, std::mt19937& mt) {
    double r = thread_safe_rand(0.0, 1.0, mt);
    int M = x.size();

    if (r <= cum_probs[0]) {
        //x[i].push_back((thread_safe_rand(0.0, 1.0, mts.at(i)) > 0.5) ? 1 : 0);
        x.push_back((thread_safe_rand(0.0, 1.0, mt) > 0.5) ? 1 : 0);
    }
    else if (r < cum_probs[1] && M > 1) {
        x.pop_back();
    }
    else {
        //int j = (int)round(thread_safe_rand(0.0, (double)(M-1), mts.at(i)));
        int j = (int)round(thread_safe_rand(0.0, (double)(M - 1), mt));
        x[j] = 1 - x[j];
    }
}