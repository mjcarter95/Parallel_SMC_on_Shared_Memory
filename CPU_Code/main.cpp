/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

 /*
  * File:   main.cpp
  * Author: alessandro
  *
  * Created on 7 giugno 2024, 11.46
  */

#include <iostream>
#include <omp.h>
#include <random>
#include <vector>
#include <algorithm>
#include "reduction_operations.h"
#include "importance_sampling.h"
#include "resampling.h"
#include "prefix_reduction_operations.h"
#include "redistribution.h"
#include "smc.h"

using namespace std;

double median(vector<double> a) {
    int n = a.size();
    // If size of the arr[] is even 
    if (n % 2 == 0) {
        // Applying nth_element on n/2th index 
        nth_element(a.begin(), a.begin() + n / 2, a.end());

        // Applying nth_element on (n-1)/2 th index 
        nth_element(a.begin(), a.begin() + (n - 1) / 2, a.end());

        // Find the average of value at index N/2 and (N-1)/2 
        return (double)(a[(n - 1) / 2] + a[n / 2]) / 2.0;
    }

    // If size of the arr[] is odd 
    else {
        // Applying nth_element on n/2 
        nth_element(a.begin(), a.begin() + n / 2, a.end());

        // Value at index (N/2)th is the median 
        return (double)a[n / 2];
    }
}

int main(int argc, char** argv) {
    int T = atoi(argv[1]), loc_n = 1 << atoi(argv[2]), K = atoi(argv[3]);
    int redistribution = atoi(argv[4]);
    int num_MC_runs = 1;
    vector<double> times(num_MC_runs);
    double* time = new double, *percent = new double;

    omp_set_num_threads(T);


    for (int i = 0; i < num_MC_runs; i++) {
        double est = smc_sampler(loc_n, K, T, redistribution, time, percent, i); 
        //printf("MC run = %u\n", i);
        times[i] = *time;
    }

    printf("\nMedian = %e\n", median(times));

}
