#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "model.h"
#include <random>
#include "importance_sampling.h"
#include "random.h"


void importance_sampling(std::vector<std::vector<int>>& x, std::vector<double>& logw, std::vector<std::vector<int>>& initx,
	std::vector<int>& initM, int loc_n, int T, std::vector<std::mt19937>& mts) {
	omp_set_num_threads(T);
	double cum_probs[3] = { 0.4, 0.8, 1.0 };
	
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < loc_n; i++) {
		logw[i] += -log_pdf(x[i]) + q(x[i], initx[i], initM[i]);
	}

	#pragma omp parallel for schedule(static)
	for (int i = 0; i < loc_n; i++) {
		sampling_from_q(x[i], mts.at(i));

		logw[i] += log_pdf(x[i]) - q(x[i], initx[i], initM[i]);

	}

}

void initialise(std::vector<std::vector<int>>& x, std::vector<double>& logw, std::vector<std::vector<int>>& initx, 
	std::vector<int>& initM, int loc_n, int T, int seed, std::vector<std::mt19937>& mts) {
	omp_set_num_threads(T);

	#pragma omp parallel for schedule(static)
	for (int i = 0; i < loc_n; i++) {
		mts[i].seed(seed + i);
		x[i] = sampling_from_q0(mts.at(i));
		logw[i] = log_pdf(x[i]) - q0(x[i]);
		initM[i] = (int)x[i].size();
		initx[i] = x[i];
	}
}