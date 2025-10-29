#include <omp.h>
#include <vector>
#include "reduction_operations.h"
#include "prefix_sum_load_balancing.h"

void pad(std::vector<std::vector<int>>& x, int loc_n, int maxM, int T) {
	omp_set_num_threads(T);

	#pragma omp parallel for firstprivate(maxM) schedule(static)
	for (int i = 0; i < loc_n; i++) {
		for (int j = x[i].size(); j < maxM; j++) {
			x[i].push_back(-1);
		}
	}

}

void restore(std::vector<std::vector<int>>& x, int loc_n, int T) {
	omp_set_num_threads(T);
	int M;

	#pragma omp parallel for private(M) schedule(static)
	for (int i = 0; i < loc_n; i++) {
		M = x[i].size();
		
		for (int j = M - 1; j >= 0; j--) {
			if (x[i][j] == -1) {
				x[i].pop_back();
			}
		}
	}
	
}

void naive_redistribution(std::vector<std::vector<int>>& x, std::vector<int> ncopies, int loc_n, int T) {
	omp_set_num_threads(T);
	std::vector<int> M(loc_n);

	#pragma omp parallel for schedule(static)
	for (int i = 0; i < loc_n; i++) {
		M[i] = (int)x[i].size();
	}

	int maxM = max(M, loc_n, T);

	pad(x, loc_n, maxM, T);

	static_prefix_sum_load_balancing_redistribution(x, ncopies, loc_n, T);

	restore(x, loc_n, T);
}

void dynamic_redistribution(std::vector<std::vector<int>>& x, std::vector<int> ncopies, int loc_n, int T) {
	omp_set_num_threads(T);

	std::vector<int> M(loc_n);
	
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < loc_n; i++) {
		M[i] = (int)x[i].size();
	}

	dynamic_prefix_sum_load_balancing_redistribution(x, ncopies, M, loc_n, T);
}