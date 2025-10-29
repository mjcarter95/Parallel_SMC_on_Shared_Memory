#include <vector>
#include <random>

double thread_safe_rand(double lb, double ub) {
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_real_distribution<double> distribution(lb, ub);

	return distribution(generator);
}


double thread_safe_rand(double lb, double ub, std::mt19937& mt) {
	std::uniform_real_distribution<double> distribution(lb, ub);
	return distribution(mt);
}