#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

/*TODO: 
    1) Implement one prefix_sum function with template for double or int input arguments.
    2) Change prefix_sum name to prefix_reduction
    3) Make first for loop in prefix_sum and prefix_dot_product a separate procedure
    4) Add function input, either partial_sum or partial_dot_product, to prefix_reduction 
        such that it computes a generic prefix function
*/

std::vector<double> prefix_sum(std::vector<double> array, unsigned int loc_n, int T) {
    omp_set_num_threads(T);
    std::vector<double> csum(loc_n);
    int i, height = (T == 1) ? 1 : (int)(log(T) / log(2)), dist, size = (int)(loc_n / T);
    double *partial_sum = new double[T](), *partial_csum = new double[T]();

    #pragma omp parallel for firstprivate(size) schedule(static)
    for (int i = 0; i < loc_n; i++) {
        if (i % size == 0) {
            csum[i] = array[i];
        }
        else
            csum[i] = csum[i - 1] + array[i];
    }

    #pragma omp parallel for firstprivate(size) schedule(static)
    for (int my_id = 0; my_id < T; my_id++) {
        partial_csum[my_id] = csum[my_id * size + size - 1];
        partial_sum[my_id] = csum[my_id * size + size - 1];
    }

    for (i = 0, dist = 1; i < height; i++, dist *= 2) {
        #pragma omp parallel firstprivate(size) num_threads(T)
        {
            int my_id = omp_get_thread_num(), partner = my_id ^ dist;
            
            double temp = partial_sum[partner];

            partial_csum[my_id] = (my_id > partner) ? partial_csum[my_id] + temp : partial_csum[my_id];

            #pragma omp barrier
            partial_sum[my_id] += temp;

            if (i == height - 1)
                partial_csum[my_id] -= csum[my_id * size + size - 1];
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < loc_n; i++) {
        csum[i] += partial_csum[omp_get_thread_num()];
    }

    delete[] partial_sum;
    delete[] partial_csum;

    return csum;
}

std::vector<int> prefix_sum(std::vector<int> array, unsigned int loc_n, int T) {
    omp_set_num_threads(T);
    std::vector<int> csum(loc_n);
    int i, height = (T == 1) ? 1 : (int)(log(T) / log(2)), dist, size = (int)(loc_n / T);
    int* partial_sum = new int[T](), * partial_csum = new int[T]();

    #pragma omp parallel for firstprivate(size) schedule(static)
    for (int i = 0; i < loc_n; i++) {
        if (i % size == 0) {
            csum[i] = array[i];
        }
        else
            csum[i] = csum[i - 1] + array[i];
    }

    #pragma omp parallel for firstprivate(size) schedule(static)
    for (int my_id = 0; my_id < T; my_id++) {
        partial_csum[my_id] = csum[my_id * size + size - 1];
        partial_sum[my_id] = csum[my_id * size + size - 1];
    }

    for (i = 0, dist = 1; i < height; i++, dist *= 2) {
        #pragma omp parallel firstprivate(size) num_threads(T)
        {
            int my_id = omp_get_thread_num(), partner = my_id ^ dist;

            int temp = partial_sum[partner];

            partial_csum[my_id] = (my_id > partner) ? partial_csum[my_id] + temp : partial_csum[my_id];

            #pragma omp barrier
            partial_sum[my_id] += temp;

            if (i == height - 1)
                partial_csum[my_id] -= csum[my_id * size + size - 1];
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < loc_n; i++) {
        csum[i] += partial_csum[omp_get_thread_num()];
    }

    delete[] partial_sum;
    delete[] partial_csum;

    return csum;
}

std::vector<int> prefix_dot_product(std::vector<int> array1, std::vector<int> array2, unsigned int loc_n, int T) {
    omp_set_num_threads(T);
    std::vector<int> csum(loc_n);
    int i, height = (T == 1) ? 1 : (int)(log(T) / log(2)), dist, size = (int)(loc_n / T);
    int* partial_sum = new int[T](), * partial_csum = new int[T]();

    #pragma omp parallel for firstprivate(size) schedule(static)
    for (int i = 0; i < loc_n; i++) {
        if (i % size == 0) {
            csum[i] = array1[i] * array2[i];
        }
        else
            csum[i] = csum[i - 1] + array1[i] * array2[i];
    }

    #pragma omp parallel for firstprivate(size) schedule(static)
    for (int my_id = 0; my_id < T; my_id++) {
        partial_csum[my_id] = csum[my_id * size + size - 1];
        partial_sum[my_id] = csum[my_id * size + size - 1];
    }

    for (i = 0, dist = 1; i < height; i++, dist *= 2) {
        #pragma omp parallel firstprivate(size) num_threads(T)
        {
            int my_id = omp_get_thread_num(), partner = my_id ^ dist;

            int temp = partial_sum[partner];

            partial_csum[my_id] = (my_id > partner) ? partial_csum[my_id] + temp : partial_csum[my_id];

            #pragma omp barrier
            partial_sum[my_id] += temp;

            if (i == height - 1)
                partial_csum[my_id] -= csum[my_id * size + size - 1];
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < loc_n; i++) {
        csum[i] += partial_csum[omp_get_thread_num()];
    }

    delete[] partial_sum;
    delete[] partial_csum;

    return csum;
}