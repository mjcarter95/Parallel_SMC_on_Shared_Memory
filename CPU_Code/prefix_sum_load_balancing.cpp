#include <stdio.h>
#include <omp.h>
#include <vector>
#include <stdlib.h>
#include "prefix_reduction_operations.h"

int binary_search(std::vector<int> ncopies, std::vector<int> csum, int loc_n, int limit) {
    int pivot = 0, first = 0, last = loc_n - 1;
    int middle = (first + last) / 2;

    while (first <= last) {
        if (csum[middle] <= limit) 
            first = middle + 1;
        else {
            if (csum[middle] - ncopies[middle] > limit) 
                last = middle - 1;
            else {
                if (ncopies[middle] == 0) last = middle - 1;
                else {
                    pivot = middle;
                    break;
                }
            }
        }
        middle = (first + last) / 2;
    }

    return pivot;
}

int binary_search(std::vector<int> array1, std::vector<int> array2, std::vector<int> csum, int loc_n, int limit) {
    int pivot = 0, first = 0, last = loc_n - 1;
    int middle = (first + last) / 2;

    while (first <= last) {
        if (csum[middle] <= limit)
            first = middle + 1;
        else {
            if (csum[middle] - array1[middle] * array2[middle] > limit)
                last = middle - 1;
            else {
                if ((array1[middle] * array2[middle]) == 0) last = middle - 1;
                else {
                    pivot = middle;
                    break;
                }
            }
        }
        middle = (first + last) / 2;
    }

    return pivot;
}



void static_prefix_sum_load_balancing_redistribution(std::vector<std::vector<int>>& x, std::vector<int> ncopies, int loc_n, int T) {
    omp_set_num_threads(T);
    std::vector<int> csum(loc_n);
    int* pivot = new int[T](), M = x[0].size(), cdot_before = 0, cdot_after = 0;
    std::vector<std::vector<int>> temp_x(loc_n);

    csum = prefix_sum(ncopies, loc_n, T);

    #pragma omp parallel num_threads(T)
    {
        int my_id = omp_get_thread_num(), limit = my_id * loc_n / T;

        pivot[my_id] = binary_search(ncopies, csum, loc_n, limit);
    }


    #pragma omp parallel firstprivate(M) num_threads(T)
    {
        int my_id = omp_get_thread_num(), chunk_size = loc_n / T;
        int count, i;
        int copies_on_pivot; 
        if (T > 1) {
            copies_on_pivot = std::min(chunk_size, csum[pivot[my_id]] - my_id * chunk_size);
        }
        else
            copies_on_pivot = ncopies[pivot[my_id]];

        for (count = 0; count < copies_on_pivot; count++) {
            M = x[pivot[my_id]].size();
            for (int j = 0; j < M; j++) {
                temp_x[my_id * chunk_size + count].push_back(x[pivot[my_id]][j]);
            }
        }
        i = pivot[my_id] + 1;

        while (count < chunk_size) {
            for (int k = 0; k < ncopies[i] && count < chunk_size; k++) {
                M = x[i].size();
                for (int j = 0; j < M; j++) {
                    temp_x[my_id * chunk_size + count].push_back(x[i][j]);
                }
                count++;
            }
            i++;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < loc_n; i++) {
        x[i].resize(temp_x[i].size());
        x[i] = temp_x[i];
    }

    delete[] pivot;
}

void dynamic_prefix_sum_load_balancing_redistribution(std::vector<std::vector<int>>& x, 
    std::vector<int> ncopies, std::vector<int> M, int loc_n, int T) {
    
    omp_set_num_threads(T);

    std::vector<int> cdot(loc_n);
    std::vector<int> csum(loc_n);
    std::vector<int> newM(loc_n);
    
    cdot = prefix_dot_product(ncopies, M, loc_n, T);
    csum = prefix_sum(ncopies, loc_n, T);
    
    int total_workload = cdot[loc_n - 1];
    
    int* temp_x = new int[total_workload](), * pivots = new int[T](), * workloads = new int[T](),
        * cumulative_workloads = new int[T]();
    std::vector<std::vector<int>> coordinates(total_workload, std::vector<int>(2));
    
    #pragma omp parallel firstprivate(total_workload) num_threads(T)
    {
        int twl_mod_T = total_workload % T;
        int id = omp_get_thread_num(), workload = (id + 1 <= twl_mod_T)? total_workload / T + 1: total_workload / T,
            cumulative_workload = (id + 1 <= twl_mod_T) ? id * workload: twl_mod_T*(workload+1) + workload*(id - twl_mod_T);

        pivots[id] = binary_search(ncopies, M, cdot, loc_n, cumulative_workload);
        workloads[id] = workload;
        cumulative_workloads[id] = cumulative_workload;
    }
    
    #pragma omp parallel num_threads(T)
    {
        int id = omp_get_thread_num(), wl = workloads[id], cwl = cumulative_workloads[id], k = cwl,
            i = pivots[id] + (cwl == cdot[pivots[id]]), newi, newj;
        
        while (k < cwl + wl) {
            if (ncopies[i] == 0) {
                i++;
                continue;
            }
            newi = csum[i] - ncopies[i] + int((k - (cdot[i] - ncopies[i] * M[i])) / M[i]);
            newj = (k - (cdot[i] - ncopies[i] * M[i])) % M[i];
            temp_x[k] = x[i][newj];
            
            coordinates[k][0] = newi; coordinates[k][1] = newj;
            if (newj == 0) {
                newM[newi] = M[i];
            }
            
            if (k == cdot[i]-1) {
                i++;
            }
            //printf("%d\n", k);
            k++;
        }
    }
    
    //return;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < loc_n; i++) {
        x[i].resize(newM[i]);
    }

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < total_workload; k++) {
        x[coordinates[k][0]][coordinates[k][1]] = temp_x[k];
    }

    free(temp_x);
    free(pivots);
    free(workloads);
    free(cumulative_workloads);
}