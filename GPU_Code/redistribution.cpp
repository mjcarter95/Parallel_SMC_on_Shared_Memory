#include "redistribution.h"

#include <omp.h>
#include <cstddef>
#include <algorithm>

static inline int device_read_last_int(const int* d_arr, int len) {
    int out = 0;
    omp_target_memcpy(&out, d_arr + (len - 1), sizeof(int),
                      0, 0, omp_get_initial_device(), omp_get_default_device());
    return out;
}

// ======================= NAIVE (unchanged) =======================

void pad(int*& X, int& x_len,
         const int* M, const int* csumM,
         int loc_n,
         int& maxM) {
    if (loc_n <= 0) { maxM = 0; return; }
    const int dev = omp_get_default_device();

    int* d_maxM = static_cast<int*>(omp_target_alloc(sizeof(int), dev));
    d_max_int(M, loc_n, d_maxM);
    maxM = 0;
    omp_target_memcpy(&maxM, d_maxM, sizeof(int), 0, 0,
                      omp_get_initial_device(), dev);

    const std::size_t padded_len = (std::size_t)loc_n * (std::size_t)maxM;
    int* newX = static_cast<int*>(omp_target_alloc(sizeof(int) * padded_len, dev));

    #pragma omp target teams distribute parallel for \
        is_device_ptr(newX, X, M, csumM) firstprivate(maxM)
    for (int i = 0; i < loc_n; ++i) {
        const int Mi  = M[i];
        const int src = (i == 0) ? 0 : csumM[i - 1];
        const int dst = i * maxM;

        #pragma omp parallel for
        for (int j = 0; j < Mi; ++j) newX[dst + j] = X[src + j];

        #pragma omp parallel for
        for (int j = Mi; j < maxM; ++j) newX[dst + j] = -1;
    }

    if (X) omp_target_free(X, dev);
    X = newX;
    x_len = (int)padded_len;

    omp_target_free(d_maxM, dev);
}

void fixed_size_redistribution(int* X, int* M,
                               const int* ncopies,
                               int loc_n, int maxM) {
    if (loc_n <= 0 || maxM <= 0) return;

    const int dev = omp_get_default_device();

    int* d_csum = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)loc_n, dev));
    #pragma omp target teams distribute parallel for is_device_ptr(d_csum, ncopies)
    for (int i = 0; i < loc_n; ++i) d_csum[i] = ncopies[i];
    prefix_sum_int_inplace_device(d_csum, loc_n);

    int* X_tmp = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)loc_n * (size_t)maxM, dev));
    int* M_tmp = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)loc_n, dev));

    const int teams = std::min(loc_n, 256);

    #pragma omp target teams num_teams(teams) thread_limit(256) \
        is_device_ptr(X, X_tmp, M, M_tmp, ncopies, d_csum) \
        firstprivate(loc_n, teams, maxM)
    {
        const int id        = omp_get_team_num();
        const int start_out = (id * loc_n) / teams;
        const int end_out   = ((id + 1) * loc_n) / teams;
        const int chunk     = end_out - start_out;

        if (chunk > 0) {
            const int pivot = d_binary_search_ncopies_csum(ncopies, d_csum, loc_n, start_out);

            int copies_on_pivot;
            if (teams > 1) {
                const int csum_p = d_csum[pivot];
                copies_on_pivot  = csum_p - start_out;
                if (copies_on_pivot > chunk) copies_on_pivot = chunk;
            } else {
                copies_on_pivot = ncopies[pivot];
            }

            auto copy_row = [&](int src_idx, int dst_idx) {
                const int src_base = src_idx * maxM;
                const int dst_base = dst_idx * maxM;

                #pragma omp parallel for
                for (int j = 0; j < maxM; ++j) X_tmp[dst_base + j] = X[src_base + j];
                M_tmp[dst_idx] = M[src_idx];
            };

            int count = 0;
            for (; count < copies_on_pivot; ++count) copy_row(pivot, start_out + count);

            int i_src = pivot + 1;
            while (count < chunk) {
                const int nci = (i_src < loc_n) ? ncopies[i_src] : 0;
                for (int k = 0; k < nci && count < chunk; ++k) {
                    copy_row(i_src, start_out + count);
                    ++count;
                }
                ++i_src;
            }
        }
    }

    #pragma omp target teams distribute parallel for is_device_ptr(X, X_tmp) firstprivate(maxM)
    for (int idx = 0; idx < loc_n; ++idx) {
        const int base = idx * maxM;
        #pragma omp parallel for
        for (int j = 0; j < maxM; ++j) X[base + j] = X_tmp[base + j];
    }

    #pragma omp target teams distribute parallel for is_device_ptr(M, M_tmp)
    for (int i = 0; i < loc_n; ++i) M[i] = M_tmp[i];

    omp_target_free(d_csum, omp_get_default_device());
    omp_target_free(X_tmp,  omp_get_default_device());
    omp_target_free(M_tmp,  omp_get_default_device());
}

void restore(int*& X, int& x_len,
             const int* M, int* csumM,
             int loc_n, int maxM) {
    if (loc_n <= 0 || maxM <= 0) return;

    const int dev = omp_get_default_device();

    #pragma omp target teams distribute parallel for is_device_ptr(csumM, M)
    for (int i = 0; i < loc_n; ++i) csumM[i] = M[i];
    prefix_sum_int_inplace_device(csumM, loc_n);

    const int total = device_read_last_int(csumM, loc_n);

    int* newX = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)total, dev));

    #pragma omp target teams distribute parallel for \
        is_device_ptr(newX, X, M, csumM) firstprivate(maxM)
    for (int i = 0; i < loc_n; ++i) {
        const int Mi  = M[i];
        const int src = i * maxM;
        const int dst = (i == 0) ? 0 : csumM[i-1];
        #pragma omp parallel for
        for (int j = 0; j < Mi; ++j) newX[dst + j] = X[src + j];
    }

    if (X) omp_target_free(X, dev);
    X = newX;
    x_len = total;
}

void naive_variable_size_redistribution(int*& X, int& x_len,
                                        int* M, int* csumM,
                                        const int* ncopies,
                                        int loc_n) {
    if (loc_n <= 0) return;
    int maxM = 0;
    pad(X, x_len, M, csumM, loc_n, maxM);
    fixed_size_redistribution(X, M, ncopies, loc_n, maxM);
    restore(X, x_len, M, csumM, loc_n, maxM);
}

// ======================= OPTIMAL (cdot + binary search) =======================
// Order: cdot → copy X by k → redistribute M (maxM=1) → recompute csumM

void optimal_variable_size_redistribution(int*& X, int& x_len,
                                          int* M, int* csumM,
                                          const int* ncopies,
                                          int loc_n) {
    if (loc_n <= 0) return;

    const int dev = omp_get_default_device();

    // 1) cdot = prefix_sum(ncopies[i] * M[i])
    int* cdot = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)loc_n, dev));
    prefix_dot_product_inplace_device(ncopies, M, cdot, loc_n);
    const int total_workload = device_read_last_int(cdot, loc_n);

    // 2) Copy X directly into new packed buffer using linear k
    int* newX = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)total_workload, dev));

    const int teams = std::min(std::max(1, total_workload), 256);

    #pragma omp target teams num_teams(teams) thread_limit(256) \
        is_device_ptr(X, newX, M, csumM, ncopies, cdot) \
        firstprivate(loc_n, total_workload)
    {
        const int id  = omp_get_team_num();

        const int base = total_workload / teams;
        const int rem  = total_workload % teams;

        const int wl  = (id < rem) ? (base + 1) : base;
        const int cwl = (id < rem) ? id * (base + 1)
                                   : rem * (base + 1) + (id - rem) * base;

        if (wl > 0) {
            int i = d_binary_search_prod_csum(ncopies, M, cdot, loc_n, cwl);
            if (cwl == cdot[i]) ++i;

            for (int k = cwl; k < cwl + wl; ++k) {
                while (i < loc_n && ncopies[i] * M[i] == 0) ++i;

                const int Mi      = M[i];
                const int nc_i    = ncopies[i];
                const int cd_i    = cdot[i];
                const int start_i = cd_i - nc_i * Mi;
                const int offs    = k - start_i;

                const int newj     = offs % Mi;
                const int src_base = (i == 0) ? 0 : csumM[i - 1];

                newX[k] = X[src_base + newj];

                if (k == cdot[i] - 1) ++i;
            }
        }
    }

    if (X) omp_target_free(X, dev);
    X = newX;
    x_len = total_workload;

    // 3) Redistribute M via fixed-size trick with maxM=1
    int* fakeM = static_cast<int*>(omp_target_alloc(sizeof(int) * (size_t)loc_n, dev));
    #pragma omp target teams distribute parallel for is_device_ptr(fakeM)
    for (int i = 0; i < loc_n; ++i) fakeM[i] = 1;

    fixed_size_redistribution(/*X=*/M, /*M=*/fakeM, ncopies, loc_n, /*maxM=*/1);
    omp_target_free(fakeM, dev);

    // 4) Recompute csumM from new M
    #pragma omp target teams distribute parallel for is_device_ptr(csumM, M)
    for (int i = 0; i < loc_n; ++i) csumM[i] = M[i];
    prefix_sum_int_inplace_device(csumM, loc_n);

    // 5) Free temporaries
    omp_target_free(cdot, dev);
}

