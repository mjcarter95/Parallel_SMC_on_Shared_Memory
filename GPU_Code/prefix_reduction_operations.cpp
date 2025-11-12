#include "prefix_reduction_operations.h"
#include <omp.h>
#include <cstddef>
#include <stdio.h>

// -----------------------------------------------------------------------------
// Internal templated in-place inclusive scan used by both int and double variants.
// No mapping here; data must be device-resident. Two-pass block scan:
//   1) per-block inclusive scan -> block sums
//   2) scan block sums -> offsets
//   3) add offsets to subsequent blocks
// -----------------------------------------------------------------------------
template <typename T>
static void prefix_sum_inplace_device_impl(T* data, int n, int tile)
{
    if (n <= 0) return;

    const int numBlocks = (n + tile - 1) / tile;
    const int device_id = omp_get_default_device();

    T* d_blockSums   = static_cast<T*>(omp_target_alloc(sizeof(T) * numBlocks, device_id));
    T* d_blockOffset = static_cast<T*>(omp_target_alloc(sizeof(T) * numBlocks, device_id));

    // Pass 1: per-block inclusive scans + block sums
    #pragma omp target teams num_teams(numBlocks) thread_limit(256) is_device_ptr(data, d_blockSums)
    {
        int b = omp_get_team_num();
        int start = b * tile;
        int end   = (start + tile < n) ? (start + tile) : n;
        int count = end - start;

        if (count <= 0) {
            if (b < numBlocks) d_blockSums[b] = T(0);
        } else {
            #pragma omp parallel
            {
                #pragma omp single
                {
                    T run = T(0);
                    for (int k = 0; k < count; ++k) {
                        run += data[start + k];
                        data[start + k] = run;
                    }
                    d_blockSums[b] = run;
                }
            }
        }
    }

    // Pass 2: prefix of block sums -> offsets (exclusive)
    #pragma omp target is_device_ptr(d_blockSums, d_blockOffset)
    {
        T acc = T(0);
        for (int b = 0; b < numBlocks; ++b) {
            d_blockOffset[b] = acc;
            acc += d_blockSums[b];
        }
    }

    // Pass 3: add offsets to blocks b >= 1
    #pragma omp target teams num_teams(numBlocks) thread_limit(256) is_device_ptr(data, d_blockOffset)
    {
        int b = omp_get_team_num();
        if (b > 0) {
            int start = b * tile;
            int end   = (start + tile < n) ? (start + tile) : n;
            T off = d_blockOffset[b];
            #pragma omp parallel for
            for (int i = start; i < end; ++i) {
                data[i] += off;
            }
        }
    }

    omp_target_free(d_blockSums, device_id);
    omp_target_free(d_blockOffset, device_id);
}

// Public in-place scans
void prefix_sum_int_inplace_device(int* data, int n, int tile)
{
    prefix_sum_inplace_device_impl<int>(data, n, tile);
}

void prefix_sum_double_inplace_device(double* data, int n, int tile)
{
    prefix_sum_inplace_device_impl<double>(data, n, tile);
}

// -----------------------------------------------------------------------------
// Inclusive prefix dot-product on device:
//   out[i] = sum_{k=0..i} (a1[k] * a2[k])
// Steps:
//   (A) out[i] := a1[i] * a2[i]   (parallel, device)
//   (B) prefix_sum_int_inplace_device(out, n, tile)
// -----------------------------------------------------------------------------
void prefix_dot_product_inplace_device(const int* a1, const int* a2,
                                       int* out, int n, int tile)
{
    if (n <= 0) return;

    // Step A: element-wise product into 'out'
    #pragma omp target teams distribute parallel for is_device_ptr(a1, a2, out)
    for (int i = 0; i < n; ++i) {
        out[i] = a1[i] * a2[i];
    }

    // Step B: in-place inclusive prefix sum on 'out'
    prefix_sum_int_inplace_device(out, n, tile);
}


// Returns true if csumM is an inclusive prefix sum of M.
// M and csumM must be device pointers (omp_target_alloc).
bool d_validate_csumM_inclusive(const int* M,
                                       const int* csumM,
                                       int loc_n,
                                       int device)
{
    if (loc_n <= 0) return true;

    const int dev = omp_get_default_device();

    // Scalars come back to host via map(from:).
    int err = 0;
    int bad_i = -1;
    long long expected = 0;   // use wide type for safety
    long long got = 0;

    #pragma omp target is_device_ptr(M, csumM) \
                       map(from: err, bad_i, expected, got) \
                       firstprivate(loc_n)
    {
        long long acc = 0;
        err = 0;
        bad_i = -1;
        expected = 0;
        got = 0;

        for (int i = 0; i < loc_n; ++i) {
            acc += (long long)M[i];
            // Compare in 64-bit to avoid overflow surprises.
            if ((long long)csumM[i] != acc) {
                err = 1;
                bad_i = i;
                expected = acc;
                got = (long long)csumM[i];
                break; // stop at first mismatch
            }
        }
    }

    if (err) {
        // Optional: also fetch a few neighbor values around the mismatch
        // for extra debugging context (safe even if we skip it).
        int prev = (bad_i > 0) ? 0 : -1;
        if (bad_i > 0) {
            omp_target_memcpy(&prev, csumM + (bad_i - 1), sizeof(int),
                              0, 0, omp_get_initial_device(), dev);
        }

        int Mi = 0, csum_i = 0;
        omp_target_memcpy(&Mi,     M + bad_i,      sizeof(int), 0, 0, omp_get_initial_device(), dev);
        omp_target_memcpy(&csum_i, csumM + bad_i,  sizeof(int), 0, 0, omp_get_initial_device(), dev);

        printf("[CSUMM-VALIDATE] FAILED at i=%d: "
               "prev_csum=%d, M[i]=%d, csumM[i]=%d, "
               "expected(in 64-bit)=%lld, got=%lld\n",
               bad_i, prev, Mi, csum_i, expected, got);
        return false;
    }

    // Optional total consistency check: last equals sum(M)
    int last = 0;
    omp_target_memcpy(&last, csumM + (loc_n - 1), sizeof(int),
                      0, 0, omp_get_initial_device(), dev);

    long long total = 0;
    #pragma omp target teams distribute parallel for reduction(+:total) \
                         is_device_ptr(M) firstprivate(loc_n)
    for (int i = 0; i < loc_n; ++i) total += (long long)M[i];

    if ((long long)last != total) {
        printf("[CSUMM-VALIDATE] FAILED total: last=%d, sum(M)=%lld\n",
               last, total);
        return false;
    }

    // All good
    // printf("[CSUMM-VALIDATE] OK: last=%d, sum(M)=%lld\n", last, total);
    return true;
}
