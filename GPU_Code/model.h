#ifndef MODEL_H
#define MODEL_H

#include <cstddef>

// Device-visible model constants (defined in model.cpp)
extern double prior_threshold;     // p("small")
extern int    min_dim_range[2];    // inclusive
extern int    max_dim_range[2];    // inclusive
extern double cum_probs[3];        // {p_grow, p_grow+p_shrink, 1.0}

// Size-only log-prob helpers (device-usable)
double intervals_lpmf_M(int M);
double q_by_sizes(int M, int initM);

// Build initial population on the GPU (packed storage).
// NOTE: logw is NO LONGER an argument here; it will be set in `initialise`.
// Outputs (device):
//   - X (int*&)  : allocated/reallocated to total length (csumM[loc_n-1])
//   - x_len (host int&)
//   - M, csumM   : sizes and inclusive prefix
//   - initM      : copy of M
//   - initX      : allocated/reallocated packed copy of X
//   - init_x_len : host len of initX
//   - initcsumM  : copy of csumM
void sampling_from_q0(int*& X, int& x_len,
                      int* M, int* csumM,
                      int* initM,
                      int*& initX, int& init_x_len,
                      int* initcsumM,
                      int loc_n,
                      unsigned long long* rng_s0,
                      unsigned long long* rng_s1);

// One proposal step from q on the GPU (packed storage)
void sampling_from_q(int*& X, int& x_len,
                     int* M, int* csumM,
                     int loc_n,
                     unsigned long long* rng_s0,
                     unsigned long long* rng_s1);

#endif // MODEL_H

