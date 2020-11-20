__kernel void matrix_prod(__global const float *x, __global const float *y,
                          __global float *restrict z, const int M, const int K,
                          const int N) {
#ifdef GROUPS
    size_t i = get_group_id(0) * get_local_size(0) + get_local_id(0);
    size_t j = get_group_id(1) * get_local_size(1) + get_local_id(1);
#else
    size_t id = get_global_id(0);
    size_t i = id / N;
    size_t j = id % N;
#endif

    double sum = 0;
    for (unsigned k = 0; k < K; k++) {
        sum += x[i * K + k] * y[k * N + j];
    }
    z[i * N + j] = (float)sum;
}
