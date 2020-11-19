#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

__kernel void matrix_prod_groups(__global const float *f,
                                 __global const float *y,
                                 __global float *restrict z,
                                 const int M,
                                 const int N,
                                 const int D) {
    size_t i = get_group_id(0) * M / get_num_groups(0) + get_local_id(0);
    size_t j = get_group_id(1) * N / get_num_groups(1) + get_local_id(1);

    unsigned rmin = MAX(0, D / 2 - i);
    unsigned rmax = MIN(D, M-1 + D/2 - i);

    unsigned lmin = MAX(0, D / 2 - j);
    unsigned lmax = MIN(D, N-1 + D/2 - j);

    double sum = 0;

    for (unsigned r = rmin; r < rmax; r++) {
        for (unsigned l = lmin; l < lmax; l++) {
            sum += x[r*D + l] * y[(i + r - D / 2) * N + j + l - D / 2];
        }
    }

    z[i*N + j] = (float)sum;
}

