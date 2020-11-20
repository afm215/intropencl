#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

__kernel void matrix_convolution(__global const unsigned char *f,
                                 __global const float *y,
                                 __global unsigned char *restrict z, unsigned W,
                                 unsigned H, unsigned D) {
#ifdef GROUPS
    unsigned i = get_group_id(0) * get_local_size(0) + get_local_id(0);
    unsigned j = get_group_id(1) * get_local_size(1) + get_local_id(1);
#else
    unsigned id = get_global_id(0);
    unsigned i = id / W;
    unsigned j = id % W;
#endif

    float sum = 0;
    for (unsigned r = 0; r < D; r++) {
        for (unsigned c = 0; c < D; c++) {
            int rind = i + D / 2 - r;
            int cind = j + D / 2 - c;
            if (rind >= 0 && rind < H && cind >= 0 && cind <= W) {
                sum += y[r * D + c] * (float)f[rind * W + cind];
            }
        }
    }

    z[i * W + j] = (unsigned char) MAX(MIN(sum, 255.), 0.);
}

__kernel void matrix_average(__global const unsigned char *x,
                         __global const unsigned char *y,
                         __global unsigned char *restrict z, unsigned W) {
    int id = get_global_id(0);
    z[id] = (unsigned char)((x[id] + y[id]) / 2);
}
