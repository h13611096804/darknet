#ifndef IM2COL_H
#define IM2COL_H
//hanxu
#include <stdint.h>

typedef uint32_t BINARY_WORD;
void im2col_xnor(float* data_im,
	int channels, int height, int width,
	int ksize, int stride, int pad, BINARY_WORD* data_col);
void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

#ifdef GPU

void im2col_ongpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);

#endif
#endif
