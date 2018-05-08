#include "im2col.h"
#include <stdio.h>
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}
//hanxu

void im2col_xnor(float* data_im,
	int channels, int height, int width,
	int ksize, int stride, int pad, BINARY_WORD* data_col)
{
	BINARY_WORD left_move_map[32] = { 0x1,0x2,0x4,0x8,0x10,0x20,0x40,0x80,0x100,0x200,0x400,0x800,0x1000,0x2000,0x4000,0x8000,0x10000,0x20000,0x40000,0x80000,0x100000,0x200000,0x400000,0x800000,0x1000000,0x2000000,0x4000000,0x8000000,0x10000000,0x20000000,0x40000000,0x80000000 };
	int i,j, c, h, w;
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	int plane = ksize * ksize;
	int channels_sum = plane * channels;
	int c_tmp = 1;
	int bits = channels;
	int block = channels_sum;
	int tmp_b = 0;
	if (32 <= channels) {
		c_tmp = channels >> 5;
		block = plane << 32;
		bits = 32;
	}
	for(j=0;j<c_tmp;j++){
	for (c = tmp_b; c < plane + tmp_b; c++){
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		for (h = 0; h < height_col; ++h) {
			for (w = 0; w < width_col; ++w) {
				int im_row = h_offset + h * stride;
				int im_col = w_offset + w * stride;
				int col_index = (c * height_col + h) * width_col + w;
				data_col[col_index] = 0x00000000;
				for (i = 0; i < bits; i++) {
					//printf("c:%d,h:%d,w:%d,col_index:%d,i:%d;return:%lf\n",c,h,w, col_index,i, im2col_get_pixel(data_im, height, width, channels, im_row, im_col, i, pad));
						if (0 < im2col_get_pixel(data_im, height, width, channels,
							im_row, im_col, i, pad)) {
							data_col[col_index] = data_col[col_index] | (left_move_map[(32 - 1) - i]);
							
						}
					
				}
			}
		}
		
	}
	tmp_b += block;
	}
}


