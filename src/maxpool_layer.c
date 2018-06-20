#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + 2*padding)/stride;
    l.out_h = (h + 2*padding)/stride;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
	l.bflops = (l.size*l.size*l.c * l.out_h*l.out_w) / 1000000000.;
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d %5.3f BF\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + 2*l->pad)/l->stride;
    l->out_h = (h + 2*l->pad)/l->stride;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = realloc(l->indexes, output_size * sizeof(int));
    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}
void find_max(const maxpool_layer l, network_state state, int out_index,int c, int b) {
	for (int h = 0; h < l.out_h; ++h) {
		for (int w = 0; w < l.out_w; ++w, out_index += 1) {
			if (out_index != w + l.out_w*(h + l.out_h*(c + l.c*b)))
				printf("error:%d, %d\n", out_index, w + l.out_w*(h + l.out_w*(c + l.c*b)));

			float max = -FLT_MAX;
			int max_i = -1;
			for (int n = 0; n < l.size; ++n) {
				for (int m = 0; m < l.size; ++m) {
					int cur_h = -l.pad + h*l.stride + n;
					int cur_w = -l.pad + w*l.stride + m;
					int index = cur_w + l.w*(cur_h + l.h*(c + b*l.c));
					int valid = (cur_h >= 0 && cur_h < l.h &&
						cur_w >= 0 && cur_w < l.w);
					float val = (valid != 0) ? state.input[index] : -FLT_MAX;
					max_i = (val > max) ? index : max_i;
					max = (val > max) ? val : max;
				}
			}
			l.output[out_index] = max;
			l.indexes[out_index] = max_i;
		}
	}
}

void forward_maxpool_layer(const maxpool_layer l, network_state state)
{
    int c,b,h,w,m,n;
    int w_offset = -l.pad;
    int h_offset = -l.pad;

	int tmp_2 = l.out_h*l.out_w;
	int tmp_3 = l.c*l.out_h*l.out_w;
	/*clock_t time1, time2;
	time1 = clock();*/
    for(b = 0; b < l.batch; ++b,b*l.c){
		int batch = b*tmp_3;
		#pragma omp parallel for
        for(c = 0; c < l.c; ++c){
			int out_index = tmp_2*c+batch;
			find_max(l, state, out_index, c,b);
        }
    }
	//printf("%5lf", (double)(clock() - time1) / CLOCKS_PER_SEC);
	//int  i, j, k;

	//h = l.out_h;
	//w = l.out_w;
	//c = l.c;
	//time2 = clock();
	//for (b = 0; b < l.batch; ++b) {
	//	for (k = 0; k < c; ++k) {
	//		for (i = 0; i < h; ++i) {
	//			for (j = 0; j < w; ++j) {
	//				int out_index = j + w*(i + h*(k + c*b));
	//				float max = -FLT_MAX;
	//				int max_i = -1;
	//				for (n = 0; n < l.size; ++n) {
	//					for (m = 0; m < l.size; ++m) {
	//						int cur_h = h_offset + i*l.stride + n;
	//						int cur_w = w_offset + j*l.stride + m;
	//						int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
	//						int valid = (cur_h >= 0 && cur_h < l.h &&
	//							cur_w >= 0 && cur_w < l.w);
	//						float val = (valid != 0) ? state.input[index] : -FLT_MAX;
	//						max_i = (val > max) ? index : max_i;
	//						max = (val > max) ? val : max;
	//					}
	//				}
	//				l.output[out_index] = max;
	//				l.indexes[out_index] = max_i;
	//			}
	//		}
	//		
	//	}
	//}
	//printf("%5lf\n", (double)(clock() - time2) / CLOCKS_PER_SEC);
}

void backward_maxpool_layer(const maxpool_layer l, network_state state)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        state.delta[index] += l.delta[i];
    }
}

