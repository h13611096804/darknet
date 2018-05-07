#include "binary_convolution.h"
#include <nmmintrin.h>
#include "im2col.h"
#include <stdio.h>
#include "col2im.h"
#include "network.h"
#include "gemm.h"

int ai2_bin_dp(BINARY_WORD *a, BINARY_WORD *b, dim3 vdim) {     // TODO unroll
	int accumulator = 0;
	for (int z = 0; z < vdim.z / BITS_PER_BINARY_WORD; z++) {
		for (int y = 0; y < vdim.y; y++) {
			for (int x = 0; x < vdim.x; x++) {
				int idx = z*vdim.y*vdim.x + y*vdim.x + x;
				accumulator += _mm_popcnt_u64(~(a[idx] ^ b[idx]));   // count the XNOR of the two bit vectors
			}
		}
	}
	return accumulator;
}

/**
* Pre-conditions:
*                  alpha_volume is an array of size x*y*z.
*                  alpha_plane is an array of size x*y.
*                  alpha_volume (x,y,z) is transposed to (z,x,y).
*/
void ai2_calc_alpha(float *alpha_plane, float *alpha_volume, dim3 vdim, int n) {
	for (int i = 0; i < n; ++i) {
		double accum = 0.0;
		for (int y = 0; y < vdim.y*vdim.x*vdim.z; ++y) {
			accum += fabs(alpha_volume[i*vdim.y*vdim.x*vdim.z + y]);
			
		}
		alpha_plane[i] = accum / (vdim.y*vdim.x*vdim.z);
	}
}

/** @brief Wrapper function for generating the beta scaling factor */
void ai2_calc_beta(float *beta_plane, float *beta_volume, dim3 vdim, int n) {
	for (int i = 0; i < n; ++i) {
		for (int y = 0; y < vdim.y; ++y) {
			for (int x = 0; x < vdim.x; ++x) {
				int out = y * vdim.x + x;
				float accum = 0.0;
				for (int z = 0; z < vdim.z; ++z) {
					accum += fabs(beta_volume[i * vdim.y * vdim.x * vdim.z + out * vdim.z + z]);
					//printf("%f", accum);
				}
				beta_plane[i * vdim.y * vdim.x + out] = accum / vdim.z;
			}
		}
	}
}
//BINARY_WORD left_move_map [32] = {0x0001, 0x002, 0x0004, 0x0008};
///** @brief Set the bit in a binary word */
//void ai2_bitset(BINARY_WORD *bword, unsigned int position) {
//	BINARY_WORD mask = (1 << position);
//	*bword = *bword | mask;
//}

/** @brief Checks that the bit is set in a binary word */
int ai2_is_set(BINARY_WORD bword, unsigned int position) {
	unsigned int position_complement = (BITS_PER_BINARY_WORD - 1) - position;   // number of leading bits before the bit position of interest
	bword = (bword << position_complement);                                     // zero out leading bits
	bword = (bword >> (BITS_PER_BINARY_WORD - 1));                              // shift bit position of interest to the 0th position
	return (bword & 0x1);                                                       // test if bit position of interest is set
}
BINARY_WORD left_move_map[32] = { 0x1,0x2,0x4,0x8,0x10,0x20,0x40,0x80,0x100,0x200,0x400,0x800,0x1000,0x2000,0x4000,0x8000,0x10000,0x20000,0x40000,0x80000,0x100000,0x200000,0x400000,0x800000,0x1000000,0x2000000,0x4000000,0x8000000,0x10000000,0x20000000,0x40000000,0x80000000 };
void ai2_flt_to_bin(BINARY_WORD *binary_vol, float *real_vol, float *alpha, dim3 dim, int b) {
	
	//clock_t start, finish; double duration;start = clock();
	int tmp3 = dim.z * dim.y * dim.x;
	int tmp2_B = dim.y * dim.x* BITS_PER_BINARY_WORD;
	int tmp2 = dim.y * dim.x;
	BINARY_WORD tmp = 0x00000000;
	int b_tmp = 0;
	int z_tmp = 0;
	int y_tmp = 0;
	int x_tmp = 0;
	int waddr = 0;
	int bi, zi, yi, xi, i;
	float accum = 0.0;
	for (bi = 0; bi < b; bi++)
	{
		b_tmp += tmp3;
		if (b_tmp == bi * dim.z * dim.y * dim.z)
			printf("1");
		
		z_tmp = 0;
		for (zi = 0; zi < dim.z; zi += BITS_PER_BINARY_WORD)
		{
			z_tmp += tmp2_B;
			if (z_tmp == zi * dim.y * dim.x)
				printf("2");
			y_tmp = 0;
			for (yi = 0; yi < dim.y; yi++)
			{
				y_tmp += dim.x;
				if (y_tmp == dim.x * yi)
					printf("3");
				

				tmp = 0x00000000;
				for (xi = 0 ; xi < dim.x; xi++)
				{
					x_tmp = 0;
					waddr = 0;
					for (i = 0;  i < BITS_PER_BINARY_WORD; ++i)
					{
						x_tmp += tmp2;
				
						waddr = b_tmp + y_tmp + z_tmp + x_tmp + xi;
						if (waddr == b_tmp + y_tmp + z_tmp + i*dim.y*dim.x + xi)
							printf("4");
						accum += fabs(real_vol[waddr]);
						if (0 < real_vol[waddr])
						{
							tmp = ai2_bitset(tmp, left_move_map[(BITS_PER_BINARY_WORD - 1) - i]);
						}
					}
					binary_vol[(b_tmp + z_tmp) / 32 + y_tmp + xi] = tmp;
				}
			}
		}
		alpha[bi] = accum / tmp3;
	}
	//for (int bi = 0; bi < b; bi++) {
	//	int b_tmp = bi * dim.z * dim.y * dim.x;
	//	float accum = 0.0;
	//	for (int zi = 0; zi < dim.z; zi += BITS_PER_BINARY_WORD) {
	//		int  z_tmp = zi * dim.y * dim.x;
	//		for (int yi = 0; yi < dim.y; yi++) {
	//			int y_tmp = dim.x * yi;
	//			for (int xi = 0; xi < dim.x; xi++) {
	//				BINARY_WORD tmp = 0x00000000;
	//				for (int i = 0; i < BITS_PER_BINARY_WORD; ++i) {
	//					int waddr = b_tmp + y_tmp + z_tmp + i*dim.y*dim.x + xi;
	//					accum += fabs(real_vol[waddr]);
	//					if (real_vol[waddr] > 0)
	//					//if (signbit(real_vol[waddr]) > 0)
	//						ai2_bitset(&tmp, (BITS_PER_BINARY_WORD - 1) - i);
	//				}
	//				binary_vol[(b_tmp + z_tmp) / 32 + y_tmp + xi] = tmp;
	//			}
	//		}
	//	}
	//	alpha[bi] = accum / (dim.y*dim.x*dim.z);
	//}
	
	//finish = clock();duration = (double)(finish - start) / CLOCKS_PER_SEC;printf("1:%f seconds\n", duration);
	//start = clock();
	//
	//ai2_transpose3D( real_vol, real_tmp,  dim, b); // (x,y,z) -> (z,x,y)
	//int sz = dim.x * dim.y * dim.z * b;
	//for (int i = 0; i < sz; i += BITS_PER_BINARY_WORD) {
	//	BINARY_WORD tmp = 0x00000000;
	//	for (int x = 0; x < BITS_PER_BINARY_WORD; ++x) {
	//		int waddr = x + i;
	//		if (signbit(real_vol[waddr]) > 0)
	//			ai2_bitset(&tmp, (BITS_PER_BINARY_WORD - 1) - x);
	//	}
	//	//printf("bin:%x tmp:%x\n  ", binary_vol[i / BITS_PER_BINARY_WORD], tmp);
	//	if (binary_vol[i / BITS_PER_BINARY_WORD] != tmp)
	//		printf("报错：y:%d\n  ", i);
	//	//binary_vol[i / BITS_PER_BINARY_WORD] = tmp;
	//}
	//dim.z /= BITS_PER_BINARY_WORD;
	//ai2_transpose3D_reverse(binary_vol, binary_tmp, dim, b);
	//
	//finish = clock();duration = (double)(finish - start) / CLOCKS_PER_SEC;printf("2:%f seconds\n", duration);
}

void ai2_input_to_bin(BINARY_WORD *binary_vol, float *real_vol, dim3 dim, int b) {
	//clock_t start, finish; double duration;start = clock();
	int tmp3 = dim.z * dim.y * dim.x;
	int tmp2_B = dim.y * dim.x* BITS_PER_BINARY_WORD;
	int tmp2 = dim.y * dim.x;
	BINARY_WORD tmp = 0x00000000;
	int b_tmp = 0;
	int z_tmp = 0;
	int y_tmp = 0;
	int x_tmp = 0;
	int waddr = 0;
	int bi, zi, yi, xi, i;
	//int k

	for (bi = 0; bi < b; bi++)
	{
		if (b_tmp != bi * dim.z * dim.y * dim.z)
			printf("1,%d,%d\n", bi,b);
		z_tmp = 0;
		for (zi = 0; zi < dim.z; zi += BITS_PER_BINARY_WORD)
		{
			if (z_tmp != zi * dim.y * dim.x)
				printf("2");
			y_tmp = 0;
			for (yi = 0; yi < dim.y; yi++)
			{
				if (y_tmp != dim.x * yi)
					printf("3");
				tmp = 0x00000000;
				for (xi = 0; xi < dim.x; xi++)
				{
					x_tmp = 0;
					waddr = 0;
					for (i = 0; i < BITS_PER_BINARY_WORD; ++i)
					{
						waddr = b_tmp + y_tmp + z_tmp + x_tmp + xi;
						if (waddr != b_tmp + y_tmp + z_tmp + i*dim.y*dim.x + xi)
							printf("4");
						if (0 < real_vol[waddr])
						{
							tmp = ai2_bitset(tmp, left_move_map[(BITS_PER_BINARY_WORD - 1) - i]);
						}
						x_tmp += tmp2;
					}
					binary_vol[(b_tmp + z_tmp) / 32 + y_tmp + xi] = tmp;
				}
				y_tmp += dim.x;
			}
			z_tmp += tmp2_B;
		}
		b_tmp += tmp3;
	}
}

void ai2_bin_to_flt(float *real_vol, BINARY_WORD *binary_vol, dim3 dim) {   // TODO unit tests
	for (int z = 0; z < dim.z; z++) {
		for (int y = 0; y < dim.y; y++) {
			for (int x = 0; x < dim.x / BITS_PER_BINARY_WORD; x++) {    // TODO boundary checks, for uneven input
				BINARY_WORD word = binary_vol[z*dim.y*dim.x + y*dim.x + x];
				for (int t = 0; t < BITS_PER_BINARY_WORD; ++t) {
					int oidx = z*dim.y*dim.x + y*dim.x + x * BITS_PER_BINARY_WORD + t;
					if (ai2_is_set(word, t))
						real_vol[oidx] = 1.f;
					else
						real_vol[oidx] = -1.f;
				}
			}
		}
	}
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Transpose channels back to output
	//ai2_transpose3D(real_vol, dim,1); // (z,y,x) -> (x,y,z)
}


void ai2_pointwise_mul_mm(float *output, const float *input, int N, float n,float a) {
	int i = 0;
	while (i + 8 <= N) {
		//printf("pre:%f  \n", output[i + 0]);
		//input[i + 0]
		output[i + 0] *= a / n;
		//printf("later:%f\n", output[i + 0]);
		output[i + 1] *= a / n;
		output[i + 2] *= a / n;
		output[i + 3] *= a / n;
		output[i + 4] *= a / n;
		output[i + 5] *= a / n;
		output[i + 6] *= a / n;
		output[i + 7] *= a / n;

		i += 8;
	}
	while (++i < N) // Finish iteration that's leftover (e.g., last batch not divisible by 8 exactly)
		output[i] *= a / n;
}

/** @brief Performs a tiled pointwise matrix multiplication between two 2D tensors
*  Pre-conditions: wx < ix, and wy < iy
*/
void ai2_pointwise_mul_mm_2d(float *output, const float *alpha, int ix, int iy, int wx, int wy, int pad) {
	// Slower version
	//      for (int y = 0; y < iy; ++y) 
	//          for (int x = 0; x < ix; x++)
	//              output[y*ix+x] *= input[(y % wy)*wx + (x % wx)];

	// Stride prefetch optimized
	for (int s = 0; s < wy; ++s) {  // for each strip
		const float *strip_ptr = &alpha[s*wx];
		for (int y = pad; y < pad + (iy / wy); ++y) {   //
			int stride = y*((ix + 2 * pad)*wy) + s*(ix + 2 * pad);
			float *output_ptr = &output[stride];

			for (int x = 0; x < ix; ++x) {
				output_ptr[x] *= strip_ptr[x % wx];
			}
		}
	}
}

void ai2_setFltInput(layer l, network_state net) {
		// Binarize input
		BINARY_WORD *b = l.binary_input_xnor;
		float *input = net.input;
		//ai2_input_to_bin(b,  i, dim, l.batch);
		int tmp3 = l.inputs;
		int tmp2 = l.w * l.h;
		int tmp2_B = tmp2 * BITS_PER_BINARY_WORD;
		BINARY_WORD tmp = 0x00000000;
		int b_tmp = 0;
		int z_tmp = 0;
		int y_tmp = 0;
		int x_tmp = 0;
		int waddr = 0;
		int bi, zi, yi, xi, i;
		int bits = 0;
		////k-bit位量化
		//int k = 0;
		//if (l.c * k < 32)
		//	bits = l.c;
		//else
		//	bits = BITS_PER_BINARY_WORD;
		if (l.c < 32)
			bits = l.c;
		else
			bits = BITS_PER_BINARY_WORD;
		for (bi = 0; bi < l.batch; bi++)
		{
			/*if (b_tmp !=  bi * dim.z * dim.y * dim.x)
				printf("1,%d,==%d,%d,%d\n", b_tmp, bi * dim.z * dim.y * dim.x,bi, l.batch);*/
			z_tmp = 0;
			for (zi = 0; zi < l.c; zi += bits)
			{
				/*if (z_tmp != zi * dim.y * dim.x)
					printf("2");*/
				y_tmp = 0;
				for (yi = 0; yi < l.h; yi++)
				{
					/*if (y_tmp != dim.x * yi)
						printf("3");*/
					tmp = 0x00000000;
					for (xi = 0; xi < l.w; xi++)
					{
						x_tmp = 0;
						waddr = 0;
						
						for (i = 0; i < bits; ++i)
						{
							waddr = b_tmp + y_tmp + z_tmp + x_tmp + xi;
							/*if (waddr != b_tmp + y_tmp + z_tmp + i*dim.y*dim.x + xi)
								printf("4");*/
						
							if (0 < input[waddr])
							{
								tmp = ai2_bitset(tmp, left_move_map[(BITS_PER_BINARY_WORD - 1) - i]);
							}
							x_tmp += tmp2;
						}
						b[(b_tmp + z_tmp) / 32 + y_tmp + xi] = tmp;
					}
					y_tmp += l.w;
				}
				z_tmp += tmp2_B;
			}
			b_tmp += tmp3;
		}
		// layer->input is transposed to (z,x,y) already
		//可改进
		//float *beta = l.beta_xnor;
		//ai2_calc_beta(beta, i, dim, l.batch);
}


void ai2_setFltWeights(layer l) {
		BINARY_WORD *bin = l.binary_weights_xnor;
		float *w =l.weights;
		float *alpha = l.alpha_xnor;
		//ai2_flt_to_bin(b, w, a,dim,l.n);
		int tmp3 = l.weightss;
		int tmp2 = l.size * l.size;
		int tmp2_B = tmp2 * BITS_PER_BINARY_WORD;
		BINARY_WORD tmp = 0x00000000;
		int b_tmp = 0;
		int z_tmp = 0;
		int y_tmp = 0;
		int x_tmp = 0;
		int waddr = 0;
		int bi, zi, yi, xi, i;
		float accum = 0.0;
		for (bi = 0; bi < l.n; bi++)
		{
			/*if (b_tmp != bi * dim.z * dim.y * dim.x)
				printf("1,%d,==%d,%d,%d\n", b_tmp, bi * dim.z * dim.y * dim.x, bi, l.n);*/
			z_tmp = 0;
			for (zi = 0; zi < l.c; zi += BITS_PER_BINARY_WORD)
			{
				/*if (z_tmp != zi * dim.y * dim.x)
					printf("2");*/
				y_tmp = 0;
				for (yi = 0; yi < l.size; yi++)
				{
					/*if (y_tmp != dim.x * yi)
						printf("3");*/
					tmp = 0x00000000;
					for (xi = 0; xi < l.size; xi++)
					{
						x_tmp = 0;
						waddr = 0;
						for (i = 0; i < BITS_PER_BINARY_WORD; ++i)
						{
							waddr = b_tmp + y_tmp + z_tmp + x_tmp + xi;
							/*if (waddr != b_tmp + y_tmp + z_tmp + i * dim.y * dim.x + xi)
								printf("4");*/
							accum += fabs(w[waddr]);
							if (0 < w[waddr])
							{
								tmp = ai2_bitset(tmp, left_move_map[(BITS_PER_BINARY_WORD - 1) - i]);
							}
							x_tmp += tmp2;
						}
						bin[(b_tmp + z_tmp) / 32 + y_tmp + xi] = tmp;
					}
					y_tmp += l.size;
				}
				z_tmp += tmp2_B;
			}
			alpha[bi] = accum / tmp3;
			b_tmp += tmp3;
		}

		// Calculate alpha
		
	//	ai2_calc_alpha(a, w, dim, l.n);
}


void ai2_transpose3D(float *data, float *tmp,dim3 d,int b) {
	// Slow transpose for correctness
	// (x,y,z) becomes (z,x,y). Requires two transposes:
	//  (x,y,z) -> (x,z,y).
	//  (x,z,y) -> (z,x,y).
	// Intermediate buffer
	//float *tmp = (float *)calloc(b * d.x * d.y * d.z, sizeof(float));//add batch
	// Transpose y and z axis.
	// (x,y,z) -> (x,z,y);
	for (int i = 0; i < b; ++i) {//add batch 
		for (int y = 0; y < d.y; ++y) {
			for (int z = 0; z < d.z; ++z) {
				for (int x = 0; x < d.x; ++x) {
					tmp[i * d.x * d.y * d.z+y * d.x * d.z + z * d.x + x] = data[i * d.x * d.y * d.z + z * d.x * d.y + y * d.x + x];//add batch 
				}
			}
		}
	}
	// Transpose x and z axis.
	//  (x,z,y) -> (z,x,y)
	for (int i = 0; i < b; ++i) {//add batch 
		for (int y = 0; y < d.y; ++y) {
			for (int x = 0; x < d.x; ++x) {
				for (int z = 0; z < d.z; ++z) {
					data[i * d.x * d.y * d.z + y * d.z * d.x + x * d.z + z] = tmp[i * d.x * d.y * d.z + y * d.x * d.z + x + z * d.x];
				}
			}
		}
	}
	//free(tmp);
}

void ai2_transpose3D_reverse(BINARY_WORD *data, BINARY_WORD *tmp, dim3 d, int b) {
	// Slow transpose for correctness

	// (z,x,y) becomes (x,y,z). Requires two transposes:
	//  (z,x,y) -> (x,z,y).
	//  (x,z,y) -> (x,y,z).

	// Intermediate buffer
	//float *new_data = cuda_make_array(0, l.inputs*l.batch);
	//l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
	//BINARY_WORD *tmp = calloc(b * d.x * d.y * d.z, sizeof(BINARY_WORD));//add batch										
	for (int i = 0; i < b; ++i) {//add batch 
		for (int y = 0; y < d.y; ++y) {
			for (int x = 0; x < d.x; ++x) {
				for (int z = 0; z < d.z; ++z) {
					tmp[i * d.x * d.y * d.z + y * d.x * d.z + x + z * d.x] = data[i * d.x * d.y * d.z + y * d.z * d.x + x * d.z + z];
				}
			}
		}
	}
	for (int i = 0; i < b; ++i) {//add batch 
		for (int y = 0; y < d.y; ++y) {
			for (int z = 0; z < d.z; ++z) {
				for (int x = 0; x < d.x; ++x) {
					data[i * d.x * d.y * d.z + z * d.x * d.y + y * d.x + x] = tmp[i * d.x * d.y * d.z + y * d.x * d.z + z * d.x + x];//add batch 																											  //new_data[z*d.y*d.x + y*d.x + x] = data[y*d.x*d.z + z*d.x + x];
				}
			}
		}
	}
	//free(tmp);
}

int ai2_isFloatWhole(float f) { 
	return (ceilf(f) == f) ? 1 : 0;
}

ai2_bin_conv_layer ai2_make_bin_conv_layer(int batch, int n, int c, int w, int h, int size, int stride, int padding, int out_w, int out_h) {
	// http://cs231n.github.io/convolutional-networks/
	//  See: spatial arrangement section for determining what the output size will be
	if (ai2_isFloatWhole(out_w) == 0) {
		fprintf(stderr, "ERROR! conv layer of (b,c,ix,iy,s,pad) = (%d, %d, %d, %d, %d, %d) will give "
			" invalid output dimension: %fx%f\n", batch, c, w, h, stride, padding, out_w, out_w);
		exit(1);
	}
	// TODO: Support strided output
	if (stride != 1) {
		fprintf(stderr, "ERROR! Only stride values of 1 is supported\n");
		exit(1);
	}
	ai2_bin_conv_layer l = { 0 }; // initialize all to 0
	l.binary_input = calloc(c * w * h * batch / BITS_PER_BINARY_WORD, sizeof(BINARY_WORD));  
	l.binary_weights = calloc(c * n * size * size / BITS_PER_BINARY_WORD, sizeof(BINARY_WORD));
	l.beta = calloc(w * h * batch, sizeof(float));
	l.alpha= calloc(n, sizeof(float));
	l.batch = batch;
	l.n = n;
	l.c = c;
	l.h = w;
	l.w = h;
	l.stride = stride;
	l.pad = padding;
	l.size = size;
	l.out_w = out_w;
	l.out_h = out_h;
	// The following parameters are uninitialized and should be set elsewhere:
	//  l.beta  - padded
	//  l.alpha - not padded

	return l;
}

void ai2_free_bin_conv_layer(ai2_bin_conv_layer *l) {

	if (l->binary_input)
		free(l->binary_input);
	l->binary_input = NULL;
	if (l->binary_weights) 
		free(l->binary_weights);
	l->binary_weights = NULL;
	if (l->alpha)
		free(l->alpha);
	l->alpha = NULL;
	if (l->beta)
		free(l->beta);
	l->beta = NULL;
	//free(l);
}

void ai2_throw_error(char *str) {
	fprintf(stderr, "ERROR: %s\n", str);
	exit(1);
}

void ai2_bin_forward(layer l) {

	if (l.c % 32 != 0) ai2_throw_error("Channel is not divisible by 32. Need to implement mask "
		"before supporting arbitrary channel size. For now, "
		"set the channel size to the nearest multiple of 32 "
		"and ignore any ''extra'' channels unused.");

	int c_tmp = l.c >>5;   // For compensating with doing more work per word

	float *alpha = l.alpha_xnor;
	//float *beta = l.beta_xnor;
	//m是卷积核的个数，k是每个卷积核的参数数量（l.size是卷积核的大小），n是每个输出feature map的像素个数
	int m = l.n;
	int k = l.weightss>> 5;
	int n = l.out_w * l.out_h;
	int image = l.inputs>> 5;
	BINARY_WORD *b = l.b_space;
	BINARY_WORD *a = l.binary_weights_xnor;
	float *c = l.output;
	clock_t start, finish; double duration; start = clock();
	for (int i = 0; i < l.batch; ++i) {
		//im2col就是image to column,就是将图像依照卷积核的大小拉伸为列向量，方便矩阵运算
		//完成一张图的K
		 //im2col_cpu(beta, 1, l.w, l.h, l.size, l.stride, l.pad, l.a_space);
		// for (int k1 = 0; k1 <l.size*l.size; ++k1) {// k = l.size*l.size*l.c;
		//	 for (int j = 0; j < n; ++j) {//n = out_h*out_w;
		//		 l.K[j] += l.a_space[k1 * n + j];//B=state.workspace;C=output
		//	 }
		// }
		//gemm(0, 0, 1, n, k, 1, l.k_xnor, k, l.a_space, n, 1, l.K, n);
		im2col_xnor(l.binary_input_xnor, c_tmp, l.w, l.h, l.size, l.stride, l.pad, b);
		//m是卷积核的个数，k是每个卷积核的参数数量（l.size是卷积核的大小），n是每个输出feature map的像素个数
		for (int t = 0; t < m; ++t) {
			//printf("%d ", t);
			for (int k1 = 0; k1 < k; ++k1) {// k = l.size*l.size*l.c;
				BINARY_WORD A_PART = a[t * k  + k1];//A为权重
				for (int j = 0; j < n; ++j) {//n = out_h*out_w;
					BINARY_WORD B_PART = b[k1 * n + j];
					if (B_PART == 0)
						continue;
					//if (t < 1 && k1 < l.size * l.size) 
					//l.K[j] += l.a_space[k1 * n + j];
					
					else if (k1 == k - 1) {
						c[t * n + j] += (2.0*(float)_mm_popcnt_u64(~(A_PART ^ B_PART)) - 32.0);//B=state.workspace;C=output
						c[t * n + j] *= alpha[t];
					}
					else
						c[t * n + j] += (2.0*(float)_mm_popcnt_u64(~ (A_PART ^ B_PART))-32.0);//B=state.workspace;C=output
				}
			}
			//ai2_pointwise_mul_mm(c+t*n, l.K, n, 1.0, alpha[t]);
		}
		c += n * m;
		//更新输入
		//beta += l.w * l.h;
		l.binary_input_xnor += image;
	}
	finish = clock(); duration = (double)(finish - start) / CLOCKS_PER_SEC; printf("2:%f seconds\n", duration);
}
