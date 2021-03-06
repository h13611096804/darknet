#ifndef AI2_BINARY_CONVOLUTION_H
#define AI2_BINARY_CONVOLUTION_H

/** @file binary_convolution.h
 *  @brief Routines related for approximating convolutions using binary operations
 *      
 *  @author Carlo C. del Mundo (carlom)
 *  @date 05/23/2016
 */
#include "convolutional_layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
typedef uint32_t BINARY_WORD;
#define BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT)
typedef struct {
    int batch;   // number of filter batches
    int c;       // channels, z
    int h;       // height, y
	int n;
    int w;       // width, x
    int stride;
    int pad;

    int size;
	int out_h;
	int out_w;

    BINARY_WORD *binary_input;

    BINARY_WORD *binary_weights;

    float *alpha;       // we assume alpha is calculated at the beginning of initialization
    float *beta;        // we assume beta is given to us
   // float *new_beta;    // we calculate the new beta for the next layer

   // struct ai2_bin_conv_layer *next;
} ai2_bin_conv_layer;
static inline BINARY_WORD ai2_bitset(BINARY_WORD bword, BINARY_WORD mask) { return bword | mask; }
/** @brief Performs a binary convolution using XNOR and POPCOUNT between input and weights
 *
 *  @param output A 2D real-valued plane to store the outputs
 *  @param input A 2D binary-valued plane that holds the inputs
 *  @param weights A 2D binary-valued plane that holds the weights 
 *  @param ix   the input's x dimension 
 *  @param iy   the input's y dimensions
 *  @param wx   the weight's x dimension
 *  @param wy   the weight's y dimension
 *  @param pad  the amount of padding applied to input. (ix+2*pad is the x dimension of the input
 *  @param stride NOP. TODO: implement stride. the stride between sliding windows
 *  @return the count of all overlapping set bits between the two volumes.
 */
void ai2_bin_conv2D(float *output, const BINARY_WORD *input, const BINARY_WORD *weights, int ix, int iy, int wx, int wy, int pad, int stride);

/** @brief Performs a binary dot product (XNOR and POPCOUNT) for two equal sized volumes.
 *
 *  @param a A 3D binary tensor
 *  @param b A 3D binary tensor 
 *  @param vdim the dimensionality of the data. Note: we pack 32 elements in the Z element.
 *  @return the count of all overlapping set bits between the two volumes.
 */
int ai2_bin_dp(BINARY_WORD *a, BINARY_WORD *b, dim3 vdim);

/** @brief Calculates the alpha plane given an alpha volume. 
 *
 *  Each point in the yz alpha plane
 *  is the average sum of the absolute value of all elements in the z-direction.
 *
 * Pre-conditions: 
 *                  alpha_volume is an array of size x*y*z.
 *                  alpha_plane is an array of size x*y.
 *                  alpha_volume (x,y,z) is transposed to (z,x,y).
 *
 *  @param alpha_plane The 2D real-valued output plane
 *  @param alpha_volume The 3D real-valued output volume
 *  @param vdim the dimensionality of alpha_volume.
 */
void ai2_calc_alpha(float *alpha_plane, float *alpha_volume, dim3 vdim, int n);

/** @brief Wrapper function for generating the beta scaling factor */
void ai2_calc_beta(float *beta_plane, float *beta_volume, dim3 vdim,int n); 

/** @brief Set the bit in a binary word */


/** @brief Checks that the bit is set in a binary word */
int ai2_is_set(BINARY_WORD bword, unsigned int position) ;

/** @brief Converts a 3D float tensor into a 3D binary tensor.
 *
 *  The value of the ith element in the binary tensor is the sign
 *  of the ith element in the floating tensor.
 *
 *  @param binary_vol the binary tensor
 *  @param real_vol the real tensor
 *  @param vdim the size of the 3D tensor
 */
void ai2_flt_to_bin(BINARY_WORD *binary_vol, float *real_vol, float *alpha, dim3 dim, int b);
void ai2_input_to_bin(BINARY_WORD *binary_vol, float *real_vol, dim3 dim, int b);
/** @brief Converts a 3D binary tensor into a 3D float tensor.
 *
 * The ith float element will be '1' if the ith binary element is '1'.
 * Otherwise, the float element will be '-1'.
 *
 *  @param real_vol the output real tensor
 *  @param binary_vol the input binary tensor
 *  @param vdim the dimension of both binary_vol and real_vol
 */
void ai2_bin_to_flt(float *real_vol, BINARY_WORD *binary_vol, dim3 vdim); 

/** @brief Performs a pointwise matrix multication between two 2D tensors
 *  @param output A 2D real-valued plane to store the outputs
 *  @param input A 2D binary-valued plane that holds the inputs
 *  @param N the number of elements between the arrays
 */
void ai2_pointwise_mul_mm(float *output, const float *input, int N,float n,float a);

/** @brief Performs a tiled pointwise matrix multiplication between two 2D tensors
 *  
 *  Pre-conditions: wx < ix, and wy < iy
 *
 *  @param output A 2D real-valued plane of size ix, iy
 *  @param alpha A 2D binary-valued plane of size wx, wy
 *  @param ix   the output's x dimension 
 *  @param iy   the output's y dimensions
 *  @param wx   the alpha's x dimension
 *  @param wy   the alpha's y dimension
 *  @param pad  how many cells are padded, adds 2*pad to the borders of the image 
 */
void ai2_pointwise_mul_mm_2d(float *output, const float *alpha, int ix, int iy, int wx, int wy, int pad);

// --------------------------------------
//  SETTER FUNCTIONS
// --------------------------------------
/** @brief Safe function to set the float input of a conv_layer
 */
void ai2_setFltInput(layer l, network_state net);

/** @brief Safe function to set the binary weights of a conv_layer
 */
void ai2_setFltWeights(layer l);

/** @brief Safe function to set the new_beta of a conv_layer
 */
//void ai2_setFltNewBeta(ai2_bin_conv_layer *layer, float *new_new_beta);


/** @brief 3D tranpose from (x,y,z) to (z,y,x)
 *  @return a new pointer with the transposed matrix
 */
void ai2_transpose3D(float *data, float *tmp, dim3 d, int b);

void ai2_transpose3D_reverse(BINARY_WORD *data, BINARY_WORD *tmp, dim3 d, int b);
/** @brief Checks if a float is a whole number (e.g., an int)
 */
int ai2_isFloatWhole(float f);

/* @brief Allocates all memory objects in an ai2_bin_conv_layer
 * b - batches (number of filter batches)
 * c - input channels
 * ix - input width
 * iy - input height
 * wx - weight/filter width
 * wy - weight/filter height
 * s - stride between sliding windows
 * pad - the amount of padding
 */
ai2_bin_conv_layer ai2_make_bin_conv_layer(int batch, int n, int c, int w, int h, int size, int stride, int padding, int out_h, int out_w);

/* @brief Safe deallocation of  all memory objects in an ai2_bin_conv_layer
 */
void ai2_free_bin_conv_layer(ai2_bin_conv_layer *l);

/* @brief Given real-valued filter data and a conv layer, performs a forward pass
 */
void ai2_bin_forward(layer l);

#endif
