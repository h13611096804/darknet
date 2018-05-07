#include "xnor_layer.h"
#include "binary_convolution.h"
#include "convolutional_layer.h"

layer make_xnor_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation, int batch_normalize)
{
    int i;
    layer l = {0};
    l.type = XNOR;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = pad;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.biases = calloc(n, sizeof(float));

    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*out_h * out_w * n, sizeof(float));

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
    }

    l.activation = activation;

    fprintf(stderr, "XNOR Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);

    return l;
}

void forward_xnor_layer( layer l, network_state state)
{
    int b = l.batch;
	int n = l.n;
    int c = l.c;
    int w = l.w;
    int h = l.h;
    int size = l.size;
    int s = l.stride;
    int pad = l.pad;
	int out_h = l.out_h;
	int out_w = l.out_w;
    // MANDATORY: Make the binary layer
    //ai2_bin_conv_layer al = ai2_make_bin_conv_layer(b, n, c, w, h, size, s, pad, out_h,out_w);
    // OPTIONAL: You need to set the real-valued input like:
    ai2_setFltInput(l,state);
    // The above function will automatically binarize the input for the layer (channel wise).
    // If commented: using the default 0-valued input.
	//if(state.train)
		ai2_setFltWeights(l);
    // The above function will automatically binarize the input for the layer (channel wise).
    // If commented: using the default 0-valued weights.

    // MANDATORY: Call forward
    ai2_bin_forward(l);

    // MANDATORY: Free layer
  // ai2_free_bin_conv_layer(&al);
}
