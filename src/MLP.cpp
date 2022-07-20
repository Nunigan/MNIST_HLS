#include "MLP.h"

void FullyConnectedLayer(const int8_t A[], const Vec_t B[], Vec_t C[], const int8_t bias[], const int8_t scale, int K, int N, int M, int D, int W, bool relu){
#pragma HLS INLINE

	for (int n = 0; n < N / D; ++n) {
	#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1

		Vec_t_int acc[1][1];
		#pragma HLS ARRAY_PARTITION variable=acc dim=1 complete

		for (int k = 0; k < K; ++k) {
		#pragma HLS LOOP_TRIPCOUNT min = 128 max = 784

			int8_t a_buffer[1];
			for (int nd = 0; nd < D; ++nd) {
			#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
			#pragma HLS PIPELINE II=1
				a_buffer[nd] = A[n * D * K + nd * K + k];
			}

			for (int m = 0; m < 1; ++m) {
			#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
			#pragma HLS PIPELINE II=1
				const auto b_val = B[k * (M / W) + m];
				for (int nd = 0; nd < 1; ++nd) {
				#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
				#pragma HLS UNROLL
					const auto prev = (k > 0) ? acc[0][m] : Vec_t_int {static_cast<int16_t>(0) };
					for (int i = 0; i < M; ++i) {
						acc[nd][m][i] = prev[i] + a_buffer[nd] *  (int8_t)b_val[i];
					}
				}
			}
		}

		for (int nd = 0; nd < D; ++nd) {
		#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
			for (int m = 0; m < M / W; ++m) {
			#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
			#pragma HLS LOOP_FLATTEN
			#pragma HLS PIPELINE II=1
				Vec_t_int tmp = acc[nd][m];
				for (int i = 0; i < W; ++i) {
					tmp[i] = (tmp[i] + bias[i]) >> scale;
					if (relu!=true)
						C[n * D * (M / W) + nd * (M / W) + m][i] = tmp[i];
					else
						C[n * D * (M / W) + nd * (M / W) + m][i] = (tmp[i] < 0) ? 0 : tmp[i];
				}
			}
		}
	}
}


void MultilayerPerceptron(const int8_t im[], int8_t out[]) {
#pragma HLS INTERFACE m_axi port=im bundle=gmem0 offset=slave depth=7840000
#pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave depth=10000
#pragma HLS INTERFACE s_axilite port=im bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	int8_t data1[128];
	int8_t data2[256];
	int8_t data3[16];

	for(int i = 0; i<10000; ++i){
		FullyConnectedLayer(im+784*i, reinterpret_cast<Vec_t const *>(weights1), reinterpret_cast<Vec_t *>(data1), bias+bias_offset[0], scales[0], network_info[0], 1, network_info[1], 1, network_info[1], true);
		#ifndef __SYNTHESIS__
		for(int i = 0; i<128; ++i){
		 	if(data1[i]-res_layers[i] != 0)
		 		printf("Wrong result layer 3, pos %d: res = %d, correct = %d \n", i,data1[i], res_layers[i]);
		}
		#endif

		FullyConnectedLayer(data1, reinterpret_cast<Vec_t const *>(weights2), reinterpret_cast<Vec_t *>(data2), bias+bias_offset[1], scales[1], network_info[1], 1, network_info[2], 1, network_info[2], true);
		#ifndef __SYNTHESIS__
		for(int i = 0; i<256; ++i){
		 	if(data2[i]-res_layers[i+128] != 0)
		 		printf("Wrong result layer 3, pos %d: res = %d, correct = %d \n", i,data2[i], res_layers[i+128]);
		 }
		#endif

		FullyConnectedLayer(data2, reinterpret_cast<Vec_t const *>(weights3), reinterpret_cast<Vec_t *>(data3), bias+bias_offset[2], scales[2], network_info[2], 1, network_info[3], 1, network_info[3], false);
		#ifndef __SYNTHESIS__
		for(int i = 0; i<10; ++i){
		 	if(data3[i]-res_layers[i+128+256] != 0)
		 		printf("Wrong result layer 3, pos %d: res = %d, correct = %d \n", i,data3[i], res_layers[i+128+256]);
		}
		#endif

		int8_t argmax = 0;
		int8_t max = -128;
		for(int j=0; j<10; ++j)
		#pragma HLS PIPELINE II=1
		{
			if(data3[j] > max){
				argmax = j;
				max = data3[j];
			}
		}
		out[i] = argmax;
		}
}