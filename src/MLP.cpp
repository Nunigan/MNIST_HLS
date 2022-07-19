#include "MLP.h"
//const unsigned int c_size = MAX_SIZE;

//extern "C" {
//void mmult(const int* a, // Read-Only Matrix A
//           const int* b, // Read-Only Matrix B
//		   int* c,       // Output Result
//           int a_row,    // Matrix A Row Size
//           int a_col,    // Matrix A Col Size
//           int b_col     // Matrix B Col Size
//           ) {
//
//    int b_row = a_col;
//    int c_row = a_row;
//    int c_col = b_col;
//
//    // Local memory to store input and output matrices
//    int localA[MAX_SIZE][MAX_SIZE];
//#pragma HLS ARRAY_PARTITION variable = localA dim = 1 complete
//
//    int localB[MAX_SIZE][MAX_SIZE];
//#pragma HLS ARRAY_PARTITION variable = localB dim = 2 complete
//
//    int localC[MAX_SIZE][MAX_SIZE];
//#pragma HLS ARRAY_PARTITION variable = localC dim = 0 complete
//
//// Burst reads on input matrices from global memory
//// Read Input A
//// Auto-pipeline is going to apply pipeline to these loops
//readA:
//    for (int loc = 0, i = 0, j = 0; loc < a_row * a_col; loc++, j++) {
//#pragma HLS LOOP_TRIPCOUNT min = 128 max = 784
//        if (j == a_col) {
//            i++;
//            j = 0;
//        }
//        localA[i][j] = a[loc];
//    }
//
//// Read Input B
//readB:
//    for (int loc = 0, i = 0, j = 0; loc < b_row * b_col; loc++, j++) {
//#pragma HLS LOOP_TRIPCOUNT min = 256*10 max = 784*128
//#pragma HLS PIPELINE II=1
//
//        if (j == b_col) {
//            i++;
//            j = 0;
//        }
//        localB[i][j] = b[loc];
//    }
//
//// Perform systolic matrix multiply
//// local matrices localA and localB have been partitioned in dimensions
//// 1 and 2 respectively. local matrix C has been partitioned completely
//
//// This partitioning enables to access MAX_SIZE elements in parallel in
//// the local matrices. Because of the mode of access of array elements,
//// we are able to perform MAX_SIZE*MAX_SIZE operations in parallel.
//
//// Note : i, j and k loops are interchanged.
//
//// The top loop systolic1 runs only for a_col iterations instead of
//// MAX_SIZE like the inner loops. The inner loops have fixed loop
//// iteration counts to enable complete unroll
//
//// The following diagram explains how the matrix multiply happens
////
////        B_0        B_1        B_2        B_3
////         |          |          |          |
////         v          v          v          v
////        ___        ___        ___        ___
////       |   |      |   |      |   |      |   |
////  A0_->|C00| ---- |C01| ---- |C02| ---- |C03|
////       |___|      |___|      |___|      |___|
////         |          |          |          |
////        ___        ___        ___        ___
////       |   |      |   |      |   |      |   |
////  A1_->|C10| ---- |C11| ---- |C12| ---- |C13|
////       |___|      |___|      |___|      |___|
////         |          |          |          |
////        ___        ___        ___        ___
////       |   |      |   |      |   |      |   |
////  A2_->|C20| ---- |C21| ---- |C21| ---- |C21|
////       |___|      |___|      |___|      |___|
////         |          |          |          |
////        ___        ___        ___        ___
////       |   |      |   |      |   |      |   |
////  A3_->|C30| ---- |C31| ---- |C32| ---- |C33|
////       |___|      |___|      |___|      |___|
//
//systolic1:
//    for (int k = 0; k < a_col; k++) {
//	#pragma HLS LOOP_TRIPCOUNT min = 784 max = 784
//    systolic2:
//        for (int i = 0; i < 1; i++) {
//		#pragma HLS UNROLL
//        systolic3:
//            for (int j = 0; j < 128; j++) {
//			#pragma HLS UNROLL
//                // Get previous sum
//                int last = (k == 0) ? 0 : localC[i][j];
//
//                // Update current sum
//                // Handle boundary conditions
//                int a_val = (i < a_row && k < a_col) ? localA[i][k] : 0;
//                int b_val = (k < b_row && j < b_col) ? localB[k][j] : 0;
//                int result = last + a_val * b_val;
//
//                // Write back results
//                localC[i][j] = result;
//            }
//        }
//    }
//
//// Burst write from output matrices to global memory
//// Burst write from matrix C
//writeC:
//    for (int loc = 0, i = 0, j = 0; loc < c_row * c_col; loc++, j++) {
//	#pragma HLS LOOP_TRIPCOUNT min = 256 max = 256
//        if (j == c_col) {
//            i++;
//            j = 0;
//        }
//        c[loc] = localC[i][j];
//    }
//}
//}





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

void FullyConnectedLayerRelu1(const int8_t A[], Vec_t1 C[],const int8_t bias[], int8_t scale) {
#pragma HLS INLINE

	constexpr int N = 1;
	constexpr int M = 128;
	constexpr int K = 784;
	constexpr int D = 1;
	constexpr int W = 128;

	const Vec_t1* B = reinterpret_cast<Vec_t1 const*>(weights1);

	for (int n = 0; n < N / D; ++n) {

		Vec_t1_int16 acc[D][M / W];
		#pragma HLS ARRAY_PARTITION variable=acc dim=1 complete

		for (int k = 0; k < K; ++k) {

			int a_buffer[D];
			for (int nd = 0; nd < D; ++nd) {
			#pragma HLS PIPELINE II=1
				a_buffer[nd] = A[n * D * K + nd * K + k];
			}


			for (int m = 0; m < M / W; ++m) {
			#pragma HLS PIPELINE II=1
				const auto b_val = B[k * (M / W) + m];
				for (int nd = 0; nd < D; ++nd) {
				#pragma HLS UNROLL
					const auto prev = (k > 0) ? acc[nd][m] : Vec_t1_int16 {static_cast<int16_t>(0) };
					for (int i = 0; i < M; ++i) {
						acc[nd][m][i] = prev[i] + a_buffer[nd] * (int16_t)b_val[i];
					}
				}
			}
		}

		for (int nd = 0; nd < D; ++nd) {
			for (int m = 0; m < M / W; ++m) {
			#pragma HLS LOOP_FLATTEN
			#pragma HLS PIPELINE II=1
				Vec_t1_int16 tmp = acc[nd][m];
				for (int i = 0; i < 128; ++i) {
					tmp[i] = (tmp[i] + bias[i]) >> scale;
					C[n * D * (M / W) + nd * (M / W) + m][i] = (tmp[i] < 0) ? 0 : tmp[i];
				}
			}
		}
	}
}

void FullyConnectedLayerRelu2(const int8_t A[], Vec_t2 C[], const int8_t bias[], int8_t scale) {
#pragma HLS INLINE

	constexpr int N = 1;
	constexpr int M = 256;
	constexpr int K = 128;
	constexpr int D = 1;
	constexpr int W = 256;

	const Vec_t2* B = reinterpret_cast<Vec_t2 const*>(weights2);

	for (int n = 0; n < N / D; ++n) {

		Vec_t2_int16 acc[D][M / W];
		#pragma HLS ARRAY_PARTITION variable=acc dim=1 complete

		for (int k = 0; k < K; ++k) {

			int a_buffer[D];
			for (int nd = 0; nd < D; ++nd) {
			#pragma HLS PIPELINE II=1
				a_buffer[nd] = A[n * D * K + nd * K + k];
			}


			for (int m = 0; m < M / W; ++m) {
			#pragma HLS PIPELINE II=1
				const auto b_val = B[k * (M / W) + m];

				for (int nd = 0; nd < D; ++nd) {
				#pragma HLS UNROLL
					const auto prev = (k > 0) ? acc[nd][m] : Vec_t2_int16 {static_cast<int16_t>(0) };
					for (int i = 0; i < M; ++i) {
					acc[nd][m][i] = prev[i] + a_buffer[nd] * (int16_t)b_val[i];

				}
				}
			}

		}

		for (int nd = 0; nd < D; ++nd) {
			for (int m = 0; m < M / W; ++m) {
			#pragma HLS LOOP_FLATTEN
			#pragma HLS PIPELINE II=1
				Vec_t2_int16 tmp = acc[nd][m];
				for (int i = 0; i < 256; ++i) {
					tmp[i] = (tmp[i] + bias[i]) >> scale;
					C[n * D * (M / W) + nd * (M / W) + m][i] =
							(tmp[i] < 0) ? 0 : tmp[i];
				}
			}
		}
	}
}

void FullyConnectedLayerLast(const int8_t A[], Vec_t3 C[], const int8_t bias[], int8_t scale) {
#pragma HLS INLINE

	constexpr int N = 1;
	constexpr int M = 10;
	constexpr int K = 256;
	constexpr int D = 1;
	constexpr int W = 10;

	const Vec_t3* B = reinterpret_cast<Vec_t3 const*>(weights3);

	for (int n = 0; n < N / D; ++n) {

		Vec_t3_int16 acc[D][M / W];
		#pragma HLS ARRAY_PARTITION variable=acc dim=1 complete

		for (int k = 0; k < K; ++k) {

			int a_buffer[D];
			for (int nd = 0; nd < D; ++nd) {
			#pragma HLS PIPELINE II=1
				a_buffer[nd] = A[n * D * K + nd * K + k];
			}

			for (int m = 0; m < M / W; ++m) {
			#pragma HLS PIPELINE II=1
				const auto b_val = B[k * (M / W) + m];

				for (int nd = 0; nd < D; ++nd) {
				#pragma HLS UNROLL
					const auto prev = (k > 0) ? acc[nd][m] : Vec_t3_int16 {static_cast<int16_t>(0) };
					for (int i = 0; i < M; ++i) {
					acc[nd][m][i] = prev[i] + a_buffer[nd] *  (int16_t)b_val[i];
					}
				}
			}

		}

		for (int nd = 0; nd < D; ++nd) {
			for (int m = 0; m < M / W; ++m) {
			#pragma HLS LOOP_FLATTEN
			#pragma HLS PIPELINE II=1
				Vec_t3_int16 tmp = acc[nd][m];
				for (int i = 0; i < W; ++i) {
					tmp[i] = (tmp[i] + bias[i]) >> scale;
					C[n * D * (M / W) + nd * (M / W) + m][i] = tmp[i];
				}
			}
		}
	}
}


void MultilayerPerceptronCombined(const int8_t im[7840000], int8_t out[10000]) {
#pragma HLS INTERFACE m_axi port=im bundle=gmem0 offset=slave depth=7840000
#pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave depth=10000
#pragma HLS INTERFACE s_axilite port=im bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	int8_t data1[128];
	int8_t data2[256];
	int8_t data3[16];

	for(int i = 0; i<10000; ++i){
//		#pragma HLS PIPELINE II=784

		FullyConnectedLayer(im+784*i,    reinterpret_cast<Vec_t const *>(weights1),  reinterpret_cast<Vec_t *>(data1), bias+bias_offset[0], scales[0], network_info[0], 1, network_info[1], 1, network_info[1], true);
//		 for(int i = 0; i<128; ++i)
//		 	  {
//		 	    if(data1[i]-res_layers[i] != 0){
//		 	        printf("Wrong result layer 3, pos %d: res = %d, correct = %d \n", i,data1[i], res_layers[i]);
//		 	    }
//		 	  }
		FullyConnectedLayer(data1, reinterpret_cast<Vec_t const *>(weights2),  reinterpret_cast<Vec_t *>(data2), bias+bias_offset[1], scales[1], network_info[1], 1, network_info[2], 1, network_info[2], true);
//		 for(int i = 0; i<256; ++i)
//		 	  {
//		 	    if(data2[i]-res_layers[i+128] != 0){
//		 	        printf("Wrong result layer 3, pos %d: res = %d, correct = %d \n", i,data2[i], res_layers[i+128]);
//		 	    }
//		 	  }
		FullyConnectedLayer(data2, reinterpret_cast<Vec_t const *>(weights3),  reinterpret_cast<Vec_t *>(data3), bias+bias_offset[2], scales[2], network_info[2], 1, network_info[3], 1, network_info[3], false);
//		 for(int i = 0; i<10; ++i)
//		 	  {
//		 	    if(data3[i]-res_layers[i+128+256] != 0){
//		 	        printf("Wrong result layer 3, pos %d: res = %d, correct = %d \n", i,data3[i], res_layers[i+128+256]);
//		 	    }
//		 	  }

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


void test_axi(const int8_t a[10], int8_t c[10]){
#pragma HLS INTERFACE m_axi port=a depth=1000
#pragma HLS INTERFACE s_axilite port=c bundle=BUS_A
#pragma HLS INTERFACE s_axilite port=return bundle=BUS_A

	for(int i = 0; i<10; ++i){
	#pragma HLS PIPELINE II=1
		c[i] = 2*a[i];
	}
}



void MultilayerPerceptron(const int8_t im[7840000], int8_t out[10000]) {
#pragma HLS INTERFACE m_axi port=im bundle=gmem0 offset=slave depth=7840000
#pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave depth=10000
#pragma HLS INTERFACE s_axilite port=im bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	int8_t data1[128];
	int8_t data2[256];
	int8_t data3[16];

	for(int i = 0; i<1; ++i){
	FullyConnectedLayerRelu1(im+784*i,    reinterpret_cast<Vec_t1*>(data1), bias + bias_offset[0], scales[0]);

//	 for(int i = 0; i<128; ++i)
//	 	  {
//	 	    if(data1[i]-res_layers[i] != 0){
//	 	        printf("Wrong result layer 1, pos %d: res = %d, correct = %d \n", i,data1[i], res_layers[i]);
//	 	    }
//	 	  }

    FullyConnectedLayerRelu2(data1, reinterpret_cast<Vec_t2*>(data2), bias + bias_offset[1], scales[1]);

//	 for(int i = 0; i<256; ++i)
//		 	  {
//		 	    if(data2[i]-res_layers[i+128] != 0){
//		 	        printf("Wrong result layer 2, pos %d: res = %d, correct = %d \n", i,data2[i], res_layers[i+128]);
//		 	    }
//		 	  }

	FullyConnectedLayerLast(data2,      reinterpret_cast<Vec_t3*>(data3), bias + bias_offset[2], scales[2]);

//	 for(int i = 0; i<10; ++i)
//	 	  {
//	 	    if(data3[i]-res_layers[i+128+256] != 0){
//	 	        printf("Wrong result layer 3, pos %d: res = %d, correct = %d \n", i,data3[i], res_layers[i+128+256]);
//	 	    }
//	 	  }

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

//void test(const uint weights[], const uint im[]) {
//	uint data[128];
//  mmult(im, weights, data, 1,784,128);
//}
