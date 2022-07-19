#pragma once

#define AP_INT_MAX_W 4096


//#include "/home/nunigan/Documents/hw-ai/HLS/src/hlslib/include/hlslib/xilinx/DataPack.h"
#include "/media/nunigan/SSD2/work/HLS/hw-ai/HLS/src/hlslib/include/hlslib/xilinx/DataPack.h"
#include "mnist.h"
#include <hls_vector.h>


constexpr int network_info[] = {784, 128, 256, 10};
constexpr int data_offset[] =  {0, network_info[0], network_info[0]+network_info[1],  network_info[0]+network_info[1]+network_info[2], network_info[0]+network_info[1]+network_info[2]+network_info[3]};
constexpr int weights_offset[] = {0, network_info[0]*network_info[1], network_info[0]*network_info[1]+network_info[1]*network_info[2]};
constexpr int bias_offset[] = {0, network_info[1], network_info[1]+network_info[2]};
constexpr int n_layers = 3;


//typedef  hls::vector<int8_t, 256> t_int8Vec;
//typedef  hls::vector<int16_t, 256> t_int16Vec;

using Vec_t = hlslib::DataPack<int8_t, 256>;
using Vec_t_int = hlslib::DataPack<int16_t, 256>;

using Vec_t1 = hlslib::DataPack<int8_t, 128>;
using Vec_t2 = hlslib::DataPack<int8_t, 256>;
using Vec_t3 = hlslib::DataPack<int8_t, 16>;

using Vec_t1_int16 = hlslib::DataPack<int16_t, 128>;
using Vec_t2_int16 = hlslib::DataPack<int16_t, 256>;
using Vec_t3_int16 = hlslib::DataPack<int16_t, 16>;

//using Vec_t1_int = hlslib::DataPack<int, 128>;
//using Vec_t2_int = hlslib::DataPack<int, 256>;
//using Vec_t3_int = hlslib::DataPack<int, 10>;


//void FullyConnectedLayerRelu(const float A[], const Vec_t B[], Vec_t C[],const float bias[]);
//void FullyConnectedLayer(const float A[], const Vec_t B[], Vec_t C[], const float bias[]);
void MultilayerPerceptron(const int8_t im[784], int8_t out[10]);
void MultilayerPerceptronCombined(const int8_t im[], int8_t out[]);
void FullyConnectedLayerLast(const int8_t A[], Vec_t3 C[], const int8_t bias[], int8_t scale);
void FullyConnectedLayerRelu1(const int8_t A[], Vec_t1 C[], const int8_t bias[], int8_t scale);
void FullyConnectedLayerRelu2(const int8_t A[],  Vec_t2 C[], const int8_t bias[], int8_t scale);
void FullyConnectedLayer(const int8_t A[], const Vec_t B[], Vec_t C[], const int8_t bias[], const int8_t scale, int K, int N, int M, int D, int W, bool relu);
