#pragma once

#define AP_INT_MAX_W 4096

#include "DataPack.h"
#include "mnist.h"

constexpr int network_info[] = {784, 128, 256, 10};
constexpr int bias_offset[] = {0, network_info[1], network_info[1]+network_info[2]};
constexpr int n_layers = 3;

using Vec_t = hlslib::DataPack<int8_t, 256>;
using Vec_t_int = hlslib::DataPack<int16_t, 256>;

void MultilayerPerceptron(const int8_t im[], int8_t out[]);
void FullyConnectedLayer(const int8_t A[], const Vec_t B[], Vec_t C[], const int8_t bias[], const int8_t scale, int K, int N, int M, int D, int W, bool relu);
