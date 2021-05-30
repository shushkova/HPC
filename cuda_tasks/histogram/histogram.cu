#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <iostream>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

using namespace std;

const string filename = "hist_result.txt";

__global__ void hist(uint8_t* d_image_gray, int* d_hist, int h, int w) {
    int globalidx = threadIdx.x + blockDim.x * blockIdx.x;

    if(globalidx >= h * w)   return;
    unsigned char value = d_image_gray[globalidx];

    int bin = value % 256;
    atomicAdd(&d_hist[bin], 1);
}

uint8_t* to_gray(uint8_t* h_image, int h, int w) {
    uint8_t* h_image_gray = (uint8_t *)malloc(sizeof(uint8_t) * h * w);
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            h_image_gray[j * w + i] =
                    0.299 * h_image[j * w + i] +
                    0.587 * h_image[j * w + i + 1] +
                    0.114 * h_image[j * w + i + 2];
        }
    }
    return h_image_gray;
}

void write_to(ofstream& writer, int *array, int n)
{
    writer << "histogram" << "\n";
    for (int i = 0; i < n; i++)
        writer << i << " : " << array[i] << "\n";
}

int main(int argc, char **argv) {
    uint8_t* d_image_gray;
    int* d_hist;

    int width, height, bpp;

    uint8_t* h_image = stbi_load("image.png", &width, &height, &bpp, 3);


    cudaMalloc(&d_image_gray, sizeof(uint8_t) * height * width);
    cudaMalloc(&d_hist, sizeof(int) * 256);

    int* h_hist = (int *)malloc(sizeof(int) * 256);

    uint8_t* h_image_gray = to_gray(h_image, height, width);

    cudaMemcpy(d_image_gray, h_image_gray, sizeof(uint8_t) * height * width, cudaMemcpyHostToDevice);

    dim3 dimBlock(256);
    dim3 dimGrid(height * width / 256);

    hist<<<dimGrid, dimBlock>>>(d_image_gray, d_hist, height, width);

    stbi_image_free(h_image);

    cudaMemcpy(h_hist, d_hist, sizeof(int) * 256, cudaMemcpyDeviceToHost);

    ofstream writer;
    writer.open(filename);
    write_to(writer, h_hist, 256);
    return 0;
}
