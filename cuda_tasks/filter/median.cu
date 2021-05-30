#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

__global__ void filter(uint8_t* d_image, uint8_t* d_image_result, double *d_filter_kernel, int h, int w)
{
    int globalidx = threadIdx.x + blockDim.x * blockIdx.x;

    int size = h * w * 3;

    int x = globalidx / 3 / w;
    int y = globalidx / 3 - x * w;
    int s = globalidx - y * 3 - x * w * 3;

    int ind = x * w * 3 + y * 3 + s;

    uint8_t for_sorting[9] = {0,0,0,0,0,0,0,0,0};

    if (globalidx < size) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (x + i - 1 >= 0 && x + i - 1 < h && y + j - 1 >= 0 && y + j - 1 < w)
                    for_sorting[i * 3 + j] = d_filter_kernel[i * 3 + j] * d_image[(x + i - 1) * w * 3 + (y + j - 1) * 3 + s];
            }
        }
        thrust::sort(thrust::device, for_sorting, for_sorting + 9);
        d_image_result[ind] = for_sorting[4];
    }
}

int main(int argc, char **argv)
{
    double *h_filter_kernel = (double *) calloc(sizeof(double), 9);
    double *d_filter_kernel;

    uint8_t* d_image;
    uint8_t* d_image_result;

    for (int i = 0; i < 9; ++i) {
        h_filter_kernel[i] = 1;
    }
    h_filter_kernel[4] = 0.;

    cudaMalloc(&d_filter_kernel, sizeof(double) * 9);
    cudaMemcpy(d_filter_kernel, h_filter_kernel, sizeof(double) * 9, cudaMemcpyHostToDevice);

    int width, height, bpp;

    uint8_t* h_image = stbi_load("image.png", &width, &height, &bpp, 3);

    int size = height * width * 3;

    cudaMalloc(&d_image, sizeof(uint8_t) * size);
    cudaMalloc(&d_image_result, sizeof(uint8_t) * size);

    uint8_t* h_image_result = (uint8_t *)malloc(sizeof(uint8_t) * size);

    cudaMemcpy(d_image, h_image, sizeof(uint8_t) * size, cudaMemcpyHostToDevice);

    dim3 dimBlock(1024);
    dim3 dimGrid(size/1024);

    filter<<<dimGrid, dimBlock>>>(d_image, d_image_result, d_filter_kernel, height, width);

    stbi_image_free(h_image);

    cudaMemcpy(h_image_result, d_image_result, sizeof(uint8_t) * size, cudaMemcpyDeviceToHost);
    stbi_write_png("image_result.png", width, height, 3, h_image_result, width * 3);
    return 0;
}
