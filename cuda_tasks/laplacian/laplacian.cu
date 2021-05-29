#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>

using namespace std;

const int N = 128;
const int ITERATIONS = 500;
const string filename = "result.txt";

__global__ void laplacian(int n, double *d_array, double *d_result)
{
    int globalidx = threadIdx.x + blockIdx.x * blockDim.x;
    int x = globalidx / n;
    int y = globalidx - n * x;


    if(globalidx < n * n) {
        if (y == 0 || x == 0)
            d_result[y + x * n] = 1;
        else {
            d_result[y + x * n] = 0.25 * (
                    (y - 1 >= 0 ? d_array[(y - 1) + x * n] : 0) +
                    (y + 1 <= n - 1 ? d_array[(y + 1) + x * n] : 0) +
                    (x - 1 >= 0 ? d_array[y + (x - 1) * n] : 0) +
                    (x + 1 <= n - 1 ? d_array[y + (x + 1) * n] : 0));
        }
    }

    __syncthreads();
    d_array[y + x * n] = d_result[y + x * n];
}

double* create_grid(int size) {
    double *h_array = (double *) calloc(sizeof(double), size * size);
    for (int i = 0; i < size; i++) {
        h_array[i] = 1;
        h_array[i * size] = 1;
    }
    return h_array;
}

void write_to(ofstream& writer, double *res_i, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            writer << res_i[i*N + j] << "\t";
        writer << "\n";
    }
}

int main() {
    double *d_array;
    double *d_result;

    cudaMalloc(&d_array, sizeof(double) * N * N);
    cudaMalloc(&d_result, sizeof(double) * N * N);

    double *h_array = create_grid(N);
    cudaMemcpy(d_array, h_array, sizeof(double) * N * N, cudaMemcpyHostToDevice);

    int threadNum = 1024;
    dim3 dimBlock(threadNum);
    dim3 dimGrid(N * N / threadNum);

    ofstream writer;
    writer.open(filename);

    for (int k = 0; k < ITERATIONS; k++) {
        laplacian<<<dimGrid, dimBlock>>>(N, d_array, d_result);
    }
    cudaMemcpy(h_array, d_array, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
    write_to(writer, h_array, N);
    writer.close();
    return 0;
}

