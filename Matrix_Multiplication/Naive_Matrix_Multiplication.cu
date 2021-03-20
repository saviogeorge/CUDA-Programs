
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include<vector>
#include<iostream>
#include <cassert>

using std::vector;
using namespace std;

#define THREADS 16

// Forward declaration of the kernel
__global__ void matrixMul(const int* a, const int* b, int* c, int N);

// Compare the GPU results with CPU
void check_result(const vector<int>& a, const vector<int>& b, const vector<int>& c, const int N) {
    // row
    for (int row = 0; row < N; ++row) {
        //column
        for (int col = 0; col < N; ++col) {
            //resultant element is computed
            int element = 0;
            for (int i = 0; i < N; i++) {
                element += a[row * N + i] * b[i * N + col];
            }

            //Check CPU and GPU result
            assert(element == c[row * N + col]);
        }
    }
}

// Matrix multiplication - Host driver code
void MatMul_driver(const vector<int> &h_a, const vector<int> &h_b, vector<int> &h_c, size_t bytes, int no_elements )
{
    // Allocating device memory
    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    //Threads per block
    const dim3 blockSize(THREADS, THREADS, 1);
    //Number of blocks
    const dim3 gridSize(ceil(no_elements / (float)THREADS), ceil(no_elements / (float)THREADS), 1);

    // Launch kernel
    matrixMul <<<gridSize, blockSize>>> (d_a, d_b, d_c, no_elements);

    // Copy back to the host
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result
    check_result(h_a, h_b, h_c, no_elements);

    cout<< "SUCCESSFULLY COMPLETED\n";

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}


// Matrix multiplication kernel called by MatMul_driver()
// to be executed on the GPU.
// Each thread fetches the data from the device memory.
// Each thread reads one row of A and one column of B 
// and computes the corresponding element of C
// does not take advantage of shared memory
__global__ void matrixMul(const int* a, const int* b, int* c, int N) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    if (row < N && col < N)
    {
        c[row * N + col] = 0;
        for (int k = 0; k < N; k++) {
            // Accumulate results for a single element
            c[row * N + col] += a[row * N + k] * b[k * N + col];
        }
    }
}