// The constant memory is used here to gain some 
// performance improvement. We use it when we know the data
// will not change during the execution and when we know all thread
// will access data from the same part of the memory.
// Anything going into the constant memory will be cached in the constant cache.
// and this is what is helping us to improve the performance when compared to 
// accessing it from the device memory. Constant memory is always declared as global.



// This program implements a 1D convolution using CUDA,
// and stores the mask in constant memory, and loads
// reused values into shared memory (scratchpad)
// By: Nick from CoffeeBeforeArch

#include <cassert>
#include <cstdlib>
#include <iostream>

// Length of our convolution mask
#define MASK_LENGTH 7

// Allocate space for the mask in constant memory
__constant__ int mask[MASK_LENGTH];

// 1-D convolution kernel
// Some threads load multiple elements into shared memory
// All threads compute 1 element in final array.
// We have the second set of loading since we are padding
// the array beginning and the end with the radius of the 
// mask or else we could have simply loaded the 256 elements 
// per block and handle the halo (outliers) separately.
// Thus as per our approach we need to have the offset to 
// access the shared memory per thread block correctly and the 
// global offset to read the elements from the global memory correctly.
// Each Thread block with 256 threads(as we have provided) will carry out 
// this process of loading the first part and the then the remaining using the 
// offset.
__global__ void convolution_1d_tiled(int* arr, int* result, int n) {
    // Global thread ID calculation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Store all elements needed to compute output in shared memory
    extern __shared__ int s_array[];

    //The number of padded elements on either side
    int mask_radius = MASK_LENGTH / 2;

    //The total number of padded elements
    int total_padding = 2 * mask_radius;

    // Size in number of elements of the padded shared memory
    // here in our case it will be 256 + total_padding
    int n_padded = blockDim.x + total_padding;

    // Offset for the second set of loads in shared memory
    int offset = threadIdx.x + blockDim.x;

    // Global offset for the array in DRAM
    int g_offset = blockDim.x * blockIdx.x + offset;

    // Load the lower elements first starting at the halo
    s_array[threadIdx.x] = arr[tid];

    // Load in the remaining upper elements
    // and making sure it is within the bounds
    if (offset < n_padded) {
        s_array[offset] = arr[g_offset];
    }
    // Wait until all the elements are loaded.
    __syncthreads();

    // Temp value for calculation
    int temp = 0;

    // Go over each element of the mask
    // we do not need the if statement as we
    // we used in the case of constant memory
    // approach because here we pad it and 
    // initialize to 0 so its safe.
    for (int i = 0; i < MASK_LENGTH; i++) {
        temp += s_array[threadIdx.x + i] * mask[i];
    }

    // Write-back the results
    // to the global result array.
    // and here therefore we will need
    // the Global thread ID calculation.
    result[tid] = temp;
}

// Verify the result on the CPU
void verify_result(int* array, int* mask, int* result, int n) {
    int temp;
    for (int i = 0; i < n; i++) {
        temp = 0;
        for (int j = 0; j < MASK_LENGTH; j++) {
            temp += array[i + j] * mask[j];
        }
        assert(temp == result[i]);
    }
}

int main() {
    // Number of elements in result array
    int n = 1 << 20;

    // Size of the array in bytes
    int bytes_n = n * sizeof(int);

    // Size of the mask in bytes
    size_t bytes_m = MASK_LENGTH * sizeof(int);

    // Radius for padding the array
    int mask_radius = MASK_LENGTH / 2;
    // number of elements in padded array
    int n_p = n + mask_radius * 2;

    // Size of the padded array in bytes
    size_t bytes_p = n_p * sizeof(int);

    // Allocate the array to 
    // to be convoluted.
    int* h_array = new int[n_p];

    //Fill the array with elements
    for (int i = 0; i < n_p; i++) {
        // The padded region is initialized to 0
        if ((i < mask_radius) || (i >= (n + mask_radius))) {
            h_array[i] = 0;
        }
        else {
            h_array[i] = rand() % 100;
        }
    }

    // Allocate the mask and initialize it
    int* h_mask = new int[MASK_LENGTH];
    for (int i = 0; i < MASK_LENGTH; i++) {
        h_mask[i] = rand() % 10;
    }

    // Allocate space for the result
    int* h_result = new int[n];

    // Allocate space on the device
    int* d_array, * d_result;
    cudaMalloc(&d_array, bytes_p);
    cudaMalloc(&d_result, bytes_n);

    // Copy the data to the device
    cudaMemcpy(d_array, h_array, bytes_p, cudaMemcpyHostToDevice);

    // Copy the mask directly to the symbol
    // This would require 2 API calls with cudaMemcpy
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);

    // Threads per thread block
    int THREADS = 256;

    // Number of thread blocks with padding
    int GRID = (n + THREADS - 1) / THREADS;

    // Amount of space per-block for shared memory
    // This is padded by the overhanging radius on either side
    size_t SHMEM = (THREADS + mask_radius * 2) * sizeof(int);

    // Call the kernel
    convolution_1d_tiled << <GRID, THREADS, SHMEM >> > (d_array, d_result, n);

    // Copy back the result
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Verify the result
    verify_result(h_array, h_mask, h_result, n);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    // Free allocated memory on the device and host
    delete[] h_array;
    delete[] h_result;
    delete[] h_mask;
    cudaFree(d_result);

    return 0;
}

//Notes:
// https://www.youtube.com/watch?v=pBB8mZRM91A&list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU&index=19
// Shared memory is very fast(register speeds). It is shared between threads of each block and is
// therefore thread block private.
// Bank conflicts can slow down access. 
