// The constant memory is used here to gain some 
// performance improvement. We use it when we know the data
// will not change during the execution and when we know all thread
// will access data from the same part of the memory.
// Anything going into the constant memory will be cached in the constant cache.
// and this is what is helping us to improve the performance when compared to 
// accessing it from the device memory. Constant memory is always declared as global.



#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

// Length of the mask
const int MASK_SIZE = 7;

// Allocate space for the mask in constant memory
__constant__ int mask[MASK_SIZE];


//Kernel
__global__ void naive_1d_convolution(int* d_array, int* d_result, int array_size)
{
    // Global thread ID calculation
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate radius of the mask
    int r = MASK_SIZE / 2;

    // Calculate the starting point for the element
    int start = thread_id - r;

    // Temp value for calculation
    int temp = 0;

    // Go over each element of the mask
    for (int j = 0; j < MASK_SIZE; j++) {
        // Ignore elements that hang off (0s don't contribute)
        if (((start + j) >= 0) && (start + j < array_size)) {
            // accumulate partial results
            temp += d_array[start + j] * mask[j];
        }
    }

    // Write-back the results
    d_result[thread_id] = temp;
}

// Verify the result on the CPU
void verify_result(const std::vector<int>& array, int* mask, const std::vector<int>& result, int n) {
    int radius = MASK_SIZE / 2;
    int temp;
    int start;
    for (int i = 0; i < n; i++) {
        start = i - radius;
        temp = 0;
        for (int j = 0; j < MASK_SIZE; j++) {
            if ((start + j >= 0) && (start + j < n)) {
                temp += array[start + j] * mask[j];
            }
        }
        assert(temp == result[i]);
    }
}

int main() {
    // Number of elements in result array
    int array_size = 1 << 10;

    // Size of the array in bytes
    int array_bytes = array_size * sizeof(int);

    // Size of mask in bytes
    int mask_bytes = MASK_SIZE * sizeof(int);

    // Allocate the array (include edge elements)...
    std::vector<int> h_array(array_size);

    // ... and initialize it
    std::generate(begin(h_array), end(h_array), []() { return rand() % 100; });

    // Allocate the mask and initialize it
    // with random numbers.
    int* h_mask = new int[MASK_SIZE];
    for (int i = 0; i < MASK_SIZE; i++) {
        h_mask[i] = rand() % 10;
    }

    // Allocate space for the result on host
    std::vector<int> h_result(array_size);

    // Allocate space on the device
    int* d_array, * d_result;
    cudaMalloc(&d_array, array_bytes);
    cudaMalloc(&d_result, array_bytes);

    // Copy the data to the device
    cudaMemcpy(d_array, h_array.data(), array_bytes, cudaMemcpyHostToDevice);
    
    // Copy the data directly to the symbol
    cudaMemcpyToSymbol(mask, h_mask, mask_bytes);

    // Threads per Thread block
    int THREADS = 256;

    // Number of Thread blocks (with padding)
    int GRID = (array_size + THREADS - 1) / THREADS;

    // Call the kernel
    naive_1d_convolution << <GRID, THREADS >> > (d_array, d_result, array_size);

    // Copy back the result
    cudaMemcpy(h_result.data(), d_result, array_bytes, cudaMemcpyDeviceToHost);

    // Verify the result
    verify_result(h_array, h_mask, h_result, array_size);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    // Free allocated memory on the device and host
    cudaFree(d_result);
    cudaFree(d_array);
    delete[] h_mask;

    return 0;
}

//Notes:
// http://cuda-programming.blogspot.com/2013/01/what-is-constant-memory-in-cuda.html
// https://www.youtube.com/watch?v=RY2_8wB2QY4
