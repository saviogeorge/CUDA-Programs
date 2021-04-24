
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>


//Kernel
__global__ void naive_1d_convolution(int* d_array, int* d_mask, int* d_result, int array_size, int mask_size)
{
    // Global thread ID calculation
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate radius of the mask
    int r = mask_size / 2;

    // Calculate the starting point for the element
    int start = thread_id - r;

    // Temp value for calculation
    int temp = 0;

    // Go over each element of the mask
    for (int j = 0; j < mask_size; j++) {
        // Ignore elements that hang off (0s don't contribute)
        if (((start + j) >= 0) && (start + j < array_size)) {
            // accumulate partial results
            temp += d_array[start + j] * d_mask[j];
        }
    }

    // Write-back the results
    d_result[thread_id] = temp;
}

// Verify the result on the CPU
void verify_result(int* array, int* mask, int* result, int n, int m) {
    int radius = m / 2;
    int temp;
    int start;
    for (int i = 0; i < n; i++) {
        start = i - radius;
        temp = 0;
        for (int j = 0; j < m; j++) {
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

    // Number of elements in the convolution mask
    int  mask_size = 7;

    // Size of mask in bytes
    int mask_bytes = mask_size * sizeof(int);

    // Allocate the array (include edge elements)...
    std::vector<int> h_array(array_size);

    // ... and initialize it
    std::generate(begin(h_array), end(h_array), []() { return rand() % 100; });

    // Allocate the mask and initialize it
    std::vector<int> h_mask(mask_size);
    std::generate(begin(h_mask), end(h_mask), []() { return rand() % 10; });

    // Allocate space for the result
    std::vector<int> h_result(array_size);

    // Allocate space on the device
    int* d_array, * d_mask, * d_result;
    cudaMalloc(&d_array, array_bytes);
    cudaMalloc(&d_mask, mask_bytes);
    cudaMalloc(&d_result, array_bytes);

    // Copy the data to the device
    cudaMemcpy(d_array, h_array.data(), array_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask.data(), mask_bytes, cudaMemcpyHostToDevice);

    // Threads per Thread block
    int THREADS = 256;

    // Number of Thread blocks (with padding)
    int GRID = (array_size + THREADS - 1) / THREADS;

    // Call the kernel
    naive_1d_convolution << <GRID, THREADS >> > (d_array, d_mask, d_result, array_size, mask_size);

    // Copy back the result
    cudaMemcpy(h_result.data(), d_result, array_bytes, cudaMemcpyDeviceToHost);

    // Verify the result
    verify_result(h_array.data(), h_mask.data(), h_result.data(), array_size, mask_size);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    // Free allocated memory on the device and host
    cudaFree(d_result);
    cudaFree(d_mask);
    cudaFree(d_array);

    return 0;
}

//Notes:
//https://www.youtube.com/watch?v=OlLquh9Lnbc&list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU&index=17
    