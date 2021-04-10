

#include "device_launch_parameters.h"
#include <vector>
#include <algorithm> // for generate()
#include <cuda_runtime.h> // for cudaMalloc()
// #include<cuda.h>
#include <cassert>
#include <iostream>

#define THREADS 16
#define SHARED_MEM_SIZE (THREADS*THREADS)

using namespace std;

//Kernel
__global__ void matrix_mul(const int* matrice_a, const int* matrice_b, int* matrice_c, int N) {

	// Thread global row and column index is computed. 
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Allocate the static shared memory
	// this area is private per thread block
	__shared__ int Matrice_A[SHARED_MEM_SIZE];
	__shared__ int Matrice_B[SHARED_MEM_SIZE];

	// Block level thread info- within a block
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;
	int dim = blockDim.x;

	// Loop to move from one tile to the next
	// Each thread loads one element of each sub - matrix
	int temp = 0;
	for (int i = 0; i < ((N + dim - 1) / dim); i++) {
		// thread_y * dim gives us to which row we belong within the tile
		// and thread_x gives us the column.
		// row * N gives which row we belong in the global memory.
		// (i* dim) tells us in which tile are we in 
		// thread_x gives us the thread within the tile.
		Matrice_A[thread_y * dim + thread_x] = matrice_a[(row * N) + (i * dim) + thread_x];
		// thread_y * dim gives us to which row we belong within the tile
		// and thread_x gives us the column.
		// (i * dim * N) gives which row we belong in the global memory.
		// (thread_y * N) + col - each thread in the y direction is separated by N elements so.
		Matrice_B[thread_y * dim + thread_x] = matrice_b[(i * dim * N) + (thread_y * N) + col];

		// Wait for both tiles to be loaded in before doing computation
		__syncthreads();

		for (int j = 0; j < dim; j++)
		{
			temp += Matrice_A[thread_y * dim + j] * Matrice_B[j * dim + thread_x];
		}
		// Wait for the computations to complete on the loaded tile before
		// loading the next tile.
		__syncthreads();
	}
	// write back results to main memory
	matrice_c[row * N + col] = temp;
}

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

// Host driver code
int main() {

	unsigned int N = 1 << 10;
	size_t bytes = N * N * sizeof(unsigned int);

	// Allocate host and device memory

	// Matrices in host
	std::vector<int> h_a(N * N);
	std::vector<int> h_b(N * N);
	// Resultant Matrice
	std::vector<int> h_c(N * N);

	// Matrices in device
	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Initialize the matrices
	generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
	generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

	// Copy data from host to device
	cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

	//Threads per block
	const dim3 blockSize(THREADS, THREADS, 1);
	//Number of blocks
	// Here we do not necessarily need to do the padding 
	// as in our simple case we have taken a square matrice 
	// and which is of size 1024 which will be thus equally divisible 
	// by 16. However just for sake of understanding I decided to go with 
	// the padding technique.
	int blocks = (N + THREADS - 1) / THREADS;
	const dim3 gridSize(blocks, blocks, 1);

	// Launch the kernel
	matrix_mul << < gridSize, blockSize >> > (d_a, d_b, d_c, N);
	cudaDeviceSynchronize();

	// Copy back to the host
	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

	check_result(h_a, h_b, h_c, N);

	// Free dynamically allocated memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cout<< " GPU Results verified with CPU!!";
	cout<< " Found no ERRORS!!";
	cout << "Success!!";

}

//Notes/References:
//
//https://www.tutorialspoint.com/cuda/cuda_threads.htm#:~:text=The%20CUDA%20API%20has%20a,of%20them%20reaches%20the%20location.
//https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/matrixMul/tiled/mmul.cu
//https://www.youtube.com/watch?v=ga2ML1uGr5o
//https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
//https://penny-xu.github.io/blog/tiled-matrix-multiplication

