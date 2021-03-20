
#include<iostream>
#include<vector>
#include <algorithm> //for generate

using namespace std;


// Forward declaration of the driver Function in .cu file
void MatMul_driver(const vector<int>& h_a, const vector<int>& h_b, vector<int>& h_c, size_t bytes, int no_elements);


int main(int argc, char** argv) {

	int N = 1 << 10;

	size_t bytes = N * N * sizeof(int);

	//Host Matrix
	vector<int> h_a(N * N);
	vector<int> h_b(N * N);
	vector<int> h_c(N * N);

	// Initialize the Matrix with random values
	generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
	generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

	// Matrix Multiplication driver function.
	MatMul_driver(h_a, h_b, h_c, bytes, N);

}





