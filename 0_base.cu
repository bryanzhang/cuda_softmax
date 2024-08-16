#include <cstdio>

#include <chrono>
#include <iostream>

constexpr int BATCH_SIZE = 1024;
constexpr int N = 8192;
constexpr int BLOCK_SIZE = 1024;
constexpr int WARP_SIZE = 32;

using namespace std;

__device__ __forceinline__ float warpReduceMax(float myMax) {
        myMax = max(__shfl_xor_sync(0xffffffff, myMax, 16), myMax);
        myMax = max(__shfl_xor_sync(0xffffffff, myMax, 8), myMax);
        myMax = max(__shfl_xor_sync(0xffffffff, myMax, 4), myMax);
        myMax = max(__shfl_xor_sync(0xffffffff, myMax, 2), myMax);
        myMax = max(__shfl_xor_sync(0xffffffff, myMax, 1), myMax);
        return myMax;
}

__device__ __forceinline__ float warpReduceSum(float mySum) {
        mySum += __shfl_xor_sync(0xffffffff, mySum, 16);
        mySum += __shfl_xor_sync(0xffffffff, mySum, 8);
        mySum += __shfl_xor_sync(0xffffffff, mySum, 4);
        mySum += __shfl_xor_sync(0xffffffff, mySum, 2);
        mySum += __shfl_xor_sync(0xffffffff, mySum, 1);
        return mySum;
}

__device__ __forceinline__ float reduce_max(float* arr, int tid) {
        __shared__ int smm[WARP_SIZE];

        float a0 = arr[tid];
        float a1 = arr[tid + BLOCK_SIZE];
        float a2 = arr[tid + 2 * BLOCK_SIZE];
        float a3 = arr[tid + 3 * BLOCK_SIZE];
        float a4 = arr[tid + 4 * BLOCK_SIZE];
        float a5 = arr[tid + 5 * BLOCK_SIZE];
        float a6 = arr[tid + 6 * BLOCK_SIZE];
        float a7 = arr[tid + 7 * BLOCK_SIZE];
        float myMax = max(max(max(max(max(max(max(a0, a1), a2), a3), a4), a5), a6), a7);
        myMax = warpReduceMax(myMax);
        int laneIdx = (tid % WARP_SIZE);
        int warpIdx = tid / WARP_SIZE;
        if (laneIdx == 0) {
                smm[warpIdx] = myMax;
        }
        __syncthreads();

        if (warpIdx == 0) {
                myMax = smm[laneIdx];
                myMax = warpReduceMax(myMax);
                if (laneIdx == 0) {
                        smm[0] = myMax;
                }
        }
        __syncthreads();
        return smm[0];
}

__device__ __forceinline__ float reduce_sum(float* arr, int tid) {
        __shared__ int smm[WARP_SIZE];

        float a0 = arr[tid];
        float a1 = arr[tid + BLOCK_SIZE];
        float a2 = arr[tid + 2 * BLOCK_SIZE];
        float a3 = arr[tid + 3 * BLOCK_SIZE];
        float a4 = arr[tid + 4 * BLOCK_SIZE];
        float a5 = arr[tid + 5 * BLOCK_SIZE];
        float a6 = arr[tid + 6 * BLOCK_SIZE];
        float a7 = arr[tid + 7 * BLOCK_SIZE];
        float mySum = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
        mySum = warpReduceSum(mySum);
        int laneIdx = (tid % WARP_SIZE);
        int warpIdx = tid / WARP_SIZE;
        if (laneIdx == 0) {
                smm[warpIdx] = mySum;
        }
        __syncthreads();

        if (warpIdx == 0) {
                mySum = smm[laneIdx];
                mySum = warpReduceSum(mySum);
                if (laneIdx == 0) {
                        smm[0] = mySum;
                }
        }
        __syncthreads();
        return smm[0];
}

__global__ void softmax(float* out, float* in) {
        float max = reduce_max(in + blockIdx.x * N, threadIdx.x);
        out[blockIdx.x * N + threadIdx.x] = exp(in[blockIdx.x * N + threadIdx.x] - max);
        out[blockIdx.x * N + BLOCK_SIZE + threadIdx.x] = exp(in[blockIdx.x * N + BLOCK_SIZE + threadIdx.x] - max);
        out[blockIdx.x * N + 2 * BLOCK_SIZE + threadIdx.x] = exp(in[blockIdx.x * N + 2 * BLOCK_SIZE + threadIdx.x] - max);
        out[blockIdx.x * N + 3 * BLOCK_SIZE + threadIdx.x] = exp(in[blockIdx.x * N + 3 * BLOCK_SIZE + threadIdx.x] - max);
        out[blockIdx.x * N + 4 * BLOCK_SIZE + threadIdx.x] = exp(in[blockIdx.x * N + 4 * BLOCK_SIZE + threadIdx.x] - max);
        out[blockIdx.x * N + 5 * BLOCK_SIZE + threadIdx.x] = exp(in[blockIdx.x * N + 5 * BLOCK_SIZE + threadIdx.x] - max);
        out[blockIdx.x * N + 6 * BLOCK_SIZE + threadIdx.x] = exp(in[blockIdx.x * N + 6 * BLOCK_SIZE + threadIdx.x] - max);
        out[blockIdx.x * N + 7 * BLOCK_SIZE + threadIdx.x] = exp(in[blockIdx.x * N + 7 * BLOCK_SIZE + threadIdx.x] - max);
        __syncthreads();
        float sum = reduce_sum(out + blockIdx.x * N, threadIdx.x);
        out[blockIdx.x * N + threadIdx.x] /= sum;
        out[blockIdx.x * N + BLOCK_SIZE + threadIdx.x] /= sum;
        out[blockIdx.x * N + 2 * BLOCK_SIZE + threadIdx.x] /= sum;
        out[blockIdx.x * N + 3 * BLOCK_SIZE + threadIdx.x] /= sum;
        out[blockIdx.x * N + 4 * BLOCK_SIZE + threadIdx.x] /= sum;
        out[blockIdx.x * N + 5 * BLOCK_SIZE + threadIdx.x] /= sum;
        out[blockIdx.x * N + 6 * BLOCK_SIZE + threadIdx.x] /= sum;
        out[blockIdx.x * N + 7 * BLOCK_SIZE + threadIdx.x] /= sum;
}

double test(float* d_in, float* d_out) {
        cudaMemset(d_out, 0, sizeof(float) * N * BATCH_SIZE);
        dim3 block(BLOCK_SIZE);
        dim3 grid(BATCH_SIZE);
        auto start = chrono::high_resolution_clock::now();
        softmax<<<grid, block>>>(d_out, d_in);
        cudaDeviceSynchronize();
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        return elapsed.count();
}

int main() {
        srand(3);
        float* h_in = new float[BATCH_SIZE * N];
        for (int i = 0; i < BATCH_SIZE * N; ++i) {
                h_in[i] = (float)rand() / RAND_MAX * 2000 - 1000;
        }
        // float* h_out = new float[BATCH_SIZE * N];

        float* d_in = nullptr;
        cudaMalloc(&d_in, BATCH_SIZE * N * sizeof(float));
        cudaMemcpy(d_in, h_in, BATCH_SIZE * N * sizeof(float), cudaMemcpyHostToDevice);
        float* d_out = nullptr;
        cudaMalloc(&d_out, BATCH_SIZE * N * sizeof(float));

        double elapsed = 0.0;
        constexpr int tries = 50;
        for (int i = 0; i < tries; ++i) {
                elapsed += test(d_in, d_out);
        }
        cout << "Tries=" << tries << ", average elapsed time: " << elapsed * 1000 / tries << " ms." << endl;

        delete[] h_in;
        // delete[] h_out;
        cudaFree(d_in);
        cudaFree(d_out);
        return 0;
}
