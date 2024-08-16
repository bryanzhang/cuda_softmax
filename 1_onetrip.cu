#include <cstdio>

#include <chrono>
#include <iostream>

constexpr int BATCH_SIZE = 1024;
constexpr int N = 8192;
constexpr int BLOCK_SIZE = 1024;
constexpr int WARP_SIZE = 32;

using namespace std;

struct __align__(8) ReduceResult {
        float max;  // 局部最大值
        float sum;  // 局部sum
};

__device__ __forceinline__ ReduceResult warpReduceMaxAndSum(float myMax, float mySum) {
        float originalMyMax = myMax;
        myMax = max(__shfl_xor_sync(0xffffffff, myMax, 16), myMax);
        myMax = max(__shfl_xor_sync(0xffffffff, myMax, 8), myMax);
        myMax = max(__shfl_xor_sync(0xffffffff, myMax, 4), myMax);
        myMax = max(__shfl_xor_sync(0xffffffff, myMax, 2), myMax);
        myMax = max(__shfl_xor_sync(0xffffffff, myMax, 1), myMax);

        float adjustedMySum = mySum * __expf(originalMyMax - myMax);
        adjustedMySum += __shfl_xor_sync(0xffffffff, adjustedMySum, 16);
        adjustedMySum += __shfl_xor_sync(0xffffffff, adjustedMySum, 8);
        adjustedMySum += __shfl_xor_sync(0xffffffff, adjustedMySum, 4);
        adjustedMySum += __shfl_xor_sync(0xffffffff, adjustedMySum, 2);
        adjustedMySum += __shfl_xor_sync(0xffffffff, adjustedMySum, 1);

        return { myMax, adjustedMySum };
}

__device__ __forceinline__ ReduceResult reduce_max_and_sum(float* arr, int tid) {
        __shared__ ReduceResult smm[WARP_SIZE];

        float a0 = arr[tid];
        float a1 = arr[tid + BLOCK_SIZE];
        float a2 = arr[tid + 2 * BLOCK_SIZE];
        float a3 = arr[tid + 3 * BLOCK_SIZE];
        float a4 = arr[tid + 4 * BLOCK_SIZE];
        float a5 = arr[tid + 5 * BLOCK_SIZE];
        float a6 = arr[tid + 6 * BLOCK_SIZE];
        float a7 = arr[tid + 7 * BLOCK_SIZE];
        float myMax = max(max(max(max(max(max(max(a0, a1), a2), a3), a4), a5), a6), a7);
        float mySum = __expf(a0 - myMax) + __expf(a1 - myMax) + __expf(a2 - myMax) + __expf(a3 - myMax) + __expf(a4 - myMax) + __expf(a5 - myMax) + __expf(a6 - myMax) + __expf(a7 - myMax);
        ReduceResult result = warpReduceMaxAndSum(myMax, mySum);
        int laneIdx = (tid % WARP_SIZE);
        int warpIdx = tid / WARP_SIZE;
        if (laneIdx == 0) {
                smm[warpIdx] = result;
        }
        __syncthreads();

        if (warpIdx == 0) {
                ReduceResult res = warpReduceMaxAndSum(smm[laneIdx].max, smm[laneIdx].sum);
                if (laneIdx == 0) {
                        smm[0] = res;
                }
        }
        __syncthreads();
        return smm[0];
}

__global__ void softmax(float* out, float* in) {
        ReduceResult res = reduce_max_and_sum(in + blockIdx.x * N, threadIdx.x);
        out[blockIdx.x * N + threadIdx.x] = __expf(in[blockIdx.x * N + threadIdx.x] - res.max) / res.sum;  // NOTE：重复计算
        out[blockIdx.x * N + BLOCK_SIZE + threadIdx.x] = __expf(in[blockIdx.x * N + BLOCK_SIZE + threadIdx.x] - res.max) / res.sum;  // NOTE：重复计算
        out[blockIdx.x * N + 2 * BLOCK_SIZE + threadIdx.x] = __expf(in[blockIdx.x * N + 2 * BLOCK_SIZE + threadIdx.x] - res.max) / res.sum;  // NOTE：重复计算
        out[blockIdx.x * N + 3 * BLOCK_SIZE + threadIdx.x] = __expf(in[blockIdx.x * N + 3 * BLOCK_SIZE + threadIdx.x] - res.max) / res.sum;  // NOTE：重复计算
        out[blockIdx.x * N + 4 * BLOCK_SIZE + threadIdx.x] = __expf(in[blockIdx.x * N + 4 * BLOCK_SIZE + threadIdx.x] - res.max) / res.sum;  // NOTE：重复计算
        out[blockIdx.x * N + 5 * BLOCK_SIZE + threadIdx.x] = __expf(in[blockIdx.x * N + 5 * BLOCK_SIZE + threadIdx.x] - res.max) / res.sum;  // NOTE：重复计算
        out[blockIdx.x * N + 6 * BLOCK_SIZE + threadIdx.x] = __expf(in[blockIdx.x * N + 6 * BLOCK_SIZE + threadIdx.x] - res.max) / res.sum;  // NOTE：重复计算
        out[blockIdx.x * N + 7 * BLOCK_SIZE + threadIdx.x] = __expf(in[blockIdx.x * N + 7 * BLOCK_SIZE + threadIdx.x] - res.max) / res.sum;  // NOTE：重复计算
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
