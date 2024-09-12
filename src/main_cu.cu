#include <cuda_runtime.h>
#include <ScriptMemory.hpp>
#include <iostream>

__global__ void kernel_ref(int* A) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  A[tid] = threadIdx.x % 32;
  __shared__ int ptr[64];
  ptr[threadIdx.x] = A[tid];
  __syncthreads();
  int sum = 0;
  for (int i = 0; i < 64; i++) {
    sum += ptr[i];
  }
  A[tid] = sum;
}

__global__ void kernel(int* A) {
  auto& sm = MallocInstance();
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  A[tid] = threadIdx.x % 32;
  auto ptr = sm.MallocCache<int>(A, 64);
  sm.CacheReadFromGlobal(ptr, threadIdx.x, A, tid, 1, 0);
  sm.CacheSync();

  ptr[threadIdx.x] = A[tid];
  int sum = 0;
  for (int i = 0; i < 64; i++) {
    sum += ptr[i];
  }
  A[tid] = sum;
  if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
    sm.clear();
  }
}

void ResultCheck(int* d_A);

int main() {
  int* d_A;
  cudaMalloc(&d_A, 4096 * 2 * sizeof(int));
  cudaMemset(d_A, 0, 4096 * 2 * sizeof(int));

  dim3 grid(128, 1, 1);
  dim3 block(64, 1, 1);

  kernel_ref<<<grid, block>>>(d_A);

  cudaMemset(d_A, 0, 4096 * 2 * sizeof(int));
  cudaEvent_t start_ref, stop_ref;
  cudaEventCreate(&start_ref);
  cudaEventCreate(&stop_ref);
  cudaEventRecord(start_ref);
  for (int i = 0; i < 10000; i++)
    kernel_ref<<<grid, block>>>(d_A);
  cudaEventRecord(stop_ref);
  cudaEventSynchronize(stop_ref);
  float milliseconds_ref = 0;
  cudaEventElapsedTime(&milliseconds_ref, start_ref, stop_ref);
  std::cout << "Time_ref: " << milliseconds_ref << "ms" << std::endl;
  /*check*/
  ResultCheck(d_A);

  kernel<<<grid, block, 128 * 8>>>(d_A);
  cudaMemset(d_A, 0, 4096 * 2 * sizeof(int));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < 10000; i++)
    kernel<<<grid, block, 128 * 8>>>(d_A);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Time: " << milliseconds << "ms" << std::endl;
  /*check*/
  ResultCheck(d_A);
  return 0;
}

void ResultCheck(int* d_A) {
  int* h_A = new int[4096 * 2];
  cudaMemcpy(h_A, d_A, 4096 * 2 * sizeof(int), cudaMemcpyDeviceToHost);
  /*check*/
  bool flag = true;
  for (int i = 0; i < 4096 * 2; i++) {
    if (h_A[i] != 496 * 2) {
      flag = false;
      std::cout << "Error: " << i << " " << h_A[i] << std::endl;
    }
  }
  if (flag) {
    std::cout << "Success" << std::endl;
  }
}