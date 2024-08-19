#include <cuda_runtime.h>
#include <ScriptMemory.hpp>
#include <iostream>

__device__ void wirte_sm(int* A, ScriptMemory& sm) {
  auto ptr = sm.MallocCache<int>(A, 32);
  sm.CacheWrite(ptr, threadIdx.x, threadIdx.x, 0);
}

__global__ void kernel(int* A) {
  auto sm = ScriptMemory::MallocInstance();
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  A[tid] = threadIdx.x;
  auto ptr = sm.MallocCache<int>(A, 32);
  sm.CacheWrite(ptr, threadIdx.x, threadIdx.x, 0);
  sm.CacheSync();
  wirte_sm(A, sm);
  int sum = 0;
  for (int i = 0; i < 32; i++) {
    sum += ptr[i];
  }
  A[tid] = sum;
}

int main() {
  int* d_A;
  cudaMalloc(&d_A, 1024 * sizeof(int));
  cudaMemset(d_A, 0, 1024 * sizeof(int));

  dim3 grid(32, 1, 1);
  dim3 block(32, 1, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  kernel<<<grid, block, 8 * 1024>>>(d_A);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Time: " << milliseconds << "ms" << std::endl;
  cudaDeviceSynchronize();
  int* h_A = new int[1024];
  cudaMemcpy(h_A, d_A, 1024 * sizeof(int), cudaMemcpyDeviceToHost);
  /*check*/
  bool flag = true;
  for (int i = 0; i < 1024; i++) {
    if (h_A[i] != 496) {
      flag = false;
      std::cout << "Error: " << i << " " << h_A[i] << std::endl;
    }
  }
  if (flag) {
    std::cout << "Success" << std::endl;
  }
  return 0;
}