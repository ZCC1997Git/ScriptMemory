#include <cuda_runtime.h>
#include <ScriptMemory.hpp>
#include <iostream>

// __device__ void wirte_sm(int* A, ScriptMemory& sm) {
//   auto ptr = sm.MallocCache<int>(A, 32);
//   sm.CacheReadFromGlobal(ptr, threadIdx.x, threadIdx.x, 0);
// }

__global__ void kernel(int* A) {
  auto sm = ScriptMemory<DEVICE::CUDA>::MallocInstance();
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  A[tid] = threadIdx.x % 32;
  auto ptr = sm.MallocCache<int>(A, 64);
  sm.CacheReadFromGlobalSync(ptr, threadIdx.x, A, tid, 1, 0);
  int sum = 0;
  for (int i = 0; i < 64; i++) {
    sum += ptr[i];
  }
  A[tid] = sum;
}

int main() {
  int* d_A;
  cudaMalloc(&d_A, 4096 * sizeof(int));
  cudaMemset(d_A, 0, 4096 * sizeof(int));

  dim3 grid(64, 1, 1);
  dim3 block(64, 1, 1);

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
  int* h_A = new int[4096];
  cudaMemcpy(h_A, d_A, 4096 * sizeof(int), cudaMemcpyDeviceToHost);
  /*check*/
  bool flag = true;
  for (int i = 0; i < 4096; i++) {
    if (h_A[i] != 496 * 2) {
      flag = false;
      std::cout << "Error: " << i << " " << h_A[i] << std::endl;
    }
  }
  if (flag) {
    std::cout << "Success" << std::endl;
  }
  return 0;
}