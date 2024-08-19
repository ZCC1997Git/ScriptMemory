#include <cuda_runtime.h>
#include <ScriptMemory.hpp>
#include <iostream>

__global__ void kernel() {
  auto sm = ScriptMemory::MallocInstance();

  auto ptr = sm.MallocCache<int>(32);
  ptr[threadIdx.x] = threadIdx.x + 1;
  auto ptr2 = sm.MallocCache<int>(32);
  printf("%p %p\n", &sm, ptr2);
  ptr[threadIdx.x] = threadIdx.x + 2;

  __syncthreads();
  printf("threadIdx.x: %d, ptr[threadIdx.x]: %d\n", threadIdx.x,
         ptr[threadIdx.x]);
}

int main() {
  //   kernel<<<2, 4>>>();
  dim3 grid(1, 1, 1);
  dim3 block(32, 1, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  kernel<<<grid, block, 8 * 1024>>>();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Time: " << milliseconds << "ms" << std::endl;
  cudaDeviceSynchronize();
  return 0;
}