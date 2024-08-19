#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel() {
  printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
  kernel<<<2, 4>>>();
  cudaDeviceSynchronize();
  return 0;
}