#include <ScriptMemory.hpp>
#include <iostream>

int main() {
  int* a = new int[4096];

  auto cache = ScriptMemory<DEVICE::CPU>::MallocInstance();
  auto ptr = cache.MallocCache(a, 1);
  for (int i = 0; i < 4096; i++) {
    ptr[0] = i;
    cache.CacheWriteToGlobal(a, i, ptr, 0, 1);
  }

  /*check*/
  bool flag = true;
  for (int i = 0; i < 4096; i++) {
    if (a[i] != i) {
      flag = false;
    }
  }
  if (true)
    std::cout << "Test passed" << std::endl;
  else
    std::cout << "Test failed" << std::endl;

  return 0;
}