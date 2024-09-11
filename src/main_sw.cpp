#include <crts.h>
#include <ScriptMemory.hpp>
#include <chrono>
#include <iostream>

#if not defined(__sw_host__)
__thread ScriptMemory<DEVICE::SW> SM;
__attribute((slave)) auto& MallocInstance() {
  SM.get_data() = static_cast<char*>(ldm_malloc(MaxLDMMemory));
  return SM;
}
#else
ScriptMemory<DEVICE::SW> SM;
auto& MallocInstance() {
  return SM;
}
#endif

__attribute((kernel)) void test1(int* a) {
  // int b[1024];
  int* b = (int*)ldm_malloc(1024 * sizeof(int));
  auto tid = CRTS_tid;
  CRTS_dma_get(b, a + tid * 1024, 1024 * sizeof(int));
  CRTS_ssync_array();
  for (int i = 0; i < 1024; i++) {
    b[i] += i;
  }
  CRTS_dma_put(a + tid * 1024, b, 1024 * sizeof(int));
  CRTS_ssync_array();
  ldm_free(b, 1024 * sizeof(int));
}

__attribute((kernel)) void test2(int* a) {
  auto tid = CRTS_tid;
  auto sm = MallocInstance();
  auto aa = a + tid * 1024;

  auto b = sm.MallocCache(aa, 1024 * sizeof(int));
  sm.CacheReadFromGlobal(b, 0, a, tid * 1024, 1024, 0);
  sm.CacheSync();
  for (int i = 0; i < 1024; i++) {
    b[i] += i;
  }
  sm.CacheWriteToGlobal(a, tid * 1024, b, 0, 1024);
  sm.CacheSync();
}

#if not defined(__sw_slave__)
int main() {
  /*init*/
  CRTS_init();
  int a[1024 * 64];
  for (int i = 0; i < 1024 * 64; i++) {
    a[i] = 0;
  }

  auto start1 = std::chrono::high_resolution_clock::now();
  test1(a);
  CRTS_athread_join();
  auto end1 = std::chrono::high_resolution_clock::now();
  std::cout << "test1 time:"
            << std::chrono::duration_cast<std::chrono::microseconds>(end1 -
                                                                     start1)
                   .count()
            << std::endl;

  /*check*/
  bool flag = true;
  for (int i = 0; i < 1024 * 64; i++) {
    if (a[i] != i % 1024) {
      flag = false;
      std::cout << "a[" << i << "]=" << a[i] << std::endl;
      break;
    }
  }
  if (flag) {
    std::cout << "PASS" << std::endl;
  } else {
    std::cout << "FAIL" << std::endl;
  }

  for (int i = 0; i < 1024 * 64; i++) {
    a[i] = 0;
  }
  auto start2 = std::chrono::high_resolution_clock::now();
  test2(a);
  CRTS_athread_join();
  auto end2 = std::chrono::high_resolution_clock::now();
  std::cout << "test2 time:"
            << std::chrono::duration_cast<std::chrono::microseconds>(end2 -
                                                                     start2)
                   .count()
            << std::endl;

  CRTS_athread_halt();

  /*check*/
  flag = true;
  for (int i = 0; i < 1024 * 64; i++) {
    if (a[i] != i % 1024) {
      flag = false;
      std::cout << "a[" << i << "]=" << a[i] << std::endl;
      break;
    }
  }
  if (flag) {
    std::cout << "PASS" << std::endl;
  } else {
    std::cout << "FAIL" << std::endl;
  }
  return 0;
}
#endif