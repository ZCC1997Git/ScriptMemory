#pragma once
#include <cuda_runtime.h>
#include <cstddef>

inline constexpr int MaxSharedMemory = 49152;
inline constexpr int MaxCacheNum = 6;

/*struc for cache infor*/
struct CacheInfo {
  int global_addr = -1;
  void* cache_ptr = nullptr;
  int size; /*in byte*/
  int flag = -1;
};

/**
 * @file ScriptMemory.hpp
 * @brief ScriptMemory class: memory management for scripts memory(shared
 * mempory in CUDA, ldm in SW)
 * @date 2020-11-24
 */
class ScriptMemory {
 private:
  int size = 0;         /*the memory size in byte*/
  void* data = nullptr; /*the pointer that binded to a specific script memory*/
  int num_array_cache =
      0; /*the number of data in global memory cached in script memory*/
  CacheInfo array_cache[MaxCacheNum];

  __device__ ScriptMemory() {
    extern __shared__ char shared_memory[];
    this->data = shared_memory;
  }

 public:
  __device__ ScriptMemory(const ScriptMemory& other) {
    this->data = other.data;
    num_array_cache = 0;
    size = 0;
  }

  __device__ static ScriptMemory& MallocInstance() {
    // using shared memory to store the instance
    extern __shared__ ScriptMemory instance[];
    if (threadIdx.x == 0) {
      auto a = new ScriptMemory();
      instance[0] = *a;
    }
    __syncthreads();
    return instance[0];
  }

  __device__ ~ScriptMemory() { ; }

  /*
   * @brief: allocate a cache memory for the global data
   * @param: global_data: the pointer to the global data
   * @param: size: the size in the cache memory
   * @return: the pointer to the cache memory
   */
  template <class T>
  __device__ T* MallocCache(T* global_data, int size_cache) {
    int addr_low =
        static_cast<int>(reinterpret_cast<uintptr_t>(global_data) & 0xFFFF);
    /*if globda_data has been cached return T* directly*/
    for (int i = 0; i < num_array_cache; i++) {
      if (array_cache[i].global_addr == addr_low) {
        return reinterpret_cast<T*>(array_cache[i].cache_ptr);
      }
    }

    /*if the cache memory is not enough, return nullptr*/
    if (size + size_cache * sizeof(T) > MaxSharedMemory) {
      return nullptr;
    }

    if (num_array_cache >= MaxCacheNum) {
      return nullptr;
    }

    /*allocate the cache memory*/
    T* cache_data = reinterpret_cast<T*>((char*)data + size);
    array_cache[num_array_cache].global_addr = addr_low;
    array_cache[num_array_cache].cache_ptr = cache_data;
    array_cache[num_array_cache].size = size_cache * sizeof(T);
    num_array_cache++;
    size += size_cache * sizeof(T);
    return cache_data;
  }

  /*
   * @brief: allocate a cache memory that decoupled with global data
   * @param: size: the size in the cache memory
   * @return: the pointer to the cache memory
   */
  template <class T>
  __device__ T* MallocCache(int size_cache) {
    /*if globda_data has been cached return T* directly*/
    for (int i = 0; i < num_array_cache; i++) {
      if (array_cache[i].global_addr == 0) {
        return reinterpret_cast<T*>(array_cache[i].cache_ptr);
      }
    }

    /*if the cache memory is not enough, return nullptr*/
    if (size + size_cache * sizeof(T) > MaxSharedMemory) {
      return nullptr;
    }

    if (num_array_cache >= MaxCacheNum) {
      return nullptr;
    }

    /*allocate the cache memory*/
    T* cache_data = reinterpret_cast<T*>((char*)data + size);
    array_cache[num_array_cache].global_addr = (int)0;
    array_cache[num_array_cache].cache_ptr = cache_data;
    array_cache[num_array_cache].size = size_cache * sizeof(T);
    num_array_cache++;
    size += size_cache * sizeof(T);
    return cache_data;
  }

  template <class T>
  __device__ void CacheWrite(T* cache_ptr, auto index, auto data, int flag) {
    int i = 0;
    for (; i < num_array_cache; i++) {
      if (array_cache[i].cache_ptr == cache_ptr) {
        if (array_cache[i].flag == flag) {
          return;
        } else {
          break;
        }
      }
    }
    cache_ptr[index] = data;
    array_cache[i].flag = flag;
  }

  __device__ void CacheSync() { __syncthreads(); }
};
