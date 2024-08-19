#pragma once
#include <cuda_runtime.h>
#include <cstddef>

/*struc for cache infor*/
struct CacheInfo {
  int global_addr = -1;
  int offset;
  int size;
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
  CacheInfo array_cache[6];

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

  template <class T>
  __device__ T* get_data() {
    return static_cast<T*>(ScriptMemory::data);
  }

  __device__ ~ScriptMemory() { ; }

  /*
   * @brief: allocate a cache memory for the global data
   * @param: global_data: the pointer to the global data
   * @param: size: the size in the cache memory
   * @return: the pointer to the cache memory
   */
  template <class T, bool IsAligned = false>
  __device__ T* MallocCache(T* global_data, int size_cache) {
    /*if IsAligned is true, the cache should be aligned with 32 in CUDA*/
    if constexpr (IsAligned) {
      size_cache = (size_cache + 31) & ~31;
    }

    /*if globda_data has been cached return T* directly*/
    for (int i = 0; i < num_array_cache; i++) {
      if (array_cache[i].global_addr == (int)global_data) {
        return static_cast<T*>((char*)data + array_cache[i].offset);
      }
    }

    /*if the cache memory is not enough, return nullptr*/
    if (size + size_cache * sizeof(T) > 49152) {
      return nullptr;
    }

    if (num_array_cache >= 6) {
      return nullptr;
    }

    /*allocate the cache memory*/
    T* cache_data = static_cast<T*>((char*)data + size);
    array_cache[num_array_cache].global_addr = (int)global_data;
    array_cache[num_array_cache].offset = size;
    array_cache[num_array_cache].size = size_cache * sizeof(T);
    num_array_cache++;
    size += size_cache * sizeof(T);
  }

  template <class T, bool IsAligned = false>
  __device__ T* MallocCache(int size_cache) {
    /*if IsAligned is true, the cache should be aligned with 32 in CUDA*/
    if constexpr (IsAligned) {
      size_cache = (size_cache + 31) & ~31;
    }

    /*if globda_data has been cached return T* directly*/
    for (int i = 0; i < num_array_cache; i++) {
      if (array_cache[i].global_addr == 0) {
        return reinterpret_cast<T*>(static_cast<char*>(data) +
                                    array_cache[i].offset);
      }
    }

    /*if the cache memory is not enough, return nullptr*/
    if (size + size_cache * sizeof(T) > 49152) {
      return nullptr;
    }

    if (num_array_cache >= 6) {
      return nullptr;
    }

    /*allocate the cache memory*/
    T* cache_data = reinterpret_cast<T*>((char*)data + size);
    array_cache[num_array_cache].global_addr = (int)0;
    array_cache[num_array_cache].offset = size;
    array_cache[num_array_cache].size = size_cache * sizeof(T);
    num_array_cache++;
    size += size_cache * sizeof(T);
    return cache_data;
  }
};
