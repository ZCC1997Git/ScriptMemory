#pragma once
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>

/*the max size of shared memory in CUDA*/
inline constexpr int MaxSharedMemory = 49152;
/*the max size of cache memory in CPU and CUDA*/
inline constexpr int MaxCacheMemory = 16 * 1024;
/*the max size of cache memory in CPU and CUDA*/
inline constexpr int MaxLDMMemory = 8 * 1024;

/*the max number of cache in CPU, CUDA and SWUC*/
inline constexpr int MaxCacheNum = 6;

/*struc for cache infor*/
struct CacheInfo {
  int global_addr = -1;
  void* cache_ptr = nullptr;
  int size; /*in byte*/
  int flag = -1;
};

/*the device support*/
enum class DEVICE { CPU, CUDA, SW };

/**
 * @file ScriptMemory.hpp
 * @brief ScriptMemory class: memory management for scripts memory(shared
 * mempory in CUDA, ldm in SW)
 * @date 2020-11-24
 */
template <DEVICE device>
class ScriptMemory {};

#ifdef __CUDACC__
/*
 * @brief: the specialization of ScriptMemory for CUDA
 */
template <>
class ScriptMemory<DEVICE::CUDA> {
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
    /*using shared memory to store the instance*/
    extern __shared__ ScriptMemory instance[];
    if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
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
    if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
      array_cache[num_array_cache].global_addr = addr_low;
      array_cache[num_array_cache].cache_ptr = cache_data;
      array_cache[num_array_cache].size = size_cache * sizeof(T);
      num_array_cache++;
      size += size_cache * sizeof(T);
    }
    __syncthreads();
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
    if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
      array_cache[num_array_cache].global_addr = (int)0;
      array_cache[num_array_cache].cache_ptr = cache_data;
      array_cache[num_array_cache].size = size_cache * sizeof(T);
      num_array_cache++;
      size += size_cache * sizeof(T);
    }
    return cache_data;
  }

  __device__ void CacheSync() { __syncthreads(); }

  /*
   * @brief: read data from global memory to cache memory
   * @param: cache_ptr: the pointer to the cache memory
   * @param: index_cache: the index in the cache memory
   * @param: global_addr: the pointer to the global memory
   * @param: index_global: the index in the global memory
   * @param: num_element: the number of element to read
   * @param: flag: the flag to indicate the data is read or write
   */
  template <class T>
  __device__ void CacheReadFromGlobal(T* cache_ptr,
                                      auto index_cache,
                                      T* global_addr,
                                      auto index_global,
                                      int num_element,
                                      int flag) {
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
    array_cache[i].flag = flag;
    for (int j = 0; j < num_element; j++) {
      cache_ptr[index_cache + j] = global_addr[index_global + j];
    }
  }

  template <class T>
  __device__ void CacheReadFromGlobalSync(T* cache_ptr,
                                          auto index_cache,
                                          T* global_addr,
                                          auto index_global,
                                          int num_element,
                                          int flag) {
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
    array_cache[i].flag = flag;
    for (int j = 0; j < num_element; j++) {
      cache_ptr[index_cache + j] = global_addr[index_global + j];
    }
    __syncthreads();
  }

  /*
   * @brief: write data from cache memory to global memory
   * @param: global_ptr: the pointer to the global memory
   * @param: index_global: the index in the global memory
   * @param: cache_ptr: the pointer to the cache memory
   * @param: index_cache: the index in the cache memory
   * @param: num_element: the number of element to write
   */
  template <class T>
  __device__ void CacheWriteToGlobal(T* global_ptr,
                                     auto index_global,
                                     T* cache_ptr,
                                     auto index_cache,
                                     int num_element) {
    for (int j = 0; j < num_element; j++) {
      global_ptr[index_global + j] = cache_ptr[index_cache + j];
    }
  }
};
#endif
/*
 * @brief: the specialization of ScriptMemory for CPU
 */
template <>
class ScriptMemory<DEVICE::CPU> {
 private:
  int size = 0;              /*the memory size in byte*/
  char data[MaxCacheMemory]; /*the array work as a script memory*/
  int num_array_cache =
      0; /*the number of data in global memory cached in script memory*/
  CacheInfo array_cache[MaxCacheNum];
  ScriptMemory() = default;

 public:
  /*allocate the instance in heap*/
  static auto MallocInstance() { return ScriptMemory(); }

  ~ScriptMemory() = default;

  /*
   * @brief: allocate a cache memory for the global data
   * @param: global_data: the pointer to the global data
   * @param: size: the size in the cache memory
   * @return: the pointer to the cache memory
   */
  template <class T>
  T* MallocCache(T* global_data, int size_cache) {
    int addr_low =
        static_cast<int>(reinterpret_cast<uintptr_t>(global_data) & 0xFFFF);
    /*if globda_data has been cached return T* directly*/
    for (int i = 0; i < num_array_cache; i++) {
      if (array_cache[i].global_addr == addr_low) {
        return reinterpret_cast<T*>(array_cache[i].cache_ptr);
      }
    }

    /*if the cache memory is not enough, return nullptr*/
    if (size + size_cache * sizeof(T) > MaxCacheMemory) {
      return nullptr;
    }

    if (num_array_cache >= MaxCacheNum) {
      return nullptr;
    }

    /*allocate the cache memory*/
    T* cache_data = reinterpret_cast<T*>(data + size);
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
  T* MallocCache(int size_cache) {
    /*if globda_data has been cached return T* directly*/
    for (int i = 0; i < num_array_cache; i++) {
      if (array_cache[i].global_addr == 0) {
        return reinterpret_cast<T*>(array_cache[i].cache_ptr);
      }
    }

    /*if the cache memory is not enough, return nullptr*/
    if (size + size_cache * sizeof(T) > MaxCacheMemory) {
      return nullptr;
    }

    if (num_array_cache >= MaxCacheNum) {
      return nullptr;
    }

    /*allocate the cache memory*/
    T* cache_data = reinterpret_cast<T*>(data + size);
    array_cache[num_array_cache].global_addr = 0;
    array_cache[num_array_cache].cache_ptr = cache_data;
    array_cache[num_array_cache].size = size_cache * sizeof(T);
    num_array_cache++;
    size += size_cache * sizeof(T);
    return cache_data;
  }

  void CacheSync() { ; }
  /*
   * @brief: read data from global memory to cache memory
   * @param: cache_ptr: the pointer to the cache memory
   * @param: index_cache: the index in the cache memory
   * @param: global_addr: the pointer to the global memory
   * @param: index_global: the index in the global memory
   * @param: num_element: the number of element to read
   * @param: flag: the flag to indicate the data is read or write
   */
  template <class T>
  void CacheReadFromGlobal(T* cache_ptr,
                           auto index_cache,
                           T* global_addr,
                           auto index_global,
                           int num_element,
                           int flag) {
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
    array_cache[i].flag = flag;
    for (int j = 0; j < num_element; j++) {
      cache_ptr[index_cache + j] = global_addr[index_global + j];
    }
  }

  template <class T>
  void CacheReadFromGlobalSync(T* cache_ptr,
                               auto index_cache,
                               T* global_addr,
                               auto index_global,
                               int num_element,
                               int flag) {
    CacheReadFromGlobal(cache_ptr, index_cache, global_addr, index_global,
                        num_element, flag);
  }

  /*
   * @brief: write data from cache memory to global memory
   * @param: global_ptr: the pointer to the global memory
   * @param: index_global: the index in the global memory
   * @param: cache_ptr: the pointer to the cache memory
   * @param: index_cache: the index in the cache memory
   * @param: num_element: the number of element to write
   */
  template <class T>
  void CacheWriteToGlobal(T* global_ptr,
                          auto index_global,
                          T* cache_ptr,
                          auto index_cache,
                          int num_element) {
    for (int j = 0; j < num_element; j++) {
      global_ptr[index_global + j] = cache_ptr[index_cache + j];
    }
  }
};