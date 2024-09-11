#pragma once
#pragma swuc push hostslave
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>

/*the max size of shared memory in CUDA (size in byte)*/
inline constexpr int MaxSharedMemory = 49152;
/*the max size of cache memory in CPU (size in byte)*/
inline constexpr int MaxCacheMemory = 6 * 8 * 1024;
/*the max size of cache memory in SW (size in byte)*/
inline constexpr int MaxLDMMemory = 6 * 8 * 1024;
/*the max number of cache element in CPU, CUDA and SWUC*/
inline constexpr int MaxCacheNum = 6;

/*struc for cache infor*/
struct CacheInfo {
  int flag =
      std::numeric_limits<int>::min(); /*the flag is the same, means that the
                                          data has been in the cache*/
  int usable_size = 0;                 /*in byte*/
  int used_size = 0;                   /*in byte*/
  void* global_ptr = nullptr;
  void* cache_ptr = nullptr;
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

 public:
  __device__ void clear() {
    size = 0;
    num_array_cache = 0;
  }
  __device__ void*& get_data() { return data; }
  /*
   * @brief: allocate a cache memory for the global data
   * @param: global_data: the pointer to the global data
   * @param: size: the size in the cache memory
   * @return: the pointer to the cache memory
   */
  template <class T>
  __device__ T* MallocCache(T* global_data, int size_cache) {
    /*if globda_data has been cached return T* directly*/
    for (int i = 0; i < num_array_cache; i++) {
      if (array_cache[i].global_ptr == global_data &&
          array_cache[i].usable_size == size_cache) {
        return reinterpret_cast<T*>(array_cache[i].cache_ptr);
      }
    }

    /*if the cache memory is not enough, return nullptr*/
    if (size + size_cache > MaxSharedMemory) {
      return nullptr;
    }

    if (num_array_cache >= MaxCacheNum) {
      return nullptr;
    }

    /*allocate the cache memory*/
    T* cache_data = reinterpret_cast<T*>((char*)data + size);
    if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
      array_cache[num_array_cache].global_ptr = global_data;
      array_cache[num_array_cache].usable_size = size_cache;
      array_cache[num_array_cache].cache_ptr = cache_data;
      /*don't need in CUDA*/
      // array_cache[num_array_cache].global_addr = addr_low;
      // array_cache[num_array_cache].used_size = 0;
      // array_cache[num_array_cache].flag = std::numeric_limits<int>::min();
      num_array_cache++;
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
      if (array_cache[i].global_ptr == nullptr &&
          array_cache[i].usable_size <= size_cache) {
        return reinterpret_cast<T*>(array_cache[i].cache_ptr);
      }
    }

    /*if the cache memory is not enough, return nullptr*/
    if (size + size_cache > MaxSharedMemory) {
      return nullptr;
    }

    if (num_array_cache >= MaxCacheNum) {
      return nullptr;
    }

    /*allocate the cache memory*/
    T* cache_data = reinterpret_cast<T*>((char*)data + size);
    if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {
      array_cache[num_array_cache].global_ptr = nullptr;
      array_cache[num_array_cache].usable_size = size_cache;
      array_cache[num_array_cache].cache_ptr = cache_data;
      /*don't need in CUDA*/
      // array_cache[num_array_cache].global_ptr = nullptr;
      // array_cache[num_array_cache].flag = std::numeric_limits<int>::min();
      // array_cache[num_array_cache].used_size = 0;
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
        return;
        // if (array_cache[i].flag == flag) {
        //   return;
        // } else {
        //   break;
        // }
      }
    }
    // array_cache[i].flag = flag;
    for (int j = 0; j < num_element; j++) {
      cache_ptr[index_cache + j] = global_addr[index_global + j];
    }
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
    /*if globda_data has been cached return T* directly*/
    for (int i = 0; i < num_array_cache; i++) {
      if (array_cache[i].global_ptr == global_data &&
          array_cache[i].usable_size == size_cache) {
        return reinterpret_cast<T*>(array_cache[i].cache_ptr);
      }
    }

    /*if the cache memory is not enough, return nullptr*/
    if (size + size_cache > MaxCacheMemory) {
      return nullptr;
    }

    if (num_array_cache >= MaxCacheNum) {
      return nullptr;
    }

    /*allocate the cache memory*/
    T* cache_data = reinterpret_cast<T*>(data + size);
    array_cache[num_array_cache].cache_ptr = cache_data;
    array_cache[num_array_cache].usable_size = size_cache;
    array_cache[num_array_cache].used_size = 0;
    array_cache[num_array_cache].global_ptr = global_data;
    array_cache[num_array_cache].flag = std::numeric_limits<int>::min();
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
      if (array_cache[i].global_ptr == nullptr &&
          array_cache[i].usable_size <= size_cache) {
        return reinterpret_cast<T*>(array_cache[i].cache_ptr);
      }
    }

    /*if the cache memory is not enough, return nullptr*/
    if (size + size_cache > MaxCacheMemory) {
      return nullptr;
    }

    if (num_array_cache >= MaxCacheNum) {
      return nullptr;
    }

    /*allocate the cache memory*/
    T* cache_data = reinterpret_cast<T*>(data + size);
    array_cache[num_array_cache].cache_ptr = cache_data;
    array_cache[num_array_cache].usable_size = size_cache;
    array_cache[num_array_cache].used_size = 0;
    array_cache[num_array_cache].global_ptr = nullptr;
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
                           int index_cache,
                           T* global_addr,
                           int index_global,
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
                          int index_global,
                          T* cache_ptr,
                          int index_cache,
                          int num_element) {
    for (int j = 0; j < num_element; j++) {
      global_ptr[index_global + j] = cache_ptr[index_cache + j];
    }
  }
};

#ifdef __SWCC__
/*
 * @brief: the specialization of ScriptMemory for SW
 */
template <>
class ScriptMemory<DEVICE::SW> {
 private:
  int size = 0; /*the occupied memory size in byte*/
  int num_array_cache =
      0; /*the number of data in global memory cached in script memory*/
  CacheInfo array_cache[MaxCacheNum];
  char* data = nullptr;

 public:
  char*& get_data() { return data; }
  /*
   * @brief: allocate a cache memory for the global data
   * @param: global_data: the pointer to the global data
   * @param: size: the size in the cache memory
   * @is_ensure: the flag to indicate the user is sure the data has bee
   cached
   * @return: the pointer to the cache memory
   */
  template <class T>
  T* MallocCache(T* global_data, int size_cache) {
    /*if globda_data has been cached return T* directly*/
    for (int i = 0; i < num_array_cache; i++) {
      if (array_cache[i].global_ptr == global_data &&
          array_cache[i].usable_size == size_cache) {
        return reinterpret_cast<T*>(array_cache[i].cache_ptr);
      }
    }

    /*if the cache memory is not enough, return nullptr*/
    if (size + size_cache > MaxCacheMemory) {
      return nullptr;
    }

    if (num_array_cache >= MaxCacheNum) {
      return nullptr;
    }

    /*allocate the cache memory*/
    T* cache_data = reinterpret_cast<T*>(data + size);
    array_cache[num_array_cache].usable_size = size_cache;
    array_cache[num_array_cache].used_size = 0;
    array_cache[num_array_cache].cache_ptr = cache_data;
    array_cache[num_array_cache].global_ptr = global_data;
    array_cache[num_array_cache].flag = std::numeric_limits<int>::min();
    num_array_cache++;
    size += size_cache;
    /*make the data is 64 aligned*/
    size = (size + 63) & ~63;
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
      if (array_cache[i].global_ptr == nullptr &&
          array_cache[i].usable_size <= size_cache) {
        return reinterpret_cast<T*>(array_cache[i].cache_ptr);
      }
    }

    /*if the cache memory is not enough, return nullptr*/
    if (size + size_cache > MaxCacheMemory) {
      return nullptr;
    }

    if (num_array_cache >= MaxCacheNum) {
      return nullptr;
    }

    /*allocate the cache memory*/
    T* cache_data = reinterpret_cast<T*>(data + size);
    array_cache[num_array_cache].usable_size = size_cache;
    array_cache[num_array_cache].used_size = 0;
    array_cache[num_array_cache].cache_ptr = cache_data;
    array_cache[num_array_cache].global_ptr = nullptr;
    num_array_cache++;
    size += size_cache;
    /*make the data is 64 aligned*/
    size = (size + 63) & ~63;
    return cache_data;
  }

  void CacheSync() { CRTS_ssync_array(); }
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
                           int index_cache,
                           T* global_addr,
                           int index_global,
                           int num_element,
                           int flag) {
    int index = 0;
    for (int i = 0; i < num_array_cache; i++) {
      if (array_cache[i].cache_ptr == cache_ptr) {
        index = i;
        if (array_cache[i].flag == flag) {
          return;
        }
      }
    }
    array_cache[index].flag = flag;
    array_cache[index].used_size = num_element * sizeof(T);
    CRTS_dma_get(cache_ptr + index_cache, global_addr + index_global,
                 num_element * sizeof(T));
  }

  template <class T>
  void CacheReadFromGlobal_with_writeback(T* cache_ptr,
                                          int index_cache,
                                          T* global_addr,
                                          int index_global,
                                          int num_element,
                                          int flag) {
    int index = 0;
    for (int i = 0; i < num_array_cache; i++) {
      if (array_cache[i].cache_ptr == cache_ptr) {
        index = i;
        if (array_cache[i].flag == flag) {
          return;
        }
      }
    }

    /*write back the data*/
    if (array_cache[index].flag != std::numeric_limits<int>::min()) {
      CRTS_dma_put(array_cache[index].global_ptr, array_cache[index].cache_ptr,
                   array_cache[index].used_size);
      CRTS_ssync_array();
    }

    array_cache[index].flag = flag;
    array_cache[index].used_size = num_element * sizeof(T);
    CRTS_dma_get(cache_ptr + index_cache, global_addr + index_global,
                 num_element * sizeof(T));
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
                          int index_global,
                          T* cache_ptr,
                          int index_cache,
                          int num_element) {
    CRTS_dma_put(global_ptr + index_global, cache_ptr + index_cache,
                 num_element * sizeof(T));
    CRTS_ssync_array();
  }
};
#endif
#pragma swuc pop