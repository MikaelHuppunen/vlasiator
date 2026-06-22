/*
 * This file is part of Vlasiator.
 * Copyright 2010-2025 Finnish Meteorological Institute and University of Helsinki
 *
 * For details of usage, see the COPYING file and read the "Rules of the Road"
 * at http://www.physics.helsinki.fi/vlasiator/
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef GPU_BASE_H
#define GPU_BASE_H

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "arch_device_api.h"

#include <stdio.h>
#include <mutex>
#include "include/splitvector/splitvec.h"
#include "include/hashinator/hashinator.h"
#include "../definitions.h"
#include "../vlasovsolver/vec.h"
#include "../velocity_mesh_parameters.h"
#include <phiprof.hpp>

#ifndef THREADS_PER_MP
#define THREADS_PER_MP 2048
#endif
#ifndef REGISTERS_PER_MP
#define REGISTERS_PER_MP 65536
#endif

// Device properties
extern int gpuMultiProcessorCount;
extern int blocksPerMP;
extern int threadsPerMP;

extern double gpuShrinkFactor;

// Magic multipliers used to make educated guesses for initial allocations
// and for managing dynamic increases in allocation sizes. Some of these are
// scaled based on WID value for better guesses,
static const uint VLASOV_BUFFER_MINBLOCKS = 32768/WID3;
static const uint VLASOV_BUFFER_MINCOLUMNS = 2000/WID;
static const uint INIT_VMESH_SIZE (32768/WID3);
static const uint INIT_MAP_SIZE (16 - WID);
static const double BLOCK_ALLOCATION_PADDING = 1.2;
static const double BLOCK_ALLOCATION_FACTOR = 1.1;

// Used in acceleration column construction. The flattened version of the
// probe cube must store (5) counters / offsets, see vlasovsolver/gpu_acc_map.cpp for details.
static const int GPU_PROBEFLAT_N = 5;

// buffers need to be larger for translation to allow proper parallelism
// GPUTODO: Get rid of this multiplier and consolidate buffer allocations.
// WARNING: Simply removing this factor led to diffs in Flowthrough_trans_periodic, indicating that
// there is somethign wrong with the evaluation of buffers! To be investigated.
static const int TRANSLATION_BUFFER_ALLOCATION_FACTOR = 5;

#define MAXCPUTHREADS 512 // hypothetical max size for some allocation arrays

void gpu_init_device();
void gpu_clear_device();
gpuStream_t gpu_getStream();
gpuStream_t gpu_getPriorityStream();
uint gpu_getThread();
uint gpu_getMaxThreads();
int gpu_getDevice();
uint gpu_getAllocationCount();
void gpu_setAllocationCount(uint value);
int gpu_reportMemory(const size_t local_cap=0, const size_t ghost_cap=0, const size_t local_size=0, const size_t ghost_size=0);

unsigned int nextPowerOfTwo(unsigned int n);

void gpu_vlasov_allocate();
void gpu_calculateProbeAllocation(uint maxBlockCount);
void gpu_vlasov_deallocate();
void gpu_vlasov_set_allocation_sizes(size_t blockCount);
void gpu_vlasov_set_allocation_size(size_t index, size_t size);
uint gpu_vlasov_getSmallestAllocation();

void gpu_batch_allocate(uint nCells=0, uint maxNeighbours=0);

void gpu_acc_allocate(uint maxBlockCount);
void gpu_acc_reallocate(uint maxBlockCount);
void gpu_acc_allocate_perthread(uint cpuThreadID, uint firstAllocationCount, uint columnSetAllocationCount=0);
void gpu_acc_deallocate();

void gpu_trans_allocate(cuint nAllCells=0,
                        cuint largestVmesh=0,
                        cuint unionSetSize=0);
void gpu_trans_deallocate();

extern gpuStream_t gpuStreamList[];
extern gpuStream_t gpuPriorityStreamList[];


/**
 * @brief Custom allocator for unified memory (GPU and CPU accessible).
 *
 * This class provides an allocator for unified memory, which can be accessed
 * by both the GPU and the CPU. It allocates and deallocates memory using split_gpuMallocManaged
 * and split_gpuFree functions, while also providing constructors and destructors for objects.
 *
 * @tparam T Type of the allocated objects.
 */
template <class T>
class splitGpuMemoryManagerallocator {
public:
   typedef T value_type;
   typedef value_type* pointer;
   typedef const value_type* const_pointer;
   typedef value_type& reference;
   typedef const value_type& const_reference;
   typedef ptrdiff_t difference_type;
   typedef size_t size_type;
   template <class U>
   struct rebind {
      typedef splitGpuMemoryManagerallocator<U> other;
   };
   /**
    * @brief Default constructor.
    */
   splitGpuMemoryManagerallocator() throw() {}

   /**
    * @brief Copy constructor with different type.
    */
   template <class U>
   splitGpuMemoryManagerallocator(splitGpuMemoryManagerallocator<U> const&) throw() {}
   pointer address(reference x) const { return &x; }
   const_pointer address(const_reference x) const { return &x; }

   pointer allocate(size_type n, const void* /*hint*/ = 0) {
      T* ret;
      assert(n && "allocate 0");
      SPLIT_CHECK_ERR(split_gpuMallocManaged((void**)&ret, n * sizeof(value_type)));
      gpuMemoryManager.addManagedMemory(n * sizeof(value_type));
      if (ret == nullptr) {
         throw std::bad_alloc();
      }
      return ret;
   }

   static void* allocate_raw(size_type n, const void* /*hint*/ = 0) {
      void* ret;
      SPLIT_CHECK_ERR(split_gpuMallocManaged((void**)&ret, n));
      gpuMemoryManager.addManagedMemory(n);
      if (ret == nullptr) {
         throw std::bad_alloc();
      }
      return ret;
   }

   void deallocate(pointer p, size_type n) {
      if (n != 0 && p != 0) {
         SPLIT_CHECK_ERR(split_gpuFree(p));
         gpuMemoryManager.subtractManagedMemory(n);
      }
   }
   static void deallocate(void* p, size_type n) {
      if (n != 0 && p != 0) {
         SPLIT_CHECK_ERR(split_gpuFree(p));
         gpuMemoryManager.subtractManagedMemory(n);
      }
   }

   size_type max_size() const throw() {
      size_type max = static_cast<size_type>(-1) / sizeof(value_type);
      return (max > 0 ? max : 1);
   }

   template <typename U, typename... Args>
   __host__ __device__ void construct(U* p, Args&&... args) {
      ::new (p) U(std::forward<Args>(args)...);
   }

   void destroy(pointer p) { p->~value_type(); }
};

// Struct used by Vlasov Acceleration semi-Lagrangian solver
struct ColumnOffsets {
   split::SplitVector<uint, splitGpuMemoryManagerallocator<uint>> setColumnOffsets; // index from columnBlockOffsets where new set of columns starts (length nColumnSets)
   split::SplitVector<uint, splitGpuMemoryManagerallocator<uint>> setNumColumns; // how many columns in set of columns (length nColumnSets)

   split::SplitVector<uint, splitGpuMemoryManagerallocator<uint>> columnBlockOffsets; // indexes where columns start (in blocks, length totalColumns)
   split::SplitVector<uint, splitGpuMemoryManagerallocator<uint>> columnNumBlocks; // length of column (in blocks, length totalColumns)
   split::SplitVector<int, splitGpuMemoryManagerallocator<int>> minBlockK,maxBlockK;
   split::SplitVector<int, splitGpuMemoryManagerallocator<int>> kBegin;
   split::SplitVector<int, splitGpuMemoryManagerallocator<int>> i,j;
   uint colSize = 0;
   uint colSetSize = 0;
   uint colCapacity = 0;
   uint colSetCapacity = 0;

   ColumnOffsets(uint nColumns=1, uint nColumnSets=1) {
      gpuStream_t stream = gpu_getStream();
      setColumnOffsets.resize(nColumnSets);
      setNumColumns.resize(nColumnSets);
      columnBlockOffsets.resize(nColumns);
      columnNumBlocks.resize(nColumns);
      minBlockK.resize(nColumns);
      maxBlockK.resize(nColumns);
      kBegin.resize(nColumns);
      i.resize(nColumns);
      j.resize(nColumns);
      // These vectors themselves are not in unified memory, just their content data
      setColumnOffsets.optimizeGPU(stream);
      setNumColumns.optimizeGPU(stream);
      columnBlockOffsets.optimizeGPU(stream);
      columnNumBlocks.optimizeGPU(stream);
      minBlockK.optimizeGPU(stream);
      maxBlockK.optimizeGPU(stream);
      kBegin.optimizeGPU(stream);
      i.optimizeGPU(stream);
      j.optimizeGPU(stream);
      // Cached values
      colSize = nColumns;
      colSetSize = nColumnSets;
      colCapacity = columnBlockOffsets.capacity(); // Uses this as an example
      colSetCapacity = setNumColumns.capacity(); // Uses this as an example
   }
   void prefetchDevice(gpuStream_t stream) {
      setColumnOffsets.optimizeGPU(stream);
      setNumColumns.optimizeGPU(stream);
      columnBlockOffsets.optimizeGPU(stream);
      columnNumBlocks.optimizeGPU(stream);
      minBlockK.optimizeGPU(stream);
      maxBlockK.optimizeGPU(stream);
      kBegin.optimizeGPU(stream);
      i.optimizeGPU(stream);
      j.optimizeGPU(stream);
   }
   __host__ size_t sizeCols() const {
      return colSize;
   }
   __host__ size_t capacityCols() const {
      return colCapacity;
   }
   __host__ size_t capacityColSets() const {
      return colSetCapacity;
   }
   __device__ size_t dev_sizeCols() const {
      return columnBlockOffsets.size(); // Uses this as an example
   }
   __device__ size_t dev_sizeColSets() const {
      return setNumColumns.size(); // Uses this as an example
   }
   __device__ size_t dev_capacityCols() const {
      return columnBlockOffsets.capacity(); // Uses this as an example
   }
   __device__ size_t dev_capacityColSets() const {
      return setNumColumns.capacity(); // Uses this as an example
   }
   size_t capacityInBytes() const {
      return colCapacity * (2*sizeof(uint)+5*sizeof(int))
         + colSetCapacity * (2*sizeof(uint))
         + 4 * sizeof(split::SplitVector<uint>)
         + 5 * sizeof(split::SplitVector<int>);
   }
   void setSizes(size_t nCols=0, size_t nColSets=0) {
      // Ensure capacities are handled with cached values
      setCapacities(nCols,nColSets);
      // Only then resize
      setColumnOffsets.resize(nColSets,true);
      setNumColumns.resize(nColSets,true);
      columnBlockOffsets.resize(nCols,true);
      columnNumBlocks.resize(nCols,true);
      minBlockK.resize(nCols,true);
      maxBlockK.resize(nCols,true);
      kBegin.resize(nCols,true);
      i.resize(nCols,true);
      j.resize(nCols,true);
      colSize = nCols;
      colSetSize = nColSets;
   }
   __device__ void device_setSizes(size_t nCols=0, size_t nColSets=0) {
      // Cannot recapacitate
      setColumnOffsets.device_resize(nColSets);
      setNumColumns.device_resize(nColSets);
      columnBlockOffsets.device_resize(nCols);
      columnNumBlocks.device_resize(nCols);
      minBlockK.device_resize(nCols);
      maxBlockK.device_resize(nCols);
      kBegin.device_resize(nCols);
      i.device_resize(nCols);
      j.device_resize(nCols);
      colSize = nCols;
      colSetSize = nColSets;
   }
   void setCapacities(size_t nCols=0, size_t nColSets=0) {
      // check cached capacities to prevent page faults if not necessary
      if (nCols > colCapacity) {
         // Recapacitate column vectors
         colCapacity = nCols * BLOCK_ALLOCATION_PADDING;
         columnBlockOffsets.reallocate(colCapacity);
         columnNumBlocks.reallocate(colCapacity);
         minBlockK.reallocate(colCapacity);
         maxBlockK.reallocate(colCapacity);
         kBegin.reallocate(colCapacity);
         i.reallocate(colCapacity);
         j.reallocate(colCapacity);
      }
      if (nColSets > colSetCapacity) {
         // Recapacitate columnSet vectors
         colSetCapacity = nColSets * BLOCK_ALLOCATION_PADDING;
         setColumnOffsets.reallocate(colSetCapacity);
         setNumColumns.reallocate(colSetCapacity);
      }
   }
};

extern ColumnOffsets *host_columnOffsetData;
extern uint gpu_largest_columnCount;
extern size_t gpu_probeFullSize, gpu_probeFlattenedSize, gpu_probeStride;

// Hash map and splitvectors buffers used in block adjustment are declared in block_adjust_gpu.hpp
// Vector and set for use in translation are declared in vlasovsolver/gpu_trans_map_amr.hpp

// Counters used in allocations
extern std::vector<uint> gpu_vlasov_allocatedSize;
extern uint gpu_acc_allocatedColumns;
extern uint gpu_acc_foundColumnsCount;

#endif
