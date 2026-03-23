/*
 * This file is part of Vlasiator.
 * Copyright 2024-2025 University of Helsinki 
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

#include <vector>
#include "../definitions.h"
#include "../spatial_cells/spatial_cell_wrapper.hpp"
#include "../object_wrapper.h"
//#include <stdint.h>
#include <dccrg.hpp>
#include <dccrg_cartesian_geometry.hpp>

//using namespace std;
using namespace spatial_cell;

void reduce_vlasov_dt(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                      const vector<CellID>& cells,
                      Real (&dtMaxLocal)[3]) {

   phiprof::Timer computeTimestepTimer {"compute-vlasov-timestep"};
   const Real HALF = 0.5;

   for (vector<CellID>::const_iterator cell_id = cells.begin(); cell_id != cells.end(); ++cell_id) {
      SpatialCell* cell = mpiGrid[*cell_id];
      const Real dx = cell->parameters[CellParams::DX];
      const Real dy = cell->parameters[CellParams::DY];
      const Real dz = cell->parameters[CellParams::DZ];
      cell->parameters[CellParams::MAXRDT] = numeric_limits<Real>::max();

      for (uint popID = 0; popID < getObjectWrapper().particleSpecies.size(); ++popID) {
         cell->set_max_r_dt(popID, numeric_limits<Real>::max());
         const Real EPS = numeric_limits<Real>::min() * 1000;

         const uint nBlocks = cell->get_number_of_velocity_blocks(popID);
         if (nBlocks==0) {
            continue;
         }
         #ifdef USE_GPU
         const vmesh::VelocityBlockContainer *blockContainer = cell->dev_get_velocity_blocks(popID);
         #else
         const vmesh::VelocityBlockContainer *blockContainer = cell->get_velocity_blocks(popID);
         #endif

         Real threadMin = std::numeric_limits<Real>::max();
         arch::parallel_reduce<arch::min>({2, nBlocks},
            ARCH_LOOP_LAMBDA (uint i, const uint blockLID, Real *lthreadMin) -> void{
               i = i * (WID - 1); // ie, i == 0, i == WID - 1
               const Real* blockParams = blockContainer->getParameters();
               const Real Vx =
                   blockParams[blockLID * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VXCRD] +
                   (i + HALF) * blockParams[blockLID * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVX] + EPS;
               const Real Vy =
                   blockParams[blockLID * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VYCRD] +
                   (i + HALF) * blockParams[blockLID * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVY] + EPS;
               const Real Vz =
                   blockParams[blockLID * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::VZCRD] +
                   (i + HALF) * blockParams[blockLID * BlockParams::N_VELOCITY_BLOCK_PARAMS + BlockParams::DVZ] + EPS;

               const Real dt_max_cell = min({dx / fabs(Vx), dy / fabs(Vy), dz / fabs(Vz)});
               lthreadMin[0] = min(dt_max_cell,lthreadMin[0]);
         }, threadMin);
         cell->set_max_r_dt(popID, threadMin);
         cell->parameters[CellParams::MAXRDT] = min(cell->get_max_r_dt(popID), cell->parameters[CellParams::MAXRDT]);
      } // end loop over popID

      if (cell->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY ||
          (cell->sysBoundaryLayer == 1 && cell->sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY)) {
         // spatial fluxes computed also for boundary cells
         dtMaxLocal[0] = min(dtMaxLocal[0], cell->parameters[CellParams::MAXRDT]);
      }

      if (cell->parameters[CellParams::MAXVDT] != 0 &&
          (cell->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY ||
           (P::vlasovAccelerateMaxwellianBoundaries && cell->sysBoundaryFlag == sysboundarytype::MAXWELLIAN))) {
         // acceleration only done on non-boundary cells
         dtMaxLocal[1] = min(dtMaxLocal[1], cell->parameters[CellParams::MAXVDT]);
      }
   }
}

void reduce_vlasov_dt_test_test(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                      const vector<CellID>& cells,
                      Real (&dtMaxLocal)[3]) {

   phiprof::Timer computeGpuTimestepTimer {"compute-vlasov-gpu-timestep"};
   // Does not use streams
   const uint nAllCells = cells.size();
   const uint nPOP = getObjectWrapper().particleSpecies.size();

   #ifdef USE_GPU
   // Resize dev_vmeshes, one for each cell and each pop
   gpu_trans_allocate(nAllCells*nPOP,0,0);

   gpuMemoryManager.startSession(0,0);

   SESSION_HOST_ALLOCATE(gpuMemoryManager, Real, host_max_dt, nAllCells*nPOP*sizeof(Real));
   SESSION_HOST_ALLOCATE(gpuMemoryManager, Real, host_dxdydz, nAllCells*nPOP*3*sizeof(Real));
   SESSION_HOST_ALLOCATE(gpuMemoryManager, uint, host_nBlocks, nAllCells*nPOP*sizeof(uint));
   SESSION_HOST_ALLOCATE(gpuMemoryManager, vmesh::VelocityBlockContainer*, host_blockContainers, nAllCells*nPOP*sizeof(vmesh::VelocityBlockContainer*));
   SESSION_ALLOCATE(gpuMemoryManager, Real, max_dt, nAllCells*nPOP*sizeof(Real));
   SESSION_ALLOCATE(gpuMemoryManager, Real, dxdydz, nAllCells*nPOP*3*sizeof(Real));
   SESSION_ALLOCATE(gpuMemoryManager, uint, nBlocks, nAllCells*nPOP*sizeof(uint));
   SESSION_ALLOCATE(gpuMemoryManager, vmesh::VelocityBlockContainer*, blockContainers, nAllCells*nPOP*sizeof(vmesh::VelocityBlockContainer*));
   
   Real* host_max_dt = GET_SESSION_HOST_POINTER(gpuMemoryManager, Real, host_max_dt);
   Real* host_dxdydz = GET_SESSION_HOST_POINTER(gpuMemoryManager, Real, host_dxdydz);
   uint* host_nBlocks = GET_SESSION_HOST_POINTER(gpuMemoryManager, uint, host_nBlocks);
   vmesh::VelocityBlockContainer** host_blockContainers = GET_SESSION_HOST_POINTER(gpuMemoryManager, vmesh::VelocityBlockContainer*, host_blockContainers);
   Real* max_dt = GET_SESSION_POINTER(gpuMemoryManager, Real, max_dt);
   Real* dxdydz = GET_SESSION_POINTER(gpuMemoryManager, Real, dxdydz);
   uint* nBlocks = GET_SESSION_POINTER(gpuMemoryManager, uint, nBlocks);
   vmesh::VelocityBlockContainer** blockContainers = GET_SESSION_POINTER(gpuMemoryManager, vmesh::VelocityBlockContainer*, blockContainers);

   #else 

   Real host_max_dt[nAllCells*nPOP];
   Real host_dxdydz[nAllCells*nPOP*3];
   uint host_nBlocks[nAllCells*nPOP];
   vmesh::VelocityBlockContainer *host_blockContainers[nAllCells*nPOP];

   #endif

   // Gather vmeshes
   uint maxNBlocks = 0;
   for(uint celli = 0; celli < nAllCells; celli++){
      SpatialCell* cell = mpiGrid[cells[celli]];
      cell->parameters[CellParams::MAXRDT] = numeric_limits<Real>::max();
      //cell->parameters[CellParams::MAXRDT] = numeric_limits<Real>::max();
      for (uint popID = 0; popID < nPOP; ++popID) {
         host_dxdydz[3*celli*nPOP + 3*popID + 0] = cell->parameters[CellParams::DX];
         host_dxdydz[3*celli*nPOP + 3*popID + 1] = cell->parameters[CellParams::DY];
         host_dxdydz[3*celli*nPOP + 3*popID + 2] = cell->parameters[CellParams::DZ];

         #ifdef USE_GPU
         host_blockContainers[celli*nPOP + popID] =  cell->dev_get_velocity_blocks(popID);
         #else
         host_blockContainers[celli*nPOP + popID] = cell->get_velocity_blocks(popID);
         #endif

         host_nBlocks[celli*nPOP + popID] = cell->get_number_of_velocity_blocks(popID);
         maxNBlocks = std::min(maxNBlocks, host_nBlocks[celli*nPOP + popID]);
         host_max_dt[celli*nPOP + popID] = numeric_limits<Real>::max();
      }
   }

   #ifdef USE_GPU
   CHK_ERR( gpuMemcpy(dxdydz, host_dxdydz, nAllCells*nPOP*3*sizeof(Real), gpuMemcpyHostToDevice) );
   CHK_ERR( gpuMemcpy(GET_POINTER(gpuMemoryManager, vmesh::VelocityMesh*, dev_vmeshes), GET_POINTER(gpuMemoryManager, vmesh::VelocityMesh*, host_vmeshes), nAllCells*nPOP*sizeof(vmesh::VelocityMesh*), gpuMemcpyHostToDevice) );
   CHK_ERR( gpuMemcpy(nBlocks, host_nBlocks, nAllCells*nPOP*sizeof(uint), gpuMemcpyHostToDevice) );
   CHK_ERR( gpuMemcpy(max_dt, host_max_dt, nAllCells*nPOP*sizeof(Real), gpuMemcpyHostToDevice) );
   #else
   Real *max_dt = host_max_dt;
   Real *dxdydz = host_dxdydz;
   uint *nBlocks = host_nBlocks;
   vmesh::VelocityBlockContainer **blockContainers = host_blockContainers;
   #endif

   const Real EPS = numeric_limits<Real>::min() * 1000;
   const Real HALF = 0.5;
   uint dim1 = 2;
   const uint* limits[2]={&dim1, host_nBlocks};
   const uint maxLimits[2]={dim1, maxNBlocks};
   
   arch::parallel_reduce_test<arch::min>({nAllCells, nPOP}, {1, nAllCells*nPOP}, limits, maxLimits,
      ARCH_LOOP_LAMBDA (uint i, const uint blockLID, const uint cellIndex, const uint popID, Real *lthreadMin) -> void{
         const Real dx = dxdydz[3*cellIndex*nPOP + 3*popID + 0];
         const Real dy = dxdydz[3*cellIndex*nPOP + 3*popID + 1];
         const Real dz = dxdydz[3*cellIndex*nPOP + 3*popID + 2];

         i = i * (WID - 1); // ie, i == 0, i == WID - 1
         const vmesh::VelocityBlockContainer *blockContainer = blockContainers[cellIndex*nPOP + popID];

         if (blockLID < nBlocks[cellIndex*nPOP + popID]) {
            const Real* blockParams = blockContainer->getParameters(popID);

            const Real Vx = blockParams[0] + (i + HALF) * blockParams[3] + EPS;
            const Real Vy = blockParams[1] + (i + HALF) * blockParams[4] + EPS;
            const Real Vz = blockParams[2] + (i + HALF) * blockParams[5] + EPS;

            const Real dt_max_cell = min({dx / fabs(Vx), dy / fabs(Vy), dz / fabs(Vz)});
            lthreadMin[0] = min(dt_max_cell,lthreadMin[0]);
         }
   }, max_dt);

   #ifdef USE_GPU
   CHK_ERR( gpuPeekAtLastError() );
   CHK_ERR( gpuMemcpy(host_max_dt, max_dt, nAllCells*nPOP*sizeof(Real), gpuMemcpyDeviceToHost) );
   // CHK_ERR( gpuStreamSynchronize(bgStream) );
   #endif

   #pragma omp parallel for schedule(static)
   for(uint celli = 0; celli < nAllCells; celli++){
      SpatialCell* cell = mpiGrid[cells[celli]];
      for (uint popID = 0; popID < nPOP; ++popID) {
         cell->set_max_r_dt(popID, host_max_dt[celli*nPOP + popID]);
         cell->parameters[CellParams::MAXRDT] = min(cell->get_max_r_dt(popID), cell->parameters[CellParams::MAXRDT]);
      }
   }
   computeGpuTimestepTimer.stop();

   #ifdef USE_GPU
   gpuMemoryManager.endSession();
   #endif

   // GPUTODO thread this?
   phiprof::Timer computeRestTimestepTimer {"compute-vlasov-rest-timestep"};
   for (vector<CellID>::const_iterator cell_id = cells.begin(); cell_id != cells.end(); ++cell_id) {
      SpatialCell* cell = mpiGrid[*cell_id];

      if (cell->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY ||
          (cell->sysBoundaryLayer == 1 && cell->sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY)) {
         // spatial fluxes computed also for L1 boundary cells
         dtMaxLocal[0] = min(dtMaxLocal[0], cell->parameters[CellParams::MAXRDT]);
      }

      if (cell->parameters[CellParams::MAXVDT] != 0 &&
          (cell->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY ||
           (P::vlasovAccelerateMaxwellianBoundaries && cell->sysBoundaryFlag == sysboundarytype::MAXWELLIAN))) {
         // acceleration only done on non-boundary cells
         dtMaxLocal[1] = min(dtMaxLocal[1], cell->parameters[CellParams::MAXVDT]);
      }
   }
   computeRestTimestepTimer.stop();
}
