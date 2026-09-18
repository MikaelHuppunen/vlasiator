/*
 * This file is part of Vlasiator.
 * Copyright 2010-2016 Finnish Meteorological Institute
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

#ifdef _OPENMP
   #include <omp.h>
#endif

#include "ldz_magnetic_field.hpp"

#ifdef USE_GPU
#define functionDevice __device__
#else
#define functionDevice __host__
#endif

functionDevice void propagateMagneticField(fsgrids::perbspan perb,
                            fsgrids::perbspan perbdt2,
                            fsgrids::constefieldspan e,
                            fsgrids::constefieldspan edt2,
                            const fsgrid::FsStencil& stencil, Real dt, int32_t RKCase, bool doX, bool doY, bool doZ,
                            const std::array<Real, 3>& gridSpacing) {
   creal dtdx = dt / gridSpacing[0];
   creal dtdy = dt / gridSpacing[1];
   creal dtdz = dt / gridSpacing[2];

   std::array<Real, fsgrids::bfield::N_BFIELD>& perBGrid0 = perb[stencil.ooo()];

   if (doX == true) {
      switch (RKCase) {
      case RK_ORDER1: {
         const auto& EGrid0 = e[stencil.ooo()];
         const auto& EGrid1 = e[stencil.opo()];
         const auto& EGrid2 = e[stencil.oop()];
         perBGrid0[fsgrids::bfield::PERBX] += dtdz * (EGrid2[fsgrids::efield::EY] - EGrid0[fsgrids::efield::EY]) +
                                              dtdy * (EGrid0[fsgrids::efield::EZ] - EGrid1[fsgrids::efield::EZ]);
         break;
      }

      case RK_ORDER2_STEP1: {
         auto& perBDt2Grid0 = perbdt2[stencil.ooo()];
         const auto& EGrid0 = e[stencil.ooo()];
         const auto& EGrid1 = e[stencil.opo()];
         const auto& EGrid2 = e[stencil.oop()];
         perBDt2Grid0[fsgrids::bfield::PERBX] = perBGrid0[fsgrids::bfield::PERBX] +
             0.5 * (dtdz * (EGrid2[fsgrids::efield::EY] - EGrid0[fsgrids::efield::EY]) +
                    dtdy * (EGrid0[fsgrids::efield::EZ] - EGrid1[fsgrids::efield::EZ]));
         break;
      }

      case RK_ORDER2_STEP2: {
         const auto& EGrid0 = edt2[stencil.ooo()];
         const auto& EGrid1 = edt2[stencil.opo()];
         const auto& EGrid2 = edt2[stencil.oop()];
         perBGrid0[fsgrids::bfield::PERBX] += dtdz * (EGrid2[fsgrids::efield::EY] - EGrid0[fsgrids::efield::EY]) +
                                              dtdy * (EGrid0[fsgrids::efield::EZ] - EGrid1[fsgrids::efield::EZ]);
         break;
      }

      default:
         #ifdef USE_GPU
         printf("%s:%d: Invalid RK case.\n", __FILE__, __LINE__);
         assert(false);
         #else
         std::cerr << __FILE__ << ":" << __LINE__ << ":" << "Invalid RK case." << std::endl;
         abort();
         #endif
      }
   }

   if (doY == true) {
      switch (RKCase) {
      case RK_ORDER1: {
         const auto& EGrid0 = e[stencil.ooo()];
         const auto& EGrid1 = e[stencil.oop()];
         const auto& EGrid2 = e[stencil.poo()];
         perBGrid0[fsgrids::bfield::PERBY] += dtdx * (EGrid2[fsgrids::efield::EZ] - EGrid0[fsgrids::efield::EZ]) +
                                              dtdz * (EGrid0[fsgrids::efield::EX] - EGrid1[fsgrids::efield::EX]);
         break;
      }
      case RK_ORDER2_STEP1: {
         auto& perBDt2Grid0 = perbdt2[stencil.ooo()];
         const auto& EGrid0 = e[stencil.ooo()];
         const auto& EGrid1 = e[stencil.oop()];
         const auto& EGrid2 = e[stencil.poo()];
         perBDt2Grid0[fsgrids::bfield::PERBY] = perBGrid0[fsgrids::bfield::PERBY] +
             0.5 * (dtdx * (EGrid2[fsgrids::efield::EZ] - EGrid0[fsgrids::efield::EZ]) +
                    dtdz * (EGrid0[fsgrids::efield::EX] - EGrid1[fsgrids::efield::EX]));
         break;
      }
      case RK_ORDER2_STEP2: {
         const auto& EGrid0 = edt2[stencil.ooo()];
         const auto& EGrid1 = edt2[stencil.oop()];
         const auto& EGrid2 = edt2[stencil.poo()];
         perBGrid0[fsgrids::bfield::PERBY] += dtdx * (EGrid2[fsgrids::efield::EZ] - EGrid0[fsgrids::efield::EZ]) +
                                              dtdz * (EGrid0[fsgrids::efield::EX] - EGrid1[fsgrids::efield::EX]);
         break;
      }
      default:
         #ifdef USE_GPU
         printf("%s:%d: Invalid RK case.\n", __FILE__, __LINE__);
         assert(false);
         #else
         std::cerr << __FILE__ << ":" << __LINE__ << ":" << "Invalid RK case." << std::endl;
         abort();
         #endif
      }
   }

   if (doZ == true) {
      switch (RKCase) {
      case RK_ORDER1: {
         const auto& EGrid0 = e[stencil.ooo()];
         const auto& EGrid1 = e[stencil.poo()];
         const auto& EGrid2 = e[stencil.opo()];
         perBGrid0[fsgrids::bfield::PERBZ] += dtdy * (EGrid2[fsgrids::efield::EX] - EGrid0[fsgrids::efield::EX]) +
                                              dtdx * (EGrid0[fsgrids::efield::EY] - EGrid1[fsgrids::efield::EY]);
         break;
      }
      case RK_ORDER2_STEP1: {
         auto& perBDt2Grid0 = perbdt2[stencil.ooo()];
         const auto& EGrid0 = e[stencil.ooo()];
         const auto& EGrid1 = e[stencil.poo()];
         const auto& EGrid2 = e[stencil.opo()];
         perBDt2Grid0[fsgrids::bfield::PERBZ] = perBGrid0[fsgrids::bfield::PERBZ] +
             0.5 * (dtdy * (EGrid2[fsgrids::efield::EX] - EGrid0[fsgrids::efield::EX]) +
                    dtdx * (EGrid0[fsgrids::efield::EY] - EGrid1[fsgrids::efield::EY]));
         break;
      }
      case RK_ORDER2_STEP2: {
         const auto& EGrid0 = edt2[stencil.ooo()];
         const auto& EGrid1 = edt2[stencil.poo()];
         const auto& EGrid2 = edt2[stencil.opo()];
         perBGrid0[fsgrids::bfield::PERBZ] += dtdy * (EGrid2[fsgrids::efield::EX] - EGrid0[fsgrids::efield::EX]) +
                                              dtdx * (EGrid0[fsgrids::efield::EY] - EGrid1[fsgrids::efield::EY]);
         break;
      }
      default:
         #ifdef USE_GPU
         printf("%s:%d: Invalid RK case.\n", __FILE__, __LINE__);
         assert(false);
         #else
         std::cerr << __FILE__ << ":" << __LINE__ << ":" << "Invalid RK case." << std::endl;
         abort();
         #endif
      }
   }
}

/*! \brief Low-level magnetic field propagation function.
 *
 * Propagates the magnetic field according to the system boundary conditions.
 *
 * \param perb fsGrid holding the perturbed B quantities at runge-kutta t=0
 * \param perbdt2 fsGrid holding the perturbed B quantities at runge-kutta t=0.5
 * \param bgb fsGrid holding the background field B quantities
 * \param technical fsgrid holding the technical parameters
 * \param gridSpacing cell size in x,y,z
 * \param globalCoordinates cell global grid coordinate indices
 * \param stencil fsgrid stencil for cell
 * \param sysBoundaries System boundary conditions existing
 * \param RKCase Element in the enum defining the Runge-Kutta method steps
 * \param component component to compute
 *
 * \sa propagateMagneticFieldSimple propagateMagneticField
 */
void propagateSysBoundaryMagneticField(fsgrids::perbspan perb,
                                       fsgrids::perbspan perbdt2,
                                       fsgrids::constbgbspan bgb,
                                       fsgrids::consttechnicalspan technical,
                                       const std::array<Real, 3>& gridSpacing,
                                       const std::array<fsgrid::FsSize_t, 3>& globalCoordinates,
                                       const fsgrid::FsStencil& stencil, SysBoundary& sysBoundaries, int32_t RKCase,
                                       uint32_t component) {
   const bool case0 = RKCase == RK_ORDER1 || RKCase == RK_ORDER2_STEP2;
   auto& out = case0 ? perb[stencil.ooo()] : perbdt2[stencil.ooo()];
   const auto& pb = case0 ? perb : perbdt2;

   out[fsgrids::bfield::PERBX + component] =
       sysBoundaries.getSysBoundary(technical[stencil.ooo()].sysBoundaryFlag)
           ->fieldSolverBoundaryCondMagneticField(pb, bgb, technical, gridSpacing, globalCoordinates, stencil,
                                                  component);
}

/*! \brief Low-level magnetic field propagation function.
 *
 * Propagates the magnetic field according to the system boundary conditions.
 *
 * \param perb fsGrid holding the perturbed B quantities at runge-kutta t=0
 * \param perbdt2 fsGrid holding the perturbed B quantities at runge-kutta t=0.5
 * \param bgb fsGrid holding the background field B quantities
 * \param technical fsgrid holding the technical parameters
 * \param gridSpacing cell size in x,y,z
 * \param globalCoordinates cell global grid coordinate indices
 * \param stencil fsgrid stencil for cell
 * \param sysBoundaries System boundary conditions existing
 * \param RKCase Element in the enum defining the Runge-Kutta method steps
 * \param component component to compute
 *
 * \sa propagateMagneticFieldSimple propagateMagneticField
 */
__device__ void propagateSysBoundaryMagneticFieldDevice(fsgrids::perbspan perb,
                                       fsgrids::perbspan perbdt2,
                                       fsgrids::constbgbspan bgb,
                                       fsgrids::consttechnicalspan technical,
                                       const std::array<Real, 3>& gridSpacing,
                                       const std::array<fsgrid::FsSize_t, 3>& globalCoordinates,
                                       const fsgrid::FsStencil& stencil, SysBoundaryDevice *sysBoundaries, int32_t RKCase,
                                       uint32_t component) {
   const bool case0 = RKCase == RK_ORDER1 || RKCase == RK_ORDER2_STEP2;
   auto& out = case0 ? perb[stencil.ooo()] : perbdt2[stencil.ooo()];
   const auto& pb = case0 ? perb : perbdt2;

   out[fsgrids::bfield::PERBX + component] =
       sysBoundaries[0].getSysBoundary(technical[stencil.ooo()].sysBoundaryFlag)
                        ->fieldSolverBoundaryCondMagneticField(pb, bgb, technical, gridSpacing, globalCoordinates, stencil,
                                                  component);
}

__global__ void constructDoNotCompute(SBC::SysBoundaryConditionDevice* mem) {
   new (mem) SBC::DoNotComputeDevice();
}

__global__ void constructMaxwellian(SBC::SysBoundaryConditionDevice* mem, const Real (*templateB)[3]) {
   new (mem) SBC::MaxwellianDevice(templateB);
}

__global__ void constructOutflow(SBC::SysBoundaryConditionDevice* mem) {
   new (mem) SBC::OutflowDevice();
}

void uploadIndexToSysBoundary(SysBoundary& sysBoundaries, SysBoundaryDevice& host_sysBoundaries) {
   for (const auto& [type, host_condition] : sysBoundaries.getIndexToSysBoundary()) {
      SBC::SysBoundaryConditionDevice* dev_condition = nullptr;
      switch (type) {
         case sysboundarytype::OUTFLOW: {
            cudaMalloc(&dev_condition, sizeof(SBC::OutflowDevice));
            constructOutflow<<<1,1>>>(dev_condition);
            cudaDeviceSynchronize();
            break;
         }
         case sysboundarytype::MAXWELLIAN: {
            auto* host_maxwellian = static_cast<SBC::Maxwellian*>(host_condition);
            Real (*dev_templateB)[3] = nullptr;
            cudaMalloc(&dev_templateB, sizeof(Real[6][3]));
            cudaMemcpy(dev_templateB, host_maxwellian->templateB, sizeof(Real[6][3]), cudaMemcpyHostToDevice);
            cudaMalloc(&dev_condition, sizeof(SBC::MaxwellianDevice));
            constructMaxwellian<<<1,1>>>(dev_condition, dev_templateB);
            cudaDeviceSynchronize();
            cudaFree(dev_templateB);
            break;
         }
         case sysboundarytype::IONOSPHERE: {
            std::cerr << "IONOSPHERE not implemented" << std::endl;
            cudaMalloc(&dev_condition, sizeof(SBC::OutflowDevice));
            constructOutflow<<<1,1>>>(dev_condition);
            cudaDeviceSynchronize();
            break;
         }
         case sysboundarytype::DO_NOT_COMPUTE: {
            cudaMalloc(&dev_condition, sizeof(SBC::DoNotComputeDevice));
            constructDoNotCompute<<<1,1>>>(dev_condition);
            cudaDeviceSynchronize();
            break;
         }
         case sysboundarytype::COPYSPHERE: {
            std::cerr << "COPYSPHERE not implemented" << std::endl;
            cudaMalloc(&dev_condition, sizeof(SBC::OutflowDevice));
            constructOutflow<<<1,1>>>(dev_condition);
            cudaDeviceSynchronize();
            break;
         }
      }
      host_sysBoundaries.setSysBoundary(type, dev_condition);
   }
}

/*! \brief High-level magnetic field propagation function.
 *
 * Propagates the magnetic field and applies the field boundary conditions defined in project.h where needed.
 *
 * \param perb fsGrid holding the perturbed B quantities at runge-kutta t=0
 * \param perbdt2 fsGrid holding the perturbed B quantities at runge-kutta t=0.5
 * \param bgb fsGrid holding the background B quantities
 * \param e fsGrid holding the Electric field quantities at runge-kutta t=0
 * \param edt2 fsGrid holding the Electric field quantities at runge-kutta t=0.5
 * \param technical fsGrid holding technical information (such as boundary types)
 * \param fsgrid container of all fsgrids
 * \param sysBoundaries System boundary conditions existing
 * \param dt Length of the time step
 * \param RKCase Element in the enum defining the Runge-Kutta method steps
 *
 * \sa propagateMagneticField propagateSysBoundaryMagneticField
 */
void propagateMagneticFieldSimple(fsgrids::perbspan perb,
                                  fsgrids::perbspan perbdt2,
                                  fsgrids::bgbspan bgb,
                                  fsgrids::efieldspan e,
                                  fsgrids::efieldspan edt2,
                                  fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
                                  SysBoundaryDevice *sysBoundaries, creal& dt, cint& RKCase) {
   phiprof::Timer propagateBTimer{"Propagate magnetic field"};
   const auto* localSize = &fsgrid.getLocalSize()[0];
   const auto& gridSpacing = fsgrid.getGridSpacing();
   const size_t numCells = fsgrid.getNumCells();
   cudaDeviceSynchronize();

   int sysBoundaryTimerId{phiprof::initializeTimer("Magnetic Field compute sysboundary cells")};
   fsgrid.parallel_for_GPU([](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
                       phiprof::initializeTimer("Magnetic Field compute cells"), technical,
                       [=] __device__ (const fsgrid::Coordinates &coordinates, const fsgrid::FsStencil& stencil, cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
                          cuint bitfield = technical[stencil.ooo()].SOLVE;
                          propagateMagneticField(
                             perb, perbdt2, e, edt2, stencil, dt, RKCase, ((bitfield & compute::BX) == compute::BX),
                             ((bitfield & compute::BY) == compute::BY), ((bitfield & compute::BZ) == compute::BZ), coordinates.physicalGridSpacing);
                       });
                       
   cudaDeviceSynchronize();

   // This communication is needed for boundary conditions, in practice almost all
   // of the communication is going to be redone in calculateDerivativesSimple
   // TODO: do not transfer if there are no field boundaryconditions
   phiprof::Timer mpiTimer{"MPI", {"MPI"}};
   if (RKCase == RK_ORDER1 || RKCase == RK_ORDER2_STEP2) {
      // Exchange PERBX,PERBY,PERBZ with neighbours
      std::vector<fsgrids::perbElement> hostStagingBuffer(perb.size());
      cudaMemcpy(hostStagingBuffer.data(), perb.data(), perb.size_bytes(), cudaMemcpyDeviceToHost);
      fsgrid.updateGhostCells(std::span(hostStagingBuffer));
      cudaMemcpy(perb.data(), hostStagingBuffer.data(), perb.size_bytes(), cudaMemcpyHostToDevice);
   } else { // RKCase == RK_ORDER2_STEP1
      // Exchange PERBX_DT2,PERBY_DT2,PERBZ_DT2 with neighbours
      std::vector<fsgrids::perbElement> hostStagingBuffer(perbdt2.size());
      cudaMemcpy(hostStagingBuffer.data(), perbdt2.data(), perbdt2.size_bytes(), cudaMemcpyDeviceToHost);
      fsgrid.updateGhostCells(std::span(hostStagingBuffer));
      cudaMemcpy(perbdt2.data(), hostStagingBuffer.data(), perbdt2.size_bytes(), cudaMemcpyHostToDevice);
   }
   mpiTimer.stop();

   cudaDeviceSynchronize();

   // The looping below was modified in https://github.com/fmihpc/vlasiator/pull/1110/files
   // with a reported performance gain of 10% of field solver performance in production-like
   // conditions. Unfortunately as of https://github.com/fmihpc/vlasiator/pull/1099 (fsgrid
   // parallel_for mechanism) this split would become very cumbersome and is therefore
   // reversed until more optimisation is needed on CPU or GPU.

   // Propagate B on system boundary/process inner cells
   phiprof::Timer sysBoundaryTimer {sysBoundaryTimerId};

   // L1 pass
   fsgrid.parallel_for_GPU([](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
                       phiprof::initializeTimer("Magnetic field L1 pass"), technical,
                       [=] __device__(const fsgrid::Coordinates &coordinates, const fsgrid::FsStencil& stencil, cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
      if (sysBoundaryLayer == 1) {
         cuint bitfield = technical[stencil.ooo()].SOLVE;
         const auto globalCoordinates = coordinates.localToGlobal(stencil.i, stencil.j, stencil.k);
         if ((bitfield & compute::BX) != compute::BX) {
            propagateSysBoundaryMagneticFieldDevice(perb, perbdt2, bgb, technical, gridSpacing, globalCoordinates, stencil, sysBoundaries, RKCase, 0);
         }
         if ((bitfield & compute::BY) != compute::BY) {
            propagateSysBoundaryMagneticFieldDevice(perb, perbdt2, bgb, technical, gridSpacing, globalCoordinates, stencil, sysBoundaries, RKCase, 1);
         }
         if ((bitfield & compute::BZ) != compute::BZ) {
            propagateSysBoundaryMagneticFieldDevice(perb, perbdt2, bgb, technical, gridSpacing, globalCoordinates, stencil, sysBoundaries, RKCase, 2);
         }
      }
   });
   sysBoundaryTimer.stop();

   cudaDeviceSynchronize();

   mpiTimer.start();
   if (RKCase == RK_ORDER1 || RKCase == RK_ORDER2_STEP2) {
      // Exchange PERBX,PERBY,PERBZ with neighbours
      std::vector<fsgrids::perbElement> hostStagingBuffer(perb.size());
      cudaMemcpy(hostStagingBuffer.data(), perb.data(), perb.size_bytes(), cudaMemcpyDeviceToHost);
      fsgrid.updateGhostCells(std::span(hostStagingBuffer));
      cudaMemcpy(perb.data(), hostStagingBuffer.data(), perb.size_bytes(), cudaMemcpyHostToDevice);
   } else { // RKCase == RK_ORDER2_STEP1
      // Exchange PERBX_DT2,PERBY_DT2,PERBZ_DT2 with neighbours
      std::vector<fsgrids::perbElement> hostStagingBuffer(perbdt2.size());
      cudaMemcpy(hostStagingBuffer.data(), perbdt2.data(), perbdt2.size_bytes(), cudaMemcpyDeviceToHost);
      fsgrid.updateGhostCells(std::span(hostStagingBuffer));
      cudaMemcpy(perbdt2.data(), hostStagingBuffer.data(), perbdt2.size_bytes(), cudaMemcpyHostToDevice);
   }
   mpiTimer.stop();

   cudaDeviceSynchronize();

   sysBoundaryTimer.start();
   // L2 pass
   fsgrid.parallel_for_GPU([](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
                       phiprof::initializeTimer("Magnetic field L2 pass"), technical,
                       [=] __device__(const fsgrid::Coordinates &coordinates, const fsgrid::FsStencil& stencil, cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
      if(sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY &&
         sysBoundaryLayer == 2
      ) {
         cuint bitfield = technical[stencil.ooo()].SOLVE;
         const auto globalCoordinates = coordinates.localToGlobal(stencil.i, stencil.j, stencil.k);
         if ((bitfield & compute::BX) != compute::BX) {
            propagateSysBoundaryMagneticFieldDevice(perb, perbdt2, bgb, technical, gridSpacing, globalCoordinates, stencil, sysBoundaries, RKCase, 0);
         }
         if ((bitfield & compute::BY) != compute::BY) {
            propagateSysBoundaryMagneticFieldDevice(perb, perbdt2, bgb, technical, gridSpacing, globalCoordinates, stencil, sysBoundaries, RKCase, 1);
         }
         if ((bitfield & compute::BZ) != compute::BZ) {
            propagateSysBoundaryMagneticFieldDevice(perb, perbdt2, bgb, technical, gridSpacing, globalCoordinates, stencil, sysBoundaries, RKCase, 2);
         }
      }
   });
   sysBoundaryTimer.stop();
   propagateBTimer.stop(numCells, "Spatial Cells");

   cudaDeviceSynchronize();
}
