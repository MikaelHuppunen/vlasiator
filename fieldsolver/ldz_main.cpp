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

/*! \file ldz_main.cpp
 * \brief Londrillo -- Del Zanna upwind constrained transport field solver.
 *
 * On the divergence-free condition in Godunov-type schemes for
 * ideal magnetohydrodynamics: the upwind constrained transport method,
 * P. Londrillo and L. Del Zanna, J. Comp. Phys., 195, 2004.
 * http://dx.doi.org/10.1016/j.jcp.2003.09.016
 *
 * Reconstructions taken from:
 * Efficient, high accuracy ADER-WENO schemes for hydrodynamics and
 * divergence-free magnetohydrodynamics, D. S. Balsara, T. Rumpf,
 * M. Dumbser, C.-D. Munz, J. Comp. Phys, 228, 2480-2516, 2009.
 * http://dx.doi.org/10.1016/j.jcp.2008.12.003
 * and
 * Divergence-free reconstruction of magnetic fields and WENO
 * schemes for magnetohydrodynamics, D. S. Balsara, J. Comp. Phys.,
 * 228, 5040-5056, 2009.
 * http://dx.doi.org/10.1016/j.jcp.2009.03.038
 *
 * *****  NOTATION USED FOR VARIABLES FOLLOWS THE ONES USED  *****\n
 * *****      IN THE ABOVEMENTIONED PUBLICATION(S)           *****
 */

#include "ldz_electric_field.hpp"
#include "ldz_magnetic_field.hpp"
#include "ldz_hall.hpp"
#include "ldz_gradpe.hpp"
#include "ldz_volume.hpp"
#include "fs_common.h"
#include "derivatives.hpp"
#include "fs_limiters.h"
#include "mpiconversion.h"
#include "../fieldtracing/fieldtracing.h"
#include "../logger.h"
extern Logger logFile;

/*! \brief Top-level field propagation function.
 *
 * Propagates the magnetic field, computes the derivatives and the upwinded
 * electric field, then computes the volume-averaged field values. Takes care
 * of the Runge-Kutta iteration at the top level, the functions called get as
 * an argument the element from the enum defining the current stage and handle
 * their job correspondingly.
 *
 * \param perb fsgrid holding perturbed magnetic field
 * \param perbdt2 fsgrid holding perturbed magnetic field at Runge-Kutta half-step
 * \param e fsgrid holding electric field
 * \param edt2 fsgrid holding electric field
 * \param ehall fsgrid holding Hall term electric field
 * \param egradpe fsgrid holding gradient of electron pressure electric field
 * \param egradpedt2 fsgrid holding gradient of electron pressure electric field at Runge-Kutta half-step
 * \param moments fsgrid holding moments
 * \param momentsdt2 fsgrid holding moments at Runge-Kutta half-step
 * \param dperb fsgrid holding perturbed magnetic field derivatives
 * \param dmoments fsgrid holding moments derivatives
 * \param dmomentsdt2 fsgrid holding moments derivatives at Runge-Kutta half-step
 * \param bgb fsgrid holding background magnetic field
 * \param vol fsgrid holding volume magnetic field
 * \param technical fsgrid holding technical parameters
 * \param fsgrid fsgrids container
 * \param sysBoundaries Container of existing system boundaries
 * \param dt Length of the time step
 * \param subcycles Number of subcycles to compute.
 *
 * \sa propagateMagneticFieldSimple calculateDerivativesSimple calculateUpwindedElectricFieldSimple
 * calculateVolumeAveragedFields calculateBVOLDerivativesSimple
 *
 */
bool propagateFields(fsgrids::perbspan perb,
                     fsgrids::perbspan perbdt2,
                     fsgrids::efieldspan e,
                     fsgrids::efieldspan edt2,
                     fsgrids::ehallspan ehall,
                     fsgrids::egradpespan egradpe,
                     fsgrids::egradpespan egradpedt2,
                     fsgrids::momentsspan moments,
                     fsgrids::momentsspan momentsdt2,
                     fsgrids::dperbspan dperb,
                     fsgrids::dmomentsspan dmoments,
                     fsgrids::dmomentsspan dmomentsdt2,
                     fsgrids::bgbspan bgb,
                     fsgrids::volspan vol,
                     fsgrids::technicalspan technical,
                     FieldSolverGrid &fsgrid,
                     SysBoundary& sysBoundaries,
                     creal& dt,
                     cuint subcycles) {

   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_perb);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_perbdt2);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_e);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_edt2);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_bgb);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, dev_sysBoundaries);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_technical);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_dmoments);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_dperb);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_dmomentsdt2);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_dperbdt2);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_moments);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_momentsdt2);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_ehall);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_egradpe);
   CREATE_UNIQUE_POINTER(gpuMemoryManager, d_egradpedt2);

   if (subcycles == 0) {
      cerr << "Field solver subcycles cannot be 0." << endl;
      exit(1);
   }

   const auto* localSize = &fsgrid.getLocalSize()[0];

   ALLOCATE_GPU(gpuMemoryManager, d_technical, technical.size() * sizeof(fsgrids::technical));
   fsgrids::technical *d_technical = GET_POINTER(gpuMemoryManager, fsgrids::technical, d_technical);
   cudaMemcpy(d_technical, technical.data(), technical.size() * sizeof(fsgrids::technical), cudaMemcpyHostToDevice);
   std::span<fsgrids::technical> dev_technical(d_technical, technical.size());

   fsgrid.parallel_for_GPU([](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
                     phiprof::initializeTimer("Initialize technical.maxFsDt"), dev_technical,
                     [=] __device__(const fsgrid::Coordinates &coordinates, const fsgrid::FsStencil& stencil, cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
                        dev_technical[stencil.ooo()].maxFsDt = std::numeric_limits<Real>::max();
                     });

   if (subcycles == 1) {
      SysBoundaryDevice host_sysBoundaries;
      uploadParametersToDevice();
      uploadIndexToSysBoundary(sysBoundaries,  host_sysBoundaries);
      ALLOCATE_GPU(gpuMemoryManager, d_perb, perb.size() * sizeof(fsgrids::perbElement));
      ALLOCATE_GPU(gpuMemoryManager, d_perbdt2, perbdt2.size() * sizeof(fsgrids::perbElement));
      ALLOCATE_GPU(gpuMemoryManager, d_e, e.size() * sizeof(fsgrids::efieldElement));
      ALLOCATE_GPU(gpuMemoryManager, d_edt2, edt2.size() * sizeof(fsgrids::efieldElement));
      ALLOCATE_GPU(gpuMemoryManager, d_bgb, bgb.size() * sizeof(fsgrids::bgbElement));
      ALLOCATE_GPU(gpuMemoryManager, dev_sysBoundaries, sizeof(SysBoundaryDevice));
      ALLOCATE_GPU(gpuMemoryManager, d_dmoments, dmoments.size() * sizeof(fsgrids::dmomentsElement));
      ALLOCATE_GPU(gpuMemoryManager, d_dperb, dperb.size() * sizeof(fsgrids::dperbElement));
      ALLOCATE_GPU(gpuMemoryManager, d_moments, moments.size() * sizeof(fsgrids::momentsElement));
      ALLOCATE_GPU(gpuMemoryManager, d_dmomentsdt2, dmomentsdt2.size() * sizeof(fsgrids::dmomentsElement));
      ALLOCATE_GPU(gpuMemoryManager, d_momentsdt2, momentsdt2.size() * sizeof(fsgrids::momentsElement));
      ALLOCATE_GPU(gpuMemoryManager, d_ehall, ehall.size() * sizeof(fsgrids::ehallElement));
      ALLOCATE_GPU(gpuMemoryManager, d_egradpe, egradpe.size() * sizeof(fsgrids::egradpeElement));
      ALLOCATE_GPU(gpuMemoryManager, d_egradpedt2, egradpedt2.size() * sizeof(fsgrids::egradpeElement));
      fsgrids::perbElement *d_perb = GET_POINTER(gpuMemoryManager, fsgrids::perbElement, d_perb);
      fsgrids::perbElement *d_perbdt2 = GET_POINTER(gpuMemoryManager, fsgrids::perbElement, d_perbdt2);
      fsgrids::efieldElement *d_e = GET_POINTER(gpuMemoryManager, fsgrids::efieldElement, d_e);
      fsgrids::efieldElement *d_edt2 = GET_POINTER(gpuMemoryManager, fsgrids::efieldElement, d_edt2);
      fsgrids::bgbElement *d_bgb = GET_POINTER(gpuMemoryManager, fsgrids::bgbElement, d_bgb);
      SysBoundaryDevice *dev_sysBoundaries = GET_POINTER(gpuMemoryManager, SysBoundaryDevice, dev_sysBoundaries);
      fsgrids::dmomentsElement *d_dmoments = GET_POINTER(gpuMemoryManager, fsgrids::dmomentsElement, d_dmoments);
      fsgrids::dperbElement *d_dperb = GET_POINTER(gpuMemoryManager, fsgrids::dperbElement, d_dperb);
      fsgrids::momentsElement *d_moments = GET_POINTER(gpuMemoryManager, fsgrids::momentsElement, d_moments);
      fsgrids::dmomentsElement *d_dmomentsdt2 = GET_POINTER(gpuMemoryManager, fsgrids::dmomentsElement, d_dmomentsdt2);
      fsgrids::momentsElement *d_momentsdt2 = GET_POINTER(gpuMemoryManager, fsgrids::momentsElement, d_momentsdt2);
      fsgrids::ehallElement *d_ehall = GET_POINTER(gpuMemoryManager, fsgrids::ehallElement, d_ehall);
      fsgrids::egradpeElement *d_egradpe = GET_POINTER(gpuMemoryManager, fsgrids::egradpeElement, d_egradpe);
      fsgrids::egradpeElement *d_egradpedt2 = GET_POINTER(gpuMemoryManager, fsgrids::egradpeElement, d_egradpedt2);
      cudaMemcpy(d_perb, perb.data(),  perb.size() * sizeof(fsgrids::perbElement), cudaMemcpyHostToDevice);
      cudaMemcpy(d_perbdt2, perbdt2.data(),  perbdt2.size() * sizeof(fsgrids::perbElement), cudaMemcpyHostToDevice);
      cudaMemcpy(d_e, e.data(),  e.size() * sizeof(fsgrids::efieldElement), cudaMemcpyHostToDevice);
      cudaMemcpy(d_edt2, edt2.data(),  edt2.size() * sizeof(fsgrids::efieldElement), cudaMemcpyHostToDevice);
      cudaMemcpy(d_bgb, bgb.data(),  bgb.size() * sizeof(fsgrids::bgbElement), cudaMemcpyHostToDevice);
      cudaMemcpy(dev_sysBoundaries, &host_sysBoundaries,  sizeof(SysBoundaryDevice), cudaMemcpyHostToDevice);
      cudaMemcpy(d_dmoments, dmoments.data(),  dmoments.size() * sizeof(fsgrids::dmomentsElement), cudaMemcpyHostToDevice);
      cudaMemcpy(d_dperb, dperb.data(),  dperb.size() * sizeof(fsgrids::dperbElement), cudaMemcpyHostToDevice);
      cudaMemcpy(d_moments, moments.data(),  moments.size() * sizeof(fsgrids::momentsElement), cudaMemcpyHostToDevice);
      cudaMemcpy(d_dmomentsdt2, dmomentsdt2.data(),  dmomentsdt2.size() * sizeof(fsgrids::dmomentsElement), cudaMemcpyHostToDevice);
      cudaMemcpy(d_momentsdt2, momentsdt2.data(),  momentsdt2.size() * sizeof(fsgrids::momentsElement), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ehall, ehall.data(),  ehall.size() * sizeof(fsgrids::ehallElement), cudaMemcpyHostToDevice);
      cudaMemcpy(d_egradpe, egradpe.data(),  egradpe.size() * sizeof(fsgrids::egradpeElement), cudaMemcpyHostToDevice);
      cudaMemcpy(d_egradpedt2, egradpedt2.data(),  egradpedt2.size() * sizeof(fsgrids::egradpeElement), cudaMemcpyHostToDevice);
      std::span<fsgrids::perbElement> dev_perb(d_perb, perb.size());
      std::span<fsgrids::perbElement> dev_perbdt2(d_perbdt2, perbdt2.size());
      std::span<fsgrids::efieldElement> dev_e(d_e, e.size());
      std::span<fsgrids::efieldElement> dev_edt2(d_edt2, edt2.size());
      std::span<fsgrids::bgbElement> dev_bgb(d_bgb, bgb.size());
      std::span<fsgrids::dmomentsElement> dev_dmoments(d_dmoments, dmoments.size());
      std::span<fsgrids::dperbElement> dev_dperb(d_dperb, dperb.size());
      std::span<fsgrids::momentsElement> dev_moments(d_moments, moments.size());
      std::span<fsgrids::dmomentsElement> dev_dmomentsdt2(d_dmomentsdt2, dmomentsdt2.size());
      std::span<fsgrids::momentsElement> dev_momentsdt2(d_momentsdt2, moments.size());
      std::span<fsgrids::ehallElement> dev_ehall(d_ehall, ehall.size());
      std::span<fsgrids::egradpeElement> dev_egradpe(d_egradpe, egradpe.size());
      std::span<fsgrids::egradpeElement> dev_egradpedt2(d_egradpedt2, egradpedt2.size());
#ifdef FS_1ST_ORDER_TIME
      propagateMagneticFieldSimple(dev_perb, dev_perbdt2, dev_bgb, dev_e, dev_edt2, dev_technical, fsgrid, dev_sysBoundaries, dt, RK_ORDER1);
      calculateDerivativesSimpleDevice(dev_perb, dev_moments, dev_dperb, dev_dmoments, dev_technical, fsgrid, true /*doMoments*/);
      if (P::ohmGradPeTerm > 0) {
         calculateGradPeTermSimple(egradpe, egradpedt2, moments, momentsdt2, dmoments, dmomentsdt2, technical, fsgrid, sysBoundaries, RK_ORDER1);
      }
      if (P::ohmHallTerm > 0) {
         calculateHallTermSimple(
            perb,
            perbdt2,
            ehall,
            moments,
            momentsdt2,
            dperb,
            dmoments,
            dmomentsdt2,
            bgb,
            technical,
            fsgrid,
            sysBoundaries,
            RK_ORDER1,
            true // communicateMomentsDerivatives
         );
      }
      calculateUpwindedElectricFieldSimpleDevice(
         dev_perb,
         dev_perbdt2,
         dev_e,
         dev_edt2,
         dev_ehall,
         dev_egradpe,
         dev_egradpedt2,
         dev_moments,
         dev_momentsdt2,
         dev_dperb,
         dev_dmoments,
         dev_dmomentsdt2,
         dev_bgb,
         dev_technical,
         fsgrid,
         dev_sysBoundaries,
         RK_ORDER1,
         true // communicateEGradPeOrMomentsDerivatives
      );
#else
      propagateMagneticFieldSimple(dev_perb, dev_perbdt2, dev_bgb, dev_e, dev_edt2, dev_technical, fsgrid, dev_sysBoundaries, dt, RK_ORDER2_STEP1);
      calculateDerivativesSimpleDevice(dev_perbdt2, dev_momentsdt2, dev_dperb, dev_dmomentsdt2, dev_technical, fsgrid, true /*doMoments*/);
      if (P::ohmGradPeTerm > 0) {
         calculateGradPeTermSimple(egradpe, egradpedt2, moments, momentsdt2, dmoments, dmomentsdt2, technical, fsgrid, sysBoundaries, RK_ORDER2_STEP1);
      }
      if (P::ohmHallTerm > 0) {
         calculateHallTermSimple(
            perb,
            perbdt2,
            ehall,
            moments,
            momentsdt2,
            dperb,
            dmoments,
            dmomentsdt2,
            bgb,
            technical,
            fsgrid,
            sysBoundaries,
            RK_ORDER2_STEP1,
            true // communicateMomentsDerivatives
         );
      }
      calculateUpwindedElectricFieldSimpleDevice(
         dev_perb,
         dev_perbdt2,
         dev_e,
         dev_edt2,
         dev_ehall,
         dev_egradpe,
         dev_egradpedt2,
         dev_moments,
         dev_momentsdt2,
         dev_dperb,
         dev_dmoments,
         dev_dmomentsdt2,
         dev_bgb,
         dev_technical,
         fsgrid,
         dev_sysBoundaries,
         RK_ORDER2_STEP1,
         true // communicateEGradPeOrMomentsDerivatives
      );

      propagateMagneticFieldSimple(dev_perb, dev_perbdt2, dev_bgb, dev_e, dev_edt2, dev_technical, fsgrid, dev_sysBoundaries, dt, RK_ORDER2_STEP2);
      calculateDerivativesSimpleDevice(dev_perb, dev_moments, dev_dperb, dev_dmoments, dev_technical, fsgrid, true /*doMoments*/);
      if (P::ohmGradPeTerm > 0) {
         calculateGradPeTermSimple(egradpe, egradpedt2, moments, momentsdt2, dmoments, dmomentsdt2, technical, fsgrid, sysBoundaries, RK_ORDER2_STEP2);
      }
      if (P::ohmHallTerm > 0) {
         calculateHallTermSimple(
            perb,
            perbdt2,
            ehall,
            moments,
            momentsdt2,
            dperb,
            dmoments,
            dmomentsdt2,
            bgb,
            technical,
            fsgrid,
            sysBoundaries,
            RK_ORDER2_STEP2,
            true // communicateMomentsDerivatives
         );
      }
      calculateUpwindedElectricFieldSimpleDevice(
         dev_perb,
         dev_perbdt2,
         dev_e,
         dev_edt2,
         dev_ehall,
         dev_egradpe,
         dev_egradpedt2,
         dev_moments,
         dev_momentsdt2,
         dev_dperb,
         dev_dmoments,
         dev_dmomentsdt2,
         dev_bgb,
         dev_technical,
         fsgrid,
         dev_sysBoundaries,
         RK_ORDER2_STEP2,
         true // communicateEGradPeOrMomentsDerivatives
      );
#endif
      cudaMemcpy(perb.data(), d_perb, perb.size() * sizeof(fsgrids::perbElement), cudaMemcpyDeviceToHost);
      cudaMemcpy(perbdt2.data(), d_perbdt2, perbdt2.size() * sizeof(fsgrids::perbElement), cudaMemcpyDeviceToHost);
      cudaMemcpy(dperb.data(), d_dperb, dperb.size() * sizeof(fsgrids::dperbElement), cudaMemcpyDeviceToHost);
      cudaMemcpy(dmoments.data(), d_dmoments, dmoments.size() * sizeof(fsgrids::dmomentsElement), cudaMemcpyDeviceToHost);
      cudaMemcpy(dmomentsdt2.data(), d_dmomentsdt2, dmomentsdt2.size() * sizeof(fsgrids::dmomentsElement), cudaMemcpyDeviceToHost);
      cudaMemcpy(technical.data(), d_technical, technical.size() * sizeof(fsgrids::technical), cudaMemcpyDeviceToHost);
   } else {
      Real subcycleDt = dt / convert<Real>(subcycles);
      Real subcycleT = P::t;
      creal targetT = P::t + dt;
      uint subcycleCount = 0;
      uint maxSubcycleCount = std::numeric_limits<uint>::max();
      int myRank = fsgrid.getRank();

      while (subcycleCount < maxSubcycleCount) {
         // In case of subcycling, we decided to go for a blunt Runge-Kutta subcycling even though e.g. moments are not
         // going along. Result of the Summer of Debugging 2016, the behaviour in wave dispersion was much improved with
         // this.
         {
            SysBoundaryDevice host_sysBoundaries;
            uploadParametersToDevice();
            uploadIndexToSysBoundary(sysBoundaries,  host_sysBoundaries);
            ALLOCATE_GPU(gpuMemoryManager, d_perb, perb.size() * sizeof(fsgrids::perbElement));
            ALLOCATE_GPU(gpuMemoryManager, d_perbdt2, perbdt2.size() * sizeof(fsgrids::perbElement));
            ALLOCATE_GPU(gpuMemoryManager, d_e, e.size() * sizeof(fsgrids::efieldElement));
            ALLOCATE_GPU(gpuMemoryManager, d_edt2, edt2.size() * sizeof(fsgrids::efieldElement));
            ALLOCATE_GPU(gpuMemoryManager, d_bgb, bgb.size() * sizeof(fsgrids::bgbElement));
            ALLOCATE_GPU(gpuMemoryManager, dev_sysBoundaries, sizeof(SysBoundaryDevice));
            ALLOCATE_GPU(gpuMemoryManager, d_technical, technical.size() * sizeof(fsgrids::technical));
            ALLOCATE_GPU(gpuMemoryManager, d_dmomentsdt2, dmomentsdt2.size() * sizeof(fsgrids::dmomentsElement));
            ALLOCATE_GPU(gpuMemoryManager, d_dperb, dperb.size() * sizeof(fsgrids::dperbElement));
            ALLOCATE_GPU(gpuMemoryManager, d_momentsdt2, momentsdt2.size() * sizeof(fsgrids::momentsElement));
            fsgrids::perbElement *d_perb = GET_POINTER(gpuMemoryManager, fsgrids::perbElement, d_perb);
            fsgrids::perbElement *d_perbdt2 = GET_POINTER(gpuMemoryManager, fsgrids::perbElement, d_perbdt2);
            fsgrids::efieldElement *d_e = GET_POINTER(gpuMemoryManager, fsgrids::efieldElement, d_e);
            fsgrids::efieldElement *d_edt2 = GET_POINTER(gpuMemoryManager, fsgrids::efieldElement, d_edt2);
            fsgrids::bgbElement *d_bgb = GET_POINTER(gpuMemoryManager, fsgrids::bgbElement, d_bgb);
            SysBoundaryDevice *dev_sysBoundaries = GET_POINTER(gpuMemoryManager, SysBoundaryDevice, dev_sysBoundaries);
            fsgrids::technical *d_technical = GET_POINTER(gpuMemoryManager, fsgrids::technical, d_technical);
            fsgrids::dmomentsElement *d_dmomentsdt2 = GET_POINTER(gpuMemoryManager, fsgrids::dmomentsElement, d_dmomentsdt2);
            fsgrids::dperbElement *d_dperb = GET_POINTER(gpuMemoryManager, fsgrids::dperbElement, d_dperb);
            fsgrids::momentsElement *d_momentsdt2 = GET_POINTER(gpuMemoryManager, fsgrids::momentsElement, d_momentsdt2);
            cudaMemcpy(d_perb, perb.data(),  perb.size() * sizeof(fsgrids::perbElement), cudaMemcpyHostToDevice);
            cudaMemcpy(d_perbdt2, perbdt2.data(),  perbdt2.size() * sizeof(fsgrids::perbElement), cudaMemcpyHostToDevice);
            cudaMemcpy(d_e, e.data(),  e.size() * sizeof(fsgrids::efieldElement), cudaMemcpyHostToDevice);
            cudaMemcpy(d_edt2, edt2.data(),  edt2.size() * sizeof(fsgrids::efieldElement), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bgb, bgb.data(),  bgb.size() * sizeof(fsgrids::bgbElement), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_sysBoundaries, &host_sysBoundaries,  sizeof(SysBoundaryDevice), cudaMemcpyHostToDevice);
            cudaMemcpy(d_technical, technical.data(), technical.size() * sizeof(fsgrids::technical), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dmomentsdt2, dmomentsdt2.data(),  dmomentsdt2.size() * sizeof(fsgrids::dmomentsElement), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dperb, dperb.data(),  dperb.size() * sizeof(fsgrids::dperbElement), cudaMemcpyHostToDevice);
            cudaMemcpy(d_momentsdt2, momentsdt2.data(),  momentsdt2.size() * sizeof(fsgrids::momentsElement), cudaMemcpyHostToDevice);
            std::span<fsgrids::perbElement> dev_perb(d_perb, perb.size());
            std::span<fsgrids::perbElement> dev_perbdt2(d_perbdt2, perbdt2.size());
            std::span<fsgrids::efieldElement> dev_e(d_e, e.size());
            std::span<fsgrids::efieldElement> dev_edt2(d_edt2, edt2.size());
            std::span<fsgrids::bgbElement> dev_bgb(d_bgb, bgb.size());
            std::span<fsgrids::technical> dev_technical(d_technical, technical.size());
            std::span<fsgrids::dmomentsElement> dev_dmomentsdt2(d_dmomentsdt2, dmomentsdt2.size());
            std::span<fsgrids::dperbElement> dev_dperb(d_dperb, dperb.size());
            std::span<fsgrids::momentsElement> dev_momentsdt2(d_momentsdt2, moments.size());
            propagateMagneticFieldSimple(dev_perb, dev_perbdt2, dev_bgb, dev_e, dev_edt2, dev_technical, fsgrid, dev_sysBoundaries, subcycleDt, RK_ORDER2_STEP1);
            // We need to calculate derivatives of the moments at every substep, but the moments only
            // need to be communicated in the first one.
            calculateDerivativesSimpleDevice(dev_perbdt2, dev_momentsdt2, dev_dperb, dev_dmomentsdt2, dev_technical, fsgrid, (subcycleCount == 0) /*doMoments*/);
            cudaMemcpy(perb.data(), d_perb, perb.size() * sizeof(fsgrids::perbElement), cudaMemcpyDeviceToHost);
            cudaMemcpy(perbdt2.data(), d_perbdt2, perbdt2.size() * sizeof(fsgrids::perbElement), cudaMemcpyDeviceToHost);
            cudaMemcpy(dperb.data(), d_dperb, dperb.size() * sizeof(fsgrids::dperbElement), cudaMemcpyDeviceToHost);
            cudaMemcpy(dmomentsdt2.data(), d_dmomentsdt2, dmomentsdt2.size() * sizeof(fsgrids::dmomentsElement), cudaMemcpyDeviceToHost);
         }

         if (P::ohmGradPeTerm > 0 && subcycleCount == 0) {
            calculateGradPeTermSimple(egradpe, egradpedt2, moments, momentsdt2, dmoments, dmomentsdt2, technical, fsgrid, sysBoundaries, RK_ORDER2_STEP1);
         }
         if (P::ohmHallTerm > 0) {
            calculateHallTermSimple(
               perb,
               perbdt2,
               ehall,
               moments,
               momentsdt2,
               dperb,
               dmoments,
               dmomentsdt2,
               bgb,
               technical,
               fsgrid,
               sysBoundaries,
               RK_ORDER2_STEP1,
               subcycleCount == 0 // communicateMomentsDerivatives
            );
         }
         calculateUpwindedElectricFieldSimple(
            perb,
            perbdt2,
            e,
            edt2,
            ehall,
            egradpe,
            egradpedt2,
            moments,
            momentsdt2,
            dperb,
            dmoments,
            dmomentsdt2,
            bgb,
            technical,
            fsgrid,
            sysBoundaries,
            RK_ORDER2_STEP1,
            subcycleCount == 0 // communicateEGradPeOrMomentsDerivatives
         );

         {
            SysBoundaryDevice host_sysBoundaries;
            uploadParametersToDevice();
            uploadIndexToSysBoundary(sysBoundaries,  host_sysBoundaries);
            ALLOCATE_GPU(gpuMemoryManager, d_perb, perb.size() * sizeof(fsgrids::perbElement));
            ALLOCATE_GPU(gpuMemoryManager, d_perbdt2, perbdt2.size() * sizeof(fsgrids::perbElement));
            ALLOCATE_GPU(gpuMemoryManager, d_e, e.size() * sizeof(fsgrids::efieldElement));
            ALLOCATE_GPU(gpuMemoryManager, d_edt2, edt2.size() * sizeof(fsgrids::efieldElement));
            ALLOCATE_GPU(gpuMemoryManager, d_bgb, bgb.size() * sizeof(fsgrids::bgbElement));
            ALLOCATE_GPU(gpuMemoryManager, dev_sysBoundaries, sizeof(SysBoundaryDevice));
            ALLOCATE_GPU(gpuMemoryManager, d_technical, technical.size() * sizeof(fsgrids::technical));
            ALLOCATE_GPU(gpuMemoryManager, d_dmoments, dmoments.size() * sizeof(fsgrids::dmomentsElement));
            ALLOCATE_GPU(gpuMemoryManager, d_dperb, dperb.size() * sizeof(fsgrids::dperbElement));
            ALLOCATE_GPU(gpuMemoryManager, d_moments, moments.size() * sizeof(fsgrids::momentsElement));
            fsgrids::perbElement *d_perb = GET_POINTER(gpuMemoryManager, fsgrids::perbElement, d_perb);
            fsgrids::perbElement *d_perbdt2 = GET_POINTER(gpuMemoryManager, fsgrids::perbElement, d_perbdt2);
            fsgrids::efieldElement *d_e = GET_POINTER(gpuMemoryManager, fsgrids::efieldElement, d_e);
            fsgrids::efieldElement *d_edt2 = GET_POINTER(gpuMemoryManager, fsgrids::efieldElement, d_edt2);
            fsgrids::bgbElement *d_bgb = GET_POINTER(gpuMemoryManager, fsgrids::bgbElement, d_bgb);
            SysBoundaryDevice *dev_sysBoundaries = GET_POINTER(gpuMemoryManager, SysBoundaryDevice, dev_sysBoundaries);
            fsgrids::technical *d_technical = GET_POINTER(gpuMemoryManager, fsgrids::technical, d_technical);
            fsgrids::dmomentsElement *d_dmoments = GET_POINTER(gpuMemoryManager, fsgrids::dmomentsElement, d_dmoments);
            fsgrids::dperbElement *d_dperb = GET_POINTER(gpuMemoryManager, fsgrids::dperbElement, d_dperb);
            fsgrids::momentsElement *d_moments = GET_POINTER(gpuMemoryManager, fsgrids::momentsElement, d_moments);
            cudaMemcpy(d_perb, perb.data(),  perb.size() * sizeof(fsgrids::perbElement), cudaMemcpyHostToDevice);
            cudaMemcpy(d_perbdt2, perbdt2.data(),  perbdt2.size() * sizeof(fsgrids::perbElement), cudaMemcpyHostToDevice);
            cudaMemcpy(d_e, e.data(),  e.size() * sizeof(fsgrids::efieldElement), cudaMemcpyHostToDevice);
            cudaMemcpy(d_edt2, edt2.data(),  edt2.size() * sizeof(fsgrids::efieldElement), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bgb, bgb.data(),  bgb.size() * sizeof(fsgrids::bgbElement), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_sysBoundaries, &host_sysBoundaries,  sizeof(SysBoundaryDevice), cudaMemcpyHostToDevice);
            cudaMemcpy(d_technical, technical.data(), technical.size() * sizeof(fsgrids::technical), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dmoments, dmoments.data(),  dmoments.size() * sizeof(fsgrids::dmomentsElement), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dperb, dperb.data(),  dperb.size() * sizeof(fsgrids::dperbElement), cudaMemcpyHostToDevice);
            cudaMemcpy(d_moments, moments.data(),  moments.size() * sizeof(fsgrids::momentsElement), cudaMemcpyHostToDevice);
            std::span<fsgrids::perbElement> dev_perb(d_perb, perb.size());
            std::span<fsgrids::perbElement> dev_perbdt2(d_perbdt2, perbdt2.size());
            std::span<fsgrids::efieldElement> dev_e(d_e, e.size());
            std::span<fsgrids::efieldElement> dev_edt2(d_edt2, edt2.size());
            std::span<fsgrids::bgbElement> dev_bgb(d_bgb, bgb.size());
            std::span<fsgrids::technical> dev_technical(d_technical, technical.size());
            std::span<fsgrids::dmomentsElement> dev_dmoments(d_dmoments, dmoments.size());
            std::span<fsgrids::dperbElement> dev_dperb(d_dperb, dperb.size());
            std::span<fsgrids::momentsElement> dev_moments(d_moments, moments.size());
            propagateMagneticFieldSimple(dev_perb, dev_perbdt2, dev_bgb, dev_e, dev_edt2, dev_technical, fsgrid, dev_sysBoundaries, subcycleDt, RK_ORDER2_STEP2);
            // We need to calculate derivatives of the moments at every substep, but the moments only
            // need to be communicated in the first one.
            calculateDerivativesSimpleDevice(dev_perb, dev_moments, dev_dperb, dev_dmoments, dev_technical, fsgrid, (subcycleCount == 0) /*doMoments*/);
            cudaMemcpy(perb.data(), d_perb, perb.size() * sizeof(fsgrids::perbElement), cudaMemcpyDeviceToHost);
            cudaMemcpy(perbdt2.data(), d_perbdt2, perbdt2.size() * sizeof(fsgrids::perbElement), cudaMemcpyDeviceToHost);
            cudaMemcpy(dperb.data(), d_dperb, dperb.size() * sizeof(fsgrids::dperbElement), cudaMemcpyDeviceToHost);
            cudaMemcpy(dmoments.data(), d_dmoments, dmoments.size() * sizeof(fsgrids::dmomentsElement), cudaMemcpyDeviceToHost);
         }
         if (P::ohmGradPeTerm > 0 && subcycleCount == 0) {
            calculateGradPeTermSimple(egradpe, egradpedt2, moments, momentsdt2, dmoments, dmomentsdt2, technical, fsgrid, sysBoundaries, RK_ORDER2_STEP2);
         }
         if (P::ohmHallTerm > 0) {
            calculateHallTermSimple(
               perb,
               perbdt2,
               ehall,
               moments,
               momentsdt2,
               dperb,
               dmoments,
               dmomentsdt2,
               bgb,
               technical,
               fsgrid,
               sysBoundaries,
               RK_ORDER2_STEP2,
               subcycleCount == 0 // communicateMomentsDerivatives
            );
         }
         calculateUpwindedElectricFieldSimple(
            perb,
            perbdt2,
            e,
            edt2,
            ehall,
            egradpe,
            egradpedt2,
            moments,
            momentsdt2,
            dperb,
            dmoments,
            dmomentsdt2,
            bgb,
            technical,
            fsgrid,
            sysBoundaries,
            RK_ORDER2_STEP2,
            subcycleCount == 0 // communicateEGradPeOrMomentsDerivatives
         );

         phiprof::Timer subcyclingTimer{"FS subcycle stuff"};
         subcycleT += subcycleDt;
         subcycleCount++;

         if (subcycleT >= targetT || subcycleCount >= maxSubcycleCount) {
            // we are done
            if (subcycleT > targetT) {
               // due to roundoff we might hit this, should add delta
               std::cerr << "subcycleT > targetT, should not happen! (values: subcycleT " << subcycleT << ", subcycleDt " << subcycleDt << ", targetT " << targetT << ")" << std::endl;
            }
            break;
         }

         // Reassess subcycle dt
         Real dtMaxGlobal = 0.0;
         Real dtMaxLocal = fsgrid.parallel_reduction([](int timerId) ->  phiprof::Timer { return phiprof::Timer{timerId}; },
                                                   phiprof::initializeTimer("compute-subcycle-dt-reduction-loop"), technical,
                                                   [](Real a, Real b) { return std::min<Real>(a, b); },
                                                   std::numeric_limits<Real>::max(),
                                                   [=](const fsgrid::Coordinates &coordinates, const fsgrid::FsStencil& stencil, cuint sysBoundaryFlag, cuint sysBoundaryLayer, creal maximum) {
            if (sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY ||
               (sysBoundaryLayer == 1 && sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY)) {
               return technical[stencil.ooo()].maxFsDt;
            } else {
               return maximum;
            }
         });

         phiprof::Timer allreduceTimer{"MPI_Allreduce"};
         fsgrid.Allreduce(&(dtMaxLocal), &(dtMaxGlobal), 1, MPI_Type<Real>(), MPI_MIN);
         allreduceTimer.stop();

         // reduce dt if it is too high
         if (subcycleDt > dtMaxGlobal * P::fieldSolverMaxCFL) {
            creal meanFieldsCFL = 0.5 * (P::fieldSolverMaxCFL + P::fieldSolverMinCFL);
            subcycleDt = meanFieldsCFL * dtMaxGlobal;
            if (myRank == MASTER_RANK) {
               logFile << "(TIMESTEP) New field solver subcycle dt = " << subcycleDt << " computed on step " << P::tstep << " and substep " << subcycleCount << " at " << P::t << " s" << std::endl;
            }
         }

         // Readjust the dt to hit targetT. Try to avoid having a very
         // short delta step at the end, instead 2 more normal ones
         if (subcycleT + 1.5 * subcycleDt > targetT) {
            subcycleDt = targetT - subcycleT;
            maxSubcycleCount = subcycleCount + 1; // 1 more steps
            // check that subcyclDt has correct CFL, take 2 if not
            if (subcycleDt > dtMaxGlobal * P::fieldSolverMaxCFL) {
               subcycleDt = (targetT - subcycleT) / 2;
               maxSubcycleCount = subcycleCount + 2;
            }
         }

         subcyclingTimer.stop();
      }

      if (subcycles != subcycleCount && myRank == MASTER_RANK) {
         logFile << "Effective field solver subcycles were " << subcycleCount << " instead of " << P::fieldSolverSubcycles << " on step " << P::tstep << std::endl;
      }
   }

   calculateVolumeAveragedFieldsSimple(perb, e, dperb, vol, technical, fsgrid);
   calculateBVOLDerivativesSimple(vol, technical, fsgrid);
   if (FieldTracing::fieldTracingParameters.doTraceFullBox || Parameters::computeCurvature) {
      fsgrid.updateGhostCells(vol);
      calculateCurvatureSimple(vol, bgb, technical, fsgrid);
   }
   return true;
}
