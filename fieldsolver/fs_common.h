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

#ifndef FS_COMMON_H
#define FS_COMMON_H

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <list>
#include <set>
#include <stdint.h>

#include <fsgrid.hpp>
#include <phiprof.hpp>

#include "../definitions.h"
#include "../common.h"
#include "../parameters.h"
#include "../projects/project.h"
#include "../sysboundary/sysboundary.h"
#include "../sysboundary/sysboundarycondition.h"

// Constants: not needed as such, but if field solver is implemented on GPUs
// these force CPU to use float accuracy, which in turn helps to compare
// CPU and GPU results.

const Real HALF    = 0.5;
const Real MINUS   = -1.0;
const Real PLUS    = +1.0;
const Real THIRD   = 1.0/3.0;
const Real FOURTH  = 1.0/4.0;
const Real SIXTH   = 1.0/6.0;
const Real EIGTH   = 1.0/8.0;
const Real TENTH   = 1.0/10.0;
const Real TWELWTH = 1.0/12.0;
const Real TWO     = 2.0;
const Real ZERO    = 0.0;

static creal EPS = 1.0e-30;

using namespace std;

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
                     fsgrids::technicalspan technical, FieldSolverGrid &fsgrid, SysBoundary& sysBoundaries,
                     creal& dt, cuint subcycles);

Real divideIfNonZero(creal rhoV, creal rho);

/*! \brief Helper function
 *
 * Divides the first value by the second or returns zero if the denominator is zero.
 *
 * \param numerator Numerator
 * \param denominator Denominator
 */
__device__ inline Real divideIfNonZeroDevice(creal numerator, creal denominator) {
   if (denominator <= 0.0) {
      return 0.0;
   } else {
      return numerator / denominator;
   }
}

/*! Namespace encompassing the enum defining the list of reconstruction coefficients used in field component reconstructions.*/
namespace Rec {
   /*! Enum defining the list of reconstruction coefficients used in field component reconstructions.*/
   enum Rec {
      a_0, a_x, a_y, a_z, a_xx, a_yy, a_zz, a_xy, a_xz, a_yz, a_xxx, a_xxy, a_xyy, a_xxz, a_xzz, a_xyz,
      b_0, b_x, b_y, b_z, b_xx, b_yy, b_zz, b_xy, b_xz, b_yz, b_xxy, b_xyy, b_yyy, b_yyz, b_yzz, b_xyz,
      c_0, c_x, c_y, c_z, c_xx, c_yy, c_zz, c_xy, c_xz, c_yz, c_xxz, c_xzz, c_yyz, c_yzz, c_xyz, c_zzz,
      N_REC_COEFFICIENTS
   };
}

std::array<Real, Rec::N_REC_COEFFICIENTS>
reconstructionCoefficients(fsgrids::perbspan perb,
                           fsgrids::constdperbspan dperb,
                           const fsgrid::FsStencil& stencil, Real reconstructionOrder);

/*! \brief Low-level helper function.
 *
 * Computes the reconstruction coefficients used for field component reconstruction.
 * Only implemented for 2nd and 3rd order.
 *
 * \param perb fsGrid holding the perturbed B quantities
 * \param dperb fsGrid holding the derivatives of perturbed B
 * \param stencil fsgrid stencil for the current cell
 * \param reconstructionOrder Reconstruction order of the fields after Balsara 2009, 2 used for BVOL, 3 used for
 * 2nd-order Hall term calculations.
 */
__device__ inline std::array<Real, Rec::N_REC_COEFFICIENTS>
reconstructionCoefficientsDevice(fsgrids::perbspan perb,
                           fsgrids::constdperbspan dperb,
                           const fsgrid::FsStencil& stencil, Real reconstructionOrder) {
   std::array<Real, Rec::N_REC_COEFFICIENTS> perturbedResult;
   const auto ooo = stencil.ooo();
   const auto poo = stencil.poo();
   const auto opo = stencil.opo();
   const auto oop = stencil.oop();
   const std::array<Real, fsgrids::dperb::N_DPERB>& der_i1j1k1 = dperb[ooo];
   const std::array<Real, fsgrids::bfield::N_BFIELD>& cep_i1j1k1 = perb[ooo];

   const bool pooOk = stencil.cellExists(1, 0, 0);
   const bool opoOk = stencil.cellExists(0, 1, 0);
   const bool oopOk = stencil.cellExists(0, 0, 1);
   const std::array<Real, fsgrids::bfield::N_BFIELD>& cep_i2j1k1 = pooOk ? perb[poo] : cep_i1j1k1;
   const std::array<Real, fsgrids::bfield::N_BFIELD>& cep_i1j2k1 = opoOk ? perb[opo] : cep_i1j1k1;
   const std::array<Real, fsgrids::bfield::N_BFIELD>& cep_i1j1k2 = oopOk ? perb[oop] : cep_i1j1k1;

#ifndef FS_1ST_ORDER_SPACE
   // Create a dummy array for containing zero values for derivatives on non-existing cells:
   std::array<Real, fsgrids::dperb::N_DPERB> dummyDerivatives;
   dummyDerivatives.fill(0.0);

   // Fetch neighbour cell derivatives, or in case the neighbour does not
   // exist, use dummyDerivatives array:
   const std::array<Real, fsgrids::dperb::N_DPERB>& der_i2j1k1 = pooOk ? dperb[poo] : dummyDerivatives;
   const std::array<Real, fsgrids::dperb::N_DPERB>& der_i1j2k1 = opoOk ? dperb[opo] : dummyDerivatives;
   const std::array<Real, fsgrids::dperb::N_DPERB>& der_i1j1k2 = oopOk ? dperb[oop] : dummyDerivatives;

   // Calculate 3rd order reconstruction coefficients:
   if (reconstructionOrder == 2) {
      perturbedResult[Rec::a_yy] = 0.0;
      perturbedResult[Rec::a_zz] = 0.0;
      perturbedResult[Rec::a_yz] = 0.0;
      perturbedResult[Rec::a_xxx] = 0.0;
      perturbedResult[Rec::a_xxy] = 0.0;
      perturbedResult[Rec::a_xxz] = 0.0;
      perturbedResult[Rec::a_xyy] = 0.0;
      perturbedResult[Rec::a_xyz] = 0.0;
      perturbedResult[Rec::a_xzz] = 0.0;
      perturbedResult[Rec::b_xx] = 0.0;
      perturbedResult[Rec::b_xz] = 0.0;
      perturbedResult[Rec::b_zz] = 0.0;
      perturbedResult[Rec::b_xxy] = 0.0;
      perturbedResult[Rec::b_xyy] = 0.0;
      perturbedResult[Rec::b_xyz] = 0.0;
      perturbedResult[Rec::b_yyy] = 0.0;
      perturbedResult[Rec::b_yyz] = 0.0;
      perturbedResult[Rec::b_yzz] = 0.0;
      perturbedResult[Rec::c_xx] = 0.0;
      perturbedResult[Rec::c_xy] = 0.0;
      perturbedResult[Rec::c_yy] = 0.0;
      perturbedResult[Rec::c_xxz] = 0.0;
      perturbedResult[Rec::c_xyz] = 0.0;
      perturbedResult[Rec::c_xzz] = 0.0;
      perturbedResult[Rec::c_yyz] = 0.0;
      perturbedResult[Rec::c_yzz] = 0.0;
      perturbedResult[Rec::c_zzz] = 0.0;
   } else if (reconstructionOrder == 3) {
      perturbedResult[Rec::a_yy] = HALF*(der_i2j1k1[fsgrids::dperb::dPERBxdyy] + der_i1j1k1[fsgrids::dperb::dPERBxdyy]);
      perturbedResult[Rec::a_zz] = HALF*(der_i2j1k1[fsgrids::dperb::dPERBxdzz] + der_i1j1k1[fsgrids::dperb::dPERBxdzz]);
      perturbedResult[Rec::a_yz] = HALF*(der_i2j1k1[fsgrids::dperb::dPERBxdyz] + der_i1j1k1[fsgrids::dperb::dPERBxdyz]);
      perturbedResult[Rec::a_xyy] = (der_i2j1k1[fsgrids::dperb::dPERBxdyy] - der_i1j1k1[fsgrids::dperb::dPERBxdyy]);
      perturbedResult[Rec::a_xyz] = (der_i2j1k1[fsgrids::dperb::dPERBxdyz] - der_i1j1k1[fsgrids::dperb::dPERBxdyz]);
      perturbedResult[Rec::a_xzz] = (der_i2j1k1[fsgrids::dperb::dPERBxdzz] - der_i1j1k1[fsgrids::dperb::dPERBxdzz]);

      perturbedResult[Rec::b_xx] = HALF*(der_i1j2k1[fsgrids::dperb::dPERBydxx] + der_i1j1k1[fsgrids::dperb::dPERBydxx]);
      perturbedResult[Rec::b_xz] = HALF*(der_i1j2k1[fsgrids::dperb::dPERBydxz] + der_i1j1k1[fsgrids::dperb::dPERBydxz]);
      perturbedResult[Rec::b_zz] = HALF*(der_i1j2k1[fsgrids::dperb::dPERBydzz] + der_i1j1k1[fsgrids::dperb::dPERBydzz]);
      perturbedResult[Rec::b_xxy] = (der_i1j2k1[fsgrids::dperb::dPERBydxx] - der_i1j1k1[fsgrids::dperb::dPERBydxx]);
      perturbedResult[Rec::b_xyz] = (der_i1j2k1[fsgrids::dperb::dPERBydxz] - der_i1j1k1[fsgrids::dperb::dPERBydxz]);
      perturbedResult[Rec::b_yzz] = (der_i1j2k1[fsgrids::dperb::dPERBydzz] - der_i1j1k1[fsgrids::dperb::dPERBydzz]);

      perturbedResult[Rec::c_xx] = HALF*(der_i1j1k2[fsgrids::dperb::dPERBzdxx] + der_i1j1k1[fsgrids::dperb::dPERBzdxx]);
      perturbedResult[Rec::c_xy] = HALF*(der_i1j1k2[fsgrids::dperb::dPERBzdxy] + der_i1j1k1[fsgrids::dperb::dPERBzdxy]);
      perturbedResult[Rec::c_yy] = HALF*(der_i1j1k2[fsgrids::dperb::dPERBzdyy] + der_i1j1k1[fsgrids::dperb::dPERBzdyy]);
      perturbedResult[Rec::c_xxz] = (der_i1j1k2[fsgrids::dperb::dPERBzdxx] - der_i1j1k1[fsgrids::dperb::dPERBzdxx]);
      perturbedResult[Rec::c_xyz] = (der_i1j1k2[fsgrids::dperb::dPERBzdxy] - der_i1j1k1[fsgrids::dperb::dPERBzdxy]);
      perturbedResult[Rec::c_yyz] = (der_i1j1k2[fsgrids::dperb::dPERBzdyy] - der_i1j1k1[fsgrids::dperb::dPERBzdyy]);

      perturbedResult[Rec::a_xxx] = -THIRD*(perturbedResult[Rec::b_xxy] + perturbedResult[Rec::c_xxz]);
      perturbedResult[Rec::a_xxy] = -FOURTH*perturbedResult[Rec::c_xyz];
      perturbedResult[Rec::a_xxz] = -FOURTH*perturbedResult[Rec::b_xyz];

      perturbedResult[Rec::b_xyy] = -FOURTH*perturbedResult[Rec::c_xyz];
      perturbedResult[Rec::b_yyy] = -THIRD*(perturbedResult[Rec::c_yyz] + perturbedResult[Rec::a_xyy]);
      perturbedResult[Rec::b_yyz] = -FOURTH*perturbedResult[Rec::a_xyz];

      perturbedResult[Rec::c_xzz] = -FOURTH*perturbedResult[Rec::b_xyz];
      perturbedResult[Rec::c_yzz] = -FOURTH*perturbedResult[Rec::a_xyz];
      perturbedResult[Rec::c_zzz] = -THIRD*(perturbedResult[Rec::a_xzz] + perturbedResult[Rec::b_yzz]);
   } else {
      assert(false);
      //cerr << __FILE__ << ":" << __LINE__ << ":" << " Not coded yet!" << endl;
      //abort();
   }

   // Calculate 2nd order reconstruction coefficients:
   perturbedResult[Rec::a_xy] = der_i2j1k1[fsgrids::dperb::dPERBxdy] - der_i1j1k1[fsgrids::dperb::dPERBxdy];
   perturbedResult[Rec::a_xz] = der_i2j1k1[fsgrids::dperb::dPERBxdz] - der_i1j1k1[fsgrids::dperb::dPERBxdz];
   perturbedResult[Rec::a_y] = HALF*(der_i2j1k1[fsgrids::dperb::dPERBxdy] + der_i1j1k1[fsgrids::dperb::dPERBxdy]) - SIXTH*perturbedResult[Rec::a_xxy];
   perturbedResult[Rec::a_z] = HALF*(der_i2j1k1[fsgrids::dperb::dPERBxdz] + der_i1j1k1[fsgrids::dperb::dPERBxdz]) - SIXTH*perturbedResult[Rec::a_xxz];

   perturbedResult[Rec::b_xy] = der_i1j2k1[fsgrids::dperb::dPERBydx] - der_i1j1k1[fsgrids::dperb::dPERBydx];
   perturbedResult[Rec::b_yz] = der_i1j2k1[fsgrids::dperb::dPERBydz] - der_i1j1k1[fsgrids::dperb::dPERBydz];
   perturbedResult[Rec::b_x] = HALF*(der_i1j2k1[fsgrids::dperb::dPERBydx] + der_i1j1k1[fsgrids::dperb::dPERBydx]) - SIXTH*perturbedResult[Rec::b_xyy];
   perturbedResult[Rec::b_z] = HALF*(der_i1j2k1[fsgrids::dperb::dPERBydz] + der_i1j1k1[fsgrids::dperb::dPERBydz]) - SIXTH*perturbedResult[Rec::b_yyz];

   perturbedResult[Rec::c_xz] = der_i1j1k2[fsgrids::dperb::dPERBzdx] - der_i1j1k1[fsgrids::dperb::dPERBzdx];
   perturbedResult[Rec::c_yz] = der_i1j1k2[fsgrids::dperb::dPERBzdy] - der_i1j1k1[fsgrids::dperb::dPERBzdy];
   perturbedResult[Rec::c_x] = HALF*(der_i1j1k2[fsgrids::dperb::dPERBzdx] + der_i1j1k1[fsgrids::dperb::dPERBzdx]) - SIXTH*perturbedResult[Rec::c_xzz];
   perturbedResult[Rec::c_y] = HALF*(der_i1j1k2[fsgrids::dperb::dPERBzdy] + der_i1j1k1[fsgrids::dperb::dPERBzdy]) - SIXTH*perturbedResult[Rec::c_yzz];

   perturbedResult[Rec::a_xx] = -HALF*(perturbedResult[Rec::b_xy] + perturbedResult[Rec::c_xz]);
   perturbedResult[Rec::b_yy] = -HALF*(perturbedResult[Rec::a_xy] + perturbedResult[Rec::c_yz]);
   perturbedResult[Rec::c_zz] = -HALF*(perturbedResult[Rec::a_xz] + perturbedResult[Rec::b_yz]);

   perturbedResult[Rec::a_x] = cep_i2j1k1[fsgrids::bfield::PERBX] - cep_i1j1k1[fsgrids::bfield::PERBX] - TENTH*perturbedResult[Rec::a_xxx];
   perturbedResult[Rec::b_y] = cep_i1j2k1[fsgrids::bfield::PERBY] - cep_i1j1k1[fsgrids::bfield::PERBY] - TENTH*perturbedResult[Rec::b_yyy];
   perturbedResult[Rec::c_z] = cep_i1j1k2[fsgrids::bfield::PERBZ] - cep_i1j1k1[fsgrids::bfield::PERBZ] - TENTH*perturbedResult[Rec::c_zzz];

#else
   perturbedResult.fill(0.0);
#endif

   // Calculate 1st order reconstruction coefficients:
   perturbedResult[Rec::a_0] = HALF*(cep_i2j1k1[fsgrids::bfield::PERBX] + cep_i1j1k1[fsgrids::bfield::PERBX]) - SIXTH*perturbedResult[Rec::a_xx];
   perturbedResult[Rec::b_0] = HALF*(cep_i1j2k1[fsgrids::bfield::PERBY] + cep_i1j1k1[fsgrids::bfield::PERBY]) - SIXTH*perturbedResult[Rec::b_yy];
   perturbedResult[Rec::c_0] = HALF*(cep_i1j1k2[fsgrids::bfield::PERBZ] + cep_i1j1k1[fsgrids::bfield::PERBZ]) - SIXTH*perturbedResult[Rec::c_zz];

   return perturbedResult;
}

std::array<Real, 3> interpolatePerturbedB(
   fsgrids::perbspan perb,
   fsgrids::constdperbspan dperb,
   fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
   std::map<std::array<int, 3>, std::array<Real, Rec::N_REC_COEFFICIENTS>>& reconstructionCoefficientsCache,
   cint i,
   cint j,
   cint k,
   const std::array<Real, 3> x
);

std::array<Real, 3> interpolateCurlB(
   fsgrids::perbspan perb,
   fsgrids::constdperbspan dperb,
   fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
   std::map<std::array<int, 3>, std::array<Real, Rec::N_REC_COEFFICIENTS>>& reconstructionCoefficientsCache,
   cint i,
   cint j,
   cint k,
   const std::array<Real, 3> x
);

#endif
