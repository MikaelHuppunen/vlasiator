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

#ifndef COPYSPHERE_H
#define COPYSPHERE_H

#include <vector>
#include "../definitions.h"
#include "../readparameters.h"
#include "../spatial_cells/spatial_cell_wrapper.hpp"
#include "sysboundarycondition.h"

using namespace projects;
using namespace std;

namespace SBC {

   struct CopysphereSpeciesParameters {
      Real rho;
      Real V0[3];
      Real T;
      Real fluffiness;
   };

   /*!\brief Copysphere is a class applying an ionosphere-ish boundary conditions.
    *
    * Copysphere is a class handling cells tagged as sysboundarytype::COPYSPHERE by this system boundary condition. It applies copy boundary conditions to perturbed magnetic field.
    *
    * These consist in:
    * - Do nothing for the distribution (keep the initial state constant in time);
    * - Copy the closest neighbors' perturbed B and average it;
    * - Null out the electric fields.
    *
    * For 3D magnetospheric simulations, you might be interesting in trying the ionosphere boundary instead!
    */
   class Copysphere: public SysBoundaryCondition {
   public:
      Copysphere();
      virtual ~Copysphere();
      
      void addParameters();
      virtual void getParameters() override;

      virtual void initSysBoundary(creal& t, Project& project) override;
      virtual void assignSysBoundary(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                     fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) override;
      virtual void applyInitialState(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                     fsgrids::technicalspan technical, FieldSolverGrid& fsgrid,
                                     fsgrids::perbspan perb,
                                     fsgrids::bgbspan bgb,
                                     Project& project) override;
      virtual void updateState(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                               fsgrids::technicalspan technical, FieldSolverGrid& fsgrid,
                               fsgrids::perbspan perb,
                               fsgrids::bgbspan bgb, creal t) override;
      virtual Real fieldSolverBoundaryCondMagneticField(fsgrids::perbspan b,
                                                        fsgrids::constbgbspan bgb,
                                                        fsgrids::consttechnicalspan technical,
                                                        const std::array<Real, 3>& gridSpacing,
                                                        const std::array<fsgrid::FsSize_t, 3>& globalCoordinates,
                                                        const fsgrid::FsStencil& stencil, cuint component) override;
      virtual void fieldSolverBoundaryCondElectricField(fsgrids::efieldspan e,
                                                        const fsgrid::FsStencil& stencil, cuint component) override;
      virtual void fieldSolverBoundaryCondHallElectricField(fsgrids::ehallspan ehall,
                                                            const fsgrid::FsStencil& stencil, cuint component) override;
      virtual void
      fieldSolverBoundaryCondGradPeElectricField(fsgrids::egradpespan EGradPe,
                                                 const fsgrid::FsStencil& stencil, cuint component) override;
      virtual void
      fieldSolverBoundaryCondDerivatives(fsgrids::dperbspan dperb,
                                         fsgrids::dmomentsspan dmoments,
                                         const fsgrid::FsStencil& stencil, cuint RKCase, cuint component) override;
      virtual void fieldSolverBoundaryCondBVOLDerivatives(fsgrids::volspan vols,
                                                          const fsgrid::FsStencil& stencil, cuint component) override;
      virtual void vlasovBoundaryCondition(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                           const CellID& cellID, const uint popID,
                                           const bool calculate_V_moments) override;

      void getFaces(bool* faces) override;
      virtual std::string getName() const override;
      virtual uint getIndex() const override;

      void generateTemplateCell(Project &project);
      void setCellFromTemplate(SpatialCell* cell,const uint popID);

      std::array<Real, 3> fieldSolverGetNormalDirection(
         fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
         cint i,
         cint j,
         cint k
      );

      Real center[3]; /*!< Coordinates of the centre of the copy sphere. */
      Real radius; /*!< Radius of the copy sphere. */
      uint geometry; /*!< Geometry of the copy sphere, 0: inf-norm (diamond), 1: 1-norm (square), 2: 2-norm (circle, DEFAULT), 3: polar-plane cylinder with line dipole. */

      static std::vector<CopysphereSpeciesParameters> speciesParams;
      static std::vector<CopysphereSpeciesParameters*> speciesParamsRead;
      bool zeroPerB;

      spatial_cell::SpatialCell templateCell;

   };

   class CopysphereDevice: public SysBoundaryConditionDevice {
   public:
      __device__ CopysphereDevice(const bool* periodic_input, bool zeroPerB_input) : SysBoundaryConditionDevice(periodic_input) {
         zeroPerB = zeroPerB_input;
      };

      virtual ~CopysphereDevice() = default;

      __device__ inline Real fieldSolverBoundaryCondMagneticField(
         fsgrids::perbspan b, fsgrids::constbgbspan bgb, fsgrids::consttechnicalspan technical, const std::array<Real, 3>& gridSpacing,
         const std::array<fsgrid::FsSize_t, 3>& globalCoordinates, const fsgrid::FsStencil& stencil, cuint component, const DeviceParameters& parameters)
      {
         const uint32_t perbComponent = fsgrids::bfield::PERBX + component;
         const uint32_t bitfield = 1 << component;

         // clang-format off
         static constexpr std::array permutations = {
            std::array {
               0, 1, 2, 3, 4, 5,
            },
            std::array {
               2, 3, 0, 1, 4, 5,
            },
            std::array {
               4, 5, 0, 1, 2, 3,
            },
         };

         const std::array permutation = permutations[component];

         const std::array<size_t, 6> inds = {
            stencil.moo(),
            stencil.poo(),
            stencil.omo(),
            stencil.opo(),
            stencil.oom(),
            stencil.oop(),
         };
         // clang-format on

         auto bitFieldSet = [&bitfield](auto& tech) { return (tech.SOLVE & bitfield) == bitfield; };
         auto sbLayerIsOne = [](auto& tech) { return tech.sysBoundaryLayer == 1; };
         auto averageNeigbours = [&technical, &b, &inds, &permutation, &perbComponent,
                                 &bitFieldSet](auto begin, auto end, auto& sum, auto& nCells) {
            for (size_t i = begin; i < end; i++) {
               const auto j = inds[permutation[i]];
               if (bitFieldSet(technical[j])) {
                  sum += b[j][perbComponent];
                  nCells++;
               }
            }
         };

         auto averageAllNeighbours = [&stencil, &technical, &b, &perbComponent](auto predicateLambda, auto& sum,
                                                                              auto& nCells) {
            for (const auto& i : stencil.indices()) {
               if (predicateLambda(technical[i])) {
                  sum += b[i][perbComponent];
                  nCells++;
               }
            }
         };

         Real sum = 0.0;
         uint nCells = 0;
         if (this->zeroPerB == true) {
            sum = b[stencil.ooo()][perbComponent];
            nCells = 1;
         } else {
            if (sbLayerIsOne(technical[stencil.ooo()])) {
               averageNeigbours(0ul, 2ul, sum, nCells);

               if (nCells == 0) {
                  averageNeigbours(2ul, 6ul, sum, nCells);
               }

               if (nCells == 0) {
                  averageAllNeighbours(bitFieldSet, sum, nCells);
               }
            } else {
               // L2 cells
               averageAllNeighbours(sbLayerIsOne, sum, nCells);
            }
         }

         if (nCells == 0) {
            printf(": ERROR: this should not have fallen through.\n");
            sum = 0.0;
            nCells = 1;
         }

         return sum / nCells;
      }

      __device__ inline void fieldSolverBoundaryCondElectricField(
         fsgrids::efieldspan e, const fsgrid::FsStencil& stencil, cuint component)
      {
         e[stencil.ooo()][fsgrids::efield::EX + component] = 0.0;
      }

      __device__ inline void fieldSolverBoundaryCondGradPeElectricField(
         fsgrids::egradpespan EGradPe, const fsgrid::FsStencil& stencil, cuint component)
      {
         EGradPe[stencil.ooo()][fsgrids::egradpe::EXGRADPE + component] = 0.0;
      }

      __device__ inline void fieldSolverBoundaryCondHallElectricField(
         fsgrids::ehallspan ehall, const fsgrid::FsStencil& stencil, cuint component)
      {
         std::array<Real, fsgrids::ehall::N_EHALL>& cp = ehall[stencil.ooo()];
         switch (component) {
         case 0:
            cp[fsgrids::ehall::EXHALL_000_100] = 0.0;
            cp[fsgrids::ehall::EXHALL_010_110] = 0.0;
            cp[fsgrids::ehall::EXHALL_001_101] = 0.0;
            cp[fsgrids::ehall::EXHALL_011_111] = 0.0;
            break;
         case 1:
            cp[fsgrids::ehall::EYHALL_000_010] = 0.0;
            cp[fsgrids::ehall::EYHALL_100_110] = 0.0;
            cp[fsgrids::ehall::EYHALL_001_011] = 0.0;
            cp[fsgrids::ehall::EYHALL_101_111] = 0.0;
            break;
         case 2:
            cp[fsgrids::ehall::EZHALL_000_001] = 0.0;
            cp[fsgrids::ehall::EZHALL_100_101] = 0.0;
            cp[fsgrids::ehall::EZHALL_010_011] = 0.0;
            cp[fsgrids::ehall::EZHALL_110_111] = 0.0;
            break;
         default:
            printf("Invalid component\n");
         }
      }

      bool zeroPerB;
   };
}

#endif
