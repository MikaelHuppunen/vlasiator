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

#ifndef SYSBOUNDARYCONDITION_H
#define SYSBOUNDARYCONDITION_H

#include <dccrg.hpp>
#include <dccrg_cartesian_geometry.hpp>
#include <fsgrid.hpp>

#include <vector>
#include "../common.h"
#include "../definitions.h"
#include "../spatial_cells/spatial_cell_wrapper.hpp"
#include "../projects/project.h"
#include "../parameters.h"

using namespace spatial_cell;
using namespace projects;

namespace SBC {
   /*!\brief SBC::SysBoundaryCondition is the base class for system boundary conditions.
    *
    * SBC::SysBoundaryCondition defines a base class for applying boundary conditions.
    * Specific system boundary conditions inherit from this base class, that's why most
    * functions defined here are not meant to be called and contain a corresponding error
    * message. The functions to be called are the inherited class members.
    *
    * The initSysBoundary function is used to initialise the internal workings needed by the
    * system boundary condition to run (e.g. importing parameters, initialising class
    * members). assignSysBoundary is used to determine whether a given cell is within the
    * domain of system boundary condition. applyInitialState is called to initialise a system
    * boundary cell's parameters and velocity space.
    *
    * If needed, a user can write his or her own SBC::SysBoundaryConditions, which
    * are loaded when the simulation initializes.
    */
   class SysBoundaryCondition {
      public:
         SysBoundaryCondition();
         virtual ~SysBoundaryCondition();

         static void addParameters();
         virtual void getParameters() {
            std::cerr << "ERROR: base class SysBoundaryCondition::getParameters called!" << std::endl;
         }
         virtual void generateTemplateCell() {
            std::cerr << "ERROR: base class SysBoundaryCondition::generateTemplateCells called!" << std::endl;
         }

         virtual void initSysBoundary(
            creal& t,
            Project &project
         )=0;
         virtual void assignSysBoundary(dccrg::Dccrg<SpatialCell,
                                        dccrg::Cartesian_Geometry>& mpiGrid,
                                        fsgrids::technicalspan technical, FieldSolverGrid &fsgrid)=0;
         virtual void applyInitialState(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                        fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
                                        fsgrids::perbspan perb,
                                        fsgrids::bgbspan bgb,
                                        Project& project) = 0;
         virtual void updateState(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                  fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
                                  fsgrids::perbspan perb,
                                  fsgrids::bgbspan bgb, creal t) = 0;
         virtual Real
         fieldSolverBoundaryCondMagneticField(fsgrids::perbspan b,
                                              fsgrids::constbgbspan bgb,
                                              fsgrids::consttechnicalspan technical,
                                              const std::array<Real, 3>& gridSpacing,
                                              const std::array<fsgrid::FsSize_t, 3>& globalCoordinates,
                                              const fsgrid::FsStencil& stencil, cuint component) = 0;
         virtual void fieldSolverBoundaryCondElectricField(fsgrids::efieldspan e,
                                                           const fsgrid::FsStencil& stencil, cuint component) = 0;
         virtual void
         fieldSolverBoundaryCondHallElectricField(fsgrids::ehallspan ehall,
                                                  const fsgrid::FsStencil& stencil, cuint component) = 0;
         virtual void
         fieldSolverBoundaryCondGradPeElectricField(fsgrids::egradpespan EGradPe,
                                                    const fsgrid::FsStencil& stencil, cuint component) = 0;
         virtual void
         fieldSolverBoundaryCondDerivatives(fsgrids::dperbspan dperb,
                                            fsgrids::dmomentsspan dmoments,
                                            const fsgrid::FsStencil& stencil, cuint RKCase, cuint component) = 0;
         virtual void
         fieldSolverBoundaryCondBVOLDerivatives(fsgrids::volspan vols,
                                                const fsgrid::FsStencil& stencil, cuint component) = 0;
         static void setCellDerivativesToZero(fsgrids::dperbspan dperb,
                                              fsgrids::dmomentsspan dmoments,
                                              const fsgrid::FsStencil& stencil, cuint component);
         static void setCellBVOLDerivativesToZero(fsgrids::volspan vols,
                                                  const fsgrid::FsStencil& stencil, cuint component);

         virtual void mapCellPotentialAndGetEXBDrift(
            std::array<Real, CellParams::N_SPATIAL_CELL_PARAMS>& cellParams
         );

         /** This function computes the Vlasov (distribution function)
          * boundary condition for the given particle species only.
          * It is not! allowed to change block structure in cell.
          * @param mpiGrid Parallel grid.
          * @param cellID Spatial cell ID.
          * @param popID Particle species ID.*/
        virtual void vlasovBoundaryCondition(
            dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
            const CellID& cellID,
            const uint popID,
            const bool calculate_V_moments
        )=0;

        virtual void setupL2OutflowAtRestart(
            dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid
        ) {
            std::cerr << "ERROR: base class SysBoundaryCondition::setupL2OutflowAtRestart called!" << std::endl;
        }



         /*! Function used to know which faces the boundary condition is applied to.
          * @param faces Pointer to array of 6 bool in which the values are returned whether the corresponding face is of that
          * type. Order: 0 x+; 1 x-; 2 y+; 3 y-; 4 z+; 5 z-
          */
         virtual void getFaces(bool *faces) = 0;
         virtual std::string getName() const {
            std::cerr << "ERROR: base class SysBoundaryCondition::getName called!" << std::endl;
            return "ERROR";
         }
         virtual uint getIndex() const = 0; // {
//            std::cerr << "ERROR: base class SysBoundaryCondition::getIndex called!" << std::endl;
//            return sysboundarytype::N_SYSBOUNDARY_CONDITIONS;
//         }
         uint getPrecedence() const;
         bool isDynamic() const;

         bool updateSysBoundaryConditionsAfterLoadBalance(
            const dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
            const std::vector<CellID> & local_cells_on_boundary
         );
         bool doApplyUponRestart() const;
         void setPeriodicity(
            std::array<bool, 3> isFacePeriodic
         );
      protected:
         void determineFace(
            bool* isThisCellOnAFace,
            const creal x,const  creal y,const creal z,
            const creal dx,const creal dy,const creal dz,
            const bool excludeSlicesAndPeriodicDimensions = false
         ) const;
         void determineFace(
            std::array<bool, 6> &isThisCellOnAFace,
            const dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
            CellID id,
            const bool excludeSlicesAndPeriodicDimensions = false
         );
         void copyCellData(
            const SpatialCell *from,
            SpatialCell *to,
            const bool copyMomentsOnly,
            const uint popID,
            const bool copy_V_moments
         );
         std::array<SpatialCell*,27> & getFlowtoCells(
               const CellID& cellID
         );

         std::array<Realf*,27> getFlowtoCellsBlock(
               const std::array<SpatialCell*,27> flowtoCells,
               const vmesh::GlobalID blockGID,
               const uint popID
         );


      /*! Helper function to get the index of a neighboring cell in the arrays in allFlowtoCells.
       * \param i Offset in x direction (-1, 0 or 1)
       * \param j Offset in y direction (-1, 0 or 1)
       * \param k Offset in z direction (-1, 0 or 1)
       * \retval int Index in the flowto cell array (0 to 26, indexed from - to + x, y, z.
       */
      inline int nbrID(const int i, const int j, const int k){
         return (k+1)*9 + (j+1)*3 + i + 1;
      }

         void vlasovBoundaryCopyFromTheClosestNbr(
            dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
            const CellID& cellID,
            const bool& copyMomentsOnly,
            const uint popID,
            const bool calculate_V_moments
         );
         void vlasovBoundaryCopyFromTheClosestL1OutflowNbr(
            dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
            const CellID& cellID,
            const bool& copyMomentsOnly,
            const uint popID,
            const bool calculate_V_moments
         );
         void vlasovBoundaryCopyFromAllClosestNbrs(
            dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
            const CellID& cellID,
            const uint popID,
            const bool calculate_V_moments
         );
         void vlasovBoundaryFluffyCopyFromAllCloseNbrs(
            dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
            const CellID& cellID,
            const uint popID,
            const bool calculate_V_moments,
            creal fluffiness
         );
         std::array<int, 3> getTheClosestNonsysboundaryCell(
            fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
            cint i,
            cint j,
            cint k
         );
         std::vector< std::array<int, 3> > getAllClosestNonsysboundaryCells(
            fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
            cint i,
            cint j,
            cint k
         );
         CellID & getTheClosestNonsysboundaryCell(
            const CellID& cellID
         );
         std::vector<CellID> & getAllClosestNonsysboundaryCells(
            const CellID& cellID
         );
         std::vector<CellID> & getAllCloseNonsysboundaryCells(
            const CellID& cellID
         );
         Real
         fieldBoundaryCopyFromSolvingNbrMagneticField(fsgrids::perbspan b,
                                                      fsgrids::consttechnicalspan technical,
                                                      const fsgrid::FsStencil& stencil, cuint component, cuint mask);

         CellID & getTheClosestL1OutflowCell(
            const CellID& cellID
         );
         std::vector<CellID> & getAllClosestL1OutflowCells(
            const CellID& cellID
         );

         /*! Precedence value of the system boundary condition. */
         uint precedence;
         /*! Is the boundary condition dynamic in time or not. */
         bool dynamic;
         /*! Array of bool telling whether the system is periodic in any direction. */
         std::array<bool, 3> periodic;
         /*! Map of closest nonsysboundarycells. Used in getAllClosestNonsysboundaryCells. */
         std::unordered_map<CellID, std::vector<CellID>> allClosestNonsysboundaryCells;
         /*! Map of close nonsysboundarycells. Used in getAllCloseNonsysboundaryCells. */
         std::unordered_map<CellID, std::vector<CellID>> allCloseNonsysboundaryCells;
         /*! Map of closest Outflow L1 cells. Used in getAllClosestL1OutflowCells. */
         std::unordered_map<CellID, std::vector<CellID>> allClosestL1OutflowCells;
         /*! Map of close Outflow L1 cells. Used in getAllCloseL1OutflowCells. */
         std::unordered_map<CellID, std::vector<CellID>> allCloseL1OutflowCells;

         /*! bool telling whether to call again applyInitialState upon restarting the simulation. */
         bool applyUponRestart;
   };

   /*!\brief SBC::SysBoundaryCondition is the base class for system boundary conditions.
    *
    * SBC::SysBoundaryCondition defines a base class for applying boundary conditions.
    * Specific system boundary conditions inherit from this base class, that's why most
    * functions defined here are not meant to be called and contain a corresponding error
    * message. The functions to be called are the inherited class members.
    *
    * The initSysBoundary function is used to initialise the internal workings needed by the
    * system boundary condition to run (e.g. importing parameters, initialising class
    * members). assignSysBoundary is used to determine whether a given cell is within the
    * domain of system boundary condition. applyInitialState is called to initialise a system
    * boundary cell's parameters and velocity space.
    *
    * If needed, a user can write his or her own SBC::SysBoundaryConditions, which
    * are loaded when the simulation initializes.
    */
   class SysBoundaryConditionDevice {
      public:
         SysBoundaryConditionDevice() = default;
         virtual ~SysBoundaryConditionDevice() = default;
         virtual Real __device__ fieldSolverBoundaryCondMagneticField(fsgrids::perbspan b,
                                              fsgrids::constbgbspan bgb,
                                              fsgrids::consttechnicalspan technical,
                                              const std::array<Real, 3>& gridSpacing,
                                              const std::array<fsgrid::FsSize_t, 3>& globalCoordinates,
                                              const fsgrid::FsStencil& stencil, cuint component) = 0;
         
         virtual void __device__ fieldSolverBoundaryCondElectricField(fsgrids::efieldspan e,
                                                     const fsgrid::FsStencil& stencil, cuint component) = 0;

         __device__ inline Real fieldBoundaryCopyFromSolvingNbrMagneticField(
            fsgrids::perbspan b, fsgrids::consttechnicalspan technical,
            const fsgrid::FsStencil& stencil, cuint component, cuint mask) {
            int distance = numeric_limits<int>::max();
            auto closestCellIndex = 0;

            for (auto kk = -2; kk < 3; kk++) {
               for (auto jj = -2; jj < 3; jj++) {
                  for (auto ii = -2; ii < 3; ii++) {
                     if (stencil.cellExists(ii, jj, kk)) {
                        const auto index = stencil.indexFromOffset(ii, jj, kk);
                        const auto& tech = technical[index];
                        const bool copyable = (tech.SOLVE & mask) == mask &&
                                             tech.sysBoundaryFlag != sysboundarytype::DO_NOT_COMPUTE &&
                                             tech.sysBoundaryFlag != sysboundarytype::OUTER_BOUNDARY_PADDING;
                        const int d = ii * ii + jj * jj + kk * kk;
                        if (copyable && d < distance) {
                           distance = d;
                           closestCellIndex = index;
                        }
                     }
                  }
               }
            }

            if (distance == numeric_limits<int>::max()) {
               //abort_mpi("No closest cell found!", 1);
            }

            return b[closestCellIndex][fsgrids::bfield::PERBX + component];
         }

         __device__ inline void determineFace(
            bool* isThisCellOnAFace,
            const creal x,const  creal y,const creal z,
            const creal dx,const creal dy,const creal dz,
            const bool excludeSlicesAndPeriodicDimensions = false
         ){
            for(uint i=0; i<6; i++) {
               isThisCellOnAFace[i] = false;
            }
            if(x > dev_xmax - dx * 2) {
               isThisCellOnAFace[0] = true;
            }
            if(x < dev_xmin + dx * 2) {
               isThisCellOnAFace[1] = true;
            }
            if(y > dev_ymax - dy * 2) {
               isThisCellOnAFace[2] = true;
            }
            if(y < dev_ymin + dy * 2) {
               isThisCellOnAFace[3] = true;
            }
            if(z > dev_zmax - dz * 2) {
               isThisCellOnAFace[4] = true;
            }
            if(z < dev_zmin + dz * 2) {
               isThisCellOnAFace[5] = true;
            }
            if(excludeSlicesAndPeriodicDimensions == true) {
               if (dev_xcells_ini == 1 || this->periodic[0]) {
                  isThisCellOnAFace[0] = false;
                  isThisCellOnAFace[1] = false;
               }
               if (dev_ycells_ini == 1 || this->periodic[1]) {
                  isThisCellOnAFace[2] = false;
                  isThisCellOnAFace[3] = false;
               }
               if (dev_zcells_ini == 1 || this->periodic[2]) {
                  isThisCellOnAFace[4] = false;
                  isThisCellOnAFace[5] = false;
               }
            }
         }

         __device__ static inline void setCellDerivativesToZero(fsgrids::dperbspan dperb,
                                                      fsgrids::dmomentsspan dmoments,
                                                      const fsgrid::FsStencil& stencil, cuint component) {
            auto& dPerBGrid0 = dperb[stencil.ooo()];
            auto& dMomentsGrid0 = dmoments[stencil.ooo()];
            switch(component) {
               case 0: // x, xx
                  dMomentsGrid0[fsgrids::dmoments::drhomdx] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::drhoqdx] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dp11dx] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dp22dx] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dp33dx] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dVxdx] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dVydx] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dVzdx] = 0.0;

                  dPerBGrid0[fsgrids::dperb::dPERBydx] = 0.0;
                  dPerBGrid0[fsgrids::dperb::dPERBzdx] = 0.0;
                  dPerBGrid0[fsgrids::dperb::dPERBydxx] = 0.0;
                  dPerBGrid0[fsgrids::dperb::dPERBzdxx] = 0.0;
                  break;
               case 1: // y, yy
                  dMomentsGrid0[fsgrids::dmoments::drhomdy] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::drhoqdy] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dp11dy] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dp22dy] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dp33dy] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dVxdy] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dVydy] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dVzdy] = 0.0;

                  dPerBGrid0[fsgrids::dperb::dPERBxdy] = 0.0;
                  dPerBGrid0[fsgrids::dperb::dPERBzdy] = 0.0;
                  dPerBGrid0[fsgrids::dperb::dPERBxdyy] = 0.0;
                  dPerBGrid0[fsgrids::dperb::dPERBzdyy] = 0.0;
                  break;
               case 2: // z, zz
                  dMomentsGrid0[fsgrids::dmoments::drhomdz] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::drhoqdz] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dp11dz] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dp22dz] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dp33dz] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dVxdz] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dVydz] = 0.0;
                  dMomentsGrid0[fsgrids::dmoments::dVzdz] = 0.0;

                  dPerBGrid0[fsgrids::dperb::dPERBxdz] = 0.0;
                  dPerBGrid0[fsgrids::dperb::dPERBydz] = 0.0;
                  dPerBGrid0[fsgrids::dperb::dPERBxdzz] = 0.0;
                  dPerBGrid0[fsgrids::dperb::dPERBydzz] = 0.0;
                  break;
               case 3: // xy
                  dPerBGrid0[fsgrids::dperb::dPERBzdxy] = 0.0;
                  break;
               case 4: // xz
                  dPerBGrid0[fsgrids::dperb::dPERBydxz] = 0.0;
                  break;
               case 5: // yz
                  dPerBGrid0[fsgrids::dperb::dPERBxdyz] = 0.0;
                  break;
               default:
                  assert(false);
                  //cerr << __FILE__ << ":" << __LINE__ << ":" << " Invalid component" << endl;
                  //abort_mpi("Invalid component", 1);
            }
         }
      protected:
         /*! Array of bool telling whether the system is periodic in any direction. */
         std::array<bool, 3> periodic;
   };

   class OuterBoundaryCondition: public SysBoundaryCondition {
      public:
         virtual void assignSysBoundary(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid, fsgrids::technicalspan technical, FieldSolverGrid &fsgrid);
      protected:
         /*! Array of bool telling which faces are going to be processed by the system boundary condition.*/
         std::array<bool, 6> facesToProcess;
   };

   class OuterBoundaryConditionDevice: public SysBoundaryConditionDevice {
      public:
      protected:
   };

   // Moved outside the class since it's a helper function that doesn't require member access
   void averageCellData (
      const dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      std::vector<CellID> cellList,
      SpatialCell *to,
      const uint popID,
      const creal fluffiness = 0
   );

   /*!\brief SBC::findMaxwellianBlocksToInitialize returns a list of blocks to construct the VDF with.
    *
    *  Here the while loop iterates  from the centre of the maxwellian in blocksize (4*dvx) increments,
    *  and looks at the centre of the first velocity cell in the block (+0.5dvx), checking if the
    *  phase-space density there is large enough to be included due to sparsity threshold.
    *  That results in a "blocks radius"  vRadiusSquared from the centre of the Maxwellian distribution.
    *  Then we iterate through the actual blocks and calculate their radius R2 based on their velocity coordinates
    *  and the plasma bulk velocity. Blocks that fullfil R2<vRadiusSquared are included to blocksToInitialize.
    */
   vmesh::LocalID findMaxwellianBlocksToInitialize(
      const uint popID,
      spatial_cell::SpatialCell& cell,
      creal& rho,
      creal& T,
      creal& VX0,
      creal& VY0,
      creal& VZ0);

   void determineFaceNoClassMembers(
      bool* isThisCellOnAFace,
      creal x, creal y, creal z,
      creal dx, creal dy, creal dz,
      const std::array<bool, 3> periodicity,
      const bool excludeSlicesAndPeriodicDimensions=false // (default)
   );

} // namespace SBC

#endif
