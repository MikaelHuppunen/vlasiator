/*
 * This file is part of Vlasiator.
 * Copyright 2010-2016 Finnish Meteorological Institute
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

#ifndef DERIVATIVES_HPP
#define DERIVATIVES_HPP

#include <array>

#include "../definitions.h"
//#include "../spatial_cells/spatial_cell_wrapper.hpp"
#include "../sysboundary/sysboundary.h"

#include "fs_limiters.h"

#ifdef USE_GPU
void calculateDerivativesSimple(fsgrids::perbspan perb,
                                fsgrids::momentsspan moments,
                                fsgrids::dperbspan dperb,
                                fsgrids::dmomentsspan dmoments,
                                fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
                                const bool doMoments, const DeviceParameters& parameters);
#else
void calculateDerivativesSimple(fsgrids::perbspan perb,
                                fsgrids::momentsspan moments,
                                fsgrids::dperbspan dperb,
                                fsgrids::dmomentsspan dmoments,
                                fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
                                const bool doMoments);
#endif

#ifdef USE_GPU
void calculateBVOLDerivativesSimple(fsgrids::volspan vol,
                                    fsgrids::technicalspan technical, FieldSolverGrid &fsgrid, const DeviceParameters& parameters);
#else
void calculateBVOLDerivativesSimple(fsgrids::volspan vol,
                                    fsgrids::technicalspan technical, FieldSolverGrid &fsgrid);
#endif

#ifdef USE_GPU
void calculateCurvatureSimple(fsgrids::volspan vol,
                              fsgrids::constbgbspan bgb,
                              fsgrids::technicalspan technical, FieldSolverGrid &fsgrid);
#else
void calculateCurvatureSimple(fsgrids::volspan vol,
                              fsgrids::constbgbspan bgb,
                              fsgrids::technicalspan technical, FieldSolverGrid &fsgrid);
#endif

void calculateScaledDeltasSimple(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid);

#endif
