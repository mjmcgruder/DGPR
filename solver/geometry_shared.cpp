/* DGPR: Discontinuous Galerkin Performance Research                          */
/* Copyright (C) 2023  Miles McGruder                                         */
/*                                                                            */
/* This program is free software: you can redistribute it and/or modify       */
/* it under the terms of the GNU General Public License as published by       */
/* the Free Software Foundation, either version 3 of the License, or          */
/* (at your option) any later version.                                        */
/*                                                                            */
/* This program is distributed in the hope that it will be useful,            */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of             */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              */
/* GNU General Public License for more details.                               */
/*                                                                            */
/* You should have received a copy of the GNU General Public License          */
/* along with this program.  If not, see <https://www.gnu.org/licenses/>.     */

#pragma once

#include "geometry_common.cpp"

struct shared_geometry
{
  core_geometry core;

  array<s64> interior_face_list;
  array<s64> boundary_face_list;
  array<real, 3> n_;
  array<real> fJ_;
  array<real, 2> interior_h;
  array<real> boundary_h;

  shared_geometry();
  shared_geometry(u64 sol_order, u64 geo_order, u64 nelem_x, u64 nelem_y,
                  u64 nelem_z, s64 boundaries[6]);
  shared_geometry(core_geometry& core_geom);

  void precompute_face_geometry();

  void precompute();

  real* n(u64 face, u64 qnode);
  real& fJ(u64 face, u64 qnode);

  u64 nface() const;
  u64 num_interior_faces() const;
  u64 num_boundary_faces() const;

  u64 fJ_size() const;
  u64 n_size() const;
  u64 interior_h_size() const;
  u64 boundary_h_size() const;
  u64 interior_face_list_size() const;
  u64 boundary_face_list_size() const;
};

real* shared_geometry::n(u64 face, u64 qnode)
{
  return n_ + 3 * (core.refp.fqrule.n * face + qnode);
}
real& shared_geometry::fJ(u64 face, u64 qnode)
{
  return fJ_[core.refp.fqrule.n * face + qnode];
}

u64 shared_geometry::nface() const
{
  return core.nelem * 3 + ((core.bounds[0] != periodic) * core.ny * core.nz +
                           (core.bounds[2] != periodic) * core.nz * core.nx +
                           (core.bounds[4] != periodic) * core.nx * core.ny);
}
u64 shared_geometry::num_interior_faces() const
{
  return nface() - num_boundary_faces();
}
u64 shared_geometry::num_boundary_faces() const
{
  return (core.ny * core.nz * (core.bounds[0] < -2)) +
         (core.ny * core.nz * (core.bounds[1] < -2)) +
         (core.nx * core.nz * (core.bounds[2] < -2)) +
         (core.nx * core.nz * (core.bounds[3] < -2)) +
         (core.nx * core.ny * (core.bounds[4] < -2)) +
         (core.nx * core.ny * (core.bounds[5] < -2));
}

u64 shared_geometry::fJ_size() const
{
  return nface() * core.refp.fqrule.n;
}
u64 shared_geometry::n_size() const
{
  return nface() * core.refp.fqrule.n * 3;
}
u64 shared_geometry::interior_h_size() const
{
  return num_interior_faces() * 2;
}
u64 shared_geometry::boundary_h_size() const
{
  return num_boundary_faces();
}
u64 shared_geometry::interior_face_list_size() const
{
  return num_interior_faces() * 5;
}
u64 shared_geometry::boundary_face_list_size() const
{
  return num_boundary_faces() * 4;
}

shared_geometry::shared_geometry() :
core(),
interior_face_list(),
boundary_face_list(),
n_(),
fJ_(),
interior_h(),
boundary_h()
{}

shared_geometry::shared_geometry(u64 sol_order, u64 geo_order, u64 nelem_x,
                                 u64 nelem_y, u64 nelem_z, s64 boundaries[6])
{
  if ((boundaries[0] == periodic && boundaries[1] != periodic) ||
      (boundaries[1] == periodic && boundaries[0] != periodic) ||
      (boundaries[2] == periodic && boundaries[3] != periodic) ||
      (boundaries[3] == periodic && boundaries[2] != periodic) ||
      (boundaries[4] == periodic && boundaries[5] != periodic) ||
      (boundaries[5] == periodic && boundaries[4] != periodic))
  {
    errout("periodic boundaries are not matching!");
  }

  core =
  core_geometry(sol_order, geo_order, nelem_x, nelem_y, nelem_z, boundaries);

  interior_face_list = array<s64>(interior_face_list_size());
  boundary_face_list = array<s64>(boundary_face_list_size());
  fJ_                = array<real>(fJ_size());
  n_                 = array<real, 3>({3, core.refp.fqrule.n, nface()});
  interior_h         = array<real, 2>({2, num_interior_faces()});
  boundary_h         = array<real>(boundary_h_size());
}

shared_geometry::shared_geometry(core_geometry& core_geom)
{
  core = core_geom;

  interior_face_list = array<s64>(interior_face_list_size());
  boundary_face_list = array<s64>(boundary_face_list_size());
  fJ_                = array<real>(fJ_size());
  n_                 = array<real, 3>({3, core.refp.fqrule.n, nface()});
  interior_h         = array<real, 2>({2, num_interior_faces()});
  boundary_h         = array<real>(boundary_h_size());
}

void shared_geometry::precompute_face_geometry()
{

  u64 nx  = core.nx;
  u64 ny  = core.ny;
  u64 nz  = core.nz;
  u64 gfn = 0, ifn = 0, bfn = 0;
  s64 faces_to_compute[6] = {1, 3, 5, -1, -1, -1};

  for (u64 k = 0; k < nz; ++k)
  {
    for (u64 j = 0; j < ny; ++j)
    {
      for (u64 i = 0; i < nx; ++i)
      {
        u64 e = nx * ny * k + nx * j + i;

        // generate list of local faces to pre-compute for this element
        for (u64 fn = 3; fn < 6; ++fn)
          faces_to_compute[fn] = -1;
        if (i == 0)
          faces_to_compute[3] = 0;
        if (j == 0)
          faces_to_compute[4] = 2;
        if (k == 0)
          faces_to_compute[5] = 4;

        // pre-compute the appropriate list of faces
        for (u64 if2comp = 0; if2comp < 6; ++if2comp)
        {
          if (faces_to_compute[if2comp] == -1)
            continue;

          u64 lfn = (u64)faces_to_compute[if2comp];

          // skipping high side boundary faces if periodc to not double count
          if (core.bounds[1] == periodic && i == nx - 1 && lfn == 1)
            continue;
          if (core.bounds[3] == periodic && j == ny - 1 && lfn == 3)
            continue;
          if (core.bounds[5] == periodic && k == nz - 1 && lfn == 5)
            continue;

          s64 iadj, jadj, kadj;
          s64 bound_indx =
          find_boundary_index(i, j, k, lfn, nx, ny, nz, &iadj, &jadj, &kadj);

          bool isbound = false;
          s64 eL = 0, eR = 0;
          if (bound_indx == -1 || core.bounds[bound_indx] == periodic)
          {
            eL = e;
            eR = nx * ny * ((kadj + nz) % nz) + nx * ((jadj + ny) % ny) +
                 ((iadj + nx) % nx);

            interior_face_list[5 * ifn + 0] = gfn;
            interior_face_list[5 * ifn + 1] = eL;
            interior_face_list[5 * ifn + 2] = eR;
            interior_face_list[5 * ifn + 3] = lfn;
            interior_face_list[5 * ifn + 4] = opposite_local_face[lfn];

            ++ifn;
          }
          else
          {
            isbound = true;
            eL = e;
            boundary_face_list[4 * bfn + 0] = gfn;
            boundary_face_list[4 * bfn + 1] = eL;
            boundary_face_list[4 * bfn + 2] = core.bounds[bound_indx];
            boundary_face_list[4 * bfn + 3] = lfn;

            ++bfn;
          }

          // evaluate normal and surface element at each quad point
          real face_area = 0.;
          for (u64 qnn = 0; qnn < core.refp.fqrule.n; ++qnn)
          {
            real* normal = n(gfn, qnn);
            real& surfJ  = fJ(gfn, qnn);

            compute_surface_geometry(e, lfn, qnn, core, iadj, jadj, kadj,
                                     normal, &surfJ);

            face_area += surfJ * core.refp.fqrule.wgt(qnn);
          }

          // evaluate and store ratio of face area to adjacent elem volumes
          real volL = 0., volR = 0.;
          for (u64 qi = 0; qi < core.refp.vqrule.n; ++qi)
          {
            volL += core.vJ(eL, qi) * core.refp.vqrule.wgt(qi);
            if (!isbound)
              volR += core.vJ(eR, qi) * core.refp.vqrule.wgt(qi);
          }
          if (!isbound)
          {
            interior_h[2 * (ifn - 1) + 0] = volL / face_area;
            interior_h[2 * (ifn - 1) + 1] = volR / face_area;
          }
          else
          {
            boundary_h[bfn - 1] = volL / face_area;
          }

          ++gfn;
        }
      }
    }
  }
}

void shared_geometry::precompute()
{
  core.precompute();
  precompute_face_geometry();
}

shared_geometry bump_shared(u64 p, u64 q, u64 nx, u64 ny, u64 nz, real lx,
                            real ly, real lz, real a, real bmp_height,
                            real bmp_center, real bmp_variance, btype* bounds)
{

  shared_geometry geom(p, q, nx, ny, nz, (s64*)bounds);

  // (rectangular) element sizes
  real elx = lx / (real)nx;
  real ely = ly / (real)ny;
  real elz = lz / (real)nz;

  generate_bump_nodes(elx, ely, elz, ly, geom.core.ny, a, bmp_height,
                      bmp_center, bmp_variance, 0, 0, 0, geom.core);

  geom.precompute();
  return geom;
}
