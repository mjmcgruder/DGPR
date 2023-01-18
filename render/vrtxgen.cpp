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

#include <vector>

#include "buffers.cpp"
#include "geometry_common.cpp"
#include "io.cpp"

real eval_output(output_type rend_var, core_geometry& geom, simstate& U, u64 eL,
                 u64 node, array<real>& bfunc_preeval, real gamma)
{
  // evaluate desired output
  real state[5] = {};
  for (u64 bi = 0; bi < geom.refp.nbf3d; ++bi)
  {
    real eval = bfunc_preeval[geom.refp.nbf3d * node + bi];
    state[0] += U(eL, 0, bi) * eval;
    state[1] += U(eL, 1, bi) * eval;
    state[2] += U(eL, 2, bi) * eval;
    state[3] += U(eL, 3, bi) * eval;
    state[4] += U(eL, 4, bi) * eval;
  }
  real output_val;
  switch (rend_var)
  {
    case output_type::mach: {
      real u     = state[1] / state[0];
      real v     = state[2] / state[0];
      real w     = state[3] / state[0];
      real s     = sqrt(u * u + v * v + w * w);
      real p     = (gamma - 1.f) * (state[4] - 0.5 * s * s);
      real c     = sqrt(gamma * p / state[0]);
      output_val = s / c;
    }
    break;
    case output_type::rho: output_val = state[0]; break;
    case output_type::u: output_val = state[1] / state[0]; break;
    case output_type::v: output_val = state[2] / state[0]; break;
    case output_type::w: output_val = state[3] / state[0]; break;
  }

  return output_val;
}

struct tetslice
{
  u64 ntri;
  v3 nodes[6];
  real vals[6];
  v3 ref_nodes[6];
};

template<typename T>
inline void swap(T* a, T* b)
{
  T tmp = *a;
  *a    = *b;
  *b    = tmp;
}

real plane_quad_area(v3 nodes[4])
{
  // gather quad mid points for parallelogram
  v3 mids[4];
  for (u64 i = 0; i < 4; ++i)
  {
    u64 nxt   = (i + 1) % 4;
    mids[i].x = nodes[i].x + nodes[nxt].x / 2.;
    mids[i].y = nodes[i].y + nodes[nxt].y / 2.;
    mids[i].z = nodes[i].z + nodes[nxt].z / 2.;
  }
  // calculate area via parallelogram (double parallelogram area)
  v3 a(mids[1].x - mids[0].x, mids[1].y - mids[0].y, mids[1].z - mids[0].z);
  v3 b(mids[2].x - mids[1].x, mids[2].y - mids[1].y, mids[2].z - mids[1].z);
  v3 cp(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);

  return 2. * sqrt(cp.x * cp.x + cp.y * cp.y + cp.z * cp.z);
}

// Based on John Burkardt's plane_normal_tetrahedron_intersect.m
// https://people.sc.fsu.edu/~jburkardt/m_src/tetrahedron_slice_display/tetrahedron_slice_display.html
tetslice tet_plane_slice(v3& pp, v3& pn, v3 tet[4], real tet_vals[4],
                         v3 tet_ref[4])
{

  tetslice slice{};
  slice.ntri = 0;

  u64 nnode = 0;
  v3 nodes[4];
  real node_vals[6];
  v3 ref_nodes[4];

  // distance of each tet vertex above and below the plane
  real d[4];
  for (u64 i = 0; i < 4; ++i)
  {
    d[i] = pn.x * (pp.x - tet[i].x) + pn.y * (pp.y - tet[i].y) +
           pn.z * (pp.z - tet[i].z);
  }

  // exit if no intersection
  if ((d[0] < 0. && d[1] < 0. && d[2] < 0. && d[3] < 0.) ||
      (d[0] > 0. && d[1] > 0. && d[2] > 0. && d[3] > 0.))
    return slice;

  // add to slice node list
  for (u64 i = 0; i < 4; ++i)
  {
    if (d[i] == 0.)
    {
      nodes[nnode]     = tet[i];
      node_vals[nnode] = tet_vals[i];
      ref_nodes[nnode] = tet_ref[i];
      ++nnode;
    }
    else
    {
      for (u64 j = i + 1; j < 4; ++j)
      {
        if (d[i] * d[j] < 0.)  // checks for opposite sign
        {
          real den         = d[i] - d[j];
          nodes[nnode].x   = (d[i] * tet[j].x - d[j] * tet[i].x) / den;
          nodes[nnode].y   = (d[i] * tet[j].y - d[j] * tet[i].y) / den;
          nodes[nnode].z   = (d[i] * tet[j].z - d[j] * tet[i].z) / den;
          node_vals[nnode] = (d[i] * tet_vals[j] - d[j] * tet_vals[i]) / den;
          ref_nodes[nnode].x =
          (d[i] * tet_ref[j].x - d[j] * tet_ref[i].x) / den;
          ref_nodes[nnode].y =
          (d[i] * tet_ref[j].y - d[j] * tet_ref[i].y) / den;
          ref_nodes[nnode].z =
          (d[i] * tet_ref[j].z - d[j] * tet_ref[i].z) / den;
          ++nnode;
        }
      }
    }
  }

  if (nnode == 4)
  {
    real area0 = plane_quad_area(nodes);
    swap(&nodes[2], &nodes[3]);
    swap(&ref_nodes[2], &ref_nodes[3]);
    // v3 node2     = nodes[2];  // swap nodes
    // nodes[2]     = nodes[3];
    // nodes[3]     = node2;
    swap(&node_vals[2], &node_vals[3]);
    // real val2    = node_vals[2];  // swap vals
    // node_vals[2] = node_vals[3];
    // node_vals[3] = val2;
    real area1 = plane_quad_area(nodes);
    if (area1 < area0)  // flip back if the initial configuration was correct
    {
      swap(&nodes[2], &nodes[3]);
      swap(&ref_nodes[2], &ref_nodes[3]);
      // node2        = nodes[2];  // swap nodes back
      // nodes[2]     = nodes[3];
      // nodes[3]     = node2;
      swap(&node_vals[2], &node_vals[3]);
      // val2         = node_vals[2];  // swap vals back
      // node_vals[2] = node_vals[3];
      // node_vals[3] = val2;
    }
    // finally, assign to triangles
    slice.ntri         = 2;
    slice.nodes[0]     = nodes[0];
    slice.nodes[1]     = nodes[1];
    slice.nodes[2]     = nodes[3];
    slice.nodes[3]     = nodes[2];
    slice.nodes[4]     = nodes[3];
    slice.nodes[5]     = nodes[1];
    slice.vals[0]      = node_vals[0];
    slice.vals[1]      = node_vals[1];
    slice.vals[2]      = node_vals[3];
    slice.vals[3]      = node_vals[2];
    slice.vals[4]      = node_vals[3];
    slice.vals[5]      = node_vals[1];
    slice.ref_nodes[0] = ref_nodes[0];
    slice.ref_nodes[1] = ref_nodes[1];
    slice.ref_nodes[2] = ref_nodes[3];
    slice.ref_nodes[3] = ref_nodes[2];
    slice.ref_nodes[4] = ref_nodes[3];
    slice.ref_nodes[5] = ref_nodes[1];
  }
  else if (nnode == 3)
  {
    slice.ntri         = 1;
    slice.nodes[0]     = nodes[0];
    slice.nodes[1]     = nodes[1];
    slice.nodes[2]     = nodes[2];
    slice.vals[0]      = node_vals[0];
    slice.vals[1]      = node_vals[1];
    slice.vals[2]      = node_vals[2];
    slice.ref_nodes[0] = ref_nodes[0];
    slice.ref_nodes[1] = ref_nodes[1];
    slice.ref_nodes[2] = ref_nodes[2];
  }

  return slice;
}

// Axis Aligned Bounding Box
struct AABB
{
  real xmin = +REAL_MAX;
  real xmax = -REAL_MAX;
  real ymin = +REAL_MAX;
  real ymax = -REAL_MAX;
  real zmin = +REAL_MAX;
  real zmax = -REAL_MAX;

  void grow(v3 pos)
  {
    if      (pos.x > xmax) xmax = pos.x;
    else if (pos.x < xmin) xmin = pos.x;

    if      (pos.y > ymax) ymax = pos.y;
    else if (pos.y < ymin) ymin = pos.y;

    if      (pos.z > zmax) zmax = pos.z;
    else if (pos.z < zmin) zmin = pos.z;
  }
};

// Stores pre-computed information to speed up geometry generation along with
// other useful parameters.
struct vrtxgen_metadata
{
  // gpu compute functions
  compute_pipeline precomp_sol_basis;

  // pre-computes
  array<AABB> elem_bboxes;
  array<float> bfunc_precomp;  // [render node [basis func]]
  array<float> gfunc_precomp;  // [render node [basis func]]
  // other metadata
  real global_output_min = +REAL_MAX;
  real global_output_max = -REAL_MAX;
  v3 domain_center;

  vrtxgen_metadata();

  void compute_metadata_cpu(u64 N, output_type rend_var, simstate& U,
                            core_geometry& geom, real gamma);
  void compute_metadata_gpu(u32 N, core_geometry& geom);
};

vrtxgen_metadata::vrtxgen_metadata() :
precomp_sol_basis("compute_precomp_basis.spv", 3){}

void vrtxgen_metadata::compute_metadata_gpu(u32 N, core_geometry& geom)
{
  u32 p     = geom.p;
  u32 Np1   = N + 1;
  u32 Np1p3 = Np1 * Np1 * Np1;

  array<float> h_bf(Np1p3 * geom.refp.nbf3d);

  buffer_set<u32> d_N(&precomp_sol_basis.dset, 0);
  buffer_set<u32> d_p(&precomp_sol_basis.dset, 1);
  buffer_set<float> d_bf(&precomp_sol_basis.dset, 2);

  d_N.update(&N, 1);
  d_p.update(&p, 1);
  d_bf.update(h_bf.data, h_bf.len);

  precomp_sol_basis.run((h_bf.len + (256 - 1)) / 256, 1, 1);

  d_bf.retrieve(h_bf.data);

  // for (u64 i = 0; i < bfunc_precomp.len; ++i)
  // {
  //   printf("%+.3e %+.3e\n", bfunc_precomp[i], h_bf[i]);
  // }
}

void vrtxgen_metadata::compute_metadata_cpu(u64 N, output_type rend_var,
                                            simstate& U, core_geometry& geom,
                                            real gamma)
{
#define NI(i, j, k) (Np1 * Np1 * (k) + Np1 * (j) + (i))  // sub-elem node index

  elem_bboxes   = array<AABB>(geom.nelem);
  bfunc_precomp = array<float>((N + 1) * (N + 1) * (N + 1) * geom.refp.nbf3d);
  gfunc_precomp = array<float>((N + 1) * (N + 1) * (N + 1) * geom.refq.nbf3d);
  domain_center = v3();

  u64 Np1    = N + 1;
  u64 Np1p3  = Np1 * Np1 * Np1;
  real runit = 1. / (real)N;

  AABB domain_bounds;

  for (u64 k = 0; k < Np1; ++k)
  {
    for (u64 j = 0; j < Np1; ++j)
    {
      for (u64 i = 0; i < Np1; ++i)
      {
        u64 node_num = NI(i, j, k);
        v3 ref((real)i * runit, (real)j * runit, (real)k * runit);

        for (u64 bi = 0; bi < geom.refp.nbf3d; ++bi)
        {
          bfunc_precomp[geom.refp.nbf3d * node_num + bi] =
          lagrange3d(bi, geom.p, ref);
        }

        for (u64 bi = 0; bi < geom.refq.nbf3d; ++bi)
        {
          gfunc_precomp[geom.refq.nbf3d * node_num + bi] =
          lagrange3d(bi, geom.q, ref);
        }
      }
    }
  }

  for (u64 ei = 0; ei < geom.nelem; ++ei)
  {
    for (u64 k = 0; k < Np1; ++k)
    {
      for (u64 j = 0; j < Np1; ++j)
      {
        for (u64 i = 0; i < Np1; ++i)
        {
          u64 node_num = NI(i, j, k);

          // global space node location
          v3 glo = geom.ref2glo(ei, gfunc_precomp + geom.refq.nbf3d * node_num);

          // update bounding boxes
          domain_bounds.grow(glo);
          elem_bboxes[ei].grow(glo);

          real target_val =
          eval_output(rend_var, geom, U, ei, node_num, bfunc_precomp, gamma);

          // update global min and max value
          if (target_val > global_output_max) global_output_max = target_val;
          if (target_val < global_output_min) global_output_min = target_val;
        }
      }
    }
  }

  domain_center.x = 0.5 * (domain_bounds.xmax - domain_bounds.xmin);
  domain_center.y = 0.5 * (domain_bounds.ymax - domain_bounds.ymin);
  domain_center.z = 0.5 * (domain_bounds.zmax - domain_bounds.zmin);

#undef NI
}

// N is the interval count in each direction on the face
void generate_elem_surface_vertices(entity& surface_geometry,
                                    u64 swap_chain_image, core_geometry& geom,
                                    simstate& U, u64 N, output_type rend_var,
                                    vrtxgen_metadata& metadata, real gamma,
                                    colormap* cmap)
{
  u64 Np1       = N + 1;
  real samp_int = 1. / (real)N;
  u64 nbface =
  2 * ((geom.ny * geom.nz) + (geom.nx * geom.nz) + (geom.nx * geom.ny));

  array<vertex> vertices(nbface * Np1 * Np1);
  array<u32> indices(nbface * N * N * 6);

  u64 vi_base = 0;
  u64 ii_base = 0;
  for (u64 k = 0; k < geom.nz; ++k)
  {
    for (u64 j = 0; j < geom.ny; ++j)
    {
      for (u64 i = 0; i < geom.nx; ++i)
      {
        u64 e = geom.nx * geom.ny * k + geom.nx * j + i;

        for (u64 lf = 0; lf < 6; ++lf)
        {
          if ((i == 0           && lf == 0) || 
              (j == 0           && lf == 2) ||
              (k == 0           && lf == 4) || 
              (i == geom.nx - 1 && lf == 1) ||
              (j == geom.ny - 1 && lf == 3) || 
              (k == geom.nz - 1 && lf == 5))
          {
            // generate vertices for this face
            for (u64 fj = 0; fj < Np1; ++fj)
            {
              for (u64 fi = 0; fi < Np1; ++fi)
              {
                v2 face_pos_ref(samp_int * (real)fi, samp_int * (real)fj);
                v3 elem_pos_ref = surf2vol(lf, face_pos_ref);

                u64 vi, vj, vk;
                surf2vol(lf, fi, fj, Np1, &vi, &vj, &vk);
                u64 node_num = Np1 * Np1 * vk + Np1 * vj + vi;

                real output_val = eval_output(rend_var, geom, U, e, node_num,
                                              metadata.bfunc_precomp, gamma);

                v3 elem_pos_glo = geom.ref2glo(
                e, metadata.gfunc_precomp + geom.refq.nbf3d * node_num);

                vertices[vi_base + (fj * Np1 + fi)] = {
                {elem_pos_glo.x, elem_pos_glo.y, elem_pos_glo.z},
                to_color(output_val, metadata.global_output_min,
                         metadata.global_output_max, cmap),
                {elem_pos_ref.x, elem_pos_ref.y, elem_pos_ref.z}};
              }
            }

            // generate indices for this face iterating over blocks
            for (u64 fj = 0; fj < N; ++fj)
            {
              for (u64 fi = 0; fi < N; ++fi)
              {
                u64 bvib    = vi_base + Np1 * fj + fi;
                u64 bi_base = ii_base + 6 * (fj * N + fi);

                indices[bi_base + 0] = bvib; 
                indices[bi_base + 1] = bvib + 1; 
                indices[bi_base + 2] = bvib + Np1; 

                indices[bi_base + 3] = bvib + 1; 
                indices[bi_base + 4] = bvib + Np1 + 1; 
                indices[bi_base + 5] = bvib + Np1; 
              }
            }

            vi_base += Np1 * Np1;
            ii_base += N * N * 6;
          }
        }
      }
    }
  }

  surface_geometry.update_geometry(swap_chain_image, vertices.data,
                                   vertices.len, indices.data, indices.len);
}

void sample_element(u64 ei, u64 N, output_type rend_var, core_geometry& geom,
                    simstate& U, vrtxgen_metadata& metadata, real gamma,
                    v3* glo_positions, real* values, v3* ref_positions)
{
#define NI(i, j, k) (Np1 * Np1 * (k) + Np1 * (j) + (i))  // sub-elem node index
  u64 Np1       = N + 1;
  real samp_int = 1. / (real)N;  // sampling interval
  for (u64 k = 0; k < Np1; ++k)
  {
    for (u64 j = 0; j < Np1; ++j)
    {
      for (u64 i = 0; i < Np1; ++i)
      {
        u64 node_num = Np1 * Np1 * k + Np1 * j + i;

        v3 ref(samp_int * (real)i, samp_int * (real)j, samp_int * (real)k);

        real output_val = eval_output(rend_var, geom, U, ei, node_num,
                                      metadata.bfunc_precomp, gamma);

        glo_positions[NI(i, j, k)] =
        geom.ref2glo(ei, metadata.gfunc_precomp + geom.refq.nbf3d * node_num);
        values[NI(i, j, k)]        = output_val;
        ref_positions[NI(i, j, k)] = ref;
      }
    }
  }
#undef NI
}

template<typename T>
void cube_to_tets(T cube[8], T t0[4], T t1[4], T t2[4], T t3[4], T t4[4])
{
  t0[0] = cube[1];
  t0[1] = cube[7];
  t0[2] = cube[2];
  t0[3] = cube[4];

  t1[0] = cube[0];
  t1[1] = cube[1];
  t1[2] = cube[2];
  t1[3] = cube[4];

  t2[0] = cube[1];
  t2[1] = cube[2];
  t2[2] = cube[7];
  t2[3] = cube[3];

  t3[0] = cube[1];
  t3[1] = cube[7];
  t3[2] = cube[4];
  t3[3] = cube[5];

  t4[0] = cube[4];
  t4[1] = cube[7];
  t4[2] = cube[2];
  t4[3] = cube[6];
}

void generate_slice_vertices(entity& slice_geometry, u64 swap_chain_image,
                             core_geometry& geom, simstate& U, u64 N, v3 pp,
                             v3 pn, output_type rend_var,
                             vrtxgen_metadata& metadata, real gamma,
                             colormap* cmap)
{
  u64 Np1       = N + 1;
  u64 Np1p3     = Np1 * Np1 * Np1;
  real samp_int = 1. / (real)N;  // sampling interval
  real min      = metadata.global_output_min;
  real max      = metadata.global_output_max;
  std::vector<vertex> vertices;
  std::vector<u32> indices;
  u32 index_counter = 0;

#define NI(i, j, k) (Np1 * Np1 * (k) + Np1 * (j) + (i))  // sub-elem node index

  v3* elem_sample_positions     = new v3[Np1p3];
  real* elem_sample_values      = new real[Np1p3];
  v3* elem_sample_ref_positions = new v3[Np1p3];

  for (u64 ei = 0; ei < geom.nelem; ++ei)
  {

    // AABB intersection based on that found at:
    // https://gdbooks.gitbooks.io/3dcollisions/content/Chapter2/static_aabb_plane.html

    AABB bbox = metadata.elem_bboxes[ei];

    v3 c((bbox.xmin + bbox.xmax) / 2., (bbox.ymin + bbox.ymax) / 2.,
         (bbox.zmin + bbox.zmax) / 2.);
    v3 e(bbox.xmax - c.x, bbox.ymax - c.y, bbox.zmax - c.z);

    real r = dot(e, abs(pn));

    real s = dot(pn, c) - dot(pn, pp);

    bool intersect = std::abs(s) <= r;

    if (!intersect)
      continue;

    // find sample positions and values on this element
    sample_element(ei, N, rend_var, geom, U, metadata, gamma,
                   elem_sample_positions, elem_sample_values,
                   elem_sample_ref_positions);

    for (u64 k = 0; k < N; ++k)
    {
      for (u64 j = 0; j < N; ++j)
      {
        for (u64 i = 0; i < N; ++i)
        {
          v3 glo[8] = {elem_sample_positions[NI(i + 0, j + 0, k + 0)],
                       elem_sample_positions[NI(i + 1, j + 0, k + 0)],
                       elem_sample_positions[NI(i + 0, j + 1, k + 0)],
                       elem_sample_positions[NI(i + 1, j + 1, k + 0)],
                       elem_sample_positions[NI(i + 0, j + 0, k + 1)],
                       elem_sample_positions[NI(i + 1, j + 0, k + 1)],
                       elem_sample_positions[NI(i + 0, j + 1, k + 1)],
                       elem_sample_positions[NI(i + 1, j + 1, k + 1)]};

          real val[8] = {elem_sample_values[NI(i + 0, j + 0, k + 0)],
                         elem_sample_values[NI(i + 1, j + 0, k + 0)],
                         elem_sample_values[NI(i + 0, j + 1, k + 0)],
                         elem_sample_values[NI(i + 1, j + 1, k + 0)],
                         elem_sample_values[NI(i + 0, j + 0, k + 1)],
                         elem_sample_values[NI(i + 1, j + 0, k + 1)],
                         elem_sample_values[NI(i + 0, j + 1, k + 1)],
                         elem_sample_values[NI(i + 1, j + 1, k + 1)]};

          v3 ref[8] = {elem_sample_ref_positions[NI(i + 0, j + 0, k + 0)],
                       elem_sample_ref_positions[NI(i + 1, j + 0, k + 0)],
                       elem_sample_ref_positions[NI(i + 0, j + 1, k + 0)],
                       elem_sample_ref_positions[NI(i + 1, j + 1, k + 0)],
                       elem_sample_ref_positions[NI(i + 0, j + 0, k + 1)],
                       elem_sample_ref_positions[NI(i + 1, j + 0, k + 1)],
                       elem_sample_ref_positions[NI(i + 0, j + 1, k + 1)],
                       elem_sample_ref_positions[NI(i + 1, j + 1, k + 1)]};

          v3 t0[4], t1[4], t2[4], t3[4], t4[4];
          v3 t0r[4], t1r[4], t2r[4], t3r[4], t4r[4];
          real varr0[4], varr1[4], varr2[4], varr3[4], varr4[4];

          cube_to_tets(glo, t0, t1, t2, t3, t4);
          cube_to_tets(ref, t0r, t1r, t2r, t3r, t4r);
          cube_to_tets(val, varr0, varr1, varr2, varr3, varr4);

          // v3 t0[4]      = {n1, n7, n2, n4};  // cntr
          // real varr0[4] = {val1, val7, val2, val4};
          // v3 t0r[4]     = {r1, r7, r2, r4};

          // v3 t1[4]      = {n0, n1, n2, n4};  // low near
          // real varr1[4] = {val0, val1, val2, val4};
          // v3 t1r[4]     = {r0, r1, r2, r4};

          // v3 t2[4]      = {n1, n2, n7, n3};  // low far
          // real varr2[4] = {val1, val2, val7, val3};
          // v3 t2r[4]     = {r1, r2, r7, r3};

          // v3 t3[4]      = {n1, n7, n4, n5};  // right
          // real varr3[4] = {val1, val7, val4, val5};
          // v3 t3r[4]     = {r1, r7, r4, r5};

          // v3 t4[4]      = {n4, n7, n2, n6};  // left
          // real varr4[4] = {val4, val7, val2, val6};
          // v3 t4r[4]     = {r4, r7, r2, r6};

          tetslice slices[5];
          slices[0] = tet_plane_slice(pp, pn, t0, varr0, t0r);
          slices[1] = tet_plane_slice(pp, pn, t1, varr1, t1r);
          slices[2] = tet_plane_slice(pp, pn, t2, varr2, t2r);
          slices[3] = tet_plane_slice(pp, pn, t3, varr3, t3r);
          slices[4] = tet_plane_slice(pp, pn, t4, varr4, t4r);
          // store to vertices and indices
          for (u64 si = 0; si < 5; ++si)
          {
            for (u64 ti = 0; ti < slices[si].ntri; ++ti)
            {
              vertices.push_back(vertex{
              {slices[si].nodes[3 * ti + 0].x, slices[si].nodes[3 * ti + 0].y,
               slices[si].nodes[3 * ti + 0].z},
              to_color(slices[si].vals[3 * ti + 0], min, max, cmap),
              {
              slices[si].ref_nodes[3 * ti + 0].x,
              slices[si].ref_nodes[3 * ti + 0].y,
              slices[si].ref_nodes[3 * ti + 0].z,
              }});
              vertices.push_back(vertex{
              {slices[si].nodes[3 * ti + 1].x, slices[si].nodes[3 * ti + 1].y,
               slices[si].nodes[3 * ti + 1].z},
              to_color(slices[si].vals[3 * ti + 1], min, max, cmap),
              {
              slices[si].ref_nodes[3 * ti + 1].x,
              slices[si].ref_nodes[3 * ti + 1].y,
              slices[si].ref_nodes[3 * ti + 1].z,
              }});
              vertices.push_back(vertex{
              {slices[si].nodes[3 * ti + 2].x, slices[si].nodes[3 * ti + 2].y,
               slices[si].nodes[3 * ti + 2].z},
              to_color(slices[si].vals[3 * ti + 2], min, max, cmap),
              {
              slices[si].ref_nodes[3 * ti + 2].x,
              slices[si].ref_nodes[3 * ti + 2].y,
              slices[si].ref_nodes[3 * ti + 2].z,
              }});
              indices.push_back(index_counter + 0);
              indices.push_back(index_counter + 1);
              indices.push_back(index_counter + 2);
              index_counter += 3;
            }
          }
        }
      }
    }
  }

  delete[] elem_sample_positions;
  delete[] elem_sample_values;
  delete[] elem_sample_ref_positions;

  slice_geometry.update_geometry(swap_chain_image, vertices.data(),
                                 vertices.size(), indices.data(),
                                 indices.size());
#undef NI
}

void line_probe(core_geometry& geom, simstate& U, v3 point1, v3 point2,
                u64 num_samples, u64 N, output_type rend_var,
                vrtxgen_metadata& metadata, real gamma)
{
  u64 Np1   = N + 1;
  u64 Np1p3 = Np1 * Np1 * Np1;
#define NI(i, j, k) (Np1 * Np1 * (k) + Np1 * (j) + (i))  // sub-elem node index

  v3* elem_sample_positions     = new v3[Np1p3];
  v3* elem_sample_ref_positions = new v3[Np1p3];
  real* elem_sample_values      = new real[Np1p3];

  v3* probe_points   = new v3[num_samples];
  real* probe_values = new real[num_samples]{};

  v3 point_diff = point2 - point1;
  for (u64 sample = 0; sample < num_samples; ++sample)
  {
    v3 point = point1 + point_diff * ((real)sample / (real)(num_samples - 1));
    probe_points[sample] = point;

    for (u64 ei = 0; ei < geom.nelem; ++ei)
    {
      AABB bbox = metadata.elem_bboxes[ei];

      if (point.x >= bbox.xmin && point.x <= bbox.xmax &&
          point.y >= bbox.ymin && point.y <= bbox.ymax &&
          point.z >= bbox.zmin && point.z <= bbox.zmax)
      {
        sample_element(ei, N, rend_var, geom, U, metadata, gamma,
                       elem_sample_positions, elem_sample_values,
                       elem_sample_ref_positions);

        for (u64 k = 0; k < N; ++k)
        {
          for (u64 j = 0; j < N; ++j)
          {
            for (u64 i = 0; i < N; ++i)
            {
              v3 glo[8] = {elem_sample_positions[NI(i + 0, j + 0, k + 0)],
                           elem_sample_positions[NI(i + 1, j + 0, k + 0)],
                           elem_sample_positions[NI(i + 0, j + 1, k + 0)],
                           elem_sample_positions[NI(i + 1, j + 1, k + 0)],
                           elem_sample_positions[NI(i + 0, j + 0, k + 1)],
                           elem_sample_positions[NI(i + 1, j + 0, k + 1)],
                           elem_sample_positions[NI(i + 0, j + 1, k + 1)],
                           elem_sample_positions[NI(i + 1, j + 1, k + 1)]};

              real val[8] = {elem_sample_values[NI(i + 0, j + 0, k + 0)],
                             elem_sample_values[NI(i + 1, j + 0, k + 0)],
                             elem_sample_values[NI(i + 0, j + 1, k + 0)],
                             elem_sample_values[NI(i + 1, j + 1, k + 0)],
                             elem_sample_values[NI(i + 0, j + 0, k + 1)],
                             elem_sample_values[NI(i + 1, j + 0, k + 1)],
                             elem_sample_values[NI(i + 0, j + 1, k + 1)],
                             elem_sample_values[NI(i + 1, j + 1, k + 1)]};

              v3 tets[5][4];
              real vals[5][4];
              cube_to_tets(glo, tets[0], tets[1], tets[2], tets[3], tets[4]);
              cube_to_tets(val, vals[0], vals[1], vals[2], vals[3], vals[4]);

              for (u64 ti = 0; ti < 5; ++ti)
              {
                v3 r1 = tets[ti][0];
                v3 r2 = tets[ti][1];
                v3 r3 = tets[ti][2];
                v3 r4 = tets[ti][3];

                // clang-format off
                real T[9] = {r1.x - r4.x, r2.x - r4.x, r3.x - r4.x,
                             r1.y - r4.y, r2.y - r4.y, r3.y - r4.y,
                             r1.z - r4.z, r2.z - r4.z, r3.z - r4.z};
                // clang-format on

                real Tinv[9] = {};
                inv3x3(T, Tinv);

                v3 rhs          = point - r4;
                real rhs_arr[3] = {rhs.x, rhs.y, rhs.z};

                real lam[4] = {};
                matmul(3, 3, 1, 1., Tinv, rhs_arr, 1., lam);
                lam[3] = 1. - lam[0] - lam[1] - lam[2];

                bool inside = true;
                for (int li = 0; li < 4; ++li)
                {
                  if (lam[li] < 0. || lam[li] > 1.)
                    inside = false;
                }

                if (inside)
                {
                  real interpolated_value = 0;
                  for (int li = 0; li < 4; ++li)
                  {
                    interpolated_value += lam[li] * vals[ti][li];
                  }
                  probe_values[sample] = interpolated_value;
                  goto end_sample;
                }
              }
            }
          }
        }
      }
    }
  end_sample:;
  }

  for (u64 i = 0; i < num_samples; ++i)
  {
    printf("%f %f %f: %f\n", probe_points[i].x, probe_points[i].y,
           probe_points[i].z, probe_values[i]);
  }

  delete[] elem_sample_positions;
  delete[] elem_sample_ref_positions;
  delete[] elem_sample_values;
  delete[] probe_values;
#undef NI
}
