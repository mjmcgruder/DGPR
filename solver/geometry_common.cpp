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

#include "basis.cpp"
#include "boundaries.cpp"
#include "data_structures.cpp"
#include "dgmath.cpp"
#include "error.cpp"

struct core_geometry
{
  u64 p;  // solution order (constant for now)
  u64 q;  // gometry order (also constant for now)

  u64 nx;  // element count in x
  u64 ny;  //                  y
  u64 nz;  //                  z, elements are ordered [z [y [x]]]
  u64 nelem;

  basis refp;
  basis refq;

  // Physical locations of all geometry nodes. Adjacent elements do not share
  // the same node, this should support the general case where geometry nodes do
  // not necessarily appear on element edges, and non-constant geometry order.
  // Storing geometry nodes shared on edges also complicates indexing quite a
  // bit... (this was initially implemented this with shared nodes on edges, it
  // was bad, don't do it again!).
  // The format is as follows:
  //   each element in the domain
  //     each geometry node ordered [z [y [x]]]
  //       coordinates stored [x y z]
  array<real, 3> node_;

  // Real space element mass matrices for each element in the domain.
  // A single mass matrix is stored for each element in the order of the
  // elements. Each matrix is stored in a row major upper triangular packed
  // form.
  array<real> Minv_;

  // Jacobian determinant storage for every volume quadrature point on each
  // element in the geometry. The format is as follows:
  //   each element ordered [z [y [x]]]
  //     each volume quadrature point (ordered by "basis" struct)
  array<real> vJ_;

  // Gradient storage on each volume quadrature point on each element in the
  // geometry. The format is as follows:
  //   for each element ordered [z [y [x]]]
  //     each volume quadrautre point (ordered by 3D "quad" struct)
  //       for each each basis function (ordered by "basis" struct)
  //         gradient elements stored [d_dx d_dy d_dz]
  array<real> vgrad_;

  // Gradient storage at each face quadrature point of every face on every
  // element in the geometry. The format is:
  //   each element ordered [z [y [x]]]
  //     each local face ordered as in basis.feval
  //       each face quadrature point as ordered in a 2D "quad" struct
  //         each basis function ordered by "basis" struct
  //           the gradient stored [d_dx, d_dy, d_dz]
  array<real> fgrad_;

  // Stores neighboring procs over -x +x -y +y -z +z bounds. If there is no
  // neighboring processor there must be a boundary on that domain face and
  // instead the boundary type will be stored. Note that btype and this array
  // are both signed integers.
  s64 bounds[6];

  core_geometry();
  core_geometry(u64 sol_order, u64 geo_order, u64 nelem_x, u64 nelem_y,
                u64 nelem_z, s64 boundaries[6]);

  // Maps a given reference space location on an element to the global space
  // position using a transfinite mapping (I think that's what it's called, the
  // thing where you use Lagrange interpolating polynomials to interpolate
  // between geometry nodes yielding the full curved geometry).
  // Instead of taking an arbitrary reference space location, the second version
  // takes pre-computed basis function values. This version will be faster in
  // cases where ref2glo is evaluated repeatedly on the same reference space
  // locations due to pre-computeation of the lagrange polynomial evaluations.
  v3 ref2glo(u64 e, v3 ref);
  v3 ref2glo(u64 e, real* phi);

  // Computes the Jacobian matrix for any given reference space location on any
  // element in the domain.
  void jacobian(u64 e, u64 qn, u64* f, basis& refq_precmp, real J[9]);

  // Pre-computes all relevant information for residual computation assuming nx,
  // ny, nz, p, q, and nodes, are already filled and all other memory is
  // allocated.
  void precompute();

  // indexing functions

  real* node(u64 e, u64 n);
  real* Minv(u64 e);
  real* vgrad(u64 e, u64 bf, u64 qn);
  real* fgrad(u64 e, u64 f, u64 bf, u64 qn);

  real& vJ(u64 e, u64 qn);

  // counting functions

  u64 num_entry_Minv() const;

  // array size functions

  u64 node_size() const;
  u64 Minv_size() const;
  u64 vJ_size() const;
  u64 vgrad_size() const;
  u64 fgrad_size() const;
};

struct simstate
{
  u64 rank;
  u64 nbfnc;
  u64 nelem;
  array<real, 3> U;

  simstate();
  simstate(core_geometry& geom);

  real& operator()(u64 elm, u64 rnk, u64 bfn);
  real& operator[](u64 indx);
  real* operator+(u64 num);
  u64 size();
};

/* ---------------------------- */
/* core_geometry implementation */
/* ---------------------------- */

core_geometry::core_geometry() :
p(0),
q(0),
nx(0),
ny(0),
nz(0),
nelem(0),
refp(),
refq(),
node_(),
Minv_(),
vJ_(),
vgrad_(),
fgrad_(),
bounds{}
{}

core_geometry::core_geometry(u64 sol_order, u64 geo_order, u64 nelem_x,
                             u64 nelem_y, u64 nelem_z, s64 boundaries[6]) :
p(sol_order),
q(geo_order),
nx(nelem_x),
ny(nelem_y),
nz(nelem_z),
nelem(nx * ny * nz),
refp(p, 2 * p + 1),
refq(q, 2 * p + 1),
node_({3, refq.nbf3d, nelem}),
Minv_(Minv_size()),
vJ_(vJ_size()),
vgrad_(vgrad_size()),
fgrad_(fgrad_size())
{
  for (int i = 0; i < 6; ++i)
    bounds[i] = boundaries[i];
}

u64 core_geometry::num_entry_Minv() const
{
  return ((refp.nbf3d * (refp.nbf3d + 1)) / 2);
}

real* core_geometry::node(u64 e, u64 n)
{
  return node_ + (refq.nbf3d * e + n) * 3;
}
real* core_geometry::Minv(u64 e)
{
  return Minv_ + e * num_entry_Minv();
}
real* core_geometry::vgrad(u64 e, u64 bf, u64 qn)
{
  return vgrad_ + (refp.nbf3d * (refp.vqrule.n * e + qn) + bf) * 3;
}
real* core_geometry::fgrad(u64 e, u64 f, u64 bf, u64 qn)
{
  return fgrad_ + (refp.nbf3d * (refp.fqrule.n * (6 * e + f) + qn) + bf) * 3;
}
real& core_geometry::vJ(u64 e, u64 qn)
{
  return vJ_[e * refp.vqrule.n + qn];
}

u64 core_geometry::node_size() const
{
  return nelem * refq.nbf3d * 3;
}
u64 core_geometry::Minv_size() const
{
  return nelem * num_entry_Minv();
}
u64 core_geometry::vJ_size() const
{
  return nelem * refp.vqrule.n;
}
u64 core_geometry::vgrad_size() const
{
  return nelem * refp.nbf3d * refp.vqrule.n * 3;
}
u64 core_geometry::fgrad_size() const
{
  return nelem * refp.nbf3d * refp.fqrule.n * 6 * 3;
}

v3 core_geometry::ref2glo(u64 e, v3 ref)
{
  v3 glo(0., 0., 0.);
  for (u64 bi = 0; bi < refq.nbf3d; ++bi)
  {
    real phi = lagrange3d(bi, q, ref);
    real* n  = node(e, bi);

    glo.x += n[0] * phi;
    glo.y += n[1] * phi;
    glo.z += n[2] * phi;
  }

  return glo;
}

v3 core_geometry::ref2glo(u64 e, real* phi)
{
  v3 glo(0., 0., 0.);
  for (u64 bi = 0; bi < refq.nbf3d; ++bi)
  {
    real* n = node(e, bi);
    glo.x += n[0] * phi[bi];
    glo.y += n[1] * phi[bi];
    glo.z += n[2] * phi[bi];
  }
  return glo;
}

void core_geometry::jacobian(u64 e, u64 qn, u64* f, basis& refq_precmp,
                             real J[9])
{
  for (u64 i = 0; i < 9; ++i)
    J[i] = 0.;

  for (u64 gni = 0; gni < refq_precmp.nbf3d; ++gni)
  {
    real* n = node(e, gni);

    real* ref_grad;
    if (f)
      ref_grad = refq_precmp.fgrad_at(*f, gni, qn);
    else
      ref_grad = refq_precmp.vgrad_at(gni, qn);

    J[0] += n[0] * ref_grad[0];
    J[1] += n[0] * ref_grad[1];
    J[2] += n[0] * ref_grad[2];
    J[3] += n[1] * ref_grad[0];
    J[4] += n[1] * ref_grad[1];
    J[5] += n[1] * ref_grad[2];
    J[6] += n[2] * ref_grad[0];
    J[7] += n[2] * ref_grad[1];
    J[8] += n[2] * ref_grad[2];
  }
}

void core_geometry::precompute()
{
  {
    real jac[9];
    u64 nbf = refp.nbf3d;

    basis refq_2p(q, 2 * p);
    basis refp_2p(p, 2 * p);

    real* detJ_precomp = new real[refq_2p.vqrule.n];
    real* M            = new real[nbf * nbf];

    for (u64 e = 0; e < nelem; ++e)
    {
      // pre-computing J is much quicker than re-computing it in the inner loop
      for (u64 qnn = 0; qnn < refq_2p.vqrule.n; ++qnn)
      {
        jacobian(e, qnn, nullptr, refq_2p, jac);
        detJ_precomp[qnn] = det3x3(jac);
      }

      // use symmetry to construct a full mass matrix
      for (u64 tfn = 0; tfn < nbf; ++tfn)  // loop over only the top half of the
      {                                    // mass matrix
        for (u64 bfn = tfn; bfn < nbf; ++bfn)
        {
          M[nbf * tfn + bfn] = 0.;
          for (u64 qnn = 0; qnn < refp_2p.vqrule.n; ++qnn)
          {
            M[nbf * tfn + bfn] += detJ_precomp[qnn] * refp_2p.vqrule.wgt(qnn) *
                                  refp_2p.veval_at(tfn, qnn) *
                                  refp_2p.veval_at(bfn, qnn);
          }
          M[nbf * bfn + tfn] = M[nbf * tfn + bfn];
        }
      }

      // invert the mass matrix
      int err = invert(M, nbf);
      if (err != 0)
        errout("mass matrix inversion failed! (status %d)\n", err);

      // pull out the upper triangular part for final storage
      u64 iM      = 0;
      real* Msymm = Minv(e);
      for (u64 ir = 0; ir < nbf; ++ir)
      {
        for (u64 ic = ir; ic < nbf; ++ic)
        {
          Msymm[iM] = M[nbf * ir + ic];
          ++iM;
        }
      }
    }

    delete[] detJ_precomp;
    delete[] M;
  }

  {
    real J[9];
    real Jinv[9];
    for (u64 e = 0; e < nelem; ++e)
    {
      for (u64 qn = 0; qn < refp.vqrule.n; ++qn)
      {
        // compute and store jacobian determinant
        jacobian(e, qn, nullptr, refq, J);
        vJ(e, qn) = det3x3(J);

        // compute and store real-space basis function gradients
        inv3x3(J, Jinv);
        for (u64 bf = 0; bf < refp.nbf3d; ++bf)
        {
          real* grad_vec = refp.vgrad_at(bf, qn);
          matmul(1, 3, 3, 1., grad_vec, Jinv, 0., vgrad(e, bf, qn));
        }
      }
    }
  }

  {
    real J[9];
    real Jinv[9];
    for (u64 ei = 0; ei < nelem; ++ei)
    {
      for (u64 fi = 0; fi < 6; ++fi)
      {
        for (u64 qi = 0; qi < refp.fqrule.n; ++qi)
        {
          jacobian(ei, qi, &fi, refq, J);
          inv3x3(J, Jinv);

          // compute and store real-space basis gradients at face quad points
          for (u64 bi = 0; bi < refp.nbf3d; ++bi)
          {
            real* grad = refp.fgrad_at(fi, bi, qi);
            matmul(1, 3, 3, 1., grad, Jinv, 0., fgrad(ei, fi, bi, qi));
          }
        }
      }
    }
  }
}

s64 find_boundary_index(u64 i, u64 j, u64 k, u64 lfn, u64 nx, u64 ny, u64 nz,
                        s64* iadj, s64* jadj, s64* kadj)
{
  s64 adjmodi[6] = {-1, 1, 0, 0, 0, 0};
  s64 adjmodj[6] = {0, 0, -1, 1, 0, 0};
  s64 adjmodk[6] = {0, 0, 0, 0, -1, 1};

  *iadj = i + adjmodi[lfn];
  *jadj = j + adjmodj[lfn];
  *kadj = k + adjmodk[lfn];

  s64 bound_indx = -1;
  bound_indx += ((*iadj < 0) * 1) + ((*iadj > (s64)nx - 1) * 2) +
                ((*jadj < 0) * 3) + ((*jadj > (s64)ny - 1) * 4) +
                ((*kadj < 0) * 5) + ((*kadj > (s64)nz - 1) * 6);

  return bound_indx;
}

void compute_surface_geometry(u64 e, u64 lfn, u64 qnn, core_geometry& core,
                              s64 iadj, s64 jadj, s64 kadj, real* normal,
                              real* surfJ)
{
  real dx_ds[3] = {};
  real dx_dt[3] = {};

  for (u64 bfn = 0; bfn < core.refq.nbf2d; ++bfn)
  {
    // gather global node position and referene space gradient
    v3 x = core.ref2glo(e, surf2vol(lfn, lagrange_node2d(bfn, core.refq.p)));
    v2 ref_grad =
    lagrange2d_grad(bfn, core.refq.p, core.refq.fqrule.pos2d(qnn));

    // outer product for the tangent in each parametric direction
    dx_ds[0] += x.x * ref_grad.x;
    dx_ds[1] += x.y * ref_grad.x;
    dx_ds[2] += x.z * ref_grad.x;
    dx_dt[0] += x.x * ref_grad.y;
    dx_dt[1] += x.y * ref_grad.y;
    dx_dt[2] += x.z * ref_grad.y;
  }

  cross(dx_ds, dx_dt, normal);  // cross for normal vector
  *surfJ = l2(normal, 3);       // magnitude for suface element

  // normalize for normal vector
  normal[0] /= *surfJ;
  normal[1] /= *surfJ;
  normal[2] /= *surfJ;

  // negate normal in cases where it is pointing into "this" element
  if (iadj < 0 || jadj < 0 || kadj < 0)
  {
    normal[0] *= -1.;
    normal[1] *= -1.;
    normal[2] *= -1.;
  }
}

/* ----------------------- */
/* simstate implementation */
/* ----------------------- */

simstate::simstate() : rank(0), nbfnc(0), nelem(0), U()
{}

simstate::simstate(core_geometry& geom) :
rank(5),
nbfnc(geom.refp.nbf3d),
nelem(geom.nx * geom.ny * geom.nz),
U({nbfnc, rank, nelem})
{}

real& simstate::operator()(u64 elm, u64 rnk, u64 bfn)
{
  return U[nbfnc * rank * elm + nbfnc * rnk + bfn];
}

real& simstate::operator[](u64 indx)
{
  return U[indx];
}

u64 simstate::size()
{
  return nelem * rank * nbfnc;
}

real* simstate::operator+(u64 num)
{
  return U + num;
}

/* ------------------- */
/* geometry generation */
/* ------------------- */

// initializes a gaussian bump domain and performs pre-computation of
// interesting quantities. this specific geometry generator should be replaced
// by one that reads a mesh file in the future...
void generate_bump_nodes(real elx, real ely, real elz, real gly, u64 gny,
                         real a, real bmp_height, real bmp_center,
                         real bmp_variance, u64 dsx, u64 dsy, u64 dsz,
                         core_geometry& core)
{
  u64 qp1   = core.q + 1;
  u64 nx = core.nx, ny = core.ny;
  real nsx = elx / (real)core.q;  // node spacings (within an element)
  real nsz = elz / (real)core.q;

  // fill "nodes" entries of the geometry
  for (u64 ez = dsz; ez < dsz + core.nz; ++ez)
  {
    real esz = ez * elz;  // element start location in z direction
    for (u64 ey = dsy; ey < dsy + core.ny; ++ey)
    {
      /* doing some non-linear spacing calculations */
      real ys = 0.;

      real esy, eey;
      if (a != 0.)
      {
        real xi = -1. + 2. * ((real)ey / gny);
        real y  = 0.5 * tanh(xi * atanh(a));
        esy     = ys + gly * ((y * (2. / a) + 1.) / 2.);

        xi  = -1. + 2. * (((real)(ey + 1.)) / gny);
        y   = 0.5 * tanh(xi * atanh(a));
        eey = ys + gly * ((y * (2. / a) + 1.) / 2.);
      }
      else
      {
        esy = ey * ely;
        eey = (ey + 1.) * ely;
      }
      /* doing some non-linear spacing calculations */

      real nsy = (eey - esy) / (real)core.q;
      for (u64 ex = dsx; ex < dsx + nx; ++ex)
      {
        real esx = ex * elx;
        u64 e    = nx * ny * (ez - dsz) + nx * (ey - dsy) + (ex - dsx);
        for (u64 k = 0; k < qp1; ++k)
        {
          for (u64 j = 0; j < qp1; ++j)
          {
            for (u64 i = 0; i < qp1; ++i)
            {
              u64 node_num = qp1 * qp1 * k + qp1 * j + i;
              real* node   = core.node(e, node_num);
              node[0]      = esx + nsx * (real)i;
              node[1]      = esy + nsy * (real)j;
              node[2]      = esz + nsz * (real)k;
            }
          }
        }
      }
    }
  }

  // throw a bump in here
  core_geometry box_domain = core;  // store simple box for reference
  for (u64 e = 0; e < core.nelem; ++e)
  {
    for (u64 node_num = 0; node_num < qp1 * qp1 * qp1; ++node_num)
    {
      real* lin_node = box_domain.node(e, node_num);
      real x         = lin_node[0];
      real lin_y     = lin_node[1];
      real func      = bmp_height * exp(-((x - bmp_center) * (x - bmp_center)) /
                                   (2. * bmp_variance));
      real t1        = gly;
      real t2        = gly;
      real b1        = 0.;
      real b2        = func;
      real p1        = lin_y;
      real p2        = ((p1 - b1) / (t1 - b1)) * (t2 - b2) + b2;

      real* node = core.node(e, node_num);
      node[1]    = p2;
    }
  }
}
