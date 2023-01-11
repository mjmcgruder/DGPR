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

#include <utility>

#include "quadrature.cpp"
#include "utils.cpp"

// basis functions -------------------------------------------------------------

// split a 2d tensor product basis function number into its 1d components
static void bfunc_split2(u64 bf, u64 p, u64* bfx, u64* bfy)
{
  u64 pp1 = p + 1;
  split2_cartesian(bf, pp1, bfx, bfy);
}

// split a 3d tensor product basis function number into its 1d components
static void bfunc_split3(u64 bf, u64 p, u64* bfx, u64* bfy, u64* bfz)
{
  u64 pp1 = p + 1;
  split3_cartesian(bf, pp1, pp1, bfx, bfy, bfz);
}

// Returns the number of tensor product basis functions for each dimension
// count.

u64 nbf1d(u64 p)
{
  return (p + 1);
}
u64 nbf2d(u64 p)
{
  return (p + 1) * (p + 1);
}
u64 nbf3d(u64 p)
{
  return (p + 1) * (p + 1) * (p + 1);
}

// Evaluate the lagrange node location of order "p" basis function number "bf".
// These functions generate linearly spaced lagrage nodes over a [0, 1]
// reference space in each direction. For p = 0 the node is arbitrarily placed
// at 0, I think this should work... The Lagrange basis functions are ordered
// such that their nodes appear in the following order:
//   z positions
//     y positions
//       x positions
real lagrange_node1d(u64 bf, u64 p)
{
  if (p == 0)
    return 0.;
  else
    return (real)bf / (real)p;
}

v2 lagrange_node2d(u64 bf, u64 p)
{
  u64 bfx, bfy;
  bfunc_split2(bf, p, &bfx, &bfy);
  return v2(lagrange_node1d(bfx, p), lagrange_node1d(bfy, p));
}

v3 lagrange_node3d(u64 bf, u64 p)
{
  u64 bfx, bfy, bfz;
  bfunc_split3(bf, p, &bfx, &bfy, &bfz);
  return v3(lagrange_node1d(bfx, p), lagrange_node1d(bfy, p),
            lagrange_node1d(bfz, p));
}

// Evaluates Lagrange polynomial number "bf" (zero based) of order "p" at the
// requested position "pos". The order of the polynomials is as specified above
// for their node locations.

real lagrange1d(u64 i, u64 p, real x)
{
  real xj, eval = 1.;
  const real xi = (real)i / (real)p;
  for (u64 j = 0; j < p + 1; ++j)
  {
    if (i != j)
    {
      xj = (real)j / (real)p;
      eval *= (x - xj) / (xi - xj);
    }
  }
  return eval;
}

real lagrange2d(u64 bf, u64 p, v2 pos)
{
  u64 bfx, bfy;
  bfunc_split2(bf, p, &bfx, &bfy);
  return lagrange1d(bfx, p, pos.x) * lagrange1d(bfy, p, pos.y);
}

real lagrange3d(u64 bf, u64 p, v3 pos)
{
  u64 bfx, bfy, bfz;
  bfunc_split3(bf, p, &bfx, &bfy, &bfz);
  return lagrange1d(bfx, p, pos.x) * lagrange1d(bfy, p, pos.y) *
         lagrange1d(bfz, p, pos.z);
}

// Evaluates the reference space gradient of the order "p" Lagrange basis
// function number "bf" at the requested location. The ordering here is also
// specified by the node ordering above.

real lagrange1d_deriv(u64 i, u64 p, real x)
{
  real a_acc = 0.0;  // additive accumulator
  real m_acc;        // multiplicative accumulator
  const real xi = (real)i / (real)p;
  for (u64 j = 0; j < p + 1; ++j)
  {
    if (j != i)
    {
      m_acc   = 1.0;
      real xj = (real)j / (real)p;
      for (u64 m = 0; m < p + 1; ++m)
      {
        if (m != i && m != j)
        {
          real xm = (real)m / (real)p;
          m_acc *= (x - xm) / (xi - xm);
        }
      }
      a_acc += (1.0 / (xi - xj)) * m_acc;
    }
  }
  return a_acc;
}

v2 lagrange2d_grad(u64 bf, u64 p, v2 pos)
{
  u64 bfx, bfy;
  bfunc_split2(bf, p, &bfx, &bfy);
  const real phix = lagrange1d(bfx, p, pos.x);
  const real phiy = lagrange1d(bfy, p, pos.y);
  return v2(lagrange1d_deriv(bfx, p, pos.x) * phiy,
            phix * lagrange1d_deriv(bfy, p, pos.y));
}

v3 lagrange3d_grad(u64 bf, u64 p, v3 pos)
{
  u64 bfx, bfy, bfz;
  bfunc_split3(bf, p, &bfx, &bfy, &bfz);
  const real phix = lagrange1d(bfx, p, pos.x);
  const real phiy = lagrange1d(bfy, p, pos.y);
  const real phiz = lagrange1d(bfz, p, pos.z);
  return v3(lagrange1d_deriv(bfx, p, pos.x) * phiy * phiz,
            phix * lagrange1d_deriv(bfy, p, pos.y) * phiz,
            phix * phiy * lagrange1d_deriv(bfz, p, pos.z));
}

// For a hexagonal element, converts a 2D reference space location on face
// number "face" to a 3D reference space location. This function accounts for
// the orientation of the 2D reference space axes in 3D space to make the
// conversion. The intent of this function is to formalize that conversion.
// The face ordering:
// 0: -x, 1: +x, 2: -y, 3: +y, 4: -z, 5: +z
// TODO: you should probably provide some ascii art here...

v3 surf2vol(u64 face, v2 s)
{
  switch (face)
  {
    case 0: return v3(0.0, s.x, s.y); break;
    case 1: return v3(1.0, s.x, s.y); break;
    case 2: return v3(s.y, 0.0, s.x); break;
    case 3: return v3(s.y, 1.0, s.x); break;
    case 4: return v3(s.x, s.y, 0.0); break;
    case 5: return v3(s.x, s.y, 1.0); break;
  }
  return v3();
}

void surf2vol(u64 face, u64 i_i, u64 i_j, u64 n, u64* o_i, u64* o_j, u64* o_k)
{
  switch (face)
  {
    case 0:
      *o_i = 0;
      *o_j = i_i;
      *o_k = i_j;
      break;
    case 1:
      *o_i = n - 1;
      *o_j = i_i;
      *o_k = i_j;
      break;
    case 2:
      *o_i = i_j;
      *o_j = 0;
      *o_k = i_i;
      break;
    case 3:
      *o_i = i_j;
      *o_j = n - 1;
      *o_k = i_i;
      break;
    case 4:
      *o_i = i_i;
      *o_j = i_j;
      *o_k = 0;
      break;
    case 5:
      *o_i = i_i;
      *o_j = i_j;
      *o_k = n - 1;
      break;
  }
}

// For a hexagonal element on this code's structured domain where relative
// orientation of elements is always the same, converts a given local face
// number into the opposite local face number over a face.
u64 opposite_local_face[6] = {1, 0, 3, 2, 5, 4};

// basis pre-compute structure -------------------------------------------------

// This struct provides pre-evaluations and quadrature information over a unit
// cube reference space ([0, 1] in the x, y, and z directions) for an order "p"
// reference element. The quadrature rules will integrate a 3D
// polynomial of order "p" or less in each direction exactly. This makes the
// rules unsuitable for the mass matrix calculation so that is (should be
// anyway) handled with the "real space" geometry pre-computes like Jacobians.
// Actually that has to be the case for all except affine reference to global
// mappings (which I believe only occur for linear simplex elements) because the
// reference to global mapping needs to be included in the integral for
// generating the mass matrix for a nonlinear mapping.
struct basis
{
  u64 p;         // polynomial order of this reference space info
  u64 quad_ord;  // this basis is pre-evaluated for this quad order
  u64 nbf2d;     // 2d basis function count
  u64 nbf3d;     // 3d basis function count

  // 3D quad rule over the volume of this reference space
  quad vqrule;  // "volume quadrature rule"

  // 2D quadrule for integration over faces
  quad fqrule;  // "face quadrature rule"

  // Stores each basis function pre-evaluated at element interior (volume)
  // quadrature points. The array format is as follows:
  //   quad nodes (ordered by "quad" structure)
  //     basis functions (ordered by lagrange_node*d functions)
  array<real> veval;  // "volume basis function evaluation"

  // Stores the gradient of each basis function pre-evaluated at volume
  // quadrature points. The array format is as follows:
  //   each basis function (ordered by lagrange_node*d functions)
  //     each quadrature point (ordered by "quad" structure)
  //       each gradient stored [dphi_dx, dphi_dy, dphi_dz]
  array<real> vgrad;  // "volume basis function gradient evaluation"

  // Stores each 3D basis function evaluated at each quadrature point on each
  // face of the reference element. It's important to note that in the
  // translation from the 3D coordinate system into 2D planes for the faces,
  // that the evaluation ordering for opposite faces is kept consistent so the
  // client function does not need to keep track of multiple sets of indices to
  // compute and integrate a flux over neighboring elements.
  // The face ordering:
  // 0: -x, 1: +x, 2: -y, 3: +y, 4: -z, 5: +z
  // The array format is as follows:
  //   each local face (in order above)
  //     each quadrature point (ordered by "quad" structure (for 2D))
  //       each basis function (ordered by lagrange_node*d functions)
  array<real> feval;  // "face basis function evaluation"

  // Stores the evaluation of each basis function evaluated at each quadrature
  // point on each face of the reference element. This array uses the same face
  // ordering aas feval above.
  // The format is as follows:
  //   each local face (same order as feval)
  //     each basis function (ordered by lagrange_node*d functions)
  //       each quadrature point (ordered by "quad" structure (for 2D))
  //         gradient stored [d_dx, d_dy, d_dz]
  array<real> fgrad;

  // constructors

  basis();                      // just convenient
  basis(u64 p_, u64 quad_ord);  // the one true constructor

  // pre-compute accessors (for sheer convenience)

  real& veval_at(u64 bfnc, u64 qpnt);
  real* vgrad_at(u64 bfnc, u64 qpnt);
  real& vgrad_at(u64 bfnc, u64 qpnt, component comp);
  real& feval_at(u64 local_face_indx, u64 bfnc, u64 qpnt);
  real* fgrad_at(u64 local_face_indx, u64 bfnc, u64 qpnt);
  real& fgrad_at(u64 local_face_indx, u64 bfnc, u64 qpnt, component comp);

 private:
  // some functions made for clarity, they're just called on construction...
  void pre_eval_veval();
  void pre_eval_vgrad();
  void pre_eval_feval();
  void pre_eval_fgrad();
};

// indexing functions ----------------------------------------------------------

real& basis::veval_at(u64 bfnc, u64 qpnt)
{
  return veval[nbf3d * qpnt + bfnc];
}
real* basis::vgrad_at(u64 bfnc, u64 qpnt)
{
  return vgrad + (3 * (vqrule.n * bfnc + qpnt));
}
real& basis::vgrad_at(u64 bfnc, u64 qpnt, component comp)
{
  return vgrad[3 * (vqrule.n * bfnc + qpnt) + (u64)comp];
}
real& basis::feval_at(u64 local_face_indx, u64 bfnc, u64 qpnt)
{
  return feval[nbf3d * fqrule.n * local_face_indx + nbf3d * qpnt + bfnc];
}
real* basis::fgrad_at(u64 local_face_indx, u64 bfnc, u64 qpnt)
{
  return fgrad +
         ((nbf3d * fqrule.n * local_face_indx + fqrule.n * bfnc + qpnt) * 3);
}
real& basis::fgrad_at(u64 local_face_indx, u64 bfnc, u64 qpnt, component comp)
{
  real* grad = fgrad_at(local_face_indx, bfnc, qpnt);
  return grad[(u64)comp];
}

// basis pre-compute functions -------------------------------------------------

void basis::pre_eval_veval()
{
  for (u64 bfn = 0; bfn < nbf3d; ++bfn)
  {
    for (u64 qnn = 0; qnn < vqrule.n; ++qnn)
    {
      veval_at(bfn, qnn) = lagrange3d(bfn, p, vqrule.pos3d(qnn));
    }
  }
}

void basis::pre_eval_vgrad()
{
  for (u64 bfn = 0; bfn < nbf3d; ++bfn)
  {
    for (u64 qnn = 0; qnn < vqrule.n; ++qnn)
    {
      v3 grad         = lagrange3d_grad(bfn, p, vqrule.pos3d(qnn));
      real* vgrad_pos = vgrad_at(bfn, qnn);
      vgrad_pos[0]    = grad.x;
      vgrad_pos[1]    = grad.y;
      vgrad_pos[2]    = grad.z;
    }
  }
}

void basis::pre_eval_feval()
{
  for (u64 face = 0; face < 6; ++face)
  {
    for (u64 bfn = 0; bfn < nbf3d; ++bfn)
    {
      for (u64 qnn = 0; qnn < fqrule.n; ++qnn)
      {
        feval_at(face, bfn, qnn) =
        lagrange3d(bfn, p, surf2vol(face, fqrule.pos2d(qnn)));
      }
    }
  }
}

void basis::pre_eval_fgrad()
{
  for (u64 face = 0; face < 6; ++face)
  {
    for (u64 bfn = 0; bfn < nbf3d; ++bfn)
    {
      for (u64 qnn = 0; qnn < fqrule.n; ++qnn)
      {
        v3 grad = lagrange3d_grad(bfn, p, surf2vol(face, fqrule.pos2d(qnn)));
        fgrad_at(face, bfn, qnn, component::x) = grad.x;
        fgrad_at(face, bfn, qnn, component::y) = grad.y;
        fgrad_at(face, bfn, qnn, component::z) = grad.z;
      }
    }
  }
}

// basis construction and memory management ------------------------------------

basis::basis() :
p(0), nbf2d(0), nbf3d(0), vqrule(), fqrule(), veval(), vgrad(), feval(), fgrad()
{}

basis::basis(u64 p_, u64 quad_ord_) :
p(p_),
quad_ord(quad_ord_),
nbf2d((p + 1) * (p + 1)),
nbf3d((p + 1) * (p + 1) * (p + 1))
{
  vqrule =
  gauss_legendre_3d(quad_ord, quad_ord, quad_ord, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  fqrule = gauss_legendre_2d(quad_ord, quad_ord, 0.0, 1.0, 0.0, 1.0);
  veval  = array<real>(nbf3d * vqrule.n);
  vgrad  = array<real>(nbf3d * vqrule.n * 3);
  feval  = array<real>(nbf3d * fqrule.n * 6);
  fgrad  = array<real>(nbf3d * fqrule.n * 6 * 3);
  pre_eval_veval();
  pre_eval_vgrad();
  pre_eval_feval();
  pre_eval_fgrad();
}
