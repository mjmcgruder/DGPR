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

#include <cmath>
#include <cstring>
#include <limits>

#include "helper_types.cpp"

#define det2(a, b, c, d) ((a) * (d) - (b) * (c))

v3 abs(v3 a)
{
  return v3(std::abs(a.x), std::abs(a.y), std::abs(a.z));
}

real dot(v3 a, v3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Computes the cross product of three element vectors "a" and "b" and stores
// the result in "c."
void cross(real* a, real* b, real* c)
{
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

// Computes the L2 norm of the real-valued vector "a" with length "len."
real l2(real* a, u64 len)
{
  real acc = 0.0;
  for (u64 i = 0; i < len; ++i)
  {
    acc += a[i] * a[i];
  }
  return sqrt(acc);
}

// Normalizes the vector "a" with its L2 norm in place and returns the result in
// "b." Note that a and b could be the same vector for this function allowing
// the operation to be performed in place. This is possible because the
// magnitude is computed prior to the division.
void normalize(real* a, real* b, u64 len)
{
  real mag = l2(a, len);
  for (u64 i = 0; i < len; ++i)
  {
    b[i] = a[i] / mag;
  }
}

// Computes the determinant of the 3x3 matrix m, m is stored in row-major order.
real det3x3(real* m)
{
  return m[0] * m[4] * m[8] + m[1] * m[5] * m[6] + m[2] * m[3] * m[7] -
         m[2] * m[4] * m[6] - m[1] * m[3] * m[8] - m[0] * m[5] * m[7];
}

// Computes the inverse of the given 3x3 matrix m and stores the result in the
// 3x3 matrix inv. Both matrices are stored in row-major order.
void inv3x3(real* m, real* inv)
{
  real inv_det = 1.0 / det3x3(m);
  inv[0]       = inv_det * det2(m[4], m[5], m[7], m[8]);
  inv[1]       = inv_det * det2(m[2], m[1], m[8], m[7]);
  inv[2]       = inv_det * det2(m[1], m[2], m[4], m[5]);
  inv[3]       = inv_det * det2(m[5], m[3], m[8], m[6]);
  inv[4]       = inv_det * det2(m[0], m[2], m[6], m[8]);
  inv[5]       = inv_det * det2(m[2], m[0], m[5], m[3]);
  inv[6]       = inv_det * det2(m[3], m[4], m[6], m[7]);
  inv[7]       = inv_det * det2(m[1], m[0], m[7], m[6]);
  inv[8]       = inv_det * det2(m[0], m[1], m[3], m[4]);
}

// Performs the matrix multiplication C = alpha * A * B + beta * C (similar to
// BLAS) via the schoolbook method (for now). A is a n x m matrix, B is a
// m x p matrix, and C is an n x p matrix.  Each matrix is stored in row major
// order.
// It is safe (I think) to use this function on unitinialized data. The natural
// thing to do when C is uninitialized is to simply set beta = 0 so the
// uninitialized data doesn't matter. But this doesn't work if the uninitialized
// data is Inf or NaN, in which case simple multiplication by zero will preserve
// the special value. As a result this routine does a little extra work to set
// Inf or NaN input values to zero before proceeding with the calculation.
void matmul(u64 n, u64 m, u64 p, real alpha, real* A, real* B, real beta, 
            real* C)
{
  u64 irow, icol, iacc;
  u64 indC;
  for (irow = 0; irow < n; ++irow)
  {
    for (icol = 0; icol < p; ++icol)
    {
      indC = p * irow + icol;
      real tmp = 0.;
      C[indC] *= beta;
      for (iacc = 0; iacc < m; ++iacc)
      {
        tmp += alpha * A[m * irow + iacc] * B[iacc * p + icol];
      }
      C[indC] += tmp;
    }
  }
}

// Computes the symmetric matrix vector multiplication with a constant multiple
// r = c * M * v.
void symm_mvmul(u64 n, real* M, real* v, real c, real* r)
{
  memset(r, 0, n * sizeof(*r));

  for (u64 irow = 0; irow < n; ++irow)
  {
    real tmp  = 0.;
    real vrow = c * v[irow];
    for (u64 icol = irow + 1; icol < n; ++icol)
    {
      tmp += M[irow * n + icol] * c * v[icol];
      r[icol] += M[irow * n + icol] * vrow;
    }
    r[irow] += M[irow * n + irow] * vrow + tmp;
  }
}

void symmstore_mvmul(u64 n, real* M, real* v, real c, real* r)
{
  memset(r, 0, n * sizeof(*r));

  u64 iM = 0;
  for (u64 irow = 0; irow < n; ++irow)
  {
    ++iM;
    real tmp  = 0.;
    real vrow = c * v[irow];
    for (u64 icol = irow + 1; icol < n; ++icol)
    {
      tmp += M[iM] * c * v[icol];
      r[icol] += M[iM] * vrow;
      ++iM;
    }
    r[irow] += M[iM - (n - irow)] * vrow + tmp;
  }
}

// __attribute__((noinline)) void spmm(u64 n, u64 p, real* M, real* v, real c,
//                                     real* r)
// {
//   memset(r, 0, n * p * sizeof(*r));
// 
//   u64 iM = 0;
//   for (u64 irow = 0; irow < n; ++irow)
//   {
//     ++iM;
//     for (u64 icol = irow + 1; icol < n; ++icol)
//     {
//       real thisM = M[iM];
//       for (u64 ivec = 0; ivec < p; ++ivec)
//       {
//         r[n * ivec + irow] += thisM * c * v[n * ivec + icol];
//         r[n * ivec + icol] += thisM * c * v[n * ivec + irow];
//       }
//       ++iM;
//     }
//     real diagM = M[iM - (n - irow)];
//     for (u64 ivec = 0; ivec < p; ++ivec)
//     {
//       r[n * ivec + irow] += diagM * c * v[n * ivec + irow];
//     }
//   }
// }

void spmm(u64 n, u64 p, real* M, real* v, real c, real* r)
{
  memset(r, 0, n * p * sizeof(*r));

  u64 iM = 0;
  for (u64 irow = 0; irow < n; ++irow)
  {
    ++iM;
    u64 iMstart = iM;
    for (u64 ivec = 0; ivec < p; ++ivec)
    {
      iM        = iMstart;
      real tmp  = 0.;
      real vrow = c * v[n * ivec + irow];
      for (u64 icol = irow + 1; icol < n; ++icol)
      {
        real thisM = M[iM];

        tmp                += thisM * c * v[n * ivec + icol];
        r[n * ivec + icol] += thisM * vrow;

        ++iM;
      }
      r[n * ivec + irow] += M[iM - (n - irow)] * vrow + tmp;
    }
  }
}

// Performs Gaussian elimination with partial pivoting (in place) on the given
// matrix "A" with row-major storage. The resulting matrix will be in row
// echelon form.
void gauss(real* A, u64 nr, u64 nc)
{
#define i2(r, c) (nc * (r) + (c))
  u64 pr, r, c;  // iterators: "pivot row", "row", "column"
  // send to row echelon form
  for (pr = 0; pr < nr - 1; ++pr)
  {
    // perform partial pivoting
    u64 max_abs_row = 0;
    real max_abs    = -std::numeric_limits<real>::max();
    for (r = pr; r < nr; ++r)
    {
      real this_abs = std::abs(A[i2(r, pr)]);
      if (this_abs > max_abs)
      {
        max_abs     = this_abs;
        max_abs_row = r;
      }
    }
    if (max_abs_row != pr)
    {
      for (c = pr; c < nc; ++c)
      {
        real tmp              = A[i2(pr, c)];
        A[i2(pr, c)]          = A[i2(max_abs_row, c)];
        A[i2(max_abs_row, c)] = tmp;
      }
    }
    // zero-out elements under the current pivot
    for (r = pr + 1; r < nr; ++r)
    {
      real f       = -A[i2(r, pr)] / A[i2(pr, pr)];
      A[i2(r, pr)] = 0.0;
      for (c = pr + 1; c < nc; ++c)
      {
        A[i2(r, c)] += f * A[i2(pr, c)];
      }
    }
  }
#undef i2
}

// Takes a matrix "A" in row echelon form (likely just output from the gaussian
// elimination function) and sends it to reduced row echelon
// form (in place).
void jordan(real* A, u64 nr, u64 nc)
{
#define i2(r, c) (nc * (r) + (c))
  u64 pr, pc, r, c;  // iterators: "pivot row", "pivot column", "row", "column"
  // send to reduced row echelon form by setting pivots 1 and entries above to 0
  for (pr = nr - 1; pr < nr; --pr)
  {
    // find the column of this row's pivot
    for (c = pr; c < nc; ++c)
    {
      if (A[i2(pr, c)] != 0.0)
      {
        pc = c;
        break;
      }
    }
    // send the first entry in this row to 1
    real old_pivot = A[i2(pr, pc)];
    A[i2(pr, pc)]  = 1.0;
    for (c = pc + 1; c < nc; ++c)
    {
      A[i2(pr, c)] /= old_pivot;
    }
    // set pivot column in rows above to zero
    for (r = pr - 1; r < pr; --r)
    {
      real f = -A[i2(r, pc)] / A[i2(pr, pc)];
      for (c = r; c < nc; ++c)
      {
        if (c == pc)
        {
          A[i2(r, c)] = 0.0;
        }
        else
        {
          A[i2(r, c)] += f * A[i2(pr, c)];
        }
      }
    }
  }
#undef i2
}

// Inverts the row-major matrix "mat" using gauss-jordan elimination and returns
// the result in the row-major matrix "inv" of the same size;
int invert(real* mat, u64 n)
{
#define imat(r, c) (n * (r) + c)
#define iaug(r, c) (n * 2 * (r) + c)
  real* augmented = new real[2 * n * n];
  // form augmented matrix
  u64 r, c;
  for (r = 0; r < n; ++r)
  {
    for (c = 0; c < n; ++c)
    {
      augmented[iaug(r, c)] = mat[imat(r, c)];
      if (r == c)
      {
        augmented[iaug(r, c + n)] = 1.0;
      }
      else
      {
        augmented[iaug(r, c + n)] = 0.0;
      }
    }
  }
  // invert (hopefully)
  gauss(augmented, n, n * 2);
  jordan(augmented, n, n * 2);
  // check for singularity
  for (r = 0; r < n; ++r)
  {
    for (c = 0; c < n; ++c)
    {
      if (r == c)
      {
        if (augmented[iaug(r, c)] != 1.0)
        {
          delete[] augmented;
          return 1;
        }
      }
      else
      {
        if (augmented[iaug(r, c)] != 0.0)
        {
          delete[] augmented;
          return 1;
        }
      }
    }
  }
  // extract the inversion
  for (r = 0; r < n; ++r)
  {
    for (c = 0; c < n; ++c)
    {
      mat[imat(r, c)] = augmented[iaug(r, c + n)];
    }
  }
  delete[] augmented;
  return 0;
#undef imat
#undef iaug
}
