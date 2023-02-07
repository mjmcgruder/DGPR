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

#include <random>

#include <mpitest.h>

#include "dgmath.cpp"
#include "solve_cuda2.cu"
#include "data_structures.cpp"

void run_cugpu_test(u32 m, u32 n, u32 k)
{
  std::mt19937 mt(42);
  std::uniform_real_distribution<float> dst(0., 1.);

  array<float> A(m * k);
  array<float> B(k * n);
  array<float> Ccpu(m * n);
  array<float> Cgpu0(m * n);
  array<float> Cgpu1(m * n);

  for (u32 i = 0; i < m * k; ++i)
    A[i] = dst(mt);
  for (u32 i = 0; i < k * n; ++i)
    B[i] = dst(mt);
  for (u32 i = 0; i < m * n; ++i)
    Ccpu[i] = 0.f;

  float *d_A, *d_B, *d_C0, *d_C1;

  cudaMalloc(&d_A, A.len * sizeof(*d_A));
  cudaMalloc(&d_B, B.len * sizeof(*d_B));
  cudaMalloc(&d_C0, Cgpu0.len * sizeof(*d_C0));
  cudaMalloc(&d_C1, Cgpu1.len * sizeof(*d_C1));

  cudaMemcpy(d_A, A.data, A.len * sizeof(*d_A), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data, B.len * sizeof(*d_B), cudaMemcpyHostToDevice);

  double cpu_start = MPI_Wtime();
  matmul(m, k, n, 1.f, A.data, B.data, 0.f, Ccpu.data);
  double cpu_end = MPI_Wtime();

  dim3 block(16, 16);
  dim3 grid((n + (block.x - 1)) / block.x, 
            (m + (block.y - 1)) / block.y);

  double gpu_start0 = MPI_Wtime();
  cuda_gemm_0<<<grid, block>>>(m, k, n, d_A, d_B, d_C0);
  cudaDeviceSynchronize();
  double gpu_end0 = MPI_Wtime();

  double gpu_start1 = MPI_Wtime();
  cuda_gemm_1<<<grid, block>>>(m, k, n, d_A, d_B, d_C1);
  cudaDeviceSynchronize();
  double gpu_end1 = MPI_Wtime();

  printf("cpu: %.5fs gpu0: %.5fs gpu1: %.5fs\n", 
         cpu_end - cpu_start, gpu_end0 - gpu_start0, gpu_end1 - gpu_start1);

  cudaMemcpy(Cgpu0.data, d_C0, Cgpu0.len * sizeof(*d_C0), 
             cudaMemcpyDeviceToHost);
  cudaMemcpy(Cgpu1.data, d_C1, Cgpu1.len * sizeof(*d_C1), 
             cudaMemcpyDeviceToHost);

  for (u64 i = 0; i < Ccpu.len; ++i)
  {
    EXPECT_FLOAT_EQ(Ccpu[i], Cgpu0[i], 10);
    EXPECT_FLOAT_EQ(Ccpu[i], Cgpu1[i], 10);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C0);
  cudaFree(d_C1);
}

TEST(cumath_test_matmul, 1)
{
  run_cugpu_test(1024, 1024, 1024);
}
