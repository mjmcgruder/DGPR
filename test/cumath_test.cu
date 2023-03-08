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
  array<float> Cgpu(m * n);

  for (u32 i = 0; i < m * k; ++i)
    A[i] = dst(mt);
  for (u32 i = 0; i < k * n; ++i)
    B[i] = dst(mt);
  for (u32 i = 0; i < m * n; ++i)
    Ccpu[i] = 0.f;

  float *d_A, *d_B, *d_C;

  cudaMalloc(&d_A, A.len * sizeof(*d_A));
  cudaMalloc(&d_B, B.len * sizeof(*d_B));
  cudaMalloc(&d_C, Cgpu.len * sizeof(*d_C));

  cudaMemcpy(d_A, A.data, A.len * sizeof(*d_A), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data, B.len * sizeof(*d_B), cudaMemcpyHostToDevice);

  double cpu_start = MPI_Wtime();
  matmul(m, k, n, 1.f, A.data, B.data, 0.f, Ccpu.data);
  double cpu_end = MPI_Wtime();

  dim3 block(32, 32);
  dim3 grid(idiv_ceil(n, block.x), idiv_ceil(m, block.y));

  double gpu_start = MPI_Wtime();
  cuda_gemm<<<grid, block>>>(m, k, n, d_A, d_B, d_C);
  cudaDeviceSynchronize();
  double gpu_end = MPI_Wtime();

  printf("cpu: %.5fs gpu: %.5fs\n", 
         cpu_end - cpu_start, gpu_end - gpu_start);

  cudaMemcpy(Cgpu.data, d_C, Cgpu.len * sizeof(*d_C), cudaMemcpyDeviceToHost);

  for (u64 i = 0; i < Ccpu.len; ++i)
  {
    EXPECT_FLOAT_EQ(Ccpu[i], Cgpu[i], 10);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

TEST(cumath_test_matmul, 1)
{
  // run_cugpu_test(17, 17, 17);     // small stuff
  // run_cugpu_test(128, 128, 128);  // square, tile divides evenly
  // run_cugpu_test(100, 100, 100);  // square, tile divides unevenly
  // run_cugpu_test(128, 128, 150);  // short and fat C, inner dim divs evenly
  // run_cugpu_test(150, 128, 128);  // tall and skinny C, inner dim divs evenly
  // run_cugpu_test(128, 175, 150);  // short and fat C, inner dim divs unevenly
  // run_cugpu_test(150, 175, 128);  // tall n skinny C, inner dim divs unevenly

  // run_cugpu_test(1024, 1024, 1024);  // largeish, divs evenly
  // run_cugpu_test(900, 1000, 950);    // largeish, divs unevenly
  run_cugpu_test(2000, 4000, 1500);  // more largeish, divs unevenly
}
