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
#include "buffers.cpp"
#include "init.cpp"
#include "rendering.cpp"

#define TILE_SIZE 16

void run_vkgpu_test(u32 m, u32 k, u32 n)
{
  compute_pipeline vkmatmul("compute_gemm.spv", 6);

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
  {
    Ccpu[i] = 0.f;
    Cgpu[i] = 0.f;
  }

  dbuffer<u32> d_m, d_k, d_n;
  dbuffer<float> d_A, d_B, d_C;

  d_m.bind(&vkmatmul.dset, 0);
  d_k.bind(&vkmatmul.dset, 1);
  d_n.bind(&vkmatmul.dset, 2);
  d_A.bind(&vkmatmul.dset, 3);
  d_B.bind(&vkmatmul.dset, 4);
  d_C.bind(&vkmatmul.dset, 5);

  d_m.update(&m, 1);
  d_k.update(&k, 1);
  d_n.update(&n, 1);
  d_A.update(A.data, A.len);
  d_B.update(B.data, B.len);
  d_C.allocate(Cgpu.len);  // don't like that you have to do this
                           // seems like this issue stems from malloc
                           // and copy being the same operation

  double cpu_start = MPI_Wtime();
  matmul(m, k, n, 1.f, A.data, B.data, 0.f, Ccpu.data);
  double cpu_end = MPI_Wtime();

  double gpu_start = MPI_Wtime();
  vkmatmul.run((n + (TILE_SIZE - 1)) / TILE_SIZE,
               (m + (TILE_SIZE - 1)) / TILE_SIZE, 1);
  double gpu_end = MPI_Wtime();

  printf("cpu: %.5fs gpu: %.5fs\n", cpu_end - cpu_start, gpu_end - gpu_start);

  d_C.retrieve(Cgpu.data);

  for (u64 i = 0; i < m * n; ++i)
  {
    EXPECT_FLOAT_EQ(Ccpu[i], Cgpu[i], 10);
  }
}

TEST(vkmath_test_matmul, 1)
{
  vkinit(false);

  make_swap_chain();

  run_vkgpu_test(128, 128, 128);  // square, tile divides evenly
  run_vkgpu_test(100, 100, 100);  // square, tile divides unevenly
  run_vkgpu_test(128, 128, 150);  // short and fat C, inner dim divs evenly
  run_vkgpu_test(150, 128, 128);  // tall and skinny C, inner dim divs evenly
  run_vkgpu_test(128, 175, 150);  // short and fat C, inner dim divs unevenly
  run_vkgpu_test(150, 175, 128);  // tall n skinny C, inner dim divs unevenly

  run_vkgpu_test(1024, 1024, 1024);  // largeish, divs evenly
  run_vkgpu_test(2000, 4000, 1500);  // largeish, divs unevenly

  clean();
}
