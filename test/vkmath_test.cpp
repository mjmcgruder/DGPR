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

TEST(vkmath_test_matmul, 1)
{
  vkinit(false);

  {
    u32 N = 100;

    graphics_pipeline scene_pipeline;
    graphics_pipeline ui_pipeline;
    make_swap_chain_dependencies(scene_pipeline, ui_pipeline);

    compute_pipeline vkmatmul("compute_gemm.spv", 6);

    std::mt19937 mt(42);
    std::uniform_real_distribution<float> dst(0., 1.);

    array<float> A(N * N);
    array<float> B(N * N);
    array<float> Ccpu(N * N);
    array<float> Cgpu(N * N);

    for (u32 r = 0; r < N; ++r)
    {
      for (u32 c = 0; c < N; ++c)
      {
        A[r * N + c]    = dst(mt);
        B[r * N + c]    = dst(mt);
        Ccpu[r * N + c] = 0.f;
        Cgpu[r * N + c] = 0.f;
      }
    }

    buffer_set<u32> d_m(&vkmatmul.dset, 0);
    buffer_set<u32> d_k(&vkmatmul.dset, 1);
    buffer_set<u32> d_n(&vkmatmul.dset, 2);
    buffer_set<float> d_A(&vkmatmul.dset, 3);
    buffer_set<float> d_B(&vkmatmul.dset, 4);
    buffer_set<float> d_C(&vkmatmul.dset, 5);

    d_m.update(&N, 1);
    d_k.update(&N, 1);
    d_n.update(&N, 1);
    d_A.update(A.data, A.len);
    d_B.update(B.data, B.len);
    d_C.update(Cgpu.data, Cgpu.len);  // don't like that you have to do this
                                      // seems like this issue stems from malloc
                                      // and copy being the same operation

    matmul(N, N, N, 1.f, A.data, B.data, 0.f, Ccpu.data);

    vkmatmul.run((N + (TILE_SIZE - 1)) / TILE_SIZE,
                 (N + (TILE_SIZE - 1)) / TILE_SIZE, 1);

    d_C.retrieve(Cgpu.data);

    for (u64 i = 0; i < N * N; ++i)
      EXPECT_FLOAT_EQ(Ccpu[i], Cgpu[i], 10);
  }

  clean();
}
