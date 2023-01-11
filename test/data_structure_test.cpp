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

#include <cinttypes>

#include <mpitest.h>

#include "data_structures.cpp"

TEST(string_test, 1)
{
  string bar;
  string foo("top of the morning");

  bar = foo;

  EXPECT_TRUE(bar == foo);
  EXPECT_TRUE(!(bar != foo));

  EXPECT_EQ(foo.len(), (u64)18);
  EXPECT_EQ(bar.len(), (u64)18);

  EXPECT_EQ(foo[4], 'o');
  EXPECT_EQ(bar[4], foo[4]);
}

TEST(linked_lisit_test, 1)
{
  int meaning = 42;
  int leet    = 1337;
  int leet2   = 9002;
  int lucky   = 7;

  list<int> storage;

  storage.add("meaning", meaning);
  storage.add("leet", leet);
  storage.add("lucky", lucky);
  storage.add("two power five", 32);
  storage.add("devil or something", 666);

  EXPECT_EQ(storage.get("devil or something"), 666);
  EXPECT_EQ(storage.get("two power five"), 32);
  EXPECT_EQ(storage.get("lucky"), lucky);
  EXPECT_EQ(storage.get("leet"), leet);
  EXPECT_EQ(storage.get("meaning"), meaning);

  storage.add("leet", 9001);

  EXPECT_EQ(storage.get("leet"), 9001);

  int& stuff = storage.get("stuff");
  stuff      = leet2;

  EXPECT_EQ(storage.get("stuff"), leet2);

  storage.remove("two power five");
  storage.remove("lucky");
  storage.remove("clutter");
  storage.remove("anger");
  storage.remove("meaning");  // :(

  int c = 0;
  for (auto i = storage.begin(); i != storage.end(); ++i, ++c)
  {
    if (c == 0)
      EXPECT_EQ(i.val(), 9001);
    else if (c == 1)
      EXPECT_EQ(i.val(), 666);
    else if (c == 2)
      EXPECT_EQ(i.val(), 9002);
  }
}

TEST(hash_table_test, 1)
{
  int meaning = 42;
  int leet    = 1337;
  int leet2   = 9002;
  int lucky   = 7;

  map<int> storage;

  storage.add("meaning", meaning);
  storage.add("leet", leet);
  storage.add("lucky", lucky);
  storage.add("two power five", 32);
  storage.add("devil or something", 666);

  EXPECT_EQ(storage.get("devil or something"), 666);
  EXPECT_EQ(storage.get("two power five"), 32);
  EXPECT_EQ(storage.get("lucky"), lucky);
  EXPECT_EQ(storage.get("leet"), leet);
  EXPECT_EQ(storage.get("meaning"), meaning);

  storage.add("leet", 9001);

  EXPECT_EQ(storage.get("leet"), 9001);

  int& stuff = storage.get("stuff");
  stuff      = leet2;

  EXPECT_EQ(storage.get("stuff"), leet2);

  storage.remove("two power five");
  storage.remove("lucky");
  storage.remove("clutter");
  storage.remove("anger");
  storage.remove("meaning");

  int c = 0;
  int found = 0;
  for (auto i = storage.begin(); i != storage.end(); ++i, ++c)
  {
    if (i.val() == 9001 || i.val() == 666 || i.val() == 9002)
      ++found;
  }
  EXPECT_EQ(found, 3);
  EXPECT_EQ(c, 3);
}
