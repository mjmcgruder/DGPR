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

#include <cstring>
#include <utility>

#include "error.cpp"
#include "helper_types.cpp"
#include "utils.cpp"

/* array -------------------------------------------------------------------- */

template<typename T, u64 ndim = 1>
struct array
{
  u64 len;
  u64 dims[ndim];
  T* data;

  // --

  array();
  array(u64 len_);
  array(std::initializer_list<u64>);

  array(const array& oth);
  array(array&& oth) noexcept;

  array& operator=(const array& oth);
  array& operator=(array&& oth) noexcept;

  ~array();

  // --

  T& operator[](u64 indx);
  const T& operator[](u64 indx) const;
  T* operator+(u64 num);
  T operator*();

  // --

  void shuffle(std::initializer_list<size_t> permutation, T* output);
};

template<typename T, u64 ndim>
array<T, ndim>::array() : len(0), dims{}, data(nullptr)
{}

template<typename T, u64 ndim>
array<T, ndim>::array(u64 len_) : len(len_), dims{1}, data(new T[len]())
{}

template<typename T, u64 ndim>
array<T, ndim>::array(std::initializer_list<u64> _dims)
{
  if (_dims.size() != ndim)
    errout("attempted to initialize array with incorrect dimension count %d, "
           "expected %d",
           _dims.size(), ndim);

  len    = 1;
  u64 di = 0;

  for (const u64& d : _dims)
  {
    dims[di] = d;
    len *= dims[di];
    ++di;
  }

  data = new T[len];
}

template<typename T, u64 ndim>
void array<T, ndim>::shuffle(std::initializer_list<u64> permutation, T* output)
{
  if (permutation.size() != ndim)
    errout("array permutation has incorrect dimension count %d, expected %d",
           permutation.size(), ndim);

  u64 permuted_dims[ndim];
  {
    u64 di = 0;
    for (const u64& p : permutation)
    {
      permuted_dims[di] = dims[p];
      ++di;
    }
  }

  u64 indx_i[ndim] = {};
  for (u64 i = 0; i < len; ++i)
  {
    /* shuffle data */

    u64 indx_o[ndim];
    {
      u64 di = 0;
      for (const u64& p : permutation)
      {
        indx_o[di] = indx_i[p];
        ++di;
      }
    }

    u64 lindx_i = linear_index(ndim, dims, indx_i);
    u64 lindx_o = linear_index(ndim, permuted_dims, indx_o);

    output[lindx_o] = data[lindx_i];

    /* update index */

    if (indx_i[0] == (dims[0] - 1))
    {
      for (u64 di = 1; di < ndim; ++di)
      {
        if (indx_i[di] != dims[di] - 1)
        {
          ++indx_i[di];
          for (int dj = di - 1; dj > -1; --dj)
          {
            indx_i[dj] = 0;
          }
          break;
        }
      }
    }
    else
    {
      ++indx_i[0];
    }
  }
}

template<typename T, u64 ndim>
array<T, ndim>::array(const array<T, ndim>& oth) :
len(oth.len), data(new T[len]())
{
  for (u64 i = 0; i < len; ++i)
    data[i] = oth.data[i];
}

template<typename T, u64 ndim>
array<T, ndim>::array(array<T, ndim>&& oth) noexcept :
len(oth.len), data(oth.data)
{
  oth.len  = 0;
  oth.data = nullptr;
}

template<typename T, u64 ndim>
array<T, ndim>& array<T, ndim>::operator=(const array<T, ndim>& oth)
{
  delete[] data;
  len  = oth.len;
  data = new T[len]();
  for (u64 i = 0; i < len; ++i)
    data[i] = oth.data[i];
  return *this;
}

template<typename T, u64 ndim>
array<T, ndim>& array<T, ndim>::operator=(array<T, ndim>&& oth) noexcept
{
  delete[] data;
  len      = oth.len;
  data     = oth.data;
  oth.len  = 0;
  oth.data = nullptr;
  return *this;
}

template<typename T, u64 ndim>
array<T, ndim>::~array()
{
  delete[] data;
}

template<typename T, u64 ndim>
T& array<T, ndim>::operator[](u64 indx)
{
  return data[indx];
}

template<typename T, u64 ndim>
const T& array<T, ndim>::operator[](u64 indx) const
{
  return data[indx];
}

template<typename T, u64 ndim>
T* array<T, ndim>::operator+(u64 num)
{
  return data + num;
}

template<typename T, u64 ndim>
T array<T, ndim>::operator*()
{
  return *data;
}

/* string ------------------------------------------------------------------- */

struct string
{
  array<char> data;

  string();
  string(const char* str);

  u64 len() const;
  const char* c_str() const;
  void copy_to(char* buff) const;

  char& operator[](u64 indx);
  const char& operator[](u64 indx) const;
  bool operator==(const string& oth) const;
  bool operator!=(const string& oth) const;
};

string::string() : data()
{}

string::string(const char* str)
{
  size_t len = strlen(str);
  data       = array<char>(len + 1);
  for (u64 i = 0; i < len; ++i)
    data[i] = str[i];
  data[len] = '\0';
}

u64 string::len() const
{
  return data.len - 1;
}

const char* string::c_str() const
{
  return data.data;
}

void string::copy_to(char* buff) const
{
  for (u64 i = 0; i < data.len; ++i)
    buff[i] = data[i];
}

char& string::operator[](u64 indx)
{
  return data[indx];
}

const char& string::operator[](u64 indx) const
{
  return data[indx];
}

bool string::operator==(const string& oth) const
{
  if (data.len != oth.data.len)
    return false;

  for (u64 i = 0; i < data.len; ++i)
  {
    if (operator[](i) != oth[i])
      return false;
  }

  return true;
}

bool string::operator!=(const string& oth) const
{
  return !operator==(oth);
}

/* linked list -------------------------------------------------------------- */

template<typename T>
struct list
{
  struct node
  {
    node* next;

    string key;
    T val;

    node(const string& key_, const T& val_);
    node(const string& key_, T&& val_);
  };

  struct iterator
  {
    node* here;

    iterator();
    iterator(node* here_);

    const string& key();
    T& val();
    iterator& operator++();    // prefix increment
    iterator operator++(int);  // postfix increment
    bool operator==(const iterator& oth) const;
    bool operator!=(const iterator& oth) const;
  };

  node* start;

  list();
  ~list();

  iterator begin();
  iterator end();

  T& add(const string& key, const T& obj);
  T& add(const string& key, T&& obj);
  void remove(const string& key);
  T& get(const string& key);
  T* get_strict(const string& key);
  void clear();
};

// node implementation

template<typename T>
list<T>::node::node(const string& key_, const T& val_) :
next(nullptr), key(key_), val(val_)
{}

template<typename T>
list<T>::node::node(const string& key_, T&& val_) :
next(nullptr), key(key_), val(std::move(val_))
{}

// iterator implementation

template<typename T>
list<T>::iterator::iterator() : here(nullptr)
{}

template<typename T>
list<T>::iterator::iterator(node* here_) : here(here_)
{}

template<typename T>
const string& list<T>::iterator::key()
{
  return here->key;
}

template<typename T>
T& list<T>::iterator::val()
{
  return here->val;
}

template<typename T>
typename list<T>::iterator& list<T>::iterator::operator++()
{
  here = here->next;
  return *this;
}

template<typename T>
typename list<T>::iterator list<T>::iterator::operator++(int)
{
  list<T>::iterator tmp(here);
  ++(*this);
  return tmp;
}

template<typename T>
bool list<T>::iterator::operator==(const iterator& oth) const
{
  if (here == oth.here)
    return true;
  return false;
}

template<typename T>
bool list<T>::iterator::operator!=(const iterator& oth) const
{
  return !operator==(oth);
}

// list implementation

template<typename T>
list<T>::list() : start(nullptr)
{}

template<typename T>
list<T>::~list<T>()
{
  clear();
}

template<typename T>
typename list<T>::iterator list<T>::begin()
{
  return iterator(start);
}

template<typename T>
typename list<T>::iterator list<T>::end()
{
  return iterator();
}

template<typename T>
void list<T>::clear()
{
  node* here = start;
  while (here != nullptr)
  {
    node* next = here->next;
    delete here;
    here = next;
  }
  start = nullptr;
}

template<typename T>
T& list<T>::add(const string& key, const T& obj)
{
  node* here = start;
  while (here != nullptr)
  {
    if (key == here->key)
    {
      here->val = obj;
      return here->val;
    }
    if (here->next == nullptr)
    {
      here->next = new node(key, obj);
      return here->next->val;
    }
    here = here->next;
  }
  start = new node(key, obj);
  return start->val;
}

template<typename T>
T& list<T>::add(const string& key, T&& obj)
{
  node* here = start;
  while (here != nullptr)
  {
    if (key == here->key)
    {
      here->val = std::move(obj);
      return here->val;
    }
    if (here->next == nullptr)
    {
      here->next = new node(key, std::move(obj));
      return here->next->val;
    }
    here = here->next;
  }
  start = new node(key, std::move(obj));
  return start->val;
}

template<typename T>
void list<T>::remove(const string& key)
{
  node* last = nullptr;
  node* here = start;
  while (here != nullptr && key != here->key)
  {
    last = here;
    here = here->next;
  }
  if (here == start && start != nullptr)
  {
    node* next = here->next;
    delete here;
    start = next;
  }
  else if (here != nullptr)
  {
    node* next = here->next;
    delete here;
    last->next = next;
  }
}

template<typename T>
T& list<T>::get(const string& key)
{
  node* here = start;
  while (here != nullptr && key != here->key)
    here = here->next;
  if (here == nullptr)
    return add(key, T{});
  else
    return here->val;
}

template<typename T>
T* list<T>::get_strict(const string& key)
{
  node* here = start;
  while (here != nullptr && key != here->key)
    here = here->next;
  if (here == nullptr)
    return nullptr;
  else
    return &(here->val);
}

/* hash table --------------------------------------------------------------- */

// djb2 hash function
u32 djb2(const string& key)
{
  u32 hash = 5381;
  for (u32 i = 0; i < key.len(); ++i)
    hash = hash * 33 + (u32)key[i];
  return hash;
}

template<typename T>
struct map
{
  static constexpr u64 entries_exponent = 8;
  static constexpr u64 entries          = 1 << entries_exponent;

  static u32 hash_func(const string& key);

  struct iterator
  {
    typename list<T>::node* here;  // next node to be read
    list<T>* table;
    u32 entry;

    iterator();                 // gives off the end iterator
    iterator(list<T>* table_);  // gives iterator to first valid node (if any)

    const string& key();
    T& val();
    iterator& operator++();    // prefix increment
    iterator operator++(int);  // postfix increment
    bool operator==(const iterator& oth) const;
    bool operator!=(const iterator& oth) const;
  };

  list<T> table[entries];

  map();

  iterator begin();
  iterator end();

  T& add(const string& key, const T& val);
  T& add(const string& key, const T&& val);
  void remove(const string& key);
  T& get(const string& key);
  T* get_strict(const string& key);
};

// iterator implementation

template<typename T>
map<T>::iterator::iterator() : here(nullptr), table(nullptr), entry(entries)
{}

template<typename T>
map<T>::iterator::iterator(list<T>* table_) :
here(nullptr), table(table_), entry(0)
{
  operator++();  // upate iterator to first valid node
}

template<typename T>
const string& map<T>::iterator::key()
{
  return here->key;
}

template<typename T>
T& map<T>::iterator::val()
{
  return here->val;
}

template<typename T>
typename map<T>::iterator& map<T>::iterator::operator++()
{
  // update iterator if already at a node inside a linked list
  if (here != nullptr)
  {
    for (typename list<T>::iterator i = ++(typename list<T>::iterator(here));
         i != table[entry].end();
         ++i)
    {
      here = i.here;
      return *this;
    }
    ++entry;
  }

  // if no new node found in the same list, search later lists in the table
  while (entry < map<T>::entries)
  {
    for (typename list<T>::iterator i = table[entry].begin();
         i != table[entry].end();
         ++i)
    {
      here = i.here;
      return *this;
    }
    ++entry;
  }

  // return the one off the end iterator (entry will be 1 past valid range)
  here  = nullptr;
  table = nullptr;
  return *this;
}

template<typename T>
typename map<T>::iterator map<T>::iterator::operator++(int)
{
  iterator tmp = *this;
  ++(*this);
  return tmp;
}

template<typename T>
bool map<T>::iterator::operator==(const iterator& oth) const
{
  if (here == oth.here && table == oth.table && entry == oth.entry)
    return true;
  return false;
}

template<typename T>
bool map<T>::iterator::operator!=(const iterator& oth) const
{
  return !operator==(oth);
}

// map implementation

template<typename T>
u32 map<T>::hash_func(const string& key)
{
  return djb2(key) % entries;
}

template<typename T>
map<T>::map() : table{}
{}

template<typename T>
typename map<T>::iterator map<T>::begin()
{
  return iterator(table);
}

template<typename T>
typename map<T>::iterator map<T>::end()
{
  return iterator();
}

template<typename T>
T& map<T>::add(const string& key, const T& val)
{
  u32 hash = hash_func(key);
  return table[hash].add(key, val);
}

template<typename T>
T& map<T>::add(const string& key, const T&& val)
{
  u32 hash = hash_func(key);
  return table[hash].add(key, std::move(val));
}

template<typename T>
void map<T>::remove(const string& key)
{
  u32 hash = hash_func(key);
  table[hash].remove(key);
}

template<typename T>
T& map<T>::get(const string& key)
{
  u32 hash = hash_func(key);
  return table[hash].get(key);
}

template<typename T>
T* map<T>::get_strict(const string& key)
{
  u32 hash = hash_func(key);
  return table[hash].get_strict(key);
}
