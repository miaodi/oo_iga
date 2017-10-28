//
// Created by di miao on 10/16/17.
//

#pragma once

#include "AbstractMapper.hpp"

template <int N, typename T>
class PoissonMapper : public AbstractMapper<0, N, T>
{
public:
  PoissonMapper(DofMapper<N, T> &dofMap) : AbstractMapper<0, N, T>(dofMap)
  {
  }
  ~PoissonMapper()
  {
  }
};