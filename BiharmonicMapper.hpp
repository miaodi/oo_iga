//
// Created by di miao on 10/23/17.
//

#pragma once

#include "AbstractMapper.hpp"

template<int N, typename T>
class BiharmonicMapper : public AbstractMapper<1, N, T>
{
public:
    BiharmonicMapper(DofMapper<N, T> &dofMap) : AbstractMapper<1, N, T>(dofMap)
    {
    }
};
