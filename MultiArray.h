//
// Created by di miao on 12/29/16.
//

#ifndef OO_IGA_MULTIARRAY_H
#define OO_IGA_MULTIARRAY_H

#include <array>
#include <vector>

template <class T, size_t I, size_t... J>
struct MultiDimArray
{
    using Nested = typename MultiDimArray<T, J...>::type;
    // typedef typename MultiDimArray<T, J...>::type Nested;
    using type = std::array<Nested, I>;
    // typedef std::array<Nested, I> type;
};

template <class T, size_t I>
struct MultiDimArray<T, I>
{
    using type = std::array<T, I>;
    // typedef std::array<T, I> type;
};


#endif //OO_IGA_MULTIARRAY_H
