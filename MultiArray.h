//
// Created by di miao on 12/29/16.
//

#ifndef OO_IGA_MULTIARRAY_H
#define OO_IGA_MULTIARRAY_H

#include <array>
#include <vector>

namespace details {
    template<class T>struct tag{using type=T;};
    template<class Tag>using type=typename Tag::type;

    template<class T, size_t n>
    struct n_dim_vec:tag< std::vector< type< n_dim_vec<T, n-1> > > > {};
    template<class T>
    struct n_dim_vec<T, 0>:tag<T>{};
    template<class T, size_t n>
    using n_dim_vec_t = type<n_dim_vec<T,n>>;

    template <class T, class R=std::vector<T>>
    R make_vector(size_t size) {
        return R(size);
    }

    template<class T, class...Args, class R=n_dim_vec_t<T, sizeof...(Args)+1>>
    R make_vector(size_t top, Args...args){
        return R(top, make_vector<T>(std::forward<Args>(args)...));
    }
}


template <class T, class... Args, class R=details::n_dim_vec_t<T, sizeof...(Args)>>
R make_vector(Args... args)
{
    return details::make_vector<T>(std::forward<Args>(args)...);
}

#endif //OO_IGA_MULTIARRAY_H
