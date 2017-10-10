//
// Created by di miao on 10/9/17.
//

#ifndef OO_IGA_VERTEX_HPP
#define OO_IGA_VERTEX_HPP

#include "Topology.hpp"

template<int d, int N, typename T>
class Visitor;

template<int N, typename T>
class Vertex : public Element<0, N, T> {
public:
    typedef typename Element<0, N, T>::DomainShared_ptr DomainShared_ptr;
    typedef typename Element<0, N, T>::Coordinate Coordinate;
    typedef typename Element<0, N, T>::PhyPts PhyPts;
    typedef typename Element<0, N, T>::CoordinatePairList CoordinatePairList;
    Vertex():Element<0, N, T>(){};
    Vertex(const DomainShared_ptr & domain):Element<0, N, T>(domain){};

    Vertex(const PhyPts & point):Element<0, N, T>(std::make_shared<PhyTensorBsplineBasis<0,N,T>>(point)){};

    T Measure() const{
        return 0;
    }



    void Accept(Visitor<0, N, T> &) {};
};

#endif //OO_IGA_VERTEX_HPP
