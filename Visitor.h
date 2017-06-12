//
// Created by miaodi on 08/06/2017.
//

#ifndef OO_IGA_VISITOR_H
#define OO_IGA_VISITOR_H

#include "Topology.h"
#include "DofMapper.h"

template<typename T>
class Element;

template<typename T>
class Edge;

template<typename T>
class Cell;
template<typename T>
class DofMapper;

template<typename T>
class Visitor {
public:
    using Coordinate = typename Element<T>::Coordinate;
    using CoordinatePairList = typename Element<T>::CoordinatePairList;
    using Quadlist = typename QuadratureRule<T>::QuadList;
    using DomainShared_ptr = typename Element<T>::DomainShared_ptr;
    using IndexedValue = Eigen::Triplet<T>;
    using IndexedValueList = std::vector<IndexedValue>;
    using LoadFunctor = std::function<std::vector<T>(const Coordinate &)>;


    Visitor() {};

    virtual void visit(Edge<T> *g) = 0;

    virtual void visit(Cell<T> *g) = 0;


};
template<typename T>
class PoissonMapperInitiator:public Visitor<T>{
public:
    PoissonMapperInitiator(DofMapper<T> & dofMap):_dofMap(dofMap){

    }
    void visit(Edge<T> *g){
        if(!g->GetMatchInfo()){
            auto tmp = g->AllActivatedDofsOfLayers(0);
            for(const auto &i:*tmp){
                _dofMap.FreezedDofInserter(g->GetDomain(),i);
            }
        }
    }
    void visit(Cell<T> *g){
        _dofMap.DomainLabel(g->GetDomain());
        _dofMap.PatchDofSetter(g->GetDomain(),g->GetDof());
    }

private:
    DofMapper<T> & _dofMap;
};

#endif //OO_IGA_VISITOR_H
