//
// Created by di miao on 9/12/17.
//

#ifndef OO_IGA_POSTPROCESS_H
#define OO_IGA_POSTPROCESS_H

#include "Topology.h"
#include "QuadratureRule.h"
#include "PhyTensorBsplineBasis.h"

template<typename T>
class PostProcess {
public:
    using Coordinate = typename Element<T>::Coordinate;
    using CoordinatePairList = typename Element<T>::CoordinatePairList;
    using Quadlist = typename QuadratureRule<T>::QuadList;
    using Functor = std::function<std::vector<T>(const Coordinate &)>;
    using ResultPtr = std::shared_ptr<PhyTensorBsplineBasis<2, 1, T>>;
public:
    PostProcess(const std::vector<std::shared_ptr<Cell<T>>> &dom, const std::vector<ResultPtr> &res,
                const Functor &target) : _domains(dom), _results(res), _targetFunction(target) {};

    T RelativeL2Error() {
        T error = 0;
        for (const auto &domain:_domains) {
            auto deg_x = domain->GetDegree(0);
            auto deg_y = domain->GetDegree(1);
            QuadratureRule<T> quadrature;
            quadrature.SetUpQuadrature(deg_x >= deg_y ? (deg_x + 1) : (deg_y + 1));
            CoordinatePairList elements;
            domain->KnotSpansGetter(elements);
            for(const auto &element:elements){
                Quadlist quadratures;
                quadrature.MapToQuadrature(element, quadratures);
                
            }
        }
    }

private:
    std::vector<std::shared_ptr<Cell<T>>> _domains;
    std::vector<ResultPtr> _results;
    Functor _targetFunction;

};


#endif //OO_IGA_POSTPROCESS_H
