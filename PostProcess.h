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
    using ResultPtr = std::pair<std::shared_ptr<Cell<T>>, std::shared_ptr<PhyTensorBsplineBasis<2, 1, T>>>;
public:
    PostProcess(const std::vector<std::shared_ptr<Cell<T>>> &dom,
                const std::vector<std::shared_ptr<PhyTensorBsplineBasis<2, 1, T>>> &res,
                const Functor &target) : _targetFunction(target) {
        ASSERT(dom.size() == res.size(),
               "The number of computational domains and that of result domains are not consistent.");
        for (int i = 0; i < dom.size(); ++i) {
            _results.push_back(std::make_pair(dom[i], res[i]));
        }
    };

    T RelativeL2Error() {
        T errorNorm = 0;
        T norm = 0;
        for (const auto &result:_results) {
            auto deg_x = result.first->GetDegree(0);
            auto deg_y = result.first->GetDegree(1);
            QuadratureRule<T> quadrature;
            quadrature.SetUpQuadrature(deg_x >= deg_y ? (deg_x + 1) : (deg_y + 1));
            CoordinatePairList elements;
            result.first->KnotSpansGetter(elements);
            for (const auto &element:elements) {
                Quadlist quadratures;
                quadrature.MapToQuadrature(element, quadratures);
                for (const auto &i:quadratures) {
                    norm += i.second * pow(_targetFunction(result.first->GetDomain()->AffineMap(i.first))[0], 2);
                    errorNorm += i.second * pow(_targetFunction(result.first->GetDomain()->AffineMap(i.first))[0] -
                                                result.second->AffineMap(i.first)(0), 2);
                }
            }
        }
        return sqrt(errorNorm/norm);
    }

private:
    std::vector<ResultPtr> _results;
    Functor _targetFunction;

};


#endif //OO_IGA_POSTPROCESS_H
