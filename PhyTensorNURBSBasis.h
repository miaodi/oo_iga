//
// Created by miaodi on 01/10/2017.
//

#ifndef OO_IGA_PHYTENSORNURBSBASIS_H
#define OO_IGA_PHYTENSORNURBSBASIS_H

#include "PhyTensorBsplineBasis.h"

template<int d, int N, typename T=double>
class PhyTensorNURBSBasis : public PhyTensorBsplineBasis<d, N, T> {
public:
    using Pts = PhyTensorBsplineBasis<d, N, T>::Pts;
    using PhyPts = PhyTensorBsplineBasis<d, N, T>::PhyPts;
    using GeometryVector = PhyTensorBsplineBasis<d, N, T>::GeometryVector;
    using WeightVector = PhyTensorBsplineBasis<d, 1, T>::GeometryVector;
    using DiffPattern = PhyTensorBsplineBasis<d, N, T>::DiffPattern;
    typedef typename PhyTensorBsplineBasis<d, N, T>::BasisFunValDerAll BasisFunValDerAll;
    typedef typename PhyTensorBsplineBasis<d, N, T>::BasisFunValDerAllList BasisFunValDerAllList;
    typedef typename PhyTensorBsplineBasis<d, N, T>::BasisFunValDerAllList_ptr BasisFunValDerAllList_ptr;
    typedef typename TensorBsplineBasis<d, T>::BasisFunVal BasisFunVal;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValPac BasisFunValPac;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValPac_ptr BasisFunValPac_ptr;
    using vector=Eigen::Matrix<T, Eigen::Dynamic, 1>;

    PhyTensorNURBSBasis(const std::vector<KnotVector<T>> &, const GeometryVector &, const WeightVector &);

    BasisFunValPac_ptr EvalTensor(const vector &u, const DiffPattern &i = DiffPattern(d, 0)) const;

protected:
    PhyTensorBsplineBasis<d, 1, T> _weightFunction;
};

template<int d, int N, typename T=double>
PhyTensorNURBSBasis<d, N, T>::PhyTensorNURBSBasis(const std::vector<KnotVector<T>> &base, const PhyTensorNURBSBasis::GeometryVector &geometry,
                                                  const PhyTensorNURBSBasis::WeightVector &weight): PhyTensorBsplineBasis<d, N, T>(base, geometry),
                                                                                                    _weightFunction(base, geometry) {
}

template<int d, int N, typename T=double>
PhyTensorNURBSBasis<d, N, T>::BasisFunValPac_ptr
PhyTensorNURBSBasis<d, N, T>::EvalTensor(const PhyTensorNURBSBasis<d, N, T>::vector &u, const PhyTensorNURBSBasis<d, N, T>::DiffPattern &i) const {
    ASSERT((u.size() == d) && (i.size() == d), "Invalid input vector size.");
    std::vector<int> indexes(d, 0);
    std::vector<int> endPerIndex;
    std::array<BasisFunValPac_ptr, d> OneDResult;
    for (int direction = 0; direction != d; ++direction) {
        OneDResult[direction] = _basis[direction].Eval(u(direction), i[direction]);
        endPerIndex.push_back(OneDResult[direction]->size());
    }
    std::vector<int> MultiIndex(d);
    std::vector<T> Value(d);
    BasisFunValPac_ptr Result(new BasisFunValPac);

    std::function<void(std::vector<int> &, const std::vector<int> &, int)> recursive;

    recursive = [this, &OneDResult, &MultiIndex, &Value, &Result, &recursive](std::vector<int> &indexes,
                                                                              const std::vector<int> &endPerIndex,
                                                                              int direction) {
        if (direction == indexes.size()) {
            T result = 1;
            for (int ii = 0; ii < d; ii++)
                result *= Value[ii];
            Result->push_back(BasisFunVal(Index(MultiIndex), result));
        }
        else {
            for (indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++) {
                Value[direction] = (*OneDResult[direction])[indexes[direction]].second;
                MultiIndex[direction] = (*OneDResult[direction])[indexes[direction]].first;
                recursive(indexes, endPerIndex, direction + 1);
            }
        }
    };
    recursive(indexes, endPerIndex, 0);
}


#endif //OO_IGA_PHYTENSORNURBSBASIS_H
