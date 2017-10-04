//
// Created by miaodi on 01/10/2017.
//

#ifndef OO_IGA_PHYTENSORNURBSBASIS_H
#define OO_IGA_PHYTENSORNURBSBASIS_H

#include "PhyTensorBsplineBasis.h"
#include <boost/math/special_functions/binomial.hpp>

template<int d, int N, typename T=double>
class PhyTensorNURBSBasis : public PhyTensorBsplineBasis<d, N, T> {
public:
    using Pts = typename PhyTensorBsplineBasis<d, N, T>::Pts;
    using PhyPts = typename PhyTensorBsplineBasis<d, N, T>::PhyPts;
    using GeometryVector = typename PhyTensorBsplineBasis<d, N, T>::GeometryVector;
    using WeightVector = typename PhyTensorBsplineBasis<d, 1, T>::GeometryVector;
    using DiffPattern = typename PhyTensorBsplineBasis<d, N, T>::DiffPattern;
    typedef typename PhyTensorBsplineBasis<d, N, T>::BasisFunValDerAll BasisFunValDerAll;
    typedef typename PhyTensorBsplineBasis<d, N, T>::BasisFunValDerAllList BasisFunValDerAllList;
    typedef typename PhyTensorBsplineBasis<d, N, T>::BasisFunValDerAllList_ptr BasisFunValDerAllList_ptr;
    typedef typename TensorBsplineBasis<d, T>::BasisFunVal BasisFunVal;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValPac BasisFunValPac;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValPac_ptr BasisFunValPac_ptr;
    using vector=Eigen::Matrix<T, Eigen::Dynamic, 1>;

    PhyTensorNURBSBasis(const std::vector<KnotVector<T>> &, const GeometryVector &, const WeightVector &,
                        const bool swtch = false);

    BasisFunValPac_ptr EvalTensor(const vector &u, const DiffPattern &i = DiffPattern(d, 0)) const;

    BasisFunValDerAllList_ptr EvalDerAllTensor(const vector &u, const int i = 0) const;

protected:
    PhyTensorBsplineBasis<d, 1, T> _weightFunction;

    bool _nurbsSwtch;
};
template<int d, int N, typename T>
PhyTensorNURBSBasis<d, N, T>::PhyTensorNURBSBasis(const std::vector<KnotVector<T>> &base,
                                                  const PhyTensorNURBSBasis::GeometryVector &geometry,
                                                  const PhyTensorNURBSBasis::WeightVector &weight, const bool swtch)
        :PhyTensorBsplineBasis<d, N, T>(base, geometry), _weightFunction(base, weight), _nurbsSwtch{swtch} {
}


//! not finished yet.
template<int d, int N, typename T>
typename PhyTensorNURBSBasis<d, N, T>::BasisFunValPac_ptr
PhyTensorNURBSBasis<d, N, T>::EvalTensor(const PhyTensorNURBSBasis<d, N, T>::vector &u,
                                         const PhyTensorNURBSBasis<d, N, T>::DiffPattern &i) const {

}
template<int d, int N, typename T>
typename PhyTensorNURBSBasis<d, N, T>::BasisFunValDerAllList_ptr
PhyTensorNURBSBasis<d, N, T>::EvalDerAllTensor(const PhyTensorNURBSBasis<d, N, T>::vector &u, const int i) const {
    using namespace boost::math;
    if (_nurbsSwtch == false) {
        return TensorBsplineBasis<d, T>::EvalDerAllTensor(u, i);
    }
    auto bspline_result = TensorBsplineBasis<d, T>::EvalDerAllTensor(u, i);
    auto weights_pts = _weightFunction.CtrPtsGetter();
    for(auto &it:*bspline_result){
        for(auto &j:it.second){
            j*=weights_pts[it.first](0);
        }
    }

    switch (d) {
        case d == 1: {
            std::vector<T> weight_ders(i+1);
            for(int j=0;j<i+1;j++){
                weight_ders[j]=_weightFunction.AffineMap(u,std::vector<int>{j})(0);
            }
            for(auto &it:*bspline_result){
                auto temp=it.second;
                for(int k=0;k<=i;k++){
                    auto v = temp[k];
                    for(int j=1;j<=k;j++){
                        v-=binomial_coefficient<T>(k, j)*weight_ders[j]*it.second[k-j];
                    }
                    it.second[k]=v/weight_ders[0];
                }
            }
            break;
        }
        /*
        case d == 2:{
            Accessory::DifferentialPatternList differentialPatternList;
            for (int order = 0; order <= i; ++order) {
                auto temp = Accessory::PartialDerPattern<d>(order);
                differentialPatternList.insert(differentialPatternList.end(), temp->begin(), temp->end());
            }
            int derivativeAmount = differentialPatternList.size();
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> weight_ders(i+1,i+1);
            for(const auto &j:differentialPatternList){
                weight_ders(j[0],j[1])=_weightFunction.AffineMap(u,j)(0);
            }
            for(auto &it:*bspline_result){
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> temp(i+1,i+1);
                for(int j=0;j<derivativeAmount;j++){
                    temp(differentialPatternList[j][0],differentialPatternList[j][1])=it.second[j];
                }
                for(int k=0;k<=i;k++){
                    auto v = temp[k];
                    for(int j=1;j<=k;j++){
                        v-=binomial_coefficient<T>(k, j)*weight_ders[j]*it.second[k-j];
                    }
                    it.second[k]=v/weight_ders[0];
                }
            }
            break;
        }
        */
    }
    return bspline_result;
}


#endif //OO_IGA_PHYTENSORNURBSBASIS_H
