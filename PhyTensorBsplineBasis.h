//
// Created by di miao on 12/29/16.
//

#ifndef OO_IGA_PHYTENSORBSPLINEBASIS_H
#define OO_IGA_PHYTENSORBSPLINEBASIS_H

#include "TensorBsplineBasis.h"
#include "MultiArray.h"

template<unsigned d,  typename T=double>
class PhyTensorBsplineBasis : public TensorBsplineBasis<d, T> {

public:
    using EigenVector=Eigen::Matrix<T, Eigen::Dynamic, 1>;

    using GeometryVector = std::vector<EigenVector>;

    PhyTensorBsplineBasis();

    PhyTensorBsplineBasis(BsplineBasis<T> *baseX, GeometryVector geometry);

    PhyTensorBsplineBasis(BsplineBasis<T> *baseX, BsplineBasis<T> *baseY);

    PhyTensorBsplineBasis(BsplineBasis<T> *baseX, BsplineBasis<T> *baseY, BsplineBasis<T> *baseZ);

    PhyTensorBsplineBasis(const std::vector<KnotVector<T>> &KV, GeometryVector geometry);

    EigenVector AffineMap(const EigenVector & u) const;

    virtual ~PhyTensorBsplineBasis() {

    }

private:
    GeometryVector _geometricInfo;
};

template<unsigned d, typename T>
PhyTensorBsplineBasis<d, T>::PhyTensorBsplineBasis():TensorBsplineBasis<d, T>() {

}

template<unsigned d, typename T>
PhyTensorBsplineBasis<d, T>::PhyTensorBsplineBasis(const std::vector<KnotVector<T>> &KV, GeometryVector geometry) : TensorBsplineBasis<d,T>(KV) {
    ASSERT((this->TensorBsplineBasis<d,T>::GetDof())==geometry.size(),"Invalid geometrical information input, check size bro.");
    _geometricInfo=geometry;
}

template<unsigned d, typename T>
typename PhyTensorBsplineBasis<d, T>::EigenVector PhyTensorBsplineBasis<d, T>::AffineMap(const PhyTensorBsplineBasis<d, T>::EigenVector &u) const {
    PhyTensorBsplineBasis<d,T>::EigenVector result(d);
    result.setZero();
    auto p=this->TensorBsplineBasis<d,T>::EvalTensor(u);
    for(auto it=p->begin();it!=p->end();++it){
        result+=_geometricInfo[it->first]*(it->second)[0][0];
    }
    return result;
}


#endif //OO_IGA_PHYTENSORBSPLINEBASIS_H
