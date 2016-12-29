//
// Created by di miao on 12/29/16.
//

#ifndef OO_IGA_PHYTENSORBSPLINEBASIS_H
#define OO_IGA_PHYTENSORBSPLINEBASIS_H

#include "TensorBsplineBasis.h"
#include "MultiArray.h"

template<unsigned d, typename T=double>
class PhyTensorBsplineBasis : public TensorBsplineBasis<d, T> {

public:
    using EigenVector=Eigen::Matrix<T, Eigen::Dynamic, 1>;

    using GeometryVector = std::vector<EigenVector>;

    PhyTensorBsplineBasis();

    PhyTensorBsplineBasis(BsplineBasis<T> *baseX, GeometryVector geometry);

    PhyTensorBsplineBasis(BsplineBasis<T> *baseX, BsplineBasis<T> *baseY);

    PhyTensorBsplineBasis(BsplineBasis<T> *baseX, BsplineBasis<T> *baseY, BsplineBasis<T> *baseZ);

    PhyTensorBsplineBasis(const std::vector<KnotVector<T>> &KV, GeometryVector geometry);

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


#endif //OO_IGA_PHYTENSORBSPLINEBASIS_H
