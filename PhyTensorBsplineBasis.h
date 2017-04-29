//
// Created by di miao on 12/29/16.
//

#ifndef OO_IGA_PHYTENSORBSPLINEBASIS_H
#define OO_IGA_PHYTENSORBSPLINEBASIS_H

#include "TensorBsplineBasis.h"
#include "MultiArray.h"
#include <algorithm>



template<unsigned d, unsigned N, typename T=double>
class PhyTensorBsplineBasis : public TensorBsplineBasis<d, T> {

public:
    using Ptr=Eigen::Matrix<T, N, 1>;
    using GeometryVector = std::vector<Ptr>;
    typedef std::vector<unsigned> DiffPattern;

    PhyTensorBsplineBasis();

    PhyTensorBsplineBasis(const BsplineBasis<T> &, const GeometryVector &);

    PhyTensorBsplineBasis(const BsplineBasis<T> &, const BsplineBasis<T> &, const GeometryVector &);

    PhyTensorBsplineBasis(const BsplineBasis<T> &, const BsplineBasis<T> &, const BsplineBasis<T> &,
                          const GeometryVector &);

    Ptr AffineMap(const Ptr &, const DiffPattern &i = DiffPattern(d, 0)) const;

    T Jacobian(const Ptr &) const;

    void DegreeElevate(unsigned, unsigned);

    void UniformRefine(unsigned, unsigned, unsigned m = 1);

    void PrintCtrPtr() const;

    virtual ~PhyTensorBsplineBasis() {

    }

private:
    GeometryVector _geometricInfo;
};

template<unsigned d, unsigned N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis():TensorBsplineBasis<d, T>() {

}

template<unsigned d, unsigned N, typename T>
typename PhyTensorBsplineBasis<d, N, T>::Ptr
PhyTensorBsplineBasis<d, N, T>::AffineMap(const PhyTensorBsplineBasis<d, N, T>::Ptr &u, const DiffPattern &i) const {
    PhyTensorBsplineBasis<d, N, T>::Ptr result;
    result.setZero();
    auto p = this->TensorBsplineBasis<d, T>::EvalTensor(u,i);
    for (auto it = p->begin(); it != p->end(); ++it) {
        result += _geometricInfo[it->first] * it->second;
    }
    return result;
}

template<unsigned d, unsigned N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis(const BsplineBasis<T> &baseX,
                                                      const PhyTensorBsplineBasis<d, N, T>::GeometryVector &geometry)
        :TensorBsplineBasis<d, T>(baseX), _geometricInfo(geometry) {
    ASSERT((this->TensorBsplineBasis<d, T>::GetDof()) == geometry.size(),
           "Invalid geometrical information input, check size bro.");
}

template<unsigned d, unsigned N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY,
                                                      const PhyTensorBsplineBasis::GeometryVector &geometry)
        :TensorBsplineBasis<d, T>(baseX, baseY), _geometricInfo(geometry) {
    ASSERT((this->TensorBsplineBasis<d, T>::GetDof()) == geometry.size(),
           "Invalid geometrical information input, check size bro.");
}

template<unsigned d, unsigned N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY,
                                                      const BsplineBasis<T> &baseZ,
                                                      const PhyTensorBsplineBasis::GeometryVector &geometry)
        :TensorBsplineBasis<d, T>(baseX, baseY, baseZ), _geometricInfo(geometry) {
    ASSERT((this->TensorBsplineBasis<d, T>::GetDof()) == geometry.size(),
           "Invalid geometrical information input, check size bro.");
}

template<unsigned d, unsigned N, typename T>
void PhyTensorBsplineBasis<d, N, T>::DegreeElevate(unsigned orientation, unsigned r) {
    std::vector<unsigned> indexes(d, 0);
    std::vector<unsigned> endPerIndex;
    for (unsigned direction = 0; direction != d; ++direction) {
        endPerIndex.push_back(this->GetDof(direction));
    }
    ASSERT(orientation < d, "Invalid degree elevate orientation");
    GeometryVector temp1;
    std::vector<unsigned> MultiIndex(d);
    std::function<void(std::vector<unsigned> &, const std::vector<unsigned> &, unsigned)> recursive;
    TensorBsplineBasis<d, T> tmp1;
    KnotVector<T> knot_temp_storage;
    bool called = false;
    recursive = [this, &orientation, &called, &knot_temp_storage, &tmp1, r, &temp1, &MultiIndex, &recursive](
            std::vector<unsigned> &indexes,
            const std::vector<unsigned> &endPerIndex,
            unsigned direction) {
        if (direction == indexes.size()) {

            Accessory::ContPtrList<T, N> ElevateList;
            for (unsigned i = 0; i != endPerIndex[orientation]; ++i) {
                MultiIndex[orientation] = i;
                auto index = this->Index(MultiIndex);
                ElevateList.push_back(_geometricInfo[index]);
            }
            KnotVector<T> tmp(this->_basis[orientation].Knots());
            Accessory::degreeElevate<T, N>(r, tmp, ElevateList);
            if (knot_temp_storage.GetSize() == 0) {
                for (unsigned i = 0; i != d; ++i) {
                    if (i == orientation) {
                        tmp1.ChangeKnots(tmp, i);
                    } else {
                        tmp1.ChangeKnots(this->_basis[i].Knots(), i);
                    }
                }
                temp1.resize(tmp1.GetDof());
                knot_temp_storage = tmp;
            }
            for (unsigned i = 0; i != ElevateList.size(); ++i) {
                MultiIndex[orientation] = i;
                auto index = tmp1.Index(MultiIndex);
                temp1[index] = ElevateList[i];
            }
        } else {
            for (indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++) {
                MultiIndex[direction] = indexes[direction];
                if (direction == orientation && called == false) {
                    called = true;
                } else if (direction == orientation && called == true) {
                    break;
                }
                called = false;
                recursive(indexes, endPerIndex, direction + 1);
            }
        }
    };
    recursive(indexes, endPerIndex, 0);
    TensorBsplineBasis<d, T>::_basis[orientation] = knot_temp_storage;
    _geometricInfo = temp1;
}

template<unsigned d, unsigned N, typename T>
void PhyTensorBsplineBasis<d, N, T>::UniformRefine(unsigned orientation, unsigned r, unsigned m) {
    ASSERT(orientation < d, "Invalid degree elevate orientation");
    std::vector<unsigned> indexes(d, 0);
    std::vector<unsigned> endPerIndex;
    for (unsigned direction = 0; direction != d; ++direction) {
        endPerIndex.push_back(this->GetDof(direction));
    }
    GeometryVector temp1;
    std::vector<unsigned> MultiIndex(d);
    std::function<void(std::vector<unsigned> &, const std::vector<unsigned> &, unsigned)> recursive;
    TensorBsplineBasis<d, T> tmp1;
    KnotVector<T> knot_temp_storage(this->_basis[orientation].Knots());
    knot_temp_storage.UniformRefine(r, m);
    KnotVector<T> X = knot_temp_storage.Difference(this->_basis[orientation].Knots());
    knot_temp_storage.resize(0);
    bool called = false;
    recursive = [this, &X, &orientation, &called, &knot_temp_storage, &tmp1, &temp1, &MultiIndex, &recursive](
            std::vector<unsigned> &indexes, const std::vector<unsigned> &endPerIndex, unsigned direction) {
        if (direction == indexes.size()) {
            Accessory::ContPtrList<T, N> RefineList;
            for (unsigned i = 0; i != endPerIndex[orientation]; ++i) {
                MultiIndex[orientation] = i;
                auto index = this->Index(MultiIndex);
                RefineList.push_back(_geometricInfo[index]);
            }
            KnotVector<T> tmp(this->_basis[orientation].Knots());
            Accessory::refineKnotVectorCurve<T, N>(X, tmp, RefineList);
            if (knot_temp_storage.GetSize() == 0) {
                for (unsigned i = 0; i != d; ++i) {
                    if (i == orientation) {
                        tmp1.ChangeKnots(tmp, i);
                    } else {
                        tmp1.ChangeKnots(this->_basis[i].Knots(), i);
                    }
                }
                temp1.resize(tmp1.GetDof());
                knot_temp_storage = tmp;
            }
            for (unsigned i = 0; i != RefineList.size(); ++i) {
                MultiIndex[orientation] = i;
                auto index = tmp1.Index(MultiIndex);
                temp1[index] = RefineList[i];
            }
        } else {
            for (indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++) {
                MultiIndex[direction] = indexes[direction];
                if (direction == orientation && called == false) {
                    called = true;
                } else if (direction == orientation && called == true) {
                    break;
                }
                called = false;
                recursive(indexes, endPerIndex, direction + 1);
            }
        }
    };
    recursive(indexes, endPerIndex, 0);
    TensorBsplineBasis<d, T>::_basis[orientation] = knot_temp_storage;
    _geometricInfo = temp1;
}

template<unsigned d, unsigned N, typename T>
void PhyTensorBsplineBasis<d, N, T>::PrintCtrPtr() const {
    for (int i = 0; i != _geometricInfo.size(); ++i)
        std::cout << _geometricInfo[i].transpose() << std::endl;
}

template<unsigned d, unsigned N, typename T>
T PhyTensorBsplineBasis<d, N, T>::Jacobian(const PhyTensorBsplineBasis::Ptr &u) const {
    ASSERT(d == N, "Dimension should be the same between parametric space and Physical space.");
    using JacobianMatrix=Eigen::Matrix<T, d, d>;
    JacobianMatrix a;
    for (int i = 0; i != d; i++) {
        std::vector<unsigned> differentiation(d, 0);
        differentiation[i] = 1;
        auto aa = AffineMap(u,differentiation);
        a.col(i)=aa;
    }
    return a.determinant();
};
#endif //OO_IGA_PHYTENSORBSPLINEBASIS_H
