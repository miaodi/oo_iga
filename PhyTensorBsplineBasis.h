//
// Created by di miao on 12/29/16.
//

#ifndef OO_IGA_PHYTENSORBSPLINEBASIS_H
#define OO_IGA_PHYTENSORBSPLINEBASIS_H

#include "TensorBsplineBasis.h"


template<int d, int N, typename T=double>
class PhyTensorBsplineBasis : public TensorBsplineBasis<d, T> {

public:
    using Ptr=Eigen::Matrix<T, N, 1>;
    using GeometryVector = std::vector<Ptr>;
    typedef std::vector<int> DiffPattern;
    using vector=Eigen::Matrix<T, Eigen::Dynamic, 1>;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValDerAll BasisFunValDerAll;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValDerAllList BasisFunValDerAllList;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValDerAllList_ptr BasisFunValDerAllList_ptr;

    PhyTensorBsplineBasis();

    PhyTensorBsplineBasis(const BsplineBasis<T> &, const GeometryVector &);

    PhyTensorBsplineBasis(const BsplineBasis<T> &, const BsplineBasis<T> &, const GeometryVector &);

    PhyTensorBsplineBasis(const BsplineBasis<T> &, const BsplineBasis<T> &, const BsplineBasis<T> &,
                          const GeometryVector &);

    Ptr AffineMap(const Ptr &, const DiffPattern &i = DiffPattern(d, 0)) const;

    T Jacobian(const Ptr &) const;

    void DegreeElevate(int, int);

    void UniformRefine(int, int, int m = 1);

    void PrintCtrPtr() const;

    virtual ~PhyTensorBsplineBasis() {

    }

    BasisFunValDerAllList_ptr Eval1DerAllTensor(const vector &u) const;

    BasisFunValDerAllList_ptr Eval2DerAllTensor(const vector &u) const;

private:
    GeometryVector _geometricInfo;
};

template<int d, int N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis():TensorBsplineBasis<d, T>() {

}

template<int d, int N, typename T>
typename PhyTensorBsplineBasis<d, N, T>::Ptr
PhyTensorBsplineBasis<d, N, T>::AffineMap(const PhyTensorBsplineBasis<d, N, T>::Ptr &u, const DiffPattern &i) const {
    PhyTensorBsplineBasis<d, N, T>::Ptr result;
    result.setZero();
    auto p = this->TensorBsplineBasis<d, T>::EvalTensor(u, i);
    for (auto it = p->begin(); it != p->end(); ++it) {
        result += _geometricInfo[it->first] * it->second;
    }
    return result;
}

template<int d, int N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis(const BsplineBasis<T> &baseX,
                                                      const PhyTensorBsplineBasis<d, N, T>::GeometryVector &geometry)
        :TensorBsplineBasis<d, T>(baseX), _geometricInfo(geometry) {
    ASSERT((this->TensorBsplineBasis<d, T>::GetDof()) == geometry.size(),
           "Invalid geometrical information input, check size bro.");
}

template<int d, int N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY,
                                                      const PhyTensorBsplineBasis::GeometryVector &geometry)
        :TensorBsplineBasis<d, T>(baseX, baseY), _geometricInfo(geometry) {
    ASSERT((this->TensorBsplineBasis<d, T>::GetDof()) == geometry.size(),
           "Invalid geometrical information input, check size bro.");
}

template<int d, int N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis(const BsplineBasis<T> &baseX, const BsplineBasis<T> &baseY,
                                                      const BsplineBasis<T> &baseZ,
                                                      const PhyTensorBsplineBasis::GeometryVector &geometry)
        :TensorBsplineBasis<d, T>(baseX, baseY, baseZ), _geometricInfo(geometry) {
    ASSERT((this->TensorBsplineBasis<d, T>::GetDof()) == geometry.size(),
           "Invalid geometrical information input, check size bro.");
}

template<int d, int N, typename T>
void PhyTensorBsplineBasis<d, N, T>::DegreeElevate(int orientation, int r) {
    std::vector<int> indexes(d, 0);
    std::vector<int> endPerIndex;
    for (int direction = 0; direction != d; ++direction) {
        endPerIndex.push_back(this->GetDof(direction));
    }
    ASSERT(orientation < d, "Invalid degree elevate orientation");
    GeometryVector temp1;
    std::vector<int> MultiIndex(d);
    std::function<void(std::vector<int> &, const std::vector<int> &, int)> recursive;
    TensorBsplineBasis<d, T> tmp1;
    KnotVector<T> knot_temp_storage;
    bool called = false;
    recursive = [this, &orientation, &called, &knot_temp_storage, &tmp1, r, &temp1, &MultiIndex, &recursive](
            std::vector<int> &indexes,
            const std::vector<int> &endPerIndex,
            int direction) {
        if (direction == indexes.size()) {

            Accessory::ContPtrList<T, N> ElevateList;
            for (int i = 0; i != endPerIndex[orientation]; ++i) {
                MultiIndex[orientation] = i;
                auto index = this->Index(MultiIndex);
                ElevateList.push_back(_geometricInfo[index]);
            }
            KnotVector<T> tmp(this->_basis[orientation].Knots());
            Accessory::degreeElevate<T, N>(r, tmp, ElevateList);
            if (knot_temp_storage.GetSize() == 0) {
                for (int i = 0; i != d; ++i) {
                    if (i == orientation) {
                        tmp1.ChangeKnots(tmp, i);
                    } else {
                        tmp1.ChangeKnots(this->_basis[i].Knots(), i);
                    }
                }
                temp1.resize(tmp1.GetDof());
                knot_temp_storage = tmp;
            }
            for (int i = 0; i != ElevateList.size(); ++i) {
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

template<int d, int N, typename T>
void PhyTensorBsplineBasis<d, N, T>::UniformRefine(int orientation, int r, int m) {
    ASSERT(orientation < d, "Invalid degree elevate orientation");
    std::vector<int> indexes(d, 0);
    std::vector<int> endPerIndex;
    for (int direction = 0; direction != d; ++direction) {
        endPerIndex.push_back(this->GetDof(direction));
    }
    GeometryVector temp1;
    std::vector<int> MultiIndex(d);
    std::function<void(std::vector<int> &, const std::vector<int> &, int)> recursive;
    TensorBsplineBasis<d, T> tmp1;
    KnotVector<T> knot_temp_storage(this->_basis[orientation].Knots());
    knot_temp_storage.UniformRefine(r, m);
    KnotVector<T> X = knot_temp_storage.Difference(this->_basis[orientation].Knots());
    knot_temp_storage.resize(0);
    bool called = false;
    recursive = [this, &X, &orientation, &called, &knot_temp_storage, &tmp1, &temp1, &MultiIndex, &recursive](
            std::vector<int> &indexes, const std::vector<int> &endPerIndex, int direction) {
        if (direction == indexes.size()) {
            Accessory::ContPtrList<T, N> RefineList;
            for (int i = 0; i != endPerIndex[orientation]; ++i) {
                MultiIndex[orientation] = i;
                auto index = this->Index(MultiIndex);
                RefineList.push_back(_geometricInfo[index]);
            }
            KnotVector<T> tmp(this->_basis[orientation].Knots());
            Accessory::refineKnotVectorCurve<T, N>(X, tmp, RefineList);
            if (knot_temp_storage.GetSize() == 0) {
                for (int i = 0; i != d; ++i) {
                    if (i == orientation) {
                        tmp1.ChangeKnots(tmp, i);
                    } else {
                        tmp1.ChangeKnots(this->_basis[i].Knots(), i);
                    }
                }
                temp1.resize(tmp1.GetDof());
                knot_temp_storage = tmp;
            }
            for (int i = 0; i != RefineList.size(); ++i) {
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

template<int d, int N, typename T>
void PhyTensorBsplineBasis<d, N, T>::PrintCtrPtr() const {
    for (int i = 0; i != _geometricInfo.size(); ++i)
        std::cout << _geometricInfo[i].transpose() << std::endl;
}

template<int d, int N, typename T>
T PhyTensorBsplineBasis<d, N, T>::Jacobian(const PhyTensorBsplineBasis::Ptr &u) const {
    ASSERT(d == N, "Dimension should be the same between parametric space and Physical space.");
    using JacobianMatrix=Eigen::Matrix<T, d, d>;
    JacobianMatrix a;
    for (int i = 0; i != d; i++) {
        std::vector<int> differentiation(d, 0);
        differentiation[i] = 1;
        auto aa = AffineMap(u, differentiation);
        a.col(i) = aa;
    }
    return a.determinant();
}

template<>
typename PhyTensorBsplineBasis<2, 2, double>::BasisFunValDerAllList_ptr
PhyTensorBsplineBasis<2, 2, double>::Eval1DerAllTensor(const vector &u) const {
    auto parametric = TensorBsplineBasis<2, double>::EvalDerAllTensor(u, 1);
    Eigen::Vector2d Pxi, Peta;
    Pxi.setZero();
    Peta.setZero();
    for (const auto &i:*parametric) {
        Pxi += i.second[1] * _geometricInfo[i.first];
        Peta += i.second[2] * _geometricInfo[i.first];
    }
    Eigen::Matrix2d Jacobian;
    Jacobian.row(0) = Pxi.transpose();
    Jacobian.row(1) = Peta.transpose();
    for (auto &i:*parametric) {
        auto solution = Jacobian.partialPivLu().solve(Eigen::Vector2d::Map(i.second.data() + 1, i.second.size() - 1));
        i.second[1] = solution(0);
        i.second[2] = solution(1);
    }
    return parametric;

}

template<>
typename PhyTensorBsplineBasis<2, 2, double>::BasisFunValDerAllList_ptr
PhyTensorBsplineBasis<2, 2, double>::Eval2DerAllTensor(const PhyTensorBsplineBasis::vector &u) const {
    auto parametric = TensorBsplineBasis<2, double>::EvalDerAllTensor(u, 2);
    Eigen::Vector2d Pxi, Peta, PxiPxi, PxiPeta, PetaPeta;
    Pxi.setZero();
    Peta.setZero();
    PxiPxi.setZero();
    PxiPeta.setZero();
    PetaPeta.setZero();
    for (const auto &i:*parametric) {
        Pxi += i.second[1] * _geometricInfo[i.first];
        Peta += i.second[2] * _geometricInfo[i.first];
        PxiPxi += i.second[3] * _geometricInfo[i.first];
        PxiPeta += i.second[4] * _geometricInfo[i.first];
        PetaPeta += i.second[5] * _geometricInfo[i.first];
    }
    Eigen::Matrix<double, 5, 5> Hessian;
    Hessian << Pxi(0), Pxi(1), 0, 0, 0, Peta(0), Peta(1), 0, 0, 0, PxiPxi(0), PxiPxi(1), Pxi(0) * Pxi(0), 2 * Pxi(0) *
                                                                                                          Pxi(1),
            Pxi(1) * Pxi(1), PxiPeta(0), PxiPeta(1), Pxi(0) * Peta(0), Pxi(0) * Peta(1) + Peta(0) * Pxi(1), Pxi(1) *
                                                                                                             Peta(1), PetaPeta(
            0), PetaPeta(1), Peta(0) * Peta(0), 2 * Peta(0) * Peta(1), Peta(1) * Peta(1);
    for (auto &i:*parametric) {
        auto solution = Hessian.partialPivLu().solve(Eigen::Matrix<double,5,1>::Map(i.second.data() + 1, i.second.size() - 1));
        i.second[1] = solution(0);
        i.second[2] = solution(1);
        i.second[3] = solution(2);
        i.second[4] = solution(3);
        i.second[5] = solution(4);
    }
    return parametric;
};


#endif //OO_IGA_PHYTENSORBSPLINEBASIS_H
