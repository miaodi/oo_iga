//
// Created by di miao on 12/29/16.
//

#ifndef OO_IGA_PHYTENSORBSPLINEBASIS_H
#define OO_IGA_PHYTENSORBSPLINEBASIS_H

#include "TensorBsplineBasis.h"


template<int d, int N, typename T=double>
class PhyTensorBsplineBasis : public TensorBsplineBasis<d, T> {

public:
    using Pts=Eigen::Matrix<T, d, 1>;
    using PhyPts=Eigen::Matrix<T, N, 1>;
    using GeometryVector = std::vector<PhyPts>;
    typedef std::vector<int> DiffPattern;
    using vector=Eigen::Matrix<T, Eigen::Dynamic, 1>;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValDerAll BasisFunValDerAll;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValDerAllList BasisFunValDerAllList;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValDerAllList_ptr BasisFunValDerAllList_ptr;
    using HyperPlane = PhyTensorBsplineBasis<d - 1, N, T>;
    using HyperPlaneSharedPts = std::shared_ptr<PhyTensorBsplineBasis<d - 1, N, T>>;

    PhyTensorBsplineBasis();

    PhyTensorBsplineBasis(const BsplineBasis<T> &, const GeometryVector &);

    PhyTensorBsplineBasis(const BsplineBasis<T> &, const BsplineBasis<T> &, const GeometryVector &);

    PhyTensorBsplineBasis(const BsplineBasis<T> &, const BsplineBasis<T> &, const BsplineBasis<T> &,
                          const GeometryVector &);

    PhyTensorBsplineBasis(const std::vector<KnotVector<T>> &, const GeometryVector &);

    PhyPts AffineMap(const Pts &, const DiffPattern &i = DiffPattern(d, 0)) const;

    T Jacobian(const Pts &) const;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> JacobianMatrix(const Pts &) const;

    Pts Middle() const {
        Pts u;
        for (int i = 0; i != d; i++)
            u(i) = (this->_basis[i].DomainStart() + this->_basis[i].DomainEnd()) * .5;
        return u;
    }

    PhyPts PhyMiddle() const {
        return AffineMap(Middle());
    }

    bool InversePts(const PhyPts &, Pts &, int = 1000, T = 1e-10) const;

    void DegreeElevate(int, int);

    void UniformRefine(int, int, int m = 1);

    void KnotInsertion(int , T, int =1);

    void DegreeElevate(int p) {
        for (int i = 0; i != d; ++i)
            DegreeElevate(i, p);
    }

    void UniformRefine(int r) {
        for (int i = 0; i != d; ++i)
            UniformRefine(i, r);
    }


    void PrintCtrPts() const;

    HyperPlaneSharedPts MakeHyperPlane(const int &orientation, const int &layer) const;

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
typename PhyTensorBsplineBasis<d, N, T>::PhyPts
PhyTensorBsplineBasis<d, N, T>::AffineMap(const PhyTensorBsplineBasis<d, N, T>::Pts &u, const DiffPattern &i) const {
    PhyTensorBsplineBasis<d, N, T>::PhyPts result;
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
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis(const std::vector<KnotVector<T>> &base,
                                                      const PhyTensorBsplineBasis::GeometryVector &geometry)
        :TensorBsplineBasis<d, T>(base), _geometricInfo(geometry) {

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

            Accessory::ContPtsList<T, N> ElevateList;
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
            Accessory::ContPtsList<T, N> RefineList;
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
void PhyTensorBsplineBasis<d, N, T>::KnotInsertion(int orientation, T knot, int m) {
    std::vector<int> indexes(d, 0);
    std::vector<int> endPerIndex;
    for (int direction = 0; direction != d; ++direction) {
        endPerIndex.push_back(this->GetDof(direction));
    }
    ASSERT(orientation < d, "Invalid insertion orientation");
    GeometryVector temp1;
    std::vector<int> MultiIndex(d);
    std::function<void(std::vector<int> &, const std::vector<int> &, int)> recursive;
    TensorBsplineBasis<d, T> tmp1;
    KnotVector<T> knot_temp_storage;
    bool called = false;
    recursive = [this, &orientation, &called, &knot_temp_storage, &tmp1, knot, m, &temp1, &MultiIndex, &recursive](
            std::vector<int> &indexes,
            const std::vector<int> &endPerIndex,
            int direction) {
        if (direction == indexes.size()) {

            Accessory::ContPtsList<T, N> ElevateList;
            for (int i = 0; i != endPerIndex[orientation]; ++i) {
                MultiIndex[orientation] = i;
                auto index = this->Index(MultiIndex);
                ElevateList.push_back(_geometricInfo[index]);
            }
            KnotVector<T> tmp(this->_basis[orientation].Knots());
            Accessory::knotInsertion<T, N>(knot, m, tmp, ElevateList);
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
void PhyTensorBsplineBasis<d, N, T>::PrintCtrPts() const {
    for (int i = 0; i != _geometricInfo.size(); ++i)
        std::cout << _geometricInfo[i].transpose() << std::endl;
}

template<int d, int N, typename T>
T PhyTensorBsplineBasis<d, N, T>::Jacobian(const PhyTensorBsplineBasis::Pts &u) const {
    return JacobianMatrix(u).determinant();
}

template<int d, int N, typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
PhyTensorBsplineBasis<d, N, T>::JacobianMatrix(const PhyTensorBsplineBasis::Pts &u) const {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> result(N, d);
    for (int i = 0; i != d; i++) {
        std::vector<int> differentiation(d, 0);
        differentiation[i] = 1;
        auto aa = AffineMap(u, differentiation);
        result.col(i) = aa;
    }
    return result;
};

template<int d, int N, typename T>
bool PhyTensorBsplineBasis<d, N, T>::InversePts(const PhyTensorBsplineBasis::PhyPts &phyu, PhyTensorBsplineBasis::Pts &result, int maxLoop, T error) const {
    result=Middle();
    Pts suppBegin, suppEnd;
    for (int i = 0; i != d; ++i) {
        suppBegin(i) = this->DomainStart(i);
        suppEnd(i) = this->DomainEnd(i);
    }
    int iter = 0;
    do {
        result = result.cwiseMax(suppBegin).cwiseMin(suppEnd);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> residual = phyu;
        residual -= AffineMap(result);
        if (residual.norm() <= error)
            return true;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> jacobianMatrix(JacobianMatrix(result));

        if (jacobianMatrix.cols() == jacobianMatrix.rows()) {
            residual = jacobianMatrix.partialPivLu().solve(residual);
        } else {
            residual = jacobianMatrix.colPivHouseholderQr().solve(
                    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(N, N)) * residual;
        }
        result += residual;
    } while (++iter <= maxLoop);
    return false;
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
        Eigen::Map<Eigen::VectorXd> temp(i.second.data() + 1, i.second.size() - 1);
        Eigen::VectorXd solution = Jacobian.partialPivLu().solve(temp);
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
        Eigen::Map<Eigen::VectorXd> temp(i.second.data() + 1, i.second.size() - 1);
        Eigen::VectorXd solution = Hessian.partialPivLu().solve(temp);
        i.second[1] = solution(0);
        i.second[2] = solution(1);
        i.second[3] = solution(2);
        i.second[4] = solution(3);
        i.second[5] = solution(4);
    }
    return parametric;
}

template<int d, int N, typename T>
typename PhyTensorBsplineBasis<d, N, T>::HyperPlaneSharedPts
PhyTensorBsplineBasis<d, N, T>::MakeHyperPlane(const int &orientation, const int &layer) const {
    ASSERT(orientation < d, "Invalid input vector size.");
    std::vector<KnotVector<T>> hpknotvector;
    for (int i = 0; i != d; ++i) {
        if (i != orientation) hpknotvector.push_back(this->KnotVectorGetter(i));
    }
    auto indexList = this->AllActivatedDofsOnBoundary(orientation, layer);
    GeometryVector tempGeometry;
    for (const auto &i:*indexList) {
        tempGeometry.push_back(_geometricInfo[i]);
    }
    return std::make_shared<HyperPlane>(hpknotvector, tempGeometry);
}




#endif //OO_IGA_PHYTENSORBSPLINEBASIS_H
