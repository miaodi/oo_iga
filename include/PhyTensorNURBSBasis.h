//
// Created by miaodi on 01/10/2017.
//

#ifndef OO_IGA_PHYTENSORNURBSBASIS_H
#define OO_IGA_PHYTENSORNURBSBASIS_H

#include "PhyTensorBsplineBasis.h"
#include <boost/math/special_functions/binomial.hpp>

template <int d, int N, typename T = double>
class PhyTensorNURBSBasis : public PhyTensorBsplineBasis<d, N, T>
{
  public:
    using Pts = typename PhyTensorBsplineBasis<d, N, T>::Pts;
    using PhyPts = typename PhyTensorBsplineBasis<d, N, T>::PhyPts;
    using GeometryVector = typename PhyTensorBsplineBasis<d, N, T>::GeometryVector;
    using WtPts = typename PhyTensorBsplineBasis<d, 1, T>::PhyPts;
    using WeightVector = typename PhyTensorBsplineBasis<d, 1, T>::GeometryVector;
    using DiffPattern = typename PhyTensorBsplineBasis<d, N, T>::DiffPattern;
    typedef typename PhyTensorBsplineBasis<d, N, T>::BasisFunValDerAll BasisFunValDerAll;
    typedef typename PhyTensorBsplineBasis<d, N, T>::BasisFunValDerAllList BasisFunValDerAllList;
    typedef typename PhyTensorBsplineBasis<d, N, T>::BasisFunValDerAllList_ptr BasisFunValDerAllList_ptr;
    typedef typename TensorBsplineBasis<d, T>::BasisFunVal BasisFunVal;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValPac BasisFunValPac;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValPac_ptr BasisFunValPac_ptr;
    using vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    using PhyTensorBsplineBasis<d, N, T>::DegreeElevate;
    using PhyTensorBsplineBasis<d, N, T>::UniformRefine;
    using HyperPlane = typename PhyTensorBsplineBasis<d, N, T>::HyperPlane;
    using HyperPlaneSharedPts = typename PhyTensorBsplineBasis<d, N, T>::HyperPlaneSharedPts;

    PhyTensorNURBSBasis(const std::vector<KnotVector<T>> &,
                        const GeometryVector &,
                        const WeightVector &);

    BasisFunValPac_ptr
    EvalTensor(const vector &u,
               const DiffPattern &i = DiffPattern(d, 0)) const;

    BasisFunValDerAllList_ptr
    EvalDerAllTensor(const vector &u,
                     const int i = 0) const;

    void
    DegreeElevate(int,
                  int);

    void
    KnotInsertion(int,
                  T,
                  int = 1);

    void
    UniformRefine(int,
                  int,
                  int m = 1);

    WtPts
    WtPtsGetter(const int &i) const
    {
        return _weightFunction.CtrPtsGetter(i);
    }

    const WeightVector& WeightVectorGetter() const
    {
        return _weightFunction.CtrPtsVecGetter();
    }
    void
    PrintWtCtrPts() const
    {
        _weightFunction.PrintCtrPts();
    }

    virtual HyperPlaneSharedPts MakeHyperPlane(const int &orientation,
                                               const int &layer) const;

  protected:
    PhyTensorBsplineBasis<d, 1, T> _weightFunction;

};

template <int d, int N, typename T>
PhyTensorNURBSBasis<d, N, T>::PhyTensorNURBSBasis(const std::vector<KnotVector<T>> &base,
                                                  const PhyTensorNURBSBasis::GeometryVector &geometry,
                                                  const PhyTensorNURBSBasis::WeightVector &weight)
    : PhyTensorBsplineBasis<d, N, T>(base, geometry), _weightFunction(base, weight)
{
}

//! not finished yet.
template <int d, int N, typename T>
typename PhyTensorNURBSBasis<d, N, T>::BasisFunValPac_ptr
PhyTensorNURBSBasis<d, N, T>::EvalTensor(const PhyTensorNURBSBasis<d, N, T>::vector &u,
                                         const PhyTensorNURBSBasis<d, N, T>::DiffPattern &dff_pattern) const
{
    int diff = 0;
    for (const auto &i : dff_pattern)
    {
        diff += i;
    }
    auto all_ders = EvalDerAllTensor(u, diff);
    BasisFunValPac_ptr res(new BasisFunValPac);
    Accessory::DifferentialPatternList differentialPatternList;
    for (int order = 0; order <= diff; ++order)
    {
        auto temp = Accessory::PartialDerPattern<d>(order);
        differentialPatternList.insert(differentialPatternList.end(), temp->begin(), temp->end());
    }
    int i;
    for (i = 0; i < differentialPatternList.size(); i++)
    {
        if (differentialPatternList[i] == dff_pattern)
            break;
    }
    for (const auto &j : *all_ders)
    {
        res->push_back(std::make_pair(j.first, j.second[i]));
    }
    return res;
}

template <int d, int N, typename T>
typename PhyTensorNURBSBasis<d, N, T>::BasisFunValDerAllList_ptr
PhyTensorNURBSBasis<d, N, T>::EvalDerAllTensor(const PhyTensorNURBSBasis<d, N, T>::vector &u,
                                               const int i) const
{
    using namespace boost::math;
    auto bspline_result = TensorBsplineBasis<d, T>::EvalDerAllTensor(u, i);
    for (auto &it : *bspline_result)
    {
        for (auto &j : it.second)
        {
            j *= _weightFunction.CtrPtsGetter(it.first)(0);
        }
    }
    switch (d)
    {
    case 1:
    {
        std::vector<T> weight_ders(i + 1);
        for (int j = 0; j < i + 1; j++)
        {
            weight_ders[j] = _weightFunction.AffineMap(u, std::vector<int>{j})(0);
        }
        for (auto &it : *bspline_result)
        {
            auto temp = it.second;
            for (int k = 0; k <= i; k++)
            {
                auto v = temp[k];
                for (int j = 1; j <= k; j++)
                {
                    v -= binomial_coefficient<T>(k, j) * weight_ders[j] * it.second[k - j];
                }
                it.second[k] = v / weight_ders[0];
            }
        }
        break;
    }

    case 2:
    {
        Accessory::DifferentialPatternList differentialPatternList;
        for (int order = 0; order <= i; ++order)
        {
            auto temp = Accessory::PartialDerPattern<d>(order);
            differentialPatternList.insert(differentialPatternList.end(), temp->begin(), temp->end());
        }
        int derivativeAmount = differentialPatternList.size();
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> weight_ders(i + 1, i + 1);
        for (const auto &j : differentialPatternList)
        {
            weight_ders(j[0], j[1]) = _weightFunction.AffineMap(u, j)(0);
        }
        for (auto &it : *bspline_result)
        {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> temp(i + 1, i + 1);
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> SKL(i + 1, i + 1);
            for (int j = 0; j < derivativeAmount; j++)
            {
                temp(differentialPatternList[j][0], differentialPatternList[j][1]) = it.second[j];
            }
            for (int k = 0; k <= i; k++)
            {
                for (int l = 0; l <= i - k; l++)
                {
                    auto v = temp(k, l);
                    for (int n = 1; n <= l; n++)
                    {
                        v -= binomial_coefficient<T>(l, n) * weight_ders(0, n) * SKL(k, l - n);
                    }
                    for (int m = 1; m <= k; m++)
                    {
                        v -= binomial_coefficient<T>(k, m) * weight_ders(m, 0) * SKL(k - m, l);
                        T v2 = 0.0;
                        for (int n = 1; n <= l; n++)
                        {
                            v2 += binomial_coefficient<T>(l, n) * weight_ders(m, n) * SKL(k - m, l - n);
                        }
                        v -= binomial_coefficient<T>(k, m) * v2;
                    }
                    SKL(k, l) = v / weight_ders(0, 0);
                }
            }
            for (int j = 0; j < derivativeAmount; j++)
            {
                it.second[j] = SKL(differentialPatternList[j][0], differentialPatternList[j][1]);
            }
        }
        break;
    }
    }
    return bspline_result;
}


template <int d, int N, typename T>
void PhyTensorNURBSBasis<d, N, T>::DegreeElevate(int orientation,
                                                 int r)
{
    auto dof = this->GetDof();
    for (int i = 0; i < dof; i++)
    {
        this->_geometricInfo[i] *= _weightFunction.CtrPtsGetter(i)(0);
    }
    PhyTensorBsplineBasis<d, N, T>::DegreeElevate(orientation, r);
    _weightFunction.DegreeElevate(orientation, r);
    dof = this->GetDof();
    for (int i = 0; i < dof; i++)
    {
        this->_geometricInfo[i] /= _weightFunction.CtrPtsGetter(i)(0);
    }
}

template <int d, int N, typename T>
void PhyTensorNURBSBasis<d, N, T>::KnotInsertion(int orientation,
                                                 T knot,
                                                 int m)
{
    auto dof = this->GetDof();
    for (int i = 0; i < dof; i++)
    {
        this->_geometricInfo[i] *= _weightFunction.CtrPtsGetter(i)(0);
    }
    PhyTensorBsplineBasis<d, N, T>::KnotInsertion(orientation, knot, m);
    dof = this->GetDof();
    _weightFunction.KnotInsertion(orientation, knot, m);
    for (int i = 0; i < dof; i++)
    {
        this->_geometricInfo[i] /= _weightFunction.CtrPtsGetter(i)(0);
    }
}

template <int d, int N, typename T>
void PhyTensorNURBSBasis<d, N, T>::UniformRefine(int orientation,
                                                 int r,
                                                 int m)
{
    auto dof = this->GetDof();
    for (int i = 0; i < dof; i++)
    {
        this->_geometricInfo[i] *= _weightFunction.CtrPtsGetter(i)(0);
    }
    PhyTensorBsplineBasis<d, N, T>::UniformRefine(orientation, r, m);
    dof = this->GetDof();
    _weightFunction.UniformRefine(orientation, r, m);
    for (int i = 0; i < dof; i++)
    {
        this->_geometricInfo[i] /= _weightFunction.CtrPtsGetter(i)(0);
    }
}

template <int d, int N, typename T>
typename PhyTensorNURBSBasis<d, N, T>::HyperPlaneSharedPts
PhyTensorNURBSBasis<d, N, T>::MakeHyperPlane(const int &orientation,
                                             const int &layer) const
{
    ASSERT(orientation < d, "Invalid input vector size.");
    std::vector<KnotVector<T>> hpknotvector;
    for (int i = 0; i != d; ++i)
    {
        if (i != orientation)
            hpknotvector.push_back(this->KnotVectorGetter(i));
    }
    auto indexList = this->HyperPlaneIndices(orientation, layer);
    GeometryVector tempGeometry;
    for (const auto &i : *indexList)
    {
        tempGeometry.push_back(this->_geometricInfo[i]);
    }
    WeightVector tempWeight;
    for (const auto &i : *indexList)
    {
        tempWeight.push_back(this->WtPtsGetter(i));
    }
    return std::make_shared<PhyTensorNURBSBasis<d - 1, N, T>>(hpknotvector, tempGeometry, tempWeight);
}

template <int N, typename T>
class PhyTensorNURBSBasis<0, N, T> : public PhyTensorBsplineBasis<0, N, T>
{
  public:
    using HyperPlane = PhyTensorBsplineBasis<-1, N, T>;
    using HyperPlaneSharedPts = std::shared_ptr<PhyTensorBsplineBasis<-1, N, T>>;
    using GeometryVector = Accessory::ContPtsList<T, N>;
    using WeightVector = typename PhyTensorBsplineBasis<0, 1, T>::GeometryVector;
    PhyTensorNURBSBasis() {}

    PhyTensorNURBSBasis(const std::vector<KnotVector<T>> &,
                        const GeometryVector &,
                        const WeightVector &) {}

    HyperPlane MakeHyperPlane(const int &orientation,
                              const int &layer) const {}

    ~PhyTensorNURBSBasis(){};
};

#endif //OO_IGA_PHYTENSORNURBSBASIS_H
