//
// Created by di miao on 12/29/16.
//
#pragma once

#include "TensorBsplineBasis.h"
#include "Utility.hpp"
#include <Eigen/StdVector>
#include <memory>
#include <unordered_map>

template <int d, int N, typename T>
struct ComputeJacobian;

template <int d, int N, typename T = double>
class PhyTensorBsplineBasis : public TensorBsplineBasis<d, T>
{
public:
    using Pts = Eigen::Matrix<T, d, 1>;
    using PhyPts = Eigen::Matrix<T, N, 1>;

    // aligned_allocator is required by Eigen for fixed-size Eigen types
    using GeometryVector = Accessory::ContPtsList<T, N>;
    typedef std::vector<int> DiffPattern;
    using vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValDerAll BasisFunValDerAll;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValDerAllList BasisFunValDerAllList;
    typedef typename TensorBsplineBasis<d, T>::BasisFunValDerAllList_ptr BasisFunValDerAllList_ptr;
    using HyperPlane = PhyTensorBsplineBasis<d - 1, N, T>;
    using HyperPlaneSharedPts = std::shared_ptr<PhyTensorBsplineBasis<d - 1, N, T>>;

public:
    PhyTensorBsplineBasis();

    PhyTensorBsplineBasis( const BsplineBasis<T>&, const GeometryVector& );

    PhyTensorBsplineBasis( const BsplineBasis<T>&, const BsplineBasis<T>&, const GeometryVector& );

    PhyTensorBsplineBasis( const BsplineBasis<T>&, const BsplineBasis<T>&, const BsplineBasis<T>&, const GeometryVector& );

    PhyTensorBsplineBasis( const std::vector<KnotVector<T>>&, const GeometryVector& );

    PhyTensorBsplineBasis( const KnotVector<T>&, const GeometryVector& );

    PhyTensorBsplineBasis( const KnotVector<T>&, const KnotVector<T>&, const GeometryVector& );

    PhyTensorBsplineBasis( const KnotVector<T>&, const KnotVector<T>&, const KnotVector<T>&, const GeometryVector& );

    PhyTensorBsplineBasis( const std::vector<KnotVector<T>>&, const Eigen::Matrix<T, Eigen::Dynamic, 1>& );

    virtual ~PhyTensorBsplineBasis()
    {
    }

    virtual PhyPts AffineMap( const Pts&, const DiffPattern& i = DiffPattern( d, 0 ) ) const;

    virtual T Jacobian( const Pts& ) const;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> JacobianMatrix( const Pts& ) const;

    inline Pts Middle() const
    {
        Pts u;
        for ( int i = 0; i != d; i++ )
            u( i ) = ( this->_basis[i].DomainStart() + this->_basis[i].DomainEnd() ) * .5;
        return u;
    }

    //! Return physical middle of the patch
    inline PhyPts PhyMiddle() const
    {
        return AffineMap( Middle() );
    }

    bool InversePts( const PhyPts&, Pts&, int = 1e2, T = std::numeric_limits<T>::epsilon() * 1e3 ) const;

    bool InversePts( const vector&, vector&, int = 1e2, T = std::numeric_limits<T>::epsilon() * 1e3 ) const;

    virtual void DegreeElevate( int, int );

    virtual void UniformRefine( int, int, int );

    void UniformRefineDof( int, int );

    void KnotRefine( int, const KnotVector<T>& );

    virtual void KnotInsertion( int, T, int = 1 );

    void KnotsInsertion( int, const std::vector<T>& );

    inline virtual void DegreeElevate( int p )
    {
        for ( int i = 0; i != d; ++i )
            DegreeElevate( i, p );
    }

    inline virtual void UniformRefine( int r, int m = 1 )
    {
        if ( r == 0 )
            return;
        for ( int i = 0; i != d; ++i )
            UniformRefine( i, r, m );
    }

    virtual void UniformRefineDof( int dof );

    void PrintCtrPts() const;

    inline PhyPts CtrPtsGetter( const int& i ) const
    {
        return _geometricInfo[i];
    }

    const inline GeometryVector& CtrPtsVecGetter() const
    {
        return _geometricInfo;
    }

    inline void CtrPtsSetter( const int& i, const PhyPts& pt )
    {
        ASSERT( i < _geometricInfo.size(), "The control point index is out of range.\n" );
        _geometricInfo[i] = pt;
    }

    virtual HyperPlaneSharedPts MakeHyperPlane( const int& orientation, const int& layer ) const;

    std::function<std::vector<T>( const Pts& pos )> AffineMapFunc()
    {
        return [this]( const Pts& pos ) {
            PhyPts tmp = AffineMap( pos );

            return std::vector<T>( tmp.data(), tmp.data() + N );
        };
    }

    // Only defined for 2D domain represented by 2D parametric domain
    template <int D = d, int n = N>
    typename std::enable_if<D == 2 && n == 2, BasisFunValDerAllList_ptr>::type Eval1PhyDerAllTensor( const vector& u ) const;

    template <int D = d, int n = N>
    typename std::enable_if<D == 2 && n == 2, BasisFunValDerAllList_ptr>::type Eval2PhyDerAllTensor( const vector& u ) const;

    template <int D = d, int n = N>
    typename std::enable_if<D == 2 && n == 2, BasisFunValDerAllList_ptr>::type Eval3PhyDerAllTensor( const vector& u ) const;

    virtual void CreateCurrentConfig()
    {
        _currentConfig = std::unique_ptr<PhyTensorBsplineBasis<d, N, T>>(
            new PhyTensorBsplineBasis<d, N, T>( this->KnotVectorsGetter(), _geometricInfo ) );
    }

    inline const PhyTensorBsplineBasis<d, N, T>& CurrentConfigGetter() const
    {
        ASSERT( _currentConfig != nullptr, "You have not created the current configuration yet." );
        return *_currentConfig;
    }

    template <typename Derived>
    void UpdateGeometryVector( const Eigen::MatrixBase<Derived>& u )
    {
        ASSERT( u.rows() == N && u.cols() == _geometricInfo.size(), "The size of u is incorrect." );
        for ( int i = 0; i < _geometricInfo.size(); i++ )
        {
            _geometricInfo[i] += u.col( i );
        }
    }

    template <typename Derived>
    void UpdateCurrentGeometryVector( const Eigen::MatrixBase<Derived>& u )
    {
        ASSERT( _currentConfig != nullptr, "You have not created the current configuration yet." );
        for ( int i = 0; i < ( _currentConfig->_geometricInfo ).size(); i++ )
        {
            ( _currentConfig->_geometricInfo )[i] += u.col( i );
        }
    }

    void UpdateGeometryVector()
    {
        ASSERT( _currentConfig != nullptr, "You have not created the current configuration yet." );
        _geometricInfo = _currentConfig->_geometricInfo;
    }
    bool IsCurrentAvailable()
    {
        return _currentConfig != nullptr;
    }

protected:
    GeometryVector _geometricInfo;
    std::unique_ptr<PhyTensorBsplineBasis<d, N, T>> _currentConfig{nullptr};
};

template <>
PhyTensorBsplineBasis<2, 1, double>::PhyTensorBsplineBasis( const std::vector<KnotVector<double>>& base,
                                                            const Eigen::Matrix<double, Eigen::Dynamic, 1>& geometry )
    : TensorBsplineBasis<2, double>( base )
{
    ASSERT( geometry.rows() == ( this->TensorBsplineBasis<2, double>::GetDof() ),
            "Invalid geometrical information input, check size bro." );
    for ( int i = 0; i != geometry.rows(); ++i )
    {
        _geometricInfo.push_back( Eigen::Matrix<double, 1, 1>( geometry( i ) ) );
    }
}

template <>
PhyTensorBsplineBasis<2, 2, double>::PhyTensorBsplineBasis( const std::vector<KnotVector<double>>& base,
                                                            const Eigen::Matrix<double, Eigen::Dynamic, 1>& geometry )
    : TensorBsplineBasis<2, double>( base )
{
    ASSERT( geometry.rows() == ( 2 * this->TensorBsplineBasis<2, double>::GetDof() ),
            "Invalid geometrical information input, check size bro." );
    for ( int i = 0; i != this->TensorBsplineBasis<2, double>::GetDof(); ++i )
    {
        _geometricInfo.push_back( Eigen::Matrix<double, 2, 1>( geometry( 2 * i ), geometry( 2 * i + 1 ) ) );
    }
}

// template <>
// PhyTensorBsplineBasis<2, 1, long double>::PhyTensorBsplineBasis(const std::vector<KnotVector<long double>> &base,
//                                                            const Eigen::Matrix<long double, Eigen::Dynamic, 1> &geometry)
//     : TensorBsplineBasis<2, long double>(
//           base)
// {
//     ASSERT(geometry.rows() == (this->TensorBsplineBasis<2, long double>::GetDof()),
//            "Invalid geometrical information input, check size bro.");
//     for (int i = 0; i != geometry.rows(); ++i)
//     {
//         _geometricInfo.push_back(Eigen::Matrix<long double, 1, 1>(geometry(i)));
//     }
// }

// This is a helper class for completing the function definition.
template <int N, typename T>
class PhyTensorBsplineBasis<0, N, T> : public TensorBsplineBasis<0, T>
{
public:
    using HyperPlane = PhyTensorBsplineBasis<-1, N, T>;
    using HyperPlaneSharedPts = std::shared_ptr<PhyTensorBsplineBasis<-1, N, T>>;
    using GeometryVector = Accessory::ContPtsList<T, N>;
    PhyTensorBsplineBasis()
    {
    }

    PhyTensorBsplineBasis( const std::vector<KnotVector<T>>&, const GeometryVector& )
    {
    }

    HyperPlane MakeHyperPlane( const int& orientation, const int& layer ) const
    {
    }

    ~PhyTensorBsplineBasis(){};
};

#ifndef PHYTENSORBSPLINEBASIS_HPP
#include "../src/PhyTensorBsplineBasis.hpp"
#endif //
