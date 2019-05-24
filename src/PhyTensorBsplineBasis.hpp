//
// Created by di miao on 12/29/16.
//

#include "PhyTensorBsplineBasis.h"
// #include <boost/multiprecision/gmp.hpp>

template <int d, int N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis() : TensorBsplineBasis<d, T>()
{
}

template <int d, int N, typename T>
typename PhyTensorBsplineBasis<d, N, T>::PhyPts PhyTensorBsplineBasis<d, N, T>::AffineMap( const PhyTensorBsplineBasis<d, N, T>::Pts& u,
                                                                                           const DiffPattern& i ) const
{
    ASSERT( this->InDomain( u ), "Given parametric point is not in the domain.\n" );
    PhyTensorBsplineBasis<d, N, T>::PhyPts result;
    result.setZero();
    auto p = this->EvalTensor( u, i );
    for ( auto it = p->begin(); it != p->end(); ++it )
    {
        result += _geometricInfo[it->first] * it->second;
    }
    return result;
}

template <int d, int N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis( const BsplineBasis<T>& baseX,
                                                       const PhyTensorBsplineBasis<d, N, T>::GeometryVector& geometry )
    : TensorBsplineBasis<d, T>( baseX ), _geometricInfo( geometry )
{
    ASSERT( ( this->TensorBsplineBasis<d, T>::GetDof() ) == geometry.size(),
            "Invalid geometrical information input, check size bro." );
}

template <int d, int N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis( const BsplineBasis<T>& baseX,
                                                       const BsplineBasis<T>& baseY,
                                                       const PhyTensorBsplineBasis::GeometryVector& geometry )
    : TensorBsplineBasis<d, T>( baseX, baseY ), _geometricInfo( geometry )
{
    ASSERT( ( this->TensorBsplineBasis<d, T>::GetDof() ) == geometry.size(),
            "Invalid geometrical information input, check size bro." );
}

template <int d, int N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis( const BsplineBasis<T>& baseX,
                                                       const BsplineBasis<T>& baseY,
                                                       const BsplineBasis<T>& baseZ,
                                                       const PhyTensorBsplineBasis::GeometryVector& geometry )
    : TensorBsplineBasis<d, T>( baseX, baseY, baseZ ), _geometricInfo( geometry )
{
    ASSERT( ( this->TensorBsplineBasis<d, T>::GetDof() ) == geometry.size(),
            "Invalid geometrical information input, check size bro." );
}

template <int d, int N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis( const KnotVector<T>& baseX,
                                                       const PhyTensorBsplineBasis<d, N, T>::GeometryVector& geometry )
    : TensorBsplineBasis<d, T>( baseX ), _geometricInfo( geometry )
{
    ASSERT( ( this->TensorBsplineBasis<d, T>::GetDof() ) == geometry.size(),
            "Invalid geometrical information input, check size bro." );
}

template <int d, int N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis( const KnotVector<T>& baseX,
                                                       const KnotVector<T>& baseY,
                                                       const PhyTensorBsplineBasis::GeometryVector& geometry )
    : TensorBsplineBasis<d, T>( baseX, baseY ), _geometricInfo( geometry )
{
    ASSERT( ( this->TensorBsplineBasis<d, T>::GetDof() ) == geometry.size(),
            "Invalid geometrical information input, check size bro." );
}

template <int d, int N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis( const KnotVector<T>& baseX,
                                                       const KnotVector<T>& baseY,
                                                       const KnotVector<T>& baseZ,
                                                       const PhyTensorBsplineBasis::GeometryVector& geometry )
    : TensorBsplineBasis<d, T>( baseX, baseY, baseZ ), _geometricInfo( geometry )
{
    ASSERT( ( this->TensorBsplineBasis<d, T>::GetDof() ) == geometry.size(),
            "Invalid geometrical information input, check size bro." );
}

template <int d, int N, typename T>
PhyTensorBsplineBasis<d, N, T>::PhyTensorBsplineBasis( const std::vector<KnotVector<T>>& base,
                                                       const PhyTensorBsplineBasis::GeometryVector& geometry )
    : TensorBsplineBasis<d, T>( base ), _geometricInfo( geometry )
{
}

template <int d, int N, typename T>
void PhyTensorBsplineBasis<d, N, T>::DegreeElevate( int orientation, int r )
{
    if ( r == 0 )
        return;
    std::vector<int> indexes( d, 0 );
    std::vector<int> endPerIndex;
    for ( int direction = 0; direction != d; ++direction )
    {
        endPerIndex.push_back( this->GetDof( direction ) );
    }
    ASSERT( orientation < d, "Invalid degree elevate orientation" );
    GeometryVector temp1;
    std::vector<int> MultiIndex( d );
    std::function<void( std::vector<int>&, const std::vector<int>&, int )> recursive;
    TensorBsplineBasis<d, T> tmp1;
    KnotVector<T> knot_temp_storage;
    bool called = false;
    recursive = [this, &orientation, &called, &knot_temp_storage, &tmp1, r, &temp1, &MultiIndex, &recursive](
                    std::vector<int>& indexes, const std::vector<int>& endPerIndex, int direction ) {
        if ( direction == indexes.size() )
        {
            Accessory::ContPtsList<T, N> ElevateList;
            for ( int i = 0; i != endPerIndex[orientation]; ++i )
            {
                MultiIndex[orientation] = i;
                auto index = this->Index( MultiIndex );
                ElevateList.push_back( _geometricInfo[index] );
            }
            KnotVector<T> tmp( this->_basis[orientation].Knots() );
            Accessory::degreeElevate<T, N>( r, tmp, ElevateList );
            if ( knot_temp_storage.GetSize() == 0 )
            {
                for ( int i = 0; i != d; ++i )
                {
                    if ( i == orientation )
                    {
                        tmp1.ChangeKnots( tmp, i );
                    }
                    else
                    {
                        tmp1.ChangeKnots( this->_basis[i].Knots(), i );
                    }
                }
                temp1.resize( tmp1.GetDof() );
                knot_temp_storage = tmp;
            }
            for ( int i = 0; i != ElevateList.size(); ++i )
            {
                MultiIndex[orientation] = i;
                auto index = tmp1.Index( MultiIndex );
                temp1[index] = ElevateList[i];
            }
        }
        else
        {
            for ( indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++ )
            {
                MultiIndex[direction] = indexes[direction];
                if ( direction == orientation && called == false )
                {
                    called = true;
                }
                else if ( direction == orientation && called == true )
                {
                    break;
                }
                called = false;
                recursive( indexes, endPerIndex, direction + 1 );
            }
        }
    };
    recursive( indexes, endPerIndex, 0 );
    TensorBsplineBasis<d, T>::_basis[orientation] = knot_temp_storage;
    _geometricInfo = temp1;
}

template <int d, int N, typename T>
void PhyTensorBsplineBasis<d, N, T>::KnotRefine( int orientation, const KnotVector<T>& X )
{
    ASSERT( orientation < d, "Invalid knot vector refine orientation" );

    std::vector<int> indexes( d, 0 );
    std::vector<int> endPerIndex;
    for ( int direction = 0; direction != d; ++direction )
    {
        endPerIndex.push_back( this->GetDof( direction ) );
    }
    GeometryVector temp1;
    std::vector<int> MultiIndex( d );
    std::function<void( std::vector<int>&, const std::vector<int>&, int )> recursive;
    TensorBsplineBasis<d, T> tmp1;
    KnotVector<T> knot_temp_storage;
    knot_temp_storage.resize( 0 );
    bool called = false;
    recursive = [this, &X, &orientation, &called, &knot_temp_storage, &tmp1, &temp1, &MultiIndex, &recursive](
                    std::vector<int>& indexes, const std::vector<int>& endPerIndex, int direction ) {
        if ( direction == indexes.size() )
        {
            Accessory::ContPtsList<T, N> RefineList;
            for ( int i = 0; i != endPerIndex[orientation]; ++i )
            {
                MultiIndex[orientation] = i;
                auto index = this->Index( MultiIndex );
                RefineList.push_back( _geometricInfo[index] );
            }
            KnotVector<T> tmp( this->_basis[orientation].Knots() );
            Accessory::refineKnotVectorCurve<T, N>( X, tmp, RefineList );
            if ( knot_temp_storage.GetSize() == 0 )
            {
                for ( int i = 0; i != d; ++i )
                {
                    if ( i == orientation )
                    {
                        tmp1.ChangeKnots( tmp, i );
                    }
                    else
                    {
                        tmp1.ChangeKnots( this->_basis[i].Knots(), i );
                    }
                }
                temp1.resize( tmp1.GetDof() );
                knot_temp_storage = tmp;
            }
            for ( int i = 0; i != RefineList.size(); ++i )
            {
                MultiIndex[orientation] = i;
                auto index = tmp1.Index( MultiIndex );
                temp1[index] = RefineList[i];
            }
        }
        else
        {
            for ( indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++ )
            {
                MultiIndex[direction] = indexes[direction];
                if ( direction == orientation && called == false )
                {
                    called = true;
                }
                else if ( direction == orientation && called == true )
                {
                    break;
                }
                called = false;
                recursive( indexes, endPerIndex, direction + 1 );
            }
        }
    };
    recursive( indexes, endPerIndex, 0 );
    TensorBsplineBasis<d, T>::_basis[orientation] = knot_temp_storage;
    _geometricInfo = temp1;
}

template <int d, int N, typename T>
void PhyTensorBsplineBasis<d, N, T>::UniformRefine( int orientation, int r, int m )
{
    ASSERT( orientation < d, "Invalid knot vector refine orientation" );
    if ( r == 0 )
        return;
    KnotVector<T> knot_temp_storage( this->_basis[orientation].Knots() );
    knot_temp_storage.UniformRefine( r, m );
    KnotVector<T> X = knot_temp_storage.Difference( this->_basis[orientation].Knots() );
    this->KnotRefine( orientation, X );
}

template <int d, int N, typename T>
void PhyTensorBsplineBasis<d, N, T>::UniformRefineDof( int dof )
{
    ASSERT( dof >= this->GetDof(), "Given dof is too small.\n" );
    // use bisearch to find best insertion per span
    int insertion_per_span_u{0}, insertion_per_span_t{dof}, insertion_per_span{0};
    while ( insertion_per_span_t - insertion_per_span_u > 1 )
    {
        insertion_per_span = ( insertion_per_span_t + insertion_per_span_u ) / 2;
        int new_dof = 1;
        for ( int i = 0; i != d; ++i )
        {
            new_dof *= this->_basis[i].Knots().GetSpanSize() * insertion_per_span + this->_basis[i].Knots().GetDOF();
        }
        if ( new_dof > dof )
        {
            insertion_per_span_t = insertion_per_span;
        }
        else
        {
            insertion_per_span_u = insertion_per_span;
        }
    }
    insertion_per_span = insertion_per_span_u;

    for ( int i = 0; i != d; ++i )
    {
        UniformRefineDof( i, insertion_per_span * this->_basis[i].Knots().GetSpanSize() + this->_basis[i].Knots().GetDOF() );
    }
}

template <int d, int N, typename T>
void PhyTensorBsplineBasis<d, N, T>::UniformRefineDof( int orientation, int dof )
{
    ASSERT( orientation < d, "Invalid knot vector refine orientation" );
    KnotVector<T> X = this->_basis[orientation].Knots().RefineToDofKnotVector( dof );
    this->KnotRefine( orientation, X );
}

template <int d, int N, typename T>
template <int D, int n>
typename std::enable_if<D == 2 && n == 2, typename PhyTensorBsplineBasis<d, N, T>::BasisFunValDerAllList_ptr>::type PhyTensorBsplineBasis<d, N, T>::Eval1PhyDerAllTensor(
    const vector& u ) const
{
    BasisFunValDerAllList_ptr parametric = this->EvalDerAllTensor( u, 1 );

    Eigen::Matrix<T, 2, 2> Jacobian = JacobianMatrix( u ).transpose();
    for ( auto& i : *parametric )
    {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> temp( i.second.data() + 1, i.second.size() - 1 );
        Eigen::Matrix<T, Eigen::Dynamic, 1> solution = Jacobian.partialPivLu().solve( temp );
        i.second[1] = solution( 0 );
        i.second[2] = solution( 1 );
    }
    return parametric;
}

template <int d, int N, typename T>
template <int D, int n>
typename std::enable_if<D == 2 && n == 2, typename PhyTensorBsplineBasis<d, N, T>::BasisFunValDerAllList_ptr>::type PhyTensorBsplineBasis<d, N, T>::Eval2PhyDerAllTensor(
    const PhyTensorBsplineBasis::vector& u ) const
{
    auto parametric = this->EvalDerAllTensor( u, 2 );
    Eigen::Matrix<T, 2, 1> Pxi, Peta, PxiPxi, PxiPeta, PetaPeta;
    Pxi.setZero();
    Peta.setZero();
    PxiPxi.setZero();
    PxiPeta.setZero();
    PetaPeta.setZero();
    for ( const auto& i : *parametric )
    {
        Pxi += i.second[1] * _geometricInfo[i.first];
        Peta += i.second[2] * _geometricInfo[i.first];
        PxiPxi += i.second[3] * _geometricInfo[i.first];
        PxiPeta += i.second[4] * _geometricInfo[i.first];
        PetaPeta += i.second[5] * _geometricInfo[i.first];
    }
    Eigen::Matrix<T, 5, 5> Hessian;
    Hessian << Pxi( 0 ), Pxi( 1 ), 0, 0, 0, Peta( 0 ), Peta( 1 ), 0, 0, 0, PxiPxi( 0 ), PxiPxi( 1 ),
        Pxi( 0 ) * Pxi( 0 ), 2 * Pxi( 0 ) * Pxi( 1 ), Pxi( 1 ) * Pxi( 1 ), PxiPeta( 0 ), PxiPeta( 1 ),
        Pxi( 0 ) * Peta( 0 ), Pxi( 0 ) * Peta( 1 ) + Peta( 0 ) * Pxi( 1 ), Pxi( 1 ) * Peta( 1 ), PetaPeta( 0 ),
        PetaPeta( 1 ), Peta( 0 ) * Peta( 0 ), 2 * Peta( 0 ) * Peta( 1 ), Peta( 1 ) * Peta( 1 );
    for ( auto& i : *parametric )
    {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> temp( i.second.data() + 1, i.second.size() - 1 );
        Eigen::Matrix<T, Eigen::Dynamic, 1> solution = Hessian.partialPivLu().solve( temp );
        i.second[1] = solution( 0 );
        i.second[2] = solution( 1 );
        i.second[3] = solution( 2 );
        i.second[4] = solution( 3 );
        i.second[5] = solution( 4 );
    }
    return parametric;
}

template <int d, int N, typename T>
template <int D, int n>
typename std::enable_if<D == 2 && n == 2, typename PhyTensorBsplineBasis<d, N, T>::BasisFunValDerAllList_ptr>::type PhyTensorBsplineBasis<d, N, T>::Eval3PhyDerAllTensor(
    const PhyTensorBsplineBasis::vector& u ) const
{
    auto parametric = this->EvalDerAllTensor( u, 3 );
    Eigen::Matrix<T, 2, 1> Pxi, Peta, PxiPxi, PxiPeta, PetaPeta, PxiPxiPxi, PxiPxiPeta, PxiPetaPeta, PetaPetaPeta;
    Pxi.setZero();
    Peta.setZero();
    PxiPxi.setZero();
    PxiPeta.setZero();
    PetaPeta.setZero();
    PxiPxiPxi.setZero();
    PxiPxiPeta.setZero();
    PxiPetaPeta.setZero();
    PetaPetaPeta.setZero();
    for ( const auto& i : *parametric )
    {
        Pxi += i.second[1] * _geometricInfo[i.first];
        Peta += i.second[2] * _geometricInfo[i.first];
        PxiPxi += i.second[3] * _geometricInfo[i.first];
        PxiPeta += i.second[4] * _geometricInfo[i.first];
        PetaPeta += i.second[5] * _geometricInfo[i.first];
        PxiPxiPxi += i.second[6] * _geometricInfo[i.first];
        PxiPxiPeta += i.second[7] * _geometricInfo[i.first];
        PxiPetaPeta += i.second[8] * _geometricInfo[i.first];
        PetaPetaPeta += i.second[9] * _geometricInfo[i.first];
    }
    Eigen::Matrix<T, 9, 9> Hessian;
    Hessian.setZero();
    Hessian( 0, 0 ) = Pxi( 0 ), Hessian( 0, 1 ) = Pxi( 1 );
    Hessian( 1, 0 ) = Peta( 0 ), Hessian( 1, 1 ) = Peta( 1 );
    Hessian( 2, 0 ) = PxiPxi( 0 ), Hessian( 2, 1 ) = PxiPxi( 1 ), Hessian( 2, 2 ) = Pxi( 0 ) * Pxi( 0 ),
                  Hessian( 2, 3 ) = 2 * Pxi( 0 ) * Pxi( 1 ), Hessian( 2, 4 ) = Pxi( 1 ) * Pxi( 1 );
    Hessian( 3, 0 ) = PxiPeta( 0 ), Hessian( 3, 1 ) = PxiPeta( 1 ), Hessian( 3, 2 ) = Pxi( 0 ) * Peta( 0 ),
                  Hessian( 3, 3 ) = Pxi( 0 ) * Peta( 1 ) + Peta( 0 ) * Pxi( 1 ), Hessian( 3, 4 ) = Pxi( 1 ) * Peta( 1 );
    Hessian( 4, 0 ) = PetaPeta( 0 ), Hessian( 4, 1 ) = PetaPeta( 1 ), Hessian( 4, 2 ) = Peta( 0 ) * Peta( 0 ),
                  Hessian( 4, 3 ) = 2 * Peta( 0 ) * Peta( 1 ), Hessian( 4, 4 ) = Peta( 1 ) * Peta( 1 );

    Hessian( 5, 0 ) = PxiPxiPxi( 0 ), Hessian( 5, 1 ) = PxiPxiPxi( 1 ), Hessian( 5, 2 ) = 3 * Pxi( 0 ) * PxiPxi( 0 ),
                  Hessian( 5, 3 ) = 3 * Pxi( 1 ) * PxiPxi( 0 ) + 3 * Pxi( 0 ) * PxiPxi( 1 ),
                  Hessian( 5, 4 ) = 3 * Pxi( 1 ) * PxiPxi( 1 ), Hessian( 5, 5 ) = pow( Pxi( 0 ), 3 ),
                  Hessian( 5, 6 ) = 3 * pow( Pxi( 0 ), 2 ) * Pxi( 1 ),
                  Hessian( 5, 7 ) = 3 * Pxi( 0 ) * pow( Pxi( 1 ), 2 ), Hessian( 5, 8 ) = pow( Pxi( 1 ), 3 );

    Hessian( 6, 0 ) = PxiPxiPeta( 0 ), Hessian( 6, 1 ) = PxiPxiPeta( 1 ),
                  Hessian( 6, 2 ) = 2 * Pxi( 0 ) * PxiPeta( 0 ) + Peta( 0 ) * PxiPxi( 0 ),
                  Hessian( 6, 3 ) = 2 * Pxi( 1 ) * PxiPeta( 0 ) + 2 * Pxi( 0 ) * PxiPeta( 1 ) +
                                    Peta( 1 ) * PxiPxi( 0 ) + Peta( 0 ) * PxiPxi( 1 ),
                  Hessian( 6, 4 ) = 2 * Pxi( 1 ) * PxiPeta( 1 ) + Peta( 1 ) * PxiPxi( 1 ),
                  Hessian( 6, 5 ) = Peta( 0 ) * pow( Pxi( 0 ), 2 ),
                  Hessian( 6, 6 ) = Peta( 1 ) * pow( Pxi( 0 ), 2 ) + 2 * Peta( 0 ) * Pxi( 0 ) * Pxi( 1 ),
                  Hessian( 6, 7 ) = 2 * Peta( 1 ) * Pxi( 0 ) * Pxi( 1 ) + Peta( 0 ) * pow( Pxi( 1 ), 2 ),
                  Hessian( 6, 8 ) = Peta( 1 ) * pow( Pxi( 1 ), 2 );

    Hessian( 7, 0 ) = PxiPetaPeta( 0 ), Hessian( 7, 1 ) = PxiPetaPeta( 1 ),
                  Hessian( 7, 2 ) = PetaPeta( 0 ) * Pxi( 0 ) + 2 * Peta( 0 ) * PxiPeta( 0 ),
                  Hessian( 7, 3 ) = 2 * Peta( 1 ) * PxiPeta( 0 ) + 2 * Peta( 0 ) * PxiPeta( 1 ) +
                                    Pxi( 0 ) * PetaPeta( 1 ) + Pxi( 1 ) * PetaPeta( 0 ),
                  Hessian( 7, 4 ) = 2 * Peta( 1 ) * PxiPeta( 1 ) + Pxi( 1 ) * PetaPeta( 1 ),
                  Hessian( 7, 5 ) = Pxi( 0 ) * pow( Peta( 0 ), 2 ),
                  Hessian( 7, 6 ) = Pxi( 1 ) * pow( Peta( 0 ), 2 ) + 2 * Peta( 0 ) * Peta( 1 ) * Pxi( 0 ),
                  Hessian( 7, 7 ) = 2 * Peta( 0 ) * Peta( 1 ) * Pxi( 1 ) + Pxi( 0 ) * pow( Peta( 1 ), 2 ),
                  Hessian( 7, 8 ) = Pxi( 1 ) * pow( Peta( 1 ), 2 );

    Hessian( 8, 0 ) = PetaPetaPeta( 0 ), Hessian( 8, 1 ) = PetaPetaPeta( 1 ), Hessian( 8, 2 ) = 3 * Peta( 0 ) * PetaPeta( 0 ),
                  Hessian( 8, 3 ) = 3 * Peta( 1 ) * PetaPeta( 0 ) + 3 * Peta( 0 ) * PetaPeta( 1 ),
                  Hessian( 8, 4 ) = 3 * Peta( 1 ) * PetaPeta( 1 ), Hessian( 8, 5 ) = pow( Peta( 0 ), 3 ),
                  Hessian( 8, 6 ) = 3 * pow( Peta( 0 ), 2 ) * Peta( 1 ),
                  Hessian( 8, 7 ) = 3 * Peta( 0 ) * pow( Peta( 1 ), 2 ), Hessian( 8, 8 ) = pow( Peta( 1 ), 3 );
    for ( auto& i : *parametric )
    {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> temp( i.second.data() + 1, i.second.size() - 1 );
        Eigen::Matrix<T, Eigen::Dynamic, 1> solution = Hessian.partialPivLu().solve( temp );
        i.second[1] = solution( 0 );
        i.second[2] = solution( 1 );
        i.second[3] = solution( 2 );
        i.second[4] = solution( 3 );
        i.second[5] = solution( 4 );
        i.second[6] = solution( 5 );
        i.second[7] = solution( 6 );
        i.second[8] = solution( 7 );
        i.second[9] = solution( 8 );
    }
    return parametric;
}

// TODO This member function is so shitty. need improvement
template <int d, int N, typename T>
typename PhyTensorBsplineBasis<d, N, T>::HyperPlaneSharedPts PhyTensorBsplineBasis<d, N, T>::MakeHyperPlane( const int& orientation,
                                                                                                             const int& layer ) const
{
    ASSERT( orientation < d, "Invalid input vector size." );
    std::vector<KnotVector<T>> hpknotvector;
    for ( int i = 0; i != d; ++i )
    {
        if ( i != orientation )
            hpknotvector.push_back( this->KnotVectorGetter( i ) );
    }
    auto indexList = this->HyperPlaneIndices( orientation, layer );
    GeometryVector tempGeometry;
    for ( const auto& i : *indexList )
    {
        tempGeometry.push_back( _geometricInfo[i] );
    }

    return std::make_shared<HyperPlane>( hpknotvector, tempGeometry );
}

template <int d, int N, typename T>
void PhyTensorBsplineBasis<d, N, T>::KnotInsertion( int orientation, T knot, int m )
{
    std::vector<int> indexes( d, 0 );
    std::vector<int> endPerIndex;
    for ( int direction = 0; direction != d; ++direction )
    {
        endPerIndex.push_back( this->GetDof( direction ) );
    }
    ASSERT( orientation < d, "Invalid insertion orientation" );
    GeometryVector temp1;
    std::vector<int> MultiIndex( d );
    std::function<void( std::vector<int>&, const std::vector<int>&, int )> recursive;
    TensorBsplineBasis<d, T> tmp1;
    KnotVector<T> knot_temp_storage;
    bool called = false;
    recursive = [this, &orientation, &called, &knot_temp_storage, &tmp1, knot, m, &temp1, &MultiIndex, &recursive](
                    std::vector<int>& indexes, const std::vector<int>& endPerIndex, int direction ) {
        if ( direction == indexes.size() )
        {
            Accessory::ContPtsList<T, N> ElevateList;
            for ( int i = 0; i != endPerIndex[orientation]; ++i )
            {
                MultiIndex[orientation] = i;
                auto index = this->Index( MultiIndex );
                ElevateList.push_back( _geometricInfo[index] );
            }
            KnotVector<T> tmp( this->_basis[orientation].Knots() );
            Accessory::knotInsertion<T, N>( knot, m, tmp, ElevateList );
            if ( knot_temp_storage.GetSize() == 0 )
            {
                for ( int i = 0; i != d; ++i )
                {
                    if ( i == orientation )
                    {
                        tmp1.ChangeKnots( tmp, i );
                    }
                    else
                    {
                        tmp1.ChangeKnots( this->_basis[i].Knots(), i );
                    }
                }
                temp1.resize( tmp1.GetDof() );
                knot_temp_storage = tmp;
            }
            for ( int i = 0; i != ElevateList.size(); ++i )
            {
                MultiIndex[orientation] = i;
                auto index = tmp1.Index( MultiIndex );
                temp1[index] = ElevateList[i];
            }
        }
        else
        {
            for ( indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++ )
            {
                MultiIndex[direction] = indexes[direction];
                if ( direction == orientation && called == false )
                {
                    called = true;
                }
                else if ( direction == orientation && called == true )
                {
                    break;
                }
                called = false;
                recursive( indexes, endPerIndex, direction + 1 );
            }
        }
    };
    recursive( indexes, endPerIndex, 0 );
    TensorBsplineBasis<d, T>::_basis[orientation] = knot_temp_storage;
    _geometricInfo = temp1;
}

template <int d, int N, typename T>
void PhyTensorBsplineBasis<d, N, T>::KnotsInsertion( int orientation, const std::vector<T>& knots )
{
    for ( const auto& i : knots )
    {
        this->KnotInsertion( orientation, i );
    }
}

template <int d, int N, typename T>
void PhyTensorBsplineBasis<d, N, T>::PrintCtrPts() const
{
    for ( int i = 0; i != _geometricInfo.size(); ++i )
        std::cout << _geometricInfo[i].transpose() << std::endl;
}

template <int d, int N, typename T>
T PhyTensorBsplineBasis<d, N, T>::Jacobian( const PhyTensorBsplineBasis::Pts& u ) const
{
    ComputeJacobian<d, N, T> temp;
    return temp.compute( this, u );
}

template <int d, int N, typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> PhyTensorBsplineBasis<d, N, T>::JacobianMatrix( const PhyTensorBsplineBasis::Pts& u ) const
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> result( N, d );
    for ( int i = 0; i != d; i++ )
    {
        std::vector<int> differentiation( d, 0 );
        differentiation[i] = 1;
        result.col( i ) = AffineMap( u, differentiation );
    }
    return result;
};

template <int d, int N, typename T>
bool PhyTensorBsplineBasis<d, N, T>::InversePts( const PhyTensorBsplineBasis::PhyPts& phyu,
                                                 PhyTensorBsplineBasis::Pts& result,
                                                 int maxLoop,
                                                 T error ) const
{
    result = Middle();
    Pts suppBegin, suppEnd;
    for ( int i = 0; i != d; ++i )
    {
        suppBegin( i ) = this->DomainStart( i );
        suppEnd( i ) = this->DomainEnd( i );
    }
    int iter = 0;
    do
    {
        for ( int i = 0; i != d; ++i )
        {
            if ( result( i ) < suppBegin( i ) + error )
            {
                result( i ) = suppBegin( i );
            }
            else
            {
                if ( result( i ) > suppEnd( i ) - error )
                {
                    result( i ) = suppEnd( i );
                }
            }
        }
        vector residual = phyu - AffineMap( result );
        if ( residual.norm() <= error )
            return true;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> jacobianMatrix( JacobianMatrix( result ) );
        if ( jacobianMatrix.cols() == jacobianMatrix.rows() )
        {
            residual = jacobianMatrix.partialPivLu().solve( residual );
        }
        else
        {
            residual =
                jacobianMatrix.colPivHouseholderQr().solve( Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity( N, N ) ) * residual;
        }
        result += residual;
    } while ( ++iter <= maxLoop );
    return false;
}

template <int d, int N, typename T>
bool PhyTensorBsplineBasis<d, N, T>::InversePts( const PhyTensorBsplineBasis::vector& phyu,
                                                 PhyTensorBsplineBasis::vector& result,
                                                 int maxLoop,
                                                 T error ) const
{
    ASSERT( phyu.size() == N, "Given point is not consistent with the physical domain." );
    Pts tmp;
    PhyPts phy_tmp = phyu;
    if ( InversePts( phy_tmp, tmp ) )
    {
        result = tmp;
        return true;
    }
    return false;
}

template <int d, int N, typename T>
struct ComputeJacobian
{
    using Pts = typename PhyTensorBsplineBasis<d, N, T>::Pts;
    using PhyPts = typename PhyTensorBsplineBasis<d, N, T>::PhyPts;

    T compute( const PhyTensorBsplineBasis<d, N, T>* domain_ptr, const Pts& u )
    {
        return domain_ptr->JacobianMatrix( u ).determinant();
    }
};

template <typename T>
struct ComputeJacobian<1, 2, T>
{
    using Pts = typename PhyTensorBsplineBasis<1, 2, T>::Pts;
    using PhyPts = typename PhyTensorBsplineBasis<1, 2, T>::PhyPts;

    T compute( const PhyTensorBsplineBasis<1, 2, T>* domain_ptr, const Pts& u )
    {
        PhyPts normal = domain_ptr->JacobianMatrix( u );
        T res = normal.transpose() * normal;
        return sqrt( res );
    }
};

template <typename T>
struct ComputeJacobian<1, 3, T>
{
    using Pts = typename PhyTensorBsplineBasis<1, 3, T>::Pts;
    using PhyPts = typename PhyTensorBsplineBasis<1, 3, T>::PhyPts;

    T compute( const PhyTensorBsplineBasis<1, 3, T>* domain_ptr, const Pts& u )
    {
        PhyPts normal = domain_ptr->JacobianMatrix( u );
        T res = normal.transpose() * normal;
        return sqrt( res );
    }
};

template <typename T>
struct ComputeJacobian<2, 3, T>
{
    using Pts = typename PhyTensorBsplineBasis<2, 3, T>::Pts;
    using PhyPts = typename PhyTensorBsplineBasis<2, 3, T>::PhyPts;

    T compute( const PhyTensorBsplineBasis<2, 3, T>* domain_ptr, const Pts& u )
    {
        Eigen::Matrix<T, 3, 2> Jacob = domain_ptr->JacobianMatrix( u );
        PhyPts normal = Jacob.col( 0 ).cross( Jacob.col( 1 ) );
        T res = normal.transpose() * normal;
        return sqrt( res );
    }
};

// template <int N, typename T>
// PhyTensorBsplineBasis<0, N, T>::PhyTensorBsplineBasis(const T &knot, const PhyPts &point)
//     : TensorBsplineBasis<0, T>(knot), _point(point)
// {
// }

// template <int N, typename T>
// PhyTensorBsplineBasis<0, N, T>::PhyTensorBsplineBasis(const PhyPts &point)
//     : TensorBsplineBasis<0, T>(), _point(point)
// {
// }
