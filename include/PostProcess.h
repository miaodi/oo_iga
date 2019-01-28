//
// Created by di miao on 9/12/17.
//

#ifndef OO_IGA_POSTPROCESS_H
#define OO_IGA_POSTPROCESS_H

#include "PhyTensorBsplineBasis.h"
#include "QuadratureRule.h"
#include "Topology.hpp"
#include <fstream>

template <typename T, int N>
class PostProcess
{
public:
    using Coordinate = typename Element<2, 2, T>::Coordinate;
    using CoordinatePairList = typename Element<2, 2, T>::CoordinatePairList;
    using Quadlist = typename QuadratureRule<T>::QuadList;
    using Functor = std::function<std::vector<T>( const Coordinate& )>;
    using ResultPtr = std::pair<std::shared_ptr<Surface<2, T>>, std::shared_ptr<PhyTensorBsplineBasis<2, N, T>>>;

public:
    PostProcess( const std::vector<std::shared_ptr<Surface<2, T>>>& dom,
                 const std::vector<std::shared_ptr<PhyTensorNURBSBasis<2, N, T>>>& res,
                 const Functor& target )
        : _targetFunction( target )
    {
        ASSERT( dom.size() == res.size(),
                "The number of computational domains and that of result domains are not consistent." );
        for ( int i = 0; i < dom.size(); ++i )
        {
            _results.push_back( std::make_pair( dom[i], res[i] ) );
        }
    };

    T RelativeL2Error()
    {
        T err_norm{0};
        T val_norm{0};
        for ( const auto& result : _results )
        {
            T err_norm_per_patch{0};
            T val_norm_per_patch{0};
            auto deg_x = result.first->GetDomain()->GetDegree( 0 );
            auto deg_y = result.first->GetDomain()->GetDegree( 1 );
            QuadratureRule<T> quadrature;
            quadrature.SetUpQuadrature( deg_x >= deg_y ? ( deg_x + 1 ) : ( deg_y + 1 ) );
            CoordinatePairList elements;
            result.first->KnotSpansGetter( elements );
            int num_of_threads = 8;
            std::vector<std::thread> threads( num_of_threads );
            const int grainsize = elements.size() / num_of_threads;
            auto work_iter = elements.begin();
            std::mutex mtx;
            auto lambda = [&]( typename CoordinatePairList::iterator begin, typename CoordinatePairList::iterator end ) -> void {
                T err_v{0}, val_v{0};
                for ( auto it = begin; it != end; ++it )
                {
                    Quadlist quadratures;
                    quadrature.MapToQuadrature( *it, quadratures );
                    for ( const auto& i : quadratures )
                    {
                        for ( int d = 0; d < N; d++ )
                        {
                            val_v += i.second * pow( _targetFunction( result.first->GetDomain()->AffineMap( i.first ) )[d], 2 );
                            err_v += i.second * pow( _targetFunction( result.first->GetDomain()->AffineMap( i.first ) )[d] -
                                                         result.second->AffineMap( i.first )( d ),
                                                     2 );
                        }
                    }
                }
                std::lock_guard<std::mutex> lock( mtx );
                val_norm_per_patch += val_v;
                err_norm_per_patch += err_v;
            };

            for ( auto it = std::begin( threads ); it != std::end( threads ) - 1; ++it )
            {
                *it = std::thread( lambda, work_iter, work_iter + grainsize );
                work_iter += grainsize;
            }
            threads.back() = std::thread( lambda, work_iter, elements.end() );
            for ( auto& i : threads )
            {
                i.join();
            }
            err_norm += err_norm_per_patch;
            val_norm += val_norm_per_patch;
        }
        return sqrt( err_norm / val_norm );
    }

    T RelativeH2Error()
    {
        T err_norm{0};
        T val_norm{0};
        for ( const auto& result : _results )
        {
            T err_norm_per_patch{0};
            T val_norm_per_patch{0};
            auto deg_x = result.first->GetDomain()->GetDegree( 0 );
            auto deg_y = result.first->GetDomain()->GetDegree( 1 );
            QuadratureRule<T> quadrature;
            quadrature.SetUpQuadrature( deg_x >= deg_y ? ( deg_x + 1 ) : ( deg_y + 1 ) );
            CoordinatePairList elements;
            result.first->KnotSpansGetter( elements );
            int num_of_threads = 8;
            std::vector<std::thread> threads( num_of_threads );
            const int grainsize = elements.size() / num_of_threads;
            auto work_iter = elements.begin();
            std::mutex mtx;
            auto lambda = [&]( typename CoordinatePairList::iterator begin, typename CoordinatePairList::iterator end ) -> void {
                T err_v{0}, val_v{0};
                for ( auto it = begin; it != end; ++it )
                {
                    Quadlist quadratures;
                    quadrature.MapToQuadrature( *it, quadratures );
                    for ( const auto& i : quadratures )
                    {
                        auto evals = result.first->GetDomain()->Eval2PhyDerAllTensor( i.first );
                        std::vector<T> approx( 6, 0.0 );
                        for ( const auto& j : *evals )
                        {
                            approx[0] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[0];
                            approx[1] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[1];
                            approx[2] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[2];
                            approx[3] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[3];
                            approx[4] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[4];
                            approx[5] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[5];
                        }
                        auto exact = _targetFunction( result.first->GetDomain()->AffineMap( i.first ) );
                        val_v += i.second * ( pow( exact[0], 2 ) + pow( exact[1], 2 ) + pow( exact[2], 2 ) +
                                              pow( exact[3], 2 ) + pow( exact[4], 2 ) + pow( exact[5], 2 ) );
                        err_v += i.second * ( pow( exact[0] - approx[0], 2 ) + pow( exact[1] - approx[1], 2 ) +
                                              pow( exact[2] - approx[2], 2 ) + pow( exact[3] - approx[3], 2 ) +
                                              pow( exact[4] - approx[4], 2 ) + pow( exact[5] - approx[5], 2 ) );
                    }
                }
                std::lock_guard<std::mutex> lock( mtx );
                val_norm_per_patch += val_v;
                err_norm_per_patch += err_v;
            };

            for ( auto it = std::begin( threads ); it != std::end( threads ) - 1; ++it )
            {
                *it = std::thread( lambda, work_iter, work_iter + grainsize );
                work_iter += grainsize;
            }
            threads.back() = std::thread( lambda, work_iter, elements.end() );
            for ( auto& i : threads )
            {
                i.join();
            }
            err_norm += err_norm_per_patch;
            val_norm += val_norm_per_patch;
        }
        return sqrt( err_norm / val_norm );
    }

    T RelativeH1Error()
    {
        T err_norm{0};
        T val_norm{0};
        for ( const auto& result : _results )
        {
            T err_norm_per_patch{0};
            T val_norm_per_patch{0};
            auto deg_x = result.first->GetDomain()->GetDegree( 0 );
            auto deg_y = result.first->GetDomain()->GetDegree( 1 );
            QuadratureRule<T> quadrature;
            quadrature.SetUpQuadrature( deg_x >= deg_y ? ( deg_x + 1 ) : ( deg_y + 1 ) );
            CoordinatePairList elements;
            result.first->KnotSpansGetter( elements );
            int num_of_threads = 8;
            std::vector<std::thread> threads( num_of_threads );
            const int grainsize = elements.size() / num_of_threads;
            auto work_iter = elements.begin();
            std::mutex mtx;
            auto lambda = [&]( typename CoordinatePairList::iterator begin, typename CoordinatePairList::iterator end ) -> void {
                T err_v{0}, val_v{0};
                for ( auto it = begin; it != end; ++it )
                {
                    Quadlist quadratures;
                    quadrature.MapToQuadrature( *it, quadratures );
                    for ( const auto& i : quadratures )
                    {
                        auto evals = result.first->GetDomain()->Eval1PhyDerAllTensor( i.first );
                        std::vector<T> approx( 3, 0.0 );
                        for ( const auto& j : *evals )
                        {
                            approx[0] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[0];
                            approx[1] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[1];
                            approx[2] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[2];
                        }
                        auto exact = _targetFunction( result.first->GetDomain()->AffineMap( i.first ) );
                        val_v += i.second * ( pow( exact[0], 2 ) + pow( exact[1], 2 ) + pow( exact[2], 2 ) );
                        err_v += i.second * ( pow( exact[0] - approx[0], 2 ) + pow( exact[1] - approx[1], 2 ) +
                                              pow( exact[2] - approx[2], 2 ) );
                    }
                }
                std::lock_guard<std::mutex> lock( mtx );
                val_norm_per_patch += val_v;
                err_norm_per_patch += err_v;
            };

            for ( auto it = std::begin( threads ); it != std::end( threads ) - 1; ++it )
            {
                *it = std::thread( lambda, work_iter, work_iter + grainsize );
                work_iter += grainsize;
            }
            threads.back() = std::thread( lambda, work_iter, elements.end() );
            for ( auto& i : threads )
            {
                i.join();
            }
            err_norm += err_norm_per_patch;
            val_norm += val_norm_per_patch;
        }
        return sqrt( err_norm / val_norm );
    }

    T RelativeEnergyError()
    {
        T err_norm{0};
        T val_norm{0};

        T nu = 0.3;
        T E = 1e5;
        Eigen::MatrixXd constitutive( 3, 3 );
        constitutive << 1 - nu, nu, 0, nu, 1 - nu, 0, 0, 0, ( 1.0 - 2 * nu ) / 2;
        constitutive *= E / ( 1 + nu ) / ( 1 - 2 * nu );

        for ( const auto& result : _results )
        {
            T err_norm_per_patch{0};
            T val_norm_per_patch{0};
            auto deg_x = result.first->GetDomain()->GetDegree( 0 );
            auto deg_y = result.first->GetDomain()->GetDegree( 1 );
            QuadratureRule<T> quadrature;
            quadrature.SetUpQuadrature( deg_x >= deg_y ? ( deg_x + 1 ) : ( deg_y + 1 ) );
            CoordinatePairList elements;
            result.first->KnotSpansGetter( elements );
            int num_of_threads = 8;
            std::vector<std::thread> threads( num_of_threads );
            const int grainsize = elements.size() / num_of_threads;
            auto work_iter = elements.begin();
            std::mutex mtx;
            auto lambda = [&]( typename CoordinatePairList::iterator begin, typename CoordinatePairList::iterator end ) -> void {
                T err_v{0}, val_v{0};
                for ( auto it = begin; it != end; ++it )
                {
                    Quadlist quadratures;
                    quadrature.MapToQuadrature( *it, quadratures );
                    for ( const auto& i : quadratures )
                    {
                        auto evals = result.first->GetDomain()->Eval1PhyDerAllTensor( i.first );
                        Eigen::Vector3d approx;
                        approx.setZero();
                        for ( const auto& j : *evals )
                        {
                            approx( 0 ) += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[1];
                            approx( 1 ) += result.second->CtrPtsGetter( j.first )( 1 ) * j.second[2];
                            approx( 2 ) += result.second->CtrPtsGetter( j.first )( 1 ) * j.second[1] +
                                           result.second->CtrPtsGetter( j.first )( 0 ) * j.second[2];
                        }
                        Eigen::Vector3d exact;
                        exact( 0 ) = _targetFunction( result.first->GetDomain()->AffineMap( i.first ) )[2];
                        exact( 1 ) = _targetFunction( result.first->GetDomain()->AffineMap( i.first ) )[3];
                        exact( 2 ) = _targetFunction( result.first->GetDomain()->AffineMap( i.first ) )[4];
                        exact = constitutive.partialPivLu().solve( exact );
                        auto error_vector = exact - approx;
                        val_v += i.second * ( exact.transpose() * constitutive * exact )( 0 );
                        err_v += i.second * ( error_vector.transpose() * constitutive * error_vector )( 0 );
                    }
                }
                std::lock_guard<std::mutex> lock( mtx );
                val_norm_per_patch += val_v;
                err_norm_per_patch += err_v;
            };

            for ( auto it = std::begin( threads ); it != std::end( threads ) - 1; ++it )
            {
                *it = std::thread( lambda, work_iter, work_iter + grainsize );
                work_iter += grainsize;
            }
            threads.back() = std::thread( lambda, work_iter, elements.end() );
            for ( auto& i : threads )
            {
                i.join();
            }
            err_norm += err_norm_per_patch;
            val_norm += val_norm_per_patch;
        }
        return sqrt( err_norm / val_norm );
    }

    T RelativeMxError( const T D, const T nu, const T L )
    {
        const double Pi = 3.14159265358979323846264338327;
        T err_norm{0};
        T val_norm{0};
        for ( const auto& result : _results )
        {
            T err_norm_per_patch{0};
            T val_norm_per_patch{0};
            auto deg_x = result.first->GetDomain()->GetDegree( 0 );
            auto deg_y = result.first->GetDomain()->GetDegree( 1 );
            QuadratureRule<T> quadrature;
            quadrature.SetUpQuadrature( deg_x >= deg_y ? ( deg_x + 1 ) : ( deg_y + 1 ) );
            CoordinatePairList elements;
            result.first->KnotSpansGetter( elements );
            int num_of_threads = 8;
            std::vector<std::thread> threads( num_of_threads );
            const int grainsize = elements.size() / num_of_threads;
            auto work_iter = elements.begin();
            std::mutex mtx;
            auto lambda = [&]( typename CoordinatePairList::iterator begin, typename CoordinatePairList::iterator end ) -> void {
                T err_v{0}, val_v{0};
                for ( auto it = begin; it != end; ++it )
                {
                    Quadlist quadratures;
                    quadrature.MapToQuadrature( *it, quadratures );
                    for ( const auto& i : quadratures )
                    {
                        auto evals = result.first->GetDomain()->Eval2PhyDerAllTensor( i.first );
                        std::vector<T> approx( 6, 0.0 );
                        for ( const auto& j : *evals )
                        {
                            approx[0] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[0];
                            approx[1] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[1];
                            approx[2] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[2];
                            approx[3] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[3];
                            approx[4] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[4];
                            approx[5] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[5];
                        }
                        auto u = result.first->GetDomain()->AffineMap( i.first );
                        auto exact =
                            ( 1 + nu ) * pow( L, 2 ) * sin( Pi * u( 0 ) / L ) * sin( Pi * u( 1 ) / L ) / pow( Pi, 2 ) / 4;
                        val_v += i.second * ( pow( exact, 2 ) );
                        err_v += i.second * ( pow( exact + D * ( approx[3] + nu * approx[5] ), 2 ) );
                    }
                }
                std::lock_guard<std::mutex> lock( mtx );
                val_norm_per_patch += val_v;
                err_norm_per_patch += err_v;
            };

            for ( auto it = std::begin( threads ); it != std::end( threads ) - 1; ++it )
            {
                *it = std::thread( lambda, work_iter, work_iter + grainsize );
                work_iter += grainsize;
            }
            threads.back() = std::thread( lambda, work_iter, elements.end() );
            for ( auto& i : threads )
            {
                i.join();
            }
            err_norm += err_norm_per_patch;
            val_norm += val_norm_per_patch;
        }
        return sqrt( err_norm / val_norm );
    }

    T RelativeMxyError( const T D, const T nu, const T L )
    {
        const double Pi = 3.14159265358979323846264338327;
        T err_norm{0};
        T val_norm{0};
        for ( const auto& result : _results )
        {
            T err_norm_per_patch{0};
            T val_norm_per_patch{0};
            auto deg_x = result.first->GetDomain()->GetDegree( 0 );
            auto deg_y = result.first->GetDomain()->GetDegree( 1 );
            QuadratureRule<T> quadrature;
            quadrature.SetUpQuadrature( deg_x >= deg_y ? ( deg_x + 1 ) : ( deg_y + 1 ) );
            CoordinatePairList elements;
            result.first->KnotSpansGetter( elements );
            int num_of_threads = 8;
            std::vector<std::thread> threads( num_of_threads );
            const int grainsize = elements.size() / num_of_threads;
            auto work_iter = elements.begin();
            std::mutex mtx;
            auto lambda = [&]( typename CoordinatePairList::iterator begin, typename CoordinatePairList::iterator end ) -> void {
                T err_v{0}, val_v{0};
                for ( auto it = begin; it != end; ++it )
                {
                    Quadlist quadratures;
                    quadrature.MapToQuadrature( *it, quadratures );
                    for ( const auto& i : quadratures )
                    {
                        auto evals = result.first->GetDomain()->Eval2PhyDerAllTensor( i.first );
                        std::vector<T> approx( 6, 0.0 );
                        for ( const auto& j : *evals )
                        {
                            approx[0] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[0];
                            approx[1] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[1];
                            approx[2] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[2];
                            approx[3] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[3];
                            approx[4] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[4];
                            approx[5] += result.second->CtrPtsGetter( j.first )( 0 ) * j.second[5];
                        }
                        auto u = result.first->GetDomain()->AffineMap( i.first );
                        auto exact =
                            -( 1 - nu ) * pow( L, 2 ) * cos( Pi * u( 0 ) / L ) * cos( Pi * u( 1 ) / L ) / pow( Pi, 2 ) / 4;
                        val_v += i.second * ( pow( exact, 2 ) );
                        err_v += i.second * ( pow( exact + D * ( 1 - nu ) * approx[4], 2 ) );
                    }
                }
                std::lock_guard<std::mutex> lock( mtx );
                val_norm_per_patch += val_v;
                err_norm_per_patch += err_v;
            };

            for ( auto it = std::begin( threads ); it != std::end( threads ) - 1; ++it )
            {
                *it = std::thread( lambda, work_iter, work_iter + grainsize );
                work_iter += grainsize;
            }
            threads.back() = std::thread( lambda, work_iter, elements.end() );
            for ( auto& i : threads )
            {
                i.join();
            }
            err_norm += err_norm_per_patch;
            val_norm += val_norm_per_patch;
        }
        return sqrt( err_norm / val_norm );
    }

    void Plot( const int size )
    {
        for ( int i = 0; i < _results.size(); i++ )
        {
            std::ofstream file;
            std::string name;
            name = "domain_" + std::to_string( i ) + ".txt";
            file.open( name );
            for ( int x = 0; x <= size; x++ )
            {
                for ( int y = 0; y <= size; y++ )
                {
                    Coordinate u;
                    u << 1.0 * x / size, 1.0 * y / size;
                    Coordinate pos = _results[i].first->GetDomain()->AffineMap( u );
                    file << pos( 0 ) << " " << pos( 1 ) << " "
                         << _targetFunction( pos )[0] - _results[i].second->AffineMap( u )( 0 ) << std::endl;
                }
            }
            file.close();
        }
    }

private:
    std::vector<ResultPtr> _results;
    Functor _targetFunction;
};

#endif // OO_IGA_POSTPROCESS_H
