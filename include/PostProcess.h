//
// Created by di miao on 9/12/17.
//

#ifndef OO_IGA_POSTPROCESS_H
#define OO_IGA_POSTPROCESS_H

#include "Topology.hpp"
#include "QuadratureRule.h"
#include "PhyTensorBsplineBasis.h"

template <typename T>
class PostProcess
{
  public:
    using Coordinate = typename Element<2, 2, T>::Coordinate;
    using CoordinatePairList = typename Element<2, 2, T>::CoordinatePairList;
    using Quadlist = typename QuadratureRule<T>::QuadList;
    using Functor = std::function<std::vector<T>(const Coordinate &)>;
    using ResultPtr = std::pair<std::shared_ptr<Surface<2, T>>, std::shared_ptr<PhyTensorBsplineBasis<2, 1, T>>>;

  public:
    PostProcess(const std::vector<std::shared_ptr<Surface<2, T>>> &dom,
                const std::vector<std::shared_ptr<PhyTensorBsplineBasis<2, 1, T>>> &res,
                const Functor &target) : _targetFunction(target)
    {
        ASSERT(dom.size() == res.size(),
               "The number of computational domains and that of result domains are not consistent.");
        for (int i = 0; i < dom.size(); ++i)
        {
            _results.push_back(std::make_pair(dom[i], res[i]));
        }
    };

    T RelativeL2Error()
    {
        T err_norm{0};
        T val_norm{0};
        for (const auto &result : _results)
        {
            auto deg_x = result.first->GetDomain()->GetDegree(0);
            auto deg_y = result.first->GetDomain()->GetDegree(1);
            QuadratureRule<T> quadrature;
            quadrature.SetUpQuadrature(deg_x >= deg_y ? (deg_x + 1) : (deg_y + 1));
            CoordinatePairList elements;
            result.first->KnotSpansGetter(elements);
            int num_of_threads = 8;
            std::vector<std::thread> threads(num_of_threads);
            const int grainsize = elements.size() / num_of_threads;
            auto work_iter = elements.begin();
            std::mutex mtx;
            auto lambda = [&](typename CoordinatePairList::iterator begin,
                              typename CoordinatePairList::iterator end) -> void {
                T err_v{0}, val_v{0};
                for (auto it = begin; it != end; ++it)
                {
                    Quadlist quadratures;
                    quadrature.MapToQuadrature(*it, quadratures);
                    for (const auto &i : quadratures)
                    {
                        val_v += i.second * pow(_targetFunction(result.first->GetDomain()->AffineMap(i.first))[0], 2);
                        err_v += i.second * pow(_targetFunction(result.first->GetDomain()->AffineMap(i.first))[0] -
                                                    result.second->AffineMap(i.first)(0),
                                                2);
                    }
                }
                std::lock_guard<std::mutex> lock(mtx);
                val_norm += val_v;
                err_norm += err_v;
            };

            for (auto it = std::begin(threads); it != std::end(threads) - 1; ++it)
            {
                *it = std::thread(lambda, work_iter, work_iter + grainsize);
                work_iter += grainsize;
            }
            threads.back() = std::thread(lambda, work_iter, elements.end());
            for (auto &i : threads)
            {
                i.join();
            }
        }
        return sqrt(err_norm / val_norm);
    }

    T RelativeH2Error()
    {
        T err_norm{0};
        T val_norm{0};
        for (const auto &result : _results)
        {
            auto deg_x = result.first->GetDomain()->GetDegree(0);
            auto deg_y = result.first->GetDomain()->GetDegree(1);
            QuadratureRule<T> quadrature;
            quadrature.SetUpQuadrature(deg_x >= deg_y ? (deg_x + 1) : (deg_y + 1));
            CoordinatePairList elements;
            result.first->KnotSpansGetter(elements);
            int num_of_threads = 8;
            std::vector<std::thread> threads(num_of_threads);
            const int grainsize = elements.size() / num_of_threads;
            auto work_iter = elements.begin();
            std::mutex mtx;
            auto lambda = [&](typename CoordinatePairList::iterator begin,
                              typename CoordinatePairList::iterator end) -> void {
                T err_v{0}, val_v{0};
                for (auto it = begin; it != end; ++it)
                {
                    Quadlist quadratures;
                    quadrature.MapToQuadrature(*it, quadratures);
                    for (const auto &i : quadratures)
                    {
                        auto evals = result.first->GetDomain()->Eval2PhyDerAllTensor(i.first);
                        std::vector<T> approx(6, 0.0);
                        for (const auto &j : *evals)
                        {
                            approx[0] += result.second->CtrPtsGetter(j.first)(0) * j.second[0];
                            approx[1] += result.second->CtrPtsGetter(j.first)(0) * j.second[1];
                            approx[2] += result.second->CtrPtsGetter(j.first)(0) * j.second[2];
                            approx[3] += result.second->CtrPtsGetter(j.first)(0) * j.second[3];
                            approx[4] += result.second->CtrPtsGetter(j.first)(0) * j.second[4];
                            approx[5] += result.second->CtrPtsGetter(j.first)(0) * j.second[5];
                        }
                        auto exact = _targetFunction(result.first->GetDomain()->AffineMap(i.first));
                        val_v += i.second * (pow(exact[0], 2) + pow(exact[1], 2) + pow(exact[2], 2) + pow(exact[3], 2) + pow(exact[4], 2) + pow(exact[5], 2));
                        err_v += i.second * (pow(exact[0] - approx[0], 2) + pow(exact[1] - approx[1], 2) + pow(exact[2] - approx[2], 2) + pow(exact[3] - approx[3], 2) + pow(exact[4] - approx[4], 2) + pow(exact[5] - approx[5], 2));
                    }
                }
                std::lock_guard<std::mutex> lock(mtx);
                val_norm += val_v;
                err_norm += err_v;
            };

            for (auto it = std::begin(threads); it != std::end(threads) - 1; ++it)
            {
                *it = std::thread(lambda, work_iter, work_iter + grainsize);
                work_iter += grainsize;
            }
            threads.back() = std::thread(lambda, work_iter, elements.end());
            for (auto &i : threads)
            {
                i.join();
            }
        }
        return sqrt(err_norm / val_norm);
    }

  private:
    std::vector<ResultPtr> _results;
    Functor _targetFunction;
};

#endif //OO_IGA_POSTPROCESS_H
