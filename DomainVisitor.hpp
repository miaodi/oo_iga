//
// Created by di miao on 10/17/17.
//

#pragma once

#include "Visitor.hpp"
#include "Topology.hpp"
#include "QuadratureRule.h"
#include <thread>
#include <mutex>

template<typename T>
struct MatrixData
{
    std::unique_ptr<std::vector<int>> _rowIndices;
    std::unique_ptr<std::vector<int>> _colIndices;
    std::unique_ptr<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> _matrix;
    MatrixData() : _rowIndices{std::make_unique<std::vector<int>>()},
                   _colIndices{std::make_unique<std::vector<int>>()},
                   _matrix{std::make_unique<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>()} {}
};

template<typename T>
struct VectorData
{
    std::unique_ptr<std::vector<int>> _rowIndices;
    std::unique_ptr<Eigen::Matrix<T, Eigen::Dynamic, 1>> _vector;
    VectorData() : _rowIndices{std::make_unique<std::vector<int>>()},
                   _vector{std::make_unique<Eigen::Matrix<T, Eigen::Dynamic, 1>>()} {}
};

template<int d, int N, typename T>
class DomainVisitor : public Visitor<d, N, T>
{
public:
    using Knot = typename QuadratureRule<T>::Coordinate;
    using Quadrature = typename QuadratureRule<T>::Quadrature;
    using QuadList = typename QuadratureRule<T>::QuadList;
    using KnotSpan = std::pair<Knot, Knot>;
    using KnotSpanlist  = std::vector<KnotSpan>;
    using LoadFunctor = std::function<std::vector<T>(const Knot &)>;
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    DomainVisitor() {};

//    Multi thread domain visitor
    void
    Visit(Element<d, N, T> *g)
    {
        QuadratureRule<T> quad_rule;
        KnotSpanlist knot_spans;
        InitializeQuadratureRule(g, quad_rule);
        InitializeKnotSpans(g, knot_spans);
        std::mutex pmutex;
        auto n = std::thread::hardware_concurrency();
        std::vector<std::thread> threads(n);
        const int grainsize = knot_spans.size() / n;
        auto work_iter = knot_spans.begin();
        auto
            lambda = [&](typename KnotSpanlist::iterator begin, typename KnotSpanlist::iterator end) -> void
        {
            for (auto i = begin; i != end; ++i)
            {
                LocalAssemble(g, quad_rule, *i, pmutex);
            }
        };
        for (auto it = std::begin(threads); it != std::end(threads) - 1; ++it)
        {
            *it = std::thread(lambda, work_iter, work_iter + grainsize);
            work_iter += grainsize;
        }
        threads.back() = std::thread(lambda, work_iter, knot_spans.end());
        for (auto &i:threads)
        {
            i.join();
        }
    }

//    Initialize quadrature rule
    virtual void
    InitializeQuadratureRule(Element<d, N, T> *g, QuadratureRule<T> &quad_rule)
    {
        if (d == 0)
        {
            quad_rule.SetUpQuadrature(1);
        }
        else
        {
            auto domain = g->GetDomain();
            int max_degree = 0;
            for (int i = 0; i < d; i++)
            {
                max_degree = std::max(max_degree, domain->GetDegree(i));
            }
            quad_rule.SetUpQuadrature(max_degree + 1);
        }
    }

//    Initialize knot spans
    virtual void
    InitializeKnotSpans(Element<d, N, T> *g, KnotSpanlist &knot_spans)
    {
        g->GetDomain()->KnotSpanGetter(knot_spans);
    }

//    Pure virtual method local assemble algorithm is needed to be implemented here
    virtual void
    LocalAssemble(Element<d, N, T> *, const QuadratureRule<T> &, const KnotSpan &, std::mutex &) = 0;

    virtual MatrixData<T>
    LocalStiffness(const std::vector<Matrix> &weight_basis,
                   std::vector<int> &weight_basis_indices,
                   const std::vector<Matrix> &basis,
                   std::vector<int> &basis_indices,
                   const std::vector<T> &quadrature_wegiht)
    {
        Matrix tmp;
        tmp.setZero();
        std::cout<<tmp<<std::endl;

        for (int i = 0; i < quadrature_wegiht.size(); ++i)
        {
            std::cout<< weight_basis[i].transpose() * quadrature_wegiht[i] * basis[i]<<std::endl;
        }

        MatrixData<T> res;
        *(res._rowIndices) = std::move(weight_basis_indices);
        *(res._colIndices) = std::move(basis_indices);
        *(res._matrix) = std::move(tmp);
        return res;
    }

    virtual VectorData<T>
    LocalRhs(const std::vector<Matrix> &weight_basis,
             const std::vector<int> &weight_basis_indices,
             const std::vector<Matrix> &function_value,
             const std::vector<T> &quadrature_wegiht)
    {
        Vector tmp;
        tmp.setZero();
        for (int i = 0; i < quadrature_wegiht.size(); ++i)
        {
            tmp += weight_basis[i].transpose() * quadrature_wegiht[i] * function_value[i];
        }
        VectorData<T> res;
        *(res._rowIndices) = std::move(weight_basis_indices);
        *(res._vector) = std::move(tmp);
        return res;
    }
};
