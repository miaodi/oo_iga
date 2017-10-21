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

    MatrixData()
        : _rowIndices{std::make_unique<std::vector<int>>()},
          _colIndices{std::make_unique<std::vector<int>>()},
          _matrix{std::make_unique<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>()} {}
};

template<typename T>
struct VectorData
{
    std::unique_ptr<std::vector<int>> _rowIndices;
    std::unique_ptr<Eigen::Matrix<T, Eigen::Dynamic, 1>> _vector;

    VectorData()
        : _rowIndices{std::make_unique<std::vector<int>>()},
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

    DomainVisitor(const DofMapper<N, T> &dof_mapper)
        : _dofMapper(dof_mapper) {};

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
        Matrix tmp(weight_basis.size(), basis.size());
        tmp.setZero();
        for (int i = 0; i < quadrature_wegiht.size(); ++i)
        {
            tmp += weight_basis[i].transpose() * quadrature_wegiht[i] * basis[i];
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
        Vector tmp(weight_basis.size());
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

    void
    CondensedTripletVia(const std::map<int, int> &row_map,
                        const std::map<int, int> &col_map,
                        const std::vector<Eigen::Triplet<T>> &original_triplet,
                        std::vector<Eigen::Triplet<T>> &mapped_triplet) const
    {
        mapped_triplet.clear();
        for (const auto &i:original_triplet)
        {
            auto it_row = row_map.find(i.row());
            auto it_col = col_map.find(i.col());
            if (it_row != row_map.end() && it_col != col_map.end())
            {
                mapped_triplet.push_back(Eigen::Triplet<T>(it_row->second, it_col->second, i.value()));
            }
        }
    }

    void
    CondensedTripletVia(const std::map<int, int> &row_map,
                        const std::vector<Eigen::Triplet<T>> &original_triplet,
                        std::vector<Eigen::Triplet<T>> &mapped_triplet) const
    {
        mapped_triplet.clear();
        for (const auto &i:original_triplet)
        {
            auto it_row = row_map.find(i.row());
            if (it_row != row_map.end())
            {
                mapped_triplet.push_back(Eigen::Triplet<T>(it_row->second, 0, i.value()));
            }
        }
    }

    void
    LocalToGlobal(Element<d, N, T> *g, MatrixData<T> &indexed_matrix)
    {
        int start_index = _dofMapper.StartingIndex(g->GetDomain());
        std::transform(indexed_matrix._colIndices->cbegin(), indexed_matrix._colIndices->cend(),
                       indexed_matrix._colIndices->begin(), [&start_index](const int &i) { return i + start_index; });
        std::transform(indexed_matrix._rowIndices->cbegin(), indexed_matrix._rowIndices->cend(),
                       indexed_matrix._rowIndices->begin(), [&start_index](const int &i) { return i + start_index; });
    }

    void
    LocalToGlobal(Element<d, N, T> *g, VectorData<T> &indexed_matrix)
    {
        int start_index = _dofMapper.StartingIndex(g->GetDomain());
        std::transform(indexed_matrix._rowIndices->cbegin(), indexed_matrix._rowIndices->cend(),
                       indexed_matrix._rowIndices->begin(), [&start_index](const int &i) { return i + start_index; });
    }

//    Convert non-zero Symmetric MatrixData elements to Triplet
    void
    SymmetricTriplet(const MatrixData<T> &matrix, std::vector<Eigen::Triplet<T>> &triplet) const
    {
        ASSERT(matrix._rowIndices->size() == matrix._colIndices->size(),
               "Given matrix data does not fit to symmetric assembler.");
        for (int i = 0; i < matrix._rowIndices->size(); ++i)
        {
            for (int j = i; j < matrix._colIndices->size(); ++j)
            {
                T tmp{(*matrix._matrix)(i, j)};
                if (tmp != 0)
                {
                    triplet.emplace_back(Eigen::Triplet<T>((*matrix._rowIndices)[i], (*matrix._colIndices)[j], tmp));
                }
            }
        }
    }

    //    Convert non-zero MatrixData elements to Triplet
    void
    Triplet(const MatrixData<T> &matrix, std::vector<Eigen::Triplet<T>> &triplet) const
    {
        for (int i = 0; i < matrix._rowIndices->size(); ++i)
        {
            for (int j = 0; j < matrix._colIndices->size(); ++j)
            {
                T tmp{(*matrix._matrix)(i, j)};
                if (tmp != 0)
                {
                    triplet.emplace_back(Eigen::Triplet<T>((*matrix._rowIndices)[i], (*matrix._colIndices)[j], tmp));
                }
            }
        }
    }
//    Convert non-zero VectorData elements to Triplet
    void
    Triplet(const VectorData<T> &vector, std::vector<Eigen::Triplet<T>> &triplet) const
    {
        for (int i = 0; i < vector._rowIndices->size(); ++i)
        {
            T tmp{(*vector._vector)(i)};
            if (tmp != 0)
            {
                triplet.emplace_back(Eigen::Triplet<T>((*vector._rowIndices)[i], 0, tmp));
            }
        }
    }

    bool
    IndexModifier(const std::map<int, int> &index_map, int &index) const
    {
        auto it = index_map.find(index);
        if (it != index_map.end())
        {
            index = it->second;
            return true;
        }
        return false;
    }

    void
    MatrixDataIndexModifier(const std::map<int, int> &index_map, MatrixData<T> &matrix_data)
    {
//        Row operation
        for (auto it = matrix_data._rowIndices->begin(); it != matrix_data._rowIndices->end();)
        {
            if (!IndexModifier(index_map, *it))
            {
                it = matrix_data._rowIndices->erase(it);
                int row_num = it - matrix_data._rowIndices->begin();
                Accessory::removeRow(*matrix_data._matrix, row_num);
            }
            else
            {
                ++it;
            }
        }
//        Column operation
        for (auto it = matrix_data._colIndices->begin(); it != matrix_data._colIndices->end();)
        {
            if (!IndexModifier(index_map, *it))
            {
                it = matrix_data._colIndices->erase(it);
                int col_num = it - matrix_data._colIndices->begin();
                Accessory::removeColumn(*matrix_data._matrix, col_num);
            }
            else
            {
                ++it;
            }
        }
    }

    void
    VectorDataIndexModifier(const std::map<int, int> &index_map, VectorData<T> &vector_data)
    {
//        Row operation
        for (auto it = vector_data._rowIndices->begin(); it != vector_data._rowIndices->end();)
        {
            if (!IndexModifier(index_map, *it))
            {
                it = vector_data._rowIndices->erase(it);
                int row_num = it - vector_data._rowIndices->begin();
                Accessory::removeRow(*vector_data._vector, row_num);
            }
            else
            {
                ++it;
            }
        }
    }

    void
    MatrixAssembler(const int &row_dof,
                    const int &col_dof,
                    const std::vector<Eigen::Triplet<T>> &triplet,
                    Eigen::SparseMatrix<T> &matrix) const
    {
        matrix.resize(row_dof, col_dof);
        matrix.setFromTriplets(triplet.cbegin(), triplet.cend());
    }

    void
    VectorAssembler(const int &row_dof,
                    const std::vector<Eigen::Triplet<T>> &triplet,
                    Eigen::SparseMatrix<T> &vector) const
    {
        vector.resize(row_dof, 1);
        vector.setFromTriplets(triplet.cbegin(), triplet.cend());
    }

    Matrix
    Solve(const Eigen::SparseMatrix<T> &gramian, const Eigen::SparseMatrix<T> &rhs) const
    {
        Eigen::ConjugateGradient<Eigen::SparseMatrix<T>, Eigen::Lower | Eigen::Upper> cg;
        cg.compute(gramian);
        Matrix res = cg.solve(rhs);
        return res;
    }

protected:
    const DofMapper<N, T> &_dofMapper;
};
