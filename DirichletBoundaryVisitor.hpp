//
// Created by miaodi on 19/10/2017.
//

#pragma once

#include "DomainVisitor.hpp"
#include "Edge.hpp"

template<int N, typename T>
class DofMapper;

template<int N, typename T>
class DirichletBoundaryVisitor : public DomainVisitor<1, N, T>
{
public:
    using Knot = typename DomainVisitor<1, N, T>::Knot;
    using Quadrature = typename DomainVisitor<1, N, T>::Quadrature;
    using QuadList = typename DomainVisitor<1, N, T>::QuadList;
    using KnotSpan = typename DomainVisitor<1, N, T>::KnotSpan;
    using KnotSpanlist = typename DomainVisitor<1, N, T>::KnotSpanlist;
    using LoadFunctor = typename DomainVisitor<1, N, T>::LoadFunctor;
    using Matrix = typename DomainVisitor<1, N, T>::Matrix;
    using Vector = typename DomainVisitor<1, N, T>::Vector;

public:
    DirichletBoundaryVisitor(const DofMapper<N, T> &dof_mapper, const LoadFunctor &body_force)
        : DomainVisitor<2, N, T>(dof_mapper), _DirichletFunctor(body_force) {}

    void
    SolveDirichletBoundary() const;

    void
    Visit(Element<1, N, T> *);

    void
    LocalAssemble(Element<1, N, T> *, const QuadratureRule<T> &, const KnotSpan &, std::mutex &);

    void
    DirichletBoundary(Eigen::SparseMatrix<T> &dirichlet_boundary) const;

    void
    CondensedDirichletBoundary(Eigen::SparseMatrix<T> &dirichlet_boundary) const;

protected:
    std::vector<Eigen::Triplet<T>> _gramian;
    std::vector<Eigen::Triplet<T>> _rhs;
    std::vector<Eigen::Triplet<T>> _dirichlet;
    const LoadFunctor &_DirichletFunctor;
};

template<int N, typename T>
void
DirichletBoundaryVisitor<N, T>::SolveDirichletBoundary() const
{
    std::vector<Eigen::Triplet<T>> condensed_gramian;
    std::vector<Eigen::Triplet<T>> condensed_rhs;
    auto dirichlet_map = this->_dofMapper.GlobalDirichletCondensedMap();
    auto dirichlet_indices = this->_dofMapper.GlobalDirichletIndices();
    this->CondensedTripletVia(dirichlet_map, dirichlet_map, _gramian, condensed_gramian);
    this->CondensedTripletVia(dirichlet_map, _rhs, condensed_rhs);
    Eigen::SparseMatrix<T> gramian_matrix_triangle, rhs_vector, gramian_matrix;
    this->MatrixAssembler(dirichlet_map.size(), dirichlet_map.size(), _gramian, gramian_matrix_triangle);
    this->VectorAssembler(dirichlet_map.size(), _rhs, rhs_vector);
    gramian_matrix = gramian_matrix_triangle.selfadjointView<Eigen::Upper>();
    Vector res = this->Solve(gramian_matrix, rhs_vector);
    for (int i = 0; i < res.rows(); ++i)
    {
        _dirichlet.push_back(Eigen::Triplet<T>(dirichlet_indices[i], 0, res(i)));
    }
}

template<int N, typename T>
void
DirichletBoundaryVisitor<N, T>::DirichletBoundary(Eigen::SparseMatrix<T> &dirichlet_boundary) const
{
    this->VectorAssembler(_dofMapper.Dof(), _dirichlet, dirichlet_boundary);
}

template<int N, typename T>
void
DirichletBoundaryVisitor<N, T>::CondensedDirichletBoundary(Eigen::SparseMatrix<T> &dirichlet_boundary) const
{
    std::vector<Eigen::Triplet<T>> condensed_dirichlet;
    for (const auto &i : _dirichlet)
    {
        int global_dirichlet_index = i.row();
        if (this->_dofMapper.GlobalToCondensedIndex(global_dirichlet_index))
        {
            condensed_dirichlet.push_back(Eigen::Triplet<T>(global_dirichlet_index, 0, i.value()));
        }
        else
        {
            std::cout << "error happens when creates condensed richlet boundary" << std::endl;
        }
    }
    this->VectorAssembler(_dofMapper.CondensedDof(), condensed_dirichlet, dirichlet_boundary);
}

template<int N, typename T>
void
DirichletBoundaryVisitor<N, T>::Visit(Element<1, N, T> *g)
{
    auto edge = dynamic_cast<Edge<N, T> *>(g);
    if (edge->IsDirichlet())
    {
        DomainVisitor<1, N, T>::Visit(g);
    }
}

template<int N, typename T>
void
DirichletBoundaryVisitor<N, T>::LocalAssemble(Element<1, N, T> *g,
                                              const QuadratureRule<T> &quadrature_rule,
                                              const DirichletBoundaryVisitor<N, T>::KnotSpan &knot_span,
                                              std::mutex &pmutex)
{
    auto domain = g->GetDomain();
    QuadList quadrature_points;
    quadrature_rule.MapToQuadrature(knot_span, quadrature_points);

    auto index = domain->ActiveIndex(quadrature_points[0].first);
    this->_dofMapper.IndicesToGlobal(domain, index);
    auto num_of_quadrature = quadrature_points.size();

    std::vector<int> bilinear_form_test_indices{index}, bilinear_form_trial_indices{index}, linear_form_test_indices{
        index};
    std::vector<Matrix> bilinear_form_test(num_of_quadrature), bilinear_form_trial(
        num_of_quadrature), linear_form_test(num_of_quadrature), linear_form_value(num_of_quadrature);
    std::vector<T> weights;
    for (int i = 0; i<quadrature_points.size(); ++i)
    {
        weights.push_back(quadrature_points[i].second*domain->Jacobian(quadrature_points[i].first));
        IntegralElementAssembler(bilinear_form_test[i], bilinear_form_trial[i], linear_form_test[i],
                                 linear_form_value[i], domain, quadrature_points[i].first);
    }
}
