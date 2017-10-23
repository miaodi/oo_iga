//
// Created by miaodi on 19/10/2017.
//

#pragma once

#include "DomainVisitor.hpp"
#include "Edge.hpp"

namespace Accessory {

    template<int N, int d_from, int d_to, typename T>
    bool MapParametricPoint(const PhyTensorBsplineBasis<d_from, N, T>* const from_domain,
            const Eigen::Matrix<T, Eigen::Dynamic, 1>& from_point,
            const PhyTensorBsplineBasis<d_to, N, T>* const to_domain, Eigen::Matrix<T, Eigen::Dynamic, 1>& to_point)
    {
        ASSERT(from_domain->InDomain(from_point), "The point about to be mapped is out of the domain.");
        Eigen::Matrix<T, N, 1> physical_point = from_domain->AffineMap(from_point);
        return to_domain->InversePts(physical_point, to_point);
    };
}

template<int N, typename T>
class DofMapper;

template<int N, typename T>
class DirichletBoundaryVisitor : public DomainVisitor<1, N, T> {
public:
    using Knot = typename DomainVisitor<1, N, T>::Knot;
    using Quadrature = typename DomainVisitor<1, N, T>::Quadrature;
    using QuadList = typename DomainVisitor<1, N, T>::QuadList;
    using KnotSpan = typename DomainVisitor<1, N, T>::KnotSpan;
    using KnotSpanlist = typename DomainVisitor<1, N, T>::KnotSpanlist;
    using LoadFunctor = typename DomainVisitor<1, N, T>::LoadFunctor;
    using Matrix = typename DomainVisitor<1, N, T>::Matrix;
    using Vector = typename DomainVisitor<1, N, T>::Vector;
    using DomainShared_ptr = typename std::shared_ptr<PhyTensorBsplineBasis<2, N, T>>;

public:
    DirichletBoundaryVisitor(const DofMapper<N, T>& dof_mapper, const LoadFunctor& boundary_value)
            :DomainVisitor<1, N, T>(dof_mapper), _dirichletFunctor(boundary_value) { }

    void
    Visit(Element<1, N, T>*);

    void
    DirichletBoundary(Eigen::SparseMatrix<T>& dirichlet_boundary) const;

    void
    CondensedDirichletBoundary(Eigen::SparseMatrix<T>& dirichlet_boundary) const;

protected:
    void
    SolveDirichletBoundary() const;

    virtual void
    IntegralElementAssembler(Matrix& bilinear_form_trail, Matrix& bilinear_form_test, Matrix& linear_form_value,
            Matrix& linear_form_test, const DomainShared_ptr domain, const Knot& u) const = 0;

    void
    LocalAssemble(Element<1, N, T>*, const QuadratureRule<T>&, const KnotSpan&, std::mutex&);

protected:
    std::vector<Eigen::Triplet<T>> _gramian;
    std::vector<Eigen::Triplet<T>> _rhs;
    mutable std::vector<Eigen::Triplet<T>> _dirichlet;
    const LoadFunctor& _dirichletFunctor;
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
    this->MatrixAssembler(dirichlet_map.size(), dirichlet_map.size(), condensed_gramian, gramian_matrix_triangle);
    this->VectorAssembler(dirichlet_map.size(), condensed_rhs, rhs_vector);
    gramian_matrix = gramian_matrix_triangle.template selfadjointView<Eigen::Upper>();
    Vector res = this->Solve(gramian_matrix, rhs_vector);
    for (int i = 0; i<res.rows(); ++i)
    {
        _dirichlet.push_back(Eigen::Triplet<T>(dirichlet_indices[i], 0, res(i)));
    }
}

template<int N, typename T>
void
DirichletBoundaryVisitor<N, T>::DirichletBoundary(Eigen::SparseMatrix<T>& dirichlet_boundary) const
{
    if (_dirichlet.size()==0)
    {
        SolveDirichletBoundary();
    }
    this->VectorAssembler(this->_dofMapper.Dof(), _dirichlet, dirichlet_boundary);
}

template<int N, typename T>
void
DirichletBoundaryVisitor<N, T>::CondensedDirichletBoundary(Eigen::SparseMatrix<T>& dirichlet_boundary) const
{
    if (_dirichlet.size()==0)
    {
        SolveDirichletBoundary();
    }
    std::vector<Eigen::Triplet<T>> condensed_dirichlet;
    for (const auto& i : _dirichlet)
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
    this->VectorAssembler(this->_dofMapper.CondensedDof(), condensed_dirichlet, dirichlet_boundary);
}

template<int N, typename T>
void
DirichletBoundaryVisitor<N, T>::Visit(Element<1, N, T>* g)
{
    auto edge = dynamic_cast<Edge<N, T>*>(g);
    if (edge->IsDirichlet())
    {
        DomainVisitor<1, N, T>::Visit(g);
    }
}

template<int N, typename T>
void
DirichletBoundaryVisitor<N, T>::LocalAssemble(Element<1, N, T>* g,
        const QuadratureRule<T>& quadrature_rule,
        const DirichletBoundaryVisitor<N, T>::KnotSpan& knot_span,
        std::mutex& pmutex)
{

    auto edge = dynamic_cast<Edge<N, T>*>(g);
    auto test_domain = edge->GetDomain();
    auto trial_domain = (edge->Parent(0).lock())->GetDomain();
    QuadList test_quadrature_points;
    quadrature_rule.MapToQuadrature(knot_span, test_quadrature_points);

    std::vector<int> index;
    this->_dofMapper.IndicesToGlobal(trial_domain, index);
    auto num_of_quadrature = test_quadrature_points.size();

    std::vector<Matrix> bilinear_form_test(num_of_quadrature), bilinear_form_trial(
            num_of_quadrature), linear_form_test(num_of_quadrature), linear_form_value(num_of_quadrature);
    std::vector<T> weights;
    for (int i = 0; i<num_of_quadrature; ++i)
    {
        weights.push_back(test_quadrature_points[i].second*test_domain->Jacobian(test_quadrature_points[i].first));
        Vector trial_quadrature_abscissa;
        if (!Accessory::MapParametricPoint(&*test_domain, test_quadrature_points[i].first, &*trial_domain,
                trial_quadrature_abscissa))
        {
            std::cout << "MapParametericPoint failed" << std::endl;
        }
        IntegralElementAssembler(bilinear_form_trial[i], bilinear_form_test[i], linear_form_value[i],
                linear_form_test[i], trial_domain, trial_quadrature_abscissa);
        if (index.size()==0)
        {
            index = trial_domain->ActiveIndex(trial_quadrature_abscissa);
        }
    }
    this->_dofMapper.IndicesToGlobal(trial_domain, index);
    std::vector<int> bilinear_form_test_indices{index}, bilinear_form_trial_indices{index}, linear_form_test_indices{
            index};
    auto stiff = this->LocalStiffness(bilinear_form_test, bilinear_form_test_indices, bilinear_form_trial,
            bilinear_form_trial_indices, weights);
    auto load = this->LocalRhs(linear_form_test, linear_form_test_indices, linear_form_value, weights);
    std::lock_guard<std::mutex> lock(pmutex);
    this->SymmetricTriplet(stiff, _gramian);
    this->Triplet(load, _rhs);
}
