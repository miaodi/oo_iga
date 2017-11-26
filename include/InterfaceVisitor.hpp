//
// Created by miaodi on 23/10/2017.
//

#pragma once

#include "DomainVisitor.hpp"
#include "Edge.hpp"
#include "Utility.hpp"
#include <functional>

template <int N, typename T>
class InterfaceVisitor : public DomainVisitor<1, N, T>
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
    using DomainShared_ptr = typename std::shared_ptr<PhyTensorBsplineBasis<2, N, T>>;
    using ConstraintIntegralElementAssembler = std::function<void(Matrix &slave_constraint_basis,
                                                                  std::vector<int> &slave_constraint_basis_indices,
                                                                  Matrix &master_constrint_basis,
                                                                  std::vector<int> &master_constraint_basis_indices,
                                                                  Matrix &multiplier_basis,
                                                                  std::vector<int> &multiplier_basis_indices,
                                                                  T &integral_weight,
                                                                  Edge<N, T> *edge,
                                                                  const Quadrature &u)>;

  public:
    InterfaceVisitor(const DofMapper<N, T> &dof_mapper)
        : DomainVisitor<1, N, T>(dof_mapper) {}

    //    visit if given topology is matched and is slave.
    void
        Visit(Element<1, N, T> *g);

  protected:
    // Initialize quadrature rule by the highest polynomial order of coupled domains
    void
        InitializeQuadratureRule(Element<1, N, T> *g,
                                 QuadratureRule<T> &quad_rule);

    // Initialize knot spans such that new knot span is a union of two coupling knot vectors
    void
        InitializeKnotSpans(Element<1, N, T> *g,
                            KnotSpanlist &knot_spans);

    // Set all intermediate triplet containers (C^0 constaints' lhs and rhs) to empty,
    virtual void
    InitializeTripletContainer() = 0;

    virtual void
    SolveConstraint(Edge<N, T> *) = 0;

    void
        ConstraintLocalAssemble(Element<1, N, T> *,
                                const QuadratureRule<T> &,
                                const KnotSpan &,
                                ConstraintIntegralElementAssembler,
                                std::vector<Eigen::Triplet<T>> &);

    // Move master d.o.f from slave domain to master domain.
    void MoveToRhs(const DomainShared_ptr &,
                   std::vector<Eigen::Triplet<T>> &);
};

template <int N, typename T>
void
    InterfaceVisitor<N, T>::Visit(Element<1, N, T> *g)
{
    auto edge = dynamic_cast<Edge<N, T> *>(g);
    if (edge->IsMatched() && !edge->IsSlave())
    {
        InitializeTripletContainer();
        edge->GetDomain()->BezierDualInitialize();
        DomainVisitor<1, N, T>::Visit(g);
        SolveConstraint(edge);
    }
}

template <int N, typename T>
void
    InterfaceVisitor<N, T>::InitializeQuadratureRule(Element<1, N, T> *g,
                                                     QuadratureRule<T> &quad_rule)
{
    auto edge = dynamic_cast<Edge<N, T> *>(g);

    auto slave_domain = edge->Parent(0).lock()->GetDomain();
    auto master_domain = edge->Counterpart().lock()->Parent(0).lock()->GetDomain();
    quad_rule.SetUpQuadrature(std::max(slave_domain->MaxDegree(),
                                       master_domain->MaxDegree()) +
                              1);
}

template <int N, typename T>
void
    InterfaceVisitor<N, T>::InitializeKnotSpans(Element<1, N, T> *g,
                                                KnotSpanlist &knot_spans)
{
    auto edge = dynamic_cast<Edge<N, T> *>(g);
    auto slave_edge_knot_vector = edge->GetDomain()->KnotVectorGetter(0);
    auto master_edge_knot_vector_uni = edge->Counterpart().lock()->GetDomain()->KnotVectorGetter(0).GetUnique();
    Knot from(1), to(1);
    for (int i = 1; i < master_edge_knot_vector_uni.size() - 1; ++i)
    {
        from(0) = master_edge_knot_vector_uni[i];
        if (!Accessory::MapParametricPoint(&*edge->Counterpart().lock()->GetDomain(),
                                           from,
                                           &*edge->GetDomain(),
                                           to))
        {
            std::cout << " Mapping from master edge to slave edge failed. " << std::endl;
        }
        slave_edge_knot_vector.Insert(to(0));
    }

    // clean noise
    slave_edge_knot_vector.Uniquify();
    knot_spans = slave_edge_knot_vector.KnotEigenSpans();
}

template <int N, typename T>
void
    InterfaceVisitor<N, T>::ConstraintLocalAssemble(Element<1, N, T> *g,
                                                    const QuadratureRule<T> &quadrature_rule,
                                                    const KnotSpan &knot_span,
                                                    ConstraintIntegralElementAssembler IntegralElementAssembler,
                                                    std::vector<Eigen::Triplet<T>> &constraintsEquationElements)
{
    auto edge = dynamic_cast<Edge<N, T> *>(g);
    QuadList edge_quadrature_points;
    quadrature_rule.MapToQuadrature(knot_span,
                                    edge_quadrature_points);
    auto num_of_quadrature = edge_quadrature_points.size();
    std::vector<Matrix> slave_constraint_basis(num_of_quadrature), master_constraint_basis(num_of_quadrature),
        multiplier_basis(num_of_quadrature);
    std::vector<int> slave_constraint_basis_indices, master_constraint_basis_indices, multiplier_basis_indices;
    std::vector<T> weights(num_of_quadrature);
    for (int i = 0; i < num_of_quadrature; ++i)
    {
        IntegralElementAssembler(slave_constraint_basis[i],
                                 slave_constraint_basis_indices,
                                 master_constraint_basis[i],
                                 master_constraint_basis_indices,
                                 multiplier_basis[i],
                                 multiplier_basis_indices,
                                 weights[i],
                                 edge,
                                 edge_quadrature_points[i]);
    }
    auto multiplier_basis_indices_copy{multiplier_basis_indices};
    auto stiff = this->LocalStiffness(multiplier_basis,
                                      multiplier_basis_indices,
                                      slave_constraint_basis,
                                      slave_constraint_basis_indices,
                                      weights);
    auto load = this->LocalStiffness(multiplier_basis,
                                     multiplier_basis_indices_copy,
                                     master_constraint_basis,
                                     master_constraint_basis_indices,
                                     weights);

    std::lock_guard<std::mutex> lock(this->_mutex);
    this->Triplet(stiff, constraintsEquationElements);
    this->Triplet(load, constraintsEquationElements);
}

template <int N, typename T>
void InterfaceVisitor<N, T>::MoveToRhs(const DomainShared_ptr &domain,
                                       std::vector<Eigen::Triplet<T>> &triplets)
{
    for (auto &i : triplets)
    {
        if (!this->_dofMapper.IsInDomain(domain, i.col()))
        {
            Eigen::Triplet<T> tmp(i.row(), i.col(), -i.value());
            i = tmp;
        }
    }
}
