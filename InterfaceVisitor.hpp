//
// Created by miaodi on 23/10/2017.
//

#pragma once

#include "DomainVisitor.hpp"
#include "Edge.hpp"
#include "Utility.hpp"

template<int N, typename T>
class InterfaceVisitor : public DomainVisitor<1, N, T> {
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
    InterfaceVisitor(const DofMapper<N, T>& dof_mapper)
            :DomainVisitor<1, N, T>(dof_mapper) { }

//    visit if given topology is matched and is slave.
    void Visit(Element<1, N, T>* g);

protected:
    // Initialize quadrature rule by the highest polynomial order of coupled domains
    void
    InitializeQuadratureRule(Element<1, N, T>* g, QuadratureRule<T>& quad_rule);

    // Initialize knot spans such that new knot span is a union of two coupling knot vectors
    void
    InitializeKnotSpans(Element<1, N, T>* g, KnotSpanlist& knot_spans);

    void
    LocalAssemble(Element<1, N, T>*, const QuadratureRule<T>&, const KnotSpan&, std::mutex&);
};

template<int N, typename T>
void InterfaceVisitor<N, T>::Visit(Element<1, N, T>* g)
{
    auto edge = dynamic_cast<Edge<N, T>*>(g);
    if (edge->IsMatched() && edge->IsSlave())
    {
        DomainVisitor<1, N, T>::Visit(g);
    }
}

template<int N, typename T>
void InterfaceVisitor<N, T>::InitializeQuadratureRule(Element<1, N, T>* g, QuadratureRule<T>& quad_rule)
{
    auto edge = dynamic_cast<Edge<N, T>*>(g);
    auto slave_domain = edge->Parent(0).lock()->GetDomain();
    auto master_domain = edge->Counterpart().lock()->Parent(0).lock()->GetDomain();
    quad_rule.SetUpQuadrature(std::max(slave_domain->MaxDegree(), master_domain->MaxDegree())+1);
}

template<int N, typename T>
void InterfaceVisitor<N, T>::InitializeKnotSpans(Element<1, N, T>* g, KnotSpanlist& knot_spans)
{
    auto edge = dynamic_cast<Edge<N, T>*>(g);
    auto slave_edge_knot_vector = edge->GetDomain()->KnotVectorGetter(0);
    auto master_edge_knot_vector_uni = edge->Counterpart().lock()->GetDomain()->KnotVectorGetter(0).GetUnique();
    Knot from(1), to(1);
    for (int i = 1; i<master_edge_knot_vector_uni.size()-1; ++i)
    {
        from(0) = master_edge_knot_vector_uni[i];
        if (!Accessory::MapParametricPoint(&*edge->Counterpart().lock()->GetDomain(), from, &*edge->GetDomain(), to))
        {
            std::cout << " Mapping from master edge to slave edge failed. " << std::endl;
        }
        slave_edge_knot_vector.Insert(to(0));
    }
    knot_spans = slave_edge_knot_vector.KnotEigenSpans();
//    for(auto i:knot_spans){
//        std::cout<<i.first.transpose()<<", "<<i.second.transpose()<<std::endl;
//    }
}

template<int N, typename T>
void InterfaceVisitor<N, T>::LocalAssemble(Element<1, N, T>* g, const QuadratureRule<T>& quadrature_rule,
        const KnotSpan& knot_span, std::mutex& pmutex)
{

    auto edge = dynamic_cast<Edge<N, T>*>(g);
    QuadList edge_quadrature_points;
    quadrature_rule.MapToQuadrature(knot_span, edge_quadrature_points);
    auto num_of_quadrature = edge_quadrature_points.size();
    std::vector<Matrix> bilinear_form_test(num_of_quadrature), bilinear_form_trial(num_of_quadrature),
            linear_form_test(num_of_quadrature), linear_form_value(num_of_quadrature);
    std::vector<int> bilinear_form_test_indices, bilinear_form_trial_indices, linear_form_test_indices;
    std::vector<T> weights(num_of_quadrature);
    for (int i = 0; i<num_of_quadrature; ++i)
    {
        IntegralElementAssembler(bilinear_form_trial[i],
                bilinear_form_trial_indices,
                bilinear_form_test[i],
                bilinear_form_test_indices,
                linear_form_value[i],
                linear_form_test[i],
                linear_form_test_indices,
                weights[i],
                edge,
                edge_quadrature_points[i]);
    }

    auto stiff = this->LocalStiffness(bilinear_form_test, bilinear_form_test_indices, bilinear_form_trial,
            bilinear_form_trial_indices, weights);
    auto load = this->LocalRhs(linear_form_test, linear_form_test_indices, linear_form_value, weights);
    std::lock_guard<std::mutex> lock(pmutex);
    this->SymmetricTriplet(stiff, _gramian);
    this->Triplet(load, _rhs);
}