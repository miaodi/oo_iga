//
// Created by miaodi on 16/11/2017.
//

#pragma once

#include "InterfaceVisitor.hpp"

template <int N, typename T>
class BiharmonicInterfaceH1 : public InterfaceVisitor<N, T>
{
  public:
    using Knot = typename InterfaceVisitor<N, T>::Knot;
    using Quadrature = typename InterfaceVisitor<N, T>::Quadrature;
    using QuadList = typename InterfaceVisitor<N, T>::QuadList;
    using KnotSpan = typename InterfaceVisitor<N, T>::KnotSpan;
    using KnotSpanlist = typename InterfaceVisitor<N, T>::KnotSpanlist;
    using LoadFunctor = typename InterfaceVisitor<N, T>::LoadFunctor;
    using Matrix = typename InterfaceVisitor<N, T>::Matrix;
    using Vector = typename InterfaceVisitor<N, T>::Vector;
    using DomainShared_ptr = typename InterfaceVisitor<N, T>::DomainShared_ptr;
    using ConstraintIntegralElementAssembler = typename InterfaceVisitor<N, T>::ConstraintIntegralElementAssembler;

  public:
    BiharmonicInterfaceH1(const DofMapper<N, T> &dof_mapper)
        : InterfaceVisitor<N, T>(dof_mapper) {}

    void
    ConstraintMatrix(Eigen::SparseMatrix<T> &);

  protected:
    void
    InitializeTripletContainer();

    void SolveConstraint(Edge<N, T> *);

    void
        LocalAssemble(Element<1, N, T> *,
                      const QuadratureRule<T> &,
                      const KnotSpan &);

    void
    H1IntegralElementAssembler(Matrix &slave_constraint_basis,
                               std::vector<int> &slave_constraint_basis_indices,
                               Matrix &master_constrint_basis,
                               std::vector<int> &master_constraint_basis_indices,
                               Matrix &multiplier_basis,
                               std::vector<int> &multiplier_basis_indices,
                               T &integral_weight,
                               Edge<N, T> *edge,
                               const Quadrature &u);

  protected:
    std::vector<Eigen::Triplet<T>> _constraintsEquationElements;
    std::map<Edge<N, T> *, std::vector<Eigen::Triplet<T>>> _constraint;
};

template <int N, typename T>
void BiharmonicInterfaceH1<N, T>::InitializeTripletContainer()
{
    _constraintsEquationElements.clear();
}

template <int N, typename T>
void BiharmonicInterfaceH1<N, T>::SolveConstraint(Edge<N, T> *edge)
{
    // iterate across the constraint equation container and obtain activated global indices and lagrange multiplier indices
    std::vector<int> activated_indices = Accessory::ColIndicesVector(_constraintsEquationElements);
    std::vector<int> multiplier_indices = Accessory::RowIndicesVector(_constraintsEquationElements);

    // get all the slave indices on the edge
    std::vector<int> slave_indices = *(edge->ExclusiveIndices(1));
    this->_dofMapper.IndicesToGlobal(edge->Parent(0).lock()->GetDomain(), slave_indices);

    // find slave indices and master indices
    std::vector<int> activated_slave_indices, activated_master_indices;
    std::set_intersection(activated_indices.begin(), activated_indices.end(), slave_indices.begin(),
                          slave_indices.end(), std::back_inserter(activated_slave_indices));
    std::set_difference(activated_indices.begin(), activated_indices.end(), activated_slave_indices.begin(),
                        activated_slave_indices.end(), std::back_inserter(activated_master_indices));

    ASSERT(multiplier_indices.size() > 0, "Lagrange multiplier size is zero, needs more refines.\n");
    auto activated_slave_indices_inverse_map = Accessory::IndicesInverseMap(activated_slave_indices);
    auto activated_master_indices_inverse_map = Accessory::IndicesInverseMap(activated_master_indices);
    auto multiplier_indices_inverse_map = Accessory::IndicesInverseMap(multiplier_indices);
    std::vector<Eigen::Triplet<T>> condensed_gramian, condensed_rhs;
    this->CondensedTripletVia(multiplier_indices_inverse_map, activated_slave_indices_inverse_map,
                              _constraintsEquationElements, condensed_gramian);
    this->MoveToRhs(edge->Counterpart().lock()->Parent(0).lock()->GetDomain(), _constraintsEquationElements);
    this->CondensedTripletVia(multiplier_indices_inverse_map, activated_master_indices_inverse_map,
                              _constraintsEquationElements, condensed_rhs);

    Matrix gramian_matrix, rhs_matrix;
    this->MatrixAssembler(multiplier_indices_inverse_map.size(), activated_slave_indices_inverse_map.size(),
                          condensed_gramian, gramian_matrix);
    this->MatrixAssembler(multiplier_indices_inverse_map.size(), activated_master_indices_inverse_map.size(),
                          condensed_rhs, rhs_matrix);
    Matrix constraint = this->SolveNonSymmetric(gramian_matrix, rhs_matrix);
    MatrixData<T> constraint_data(constraint, activated_slave_indices, activated_master_indices);
    std::vector<Eigen::Triplet<T>> temp;
    this->Triplet(constraint_data, temp);
    _constraint[edge] = std::move(temp);
}

template <int N, typename T>
void
    BiharmonicInterfaceH1<N, T>::LocalAssemble(Element<1, N, T> *g,
                                               const QuadratureRule<T> &quadrature_rule,
                                               const KnotSpan &knot_span)
{
    // non-static member function take this pointer.
    using namespace std::placeholders;
    auto function =
        std::bind(&BiharmonicInterfaceH1<N, T>::H1IntegralElementAssembler, this, _1, _2, _3, _4, _5, _6, _7, _8, _9);
    this->ConstraintLocalAssemble(g, quadrature_rule, knot_span, function, _constraintsEquationElements);
}

template <int N, typename T>
void BiharmonicInterfaceH1<N, T>::ConstraintMatrix(Eigen::SparseMatrix<T> &sparse_constraint)
{
    std::vector<Eigen::Triplet<T>> constraint_triplet;
    for (int i = 0; i < this->_dofMapper.Dof(); ++i)
    {
        constraint_triplet.push_back(Eigen::Triplet<T>(i, i, 1));
    }
    for (const auto &i : _constraint)
    {
        constraint_triplet.insert(constraint_triplet.end(), i.second.begin(), i.second.end());
    }
    sparse_constraint.resize(this->_dofMapper.Dof(), this->_dofMapper.Dof());
    sparse_constraint.setFromTriplets(constraint_triplet.begin(), constraint_triplet.end());
}

template <int N, typename T>
void BiharmonicInterfaceH1<N, T>::H1IntegralElementAssembler(Matrix &slave_constraint_basis,
                                                             std::vector<int> &slave_constraint_basis_indices,
                                                             Matrix &master_constraint_basis,
                                                             std::vector<int> &master_constraint_basis_indices,
                                                             Matrix &multiplier_basis,
                                                             std::vector<int> &multiplier_basis_indices,
                                                             T &integral_weight,
                                                             Edge<N, T> *edge,
                                                             const Quadrature &u)
{
    auto multiplier_domain = edge->GetDomain();
    auto slave_domain = edge->Parent(0).lock()->GetDomain();
    auto master_domain = edge->Counterpart().lock()->Parent(0).lock()->GetDomain();
    //    set up integration weights
    integral_weight = u.second;

    Vector slave_quadrature_abscissa, master_quadrature_abscissa;
    if (!Accessory::MapParametricPoint(&*multiplier_domain, u.first, &*slave_domain, slave_quadrature_abscissa))
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }
    if (!Accessory::MapParametricPoint(&*multiplier_domain, u.first, &*master_domain, master_quadrature_abscissa))
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }

    auto slave_evals = slave_domain->Eval1PhyDerAllTensor(slave_quadrature_abscissa);

    auto master_evals = master_domain->Eval1PhyDerAllTensor(master_quadrature_abscissa);

    slave_constraint_basis.resize(2, slave_evals->size());
    master_constraint_basis.resize(2, master_evals->size());

    Eigen::Matrix<T, N, 1> normal = edge->NormalDirection(u.first);

    for (int j = 0; j < slave_evals->size(); ++j)
    {
        slave_constraint_basis(0, j) = (*slave_evals)[j].second[0];
        T normal_derivative{0};
        for (int i = 0; i < N; ++i)
        {
            normal_derivative += normal(i) * (*slave_evals)[j].second[1 + i];
        }
        slave_constraint_basis(1, j) = normal_derivative;
    }
    for (int j = 0; j < master_evals->size(); ++j)
    {
        master_constraint_basis(0, j) = (*master_evals)[j].second[0];
        T normal_derivative{0};
        for (int i = 0; i < N; ++i)
        {
            normal_derivative += normal(i) * (*master_evals)[j].second[1 + i];
        }
        master_constraint_basis(1, j) = normal_derivative;
    }

    multiplier_basis = slave_constraint_basis;
    // set up indices corresponding to test basis functions and trial basis functions
    if (slave_constraint_basis_indices.size() == 0)
    {
        slave_constraint_basis_indices = slave_domain->ActiveIndex(slave_quadrature_abscissa);
        this->_dofMapper.IndicesToGlobal(slave_domain, slave_constraint_basis_indices);
    }
    if (master_constraint_basis_indices.size() == 0)
    {
        master_constraint_basis_indices = master_domain->ActiveIndex(master_quadrature_abscissa);
        this->_dofMapper.IndicesToGlobal(master_domain, master_constraint_basis_indices);
    }
    if (multiplier_basis_indices.size() == 0)
    {
        multiplier_basis_indices = slave_domain->ActiveIndex(slave_quadrature_abscissa);
        auto codimension_indices = edge->IndicesMover(1);
        for (auto &i : multiplier_basis_indices)
        {
            auto it = codimension_indices.find(i);
            if (it != codimension_indices.end())
            {
                i = it->second;
            }
        }
        this->_dofMapper.IndicesToGlobal(slave_domain, multiplier_basis_indices);
        // multiplier_basis_indices = slave_constraint_basis_indices;
    }
}
