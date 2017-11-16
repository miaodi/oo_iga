//
// Created by di miao on 10/24/17.
//

#pragma once

#include "InterfaceVisitor.hpp"

template <int N, typename T>
class PoissonInterface : public InterfaceVisitor<N, T>
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
    PoissonInterface(const DofMapper<N, T> &dof_mapper)
        : InterfaceVisitor<N, T>(dof_mapper) {}

    virtual void
    ConstraintMatrix(Matrix &);

  protected:
    void
    InitializeTripletContainer();

    void SolveConstraint(Edge<N, T> *);

    void
    SolveC0Constraint(Edge<N, T> *);

    void
        LocalAssemble(Element<1, N, T> *,
                      const QuadratureRule<T> &,
                      const KnotSpan &);

    void
    C0IntegralElementAssembler(Matrix &slave_constraint_basis,
                               std::vector<int> &slave_constraint_basis_indices,
                               Matrix &master_constrint_basis,
                               std::vector<int> &master_constraint_basis_indices,
                               Matrix &multiplier_basis,
                               std::vector<int> &multiplier_basis_indices,
                               T &integral_weight,
                               Edge<N, T> *edge,
                               const Quadrature &u);

  protected:
    std::vector<Eigen::Triplet<T>> _c0ConstraintsEquationElements;
    std::map<Edge<N, T> *, Matrix> _c0Constraint;
};

template <int N, typename T>
void PoissonInterface<N, T>::InitializeTripletContainer()
{
    _c0ConstraintsEquationElements.clear();
}

template <int N, typename T>
void PoissonInterface<N, T>::SolveConstraint(Edge<N, T> *edge)
{
    SolveC0Constraint(edge);
}

// Solve the C0 constraint matrix and store it into the _c0Constraint;
template <int N, typename T>
void PoissonInterface<N, T>::SolveC0Constraint(Edge<N, T> *edge)
{

    this->MoveToRhs(edge->Parent(0).lock()->GetDomain(), _c0ConstraintsEquationElements);

    auto max_row = std::max_element(_c0ConstraintsEquationElements.begin(), _c0ConstraintsEquationElements.end(), [](Eigen::Triplet<T> lhs, Eigen::Triplet<T> rhs) {
        return lhs.row() < rhs.row();
    });

    Eigen::SparseMatrix<T> constraint;
    constraint.resize(max_row->row() + 1, this->_dofMapper.Dof());

    constraint.setFromTriplets(_c0ConstraintsEquationElements.begin(), _c0ConstraintsEquationElements.end());
    Matrix dense_constraint = Matrix(constraint);

    dense_constraint.row(2) = dense_constraint.row(0) + dense_constraint.row(1) + dense_constraint.row(2);
    dense_constraint.row(dense_constraint.rows() - 3) = dense_constraint.row(dense_constraint.rows() - 3) + dense_constraint.row(dense_constraint.rows() - 2) + dense_constraint.row(dense_constraint.rows() - 1);

    Accessory::removeRow<T>(dense_constraint, 0);
    Accessory::removeRow<T>(dense_constraint, 0);
    Accessory::removeRow<T>(dense_constraint, dense_constraint.rows() - 1);
    Accessory::removeRow<T>(dense_constraint, dense_constraint.rows() - 1);
    _c0Constraint[edge] = std::move(dense_constraint);
}

template <int N, typename T>
void
    PoissonInterface<N, T>::LocalAssemble(Element<1, N, T> *g,
                                          const QuadratureRule<T> &quadrature_rule,
                                          const KnotSpan &knot_span)
{
    // non-static member function take this pointer.
    using namespace std::placeholders;
    auto c0_function =
        std::bind(&PoissonInterface<N, T>::C0IntegralElementAssembler, this, _1, _2, _3, _4, _5, _6, _7, _8, _9);
    this->ConstraintLocalAssemble(g, quadrature_rule, knot_span, c0_function, _c0ConstraintsEquationElements);
}

template <int N, typename T>
void PoissonInterface<N, T>::ConstraintMatrix(Matrix &dense_constraint)
{
    dense_constraint.resize(0, 0);
    for (const auto &i : _c0Constraint)
    {
        int row_size = i.second.rows();
        int col_size = i.second.cols();
        int current_row_size = dense_constraint.rows();
        dense_constraint.conservativeResize(row_size + current_row_size, col_size);
        dense_constraint.block(current_row_size, 0, row_size, col_size) = i.second;
    }
}

template <int N, typename T>
void PoissonInterface<N, T>::C0IntegralElementAssembler(Matrix &slave_constraint_basis,
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

    auto slave_evals = slave_domain->EvalDerAllTensor(slave_quadrature_abscissa, 0);
    auto master_evals = master_domain->EvalDerAllTensor(master_quadrature_abscissa, 0);
    auto multiplier_evals = multiplier_domain->EvalDerAllTensor(u.first, 0);

    slave_constraint_basis.resize(1, slave_evals->size());
    master_constraint_basis.resize(1, master_evals->size());
    multiplier_basis.resize(1, multiplier_evals->size());

    for (int j = 0; j < slave_evals->size(); ++j)
    {
        slave_constraint_basis(0, j) = (*slave_evals)[j].second[0];
    }
    for (int j = 0; j < master_evals->size(); ++j)
    {
        master_constraint_basis(0, j) = (*master_evals)[j].second[0];
    }
    for (int j = 0; j < multiplier_evals->size(); ++j)
    {
        multiplier_basis(0, j) = (*multiplier_evals)[j].second[0];
    }
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
        multiplier_basis_indices = multiplier_domain->ActiveIndex(u.first);
    }
}
