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

    void
    ConstraintMatrix(Eigen::SparseMatrix<T> &);

  protected:
    void
    InitializeTripletContainer();

    void SolveConstraint(Edge<N, T> *);

    void
    SolveC0Constraint(Edge<N, T> *, const int &);

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
    std::map<Edge<N, T> *, std::vector<Eigen::Triplet<T>>> _Constraint;
    std::map<Edge<N, T> *, std::vector<int>> _slaveIndices;
};

template <int N, typename T>
void PoissonInterface<N, T>::InitializeTripletContainer()
{
    _c0ConstraintsEquationElements.clear();
}

template <int N, typename T>
void PoissonInterface<N, T>::SolveConstraint(Edge<N, T> *edge)
{
    SolveC0Constraint(edge, 1);
}

// Solve the C0 constraint matrix and store it into the _c0Constraint;
template <int N, typename T>
void PoissonInterface<N, T>::SolveC0Constraint(Edge<N, T> *edge, const int &codimension)
{
    // iterate across the constraint equation container and obtain activated global indices and lagrange multiplier indices
    std::vector<int> activated_indices = Accessory::ColIndicesVector(_c0ConstraintsEquationElements);
    std::vector<int> multiplier_indices = Accessory::RowIndicesVector(_c0ConstraintsEquationElements);

    // get all the slave indices on the edge
    std::vector<int> slave_indices = this->_dofMapper.GlobalEdgeIndicesGetter(edge);

    // find slave indices and master indices
    std::vector<int> activated_slave_indices, activated_master_indices;
    std::set_intersection(activated_indices.begin(), activated_indices.end(), slave_indices.begin(),
                          slave_indices.end(), std::back_inserter(activated_slave_indices));
    std::set_difference(activated_indices.begin(), activated_indices.end(), slave_indices.begin(),
                        slave_indices.end(), std::back_inserter(activated_master_indices));
    _slaveIndices[edge] = activated_slave_indices;

    ASSERT(multiplier_indices.size() > 0, "Lagrange multiplier size is zero, needs more refines.\n");
    auto activated_slave_indices_inverse_map = Accessory::IndicesInverseMap(activated_slave_indices);
    auto activated_master_indices_inverse_map = Accessory::IndicesInverseMap(activated_master_indices);
    auto multiplier_indices_inverse_map = Accessory::IndicesInverseMap(multiplier_indices);
    std::vector<Eigen::Triplet<T>> condensed_gramian, condensed_rhs;
    this->CondensedTripletVia(multiplier_indices_inverse_map, activated_slave_indices_inverse_map,
                              _c0ConstraintsEquationElements, condensed_gramian);
    this->MoveToRhs(edge->Counterpart().lock()->Parent(0).lock()->GetDomain(), _c0ConstraintsEquationElements);
    this->CondensedTripletVia(multiplier_indices_inverse_map, activated_master_indices_inverse_map,
                              _c0ConstraintsEquationElements, condensed_rhs);
    Matrix gramian_matrix, rhs_matrix;
    this->MatrixAssembler(multiplier_indices_inverse_map.size(), activated_slave_indices_inverse_map.size(),
                          condensed_gramian, gramian_matrix);
    this->MatrixAssembler(multiplier_indices_inverse_map.size(), activated_master_indices_inverse_map.size(),
                          condensed_rhs, rhs_matrix);

    switch (codimension)
    {
    case 1:
    {
        gramian_matrix.row(1) = gramian_matrix.row(0) + gramian_matrix.row(1);
        gramian_matrix.row(gramian_matrix.rows() - 2) = gramian_matrix.row(gramian_matrix.rows() - 2) + gramian_matrix.row(gramian_matrix.rows() - 1);

        rhs_matrix.row(1) = rhs_matrix.row(0) + rhs_matrix.row(1);
        rhs_matrix.row(rhs_matrix.rows() - 2) = rhs_matrix.row(rhs_matrix.rows() - 2) + rhs_matrix.row(rhs_matrix.rows() - 1);

        Accessory::removeRow<T>(gramian_matrix, 0);
        Accessory::removeRow<T>(gramian_matrix, gramian_matrix.rows() - 1);
        Accessory::removeRow<T>(rhs_matrix, 0);
        Accessory::removeRow<T>(rhs_matrix, rhs_matrix.rows() - 1);
        break;
    }
    case 2:
    {

        {
            gramian_matrix.row(2) = gramian_matrix.row(0) + gramian_matrix.row(1) + gramian_matrix.row(2);
            rhs_matrix.row(2) = rhs_matrix.row(0) + rhs_matrix.row(1) + rhs_matrix.row(2);
        }

        {
            gramian_matrix.row(gramian_matrix.rows() - 3) = gramian_matrix.row(gramian_matrix.rows() - 3) + gramian_matrix.row(gramian_matrix.rows() - 2) + gramian_matrix.row(gramian_matrix.rows() - 1);
            rhs_matrix.row(rhs_matrix.rows() - 3) = rhs_matrix.row(rhs_matrix.rows() - 3) + rhs_matrix.row(rhs_matrix.rows() - 2) + rhs_matrix.row(rhs_matrix.rows() - 1);
        }

        Accessory::removeRow<T>(gramian_matrix, 0);
        Accessory::removeRow<T>(gramian_matrix, 0);
        Accessory::removeRow<T>(gramian_matrix, gramian_matrix.rows() - 1);
        Accessory::removeRow<T>(gramian_matrix, gramian_matrix.rows() - 1);
        Accessory::removeRow<T>(rhs_matrix, 0);
        Accessory::removeRow<T>(rhs_matrix, 0);
        Accessory::removeRow<T>(rhs_matrix, rhs_matrix.rows() - 1);
        Accessory::removeRow<T>(rhs_matrix, rhs_matrix.rows() - 1);
        break;
    }
        std::cerr << "Undefined behavior.\n"
                  << std::endl;
    }

    Matrix constraint = this->SolveNonSymmetric(gramian_matrix, rhs_matrix);
    MatrixData<T> constraint_data(constraint, activated_slave_indices, activated_master_indices);
    std::vector<Eigen::Triplet<T>> temp;
    this->Triplet(constraint_data, temp);
    _Constraint[edge] = std::move(temp);
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
void PoissonInterface<N, T>::ConstraintMatrix(Eigen::SparseMatrix<T> &sparse_constraint)
{
    std::vector<Eigen::Triplet<T>> constraint_triplet;
    for (int i = 0; i < this->_dofMapper.Dof(); ++i)
    {
        constraint_triplet.push_back(Eigen::Triplet<T>(i, i, 1));
    }
    for (const auto &i : _Constraint)
    {
        constraint_triplet.insert(constraint_triplet.end(), i.second.begin(), i.second.end());
    }
    sparse_constraint.resize(this->_dofMapper.Dof(), this->_dofMapper.Dof());
    sparse_constraint.setFromTriplets(constraint_triplet.begin(), constraint_triplet.end());
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
    auto multiplier_evals = multiplier_domain->EvalDerAllTensor(u.first);

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
