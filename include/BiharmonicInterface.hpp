//
// Created by di miao on 10/27/17.
//

#pragma once

#include "InterfaceVisitor.hpp"

template <int N, typename T>
class BiharmonicInterface : public PoissonInterface<N, T>
{
  public:
    using Knot = typename PoissonInterface<N, T>::Knot;
    using Quadrature = typename PoissonInterface<N, T>::Quadrature;
    using QuadList = typename PoissonInterface<N, T>::QuadList;
    using KnotSpan = typename PoissonInterface<N, T>::KnotSpan;
    using KnotSpanlist = typename PoissonInterface<N, T>::KnotSpanlist;
    using LoadFunctor = typename PoissonInterface<N, T>::LoadFunctor;
    using Matrix = typename PoissonInterface<N, T>::Matrix;
    using Vector = typename PoissonInterface<N, T>::Vector;
    using DomainShared_ptr = typename PoissonInterface<N, T>::DomainShared_ptr;
    using ConstraintIntegralElementAssembler = typename PoissonInterface<N, T>::ConstraintIntegralElementAssembler;

  public:
    BiharmonicInterface(const DofMapper<N, T> &dof_mapper)
        : PoissonInterface<N, T>(dof_mapper) {}

  protected:
    void
    InitializeTripletContainer();

    void
        LocalAssemble(Element<1, N, T> *,
                      const QuadratureRule<T> &,
                      const KnotSpan &);

    void SolveConstraint(Edge<N, T> *);

    void
    SolveC1Constraint(Edge<N, T> *);

    void
    C1IntegralElementAssembler(Matrix &slave_constraint_basis,
                               std::vector<int> &slave_constraint_basis_indices,
                               Matrix &master_constrint_basis,
                               std::vector<int> &master_constraint_basis_indices,
                               Matrix &multiplier_basis,
                               std::vector<int> &multiplier_basis_indices,
                               T &integral_weight,
                               Edge<N, T> *edge,
                               const Quadrature &u);

  protected:
    std::vector<Eigen::Triplet<T>> _c1ConstraintsEquationElements;
};

template <int N, typename T>
void BiharmonicInterface<N, T>::InitializeTripletContainer()
{
    PoissonInterface<N, T>::InitializeTripletContainer();
    _c1ConstraintsEquationElements.clear();
}

template <int N, typename T>
void BiharmonicInterface<N, T>::SolveConstraint(Edge<N, T> *edge)
{
    PoissonInterface<N, T>::SolveC0Constraint(edge, 2);
    SolveC1Constraint(edge);
}

// Solve the C0 constraint matrix and store it into the _c0Constraint;
template <int N, typename T>
void BiharmonicInterface<N, T>::SolveC1Constraint(Edge<N, T> *edge)
{
    // iterate across the constraint equation container and obtain activated global indices and lagrange multiplier indices
    std::vector<int> activated_indices = Accessory::ColIndicesVector(_c1ConstraintsEquationElements);
    std::vector<int> multiplier_indices = Accessory::RowIndicesVector(_c1ConstraintsEquationElements);

    // get all the slave indices on the edge
    std::vector<int> slave_indices = this->_dofMapper.GlobalEdgeIndicesGetter(edge);

    // Get the slave indices that used for C0 constraint
    std::vector<int> c0_slave_indices = this->_slaveIndices[edge];

    // find slave indices and master indices
    std::vector<int> activated_slave_indices, activated_master_indices;
    std::set_difference(slave_indices.begin(), slave_indices.end(), c0_slave_indices.begin(),
                        c0_slave_indices.end(), std::back_inserter(activated_slave_indices));
    std::set_difference(activated_indices.begin(), activated_indices.end(), slave_indices.begin(),
                        slave_indices.end(), std::back_inserter(activated_master_indices));

    // Inverse map used for creating condensed matrix
    auto activated_slave_indices_inverse_map = Accessory::IndicesInverseMap(activated_slave_indices);
    auto c0_slave_indices_inverse_map = Accessory::IndicesInverseMap(c0_slave_indices);
    auto activated_master_indices_inverse_map = Accessory::IndicesInverseMap(activated_master_indices);
    auto multiplier_indices_inverse_map = Accessory::IndicesInverseMap(multiplier_indices);

    std::vector<Eigen::Triplet<T>> condensed_gramian, condensed_rhs_c1, condensed_rhs_c0_slave;
    this->CondensedTripletVia(multiplier_indices_inverse_map, activated_slave_indices_inverse_map,
                              _c1ConstraintsEquationElements, condensed_gramian);
    this->MoveToRhs(edge->Counterpart().lock()->Parent(0).lock()->GetDomain(), _c1ConstraintsEquationElements);
    this->CondensedTripletVia(multiplier_indices_inverse_map, activated_master_indices_inverse_map,
                              _c1ConstraintsEquationElements, condensed_rhs_c1);
    this->CondensedTripletVia(multiplier_indices_inverse_map, c0_slave_indices_inverse_map,
                              _c1ConstraintsEquationElements, condensed_rhs_c0_slave);
    Matrix gramian_matrix, rhs_matrix_c1, rhs_matrix_c0_slave;
    this->MatrixAssembler(multiplier_indices_inverse_map.size(), activated_slave_indices_inverse_map.size(),
                          condensed_gramian, gramian_matrix);
    this->MatrixAssembler(multiplier_indices_inverse_map.size(), activated_master_indices_inverse_map.size(),
                          condensed_rhs_c1, rhs_matrix_c1);
    this->MatrixAssembler(multiplier_indices_inverse_map.size(), c0_slave_indices_inverse_map.size(),
                          condensed_rhs_c0_slave, rhs_matrix_c0_slave);

    // Codimension 2 Lagrange multiplier

    {
        gramian_matrix.row(2) = gramian_matrix.row(0) + gramian_matrix.row(1) + gramian_matrix.row(2);
        rhs_matrix_c1.row(2) = rhs_matrix_c1.row(0) + rhs_matrix_c1.row(1) + rhs_matrix_c1.row(2);
        rhs_matrix_c0_slave.row(2) = rhs_matrix_c0_slave.row(0) + rhs_matrix_c0_slave.row(1) + rhs_matrix_c0_slave.row(2);
    }

    {
        gramian_matrix.row(gramian_matrix.rows() - 3) = gramian_matrix.row(gramian_matrix.rows() - 3) + gramian_matrix.row(gramian_matrix.rows() - 2) + gramian_matrix.row(gramian_matrix.rows() - 1);

        rhs_matrix_c1.row(rhs_matrix_c1.rows() - 3) = rhs_matrix_c1.row(rhs_matrix_c1.rows() - 3) + rhs_matrix_c1.row(rhs_matrix_c1.rows() - 2) + rhs_matrix_c1.row(rhs_matrix_c1.rows() - 1);

        rhs_matrix_c0_slave.row(rhs_matrix_c0_slave.rows() - 3) = rhs_matrix_c0_slave.row(rhs_matrix_c0_slave.rows() - 3) + rhs_matrix_c0_slave.row(rhs_matrix_c0_slave.rows() - 2) + rhs_matrix_c0_slave.row(rhs_matrix_c0_slave.rows() - 1);
    }

    Accessory::removeRow<T>(gramian_matrix, 0);
    Accessory::removeRow<T>(gramian_matrix, 0);
    Accessory::removeRow<T>(gramian_matrix, gramian_matrix.rows() - 1);
    Accessory::removeRow<T>(gramian_matrix, gramian_matrix.rows() - 1);
    Accessory::removeRow<T>(rhs_matrix_c1, 0);
    Accessory::removeRow<T>(rhs_matrix_c1, 0);
    Accessory::removeRow<T>(rhs_matrix_c1, rhs_matrix_c1.rows() - 1);
    Accessory::removeRow<T>(rhs_matrix_c1, rhs_matrix_c1.rows() - 1);
    Accessory::removeRow<T>(rhs_matrix_c0_slave, 0);
    Accessory::removeRow<T>(rhs_matrix_c0_slave, 0);
    Accessory::removeRow<T>(rhs_matrix_c0_slave, rhs_matrix_c0_slave.rows() - 1);
    Accessory::removeRow<T>(rhs_matrix_c0_slave, rhs_matrix_c0_slave.rows() - 1);

    Matrix c1_constraint = this->SolveNonSymmetric(gramian_matrix, rhs_matrix_c1);
    Matrix c0_c1_constraint = this->SolveNonSymmetric(gramian_matrix, rhs_matrix_c0_slave);

    auto activated_slave_indices_copy = activated_slave_indices;
    MatrixData<T> c1_constraint_data(c1_constraint, activated_slave_indices, activated_master_indices);
    MatrixData<T> c0_c1constraint_data(c0_c1_constraint, activated_slave_indices_copy, c0_slave_indices);
    MatrixData<T> c0_constraint_data = this->ToMatrixData(this->_Constraint[edge]);
    auto c0_slave_c1_constraint_data = c0_c1constraint_data * c0_constraint_data;

    this->Triplet(c1_constraint_data, this->_Constraint[edge]);
    this->Triplet(c0_slave_c1_constraint_data, this->_Constraint[edge]);
}

template <int N, typename T>
void
    BiharmonicInterface<N, T>::LocalAssemble(Element<1, N, T> *g,
                                             const QuadratureRule<T> &quadrature_rule,
                                             const KnotSpan &knot_span)
{
    PoissonInterface<N, T>::LocalAssemble(g, quadrature_rule, knot_span);
    // non-static member function take this pointer.
    using namespace std::placeholders;
    auto c1_function =
        std::bind(&BiharmonicInterface<N, T>::C1IntegralElementAssembler, this, _1, _2, _3, _4, _5, _6, _7, _8, _9);
    this->ConstraintLocalAssemble(g, quadrature_rule, knot_span, c1_function, _c1ConstraintsEquationElements);
}

template <int N, typename T>
void BiharmonicInterface<N, T>::C1IntegralElementAssembler(Matrix &slave_constraint_basis,
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
    // Set up integration weights
    integral_weight = u.second;

    // Map abscissa from Lagrange multiplier space to slave and master domain
    Vector slave_quadrature_abscissa, master_quadrature_abscissa;
    if (!Accessory::MapParametricPoint(&*multiplier_domain, u.first, &*slave_domain, slave_quadrature_abscissa))
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }
    if (!Accessory::MapParametricPoint(&*multiplier_domain, u.first, &*master_domain, master_quadrature_abscissa))
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }

    // Evaluate derivative upto 1^st order in slave and master domain
    auto slave_evals = slave_domain->EvalDerAllTensor(slave_quadrature_abscissa, 1);
    auto master_evals = master_domain->EvalDerAllTensor(master_quadrature_abscissa, 1);

    //  Evaluate Lagrange multiplier basis
    auto multiplier_evals = multiplier_domain->EvalDualAllTensor(u.first);

    // Resize integration matrices
    slave_constraint_basis.resize(1, slave_evals->size());
    master_constraint_basis.resize(1, master_evals->size());
    multiplier_basis.resize(1, multiplier_evals->size());

    // Compute the following matrix
    // +-----------+-----------+
    // | ∂ξ_m/∂ξ_s | ∂η_m/∂ξ_s |
    // +-----------+-----------+
    // | ∂ξ_m/∂η_s | ∂η_m/∂η_s |
    // +-----------+-----------+
    Matrix slave_jacobian = (slave_domain->JacobianMatrix(slave_quadrature_abscissa)).transpose();
    Matrix master_jacobian = (master_domain->JacobianMatrix(master_quadrature_abscissa)).transpose();
    // Matrix master_to_slave = slave_jacobian * master_jacobian.inverse();

    // Substitute master coordinate of master basis by slave coordinate
    for (auto &i : *master_evals)
    {
        Vector tmp = slave_jacobian * master_jacobian.partialPivLu().solve((Vector(2) << i.second[1], i.second[2]).finished());
        i.second[1] = tmp(0);
        i.second[2] = tmp(1);
    }

    // Two strategies for horizontal edge and vertical edge.
    switch (edge->GetOrient())
    {
    // For south and north edge derivative w.r.t η_s should be consistent
    case south:
    case north:
    {
        for (int j = 0; j < slave_evals->size(); ++j)
        {
            slave_constraint_basis(0, j) = (*slave_evals)[j].second[2];
        }
        for (int j = 0; j < master_evals->size(); ++j)
        {
            master_constraint_basis(0, j) = (*master_evals)[j].second[2];
        }
        break;
    }
    // For south and north edge derivative w.r.t ξ_s should be consistent
    case east:
    case west:
    {
        for (int j = 0; j < slave_evals->size(); ++j)
        {
            slave_constraint_basis(0, j) = (*slave_evals)[j].second[1];
        }
        for (int j = 0; j < master_evals->size(); ++j)
        {
            master_constraint_basis(0, j) = (*master_evals)[j].second[1];
        }
        break;
    }
    }

#pragma region
// switch (edge->GetOrient())
// {
// // For south and north edge derivative w.r.t η_s should be consistent
// case south:
// case north:
// {
//     for (int j = 0; j < slave_evals->size(); ++j)
//     {
//         slave_constraint_basis(0, j) = (*slave_evals)[j].second[2];
//     }
//     break;
// }
// // For south and north edge derivative w.r.t ξ_s should be consistent
// case east:
// case west:
// {
//     for (int j = 0; j < slave_evals->size(); ++j)
//     {
//         slave_constraint_basis(0, j) = (*slave_evals)[j].second[1];
//     }
//     break;
// }
// }
// T alpha, beta, gamma;
// switch (edge->GetOrient())
// {
// // For south and north edge derivative w.r.t η_s should be consistent
// case south:
// case north:
// {
//     auto geomDriXi = master_domain->AffineMap(master_quadrature_abscissa, {1, 0});
//     auto geomDriEta = master_domain->AffineMap(master_quadrature_abscissa, {0, 1});
//     auto geomDriEtaSlave = slave_domain->AffineMap(slave_quadrature_abscissa, {0, 1});
//     alpha = geomDriEtaSlave(0) * geomDriXi(1) - geomDriXi(0) * geomDriEtaSlave(1);
//     beta = geomDriEta(0) * geomDriEtaSlave(1) - geomDriEtaSlave(0) * geomDriEta(1);
//     gamma = geomDriEta(0) * geomDriXi(1) - geomDriXi(0) * geomDriEta(1);
//     for (int j = 0; j < master_evals->size(); ++j)
//     {
//         master_constraint_basis(0, j) = (*master_evals)[j].second[1] * beta / gamma + (*master_evals)[j].second[2] * alpha / gamma;
//     }
//     break;
// }
// // For south and north edge derivative w.r.t ξ_s should be consistent
// case east:
// case west:
// {
//     auto geomDriXi = master_domain->AffineMap(master_quadrature_abscissa, {1, 0});
//     auto geomDriEta = master_domain->AffineMap(master_quadrature_abscissa, {0, 1});
//     auto geomDriXiSlave = slave_domain->AffineMap(slave_quadrature_abscissa, {1, 0});
//     alpha = geomDriXiSlave(0) * geomDriXi(1) - geomDriXi(0) * geomDriXiSlave(1);
//     beta = geomDriEta(0) * geomDriXiSlave(1) - geomDriXiSlave(0) * geomDriEta(1);
//     gamma = geomDriEta(0) * geomDriXi(1) - geomDriXi(0) * geomDriEta(1);
//     for (int j = 0; j < slave_evals->size(); ++j)
//     {
//         master_constraint_basis(0, j) = (*master_evals)[j].second[1] * beta / gamma + (*master_evals)[j].second[2] * alpha / gamma;
//     }
//     break;
// }
// }
#pragma endregion

    // Lagrange multiplier basis
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