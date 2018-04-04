//
// Created by di miao on 10/24/17.
//

#pragma once

#include "InterfaceVisitor.hpp"
#include "PoissonInterfaceVisitor.hpp"
#include <eigen3/unsupported/Eigen/KroneckerProduct>

template <int N, typename T>
class BiharmonicInterfaceVisitor : public InterfaceVisitor<N, T>
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
    BiharmonicInterfaceVisitor() : InterfaceVisitor<N, T>() {}

    void Visit(Element<1, N, T> *g);

  protected:
    void SolveConstraint(Edge<N, T> *);

    void LocalAssemble(Element<1, N, T> *,
                       const QuadratureRule<T> &,
                       const KnotSpan &);

    template <int n = N>
    typename std::enable_if<n == 3, void>::type C1IntegralElementAssembler(Matrix &slave_constraint_basis,
                                                                           std::vector<int> &slave_constraint_basis_indices,
                                                                           Matrix &master_constrint_basis,
                                                                           std::vector<int> &master_constraint_basis_indices,
                                                                           Matrix &multiplier_basis,
                                                                           std::vector<int> &multiplier_basis_indices,
                                                                           T &integral_weight,
                                                                           Edge<N, T> *edge,
                                                                           const Quadrature &u);

    template <int n = N>
    typename std::enable_if<n == 2, void>::type C1IntegralElementAssembler(Matrix &slave_constraint_basis,
                                                                           std::vector<int> &slave_constraint_basis_indices,
                                                                           Matrix &master_constrint_basis,
                                                                           std::vector<int> &master_constraint_basis_indices,
                                                                           Matrix &multiplier_basis,
                                                                           std::vector<int> &multiplier_basis_indices,
                                                                           T &integral_weight,
                                                                           Edge<N, T> *edge,
                                                                           const Quadrature &u);

  protected:
    std::vector<Eigen::Triplet<T>> _c1Slave;
    std::vector<Eigen::Triplet<T>> _c1Master;
    PoissonInterfaceVisitor<N, T> _poisson;
};

template <int N, typename T>
void BiharmonicInterfaceVisitor<N, T>::Visit(Element<1, N, T> *g)
{
    _poisson.Visit(g);
    InterfaceVisitor<N, T>::Visit(g);
}

template <int N, typename T>
void BiharmonicInterfaceVisitor<N, T>::SolveConstraint(Edge<N, T> *edge)
{
    // iterate across the constraint equation container and obtain activated global indices and lagrange multiplier indices
    std::vector<int> slave_indices = Accessory::ColIndicesVector(_c1Slave);
    std::vector<int> master_indices = Accessory::ColIndicesVector(_c1Master);
    std::vector<int> multiplier_indices = Accessory::RowIndicesVector(_c1Slave);
    std::vector<int> c0_slave_indices = *(_poisson.ConstraintData()._rowIndices);

    std::vector<int> c1_slave_indices;
    std::set_difference(slave_indices.begin(), slave_indices.end(), c0_slave_indices.begin(),
                        c0_slave_indices.end(), std::back_inserter(c1_slave_indices));

    auto c1_slave_indices_inverse_map = Accessory::IndicesInverseMap(c1_slave_indices);
    auto c0_slave_indices_inverse_map = Accessory::IndicesInverseMap(c0_slave_indices);
    auto master_indices_inverse_map = Accessory::IndicesInverseMap(master_indices);
    auto multiplier_indices_inverse_map = Accessory::IndicesInverseMap(multiplier_indices);

    std::vector<Eigen::Triplet<T>> condensed_gramian, condensed_rhs, condensed_c0_slave;
    this->CondensedTripletVia(multiplier_indices_inverse_map, c1_slave_indices_inverse_map,
                              _c1Slave, condensed_gramian);
    this->CondensedTripletVia(multiplier_indices_inverse_map, master_indices_inverse_map,
                              _c1Master, condensed_rhs);
    this->CondensedTripletVia(multiplier_indices_inverse_map, c0_slave_indices_inverse_map,
                              _c1Slave, condensed_c0_slave);

    Matrix gramian_matrix, rhs_matrix, c0_slave_matrix;
    this->MatrixAssembler(multiplier_indices_inverse_map.size(), c1_slave_indices_inverse_map.size(),
                          condensed_gramian, gramian_matrix);
    this->MatrixAssembler(multiplier_indices_inverse_map.size(), master_indices_inverse_map.size(),
                          condensed_rhs, rhs_matrix);
    this->MatrixAssembler(multiplier_indices_inverse_map.size(), c0_slave_indices_inverse_map.size(),
                          condensed_c0_slave, c0_slave_matrix);
    Accessory::removeNoise(gramian_matrix, 1e-7 * abs(gramian_matrix(0, 0)));
    Accessory::removeNoise(rhs_matrix, 1e-14);
    Accessory::removeNoise(c0_slave_matrix, 1e-7 * abs(c0_slave_matrix(0, 0)));
    Matrix c1_constraint = this->SolveNonSymmetric(gramian_matrix, rhs_matrix);
    Matrix c0_c1_constraint = this->SolveNonSymmetric(gramian_matrix, -c0_slave_matrix);
    auto c1_slave_indices_copy = c1_slave_indices;
    MatrixData<T> c1_constraint_data(c1_constraint, c1_slave_indices, master_indices);
    MatrixData<T> c0_c1constraint_data(c0_c1_constraint, c1_slave_indices_copy, c0_slave_indices);
    auto c0_slave_c1_constraint_data = c0_c1constraint_data * _poisson.ConstraintData();
    this->_constraintData = c1_constraint_data + c0_slave_c1_constraint_data + _poisson.ConstraintData();
}

template <int N, typename T>
void BiharmonicInterfaceVisitor<N, T>::LocalAssemble(Element<1, N, T> *g,
                                                     const QuadratureRule<T> &quadrature_rule,
                                                     const KnotSpan &knot_span)
{
    // non-static member function take this pointer.
    using namespace std::placeholders;
    auto c1_function =
        std::bind(&BiharmonicInterfaceVisitor<N, T>::C1IntegralElementAssembler<>, this, _1, _2, _3, _4, _5, _6, _7, _8, _9);
    this->ConstraintLocalAssemble(g, quadrature_rule, knot_span, c1_function, _c1Slave, _c1Master);
}

template <int N, typename T>
template <int n>
typename std::enable_if<n == 3, void>::type BiharmonicInterfaceVisitor<N, T>::C1IntegralElementAssembler(Matrix &slave_constraint_basis,
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
    Matrix slave_jacobian = slave_domain->JacobianMatrix(slave_quadrature_abscissa);
    Matrix master_jacobian = master_domain->JacobianMatrix(master_quadrature_abscissa);
    // Matrix master_to_slave = slave_jacobian * master_jacobian.inverse();
    Eigen::Matrix<T, 3, 1> s_s, s_t, m_s, m_t, s_n, m_n;
    s_s = slave_jacobian.col(0);
    s_t = slave_jacobian.col(1);
    m_s = master_jacobian.col(0);
    m_t = master_jacobian.col(1);
    s_n = s_s.cross(s_t);
    m_n = m_s.cross(m_t);

    Eigen::Matrix<T, 3, 3> rotation_matrix = Accessory::RotationMatrix(m_n, s_n);

    Matrix gramian = master_jacobian.transpose() * master_jacobian;
    Matrix rhs = master_jacobian.transpose() * rotation_matrix * slave_jacobian;

    Matrix sol = gramian.partialPivLu().solve(rhs);

    // Two strategies for horizontal edge and vertical edge.
    switch (edge->GetOrient())
    {
    // For south and north edge derivative w.r.t η_s should be consistent
    case Orientation::south:
    case Orientation::north:
    {
        for (int j = 0; j < slave_evals->size(); ++j)
        {
            slave_constraint_basis(0, j) = (*slave_evals)[j].second[2];
        }
        for (int j = 0; j < master_evals->size(); ++j)
        {
            master_constraint_basis(0, j) = (*master_evals)[j].second[1] * sol(0, 1) + (*master_evals)[j].second[2] * sol(1, 1);
        }
        break;
    }
    // For south and north edge derivative w.r.t ξ_s should be consistent
    case Orientation::east:
    case Orientation::west:
    {
        for (int j = 0; j < slave_evals->size(); ++j)
        {
            slave_constraint_basis(0, j) = (*slave_evals)[j].second[1];
        }
        for (int j = 0; j < master_evals->size(); ++j)
        {
            master_constraint_basis(0, j) = (*master_evals)[j].second[1] * sol(0, 0) + (*master_evals)[j].second[2] * sol(1, 0);
        }
        break;
    }
    }

    // Lagrange multiplier basis
    for (int j = 0; j < multiplier_evals->size(); ++j)
    {
        multiplier_basis(0, j) = (*multiplier_evals)[j].second[0];
    }

    Eigen::Matrix<T, 3, 3> identity;
    identity.setIdentity();

    master_constraint_basis = kroneckerProduct(master_constraint_basis, identity).eval();
    slave_constraint_basis = kroneckerProduct(slave_constraint_basis, identity).eval();
    multiplier_basis = kroneckerProduct(multiplier_basis, identity).eval();

    master_constraint_basis = (rotation_matrix * master_constraint_basis).eval();

    // set up local indices corresponding to test basis functions and trial basis functions
    if (slave_constraint_basis_indices.size() == 0)
    {
        auto index = slave_domain->ActiveIndex(slave_quadrature_abscissa);
        for (auto &i : index)
        {
            for (int j = 0; j < 3; j++)
            {
                slave_constraint_basis_indices.push_back(3 * i + j);
            }
        }
    }
    if (master_constraint_basis_indices.size() == 0)
    {
        auto index = master_domain->ActiveIndex(master_quadrature_abscissa);
        for (auto &i : index)
        {
            for (int j = 0; j < 3; j++)
            {
                master_constraint_basis_indices.push_back(3 * i + j);
            }
        }
    }
    if (multiplier_basis_indices.size() == 0)
    {
        auto index = multiplier_domain->ActiveIndex(u.first);
        for (auto &i : index)
        {
            for (int j = 0; j < 3; j++)
            {
                multiplier_basis_indices.push_back(3 * i + j);
            }
        }
    }
}

template <int N, typename T>
template <int n>
typename std::enable_if<n == 2, void>::type BiharmonicInterfaceVisitor<N, T>::C1IntegralElementAssembler(Matrix &slave_constraint_basis,
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
    Matrix slave_jacobian = slave_domain->JacobianMatrix(slave_quadrature_abscissa);
    Matrix master_jacobian = master_domain->JacobianMatrix(master_quadrature_abscissa);

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
    case Orientation::south:
    case Orientation::north:
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
    case Orientation::east:
    case Orientation::west:
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

    // Lagrange multiplier basis
    for (int j = 0; j < multiplier_evals->size(); ++j)
    {
        multiplier_basis(0, j) = (*multiplier_evals)[j].second[0];
    }

    // set up indices corresponding to test basis functions and trial basis functions
    if (slave_constraint_basis_indices.size() == 0)
    {
        slave_constraint_basis_indices = slave_domain->ActiveIndex(slave_quadrature_abscissa);
    }
    if (master_constraint_basis_indices.size() == 0)
    {
        master_constraint_basis_indices = master_domain->ActiveIndex(master_quadrature_abscissa);
    }
    if (multiplier_basis_indices.size() == 0)
    {
        std::vector<int> indices;
        for (auto &i : *multiplier_evals)
        {
            indices.push_back(i.first);
        }
        multiplier_basis_indices = indices;
    }
}