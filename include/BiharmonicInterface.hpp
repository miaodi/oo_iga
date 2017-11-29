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

    void
    ConstraintMatrix(Matrix &);

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
    std::map<Edge<N, T> *, Matrix> _c1Constraint;
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
    PoissonInterface<N, T>::SolveC0Constraint(edge);
    SolveC1Constraint(edge);
}

// Solve the C0 constraint matrix and store it into the _c0Constraint;
template <int N, typename T>
void BiharmonicInterface<N, T>::SolveC1Constraint(Edge<N, T> *edge)
{
    this->MoveToRhs(edge->Parent(0).lock()->GetDomain(), _c1ConstraintsEquationElements);

    auto max_row = std::max_element(_c1ConstraintsEquationElements.begin(), _c1ConstraintsEquationElements.end(), [](Eigen::Triplet<T> lhs, Eigen::Triplet<T> rhs) {
        return lhs.row() < rhs.row();
    });

    Eigen::SparseMatrix<T> constraint;
    constraint.resize(max_row->row() + 1, this->_dofMapper.Dof());

    constraint.setFromTriplets(_c1ConstraintsEquationElements.begin(), _c1ConstraintsEquationElements.end());

    Matrix dense_constraint = Matrix(constraint);

    _c1Constraint[edge] = std::move(dense_constraint);
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
void BiharmonicInterface<N, T>::ConstraintMatrix(Matrix &dense_constraint)
{
    PoissonInterface<N, T>::ConstraintMatrix(dense_constraint);
    for (const auto &i : _c1Constraint)
    {
        int row_size = i.second.rows();
        int col_size = i.second.cols();
        int current_row_size = dense_constraint.rows();
        dense_constraint.conservativeResize(row_size + current_row_size, col_size);
        dense_constraint.block(current_row_size, 0, row_size, col_size) = i.second;
    }
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
    auto edge_domain = edge->GetDomain();
    auto slave_domain = edge->Parent(0).lock()->GetDomain();
    auto master_domain = edge->Counterpart().lock()->Parent(0).lock()->GetDomain();

    auto multiplier_domain = edge_domain;

    auto edge_knot = edge_domain->KnotVectorGetter(0);
    // edge_knot.erase(edge_knot.begin());
    // edge_knot.erase(edge_knot.end() - 1);
    // edge_knot.erase(edge_knot.begin());
    // edge_knot.erase(edge_knot.end() - 1);

    // auto multiplier_domain = std::make_shared<TensorBsplineBasis<1, double>>(edge_knot);

    // Set up integration weights
    integral_weight = u.second;

    // Map abscissa from Lagrange multiplier space to slave and master domain
    Vector slave_quadrature_abscissa, master_quadrature_abscissa;
    if (!Accessory::MapParametricPoint(&*edge_domain, u.first, &*slave_domain, slave_quadrature_abscissa))
    {
        std::cout << "MapParametericPoint failed" << std::endl;
    }
    if (!Accessory::MapParametricPoint(&*edge_domain, u.first, &*master_domain, master_quadrature_abscissa))
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
