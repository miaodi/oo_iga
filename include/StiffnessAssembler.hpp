#pragma once
#include "BendingStiffnessVisitor.hpp"
#include "MembraneStiffnessVisitor.hpp"
#include "Surface.hpp"
template <typename T>
class StiffnessAssembler
{
  public:
    using LoadFunctor = typename BendingStiffnessVisitor<T>::LoadFunctor;

    StiffnessAssembler(DofMapper &dof) : _dof(dof) {}

    void Assemble(const std::vector<std::shared_ptr<Surface<3, T>>> &cells, const LoadFunctor &load, Eigen::SparseMatrix<T> &stiffness_matrix, Eigen::SparseMatrix<T> &load_vector)
    {
        std::vector<std::unique_ptr<BendingStiffnessVisitor<T>>> bending_visitors;
        std::vector<std::unique_ptr<MembraneStiffnessVisitor<T>>> membrane_visitors;
        for (auto &i : cells)
        {
            std::unique_ptr<BendingStiffnessVisitor<T>> bending(new BendingStiffnessVisitor<T>(load));
            std::unique_ptr<MembraneStiffnessVisitor<T>> membrane(new MembraneStiffnessVisitor<T>(load));
            i->Accept(*bending);
            i->Accept(*membrane);
            bending_visitors.push_back(std::move(bending));
            membrane_visitors.push_back(std::move(membrane));
        }
        std::vector<Eigen::Triplet<T>> stiffness_triplet, load_triplet;
        for (auto &i : bending_visitors)
        {
            const auto stiffness_triplet_in = i->GetStiffness();
            int id = i->ID();
            int starting_dof = _dof.StartingDof(id);
            for (const auto &j : stiffness_triplet_in)
            {
                stiffness_triplet.push_back(Eigen::Triplet<T>(starting_dof + j.row(), starting_dof + j.col(), j.value()));
            }
        }
        for (auto &i : membrane_visitors)
        {
            const auto stiffness_triplet_in = i->GetStiffness();
            const auto load_triplet_in = i->GetRhs();
            int id = i->ID();
            int starting_dof = _dof.StartingDof(id);
            for (const auto &j : stiffness_triplet_in)
            {
                stiffness_triplet.push_back(Eigen::Triplet<T>(starting_dof + j.row(), starting_dof + j.col(), j.value()));
            }
            for (const auto &j : load_triplet_in)
            {
                load_triplet.push_back(Eigen::Triplet<T>(starting_dof + j.row(), j.col(), j.value()));
            }
        }
        Eigen::SparseMatrix<T> triangle_stiffness_matrix;
        triangle_stiffness_matrix.resize(_dof.TotalDof(), _dof.TotalDof());
        load_vector.resize(_dof.TotalDof(), 1);
        triangle_stiffness_matrix.setFromTriplets(stiffness_triplet.begin(), stiffness_triplet.end());
        stiffness_matrix = triangle_stiffness_matrix.template selfadjointView<Eigen::Upper>();
        load_vector.setFromTriplets(load_triplet.begin(), load_triplet.end());
    }

  protected:
    const DofMapper _dof;
};