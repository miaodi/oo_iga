#pragma once
#include "Surface.hpp"
#include "StiffnessVisitor.hpp"

template <typename T>
class StiffnessAssembler
{
  public:
    using LoadFunctor = typename T::LoadFunctor;
    using DataType = typename T::DataType;

    StiffnessAssembler(DofMapper &dof) : _dof(dof) {}

    template <typename T1>
    void Assemble(const T1 &cells, const LoadFunctor &load, Eigen::SparseMatrix<DataType> &stiffness_matrix, Eigen::SparseMatrix<DataType> &load_vector)
    {
        std::vector<std::unique_ptr<T>> bihamronic_visitors;
        for (auto &i : cells)
        {
            std::unique_ptr<T> biharmonic(new T(load));
            i->Accept(*biharmonic);
            bihamronic_visitors.push_back(std::move(biharmonic));
        }
        std::vector<Eigen::Triplet<DataType>> stiffness_triplet, load_triplet;

        for (auto &i : bihamronic_visitors)
        {
            const auto stiffness_triplet_in = i->GetStiffness();
            const auto load_triplet_in = i->GetRhs();
            int id = i->ID();
            int starting_dof = _dof.StartingDof(id);
            for (const auto &j : stiffness_triplet_in)
            {
                stiffness_triplet.push_back(Eigen::Triplet<DataType>(starting_dof + j.row(), starting_dof + j.col(), j.value()));
            }
            for (const auto &j : load_triplet_in)
            {
                load_triplet.push_back(Eigen::Triplet<DataType>(starting_dof + j.row(), j.col(), j.value()));
            }
        }
        Eigen::SparseMatrix<DataType> triangle_stiffness_matrix;
        triangle_stiffness_matrix.resize(_dof.TotalDof(), _dof.TotalDof());
        load_vector.resize(_dof.TotalDof(), 1);
        triangle_stiffness_matrix.setFromTriplets(stiffness_triplet.begin(), stiffness_triplet.end());
        stiffness_matrix = triangle_stiffness_matrix.template selfadjointView<Eigen::Upper>();
        load_vector.setFromTriplets(load_triplet.begin(), load_triplet.end());
    }

  protected:
    const DofMapper _dof;
};