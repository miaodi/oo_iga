#pragma once
#include "BiharmonicInterfaceVisitor.hpp"
#include "Surface.hpp"
template <typename T>
class ConstraintAssembler
{
  public:
    ConstraintAssembler(DofMapper &dof) : _dof(dof) {}

    int Assemble(const std::vector<std::shared_ptr<Surface<3, T>>> &cells, std::vector<Eigen::Triplet<T>> &constraint_triplets)
    {
        std::vector<std::unique_ptr<BiharmonicInterfaceVisitor<3, T>>> biharmonic_interface_visitors;
        for (auto &i : cells)
        {
            for (int j = 0; j < 4; j++)
            {
                if (i->EdgePointerGetter(j)->IsMatched()&&i->EdgePointerGetter(j)->IsSlave())
                {
                    std::unique_ptr<BiharmonicInterfaceVisitor<3, T>> biharmonic_interface(new BiharmonicInterfaceVisitor<3, T>());
                    i->EdgePointerGetter(j)->Accept(*biharmonic_interface);
                    biharmonic_interface_visitors.push_back(std::move(biharmonic_interface));
                }
            }
        }
        int num_of_constraint = 0;
        for (auto &i : biharmonic_interface_visitors)
        {
            const auto constraint_data_in = i->ConstraintData();
            int slave_id = i->SlaveID();
            int master_id = i->MasterID();
            int slave_starting_dof = _dof.StartingDof(slave_id);
            int master_starting_dof = _dof.StartingDof(master_id);
            for (int m = 0; m < (constraint_data_in._matrix)->rows(); m++)
            {
                constraint_triplets.push_back(Eigen::Triplet<T>(num_of_constraint, slave_starting_dof + (*constraint_data_in._rowIndices)[m], 1));
                for (int n = 0; n < (constraint_data_in._matrix)->cols(); n++)
                {
                    constraint_triplets.push_back(Eigen::Triplet<T>(num_of_constraint, master_starting_dof + (*constraint_data_in._colIndices)[n], -(*constraint_data_in._matrix)(m, n)));
                }
                num_of_constraint++;
            }
        }
        return num_of_constraint;
    }

  protected:
    const DofMapper _dof;
};