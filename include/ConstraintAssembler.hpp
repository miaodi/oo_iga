#pragma once
#include "BiharmonicInterfaceVisitor.hpp"
#include "Surface.hpp"
template <int d, int N, typename T>
class ConstraintAssembler
{
  public:
    ConstraintAssembler(DofMapper &dof) : _dof(dof) {}

    int Assemble(const std::vector<std::shared_ptr<Surface<N, T>>> &cells, std::vector<Eigen::Triplet<T>> &constraint_triplets)
    {
        std::vector<Eigen::SparseVector<T>> constraint_container;
        std::vector<Eigen::SparseVector<T>> kernel_vector_container;
        for (auto &i : cells)
        {
            for (int j = 0; j < 4; j++)
            {
                if (i->EdgePointerGetter(j)->IsMatched() && i->EdgePointerGetter(j)->IsSlave())
                {
                    BiharmonicInterfaceVisitor<d, T> biharmonic_interface;
                    i->EdgePointerGetter(j)->Accept(biharmonic_interface);
                    int slave_id = biharmonic_interface.SlaveID();
                    int master_id = biharmonic_interface.MasterID();
                    int slave_starting_dof = _dof.StartingDof(slave_id);
                    int master_starting_dof = _dof.StartingDof(master_id);

                    std::vector<int> vertex_indices;
                    for (int k = 0; k <= 1; k++)
                    {
                        auto slave_vert_ind = i->EdgePointerGetter(j)->VertexPointerGetter(k)->Indices(1, 1);
                        vertex_indices.insert(vertex_indices.end(), slave_vert_ind.begin(), slave_vert_ind.end());
                    }
                    auto constraint_data = biharmonic_interface.ConstraintData();
                    std::for_each(vertex_indices.begin(), vertex_indices.end(), [&](int &index) { index += slave_starting_dof; });
                    std::for_each(constraint_data._rowIndices->begin(), constraint_data._rowIndices->end(), [&](int &index) { index += slave_starting_dof; });
                    std::for_each(constraint_data._colIndices->begin(), constraint_data._colIndices->end(), [&](int &index) { index += master_starting_dof; });
                    int total_dof = _dof.TotalDof();
                    for (auto m = constraint_data._rowIndices->begin(); m != constraint_data._rowIndices->end();)
                    {
                        if (std::find(vertex_indices.begin(), vertex_indices.end(), *m) != vertex_indices.end())
                        {
                            Eigen::SparseVector<T> sparse_vector(total_dof);
                            sparse_vector.coeffRef(*m) = 1.0;
                            for (int n = 0; n < constraint_data._colIndices->size(); n++)
                            {
                                sparse_vector.coeffRef((*constraint_data._colIndices)[n]) = (*constraint_data._matrix)(m - constraint_data._rowIndices->begin(), n);
                            }
                            constraint_container.push_back(sparse_vector);
                            constraint_data.RowRemove(m - constraint_data._rowIndices->begin());
                        }
                        else
                        {
                            m++;
                        }
                    }
                    for (auto n = constraint_data._colIndices->begin(); n != constraint_data._colIndices->end(); n++)
                    {
                        Eigen::SparseVector<T> sparse_vector(total_dof);
                        sparse_vector.coeffRef(*n) = 1.0;
                        for (int m = 0; m < constraint_data._rowIndices->size(); m++)
                        {
                            sparse_vector.coeffRef((*constraint_data._rowIndices)[m]) = -(*constraint_data._matrix)(m, n - constraint_data._colIndices->begin());
                        }
                        kernel_vector_container.push_back(sparse_vector);
                    }
                }
            }
        }
        Eigen::SparseMatrix<T, Eigen::RowMajor> sparse_constraint_matrix;
        sparse_constraint_matrix.resize(constraint_container.size(), _dof.TotalDof());
        for (int i = 0; i < constraint_container.size(); i++)
        {
            sparse_constraint_matrix.row(i) = constraint_container[i].transpose();
        }
        Eigen::SparseMatrix<T, Eigen::ColMajor> sparse_pre_ker_matrix;
        sparse_pre_ker_matrix.resize(_dof.TotalDof(), kernel_vector_container.size());
        for (int i = 0; i < kernel_vector_container.size(); i++)
        {
            sparse_pre_ker_matrix.col(i) = kernel_vector_container[i];
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> modified_constraint_matrix = sparse_constraint_matrix * sparse_pre_ker_matrix;
        int num = 0;
        for (int i = 0; i < modified_constraint_matrix.cols(); i++)
        {
            if (modified_constraint_matrix.col(i).norm() == 0)
            {
                num++;
            }
        }
        std::cout << modified_constraint_matrix.cols() << " " << num << " " << modified_constraint_matrix.cols() - num << std::endl;
        int num_of_constraint = 0;
        return num_of_constraint;
    }

  protected:
    const DofMapper _dof;
};