#pragma once
#include "BiharmonicInterfaceVisitor.hpp"
#include "Surface.hpp"
#include <numeric>

template <int d, int N, typename T>
class ConstraintAssembler
{
  public:
    ConstraintAssembler(DofMapper &dof) : _dof(dof) {}

    void ConstraintCreator(const std::vector<std::shared_ptr<Surface<N, T>>> &cells)
    {
        _matrix_data_container.clear();
        _vertex_indices.clear();
        _involved_indices.clear();

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

                    for (int k = 0; k <= 1; k++)
                    {
                        auto vertex = i->EdgePointerGetter(j)->VertexPointerGetter(k);
                        auto slave_vert_ind = i->EdgePointerGetter(j)->VertexPointerGetter(k)->Indices(1, 1);
                        std::for_each(slave_vert_ind.begin(), slave_vert_ind.end(), [&](int &index) { index += slave_starting_dof; });
                        _vertex_indices.insert(_vertex_indices.end(), slave_vert_ind.begin(), slave_vert_ind.end());
                    }
                    auto constraint_data = biharmonic_interface.ConstraintData();
                    std::for_each(constraint_data._rowIndices->begin(), constraint_data._rowIndices->end(), [&](int &index) { index += slave_starting_dof; });
                    std::for_each(constraint_data._colIndices->begin(), constraint_data._colIndices->end(), [&](int &index) { index += master_starting_dof; });
                    _matrix_data_container.push_back(constraint_data);
                }
            }
        }
        std::sort(_vertex_indices.begin(), _vertex_indices.end());
        _vertex_indices.erase(unique(_vertex_indices.begin(), _vertex_indices.end()), _vertex_indices.end());
        for (const auto &i : _matrix_data_container)
        {
            _involved_indices.insert(_involved_indices.end(), i._rowIndices->begin(), i._rowIndices->end());
            _involved_indices.insert(_involved_indices.end(), i._colIndices->begin(), i._colIndices->end());
        }
        std::sort(_involved_indices.begin(), _involved_indices.end());
        _involved_indices.erase(unique(_involved_indices.begin(), _involved_indices.end()), _involved_indices.end());
    }

    void AssembleConstraint(Eigen::SparseMatrix<T, Eigen::RowMajor> &sparse_constraint_matrix)
    {
        std::vector<Eigen::SparseVector<T>> constraint_container;
        int total_dof = _dof.TotalDof();
        for (auto &i : _matrix_data_container)
        {
            for (auto m = i._rowIndices->begin(); m != i._rowIndices->end(); m++)
            {
                if (std::find(_additional_constraint.begin(), _additional_constraint.end(), *m) == _additional_constraint.end())
                {
                    Eigen::SparseVector<T> sparse_vector(total_dof);
                    sparse_vector.coeffRef(*m) = 1.0;
                    for (int n = 0; n < i._colIndices->size(); n++)
                    {
                        if (std::find(_additional_constraint.begin(), _additional_constraint.end(), (*i._colIndices)[n]) == _additional_constraint.end())
                        {
                            sparse_vector.coeffRef((*i._colIndices)[n]) = -(*i._matrix)(m - i._rowIndices->begin(), n);
                        }
                    }
                    constraint_container.push_back(sparse_vector);
                }
            }
        }
        sparse_constraint_matrix.resize(constraint_container.size(), _dof.TotalDof());
        for (int i = 0; i < constraint_container.size(); i++)
        {
            sparse_constraint_matrix.row(i) = constraint_container[i].transpose();
        }
    }

    void AssembleByReducedKernel(Eigen::SparseMatrix<T> &sparse_kernel_matrix)
    {
        std::vector<Eigen::SparseVector<T>> constraint_container;
        std::vector<Eigen::SparseVector<T>> kernel_vector_container;
        int total_dof = _dof.TotalDof();
        for (auto &i : _matrix_data_container)
        {
            for (auto m = i._rowIndices->begin(); m != i._rowIndices->end();)
            {
                if (std::find(_vertex_indices.begin(), _vertex_indices.end(), *m) != _vertex_indices.end())
                {
                    if (std::find(_additional_constraint.begin(), _additional_constraint.end(), *m) == _additional_constraint.end())
                    {
                        Eigen::SparseVector<T> sparse_vector(total_dof);
                        sparse_vector.coeffRef(*m) = 1.0;
                        for (int n = 0; n < i._colIndices->size(); n++)
                        {
                            if (std::find(_additional_constraint.begin(), _additional_constraint.end(), (*i._colIndices)[n]) == _additional_constraint.end())
                            {
                                sparse_vector.coeffRef((*i._colIndices)[n]) = -(*i._matrix)(m - i._rowIndices->begin(), n);
                            }
                        }
                        constraint_container.push_back(sparse_vector);
                    }
                    i.RowRemove(m - i._rowIndices->begin());
                }
                else
                {
                    m++;
                }
            }
        }
        MatrixData<T> global;
        for (auto &i : _matrix_data_container)
        {
            global = global + i;
        }
        for (auto n = global._colIndices->begin(); n != global._colIndices->end(); n++)
        {
            Eigen::SparseVector<T> sparse_vector(total_dof);
            sparse_vector.coeffRef(*n) = 1.0;
            for (int m = 0; m < global._rowIndices->size(); m++)
            {
                if (std::find(_additional_constraint.begin(), _additional_constraint.end(), (*global._rowIndices)[m]) == _additional_constraint.end())
                {
                    sparse_vector.coeffRef((*global._rowIndices)[m]) = (*global._matrix)(m, n - global._colIndices->begin());
                }
            }
            kernel_vector_container.push_back(sparse_vector);
        }
        std::vector<int> pure_slave_vertex_indices;

        std::set_difference(_vertex_indices.begin(), _vertex_indices.end(), global._colIndices->begin(), global._colIndices->end(),
                            std::back_inserter(pure_slave_vertex_indices));
        std::vector<int> pure_slave_vertex_indices_constrained;
        std::set_difference(pure_slave_vertex_indices.begin(), pure_slave_vertex_indices.end(), _additional_constraint.begin(), _additional_constraint.end(),
                            std::back_inserter(pure_slave_vertex_indices_constrained));
        for (const auto &i : pure_slave_vertex_indices_constrained)
        {
            Eigen::SparseVector<T> sparse_vector(total_dof);
            sparse_vector.coeffRef(i) = 1.0;
            kernel_vector_container.push_back(sparse_vector);
        }

        Eigen::SparseMatrix<T, Eigen::RowMajor> sparse_constraint_matrix;
        sparse_constraint_matrix.resize(constraint_container.size(), _dof.TotalDof());
        for (int i = 0; i < constraint_container.size(); i++)
        {
            sparse_constraint_matrix.row(i) = constraint_container[i].transpose();
        }

        auto pre_kernel_it = std::partition(kernel_vector_container.begin(), kernel_vector_container.end(), [&sparse_constraint_matrix](const Eigen::SparseVector<T> &ii) { return (sparse_constraint_matrix * ii).norm() > 0; });
        Eigen::SparseMatrix<T, Eigen::ColMajor> sparse_pre_kernel_matrix;
        sparse_pre_kernel_matrix.resize(_dof.TotalDof(), pre_kernel_it - kernel_vector_container.begin());
        for (int i = 0; i < pre_kernel_it - kernel_vector_container.begin(); i++)
        {
            sparse_pre_kernel_matrix.col(i) = kernel_vector_container[i];
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dense_constraint_matrix = sparse_constraint_matrix * sparse_pre_kernel_matrix;
        Eigen::FullPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> lu_decomp(dense_constraint_matrix);
        sparse_kernel_matrix = (sparse_pre_kernel_matrix * lu_decomp.kernel()).sparseView(1e-15);

        for (auto it = pre_kernel_it; it != kernel_vector_container.end(); it++)
        {
            sparse_kernel_matrix.conservativeResize(_dof.TotalDof(), sparse_kernel_matrix.cols() + 1);
            sparse_kernel_matrix.rightCols(1) = *it;
        }
        std::vector<int> dof_indices(total_dof);
        std::iota(dof_indices.begin(), dof_indices.end(), 0);
        std::vector<int> idle_index;

        std::set_difference(dof_indices.begin(), dof_indices.end(), _involved_indices.begin(), _involved_indices.end(),
                            std::back_inserter(idle_index));

        std::vector<int> idle_index_constrained;
        std::set_difference(idle_index.begin(), idle_index.end(), _additional_constraint.begin(), _additional_constraint.end(),
                            std::back_inserter(idle_index_constrained));

        for (const auto &i : idle_index_constrained)
        {
            sparse_kernel_matrix.conservativeResize(_dof.TotalDof(), sparse_kernel_matrix.cols() + 1);
            Eigen::SparseVector<T> sparse_vector(total_dof);
            sparse_vector.coeffRef(i) = 1.0;
            sparse_kernel_matrix.rightCols(1) = sparse_vector;
        }
        std::cout << dense_constraint_matrix.rows() << " " << dense_constraint_matrix.cols() << std::endl;
        std::cout << _involved_indices.size() << std::endl;
    }

    void Additional_Constraint(const std::vector<int> &indices)
    {
        _additional_constraint = indices;
    }

  protected:
    const DofMapper _dof;
    std::vector<MatrixData<T>> _matrix_data_container;
    std::vector<int> _vertex_indices;
    std::vector<int> _involved_indices;
    std::vector<int> _additional_constraint;
};