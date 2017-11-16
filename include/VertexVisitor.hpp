#pragma once
#include "Visitor.hpp"

template <int N, typename T>
class VertexVisitor : public Visitor<0, N, T>
{
public:
  using Matrix = typename DomainVisitor<0, N, T>::Matrix;
  using Vector = typename DomainVisitor<0, N, T>::Vector;
  using DomainShared_ptr = typename std::shared_ptr<PhyTensorBsplineBasis<2, N, T>>;

public:
  VertexVisitor(const DofMapper<N, T> &dof_mapper) : _dofMapper(dof_mapper) {}

  void Visit(Element<0, N, T> *);

  void
  ConstraintMatrix(Matrix &);

protected:
  virtual void VertexConstraint(Vertex<N, T> *, Vertex<N, T> *) = 0;

protected:
  std::vector<Matrix> _constraint;
  const DofMapper<N, T> &_dofMapper;
};

template <int N, typename T>
void VertexVisitor<N, T>::Visit(Element<0, N, T> *g)
{
  auto vertex = dynamic_cast<Vertex<N, T> *>(g);
  if (!vertex->IsMaster() || vertex->IsDirichlet())
    return;
  auto slave_vertices = vertex->Counterpart();
  for (const auto &i : slave_vertices)
  {
    VertexConstraint(vertex, &*(i.lock()));
  }
}

template <int N, typename T>
void VertexVisitor<N, T>::ConstraintMatrix(Matrix &vertex_constraint)
{
  for (const auto &i : _constraint)
  {
    int current_rows = vertex_constraint.rows();
    vertex_constraint.conservativeResize(current_rows + i.rows(), i.cols());
    vertex_constraint.block(current_rows, 0, i.rows(), i.cols()) = i;
  }
}