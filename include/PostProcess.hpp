#pragma once

#include "DomainVisitor.hpp"

template <int N, typename T>
class PostProcess : public DomainVisitor<2, N, T>
{
  public:
    using Knot = typename DomainVisitor<2, N, T>::Knot;
    using Quadrature = typename DomainVisitor<2, N, T>::Quadrature;
    using QuadList = typename DomainVisitor<2, N, T>::QuadList;
    using KnotSpan = typename DomainVisitor<2, N, T>::KnotSpan;
    using KnotSpanlist = typename DomainVisitor<2, N, T>::KnotSpanlist;
    using Matrix = typename DomainVisitor<2, N, T>::Matrix;
    using Vector = typename DomainVisitor<2, N, T>::Vector;
    using LoadFunctor = std::function<std::vector<T>(const Knot &)>;
    using DomainShared_ptr =
        typename std::shared_ptr<PhyTensorBsplineBasis<2, N, T>>;

  public:
    PostProcess(const DofMapper<N, T> &dof_mapper, const Vector &solution,
                const LoadFunctor &analytical_solution)
        : DomainVisitor<2, N, T>(dof_mapper), _solution(solution),
          _analyticalSolution(analytical_solution) {}
    T L2Norm() const;
    T H2Norm() const;

  protected:
    void LocalAssemble(Element<2, N, T> *, const QuadratureRule<T> &,
                       const KnotSpan &);

  protected:
    const LoadFunctor &_analyticalSolution;
    const Vector &_solution;
    std::map<DomainShared_ptr, std::pair<T, T>> _normContainer;
    std::map<DomainShared_ptr, std::pair<T, T>> _normContainerH2;
};

template <int N, typename T>
void PostProcess<N, T>::LocalAssemble(Element<2, N, T> *g,
                                      const QuadratureRule<T> &quadrature_rule,
                                      const KnotSpan &knot_span)
{

    auto domain = g->GetDomain();
    QuadList quadrature_points;

    quadrature_rule.MapToQuadrature(knot_span, quadrature_points);

    auto index = domain->ActiveIndex(quadrature_points[0].first);
    this->_dofMapper.IndicesToGlobal(domain, index);
    auto num_of_quadrature = quadrature_points.size();
    T relative{0}, denominator{0};
    T relativeH2{0}, denominatorH2{0};

    for (int i = 0; i < quadrature_points.size(); ++i)
    {
        T analytical =
            _analyticalSolution(domain->AffineMap(quadrature_points[i].first))[0];
        auto eval = domain->Eval2PhyDerAllTensor(quadrature_points[i].first);
        T approx{0};
        for (int j = 0; j < index.size(); j++)
        {
            approx += (*eval)[j].second[0] * _solution(index[j]);
        }
        auto analyticalH2 =
            _analyticalSolution(domain->AffineMap(quadrature_points[i].first));
        std::vector<T> approxH2(6, 0.0);
        for (int j = 0; j < index.size(); j++)
        {
            approxH2[0] += (*eval)[j].second[0] * _solution(index[j]);
            approxH2[1] += (*eval)[j].second[1] * _solution(index[j]);
            approxH2[2] += (*eval)[j].second[2] * _solution(index[j]);
            approxH2[3] += (*eval)[j].second[3] * _solution(index[j]);
            approxH2[4] += (*eval)[j].second[4] * _solution(index[j]);
            approxH2[5] += (*eval)[j].second[5] * _solution(index[j]);
        }
        relative += quadrature_points[i].second * pow(analytical - approx, 2);
        denominator += quadrature_points[i].second * pow(analytical, 2);
        denominatorH2 += quadrature_points[i].second *
                         (pow(analyticalH2[0], 2) + pow(analyticalH2[1], 2) +
                          pow(analyticalH2[2], 2) + pow(analyticalH2[3], 2) +
                          pow(analyticalH2[4], 2) + pow(analyticalH2[5], 2));
        relativeH2 +=
            quadrature_points[i].second * (pow(analyticalH2[0] - approxH2[0], 2) +
                                           pow(analyticalH2[1] - approxH2[1], 2) +
                                           pow(analyticalH2[2] - approxH2[2], 2) +
                                           pow(analyticalH2[3] - approxH2[3], 2) +
                                           pow(analyticalH2[4] - approxH2[4], 2) +
                                           pow(analyticalH2[5] - approxH2[5], 2));
    }

    auto it = _normContainer.find(domain);
    auto itH2 = _normContainerH2.find(domain);
    std::lock_guard<std::mutex> lock(this->_mutex);
    if (it != _normContainer.end())
    {
        (*it).second.first += relative;
        (*it).second.second += denominator;
    }
    else
    {
        _normContainer[domain] = std::make_pair(0, 0);
    }
    if (itH2 != _normContainerH2.end())
    {
        (*itH2).second.first += relativeH2;
        (*itH2).second.second += denominatorH2;
    }
    else
    {
        _normContainerH2[domain] = std::make_pair(0, 0);
    }
}

template <int N, typename T>
T PostProcess<N, T>::L2Norm() const
{
    T relative{0}, denominator{0};
    for (const auto &i : _normContainer)
    {
        relative += i.second.first;
        denominator += i.second.second;
    }
    return sqrt(relative / denominator);
}

template <int N, typename T>
T PostProcess<N, T>::H2Norm() const
{
    T relativeH2{0}, denominatorH2{0};
    for (const auto &i : _normContainerH2)
    {
        relativeH2 += i.second.first;
        denominatorH2 += i.second.second;
    }
    return sqrt(relativeH2 / denominatorH2);
}
