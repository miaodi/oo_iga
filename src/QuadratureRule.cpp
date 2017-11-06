//
// Created by miaodi on 07/05/2017.
//

#include "QuadratureRule.h"
#include <boost/multiprecision/gmp.hpp>

template <typename T>
QuadratureRule<T>::QuadratureRule(const int &num)
{
    SetUpQuadrature(num);
}

template <typename T>
void QuadratureRule<T>::SetUpQuadrature(int num)
{
    _size = num;
    if (std::is_floating_point<T>::value && num <= 9)
    {
        LookupReference(_size, _quadrature);
    }
    else
    {
        ComputeReference(_size, _quadrature);
    }
}

template <typename T>
void QuadratureRule<T>::PrintCurrentQuadrature() const
{
    if (_quadrature.size())
    {
        for (const auto &i : _quadrature)
        {
            std::cout << std::setprecision(std::numeric_limits<T>::digits) << "x=" << i.first(0) << std::endl;
            std::cout << std::setprecision(std::numeric_limits<T>::digits) << "w=" << i.second << std::endl;
            std::cout << std::endl;
        }
    }
}

template <typename T>
void QuadratureRule<T>::LookupReference(int num, QuadratureRule::QuadList &quadrature)
{
    quadrature.resize(num);
    for (auto &i : quadrature)
    {
        i.first.resize(1);
    }

    switch (num)
    {
    case 1:
    {
        quadrature[0].first(0) = 0.000000000000000000000000000000l;
        quadrature[0].second = 2.000000000000000000000000000000l;
        break;
    }
    case 2:
    {
        quadrature[0].first(0) = -0.577350269189625764509148780502l;
        quadrature[1].first(0) = 0.577350269189625764509148780502l;

        quadrature[0].second = 1.000000000000000000000000000000l;
        quadrature[1].second = 1.000000000000000000000000000000l;
        break;
    }

    case 3:
    {
        quadrature[0].first(0) = -0.774596669241483377035853079956l;
        quadrature[1].first(0) = 0.000000000000000000000000000000l;
        quadrature[2].first(0) = 0.774596669241483377035853079956l;

        quadrature[0].second = 0.555555555555555555555555555556l;
        quadrature[1].second = 0.888888888888888888888888888889l;
        quadrature[2].second = 0.555555555555555555555555555556l;
        break;
    }
    case 4:
    {
        quadrature[0].first(0) = -0.861136311594052575223946488893l;
        quadrature[1].first(0) = -0.339981043584856264802665759103l;
        quadrature[2].first(0) = 0.339981043584856264802665759103l;
        quadrature[3].first(0) = 0.861136311594052575223946488893l;

        quadrature[0].second = 0.347854845137453857373063949222l;
        quadrature[1].second = 0.652145154862546142626936050778l;
        quadrature[2].second = 0.652145154862546142626936050778l;
        quadrature[3].second = 0.347854845137453857373063949222l;
        break;
    }
    case 5:
    {
        quadrature[0].first(0) = -0.906179845938663992797626878299l;
        quadrature[1].first(0) = -0.538469310105683091036314420700l;
        quadrature[2].first(0) = 0.000000000000000000000000000000l;
        quadrature[3].first(0) = 0.538469310105683091036314420700l;
        quadrature[4].first(0) = 0.906179845938663992797626878299l;

        quadrature[0].second = 0.236926885056189087514264040720l;
        quadrature[1].second = 0.478628670499366468041291514836l;
        quadrature[2].second = 0.568888888888888888888888888889l;
        quadrature[3].second = 0.478628670499366468041291514836l;
        quadrature[4].second = 0.236926885056189087514264040720l;
        break;
    }
    case 6:
    {
        quadrature[0].first(0) = -0.932469514203152027812301554494l;
        quadrature[1].first(0) = -0.661209386466264513661399595020l;
        quadrature[2].first(0) = -0.238619186083196908630501721681l;
        quadrature[3].first(0) = 0.238619186083196908630501721681l;
        quadrature[4].first(0) = 0.661209386466264513661399595020l;
        quadrature[5].first(0) = 0.932469514203152027812301554494l;

        quadrature[0].second = 0.171324492379170345040296142173l;
        quadrature[1].second = 0.360761573048138607569833513838l;
        quadrature[2].second = 0.467913934572691047389870343990l;
        quadrature[3].second = 0.467913934572691047389870343990l;
        quadrature[4].second = 0.360761573048138607569833513838l;
        quadrature[5].second = 0.171324492379170345040296142173l;
        break;
    }
    case 7:
    {
        quadrature[0].first(0) = -0.949107912342758524526189684048l;
        quadrature[1].first(0) = -0.741531185599394439863864773281l;
        quadrature[2].first(0) = -0.405845151377397166906606412077l;
        quadrature[3].first(0) = 0.000000000000000000000000000000l;
        quadrature[4].first(0) = 0.405845151377397166906606412077l;
        quadrature[5].first(0) = 0.741531185599394439863864773281l;
        quadrature[6].first(0) = 0.949107912342758524526189684048l;

        quadrature[0].second = 0.129484966168869693270611432679l;
        quadrature[1].second = 0.279705391489276667901467771424l;
        quadrature[2].second = 0.381830050505118944950369775489l;
        quadrature[3].second = 0.417959183673469387755102040816l;
        quadrature[4].second = 0.381830050505118944950369775489l;
        quadrature[5].second = 0.279705391489276667901467771424l;
        quadrature[6].second = 0.129484966168869693270611432679l;
        break;
    }

    case 8:
    {
        quadrature[0].first(0) = -0.960289856497536231683560868569l;
        quadrature[1].first(0) = -0.796666477413626739591553936476l;
        quadrature[2].first(0) = -0.525532409916328985817739049189l;
        quadrature[3].first(0) = -0.183434642495649804939476142360l;
        quadrature[4].first(0) = 0.183434642495649804939476142360l;
        quadrature[5].first(0) = 0.525532409916328985817739049189l;
        quadrature[6].first(0) = 0.796666477413626739591553936476l;
        quadrature[7].first(0) = 0.960289856497536231683560868569l;

        quadrature[0].second = 0.101228536290376259152531354310l;
        quadrature[1].second = 0.222381034453374470544355994426l;
        quadrature[2].second = 0.313706645877887287337962201987l;
        quadrature[3].second = 0.362683783378361982965150449277l;
        quadrature[4].second = 0.362683783378361982965150449277l;
        quadrature[5].second = 0.313706645877887287337962201987l;
        quadrature[6].second = 0.222381034453374470544355994426l;
        quadrature[7].second = 0.101228536290376259152531354310l;
        break;
    }

    case 9:
    {
        quadrature[0].first(0) = -0.968160239507626089835576203l;
        quadrature[1].first(0) = -0.836031107326635794299429788l;
        quadrature[2].first(0) = -0.613371432700590397308702039l;
        quadrature[3].first(0) = -0.324253423403808929038538015l;
        quadrature[4].first(0) = 0.000000000000000000000000000l;
        quadrature[5].first(0) = 0.324253423403808929038538015l;
        quadrature[6].first(0) = 0.613371432700590397308702039l;
        quadrature[7].first(0) = 0.836031107326635794299429788l;
        quadrature[8].first(0) = 0.968160239507626089835576203l;

        quadrature[0].second = 0.081274388361574411971892158111l;
        quadrature[1].second = 0.18064816069485740405847203124l;
        quadrature[2].second = 0.26061069640293546231874286942l;
        quadrature[3].second = 0.31234707704000284006863040658l;
        quadrature[4].second = 0.33023935500125976316452506929l;
        quadrature[5].second = 0.31234707704000284006863040658l;
        quadrature[6].second = 0.26061069640293546231874286942l;
        quadrature[7].second = 0.18064816069485740405847203124l;
        quadrature[8].second = 0.081274388361574411971892158111l;
        break;
    }
    }
}

template <typename T>
void QuadratureRule<T>::ComputeReference(int n, QuadratureRule::QuadList &quadrature, unsigned digits)
{
    quadrature.resize(n);
    for (auto &i : quadrature)
    {
        i.first.resize(1);
    }

    std::vector<T> x(n, 0), w(n, 0);

    const unsigned int max_its = digits;
    const T tolerance = pow(T(0.1), static_cast<int>(digits));

    // Find only half the roots because of symmetry
    const unsigned int m = n / 2;

    // Three recurrence relation values and one derivative value.
    T pn(0.0),     // P_{n}
        pnm1(0.0), // P_{n-1}
        pnm2(0.0), // P_{n-2}
        dpn(0.0);  // P'_{n}

    // If n is odd, the rule always contains the point x==0.
    if (n % 2)
    {
        x[m] = T(0.0);
        pn = 1.0;
        pnm1 = 0.0;
        // Compute P'_n(0)
        for (int j = 0; j < n - 1; ++j)
        {
            pnm2 = pnm1;
            pnm1 = pn;
            pn = -j * pnm2 / static_cast<T>(j + 1);
        }
        dpn = n * pn;
        w[m] = T(2.0) / (dpn * dpn);
    }

    for (unsigned int i = 0; i < m; ++i)
    {
        // Remarkably, this simple relation provides a very
        // good initial guess for x_i.  See, for example,
        // F. G. Lether and P. R. Wenston
        // Journal of Computational and Applied Mathematics
        // Minimax approximations to the zeros of Pn(x) and
        // Gauss-Legendre quadrature, Volume 59,
        // Issue 2  (May 1995), p. 245-252, 1995
        const T pi = static_cast<T>(mpf_float_100("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148"));
        x[i] = cos(pi * (i + 0.75) / (n + 0.5));

        // Newton loop iteration counter
        unsigned int n_its = 0;

        // Begin Newton iterations
        do
        {
            // Initialize recurrence relation
            pn = 1.0;
            pnm1 = 0.0;

            // Use recurrence relation to compute P_n(x[i])
            for (int j = 0; j < n; ++j)
            {
                pnm2 = pnm1;
                pnm1 = pn;
                pn = ((2.0 * j + 1.0) * x[i] * pnm1 - j * pnm2) / static_cast<T>(j + 1);
            }

            // A recurrence relation also gives the derivative.
            dpn = n * (x[i] * pn - pnm1) / (x[i] * x[i] - 1.0);

            // Compute Newton update
            x[i] -= pn / dpn;

            // Increment iteration counter
            n_its++;
        } while ((abs(pn) > tolerance) && (n_its < max_its));

        // if (n_its>=max_its)
        //     gsWarn << "Max ("<<n_its<<") Newton iterations reached, error="
        //            <<math::abs(pn)<<" for node "<<i<<"(tolerance="<<tolerance<<").\n";
        //     gsDebug << "Newton converged in " << n_its << " iterations, ";
        //     gsDebug << "with tolerance=" << math::abs(pn) << std::endl;

        // Set x[i] and its mirror image.  We set these in increasing order.
        x[n - 1 - i] = x[i];
        x[i] = -x[i];

        // Compute the weight w[i], its mirror is the same value
        w[i] =
            w[n - 1 - i] = T(2.0) / ((1.0 - x[i] * x[i]) * dpn * dpn);
    } // end for
    for (int i = 0; i < n; ++i)
    {
        quadrature[i].first(0) = x[i];
        quadrature[i].second = w[i];
    }
}

template <typename T>
int QuadratureRule<T>::NumOfQuadrature() const
{
    return _size;
}

template <typename T>
void QuadratureRule<T>::MapToQuadrature(const CoordinatePair &range, QuadList &quadrature) const
{
    quadrature.resize(0);
    int d = range.first.size();
    std::vector<int> indexes(d, 0);
    std::vector<int> endPerIndex(d);
    int space = 1;
    std::pair<Coordinate, Coordinate> temp;
    temp.first.resize(d);
    temp.second.resize(d);
    for (int i = 0; i != d; ++i)
    {
        if (range.first(i) == range.second(i))
        {
            endPerIndex[i] = 0;
            temp.first(i) = range.first(i);
            space *= 1;
        }
        else
        {
            endPerIndex[i] = _size;
            space *= endPerIndex[i];
        }
    }
    quadrature.reserve(space);
    std::function<void(std::vector<int> &, const std::vector<int> &, int)> recursive;
    recursive = [this, &quadrature, &temp, &range, &recursive](std::vector<int> &indexes,
                                                               const std::vector<int> &endPerIndex, int direction) {
        if (direction == indexes.size())
        {
            Quadrature tmp{temp.first, temp.second.prod()};
            quadrature.push_back(std::move(tmp));
        }
        else
        {
            if (endPerIndex[direction] == 0)
            {
                temp.second(direction) = 1; ///need verify
                recursive(indexes, endPerIndex, direction + 1);
            }
            else
            {
                for (indexes[direction] = 0; indexes[direction] != endPerIndex[direction]; indexes[direction]++)
                {
                    T length = abs(range.first(direction) - range.second(direction)) / 2;
                    T middle = (range.first(direction) + range.second(direction)) / 2;
                    temp.first(direction) = _quadrature[indexes[direction]].first(0) * length + middle;
                    temp.second(direction) = _quadrature[indexes[direction]].second * length;
                    recursive(indexes, endPerIndex, direction + 1);
                }
            } //buggy, need more tests.
        }
    };
    recursive(indexes, endPerIndex, 0);
}

template class QuadratureRule<long double>;
template class QuadratureRule<double>;
template class QuadratureRule<float>;
template class QuadratureRule<boost::multiprecision::mpf_float_50>;
template class QuadratureRule<boost::multiprecision::mpf_float_100>;
template class QuadratureRule<boost::multiprecision::mpf_float_1000>;