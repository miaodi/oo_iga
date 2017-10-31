//
// Created by miaodi on 07/05/2017.
//

#ifndef OO_IGA_QUADRATURERULE_H
#define OO_IGA_QUADRATURERULE_H

#include <vector>
#include <eigen3/Eigen/Dense>
#include <iostream>

template <typename T>
class QuadratureRule
{
  public:
    using Coordinate = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Quadrature = std::pair<Coordinate, T>;
    using QuadList = std::vector<Quadrature>;
    using CoordinatePair = std::pair<Coordinate, Coordinate>;

  public:
    QuadratureRule() = default;

    ~QuadratureRule() = default;

    QuadratureRule(int num) : _size(num)
    {
        LookupReference(_size, _quadrature);
    }

    void
    SetUpQuadrature(int num)
    {
        _size = num;
        LookupReference(_size, _quadrature);
    }

    static void
    LookupReference(int num, QuadList &quadrature);

    void
    MapToQuadrature(const CoordinatePair &range, QuadList &quadrature) const
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
                        T length = std::abs(range.first(direction) - range.second(direction)) / 2;
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

    int
    NumOfQuadrature() const
    {
        return _size;
    }

  private:
    QuadList _quadrature;
    int _size;
};

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
        quadrature[0].first(0) = 0.000000000000000000000000000000;
        quadrature[0].second = 2.000000000000000000000000000000;
        break;
    }
    case 2:
    {
        quadrature[0].first(0) = -0.577350269189625764509148780502;
        quadrature[1].first(0) = 0.577350269189625764509148780502;

        quadrature[0].second = 1.000000000000000000000000000000;
        quadrature[1].second = 1.000000000000000000000000000000;
        break;
    }

    case 3:
    {
        quadrature[0].first(0) = -0.774596669241483377035853079956;
        quadrature[1].first(0) = 0.000000000000000000000000000000;
        quadrature[2].first(0) = 0.774596669241483377035853079956;

        quadrature[0].second = 0.555555555555555555555555555556;
        quadrature[1].second = 0.888888888888888888888888888889;
        quadrature[2].second = 0.555555555555555555555555555556;
        break;
    }
    case 4:
    {
        quadrature[0].first(0) = -0.861136311594052575223946488893;
        quadrature[1].first(0) = -0.339981043584856264802665759103;
        quadrature[2].first(0) = 0.339981043584856264802665759103;
        quadrature[3].first(0) = 0.861136311594052575223946488893;

        quadrature[0].second = 0.347854845137453857373063949222;
        quadrature[1].second = 0.652145154862546142626936050778;
        quadrature[2].second = 0.652145154862546142626936050778;
        quadrature[3].second = 0.347854845137453857373063949222;
        break;
    }
    case 5:
    {
        quadrature[0].first(0) = -0.906179845938663992797626878299;
        quadrature[1].first(0) = -0.538469310105683091036314420700;
        quadrature[2].first(0) = 0.000000000000000000000000000000;
        quadrature[3].first(0) = 0.538469310105683091036314420700;
        quadrature[4].first(0) = 0.906179845938663992797626878299;

        quadrature[0].second = 0.236926885056189087514264040720;
        quadrature[1].second = 0.478628670499366468041291514836;
        quadrature[2].second = 0.568888888888888888888888888889;
        quadrature[3].second = 0.478628670499366468041291514836;
        quadrature[4].second = 0.236926885056189087514264040720;
        break;
    }
    case 6:
    {
        quadrature[0].first(0) = -0.932469514203152027812301554494;
        quadrature[1].first(0) = -0.661209386466264513661399595020;
        quadrature[2].first(0) = -0.238619186083196908630501721681;
        quadrature[3].first(0) = 0.238619186083196908630501721681;
        quadrature[4].first(0) = 0.661209386466264513661399595020;
        quadrature[5].first(0) = 0.932469514203152027812301554494;

        quadrature[0].second = 0.171324492379170345040296142173;
        quadrature[1].second = 0.360761573048138607569833513838;
        quadrature[2].second = 0.467913934572691047389870343990;
        quadrature[3].second = 0.467913934572691047389870343990;
        quadrature[4].second = 0.360761573048138607569833513838;
        quadrature[5].second = 0.171324492379170345040296142173;
        break;
    }
    case 7:
    {
        quadrature[0].first(0) = -0.949107912342758524526189684048;
        quadrature[1].first(0) = -0.741531185599394439863864773281;
        quadrature[2].first(0) = -0.405845151377397166906606412077;
        quadrature[3].first(0) = 0.000000000000000000000000000000;
        quadrature[4].first(0) = 0.405845151377397166906606412077;
        quadrature[5].first(0) = 0.741531185599394439863864773281;
        quadrature[6].first(0) = 0.949107912342758524526189684048;

        quadrature[0].second = 0.129484966168869693270611432679;
        quadrature[1].second = 0.279705391489276667901467771424;
        quadrature[2].second = 0.381830050505118944950369775489;
        quadrature[3].second = 0.417959183673469387755102040816;
        quadrature[4].second = 0.381830050505118944950369775489;
        quadrature[5].second = 0.279705391489276667901467771424;
        quadrature[6].second = 0.129484966168869693270611432679;
        break;
    }

    case 8:
    {
        quadrature[0].first(0) = -0.960289856497536231683560868569;
        quadrature[1].first(0) = -0.796666477413626739591553936476;
        quadrature[2].first(0) = -0.525532409916328985817739049189;
        quadrature[3].first(0) = -0.183434642495649804939476142360;
        quadrature[4].first(0) = 0.183434642495649804939476142360;
        quadrature[5].first(0) = 0.525532409916328985817739049189;
        quadrature[6].first(0) = 0.796666477413626739591553936476;
        quadrature[7].first(0) = 0.960289856497536231683560868569;

        quadrature[0].second = 0.101228536290376259152531354310;
        quadrature[1].second = 0.222381034453374470544355994426;
        quadrature[2].second = 0.313706645877887287337962201987;
        quadrature[3].second = 0.362683783378361982965150449277;
        quadrature[4].second = 0.362683783378361982965150449277;
        quadrature[5].second = 0.313706645877887287337962201987;
        quadrature[6].second = 0.222381034453374470544355994426;
        quadrature[7].second = 0.101228536290376259152531354310;
        break;
    }

    case 9:
    {
        quadrature[0].first(0) = -0.968160239507626089835576203;
        quadrature[1].first(0) = -0.836031107326635794299429788;
        quadrature[2].first(0) = -0.613371432700590397308702039;
        quadrature[3].first(0) = -0.324253423403808929038538015;
        quadrature[4].first(0) = 0.000000000000000000000000000;
        quadrature[5].first(0) = 0.324253423403808929038538015;
        quadrature[6].first(0) = 0.613371432700590397308702039;
        quadrature[7].first(0) = 0.836031107326635794299429788;
        quadrature[8].first(0) = 0.968160239507626089835576203;

        quadrature[0].second = 0.081274388361574411971892158111;
        quadrature[1].second = 0.18064816069485740405847203124;
        quadrature[2].second = 0.26061069640293546231874286942;
        quadrature[3].second = 0.31234707704000284006863040658;
        quadrature[4].second = 0.33023935500125976316452506929;
        quadrature[5].second = 0.31234707704000284006863040658;
        quadrature[6].second = 0.26061069640293546231874286942;
        quadrature[7].second = 0.18064816069485740405847203124;
        quadrature[8].second = 0.081274388361574411971892158111;
        break;
    }
    }
}

#endif //OO_IGA_QUADRATURERULE_H
