//
// Created by miaodi on 26/12/2016.
//

#ifndef OO_IGA_TENSORBSPLINEBASIS_H
#define OO_IGA_TENSORBSPLINEBASIS_H

#include "BsplineBasis.h"

template<unsigned d, typename T=double>
class TensorBsplineBasis {
public:

    using vector=Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    TensorBsplineBasis();

    TensorBsplineBasis(BsplineBasis<T> *baseX);

    TensorBsplineBasis(BsplineBasis<T> *baseX, BsplineBasis<T> *baseY);

    TensorBsplineBasis(BsplineBasis<T> *baseX, BsplineBasis<T> *baseY, BsplineBasis<T> *baseZ);

    TensorBsplineBasis(const std::vector<KnotVector<T>> &KV);

    virtual ~TensorBsplineBasis() {
        for (unsigned i = 0; i < d; i++)
            delete _basis[i];
    };

    unsigned GetDegree(const unsigned i) const;

    unsigned GetDof() const;

    unsigned GetDof(const unsigned i) const;

    std::vector<unsigned> TensorIndex(const unsigned &m) const {
        ASSERT(m < GetDof(), "Input index is invalid.");
        std::vector<unsigned> ind(d);
        int mm = m;
        ///unsigned always >=0.
        for (int i = static_cast<int>(d - 1); i >= 0; --i) {
            ind[i] = mm % GetDof(i);
            mm -= ind[i];
            mm /= GetDof(i);
        }
        return ind;
    }

    unsigned Index(const std::vector<unsigned> &ts) const {
        unsigned index = 0;
        for (unsigned direction = 0; direction < d; ++direction) {
            index += ts[direction];
            if (direction != d - 1) { index *= GetDof(direction + 1); }
        }
        return index;
    }

    matrix Support(const unsigned &i) const {
        matrix res(static_cast<int>(d), 2);
        auto ti = TensorIndex(i);
        for (unsigned j = 0; j != d; ++j)
            res.row(j) = _basis[j]->Support(ti[j]);
        return res;
    }

    unsigned NumActive() const;

    unsigned NumActive(const unsigned &i) const;



    std::unique_ptr<std::vector<std::pair<unsigned, std::vector<std::vector<T>>>>>
    EvalTensor(const vector &u, const unsigned i = 0) const {
        std::unique_ptr<std::vector<std::pair<unsigned, std::vector<std::vector<T>>>>> result(
                new std::vector<std::pair<unsigned, std::vector<std::vector<T>>>>);
        std::unique_ptr<matrix> functionValue[d];
        unsigned deg[d];
        unsigned firstBasis[d];
        unsigned numActive[d];
        unsigned dof[d];
        for (unsigned direction = 0; direction != d; ++direction) {
            functionValue[direction] = _basis[direction]->Eval(static_cast<T>(u(direction)), i);
            std::cout << *(functionValue[direction]) << std::endl;
            deg[direction] = GetDegree(direction);
            firstBasis[direction] = _basis[direction]->FirstActive(static_cast<T>(u(direction)));
            numActive[direction] = _basis[direction]->NumActive();
            dof[direction] = GetDof(direction);
        }
        unsigned totActive = NumActive();
        for (unsigned it = 0; it != totActive; ++it) {
            unsigned index = 0;
            std::vector<unsigned> ind(d);
            int mm = it;
            for (int direction = static_cast<int>(d - 1); direction >= 0; --direction) {
                ///unsigned always >=0.
                ind[direction] = mm % numActive[direction];
                mm -= ind[direction];
                mm /= numActive[direction];
            }
            for (unsigned direction = 0; direction < d; ++direction) {
                index += (firstBasis[direction] + ind[direction]);
                if (direction != d - 1) { index *= dof[direction + 1]; }
            }
            result->emplace_back(std::make_pair(index, std::vector<std::vector<T>>()));
            for (unsigned p = 0; p != i + 1; ++p) {
                result->back().second.emplace_back(std::vector<T>());
                if (p == 0) {
                    T tmp = 1;
                    for (unsigned direction1 = 0; direction1 != d; ++direction1)
                        tmp *= (*functionValue[direction1])(0, ind[direction1]);
                    result->back().second.back().emplace_back(tmp);
                } else {
                    for (unsigned partialdirection = 0;
                         partialdirection != _derivativePattern[p - 1]->size(); ++partialdirection) {
                        T tmp = 1;
                        for (unsigned direction1 = 0; direction1 != d; ++direction1) {
                            unsigned pattern = (*_derivativePattern[p - 1])[partialdirection][direction1];
                            tmp *= (*functionValue[direction1])(pattern, ind[direction1]);
                        }
                        result->back().second.back().emplace_back(tmp);
                    }

                }

            }
        }
        return result;
    }

    T EvalSingle(const vector &u, const unsigned n, const std::vector<unsigned> i) {
        ASSERT((u.size() == d) && (i.size() == d), "Invalid input vector size.");
        auto tensorindex = TensorIndex(n);
        T result = 1;
        for (unsigned direction = 0; direction != d; ++direction) {
            result *= _basis[direction]->EvalSingle(u(direction), tensorindex[direction], i[direction]);
        }
        return result;
    }

    static std::unique_ptr<std::vector<std::vector<unsigned>>> PartialDerPattern(unsigned i) {
        std::vector<unsigned> kk(i);
        std::unique_ptr<std::vector<std::vector<unsigned>>> a(new std::vector<std::vector<unsigned>>);
        func(d, i, kk, 0, 0, a);
        return a;
    }

protected:
    BsplineBasis<T> *_basis[d];

    TensorBsplineBasis(const TensorBsplineBasis<d, T> &) {};

    TensorBsplineBasis operator=(const TensorBsplineBasis<d, T> &) {};

private:
    /// Recursion function for generating the partial derivative pattern. Don't know where to put it.
    static void func(unsigned D, unsigned i, std::vector<unsigned> &k, unsigned n, unsigned start,
                     std::unique_ptr<std::vector<std::vector<unsigned>>> &a) {
        if (n == i) {
            std::vector<unsigned> m;
            unsigned it = 0;
            for (unsigned it1 = 0; it1 < D; ++it1) {
                unsigned amount = 0;
                while (find(k.begin(), k.end(), it) != k.end()) {
                    amount++;
                    it++;
                }
                m.push_back(amount);
                it++;
            }
            a->push_back(m);
        } else {
            for (unsigned jj = start; jj < D + i - (i - n); ++jj) {
                k[n] = jj;
                func(D, i, k, n + 1, jj + 1, a
                );
            }
        }
    }

    std::vector<std::unique_ptr<std::vector<std::vector<unsigned>>>> _derivativePattern;
};

template<unsigned d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis() {
    for (unsigned i = 0; i < d; ++i)
        _basis[i] = nullptr;
}

template<unsigned d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(BsplineBasis<T> *baseX) {
    ASSERT(d == 1, "Invalid dimension.");
    _basis[0] = baseX;
    for (unsigned i = 1; i != 4; ++i)
        _derivativePattern.emplace_back(PartialDerPattern(i));

}

template<unsigned d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(BsplineBasis<T> *baseX, BsplineBasis<T> *baseY) {
    ASSERT(d == 2, "Invalid dimension.");
    _basis[0] = baseX;
    _basis[1] = baseY;
    for (unsigned i = 1; i != 4; ++i)
        _derivativePattern.emplace_back(PartialDerPattern(i));
}

template<unsigned d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(BsplineBasis<T> *baseX, BsplineBasis<T> *baseY, BsplineBasis<T> *baseZ) {
    ASSERT(d == 3, "Invalid dimension.");
    _basis[0] = baseX;
    _basis[1] = baseY;
    _basis[2] = baseZ;
    for (unsigned i = 1; i != 4; ++i)
        _derivativePattern.emplace_back(PartialDerPattern(i));
}

template<unsigned d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(const std::vector<KnotVector<T>> &knotVectors) {
    ASSERT(d == knotVectors.size(), "Invalid number of knot-vectors given.");
    for (unsigned i = 0; i != d; ++i)
        _basis[i] = new BsplineBasis<T>(knotVectors[i]);
    for (unsigned i = 1; i != 4; ++i)
        _derivativePattern.emplace_back(PartialDerPattern(i));
}

template<unsigned d, typename T>
unsigned TensorBsplineBasis<d, T>::GetDegree(const unsigned i) const {
    return _basis[i]->GetDegree();
}

template<unsigned d, typename T>
unsigned TensorBsplineBasis<d, T>::GetDof() const {
    unsigned dof = 1;
    for (unsigned i = 0; i != d; ++i)
        dof *= _basis[i]->GetDof();
    return dof;
}

template<unsigned d, typename T>
unsigned TensorBsplineBasis<d, T>::GetDof(const unsigned i) const {
    return _basis[i]->GetDof();
}

template<unsigned d, typename T>
unsigned TensorBsplineBasis<d, T>::NumActive(const unsigned &i) const {
    return _basis[i]->NumActive();
}

template<unsigned d, typename T>
unsigned TensorBsplineBasis<d, T>::NumActive() const {
    unsigned active = 1;
    for (unsigned i = 0; i != d; ++i)
        active *= _basis[i]->NumActive();
    return active;
}


#endif //OO_IGA_TENSORBSPLINEBASIS_H
