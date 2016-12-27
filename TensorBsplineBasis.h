//
// Created by miaodi on 26/12/2016.
//

#ifndef OO_IGA_TENSORBSPLINEBASIS_H
#define OO_IGA_TENSORBSPLINEBASIS_H

#include "BsplineBasis.h"

template<unsigned d, typename T>
class TensorBsplineBasis {
public:

    using vector=Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    TensorBsplineBasis();

    TensorBsplineBasis(BsplineBasis<T> *baseX);

    TensorBsplineBasis(BsplineBasis<T> *baseX, BsplineBasis<T> *baseY);

    TensorBsplineBasis(BsplineBasis<T> *baseX, BsplineBasis<T> *baseY, BsplineBasis<T> *baseZ);

    TensorBsplineBasis(const std::vector<KnotVector<T>> &KV);

    ~TensorBsplineBasis() {
        for (unsigned i = 0; i < d; i++)
            delete _basis[i];

    };


    unsigned GetDegree(const unsigned i) const;

    unsigned GetDof() const;

    unsigned GetDof(const unsigned i) const;

    vector TensorIndex(const unsigned &m) const {
        vector ind(d);
        int mm = m;
        ///unsigned always >=0.
        for (int i = static_cast<int>(d - 1); i >= 0; --i) {
            ind(i) = mm % GetDof(i);
            mm -= ind(i);
            mm /= GetDof(i);
        }
        return ind;
    }


    matrix support(const unsigned &i) const {
        matrix res(d, 2);
        matrix ti = TensorIndex(i);
        for (unsigned j = 0; j != d; ++j)
            res.row(j) = _basis[j]->support(ti(j));
        return res;
    }
///not finished yet
    std::unique_ptr<std::map<unsigned, std::vector<vector<T>>>> TensorEval(const vector &u, const unsigned i = 0) const {
        std::unique_ptr<std::map<unsigned, std::vector<vector<T>>>> result(new std::map<unsigned, std::vector<vector<T>>>);
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
        unsigned totActive = 1;
        for (unsigned it = 0; it != d; ++it)
            totActive *= numActive[it];
        for (unsigned it = 0; it != totActive; ++it) {
            unsigned index = 0;
            vector ind(d);
            int mm = it;
            for (int direction = static_cast<int>(d - 1); direction >= 0; --direction) {
                ///unsigned always >=0.
                ind(direction) = mm % numActive[direction];
                mm -= ind(direction);
                mm /= numActive[direction];
            }
            std::cout << ind << std::endl << std::endl;
            for (unsigned direction = 0; direction < d; ++direction) {
                index += (firstBasis[direction] + ind(direction));
                if (direction != d - 1) { index *= dof[direction+1]; }
            }
            std::cout << index << std::endl << std::endl;
            result->emplace(index,std::vector<vector<T>>());
            for(unsigned p = 0;p !=i+1;++p){
    

            }
        }


        return std::unique_ptr<std::map<unsigned int, std::vector<T>>>();
    }

protected:
    BsplineBasis<T> *_basis[d];

    TensorBsplineBasis(const TensorBsplineBasis<d, T> &) {};

    TensorBsplineBasis operator=(const TensorBsplineBasis<d, T> &) {};
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
}

template<unsigned d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(BsplineBasis<T> *baseX, BsplineBasis<T> *baseY) {
    ASSERT(d == 2, "Invalid dimension.");
    _basis[0] = baseX;
    _basis[1] = baseY;
}

template<unsigned d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(BsplineBasis<T> *baseX, BsplineBasis<T> *baseY, BsplineBasis<T> *baseZ) {
    ASSERT(d == 3, "Invalid dimension.");
    _basis[0] = baseX;
    _basis[1] = baseY;
    _basis[2] = baseZ;
}

template<unsigned d, typename T>
TensorBsplineBasis<d, T>::TensorBsplineBasis(const std::vector<KnotVector<T>> &knotVectors) {
    ASSERT(d == knotVectors.size(), "Invalid number of knot-vectors given.");
    for (unsigned i = 0; i != d; ++i)
        _basis[i] = new BsplineBasis<T>(knotVectors[i]);
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


#endif //OO_IGA_TENSORBSPLINEBASIS_H
