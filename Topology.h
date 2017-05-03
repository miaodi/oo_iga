//
// Created by di miao on 2017/5/3.
//

#ifndef OO_IGA_TOPOLOGY_H
#define OO_IGA_TOPOLOGY_H
#include<eigen3/Eigen/Dense>


template<unsigned d,typename T>
class element{
    using Point = Eigen::Matrix<T,d,1>;
protected:
    unsigned _dimension;

    std::pair<Point,Point> _
};
class Topology {

};


#endif //OO_IGA_TOPOLOGY_H
