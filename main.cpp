#include <iostream>
#include<eigen3/Eigen/Dense>
#include "KnotVector.h"
#include "BsplineBasis.h"
#include "TensorBsplineBasis.h"
#include "PhyTensorBsplineBasis.h"
#include "MultiArray.h"

using namespace Eigen;
using namespace std;
using namespace Accessory;

int main() {
    KnotVector<double> a({0, 0, 0, 1, 1, 1});
    Vector2d point1(0, 0);
    Vector2d point2(1, 1);
    Vector2d point3(2, 2);
    Matrix<Matrix<double, 2, 1>, Dynamic, 1> points(3);
    points << point1, point2, point3;
    degreeElevate<double, 2>(1, a, points);
    for(int i=0;i<points.rows();i++)
        cout<<points(i).transpose()<<endl;
    return 0;
}