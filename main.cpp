#include <iostream>
#include "KnotVector.h"
using namespace std;
int main() {
    KnotVector<double> a({0,0,0,.5,.5,1,1,1});
    a.printKnotVector();
    a.printUnique();
    cout<<a.GetDegree()<<endl;
    KnotVector<double> b({{-1,4},{-.5,2},{.4,3},{1,4}});
    b.printKnotVector();
    b.printUnique();
    cout<<b.GetDegree()<<endl;
    KnotVector<double> c=a;
    c.Insert(.6);
    c.printKnotVector();
    c.printUnique();
    return 0;
}