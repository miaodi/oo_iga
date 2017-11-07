//
// Created by miaodi on 08/06/2017.
//

#pragma once

template <int d, int N, typename T>
class Element;

template <int d, int N, typename T>
class Visitor
{
  public:
    Visitor(){};

    virtual void Visit(Element<d, N, T> *g) = 0;
};
