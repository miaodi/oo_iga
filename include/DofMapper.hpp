#pragma once

#include <map>
#include <iostream>

class DofMapper
{
  public:
    DofMapper() = default;

    void Insert(int ID, int Dof)
    {
        _IDToDof[ID] = Dof;
    }

    int StartingDof(int ID) const
    {
        int dof = 0;
        auto it = _IDToDof.begin();
        while (it != _IDToDof.end() && it->first != ID)
        {
            dof += it->second;
            it++;
        }
        return dof;
    }

    int TotalDof() const
    {
        int dof = 0;
        for (auto &i : _IDToDof)
        {
            dof += i.second;
        }
        return dof;
    }

    void PrintDofs() const
    {
        for (auto &i : _IDToDof)
        {
            std::cout << i.first << ", " << i.second << std::endl;
        }
    }

  protected:
    std::map<int, int> _IDToDof;
};