#include "Utility.hpp"

std::map<int, int> Accessory::IndicesInverseMap(const std::vector<int> &forward_map)
{
    std::map<int, int> inverse_map;
    for (int i = 0; i < forward_map.size(); ++i)
    {
        ASSERT(inverse_map.find(forward_map[i]) == inverse_map.end(),
               "Given forward map is not available for inverse.\n");
        inverse_map[forward_map[i]] = i;
    }
    return inverse_map;
}