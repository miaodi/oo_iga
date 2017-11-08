#include "Utility.hpp"
#include <algorithm>
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

std::vector<int> Accessory::IndicesIntersection(const std::vector<int> &indices_a, const std::vector<int> &indices_b)
{
    std::vector<int> res;
    std::set_intersection(indices_a.begin(), indices_a.end(), indices_b.begin(), indices_b.end(), std::back_inserter(res));
    return res;
}

std::vector<int> Accessory::IndicesDifferentiation(const std::vector<int> &indices_a, const std::vector<int> &indices_b)
{
    std::vector<int> res;
    std::set_difference(indices_a.begin(), indices_a.end(), indices_b.begin(), indices_b.end(), std::back_inserter(res));
    return res;
}

std::vector<int> Accessory::IndicesUnion(const std::vector<int> &indices_a, const std::vector<int> &indices_b)
{
    std::vector<int> res;
    std::set_union(indices_a.begin(), indices_a.end(), indices_b.begin(), indices_b.end(), std::back_inserter(res));
    return res;
}