#include "Utility.hpp"
#include <algorithm>
std::map<int, int> Accessory::IndicesInverseMap( const std::vector<int>& forward_map )
{
    std::map<int, int> inverse_map;
    for ( int i = 0; i < forward_map.size(); ++i )
    {
        ASSERT( inverse_map.find( forward_map[i] ) == inverse_map.end(), "Given forward map is not available for inverse.\n" );
        inverse_map[forward_map[i]] = i;
    }
    return inverse_map;
}

std::vector<int> Accessory::IndicesIntersection( const std::vector<int>& indices_a, const std::vector<int>& indices_b )
{
    std::vector<int> res;
    std::set_intersection( indices_a.begin(), indices_a.end(), indices_b.begin(), indices_b.end(), std::back_inserter( res ) );
    return res;
}

std::vector<int> Accessory::IndicesDifferentiation( const std::vector<int>& indices_a, const std::vector<int>& indices_b )
{
    std::vector<int> res;
    std::set_difference( indices_a.begin(), indices_a.end(), indices_b.begin(), indices_b.end(), std::back_inserter( res ) );
    return res;
}

std::vector<int> Accessory::IndicesUnion( const std::vector<int>& indices_a, const std::vector<int>& indices_b )
{
    std::vector<int> res;
    std::set_union( indices_a.begin(), indices_a.end(), indices_b.begin(), indices_b.end(), std::back_inserter( res ) );
    return res;
}

std::vector<int> Accessory::NClosestDof( const std::vector<int>& dofs, int target_dof, int n )
{
    ASSERT( n <= dofs.size(), "Requested dofs are larger than provided dofs.\n" );
    ASSERT( n > 0, "Requested dofs are invalid.\n" );
    auto it = std::find( dofs.begin(), dofs.end(), target_dof );
    ASSERT( it != dofs.end(), "Target dof is not listed in dofs" );
    std::vector<std::pair<int, int>> dist_to_dof;
    for ( const auto& i : dofs )
    {
        dist_to_dof.emplace_back( std::make_pair( std::abs( i - target_dof ), i ) );
    }
    std::sort( dist_to_dof.begin(), dist_to_dof.end(), []( const auto& a, const auto& b ) { return a.first <= b.first; } );
    std::vector<int> result;
    for ( auto it = dist_to_dof.begin(); it != dist_to_dof.begin() + n; ++it )
    {
        result.push_back( it->second );
    }
    std::sort( result.begin(), result.end() );
    return result;
}