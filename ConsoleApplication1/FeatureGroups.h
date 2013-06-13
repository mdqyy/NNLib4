#ifndef GROUPS_H
#define GROUPS_H

#include <vector>

class FeatureGroups
{
	std::vector< std::vector<size_t> > groups;
public:
	
	std::vector<size_t>& GetGroup(size_t ind)
	{
		return groups[ind];
	}
	
	const std::vector<size_t>& GetGroup(size_t ind) const
	{
		return groups[ind];
	}
	
	void AddGroup( const std::vector<size_t>& group)
	{
		groups.push_back(group);
	}

	size_t size() const
	{
		return groups.size();
	}
};

#endif