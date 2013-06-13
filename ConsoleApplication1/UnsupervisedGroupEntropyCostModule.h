#ifndef UNSUPERVISED_GROUP_ENTROPY_COST_MODULE_H
#define UNSUPERVISED_GROUP_ENTROPY_COST_MODULE_H

#include "GroupEntropyCostModule.h"
#include "UnsupervisedFeatureGroupProvider.h"

template <class T>
class UnsupervisedGroupEntropyCostModule : public GroupEntropyCostModule<T>
{
	UnsupervisedFeatureGroupProvider<T> group_provider;
	virtual FeatureGroups GetGroups(const Tensor<T>* net_output, const Tensor<T>*  labels, size_t feature_ind) const
	{
		return group_provider.GetGroups(net_output, labels, feature_ind);
	}

	virtual bool GroupsAreFeatureIndependent() const
	{
		return group_provider.IsFeatureIndependent();
	}

public:

	UnsupervisedGroupEntropyCostModule(size_t num_groups) : GroupEntropyCostModule(), group_provider(num_groups)
	{

	}

	size_t GetNumGroups() const
	{
		return group_provider.GetNumGroups();
	}

};

#endif