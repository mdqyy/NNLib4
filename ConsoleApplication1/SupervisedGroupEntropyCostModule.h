#ifndef SUPERVISED_GROUP_ENTROPY_COST_MODULE_H
#define SUPERVISED_GROUP_ENTROPY_COST_MODULE_H

#include "GroupEntropyCostModule.h"
#include "SupervisedFeatureGroupProvider.h"

// supports only one vs all for labels with one 1 and others zero
template <class T>
class SupervisedGroupEntropyCostModule : public GroupEntropyCostModule<T>
{
	SupervisedFeatureGroupProvider<T> group_provider;
	virtual FeatureGroups GetGroups(const Tensor<T>* net_output, const Tensor<T>*  labels, size_t feature_ind) const
	{
		return group_provider.GetGroups(net_output, labels, feature_ind);
	}

	virtual bool GroupsAreFeatureIndependent() const
	{
		return group_provider.IsFeatureIndependent();
	}
};

#endif