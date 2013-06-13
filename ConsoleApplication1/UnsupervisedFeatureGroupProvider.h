#ifndef UNSUPERVISED_FEATURE_GROUP_PROVIDER_H
#define UNSUPERVISED_FEATURE_GROUP_PROVIDER_H

#include <vector>
#include "Tensor.h"
#include "my_math.h"
#include "IGroupEntropyFeatureGroupProvider.h"

template <class T>
class UnsupervisedFeatureGroupProvider: public IGroupEntropyFeatureGroupProvider<T>
{
	size_t num_groups_;
public:

	UnsupervisedFeatureGroupProvider(size_t num_groups) : num_groups_(num_groups)
	{

	}

	virtual bool IsFeatureIndependent() const
	{
		return false;
	}

	size_t GetNumGroups() const
	{
		return num_groups_;
	}

	// ptr is used to allow nullptr
	virtual FeatureGroups GetGroups( const Tensor<T>* net_output, const Tensor<T>*  labels, size_t feature_ind) const
	{
		assert( net_output != nullptr);

		size_t batch_size = net_output->GetDimensionSize(net_output->NumDimensions()-1);
		size_t num_features = net_output->Numel() / batch_size;

		FeatureGroups feature_groups;
		std::vector<T> feature_values(batch_size);
		for (size_t i=0; i<batch_size; i++)
			feature_values[i] = (*net_output)[i*num_features+feature_ind];
		std::vector<size_t> ordered_inds = GetOrderedIndices(feature_values);

		std::vector<size_t> batch_sizes = GetEqualSplitBatchSizes(batch_size, num_groups_);
		size_t offset = 0;
		// group largest first
		for (int group_ind=num_groups_-1; group_ind>=0; group_ind--)
		{
			std::vector<size_t> group(batch_sizes[group_ind]);
			for (size_t i=0; i< batch_sizes[group_ind]; i++)
				group[i] = ordered_inds[offset+i];
			feature_groups.AddGroup(group);
			offset += batch_sizes[group_ind];
		}

		return feature_groups;
	}
};

#endif