#ifndef SUPERVISED_FEATURE_GROUP_PROVIDER_H
#define SUPERVISED_FEATURE_GROUP_PROVIDER_H

#include <vector>
#include "Tensor.h"
#include "my_math.h"
#include "IGroupEntropyFeatureGroupProvider.h"

// supports only one vs all for labels with one 1 and others zero
template <class T>
class SupervisedFeatureGroupProvider: public IGroupEntropyFeatureGroupProvider<T>
{
public:
	
	virtual bool IsFeatureIndependent() const
	{
		return true;
	}

	// ptr is used to allow nullptr
	virtual FeatureGroups GetGroups( const Tensor<T>* net_output, const Tensor<T>*  labels, size_t feature_ind) const
	{
		assert( labels != nullptr);

		size_t batch_size = net_output->GetDimensionSize(net_output->NumDimensions()-1);
		size_t num_labels = labels->Numel() / batch_size;
		std::vector< std::vector< size_t > > labels_groups(num_labels);
		for (size_t batch_ind = 0; batch_ind<batch_size; batch_ind++)
		{
			size_t offset = batch_ind*num_labels;
			for (size_t i=0; i<num_labels; i++)
				if ( (*labels)[offset+i] != 0)
				{
					labels_groups[i].push_back(batch_ind);
					break;
				}
		}
		
		FeatureGroups feature_groups;
		for(size_t i=0; i<labels_groups.size(); i++)
			feature_groups.AddGroup(labels_groups[i]);

		return feature_groups;
	}
};

#endif