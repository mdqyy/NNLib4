#ifndef IGROUP_ENTROPY_FEATURE_GROUP_PROVIDER_H
#define IGROUP_ENTROPY_FEATURE_GROUP_PROVIDER_H

#include <memory>
#include <vector>
#include "Tensor.h"
#include "FeatureGroups.h"

template <class T>
class IGroupEntropyFeatureGroupProvider
{
public:
	// ptr is used to allow nullptr
	virtual FeatureGroups GetGroups(const Tensor<T>* net_output, const Tensor<T>* labels, size_t feature_ind) const = 0;

	virtual bool IsFeatureIndependent() const = 0;

	virtual ~IGroupEntropyFeatureGroupProvider()
	{
	}
};

#endif