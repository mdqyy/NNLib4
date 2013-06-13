#ifndef GROUP_ENTROPY_COST_MODULE_H
#define GROUP_ENTROPY_COST_MODULE_H

#include "CostModule.h"
#include <algorithm>
#include "my_math.h"
#include "FeatureGroups.h"

template <class T>
class GroupEntropyCostModule : public CostModule<T>
{
	const T EPS;
	const T log2_;

	virtual FeatureGroups GetGroups(const Tensor<T>* net_output, const Tensor<T>*  labels, size_t feature_ind) const = 0;
	virtual bool GroupsAreFeatureIndependent() const = 0;
	
	virtual double sub_GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, double lambda);
	
	virtual void sub_bprop(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, Tensor<T>& output_gradients_buffer, double lambda);

public:

	GroupEntropyCostModule() : CostModule(), EPS( std::numeric_limits<T>::epsilon() ), 
		log2_( static_cast<T>(std::log(2)) )
	{

	}

};

template <class T>
double GroupEntropyCostModule<T>::sub_GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, double lambda)
{
	double cost = 0;
	double importance_sum = 0;
	size_t minibatch_size = net_output.GetDimensionSize(net_output.NumDimensions()-1);
	size_t num_features = net_output.Numel() / minibatch_size;
	if (normalize_by_importance)
		for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
			importance_sum += importance_weights[sample_ind];
	else
		importance_sum = 1;
	
	FeatureGroups feature_groups = GetGroups( &net_output, &expected_output, 0);
	for (size_t feature_ind = 0; feature_ind<num_features; feature_ind++)
	{
		if (!GroupsAreFeatureIndependent())
			feature_groups = GetGroups( &net_output, &expected_output, feature_ind);
		for (size_t group_ind = 0; group_ind<feature_groups.size(); group_ind++)
		{
			std::vector<size_t>& group_inds = feature_groups.GetGroup(group_ind);
			T group_probability = EPS;
			for (size_t i=0; i<group_inds.size(); i++)
				group_probability += net_output[ group_inds[i]*num_features + feature_ind ];
			
			T group_log2 = -log2(group_probability);
			for (size_t i=0; i<group_inds.size(); i++)
			{
				size_t sample_ind = group_inds[i];
				size_t sample_feature_ind = sample_ind*num_features + feature_ind;
				cost += importance_weights[sample_ind] * net_output[sample_feature_ind]*group_log2;
			}
		}
	}

	return lambda*minibatch_size*cost/importance_sum/num_features;
}

template <class T>
void GroupEntropyCostModule<T>::sub_bprop(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, Tensor<T>& output_gradients_buffer, double lambda)
{
	double importance_sum = 0;
	size_t minibatch_size = net_output.GetDimensionSize(net_output.NumDimensions()-1);
	assert(minibatch_size == importance_weights.size());
	size_t num_features = net_output.Numel() / minibatch_size;
	if (normalize_by_importance)
		for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
			importance_sum += importance_weights[sample_ind];
	else
		importance_sum = 1;

	FeatureGroups feature_groups = GetGroups( &net_output, &expected_output, 0);
	for (size_t feature_ind = 0; feature_ind<num_features; feature_ind++)
	{
		if (!GroupsAreFeatureIndependent())
			feature_groups = GetGroups( &net_output, &expected_output, feature_ind);
		for (size_t group_ind = 0; group_ind<feature_groups.size(); group_ind++)
		{
			std::vector<size_t>& group_inds = feature_groups.GetGroup(group_ind);
			T group_probability = EPS;
			T weighted_probability = 0;
			for (size_t i=0; i<group_inds.size(); i++)
			{
				size_t sample_ind = group_inds[i];
				size_t sample_feature_ind = sample_ind*num_features + feature_ind;
				group_probability += net_output[ sample_feature_ind ];
				weighted_probability += net_output[ sample_feature_ind ] * importance_weights[sample_ind];
			}
			
			T group_log2 = -log2(group_probability);

			
			T c1 = minibatch_size*lambda/num_features/importance_sum;
			T c2 = weighted_probability / group_probability / log2_;
			for (size_t i=0; i<group_inds.size(); i++)
			{
				size_t sample_ind = group_inds[i];
				size_t sample_feature_ind = sample_ind*num_features + feature_ind;
				output_gradients_buffer[sample_feature_ind] = static_cast<T>( c1*(importance_weights[sample_ind]*group_log2 - c2 ) );
			}
		}
	}
}

#endif