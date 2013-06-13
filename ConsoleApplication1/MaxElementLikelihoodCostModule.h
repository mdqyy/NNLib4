#ifndef MAX_ELEMENT_LIKELIHOOD_COST_MODULE_H
#define MAX_ELEMENT_LIKELIHOOD_COST_MODULE_H

#include <algorithm>
#include "CostModule.h"

template <class T>
class MaxElementLikelihoodCostModule : public CostModule<T>
{
	const T EPS_;
	const T log2_;
	
	virtual double sub_GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, double lambda);
	
	virtual void sub_bprop(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, Tensor<T>& output_gradients_buffer, double lambda);

public:

	MaxElementLikelihoodCostModule() : CostModule(), EPS_( std::numeric_limits<T>::epsilon() ), 
		log2_( static_cast<T>(std::log(2)) )
	{

	}

};

template <class T>
double MaxElementLikelihoodCostModule<T>::sub_GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, double lambda)
{
	double importance_sum = 0;
	size_t minibatch_size = net_output.GetDimensionSize(net_output.NumDimensions()-1);
	if (normalize_by_importance)
		for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
			importance_sum += importance_weights[sample_ind];
	else
		importance_sum = 1;
	
	double cost = 0;
	size_t num_features = net_output.Numel() / minibatch_size;
	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
	{
		size_t offset = num_features*sample_ind;
		size_t max_element_ind = std::distance( net_output.GetStartPtr()+offset, 
			std::max_element( net_output.GetStartPtr()+offset, net_output.GetStartPtr()+offset+num_features) );
		size_t feature_offset = offset+max_element_ind;
		double feature_probability = net_output[feature_offset];
		cost += -importance_weights[sample_ind]*log2(feature_probability+EPS_);
	}

	return lambda*cost/importance_sum;
}

template <class T>
void MaxElementLikelihoodCostModule<T>::sub_bprop(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
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
	
	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
	{
		size_t offset = num_features*sample_ind;
		double sample_importance = importance_weights[sample_ind]/importance_sum;
		size_t max_element_ind = std::distance( net_output.GetStartPtr()+offset, 
			std::max_element( net_output.GetStartPtr()+offset, net_output.GetStartPtr()+offset+num_features) );
		size_t feature_offset = offset+max_element_ind;
		double feature_probability = net_output[feature_offset];
		output_gradients_buffer[feature_offset] = static_cast<T>(-lambda*sample_importance / (feature_probability+EPS_) / log2_ );
	}
}

#endif