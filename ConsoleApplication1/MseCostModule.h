#ifndef MSE_COST_MODULE_H
#define MSE_COST_MODULE_H

#include "CostModule.h"
#include "my_math.h"

template <class T>
class MseCostModule : public CostModule<T>
{
public:

	MseCostModule() : CostModule()
	{
	}

	virtual double sub_GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, double lambda);
	
	virtual void sub_bprop(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, Tensor<T>& output_gradients_buffer, double lambda);
};

template <class T>
double MseCostModule<T>::sub_GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
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

	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
	{
		size_t offset = num_features*sample_ind;
		for (size_t feature_ind = 0; feature_ind<num_features; feature_ind++)
		{
			size_t feature_offset = offset+feature_ind;
			cost += lambda*importance_weights[sample_ind]*sqr(net_output[feature_offset]-expected_output[feature_offset]) / 2;
		}
	}

	return cost/importance_sum;
}

template <class T>
void MseCostModule<T>::sub_bprop(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, Tensor<T>& output_gradients_buffer, double lambda)
{
	double cost = 0;
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
		for (size_t feature_ind = 0; feature_ind<num_features; feature_ind++)
		{
			size_t feature_offset = offset+feature_ind;
			output_gradients_buffer[feature_offset] = static_cast<T>(lambda*importance_weights[sample_ind] / importance_sum * 
				( net_output[feature_offset]-expected_output[feature_offset] ));
		}
	}
}

#endif