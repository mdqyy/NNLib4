#ifndef LOGISTIC_COST_MODULE_H
#define LOGISTIC_COST_MODULE_H

#include "CostModule.h"

template <class T>
class LogisticCostModule : public CostModule<T>
{
	const double eps;
	double log2_;

	double log2(double x)
	{
		return (double)(std::log(x) / log2_);
	}

public:

	LogisticCostModule() : CostModule(), eps(0.000000001), log2_(std::log(2))
	{
	}

	virtual double sub_GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, double lambda);
	
	virtual void sub_bprop(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, Tensor<T>& output_gradients_buffer, double lambda);
};

template <class T>
double LogisticCostModule<T>::sub_GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
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
			cost += importance_weights[sample_ind]*(-expected_output[feature_offset]*log2(net_output[feature_offset]+eps)
				-(1-expected_output[feature_offset])*log2(1-net_output[feature_offset]+eps));
		}
	}

	return lambda*cost/importance_sum;
}

template <class T>
void LogisticCostModule<T>::sub_bprop(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
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
			output_gradients_buffer[feature_offset] = static_cast<T>(lambda*importance_weights[sample_ind]/importance_sum*( (1-expected_output[feature_offset]) / 
				(1-net_output[feature_offset]+eps) - expected_output[feature_offset] / (net_output[feature_offset]+eps) ) / log2_);
		}
	}
}

#endif