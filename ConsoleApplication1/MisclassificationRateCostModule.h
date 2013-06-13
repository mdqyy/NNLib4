#ifndef MISCLASSIFICATION_RATE_COST_MODULE_H
#define MISCLASSIFICATION_RATE_COST_MODULE_H

#include "CostModule.h"
#include <algorithm>

// for each output the maximum element is assigned 1 others zero
template <class T>
class MisclassificationRateCostModule : public CostModule<T>
{
	const double eps;
	double log2_;

	double log2(double x)
	{
		return (double)(std::log(x) / log2_);
	}

public:

	MisclassificationRateCostModule() : CostModule(), eps(0.000000001), log2_(std::log(2))
	{
	}

	virtual double sub_GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, double lambda);
	
	virtual void sub_bprop(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, Tensor<T>& output_gradients_buffer, double lambda);
};

template <class T>
double MisclassificationRateCostModule<T>::sub_GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
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
		auto max_element_ind = std::distance(net_output.GetStartPtr()+offset, 
			std::max_element(net_output.GetStartPtr()+offset, net_output.GetStartPtr()+offset+num_features));
		
		cost += importance_weights[sample_ind]*static_cast<double>(expected_output[offset+max_element_ind] != 1);
	}

	return lambda*cost/importance_sum;
}

template <class T>
void MisclassificationRateCostModule<T>::sub_bprop(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, Tensor<T>& output_gradients_buffer, double lambda)
{
	throw "Not supported";
}

#endif