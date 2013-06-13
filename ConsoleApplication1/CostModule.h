#ifndef COST_MODULE_H
#define COST_MODULE_H

#include <vector>
#include "Tensor.h"
#include "Module.h"

template <class T>
class CostModule
{
private:
	std::vector<T> output_gradients_buffer_data;
	std::shared_ptr< Tensor<T> > output_gradients_buffer_;

	virtual void sub_bprop(const Tensor<T>& output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, Tensor<T>& output_gradients_buffer,double lambda) = 0;

	void UpdateCash(const Tensor<T>& expected_output);
	
	virtual double sub_GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, double lambda) = 0;

public:

	CostModule() : output_gradients_buffer_( std::shared_ptr< Tensor<T> >(new Tensor<T>(0, std::vector<size_t>())) )
	{
	}

	virtual double GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, double lambda=1);

	std::shared_ptr< Tensor<T> > bprop(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, double lambda=1);

	virtual ~CostModule()
	{
	}
};

template <class T>
std::shared_ptr< Tensor<T> > CostModule<T>::bprop(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
	const std::vector<T>& importance_weights, bool normalize_by_importance, double lambda=1)
{
	UpdateCash(net_output);
	output_gradients_buffer_->SetZeros();
	sub_bprop(net_output, expected_output, importance_weights, normalize_by_importance, *output_gradients_buffer_, lambda);
	return output_gradients_buffer_;
}

template <class T>
double CostModule<T>::GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, double lambda=1)
{
	size_t minibatch_size = net_output.GetDimensionSize(net_output.NumDimensions()-1);
	assert(minibatch_size == importance_weights.size());

	return sub_GetCost(net_output, expected_output, importance_weights, normalize_by_importance, lambda);
}

template <class T>
void CostModule<T>::UpdateCash(const Tensor<T>& expected_output)
{
	if ( expected_output.GetDimensions() != output_gradients_buffer_->GetDimensions() )
	{
		std::vector<size_t> output_dims = expected_output.GetDimensions();
		if ( Tensor<T>::Numel(output_dims) > output_gradients_buffer_data.size())
			output_gradients_buffer_data = std::vector<T>(Tensor<T>::Numel(output_dims));
		output_gradients_buffer_ = std::shared_ptr< Tensor<T> >(new Tensor<T>(output_gradients_buffer_data.data(), output_dims));
	}
}

#endif