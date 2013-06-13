#ifndef SEMISUPERVISED_COST_MODULE_H
#define SEMISUPERVISED_COST_MODULE_H

#include "CostModule.h"

template <class T>
class SemisupervisedCostModule : public CostModule<T>
{	
	std::shared_ptr<CostModule> supervised_cost_module_;
	std::shared_ptr<CostModule> unsupervised_cost_module_;
	double supervised_lambda_;
	double unsupervised_lambda_;

	virtual double sub_GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, double lambda);
	
	virtual void sub_bprop(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, Tensor<T>& output_gradients_buffer, double lambda);

	bool HasLabel(T* arr, size_t N)
	{
		bool label_found = false;
		for (size_t i=0; i<N; i++)
			if (arr[i] == 1)
			{
				label_found = true;
				break;
			}
		return label_found;
	}

public:

	SemisupervisedCostModule(std::shared_ptr<CostModule> supervised_cost_module, std::shared_ptr<CostModule> unsupervised_cost_module, 
		double supervised_lambda = 1, double unsupervised_lambda = 1 ) : CostModule(), supervised_cost_module_(supervised_cost_module),
		unsupervised_cost_module_(unsupervised_cost_module), supervised_lambda_(supervised_lambda), unsupervised_lambda_(unsupervised_lambda)
	{

	}

};

template <class T>
double SemisupervisedCostModule<T>::sub_GetCost(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
		const std::vector<T>& importance_weights, bool normalize_by_importance, double lambda)
{
	double importance_sum = 0;
	size_t minibatch_size = net_output.GetDimensionSize(net_output.NumDimensions()-1);
	if (normalize_by_importance)
		for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
			importance_sum += importance_weights[sample_ind];
	else
		importance_sum = 1;
	
	std::vector<size_t> sample_dims = net_output.GetDimensions();
	sample_dims.pop_back();
	sample_dims.push_back(1);
	Tensor<T> sample_net_output_tensor(sample_dims);
	Tensor<T> sample_expected_output_tensor(sample_dims);
	std::vector<T> sample_importance(1,1);
	double cost = 0;
	size_t num_features = net_output.Numel() / minibatch_size;
	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
	{
		size_t sample_offset = num_features*sample_ind;
		
		// we could use SetDataPtr and not copy the data, but I did not find the right way to do this
		for (size_t i=0; i<sample_net_output_tensor.Numel(); i++)
			sample_net_output_tensor[i] = net_output[sample_offset+i];

		for (size_t i=0; i<sample_expected_output_tensor.Numel(); i++)
			sample_expected_output_tensor[i] = expected_output[sample_offset+i];

		if ( HasLabel(sample_expected_output_tensor.GetStartPtr(), num_features) )
			cost += supervised_lambda_*supervised_cost_module_->GetCost(sample_net_output_tensor, sample_expected_output_tensor, 
				sample_importance, false, importance_weights[sample_ind]);
		else
			cost += unsupervised_lambda_*unsupervised_cost_module_->GetCost(sample_net_output_tensor, sample_expected_output_tensor, 
				sample_importance, false, importance_weights[sample_ind]);
	}

	return lambda*cost/importance_sum;
}

template <class T>
void SemisupervisedCostModule<T>::sub_bprop(const Tensor<T>& net_output, const Tensor<T>& expected_output, 
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
	
	double inv_importance_sum = 1/importance_sum;
	std::vector<size_t> sample_dims = net_output.GetDimensions();
	sample_dims.pop_back();
	sample_dims.push_back(1);
	Tensor<T> sample_net_output_tensor(sample_dims);
	Tensor<T> sample_expected_output_tensor(sample_dims);
	std::vector<T> sample_importance(1,1);
	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
	{
		size_t sample_offset = num_features*sample_ind;

		// we could use SetDataPtr and not copy the data, but I did not find the right way to do this
		for (size_t i=0; i<sample_net_output_tensor.Numel(); i++)
			sample_net_output_tensor[i] = net_output[sample_offset+i];

		for (size_t i=0; i<sample_expected_output_tensor.Numel(); i++)
			sample_expected_output_tensor[i] = expected_output[sample_offset+i];

		std::shared_ptr< Tensor<T> > sample_gradients_tensor_ptr;
		if ( HasLabel(sample_expected_output_tensor.GetStartPtr(), num_features) )
			sample_gradients_tensor_ptr = supervised_cost_module_->bprop(sample_net_output_tensor, 
				sample_expected_output_tensor, sample_importance, false, lambda*supervised_lambda_*inv_importance_sum*importance_weights[sample_ind]);
		else
			sample_gradients_tensor_ptr = unsupervised_cost_module_->bprop(sample_net_output_tensor, 
				sample_expected_output_tensor, sample_importance, false, lambda*unsupervised_lambda_*inv_importance_sum*importance_weights[sample_ind]);
		Tensor<T>& sample_gradients_tensor = *sample_gradients_tensor_ptr;
		for (size_t i=0; i<num_features; i++)
			output_gradients_buffer[sample_offset+i] = sample_gradients_tensor[i];
	}
}

#endif