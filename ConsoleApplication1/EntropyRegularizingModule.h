#ifndef ENTROPY_REGULARIZING_MODULE_H
#define ENTROPY_REGULARIZING_MODULE_H

#include <iostream>
#include "Module.h"
#include "Converter.h"
#include "UnsupervisedGroupEntropyCostModule.h"
#include "BatchPureSoftmaxModule.h"

template <class ParamsType>
class EntropyRegularizingModule : public Module<ParamsType>
{
	double lambda_;
	size_t num_groups_;
	BatchPureSoftmaxModule<ParamsType> batch_softmax_module;
	UnsupervisedGroupEntropyCostModule<ParamsType> cost_module;
	Tensor<ParamsType> empty_tensor;

	// watch statistics of module
	double cost_estimate;
	double decay;
	size_t modulo;
	size_t show_modulo;
	size_t current_iteration;

public:

	EntropyRegularizingModule( std::string name, size_t num_groups, double lambda ) : Module<ParamsType>(name), 
		num_groups_(num_groups), lambda_(lambda), cost_module(num_groups_), batch_softmax_module("default"), 
		cost_estimate(-1), decay(0.95), modulo(25), show_modulo(250), current_iteration(-1)
	{
	}

	virtual std::string GetType() const
	{
		return "EntropyRegularizingModule";
	}
	
	virtual bool Equals(const Module<ParamsType>& module) const
	{
		if (!(module.GetType() == GetType()) && (module.GetName() == GetName()) )
			return false;
		
		const EntropyRegularizingModule<ParamsType>* other_module = static_cast< const EntropyRegularizingModule<ParamsType>* >( &module );
		return (lambda_ == other_module->lambda_) && (num_groups_ == other_module->num_groups_);
	}
	
	static std::shared_ptr< Module< ParamsType> > Create(IOTreeNode& data);

	virtual bool AlocateOutputBuffer() const
	{
		return false;
	}

	virtual bool AlocateInputGradientsBuffer() const
	{
		return true;
	}

	virtual double GetCost(const std::vector<ParamsType>& samples_importances)
	{
		std::shared_ptr< Tensor<ParamsType> > softmax_output = batch_softmax_module.predict_fprop(GetInputBuffer());
		return cost_module.GetCost( *softmax_output, empty_tensor, samples_importances, false, lambda_);
	}

protected:
	virtual void sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
	{
		output = input;
	}

	virtual void sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
		const std::vector<ParamsType>& samples_importances)
	{
		std::shared_ptr< Tensor<ParamsType> > softmax_output = batch_softmax_module.predict_fprop(input);

		if (current_iteration == -1 || current_iteration % modulo == 0)
		{
			size_t batch_size = input->GetDimensionSize(input->NumDimensions()-1);
			double cost = cost_module.GetCost( *softmax_output, empty_tensor, samples_importances, false, 1) / batch_size;
			if (current_iteration == -1)
			{
				cost_estimate = cost;
				current_iteration++;
			}
			else
				cost_estimate = decay*cost_estimate+(1-decay)*cost;
			if (current_iteration % show_modulo == 0)
				std::cout<<"Entropy equals "<<cost_estimate<<std::endl;
		}
		current_iteration++;

		std::shared_ptr< Tensor<ParamsType> > entropy_cost_gradients = 
			cost_module.bprop(*softmax_output, empty_tensor, samples_importances, false, lambda_);
		std::shared_ptr< Tensor<ParamsType> > softmax_gradients = 
			batch_softmax_module.bprop(entropy_cost_gradients, samples_importances);
		
		Tensor<ParamsType>& softmax_gradients_tensor = *softmax_gradients;
		Tensor<ParamsType>& output_gradients_tensor = *output_gradients;
		Tensor<ParamsType>& input_gradients_tensor = *input_gradients;
		size_t numel = input->Numel();
		for (size_t i = 0; i<numel; i++)
			input_gradients_tensor[i] = output_gradients_tensor[i];
		for (size_t i = 0; i<numel; i++)
			input_gradients_tensor[i] += softmax_gradients_tensor[i];
	}

	virtual void sub_GetState(IOTreeNode& node) const;
};

template <class ParamsType>
void EntropyRegularizingModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
	node.attributes().AppendEntry( "lambda", std::to_string(lambda_) );
	node.attributes().AppendEntry( "num_groups", std::to_string(num_groups_) );
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > EntropyRegularizingModule<ParamsType>::Create(IOTreeNode& node)
{
	double lambda = Converter::ConvertTo<double>( node.attributes().GetEntry( "lambda" ) );
	size_t num_groups = Converter::ConvertTo<size_t>( node.attributes().GetEntry( "num_groups" ) );

	return std::shared_ptr< Module< ParamsType> >( new EntropyRegularizingModule<ParamsType>( node.attributes().GetEntry( "Name" ), num_groups, lambda ));
}

#endif