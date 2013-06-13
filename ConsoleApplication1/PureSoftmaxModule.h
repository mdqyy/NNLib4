#ifndef PURE_SOFTMAX_MODULE_H
#define PURE_SOFTMAX_MODULE_H

#include "Module.h"
#include "my_math.h"

// Softmax takes exp() of its inputs before turning them to probabilities
// this module does not take exp()
template <class ParamsType>
class PureSoftmaxModule : public Module<ParamsType>
{
	ParamsType EPS;
public:

	PureSoftmaxModule( std::string name) : Module<ParamsType>(name), EPS(std::numeric_limits<ParamsType>::epsilon())
	{
	}

	virtual std::string GetType() const
	{
		return "PureSoftmaxModule";
	}
	
	virtual bool Equals(const Module<ParamsType>& module) const
	{
		return module.GetType() == GetType() && module.GetName() == GetName();
	}
	
	static std::shared_ptr< Module< ParamsType> > Create(IOTreeNode& data);

protected:
	virtual void sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);
	
	virtual void sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
		const std::vector<ParamsType>& samples_importances);

	virtual void sub_GetState(IOTreeNode& node) const;
};

template <class ParamsType>
void PureSoftmaxModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > PureSoftmaxModule<ParamsType>::Create(IOTreeNode& data)
{
	return std::shared_ptr< Module< ParamsType> >( new PureSoftmaxModule<ParamsType>(data.attributes().GetEntry( "Name" ) ));
}

template <class ParamsType>
void PureSoftmaxModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{	
	Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& output_tensor = *output;
	size_t minibatch_size = input_tensor.GetDimensionSize(input_tensor.NumDimensions()-1);
	size_t num_features = input_tensor.Numel() / minibatch_size;
	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
	{
		size_t sample_offset = num_features*sample_ind;
		ParamsType partition_val = EPS;
		for (size_t i=0; i<num_features; i++)
			partition_val += input_tensor[sample_offset+i];
		for (size_t i = 0; i<num_features; i++)
			output_tensor[sample_offset+i] = input_tensor[sample_offset+i]/partition_val;
	}
}

template <class ParamsType>
void PureSoftmaxModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
	std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
	const std::vector<ParamsType>& samples_importances)
{	
	Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& output_tensor = *output;
	Tensor<ParamsType>& input_gradients_tensor = *input_gradients;
	Tensor<ParamsType>& output_gradients_tensor = *output_gradients;
	size_t minibatch_size = output_tensor.GetDimensionSize(output_tensor.NumDimensions()-1);
	size_t num_features = output_tensor.Numel() / minibatch_size;

	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
	{
		size_t sample_offset = num_features*sample_ind;
		ParamsType partition_val = EPS;
		for (size_t i=0; i<num_features; i++)
			partition_val += input_tensor[sample_offset+i];

		ParamsType weighted_gradient = 0;
		for (size_t i=0; i<num_features; i++)
			weighted_gradient += output_tensor[sample_offset+i] / partition_val * output_gradients_tensor[sample_offset+i];

		for (size_t i = 0; i<num_features; i++)
			input_gradients_tensor[sample_offset+i] = output_gradients_tensor[sample_offset+i]/partition_val - weighted_gradient;
	}
}

#endif