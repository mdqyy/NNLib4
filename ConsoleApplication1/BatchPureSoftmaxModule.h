#ifndef BATCH_PURE_SOFTSIGN_H
#define BATCH_PURE_SOFTSIGN_H

#include <limits>
#include "Module.h"
#include "CashedTensor.h"
#include "my_math.h"

// Softmax takes exp() of its inputs before turning them to probabilities
// this module does not take exp(). In addition it takes softmax over the batch dimensions 
template <class ParamsType>
class BatchPureSoftmaxModule : public Module<ParamsType>
{
	ParamsType EPS_;
	CashedTensor<ParamsType> partition_values_buffer_;
	CashedTensor<ParamsType> weighted_gradients_buffer_;
public:

	BatchPureSoftmaxModule( std::string name) : Module<ParamsType>(name), EPS_(std::numeric_limits<ParamsType>::epsilon())
	{
	}

	virtual std::string GetType() const
	{
		return "BatchPureSoftmaxModule";
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
void BatchPureSoftmaxModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > BatchPureSoftmaxModule<ParamsType>::Create(IOTreeNode& data)
{
	return std::shared_ptr< Module< ParamsType> >( new BatchPureSoftmaxModule<ParamsType>(data.attributes().GetEntry( "Name" ) ));
}

template <class ParamsType>
void BatchPureSoftmaxModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& output_tensor = *output;
	size_t minibatch_size = input_tensor.GetDimensionSize(input_tensor.NumDimensions()-1);
	size_t num_features = input_tensor.Numel() / minibatch_size;
	
	partition_values_buffer_.Update(input->GetDimensions());
	Tensor<ParamsType>& partition_values_tensor = *partition_values_buffer_();
	partition_values_tensor.SetZeros();
	for (size_t i=0; i<num_features; i++)
		partition_values_tensor[i] = EPS_;
	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
	{
		size_t sample_offset = num_features*sample_ind;
		for (size_t i=0; i<num_features; i++)
			partition_values_tensor[i] += input_tensor[sample_offset+i];
	}

	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
	{
		size_t sample_offset = num_features*sample_ind;
		for (size_t i = 0; i<num_features; i++)
			output_tensor[sample_offset+i] = input_tensor[sample_offset+i]/partition_values_tensor[i];
	}
}

template <class ParamsType>
void BatchPureSoftmaxModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
	std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
	const std::vector<ParamsType>& samples_importances)
{	
	// partition_values_buffer_ already contains the partition values
	Tensor<ParamsType>& partition_values_tensor = *partition_values_buffer_();

	Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& output_tensor = *output;
	Tensor<ParamsType>& input_gradients_tensor = *input_gradients;
	Tensor<ParamsType>& output_gradients_tensor = *output_gradients;
	size_t minibatch_size = output_tensor.GetDimensionSize(output_tensor.NumDimensions()-1);
	size_t num_features = output_tensor.Numel() / minibatch_size;

	weighted_gradients_buffer_.Update(input->GetDimensions());
	Tensor<ParamsType>& weighted_gradients_tensor = *weighted_gradients_buffer_();
	weighted_gradients_tensor.SetZeros();
	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
	{
		size_t sample_offset = num_features*sample_ind;
		for (size_t i=0; i<num_features; i++)
			weighted_gradients_tensor[i] += output_tensor[sample_offset+i]*output_gradients_tensor[sample_offset+i]/partition_values_tensor[i];
	}

	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
	{
		size_t sample_offset = num_features*sample_ind;
		for (size_t i = 0; i<num_features; i++)
			input_gradients_tensor[sample_offset+i] = output_gradients_tensor[sample_offset+i]/partition_values_tensor[i] - weighted_gradients_tensor[i];
	}
}

#endif