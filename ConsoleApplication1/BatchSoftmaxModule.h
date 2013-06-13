#ifndef BATCH_SOFTSIGN_H
#define BATCH_SOFTSIGN_H

#include <limits>
#include <algorithm>
#include "Module.h"
#include "CashedTensor.h"
#include "my_math.h"

// Takes softmax over the batch dimensions 
template <class ParamsType>
class BatchSoftmaxModule : public Module<ParamsType>
{
	ParamsType EPS_;
	CashedTensor<ParamsType> partition_values_buffer_;
	CashedTensor<ParamsType> min_values_buffer_;
	CashedTensor<ParamsType> weighted_gradients_buffer_;
public:

	BatchSoftmaxModule( std::string name) : Module<ParamsType>(name), EPS_(std::numeric_limits<ParamsType>::epsilon())
	{
	}

	virtual std::string GetType() const
	{
		return "BatchSoftmaxModule";
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
void BatchSoftmaxModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > BatchSoftmaxModule<ParamsType>::Create(IOTreeNode& data)
{
	return std::shared_ptr< Module< ParamsType> >( new BatchSoftmaxModule<ParamsType>(data.attributes().GetEntry( "Name" ) ));
}

template <class ParamsType>
void BatchSoftmaxModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& output_tensor = *output;
	size_t minibatch_size = input_tensor.GetDimensionSize(input_tensor.NumDimensions()-1);
	size_t num_features = input_tensor.Numel() / minibatch_size;
	
	min_values_buffer_.Update(input->GetDimensions());
	Tensor<ParamsType>& min_values_tensor = *min_values_buffer_();
	
	for (size_t i=0; i<num_features; i++)
		min_values_tensor[i] = std::numeric_limits<ParamsType>::max();

	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
		for (size_t i=0; i<num_features; i++)
		{
			size_t sample_offset = num_features*sample_ind;
			min_values_tensor[i] = std::min(min_values_tensor[i], input_tensor[sample_offset+i]);
		}

	partition_values_buffer_.Update(input->GetDimensions());
	Tensor<ParamsType>& partition_values_tensor = *partition_values_buffer_();
	partition_values_tensor.SetZeros();
	for (size_t i=0; i<num_features; i++)
		partition_values_tensor[i] = EPS_;
	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
	{
		size_t sample_offset = num_features*sample_ind;
		for (size_t i=0; i<num_features; i++)
			partition_values_tensor[i] += std::exp(input_tensor[sample_offset+i]-min_values_tensor[i]);
	}

	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
	{
		size_t sample_offset = num_features*sample_ind;
		for (size_t i = 0; i<num_features; i++)
			output_tensor[sample_offset+i] = std::exp(input_tensor[sample_offset+i]-min_values_tensor[i])/partition_values_tensor[i];
	}
}

template <class ParamsType>
void BatchSoftmaxModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
	std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
	const std::vector<ParamsType>& samples_importances)
{	
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
			weighted_gradients_tensor[i] += output_tensor[sample_offset+i]*output_gradients_tensor[sample_offset+i];
	}
	
	for (size_t sample_ind = 0; sample_ind<minibatch_size; sample_ind++)
	{
		size_t sample_offset = num_features*sample_ind;
		for (size_t i = 0; i<num_features; i++)
			input_gradients_tensor[sample_offset+i] = static_cast<ParamsType>(  output_tensor[sample_offset+i] * 
				( output_gradients_tensor[sample_offset+i] - weighted_gradients_tensor[i] ) );
	}
}

#endif