#ifndef MEAN_STD_NORMALIZING_MODULE
#define MEAN_STD_NORMALIZING_MODULE

#include "Module.h"
#include "TensorIO.h"
#include "my_math.h"

template <class ParamsType>
class MeanStdNormalizingModule : public Module<ParamsType>
{
private:
	ParamsType eps_;
	ParamsType decay_;
	std::vector<ParamsType> means_;
	std::vector<ParamsType> stds_;
	bool frozen_;

	MeanStdNormalizingModule(std::string name, ParamsType decay, std::vector<ParamsType> means, std::vector<ParamsType> stds ) : 
		Module(name), decay_(decay), means_( means ), stds_( stds ),  frozen_(false), eps_(std::numeric_limits<ParamsType>::epsilon())
	{
	}

public:

	void DebugFreeze()
	{
		frozen_ = true;
	}
	
	void DebugUnfreeze()
	{
		frozen_ = false;
	}

	std::vector<ParamsType>& GetMeans()
	{
		return means_;
	}
	
	std::vector<ParamsType>& GetStds()
	{
		return stds_;
	}

	MeanStdNormalizingModule(std::string name, size_t num_inputs, ParamsType start_std = 100, ParamsType decay=0.999 ) : 
		Module(name), means_( num_inputs, 0), stds_(num_inputs, start_std), decay_(decay), frozen_(false),
		eps_(std::numeric_limits<ParamsType>::epsilon())
	{
	}

	virtual std::string GetType() const
	{
		return "MeanStdNormalizingModule";
	}
	
	virtual bool Equals(const Module<ParamsType>& module) const;
	
	static std::shared_ptr< Module< ParamsType> > Create(IOTreeNode& data);

protected:
	virtual void sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);
	virtual void sub_predict_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);
	
	virtual void sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
		const std::vector<ParamsType>& samples_importances);

	virtual void sub_GetState(IOTreeNode& node) const;
};

template <class ParamsType>
bool MeanStdNormalizingModule<ParamsType>::Equals(const Module<ParamsType>& module) const
{
	if (module.GetType() != GetType() || module.GetName() != GetName())
		return false;

	const MeanStdNormalizingModule<ParamsType>* other_module = static_cast< const MeanStdNormalizingModule<ParamsType>* >( &module );
	return (decay_ == other_module->decay_) && (means_ == other_module->means_) && (stds_ == other_module->stds_);
}

template <class ParamsType>
void MeanStdNormalizingModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
	node.attributes().AppendEntry( "decay", std::to_string(decay_) );
	node.attributes().AppendEntry( "means", Converter::ConvertVectorToString(means_) );
	node.attributes().AppendEntry( "stds", Converter::ConvertVectorToString(stds_) );
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > MeanStdNormalizingModule<ParamsType>::Create(IOTreeNode& node)
{
	ParamsType decay = Converter::ConvertTo<ParamsType>(node.attributes().GetEntry( "decay"));
	std::vector<ParamsType> means = Converter::StringToVector<ParamsType>(node.attributes().GetEntry( "means"));
	std::vector<ParamsType> stds = Converter::StringToVector<ParamsType>(node.attributes().GetEntry( "stds"));

	std::shared_ptr< MeanStdNormalizingModule<ParamsType> > module = 
		std::shared_ptr< MeanStdNormalizingModule< ParamsType> >( new MeanStdNormalizingModule<ParamsType>(node.attributes().GetEntry( "Name" ), 
		decay, means, stds) );

	return module;
}

template <class ParamsType>
void MeanStdNormalizingModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	const Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& output_tensor = *output;
	const size_t minibatch_size = output->GetDimensionSize(output_tensor.NumDimensions()-1);
	const size_t num_input_features = input_tensor.Numel() / minibatch_size;
	assert( num_input_features == means_.size() );

	if (!frozen_)
	{
		ParamsType batch_decay = static_cast<ParamsType>(pow(decay_, minibatch_size));
		std::vector<ParamsType> batch_means(means_.size(), 0);
		std::vector<ParamsType> batch_variances(means_.size(), 0);

		for (size_t sample_index = 0; sample_index<minibatch_size; sample_index++)
		{
			size_t input_offset = num_input_features*sample_index;
			for ( size_t offset = 0; offset < num_input_features; offset++ )
				batch_means[offset] += input_tensor[input_offset+offset];
		}
	
		for ( size_t i = 0; i < means_.size(); i++ )
			means_[i] = batch_decay*means_[i]+(1-batch_decay)*batch_means[i]/minibatch_size;

		for (size_t sample_index = 0; sample_index<minibatch_size; sample_index++)
		{
			size_t input_offset = num_input_features*sample_index;
			for ( size_t offset = 0; offset < num_input_features; offset++ )
				batch_variances[offset] += sqr(input_tensor[input_offset+offset] - means_[offset]);
		}
	
		for ( size_t i = 0; i < stds_.size(); i++ )
		{
			stds_[i] = batch_decay*stds_[i]+(1-batch_decay)*(sqrt(batch_variances[i] / minibatch_size));
			if (stds_[i]==0)
				stds_[i]=eps_;
		}
	}

	for (size_t sample_index = 0; sample_index<minibatch_size; sample_index++)
	{
		size_t input_offset = num_input_features*sample_index;
		for ( size_t offset = 0; offset < num_input_features; offset++ )
			output_tensor[input_offset+offset] = (input_tensor[input_offset+offset] - means_[offset]) / stds_[offset];
	}
}

template <class ParamsType>
void MeanStdNormalizingModule<ParamsType>::sub_predict_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	const Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& output_tensor = *output;
	const size_t minibatch_size = output->GetDimensionSize(output_tensor.NumDimensions()-1);
	const size_t num_input_features = input_tensor.Numel() / minibatch_size;
	assert( num_input_features == means_.size() );

	for (size_t sample_index = 0; sample_index<minibatch_size; sample_index++)
	{
		size_t input_offset = num_input_features*sample_index;
		for ( size_t offset = 0; offset < num_input_features; offset++ )
			output_tensor[input_offset+offset] = (input_tensor[input_offset+offset] - means_[offset]) / stds_[offset];
	}
}

template <class ParamsType>
void MeanStdNormalizingModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
	std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
	const std::vector<ParamsType>& samples_importances)
{	
	Tensor<ParamsType>& output_gradients_tensor = *output_gradients;
	Tensor<ParamsType>& input_gradients_tensor = *input_gradients;
	const size_t minibatch_size = output->GetDimensionSize(output_gradients_tensor.NumDimensions()-1);
	const size_t num_input_features = input_gradients_tensor.Numel() / minibatch_size;

	for (size_t sample_index = 0; sample_index<minibatch_size; sample_index++)
	{
		size_t input_offset = num_input_features*sample_index;
		for ( size_t offset = 0; offset < num_input_features; offset++ )
			input_gradients_tensor[input_offset+offset] = output_gradients_tensor[input_offset+offset] / stds_[offset];
	}
}

#endif