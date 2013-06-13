#ifndef GAUSSIAN_NOSIE_MODULE_H
#define GAUSSIAN_NOSIE_MODULE_H

#include <string>
#include "Module.h"
#include "RandomGenerator.h"
#include "Converter.h"

template <class ParamsType>
class GaussianNoiseModule : public Module<ParamsType>
{
	ParamsType noise_std_;
	virtual void sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);
	virtual void sub_predict_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);
	
	virtual void sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
		const std::vector<ParamsType>& samples_importances);
	virtual void sub_GetState(IOTreeNode& node) const;
public:
	
	virtual bool AlocateOutputBuffer() const
	{
		return true;
	}

	virtual bool AlocateInputGradientsBuffer() const
	{
		return false;
	}

	GaussianNoiseModule(std::string name, ParamsType noise_std) : Module<ParamsType>(name), noise_std_(noise_std)
	{

	}

	virtual bool Equals(const Module<ParamsType>& module) const
	{
		if ( !(module.GetType() == GetType() && module.GetName() == GetName()) )
			return false;
		
		const GaussianNoiseModule<ParamsType>* other_module = static_cast< const GaussianNoiseModule<ParamsType>* >( &module );
		return other_module->noise_std_ == noise_std_;
	}

	static std::shared_ptr< Module< ParamsType> > Create(IOTreeNode& data);
	
	virtual std::string GetType() const
	{
		return "GaussianNoiseModule";
	}
};

template <class ParamsType>
void GaussianNoiseModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
	node.attributes().AppendEntry( "noise_std", std::to_string(noise_std_) );
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > GaussianNoiseModule<ParamsType>::Create(IOTreeNode& data)
{
	ParamsType noise_std_ = Converter::ConvertTo<ParamsType>(data.attributes().GetEntry( "noise_std"));
	return std::shared_ptr< Module< ParamsType> >( new GaussianNoiseModule<ParamsType>(data.attributes().GetEntry( "Name" ), noise_std_ ) );
}

template <class ParamsType>
void GaussianNoiseModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& output_tensor = *output;
	size_t numel = input_tensor.Numel();
	if (noise_std_==0)
		for (size_t i = 0; i<numel; i++)
			output_tensor[i] = static_cast<ParamsType>(input_tensor[i]);
	else
	{
		for (size_t i = 0; i<numel; i++)
			output_tensor[i] = static_cast<ParamsType>(input_tensor[i]+RandomGenerator::GetNormalDouble(0,noise_std_));
	}
}

template <class ParamsType>
void GaussianNoiseModule<ParamsType>::sub_predict_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& output_tensor = *output;
	size_t numel = input_tensor.Numel();
	for (size_t i = 0; i<numel; i++)
		output_tensor[i] = input_tensor[i];
}

template <class ParamsType>
void GaussianNoiseModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
	std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
	const std::vector<ParamsType>& samples_importances)
{
	input_gradients = output_gradients;
}

#endif