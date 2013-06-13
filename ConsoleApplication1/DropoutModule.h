#ifndef DROPOUT_MODULE_H
#define DROPOUT_MODULE_H

#include <string>
#include "Module.h"
#include "RandomGenerator.h"
#include "Converter.h"

template <class ParamsType>
class DropoutModule : public Module<ParamsType>
{
	std::vector<ParamsType> dropout_tensor_data_;
	std::shared_ptr< Tensor<ParamsType> > dropout_tensor_;
	std::vector<size_t> cashed_input_dims;
	double dropout_probability_;

	void UpdateCash(const std::vector<size_t>& input_dims)
	{
		if (cashed_input_dims != input_dims)
		{
			cashed_input_dims = input_dims;
			dropout_tensor_data_.reserve( Tensor<ParamsType>::Numel(input_dims) );
			dropout_tensor_ = std::shared_ptr< Tensor<ParamsType> >(new Tensor<ParamsType>( dropout_tensor_data_.data(), input_dims));
		}
	}

	virtual void sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);
	virtual void sub_predict_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);
	
	virtual void sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
		const std::vector<ParamsType>& samples_importances);
	virtual void sub_GetState(IOTreeNode& node) const;
public:

	DropoutModule(std::string name, double dropout_probability) : Module<ParamsType>(name), dropout_probability_(dropout_probability)
	{

	}

	virtual bool Equals(const Module<ParamsType>& module) const
	{
		if ( !(module.GetType() == GetType() && module.GetName() == GetName()) )
			return false;
		
		const DropoutModule<ParamsType>* other_module = static_cast< const DropoutModule<ParamsType>* >( &module );
		return other_module->dropout_probability_ == dropout_probability_;
	}

	static std::shared_ptr< Module< ParamsType> > Create(IOTreeNode& data);
	
	virtual std::string GetType() const
	{
		return "DropoutModule";
	}
};

template <class ParamsType>
void DropoutModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
	node.attributes().AppendEntry( "DropoutProbability", std::to_string(dropout_probability_) );
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > DropoutModule<ParamsType>::Create(IOTreeNode& data)
{
	double dropout_probability = Converter::ConvertTo<double>(data.attributes().GetEntry( "DropoutProbability"));
	return std::shared_ptr< Module< ParamsType> >( new DropoutModule<ParamsType>(data.attributes().GetEntry( "Name" ), dropout_probability ) );
}

template <class ParamsType>
void DropoutModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	Tensor<ParamsType>& input_tensor = *input;
	UpdateCash(input_tensor.GetDimensions());
	Tensor<ParamsType>& output_tensor = *output;
	Tensor<ParamsType>& dropout_tensor = *dropout_tensor_;
	dropout_tensor.SetZeros();
	size_t numel = input_tensor.Numel();
	for (size_t i = 0; i<numel; i++)
	{
		ParamsType dropout_coeff = static_cast<ParamsType>(RandomGenerator::GetUniformDouble(0,1)>dropout_probability_);
		dropout_tensor[i] = dropout_coeff;
		output_tensor[i] = input_tensor[i]*dropout_coeff;
	}
}

template <class ParamsType>
void DropoutModule<ParamsType>::sub_predict_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& output_tensor = *output;
	size_t numel = input_tensor.Numel();
	for (size_t i = 0; i<numel; i++)
		output_tensor[i] = static_cast<ParamsType>(input_tensor[i]*dropout_probability_);
}

template <class ParamsType>
void DropoutModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
	std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
	const std::vector<ParamsType>& samples_importances)
{
	Tensor<ParamsType>& dropout_tensor = *dropout_tensor_;
	Tensor<ParamsType>& input_gradients_tensor = *input_gradients;
	Tensor<ParamsType>& output_gradients_tensor = *output_gradients;
	size_t numel = output_gradients_tensor.Numel();
	for (size_t i = 0; i<numel; i++)
		input_gradients_tensor[i] = output_gradients_tensor[i]*dropout_tensor[i];
}

#endif