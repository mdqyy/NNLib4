#ifndef SOFTSIGN_MODULE_H
#define SOFTSIGN_MODULE_H

#include "Module.h"

template <class ParamsType>
class SoftSignModule : public Module<ParamsType>
{
public:

	SoftSignModule( std::string name) : Module<ParamsType>(name)
	{
	}

	virtual std::string GetType() const
	{
		return "SoftSignModule";
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
void SoftSignModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > SoftSignModule<ParamsType>::Create(IOTreeNode& data)
{
	return std::shared_ptr< Module< ParamsType> >( new SoftSignModule<ParamsType>(data.attributes().GetEntry( "Name" ) ));
}

template <class ParamsType>
void SoftSignModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& output_tensor = *output;
	for (size_t i = 0; i<input->Numel(); i++)
	{
		ParamsType in = input_tensor[i];
		output_tensor[i] = static_cast<ParamsType>(in / (1.0 + std::abs(in)));
	}
}

template <class ParamsType>
void SoftSignModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
	std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
	const std::vector<ParamsType>& samples_importances)
{
	Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& input_gradients_tensor = *input_gradients;
	Tensor<ParamsType>& output_gradients_tensor = *output_gradients;
	Tensor<ParamsType>& output_tensor = *output;
	for (size_t i = 0; i<input_tensor.Numel(); i++)
	{
		ParamsType in = input_tensor[i];
		ParamsType out = output_tensor[i];
		input_tensor[i] = out/in*(1-std::abs(out));
		input_gradients_tensor[i] = out/in*(1-std::abs(out))*output_gradients_tensor[i];
	}
}

#endif