#ifndef SIGMOID_MODULE_H
#define SIGMOID_MODULE_H

#include "Module.h"

template <class ParamsType>
class SigmoidModule : public Module<ParamsType>
{
public:

	SigmoidModule( std::string name) : Module<ParamsType>(name)
	{
	}

	virtual std::string GetType() const
	{
		return "SigmoidModule";
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
void SigmoidModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > SigmoidModule<ParamsType>::Create(IOTreeNode& data)
{
	return std::shared_ptr< Module< ParamsType> >( new SigmoidModule<ParamsType>(data.attributes().GetEntry( "Name" ) ));
}

template <class ParamsType>
void SigmoidModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	for (size_t i = 0; i<input->Numel(); i++)
		(*output)[i] = (ParamsType)(1.0 / (1.0 + std::exp(-(*input)[i])));
}

template <class ParamsType>
void SigmoidModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
	std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
	const std::vector<ParamsType>& samples_importances)
{
	for (size_t i = 0; i<input->Numel(); i++)
	{
		ParamsType out = (*output)[i];
		(*input_gradients)[i] = out*(1-out)*(*output_gradients)[i];
	}
}

#endif