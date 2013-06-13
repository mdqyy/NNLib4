#ifndef RECTIFIED_LINEAR_UNIT_MODULE_H
#define RECTIFIED_LINEAR_UNIT_MODULE_H

#include "Module.h"

template <class ParamsType>
class RectifiedLinearUnitModule : public Module<ParamsType>
{
public:

	RectifiedLinearUnitModule( std::string name) : Module<ParamsType>(name)
	{
	}

	virtual std::string GetType() const
	{
		return "RectifiedLinearUnitModule";
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
void RectifiedLinearUnitModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > RectifiedLinearUnitModule<ParamsType>::Create(IOTreeNode& data)
{
	return std::shared_ptr< Module< ParamsType> >( new RectifiedLinearUnitModule<ParamsType>(data.attributes().GetEntry( "Name" ) ) );
}

template <class ParamsType>
void RectifiedLinearUnitModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	size_t numel = input->Numel();
	for (size_t i = 0; i<numel; i++)
	{
		ParamsType in = (*input)[i];
		(*output)[i] = (ParamsType)(in>0?in:0);
	}
}

template <class ParamsType>
void RectifiedLinearUnitModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
	std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
	const std::vector<ParamsType>& samples_importances)
{
	size_t numel = input->Numel();
	for (size_t i = 0; i<numel; i++)
	{
		ParamsType in = (*input)[i];
		(*input_gradients)[i] = ( (ParamsType)(in>0?1:0) )*(*output_gradients)[i];
	}
}

#endif