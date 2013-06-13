#ifndef TANH_MODULE_H
#define TANH_MODULE_H

#include "Module.h"

template <class ParamsType>
class TanhModule : public Module<ParamsType>
{
public:

	TanhModule( std::string name) : Module<ParamsType>(name)
	{
	}

	virtual std::string GetType() const
	{
		return "TanhModule";
	}
	
	virtual bool Equals(const Module<ParamsType>& module) const
	{
		return module.GetType() == GetType() && module.GetName() == GetName();
	}
	
	static std::shared_ptr< Module< ParamsType> > Create(IOTreeNode& data);

protected:
	virtual void sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
	{
		size_t numel = input->Numel();
		for (size_t i = 0; i<numel; i++)
			(*output)[i] = (ParamsType)(2.0 / (1.0 + std::exp(-2*(*input)[i]))-1);
	}

	virtual void sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
		const std::vector<ParamsType>& samples_importances)
	{
		size_t numel = input->Numel();
		for (size_t i = 0; i<numel; i++)
		{
			ParamsType out = (*output)[i];
			(*input_gradients)[i] = (1-out*out)*(*output_gradients)[i];
		}
	}

	virtual void sub_GetState(IOTreeNode& node) const;
};

template <class ParamsType>
void TanhModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > TanhModule<ParamsType>::Create(IOTreeNode& data)
{
	return std::shared_ptr< Module< ParamsType> >( new TanhModule<ParamsType>(data.attributes().GetEntry( "Name" ) ));
}

#endif