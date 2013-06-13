#ifndef ABS_MODULE_H
#define ABS_MODULE_H

#include "Module.h"

template <class ParamsType>
class AbsModule : public Module<ParamsType>
{
public:

	AbsModule( std::string name) : Module<ParamsType>(name)
	{
	}

	virtual std::string GetType() const
	{
		return "AbsModule";
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
void AbsModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > AbsModule<ParamsType>::Create(IOTreeNode& data)
{
	return std::shared_ptr< Module< ParamsType> >( new AbsModule<ParamsType>(data.attributes().GetEntry( "Name" ) ) );
}

template <class ParamsType>
void AbsModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	const Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& output_tensor = *output;
	for (size_t i = 0; i<input->Numel(); i++)
		output_tensor[i] = std::abs(input_tensor[i]);
}

template <class ParamsType>
void AbsModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
	std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
	const std::vector<ParamsType>& samples_importances)
{
	const Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& input_gradients_tensor = *input_gradients;
	const Tensor<ParamsType>& output_gradients_tensor = *output_gradients;
	for (size_t i = 0; i<input->Numel(); i++)
	{
		ParamsType in = input_tensor[i];
		input_gradients_tensor[i] = (in>0?1:-1)*output_gradients_tensor[i];
	}
}

#endif