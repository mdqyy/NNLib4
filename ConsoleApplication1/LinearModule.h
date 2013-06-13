#ifndef LINEAR_MODULE_H
#define LINEAR_MODULE_H

#include "Module.h"
#include "ConstantInitializer.h"
#include "EmptyRegularizer.h"
#include "IOTreeNode.h"
#include "TensorIO.h"

template <class ParamsType>
class LinearModule : public Module<ParamsType>
{
private:
	Tensor<ParamsType> parameters;
	Tensor<ParamsType> gradients;
	std::shared_ptr<ParametersInitializer<ParamsType> > params_initializer;
	std::shared_ptr<Regularizer<ParamsType> > regularizer;
public:

	virtual double GetCost(const std::vector<ParamsType>& samples_importances)
	{
		ParamsType importance_sum = static_cast<ParamsType>(std::accumulate(samples_importances.begin(),samples_importances.end(),0.0));
		return regularizer->GetCost(parameters, importance_sum);
	}
	
	LinearModule(std::string name, const std::vector<size_t>& input_case_dims, 
		const std::shared_ptr<ParametersInitializer<ParamsType> >& params_initializer = std::shared_ptr<ParametersInitializer<ParamsType> >(new ConstantInitializer<ParamsType>(0)),
		const std::shared_ptr<Regularizer<ParamsType> >& regularizer = std::shared_ptr<Regularizer<ParamsType> >(new EmptyRegularizer<ParamsType>()) ) 
			: Module(name), params_initializer(params_initializer), regularizer(regularizer), parameters(input_case_dims), gradients(input_case_dims)
	{
		parameters.SetZeros();
	}
	
	virtual void SetParameters(const ParamsType* params)
	{
		for (size_t i=0; i < parameters.Numel(); i++)
			parameters[i] = params[i];
	}

	virtual void SetParameters(const std::vector<ParamsType>& params)
	{
		assert(parameters.Numel() == params.size());
		SetParameters(params.data());
	}

	virtual void SetParameters(const Tensor<ParamsType>& params)
	{
		assert( params.GetDimensions() == parameters.GetDimensions() );
		for (size_t i=0; i < params.Numel(); i++)
			parameters[i] = params[i];
	}

	virtual void GetParameters(std::vector<ParamsType>& receiver) const
	{
		for (size_t i=0; i < parameters.Numel(); i++)
			receiver.push_back( parameters[i] );
	}

	virtual void GetGradients(std::vector<ParamsType>& receiver) const
	{
		for (size_t i=0; i < parameters.Numel(); i++)
			receiver.push_back( gradients[i] );
	}

	virtual void InitializeParameters()
	{
		Module<ParamsType>::InitializeParameters();
		params_initializer->InitializeParameters(this->parameters);
	}

	virtual size_t GetNumParams() const
	{
		return parameters.Numel();
	}

	virtual std::string GetType() const
	{
		return "LinearModule";
	}
	
	virtual bool Equals(const Module<ParamsType>& module) const;

	static std::shared_ptr< Module< ParamsType> > Create(IOTreeNode& data);

protected:
	virtual void sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);
	
	virtual void sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
		const std::vector<ParamsType>& samples_importances);

	virtual void sub_GetState(IOTreeNode& node) const;
};

template <class ParamsType>
bool LinearModule<ParamsType>::Equals(const Module<ParamsType>& module) const
{
	if (module.GetType() != GetType() || module.GetName() != GetName())
		return false;

	const LinearModule<ParamsType>* other_module = static_cast< const LinearModule<ParamsType>* >( &module );
	if (other_module->parameters != parameters)
		return false;
	if (!params_initializer->Equals(*other_module->params_initializer))
		return false;
	if (!regularizer->Equals(*other_module->regularizer))
		return false;
	return true;
}

template <class ParamsType>
void LinearModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
	node.nodes().AppendEntry( "regularizer", regularizer->GetState() );
	node.nodes().AppendEntry( "initializer", params_initializer->GetState() );
	node.nodes().AppendEntry( "Parameters", GetTensorState(parameters) );
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > LinearModule<ParamsType>::Create(IOTreeNode& data)
{
	std::shared_ptr< Regularizer<ParamsType> > regularizer = RegularizerFactory::GetRegularizer<ParamsType>(*data.nodes().GetEntry("regularizer"));
	std::shared_ptr< ParametersInitializer<ParamsType> > initializer = InitializerFactory::GetInitializer<ParamsType>(*data.nodes().GetEntry("initializer"));
	std::shared_ptr< Tensor<ParamsType> > parameters = CreateTensor<ParamsType>(*data.nodes().GetEntry("Parameters"));

	std::shared_ptr< LinearModule<ParamsType> > module = 
		std::shared_ptr< LinearModule< ParamsType> >( new LinearModule<ParamsType>(data.attributes().GetEntry( "Name" ), 
		parameters->GetDimensions(), initializer, regularizer) );
	module->SetParameters(*parameters);

	return module;
}

template <class ParamsType>
void LinearModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	const Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& output_tensor = *output;
	const size_t minibatch_size = output->GetDimensionSize(output_tensor.NumDimensions()-1);
	const size_t num_input_features = input_tensor.Numel() / minibatch_size;
	assert( GetNumParams() == num_input_features);
	for (size_t sample_index = 0; sample_index<minibatch_size; sample_index++)
	{
		size_t input_offset = num_input_features*sample_index;
		for ( size_t offset = 0; offset < num_input_features; offset++ )
			output_tensor[input_offset+offset] = input_tensor[input_offset+offset]*parameters[offset];
	}
}
	
template <class ParamsType>
void LinearModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
	std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
	const std::vector<ParamsType>& samples_importances)
{
	gradients.SetZeros();
	ParamsType importance_sum = static_cast<ParamsType>(std::accumulate(samples_importances.begin(),samples_importances.end(),0.0));
	regularizer->GetGradients(parameters, gradients, importance_sum);
	
	Tensor<ParamsType>& output_gradients_tensor = *output_gradients;
	Tensor<ParamsType>& input_gradients_tensor = *input_gradients;
	Tensor<ParamsType>& input_tensor = *input;

	const size_t minibatch_size = output->GetDimensionSize(output->NumDimensions()-1);
	const size_t num_input_features = input->Numel() / minibatch_size;
	size_t num_params = GetNumParams();
	for (size_t sample_index = 0; sample_index<minibatch_size; sample_index++)
	{
		size_t output_offset = num_input_features*sample_index;
		for ( size_t offset = 0; offset < num_input_features; offset++ )
			gradients[offset] += output_gradients_tensor[output_offset+offset]*input_tensor[output_offset+offset];
	}

	// backprop gradients
	for (size_t sample_index = 0; sample_index<minibatch_size; sample_index++)
	{
		size_t output_offset = num_input_features*sample_index;
		for ( size_t offset = 0; offset < num_input_features; offset++ )
			input_gradients_tensor[output_offset+offset] += output_gradients_tensor[output_offset+offset]*parameters[offset];
	}
}

#endif