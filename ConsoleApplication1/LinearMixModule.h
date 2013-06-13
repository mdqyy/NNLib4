#ifndef LINEAR_MIX_MODULE_H
#define LINEAR_MIX_MODULE_H

#include <algorithm>
#include "Module.h"
#include "ConstantInitializer.h"
#include "EmptyRegularizer.h"
#include "Converter.h"
#include "IOTreeNode.h"
#include "TensorIO.h"
#include "MatrixOperations.h"

template <class ParamsType>
class LinearMixModule : public Module<ParamsType>
{
private:
	Tensor<ParamsType> parameters;
	Tensor<ParamsType> gradients;
	std::shared_ptr<ParametersInitializer<ParamsType> > params_initializer;
	std::shared_ptr<Regularizer<ParamsType> > regularizer;
public:

	LinearMixModule(std::string name, size_t num_input_features, size_t num_output_features, 
		const std::shared_ptr<ParametersInitializer<ParamsType> >& params_initializer = std::shared_ptr<ParametersInitializer<ParamsType> >(new EmptyInitializer<ParamsType>()),
		const std::shared_ptr<Regularizer<ParamsType> >& regularizer = std::shared_ptr<Regularizer<ParamsType> >(new EmptyRegularizer<ParamsType>()) );
	
	size_t GetNumOutputs() const
	{
		return parameters.GetDimensionSize(0);
	}

	size_t GetNumInputs() const
	{
		return parameters.GetDimensionSize(1);
	}

	virtual double GetCost(const std::vector<ParamsType>& samples_importances)
	{
		ParamsType importance_sum = static_cast<ParamsType>(std::accumulate(samples_importances.begin(),samples_importances.end(),0.0));
		return regularizer->GetCost(parameters, importance_sum);
	}

	virtual void GetParameters(std::vector<ParamsType>& receiver) const
	{
		receiver.insert(receiver.end(), parameters.GetStartPtr(), parameters.GetStartPtr() + parameters.Numel());
	}

	virtual void SetParameters(const ParamsType* params)
	{
		size_t numel = parameters.Numel();
		std::copy(params, params + numel, parameters.GetStartPtr());
	}

	virtual void SetParameters(const std::vector<ParamsType>& params)
	{
		assert(parameters.Numel() == params.size());
		SetParameters(params.data());
	}

	virtual void SetParameters(const Tensor<ParamsType>& params)
	{
		assert( params.GetDimensions() == parameters.GetDimensions() );
		size_t numel = params.Numel();
		for (size_t i=0; i < numel; i++)
			parameters[i] = params[i];
	}

	virtual void GetGradients(std::vector<ParamsType>& receiver) const
	{
		receiver.insert(receiver.end(), gradients.GetStartPtr(), gradients.GetStartPtr() + gradients.Numel());
	}

	virtual void InitializeParameters()
	{
		params_initializer->InitializeParameters(this->parameters);
	}

	virtual std::vector<size_t> GetPerCaseOutputDims(const std::vector<size_t>& per_case_input_dims) const
	{
		return std::vector<size_t>(1,GetNumOutputs());
	}

	virtual size_t GetNumParams() const
	{
		return GetNumInputs()*GetNumOutputs();
	}

	virtual std::string GetType() const
	{
		return "LinearMixModule";
	}

	static std::shared_ptr< Module< ParamsType> > Create(IOTreeNode& data);

	virtual bool Equals(const Module<ParamsType>& module) const;

protected:
	virtual void sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);
	
	virtual void sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
		const std::vector<ParamsType>& samples_importances);

	virtual void sub_GetState(IOTreeNode& node) const;
};

template <class ParamsType>
bool LinearMixModule<ParamsType>::Equals(const Module<ParamsType>& module) const
{
	if (module.GetType() != GetType() || module.GetName() != GetName())
		return false;

	const LinearMixModule<ParamsType>* other_module = static_cast< const LinearMixModule<ParamsType>* >( &module );
	if (other_module->parameters != parameters)
		return false;
	if (!params_initializer->Equals(*other_module->params_initializer))
		return false;
	if (!regularizer->Equals(*other_module->regularizer))
		return false;
	return true;
}

template <class ParamsType>
void LinearMixModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
	node.attributes().AppendEntry( "num_inputs", std::to_string(GetNumInputs()) );
	node.attributes().AppendEntry( "num_outputs", std::to_string(GetNumOutputs()) );
	node.nodes().AppendEntry( "regularizer", regularizer->GetState() );
	node.nodes().AppendEntry( "initializer", params_initializer->GetState() );
	node.nodes().AppendEntry( "Parameters", GetTensorState(parameters) );
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > LinearMixModule<ParamsType>::Create(IOTreeNode& data)
{
	std::shared_ptr< Regularizer<ParamsType> > regularizer = RegularizerFactory::GetRegularizer<ParamsType>(*data.nodes().GetEntry("regularizer"));
	std::shared_ptr< ParametersInitializer<ParamsType> > initializer = InitializerFactory::GetInitializer<ParamsType>(*data.nodes().GetEntry("initializer"));
	std::shared_ptr< Tensor<ParamsType> > parameters = CreateTensor<ParamsType>(*data.nodes().GetEntry("Parameters"));
	
	size_t num_inputs = Converter::ConvertTo<size_t>(data.attributes().GetEntry( "num_inputs" ));
	size_t num_outputs = Converter::ConvertTo<size_t>(data.attributes().GetEntry( "num_outputs" ));
	std::shared_ptr< LinearMixModule<ParamsType> > module = 
		std::shared_ptr< LinearMixModule< ParamsType> >( new LinearMixModule<ParamsType>(data.attributes().GetEntry( "Name" ), 
		num_inputs, num_outputs, initializer, regularizer) );
	module->SetParameters(*parameters);

	return module;
}

template <class ParamsType>
LinearMixModule<ParamsType>::LinearMixModule(std::string name, size_t num_input_features, size_t num_output_features, 
		const std::shared_ptr<ParametersInitializer<ParamsType> >& params_initializer = std::shared_ptr<ParametersInitializer<ParamsType> >(new ConstantInitializer<ParamsType>(0)),
		const std::shared_ptr<Regularizer<ParamsType> >& regularizer = std::shared_ptr<Regularizer<ParamsType> >(new EmptyRegularizer<ParamsType>()) ) 
			: Module(name), params_initializer(params_initializer), regularizer(regularizer)
{
	std::vector<size_t> parameters_dims;
	parameters_dims.push_back(num_output_features);
	parameters_dims.push_back(num_input_features);
	this->parameters = Tensor<ParamsType>(parameters_dims);
	this->gradients = Tensor<ParamsType>(parameters_dims);
}

template <class ParamsType>
void LinearMixModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	size_t num_input_features = GetNumInputs();
	size_t num_output_features = GetNumOutputs();
	size_t num_samples = input->GetDimensionSize(input->NumDimensions()-1);
	assert( input->Numel() / num_samples == num_input_features);

	MatrixMultiply<ParamsType>(CblasColMajor, CblasNoTrans, CblasNoTrans, num_output_features, num_samples, 
		num_input_features, 1, parameters.GetStartPtr(), num_output_features, input->GetStartPtr(), 
		num_input_features, 0,output->GetStartPtr(), num_output_features);
}

template <class ParamsType>
void LinearMixModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
		const std::vector<ParamsType>& samples_importances)
{
	size_t num_input_features = GetNumInputs();
	size_t num_output_features = GetNumOutputs();
	size_t num_samples = input->GetDimensionSize(input->NumDimensions()-1);

	// set parameters gradients
	MatrixMultiply<ParamsType>(CblasColMajor, CblasNoTrans, CblasTrans, num_output_features, num_input_features, 
		num_samples, 1, output_gradients->GetStartPtr(), num_output_features, input->GetStartPtr(), 
		num_input_features, 0, gradients.GetStartPtr(), num_output_features);

	ParamsType importance_sum = static_cast<ParamsType>(std::accumulate(samples_importances.begin(),samples_importances.end(),0.0));
	regularizer->GetGradients(parameters, gradients,importance_sum);

	// backpropagate data
	MatrixMultiply<ParamsType>(CblasColMajor, CblasTrans, CblasNoTrans, num_input_features, num_samples, 
		num_output_features, 1, parameters.GetStartPtr(), num_output_features, output_gradients->GetStartPtr(), 
		num_output_features, 0, input_gradients->GetStartPtr(), num_input_features);
}

#endif