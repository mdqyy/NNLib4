#ifndef LINEAR_MAX_MODULE_H
#define LINEAR_MAX_MODULE_H

#include "Module.h"
#include "UniformInitializer.h"
#include "EmptyRegularizer.h"
#include "Converter.h"
#include "RegularizerFactory.h"
#include "InitializerFactory.h"
#include "IOTreeNode.h"
#include "TensorIO.h"

template <class ParamsType>
class LinearMaxModule : public Module<ParamsType>
{
private:
	const size_t SLOPE_DIM = 0;
	const size_t BIAS_DIM = 1;

	Tensor<ParamsType> parameters;
	Tensor<ParamsType> gradients;
public:

	virtual double GetCost(const std::vector<ParamsType>& samples_importances)
	{
		ParamsType importance_sum = static_cast<ParamsType>(std::accumulate(samples_importances.begin(),samples_importances.end(),0.0));
		return regularizer->GetCost(parameters, importance_sum);
	}

	virtual void GetParameters(std::vector<ParamsType>& receiver) const
	{
		for (size_t i=0; i < parameters.Numel(); i++)
			receiver.push_back( parameters[i] );
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

	virtual void GetGradients(std::vector<ParamsType>& receiver) const
	{
		for (size_t i=0; i < parameters.Numel(); i++)
			receiver.push_back( gradients[i] );
	}

	LinearMaxModule(std::string name, const std::vector<size_t>& input_case_dims, size_t num_lines) : Module(name)
	{
		std::vector<size_t> parameters_dims(input_case_dims);
		parameters_dims.push_back(2);
		parameters_dims.push_back(num_lines);
		parameters = Tensor<ParamsType>(parameters_dims);
		gradients = Tensor<ParamsType>(gradients_dims);
	}

	virtual void InitializeParameters()
	{
		Module<ParamsType>::InitializeParameters();

		 // initialize bias with small values
		UniformInitializer<ParamsType> initializer(-0.01, 0.01);
		params_initializer->InitializeParameters(this->parameters);

		// make all lines initially have slope one so that they did not spoil the gradient
		size_t num_lines = parameters->GetDimensionSize( parameters->NumDimensions() );
		size_t num_sample_features = parameters->Numel() / (2*num_lines);
		for (size_t line_ind=0; line_ind< num_lines; line_ind++)
		{
			size_t offset = (2*line_ind+SLOPE_DIM)*num_sample_features;
			for (size_t i=0; i<num_sample_features; i++)
				parameters[ offset+i ] = 1;
		}
	}

	virtual size_t GetNumParams() const
	{
		return parameters.Numel();
	}

	virtual bool AlocateOutputBuffer() const
	{
		return true;
	}

	virtual bool AlocateInputGradientsBuffer() const
	{
		return true;
	}

	virtual std::string GetType() const
	{
		return "LinearMaxModule";
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
bool LinearMaxModule<ParamsType>::Equals(const Module<ParamsType>& module) const
{
	if (module.GetType() != GetType() || module.GetName() != GetName())
		return false;

	const LinearMaxModule<ParamsType>* other_module = static_cast< const LinearMaxModule<ParamsType>* >( &module );
	if (other_module->parameters != parameters)
		return false;
	return true;
}

template <class ParamsType>
void LinearMaxModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
	node.nodes().AppendEntry( "Parameters", GetTensorState(parameters) );
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > LinearMaxModule<ParamsType>::Create(IOTreeNode& data)
{
	std::shared_ptr< Tensor<ParamsType> > parameters = CreateTensor<ParamsType>(*data.nodes().GetEntry("Parameters"));

	std::vector< size_t > sample_dims = parameters->GetDimensions();
	size_t num_lines = sample_dims[sample_dims.size()-1];
	sample_dims.pop_back();
	sample_dims.pop_back();

	std::shared_ptr< LinearMaxModule<ParamsType> > module = 
		std::shared_ptr< LinearMaxModule< ParamsType> >( new LinearMaxModule<ParamsType>(data.attributes().GetEntry( "Name" ), sample_dims, num_lines) );
	module->SetParameters(*parameters);

	return module;
}

template <class ParamsType>
void LinearMaxModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	const Tensor<ParamsType>& input_tensor = *input;
	Tensor<ParamsType>& output_tensor = *output;
	const size_t minibatch_size = output->GetDimensionSize(output_tensor.NumDimensions()-1);
	const size_t num_input_features = input_tensor.Numel() / minibatch_size;
	size_t num_lines = parameters->GetDimensionSize( parameters->NumDimensions() );
	size_t num_sample_features = parameters->Numel() / (2*num_lines);

	assert( num_sample_features == num_input_features);
	for (size_t sample_index = 0; sample_index<minibatch_size; sample_index++)
	{
		size_t input_offset = num_input_features*sample_index;
		for ( size_t offset = 0; offset < num_input_features; offset++ )
			output_tensor[input_offset+offset] = input_tensor[input_offset+offset]+parameters[offset];
	}
}
	
template <class ParamsType>
void LinearMaxModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
	std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
	const std::vector<ParamsType>& samples_importances)
{
	gradients.SetZeros();
	ParamsType importance_sum = static_cast<ParamsType>(std::accumulate(samples_importances.begin(),samples_importances.end(),0.0));
	regularizer->GetGradients(parameters, gradients, importance_sum);
	
	Tensor<ParamsType>& output_gradients_tensor = *output_gradients;

	const size_t minibatch_size = output->GetDimensionSize(output->NumDimensions()-1);
	const size_t num_input_features = input->Numel() / minibatch_size;
	size_t num_params = GetNumParams();
	for (size_t sample_index = 0; sample_index<minibatch_size; sample_index++)
	{
		size_t output_offset = num_input_features*sample_index;
		for ( size_t offset = 0; offset < num_input_features; offset++ )
			gradients[offset] += output_gradients_tensor[output_offset+offset];
	}

	// backprop gradients
	input_gradients = output_gradients;
}

#endif