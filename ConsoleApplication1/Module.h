#ifndef MODULE_H
#define MODULE_H

//#include <math.h>
#include <vector>
#include <memory>
#include "CashedTensor.h"
#include "Tensor.h"
#include "IOTreeNode.h"

template <class ParamsType>
class Module
{
	std::string name_;
	std::vector<ParamsType> output_buffer_data_;
	std::vector<ParamsType> input_gradients_buffer_data_;

	std::shared_ptr< Tensor<ParamsType> > input_buffer_;
	std::shared_ptr< Tensor<ParamsType> > output_buffer_;
	std::shared_ptr< Tensor<ParamsType> > input_gradients_buffer_;
	// buffers are passed by reference so that the modules could set them to point to other buffers without performing copying
	// Modules in train mode and predict mode can behave differently (like dropout)
	virtual void sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output) = 0;
	virtual void sub_predict_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);

	virtual void sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& upper_gradients, 
		const std::vector<ParamsType>& samples_importances) = 0;
	
	virtual void sub_GetState(IOTreeNode& node) const = 0;

	void UpdateCash(const std::shared_ptr< Tensor<ParamsType> >& input);

public:

	virtual bool Equals(const Module<ParamsType>& module) const = 0;

	// determine whether the module is responsible for allocating input gradients and output buffers
	// if returns false, the mmodule is usually transparent - it modifies and returns the inputs buffers
	// example of such unit - composite module - which uses buffers of its inner modules. 
	// In general, if a module does not change the corresponding buffer,
	// it does not need it to be allocated.
	virtual bool AlocateOutputBuffer() const
	{
		return true;
	}

	std::shared_ptr<IOTreeNode> GetState()
	{
		std::shared_ptr<IOTreeNode> node( new IOTreeNode() );
		node->attributes().AppendEntry( "Category", "Module" );
		node->attributes().AppendEntry( "Type", GetType() );
		node->attributes().AppendEntry( "Name", GetName() );
		sub_GetState( *node );
		return node;
	}

	virtual bool AlocateInputGradientsBuffer() const
	{
		return true;
	}

	std::shared_ptr< Tensor<ParamsType> >& GetOutputBuffer()
	{
		return output_buffer_;
	}
	
	std::shared_ptr< Tensor<ParamsType> >& GetInputBuffer()
	{
		return input_buffer_;
	}

	std::string GetName() const
	{
		return name_;
	}
	
	std::shared_ptr< Tensor<ParamsType> >& GetInputGradientsBuffer(const std::shared_ptr< Tensor<ParamsType> >& ouput_gradients)
	{
		return input_gradients_buffer_;
	}

	Module(std::string name) : name_(name), input_buffer_( std::shared_ptr< Tensor<ParamsType> >( new Tensor<ParamsType>(0, std::vector<size_t>())) ), 
		output_buffer_( std::shared_ptr< Tensor<ParamsType> >( new Tensor<ParamsType>(0, std::vector<size_t>())) )
	{
	}

	virtual double GetCost(const std::vector<ParamsType>& samples_importances)
	{
		return 0;
	}

	virtual void GetParameters(std::vector<ParamsType>& receiver) const
	{
	}

	virtual void SetParameters(const ParamsType* params)
	{
	}

	virtual void SetParameters(const std::vector<ParamsType>& params)
	{
	}

	virtual void SetParameters(const Tensor<ParamsType>& params)
	{
	}

	virtual void GetGradients(std::vector<ParamsType>& receiver) const
	{
	}

	virtual std::string GetType() const = 0;

	virtual void InitializeParameters()
	{
	}

	virtual size_t GetNumParams() const
	{
		return 0;
	}
	
	virtual std::vector<size_t> GetPerCaseOutputDims(const std::vector<size_t>& per_case_input_dims) const
	{
		return per_case_input_dims;
	}

	std::shared_ptr< Tensor<ParamsType> > train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input);

	std::shared_ptr< Tensor<ParamsType> > predict_fprop(const std::shared_ptr< Tensor<ParamsType> >& input);

	std::shared_ptr< Tensor<ParamsType> > bprop(const std::shared_ptr< Tensor<ParamsType> >& ouput_gradients, 
		const std::vector<ParamsType>& samples_importances);

	virtual ~Module()
	{
	}
};

template <class ParamsType>
std::shared_ptr< Tensor<ParamsType> > Module<ParamsType>::bprop(const std::shared_ptr< Tensor<ParamsType> >& ouput_gradients, 
																const std::vector<ParamsType>& samples_importances)
{
	std::shared_ptr< Tensor<ParamsType> > input_gradients = GetInputGradientsBuffer(ouput_gradients);
	
	// if we don't allocate the buffer, we should not change it here, because we don't know how it will affect the ouput_gradients buffer
	if (AlocateInputGradientsBuffer())
		input_gradients->SetZeros();

	sub_bprop(GetInputBuffer(), GetOutputBuffer(), input_gradients, ouput_gradients, samples_importances);
	return input_gradients;
}

template <class ParamsType>
std::shared_ptr< Tensor<ParamsType> > Module<ParamsType>::train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input)
{
	UpdateCash(input);
	std::shared_ptr< Tensor<ParamsType> >& output_buffer = GetOutputBuffer();
	
	// if we don't allocate the buffer, we should not change it here, because we don't know how it will affect the input buffer
	if (AlocateOutputBuffer())
		output_buffer->SetZeros();

	sub_train_fprop(input, output_buffer);
	return output_buffer;
}

template <class ParamsType>
std::shared_ptr< Tensor<ParamsType> > Module<ParamsType>::predict_fprop(const std::shared_ptr< Tensor<ParamsType> >& input)
{
	UpdateCash(input);
	std::shared_ptr< Tensor<ParamsType> >& output_buffer = GetOutputBuffer();
	
	// if we don't allocate the buffer, we should not change it here, because we don't know how it will affect the input buffer
	if (AlocateOutputBuffer())
		output_buffer->SetZeros();

	sub_predict_fprop(input, output_buffer);
	return output_buffer;
}

template <class ParamsType>
void Module<ParamsType>::sub_predict_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{	
	sub_train_fprop(input, output);
}

template <class ParamsType>
void Module<ParamsType>::UpdateCash(const std::shared_ptr< Tensor<ParamsType> >& input)
{
	if ( !input_buffer_->DimensionsEqual( *input ) )
	{
		std::vector<size_t> input_dims = input->GetDimensions();
		if (AlocateInputGradientsBuffer())
		{
			input_gradients_buffer_data_.reserve(Tensor<ParamsType>::Numel(input_dims));
			input_gradients_buffer_ = std::shared_ptr< Tensor<ParamsType> >( new Tensor<ParamsType>(input_gradients_buffer_data_.data(), input_dims));
		}

		// Get output dimensions
		if (AlocateOutputBuffer())
		{
			size_t num_samples = input_dims[input_dims.size()-1];
			input_dims.pop_back();
			std::vector<size_t> output_dims = GetPerCaseOutputDims(input_dims);
			output_dims.push_back(num_samples);
			output_buffer_data_.reserve(Tensor<ParamsType>::Numel(output_dims));
			output_buffer_ = std::shared_ptr< Tensor<ParamsType> >( new Tensor<ParamsType>(output_buffer_data_.data(), output_dims));
		}
	}
	
	input_buffer_ = input;
}

#endif