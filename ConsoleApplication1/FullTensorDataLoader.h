#ifndef FULL_TENSOR_DATA_LOADER_H
#define FULL_TENSOR_DATA_LOADER_H

#include "BaseTensorDataLoader.h"

template <class OutputType, class InputType>
class FullTensorDataLoader : public BaseTensorDataLoader<OutputType, InputType>
{
	typedef std::shared_ptr< Tensor<OutputType> > output_type_tensor_ptr;
	typedef std::shared_ptr< Tensor<InputType> > input_type_tensor_ptr;

	size_t GetSampleData( size_t sample_ind, Tensor<OutputType>& output_buffer, size_t output_buffer_offset ) const
	{
		Tensor<InputType>& sample = GetSample(sample_ind);

		size_t numel = sample.Numel();
		for (size_t i=0; i<numel; i++)
			output_buffer[output_buffer_offset+i] = static_cast<OutputType>(sample[i]);
		
		return sample.Numel();
	}
	
	virtual void sub_save(std::ostream& output_stream) const
	{
	}

public:
	
	static std::shared_ptr< FullTensorDataLoader<OutputType, InputType> > Create(std::string name, 
		const std::vector< std::shared_ptr< Tensor<InputType> > >& data, std::istream& input_stream)
	{
		return std::shared_ptr< FullTensorDataLoader<OutputType, InputType> >( new FullTensorDataLoader<OutputType, InputType>( data, name) );
	}

	virtual std::string GetType() const
	{
		return "FullTensorDataLoader";
	}

	virtual bool Equals(BaseTensorDataLoader<OutputType, InputType>& data_loader)
	{
		return BaseTensorDataLoader<OutputType, InputType>::Equals(data_loader);
	}
	
	FullTensorDataLoader(const std::vector< input_type_tensor_ptr >& data, std::string name = "Default") : 
		BaseTensorDataLoader<OutputType, InputType>(data, name)
	{

	}

	virtual output_type_tensor_ptr GetData(const std::vector<size_t>& samples_inds) const
	{
		output_type_tensor_ptr output_buffer = GetOutputBuffer(samples_inds.size(), GetSampleDims());
		size_t offset = 0;
		for (size_t sample_ind = 0; sample_ind<samples_inds.size(); sample_ind++)
			offset+=GetSampleData(samples_inds[sample_ind], *output_buffer, offset);

		assert( offset == output_buffer->Numel());

		return output_buffer;
	}
};

#endif