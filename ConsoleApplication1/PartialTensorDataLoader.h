#ifndef PARTIAL_TENSOR_DATA_LOADER_H
#define PARTIAL_TENSOR_DATA_LOADER_H

#include "BaseTensorDataLoader.h"

template <class OutputType, class InputType>
class PartialTensorDataLoader : public BaseTensorDataLoader<OutputType, InputType>
{
private:

	mutable std::vector<size_t> sample_offsets_;

	size_t GetSampleData( size_t sample_ind, const std::vector<size_t>& sample_left_offsets, 
		const std::vector<size_t>& sample_right_offsets, Tensor<OutputType>& output_buffer, size_t output_buffer_offset) const;

protected:
	
	virtual void sub_save(std::ostream& output_stream) const
	{
	}

public:

	virtual bool Equals(BaseTensorDataLoader<OutputType, InputType>& data_loader)
	{
		return BaseTensorDataLoader<OutputType, InputType>::Equals(data_loader);
	}

	PartialTensorDataLoader(const std::vector< std::shared_ptr< Tensor<InputType> > >& data, std::string name) : 
		BaseTensorDataLoader<OutputType, InputType>(data, name)
	{

	}

	std::shared_ptr< Tensor<OutputType> > GetSamplesData( const std::vector<size_t>& samples_inds, 
		const std::vector< std::vector<size_t> >& samples_left_offsets, 
		const std::vector< std::vector<size_t> >& samples_right_offsets) const;

};

template <class OutputType, class InputType>
size_t PartialTensorDataLoader<OutputType,InputType>::GetSampleData( size_t sample_ind, const std::vector<size_t>& sample_left_offsets, 
	const std::vector<size_t>& sample_right_offsets, Tensor<OutputType>& output_buffer, size_t output_buffer_offset) const
{
	Tensor<InputType>& sample = GetSample(sample_ind);

	// use tensor's method for valid positions selection
	std::vector<size_t> strides(sample.Numel());
	for (size_t i=0; i<strides.size(); i++)
		strides[i] = 1;
	sample_offsets_.clear();
	Tensor<InputType>::GetValidOffsetsInds(sample_offsets_, sample.GetDimensions(), sample.GetStrides(), 
											sample_left_offsets, sample_right_offsets, strides);
		
	for (size_t i=0; i<sample_offsets_.size(); i++)
		output_buffer[output_buffer_offset+i] = static_cast<OutputType>(sample[sample_offsets_[i]]);
		
	return sample_offsets_.size();
}

template <class OutputType, class InputType>
std::shared_ptr< Tensor<OutputType> > PartialTensorDataLoader<OutputType,InputType>::GetSamplesData( const std::vector<size_t>& samples_inds, 
		const std::vector< std::vector<size_t> >& samples_left_offsets, const std::vector< std::vector<size_t> >& samples_right_offsets) const
{
	std::vector<size_t> sample_dims = GetSampleDims();
	for (size_t i=0; i< sample_dims.size(); i++)
		sample_dims[i] -= samples_left_offsets[0][i] + samples_right_offsets[0][i];
		
	std::shared_ptr< Tensor<OutputType> > output_buffer = GetOutputBuffer(samples_inds.size(), sample_dims);

	size_t offset = 0;
	for (size_t i = 0; i<samples_inds.size(); i++)
	{
		size_t sample_ind = samples_inds[i];
		offset += GetSampleData( sample_ind, samples_left_offsets[i], 
			samples_right_offsets[i], *output_buffer, offset);
	}
		
	assert( offset == output_buffer->Numel());

	return output_buffer;
}

#endif