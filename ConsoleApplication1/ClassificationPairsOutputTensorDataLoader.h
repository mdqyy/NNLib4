#ifndef CLASSIFICATION_PAIRS_OUTPUT_TENSOR_DATA_LOADER_H
#define CLASSIFICATION_PAIRS_OUTPUT_TENSOR_DATA_LOADER_H

#include "BaseTensorDataLoader.h"
#include "RandomGenerator.h"

template <class OutputType, class InputType>
class ClassificationPairsOutputTensorDataLoader : public BaseTensorDataLoader<OutputType, InputType>
{
	typedef std::shared_ptr< Tensor<OutputType> > output_type_tensor_ptr;
	typedef std::shared_ptr< Tensor<InputType> > input_type_tensor_ptr;

	size_t GetSamplesPairData( size_t sample1_ind, size_t sample2_ind, Tensor<OutputType>& output_buffer, size_t output_buffer_offset ) const
	{
		Tensor<InputType>& sample1 = GetSample(sample1_ind);
		Tensor<InputType>& sample2 = GetSample(sample2_ind);

		if ( sample1 == sample2 )
		{
			output_buffer[output_buffer_offset] = 1;
			output_buffer[output_buffer_offset+1] = 0;
		}
		else
		{
			output_buffer[output_buffer_offset] = 0;
			output_buffer[output_buffer_offset+1] = 1;
		}

		return 2;
	}
	
	virtual void sub_save(std::ostream& output_stream) const
	{
		throw "Not implemented!";
	}
	
	virtual std::vector<size_t> GetSampleDims(size_t sample_ind = 0) const
	{
		std::vector<size_t> label_dims(1,2);
		return label_dims;
	}

public:

	virtual std::string GetType() const
	{
		return "ClassificationBalancedPairsTensorDataLoader";
	}

	virtual bool Equals(BaseTensorDataLoader<OutputType, InputType>& data_loader)
	{
		throw "Not implemented!";
		return false;
	}
	
	ClassificationPairsOutputTensorDataLoader(const std::vector< input_type_tensor_ptr >& labels_data,
		std::string name = "Default") : 
		BaseTensorDataLoader<OutputType, InputType>(labels_data, name)
	{
	}

	virtual output_type_tensor_ptr GetData(const std::vector<size_t>& samples_inds) const
	{
		output_type_tensor_ptr output_buffer = GetOutputBuffer(samples_inds.size()/2, GetSampleDims());
		size_t offset = 0;
		for (size_t sample_ind = 0; sample_ind<samples_inds.size()/2; sample_ind++)
			offset+=GetSamplesPairData(samples_inds[2*sample_ind], samples_inds[2*sample_ind+1], *output_buffer, offset);
		return output_buffer;
	}
};

#endif