#ifndef SEMISUPERVISED_TENSOR_DATA_LOADER_H
#define SEMISUPERVISED_TENSOR_DATA_LOADER_H

#include "BaseTensorDataLoader.h"
#include "RandomGenerator.h"

template <class OutputType, class InputType>
class SemisupervisedTensorDataLoader : public BaseTensorDataLoader<OutputType, InputType>
{
	typedef std::shared_ptr< Tensor<OutputType> > output_type_tensor_ptr;
	typedef std::shared_ptr< Tensor<InputType> > input_type_tensor_ptr;

	std::vector<size_t> labeled_inds_;
	std::vector<size_t> unlabeled_inds_;
	double supervised_probability_;

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
		throw "Not implemented!";
	}

public:

	virtual std::string GetType() const
	{
		return "SemisupervisedTensorDataLoader";
	}

	virtual bool Equals(BaseTensorDataLoader<OutputType, InputType>& data_loader)
	{
		throw "Not implemented!";
		return false;
	}
	
	SemisupervisedTensorDataLoader(const std::vector< input_type_tensor_ptr >& input_data, 
		const std::vector< input_type_tensor_ptr >& labels, double supervised_probability,
		std::string name = "Default") : 
		BaseTensorDataLoader<OutputType, InputType>(input_data, name), supervised_probability_( supervised_probability )
	{
		size_t num_samples = labels.size();
		size_t num_labels = labels[0]->Numel();
		for (size_t sample_ind = 0; sample_ind < num_samples; sample_ind++)
		{
			bool label_found = false;
			for (size_t i = 0; i < num_labels; i++)
				if ((*labels[sample_ind])[i] == 1)
				{
					labeled_inds_.push_back( sample_ind );
					label_found = true;
					break;
				}
			if (!label_found)
				unlabeled_inds_.push_back( sample_ind );
		}
	}

	virtual std::vector< size_t > SelectIndices(size_t num_samples) const
	{
		std::vector< size_t > out_selected_inds;
		out_selected_inds.reserve(num_samples);

		size_t offset = 0;
		for (size_t i = 0; i<num_samples; i++)
		{
			bool is_labeled = RandomGenerator::GetUniformDouble( 0, 1 ) < supervised_probability_;
			
			size_t sample_ind;
			if (is_labeled)
				sample_ind = labeled_inds_[ RandomGenerator::GetUniformInt( 0, labeled_inds_.size()-1 ) ];
			else
				sample_ind = unlabeled_inds_[ RandomGenerator::GetUniformInt( 0, unlabeled_inds_.size()-1 ) ];

			out_selected_inds.push_back(sample_ind);

		}

		return out_selected_inds;
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