#ifndef CLASSIFICATION_BALANCED_PAIRS_TENSOR_DATA_LOADER_H
#define CLASSIFICATION_BALANCED_PAIRS_TENSOR_DATA_LOADER_H

#include "BaseTensorDataLoader.h"
#include "RandomGenerator.h"

template <class OutputType, class InputType>
class ClassificationBalancedPairsTensorDataLoader : public BaseTensorDataLoader<OutputType, InputType>
{
	typedef std::shared_ptr< Tensor<OutputType> > output_type_tensor_ptr;
	typedef std::shared_ptr< Tensor<InputType> > input_type_tensor_ptr;

	std::vector< std::vector<size_t> > classes_inds;

	size_t GetSamplesPairData( size_t sample1_ind, size_t sample2_ind, Tensor<OutputType>& output_buffer, size_t output_buffer_offset ) const
	{
		Tensor<InputType>& sample1 = GetSample(sample1_ind);
		size_t sample1_numel = sample1.Numel();
		for (size_t i=0; i<sample1_numel; i++)
			output_buffer[output_buffer_offset+i] = static_cast<OutputType>(sample1[i]);
		
		output_buffer_offset+=sample1_numel;
		
		Tensor<InputType>& sample2 = GetSample(sample2_ind);
		size_t sample2_numel = sample2.Numel();
		for (size_t i=0; i<sample2_numel; i++)
			output_buffer[output_buffer_offset+i] = static_cast<OutputType>(sample2[i]);

		return sample1_numel + sample2_numel;
	}
	
	virtual void sub_save(std::ostream& output_stream) const
	{
		throw "Not implemented!";
	}
	
	virtual std::vector<size_t> GetSampleDims(size_t sample_ind = 0) const
	{
		std::vector<size_t> pair_dims =  BaseTensorDataLoader<OutputType, InputType>::GetSampleDims(sample_ind);
		pair_dims[0] *= 2;
		return pair_dims;
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
	
	ClassificationBalancedPairsTensorDataLoader(const std::vector< input_type_tensor_ptr >& input_data, 
		const std::vector< input_type_tensor_ptr >& labels,
		std::string name = "Default") : 
		BaseTensorDataLoader<OutputType, InputType>(input_data, name), classes_inds( labels[0]->Numel() )
	{
		size_t num_samples = labels.size();
		size_t num_labels = labels[0]->Numel();
		for (size_t sample_ind = 0; sample_ind < num_samples; sample_ind++)
		{
			bool label_found = false;
			for (size_t i = 0; i < num_labels; i++)
			{
				if ((*labels[sample_ind])[i] == 1)
				{
					classes_inds[i].push_back( sample_ind );
					label_found = true;
					break;
				}

			}
			if (!label_found)
				throw "Unlabeled samples not supported";
		}
	}

	virtual std::vector< size_t > SelectIndices(size_t num_pairs) const
	{
		std::vector< size_t > out_selected_inds;
		out_selected_inds.reserve(2*num_pairs);

		size_t offset = 0;
		for (size_t pair_ind = 0; pair_ind<num_pairs; pair_ind++)
		{
			// todo: remove debug change
			/*size_t num_samples = GetNumSamples();
			size_t sample1_ind = static_cast<size_t>( RandomGenerator::GetUniformInt( 0, static_cast<int>(num_samples)-1) );*/
			
			size_t sample1_class = RandomGenerator::GetUniformInt( 1, 4 );
			size_t sample1_class_ind = static_cast<size_t>( RandomGenerator::GetUniformInt( 0, classes_inds[sample1_class-1].size()-1 ) );
			size_t sample1_ind = classes_inds[sample1_class-1][ sample1_class_ind ];

			size_t sample2_class = RandomGenerator::GetUniformInt( 1, 4 );
			size_t sample2_class_ind = static_cast<size_t>( RandomGenerator::GetUniformInt( 0, classes_inds[sample2_class-1].size()-1 ) );
			size_t sample2_ind = classes_inds[sample2_class-1][ sample2_class_ind ];

			// random swap
			if (RandomGenerator::GetUniformInt(0,1) == 1)
			{
				size_t tmp = sample1_ind;
				sample1_ind = sample2_ind;
				sample2_ind = tmp;
			}
			out_selected_inds.push_back(sample1_ind);
			out_selected_inds.push_back(sample2_ind);

		}

		return out_selected_inds;
	}

	virtual output_type_tensor_ptr GetData(const std::vector<size_t>& samples_inds) const
	{
		output_type_tensor_ptr output_buffer = GetOutputBuffer(samples_inds.size()/2, GetSampleDims());
		size_t offset = 0;
		for (size_t pair_ind = 0; pair_ind<samples_inds.size()/2; pair_ind++)
			offset+=GetSamplesPairData(samples_inds[2*pair_ind], samples_inds[2*pair_ind+1], *output_buffer, offset);
		
		return output_buffer;
	}

};

#endif