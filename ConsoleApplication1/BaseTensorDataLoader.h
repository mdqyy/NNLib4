#ifndef BASE_TENSOR_DATA_LOADER_H
#define BASE_TENSOR_DATA_LOADER_H

#include <cassert>
#include  "ITensorDataLoader.h"
#include "CashedTensor.h"
#include "TensorIO.h"

template <class OutputType, class InputType>
class BaseTensorDataLoader : public ITensorDataLoader<OutputType>
{
	typedef std::shared_ptr< Tensor<OutputType> > output_type_tensor_ptr;
	typedef std::shared_ptr< Tensor<InputType> > input_type_tensor_ptr;
	
	std::string name_;
	virtual void sub_save(std::ostream& output_stream) const = 0;
	mutable CashedTensor<OutputType> cashed_data_buffer;
	mutable std::vector< input_type_tensor_ptr > data_;
protected:
	Tensor<InputType>& GetSample(size_t ind) const
	{
		return  *data_[ind];
	}

	virtual std::vector<size_t> GetSampleDims(size_t sample_ind = 0) const
	{
		return data_[sample_ind]->GetDimensions();
	}

	std::shared_ptr< Tensor<OutputType> > GetOutputBuffer(size_t num_cases, std::vector<size_t> sample_dims) const
	{
		sample_dims.push_back(num_cases);
		cashed_data_buffer.Update(sample_dims);
		return cashed_data_buffer();
	}
public:

	virtual std::string GetType() const = 0;

	std::string GetName() const
	{
		return name_;
	}

	void Save(std::ostream& output_stream) const
	{
		output_stream<<"DataLoader"<<"\n";
		output_stream<<GetType()<<"\n";
		output_stream<<GetName()<<"\n";
		SaveDataset(data_, output_stream);
		sub_save(output_stream);
	}
	
	virtual std::vector< size_t > SelectIndices(size_t num_generated_inds) const
	{
		size_t total_num_samples = GetNumSamples();
		if (num_generated_inds>total_num_samples)
		{
			std::vector<size_t> res(total_num_samples);
			for (size_t i=0; i<total_num_samples; i++)
				res[i] = i;
			return res;
		}
		else
		{
			std::vector<size_t> res(num_generated_inds);
			for (size_t i=0; i<num_generated_inds; i++)
				res[i] = RandomGenerator::GetUniformInt(0, total_num_samples-1);
			return res;
		}
	}

	BaseTensorDataLoader(const std::vector< input_type_tensor_ptr >& data, std::string name) : data_(data)
	{
		name_ = name;
	}

	virtual size_t GetNumSamples() const
	{
		return data_.size();
	}

	virtual bool Equals(BaseTensorDataLoader<OutputType, InputType>& data_loader)
	{
		if ( data_loader.GetType() != GetType() )
			return false;
		
		if ( data_loader.GetName() != GetName() )
			return false;

		for (size_t i=0; i<data_.size(); i++)
			if ( *data_[i] != *data_loader.data_[i])
				return false;

		return true;
	}

};

#endif