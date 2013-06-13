#ifndef RANDOM_SHIFT_PARTIAL_TENSOR_DATA_LOADER_H
#define RANDOM_SHIFT_PARTIAL_TENSOR_DATA_LOADER_H

#include "PartialTensorDataLoader.h"
#include "RandomGenerator.h"

template <class OutputType, class InputType>
class RandomShiftPartialTensorDataLoader : public PartialTensorDataLoader<OutputType, InputType>
{
	std::vector<size_t> max_shifts_;
		
protected:
	virtual void sub_save(std::ostream& output_stream) const
	{
		PartialTensorDataLoader<OutputType, InputType>::sub_save(output_stream);
		output_stream<<Converter::ConvertVectorToString(max_shifts_)<<"\n";
	}

public:

	virtual bool Equals(BaseTensorDataLoader<OutputType, InputType>& data_loader)
	{
		if ( !PartialTensorDataLoader<OutputType, InputType>::Equals(data_loader) )
			return false;
		RandomShiftPartialTensorDataLoader<OutputType, InputType>* other_loader = 
			static_cast< RandomShiftPartialTensorDataLoader<OutputType, InputType>* >(&data_loader);
		return other_loader->max_shifts_ == max_shifts_;
	}
	
	static std::shared_ptr< RandomShiftPartialTensorDataLoader<OutputType, InputType> > Create(std::string name, 
		const std::vector< std::shared_ptr< Tensor<InputType> > >& data, std::istream& input_stream)
	{
		std::string max_shifts_str;
		std::getline(input_stream, max_shifts_str);
		std::vector<size_t> max_shifts = Converter::StringToVector<size_t>(max_shifts_str);
		return std::shared_ptr< RandomShiftPartialTensorDataLoader<OutputType, InputType> >( 
			new RandomShiftPartialTensorDataLoader<OutputType, InputType>( data, max_shifts, name) );
	}

	virtual std::string GetType() const
	{
		return "RandomShiftPartialTensorDataLoader";
	}

	RandomShiftPartialTensorDataLoader(const std::vector< std::shared_ptr< Tensor<InputType> > >& data, 
		const std::vector<size_t>& max_shifts, std::string name = "Default") : 
		PartialTensorDataLoader<OutputType, InputType>(data, name), max_shifts_(max_shifts)
	{

	}

	virtual std::shared_ptr< Tensor<OutputType> > GetData(const std::vector<size_t>& samples_inds) const;
};

template <class OutputType, class InputType>
std::shared_ptr< Tensor<OutputType> > RandomShiftPartialTensorDataLoader<OutputType,InputType>::GetData(const std::vector<size_t>& samples_inds) const
{
	std::vector< std::vector<size_t> > samples_left_offsets(samples_inds.size());
	std::vector< std::vector<size_t> > samples_right_offsets(samples_inds.size());
	for (size_t i=0; i<samples_inds.size(); i++)
	{
		assert( GetSampleDims(samples_inds[i]).size() == max_shifts_.size() );
		for (size_t dim = 0; dim < max_shifts_.size(); dim++)
		{
			size_t dim_offset = RandomGenerator::GetUniformInt(0, max_shifts_[dim]);
			samples_left_offsets[i].push_back(dim_offset);
			samples_right_offsets[i].push_back(max_shifts_[dim] - dim_offset);
		}
	}
		
	return GetSamplesData( samples_inds, samples_left_offsets, samples_right_offsets);
}

#endif