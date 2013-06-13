#ifndef FIXED_SHIFT_PARTIAL_TENSOR_DATA_LOADER_H
#define FIXED_SHIFT_PARTIAL_TENSOR_DATA_LOADER_H

#include "PartialTensorDataLoader.h"
#include "Converter.h"

template <class OutputType, class InputType>
class FixedShiftPartialTensorDataLoader : public PartialTensorDataLoader<OutputType, InputType>
{
	std::vector<size_t> left_shifts_;
	std::vector<size_t> right_shifts_;

protected:
	virtual void sub_save(std::ostream& output_stream) const
	{
		PartialTensorDataLoader<OutputType, InputType>::sub_save(output_stream);
		output_stream<<Converter::ConvertVectorToString(left_shifts_)<<"\n";
		output_stream<<Converter::ConvertVectorToString(right_shifts_)<<"\n";
	}

public:
	
	static std::shared_ptr< FixedShiftPartialTensorDataLoader<OutputType, InputType> > Create(std::string name, 
		const std::vector< std::shared_ptr< Tensor<InputType> > >& data, std::istream& input_stream)
	{
		std::string left_shifts_str;
		std::getline(input_stream, left_shifts_str);
		std::vector<size_t> left_shifts = Converter::StringToVector<size_t>(left_shifts_str);
		std::string right_shifts_str;
		std::getline(input_stream, right_shifts_str);
		std::vector<size_t> right_shifts = Converter::StringToVector<size_t>(right_shifts_str);
		return std::shared_ptr< FixedShiftPartialTensorDataLoader<OutputType, InputType> >( 
			new FixedShiftPartialTensorDataLoader<OutputType, InputType>( data, left_shifts, right_shifts, name) );
	}
	
	virtual std::string GetType() const
	{
		return "FixedShiftPartialTensorDataLoader";
	}

	FixedShiftPartialTensorDataLoader(const std::vector< std::shared_ptr< Tensor<InputType> > >& data, 
		const std::vector<size_t>& left_shifts, const std::vector<size_t>& right_shifts, std::string name = "Default") : 
		PartialTensorDataLoader<OutputType, InputType>(data, name), left_shifts_(left_shifts), right_shifts_(right_shifts)
	{

	}

	virtual bool Equals(BaseTensorDataLoader<OutputType, InputType>& data_loader)
	{
		if ( !PartialTensorDataLoader<OutputType, InputType>::Equals(data_loader) )
			return false;
		FixedShiftPartialTensorDataLoader<OutputType, InputType>* other_loader = 
			static_cast< FixedShiftPartialTensorDataLoader<OutputType, InputType>* >(&data_loader);
		if (other_loader->left_shifts_ != left_shifts_)
			return false;
		return other_loader->right_shifts_ == right_shifts_;
	}

	virtual std::shared_ptr< Tensor<OutputType> > GetData(const std::vector<size_t>& samples_inds) const;
};

template <class OutputType, class InputType>
std::shared_ptr< Tensor<OutputType> > FixedShiftPartialTensorDataLoader<OutputType,InputType>::GetData(const std::vector<size_t>& samples_inds) const
{
	std::vector< std::vector<size_t> > samples_left_offsets(samples_inds.size());
	std::vector< std::vector<size_t> > samples_right_offsets(samples_inds.size());
	for (size_t i=0; i<samples_inds.size(); i++)
	{
		std::vector<size_t> sample_dims = GetSampleDims(samples_inds[i]);
		for (size_t dim = 0; dim < sample_dims.size(); dim++)
		{
			size_t dim_offset = left_shifts_[dim];
			samples_left_offsets[i].push_back(dim_offset);
			samples_right_offsets[i].push_back(right_shifts_[dim]);
		}
	}
		
	return GetSamplesData( samples_inds, samples_left_offsets, samples_right_offsets);
}

#endif