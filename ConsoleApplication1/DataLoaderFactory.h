#ifndef DATA_LOADER_FACTORY_H
#define DATA_LOADER_FACTORY_H

#include "IOTreeNode.h"
#include "FullTensorDataLoader.h"
#include "RandomShiftPartialTensorDataLoader.h"
#include "FixedShiftPartialTensorDataLoader.h"

class UnknownDataLoaderType : public std::runtime_error 
{
public:
	UnknownDataLoaderType(std::string const& s) : std::runtime_error(s)
    {
	}
};

class DataLoaderFactory
{
public:
	template <class OutputType, class InputType>
	static std::shared_ptr< BaseTensorDataLoader<OutputType, InputType> > GetDataLoader(std::istream& input_stream)
	{
		std::string end_of_line_char_str;
		std::string category;
		input_stream>>category;
		std::getline(input_stream, end_of_line_char_str); // remove end of line character
		assert( category == "DataLoader" );
		
		std::string type;
		input_stream>>type;
		std::getline(input_stream, end_of_line_char_str); // remove end of line character
		
		std::string name;
		input_stream>>name;
		std::getline(input_stream, end_of_line_char_str); // remove end of line character

		std::vector< std::shared_ptr< Tensor<InputType> > > data;
		LoadDataset<InputType>(input_stream, data);

		if (type == "FullTensorDataLoader")
			return FullTensorDataLoader<OutputType, InputType>::Create(name, data, input_stream);
		else if (type == "RandomShiftPartialTensorDataLoader")
			return RandomShiftPartialTensorDataLoader<OutputType, InputType>::Create(name, data, input_stream);
		else if (type == "FixedShiftPartialTensorDataLoader")
			return FixedShiftPartialTensorDataLoader<OutputType, InputType>::Create(name, data, input_stream);
		else 
			throw UnknownDataLoaderType(type);
	}
};

#endif