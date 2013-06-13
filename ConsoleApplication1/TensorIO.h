#ifndef TENSOR_IO_H
#define TENSOR_IO_H

#include <iostream>
#include <fstream>
#include "IOTreeNode.h"
#include "Tensor.h"
#include "Converter.h"

template< class T>
std::shared_ptr<IOTreeNode> GetTensorState(const Tensor<T>& tensor)
{
	std::shared_ptr< IOTreeNode> parameters_node = std::shared_ptr< IOTreeNode>( new IOTreeNode() );
	parameters_node->attributes().AppendEntry( "dims", Converter::ConvertVectorToString( tensor.GetDimensions() ) );
	parameters_node->attributes().AppendEntry( "parameters", Converter::ConvertArrayToString(tensor.GetStartPtr(), tensor.Numel()) );
	return parameters_node;
}

template< class T>
std::shared_ptr< Tensor<T> > CreateTensor(IOTreeNode& node)
{
	std::vector<size_t> dims = Converter::StringToVector<size_t>( node.attributes().GetEntry("dims") );
	std::vector<T> parameters = Converter::StringToVector<T>( node.attributes().GetEntry("parameters") );
	std::shared_ptr< Tensor<T> > res( new Tensor<T>(dims) );
	for (size_t i=0; i<parameters.size(); i++)
		(*res)[i] = parameters[i];
	return res;
}

// saves in string format for portability issues
template< class T>
void SaveDataset(const std::vector< std::shared_ptr< Tensor<T> > >& dataset, std::ostream& output_stream)
{
	output_stream << dataset.size() << "\n";
	for (size_t i=0; i< dataset.size(); i++)
	{
		Tensor<T>& tensor = *dataset[i];
		output_stream << Converter::ConvertVectorToString( tensor.GetDimensions() ) << "\n" << 
			Converter::ConvertArrayToString(tensor.GetStartPtr(), tensor.Numel()) << "\n";
	}
}

template< class T>
void LoadDataset(std::istream& input_stream, std::vector< std::shared_ptr< Tensor<T> > >& output_dataset)
{
	size_t num_samples;
	input_stream>>num_samples;
	std::string end_of_line_char_str;
	std::getline(input_stream, end_of_line_char_str);

	std::string sample_dims_str;
	std::string sample_params_str;
	for (size_t i=0; i< num_samples; i++)
	{
		std::getline(input_stream, sample_dims_str);
		std::vector<size_t> sample_dims = Converter::StringToVector<size_t>( sample_dims_str );
		std::getline(input_stream, sample_params_str);
		std::vector<T> sample_params = Converter::StringToVector<T>( sample_params_str );
		std::shared_ptr< Tensor<T> > params_tensor_ptr( new Tensor<T>( sample_dims ) );
		Tensor<T>& params_tensor = *params_tensor_ptr;
		for (size_t i=0; i< sample_params.size(); i++)
			params_tensor[i] = sample_params[i];
		output_dataset.push_back( params_tensor_ptr );
	}
}

#endif