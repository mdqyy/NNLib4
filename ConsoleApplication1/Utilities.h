#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>
#include <memory>
#include <string>
#include "Tensor.h"

std::vector<std::string> GetFiles(std::string& dir_path);

std::vector< std::shared_ptr< Tensor<float> > > ReadAiffFiles( std::vector< std::string > files);

template <class DataType>
std::shared_ptr< Tensor<DataType> > GetRandomTensorPtr(std::vector<size_t> tensor_dims, double min_val=-1, double max_val=1)
{
	Tensor<DataType> tensor(tensor_dims);
	for (size_t i=0; i<tensor.Numel(); i++)
		tensor[i] = (DataType)RandomGenerator::GetUniformDouble(min_val, max_val);
	return std::shared_ptr< Tensor<DataType> >( new Tensor<DataType>(tensor) );
}

template <class T>
std::pair< std::vector<T>, std::vector<T> > DeterministicSplitVector( std::vector<T>& vect, double fraction )
{
	size_t num_cases = vect.size();
	size_t split_point = static_cast<size_t>(fraction*num_cases);
	std::pair< std::vector<T>, std::vector<T> > res;
	for (size_t i=0; i<=split_point; i++)
		res.first.push_back( vect[i] );
	for (size_t i=split_point+1; i<num_cases; i++)
		res.second.push_back( vect[i] );
	return res;
}

template <class T>
std::pair< std::vector<T>, std::vector<T> > StochasticSplitVector( std::vector<T>& input_vect, double fraction )
{
	std::vector<T> vect = input_vect;
	RandomShuffleVector(vect);
	size_t num_cases = vect.size();
	size_t split_point = static_cast<size_t>(fraction*num_cases);
	std::pair< std::vector<T>, std::vector<T> > res;
	for (size_t i=0; i<split_point; i++)
		res.first.push_back( vect[i] );
	for (size_t i=split_point; i<num_cases; i++)
		res.second.push_back( vect[i] );
	return res;
}

#endif