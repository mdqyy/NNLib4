#ifndef CASHED_TENSOR_H
#define CASHED_TENSOR_H

#include <vector>
#include <memory>
#include "Tensor.h"

template <class T>
class CashedTensor
{
private:
	// when shallow copy is made we don't want to create a copy of the vector and tensor
	std::shared_ptr< std::vector<T> > data_; 
	std::shared_ptr< Tensor<T> > cashed_tensor_;
public:

	CashedTensor() : data_( new std::vector<T>() ), cashed_tensor_(new Tensor<T>(0, std::vector<size_t>()))
	{
	}

	bool Update(const std::vector<size_t>& dims);

	std::shared_ptr< Tensor<T> > operator()()
	{
		return cashed_tensor_;
	}
};

template <class T>
bool CashedTensor<T>::Update(const std::vector<size_t>& dims)
{
	if (!cashed_tensor_->DimensionsEqual(dims))
	{
		if ( Tensor<T>::Numel(dims) > data_->size())
			data_->resize(Tensor<T>::Numel(dims));
		cashed_tensor_ = std::shared_ptr< Tensor<T> >( new Tensor<T>(data_->data(), dims) );
		cashed_tensor_->SetZeros();
		return true;
	}
	return false;
}

#endif