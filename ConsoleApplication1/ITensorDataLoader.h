#ifndef TENSOR_DATA_LOADER_H
#define TENSOR_DATA_LOADER_H

#include <iostream>
#include <memory>
#include "Tensor.h"

template <class T>
class ITensorDataLoader
{
public:
	virtual size_t GetNumSamples() const = 0;
	
	virtual std::shared_ptr< Tensor<T> > GetData(const std::vector<size_t>& samples_inds) const = 0;
	
	virtual std::vector< size_t > SelectIndices(size_t num_pairs) const = 0;

	virtual ~ITensorDataLoader()
	{

	}
};

#endif