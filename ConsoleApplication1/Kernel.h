#ifndef KERNEL_H
#define KERNEL_H

#include "Tensor.h"
#include <assert.h>
#include <vector>

template <class DataType>
class Kernel
{
	// for reusing buffers without allocating memory
	std::vector<size_t> cashed_valid_tensor_positions_;
	std::vector<size_t> cashed_dimensions_;
	void UpdateCash(const Tensor<DataType>& input);
	// when a kernel is applied at a point,  we need to iterate over its 
	// dimensions and look at the corresponding positions of the tensor being processed.
	// We could use many for loops for this, but since number of tensor's dimensions is not known in advance, it is too difficult.
	// We use the fact that each tensor occupies contiguous parts of memory. As a result for each kernel point we can get the corresponding 
	// tensor point at a given position by always substracting the same 1d offset. This approach is not applicable near the borders of tensors, where 
	// linear offset may "overflow" the dimension of the tensor and point to the wrong data
	std::vector<int> cashed_kernel_offsets_;
	
	std::vector<size_t> strides_;
	// kernel parameters
	Tensor<DataType> kernel_;

	std::vector<int> ComputeKernelOffsetsInds(const std::vector<size_t>& input_dims);

	void UpdateCash(std::vector<size_t>& input_dims);

protected:
	std::vector<size_t>& GetValidTensorPositions(std::vector<size_t>& input_dims);
	
	std::vector<int>& GetKernelOffsets(std::vector<size_t>& input_dims)
	{
		UpdateCash(input_dims);
		return cashed_kernel_offsets_;
	}

public:
	void SetNewParameters(DataType* params_ptr);

	Kernel(const Tensor<DataType>& kernel, const std::vector<size_t>& strides);

	virtual std::vector<size_t> GetOutputTensorDimensions(const std::vector<size_t>& input_dimensions) const = 0;

	virtual size_t GetNumberOfParameters() const = 0;

	virtual DataType GetResponse(const Tensor<DataType>& data, const size_t* data_pos) const = 0;
	
	const std::vector<size_t> GetStrides() const
	{
		return strides_;
	}

	std::vector<size_t> GetKernelDimensions() const
	{
		return kernel_.GetDimensions();
	}

	const Tensor<DataType>& GetKernelTensor() const 
	{
		return kernel_;
	}

	Tensor<DataType>& GetKernelTensor() 
	{
		return kernel_;
	}

	virtual bool Equals(const Kernel<DataType>& kernel) const;

	virtual std::string GetType() const = 0;

	// output is not guaranteed to be zero
	virtual void fprop(const Tensor<DataType>& input, Tensor<DataType>& output) = 0;
	
	virtual void bprop(const Tensor<DataType>& input, const Tensor<DataType>& output, Tensor<DataType>& input_gradients, const Tensor<DataType>& upper_gradients) = 0;

	virtual void GetGradient(const Tensor<DataType>& input, const Tensor<DataType>& output, const Tensor<DataType>& upper_gradients, Tensor<DataType>& gradient) = 0;
};

template <class DataType>
bool Kernel<DataType>::Equals(const Kernel<DataType>& kernel) const
{
	if (GetType() != kernel.GetType())
		return false;
	if (strides_ != kernel.strides_)
		return false;
	return true;
}

template <class DataType>
std::vector<size_t>& Kernel<DataType>::GetValidTensorPositions(std::vector<size_t>& input_dims)
{
	UpdateCash(input_dims);
	return cashed_valid_tensor_positions_;
}

template <class DataType>
void Kernel<DataType>::SetNewParameters(DataType* params_ptr)
{
	this->kernel_.SetDataPtr(params_ptr);
}

template <class DataType>
Kernel<DataType>::Kernel(const Tensor<DataType>& kernel_params, const std::vector<size_t>& strides) : kernel_(kernel_params), strides_(strides)
{
}

template <class DataType>
std::vector<int> Kernel<DataType>::ComputeKernelOffsetsInds(const std::vector<size_t>& input_dimensions)
{
	auto input_dims = input_dimensions;
	std::vector<size_t> kernel_dimensions = kernel_.GetDimensions();
	while (input_dims.size()<kernel_dimensions.size())
		input_dims.push_back(1);
	std::vector<size_t> img_strides = Tensor<DataType>::GetStrides(input_dims);

	size_t kernel_numel = kernel_.Numel();
	std::vector<int> res(kernel_numel);
	for (size_t ind = 0; ind < kernel_numel; ind++)
	{
		std::vector<size_t> offsets = Tensor<DataType>::IndToPos(kernel_dimensions, ind);

		int offset = 0;
		for (size_t dim = 0; dim<kernel_.NumDimensions(); dim++)
			offset += offsets[dim]*img_strides[dim];
		res[ind]=offset;
	}
	return res;
}

template <class DataType>
void Kernel<DataType>::UpdateCash(std::vector<size_t>& input_dims)
{
	if (cashed_dimensions_ != input_dims)
	{
		std::vector<size_t> left_margins(kernel_.NumDimensions());
		std::vector<size_t> right_margins=kernel_.GetDimensions();
		for (size_t i=0; i<right_margins.size(); i++)
			right_margins[i]--;
		
		cashed_dimensions_ = input_dims;
		cashed_valid_tensor_positions_ = Tensor<DataType>::GetValidOffsetsInds( input_dims, 
			Tensor<DataType>::GetStrides(input_dims), left_margins, right_margins, strides_);
		cashed_kernel_offsets_ = ComputeKernelOffsetsInds(input_dims);
	}
}

#endif