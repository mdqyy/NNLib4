#ifndef MAX_POOLING_KERNEL_H
#define MAX_POOLING_KERNEL_H

#include "Kernel.h"
#include <algorithm>

template <class DataType>
class MaxPoolingKernel : public Kernel<DataType>
{
protected:
	void subGetResponse(const Tensor<DataType>& data, size_t dim_num,  size_t* data_current_pos,  size_t* kernel_current_pos, DataType& res)  const;
	DataType GetMaxElement(const DataType* input_ptr, const std::vector<int>& kernel_offsets) const;
	size_t GetMaxElementIndex(const DataType* input_ptr, const std::vector<int>& kernel_offsets) const;
public:

	virtual void fprop(const Tensor<DataType>& input, Tensor<DataType>& output);

	virtual void bprop(const Tensor<DataType>& input, const Tensor<DataType>& output, 
		Tensor<DataType>& input_gradients, const Tensor<DataType>& upper_gradients);

	virtual void GetGradient(const Tensor<DataType>& input, const Tensor<DataType>& output, 
		const Tensor<DataType>& upper_gradients, Tensor<DataType>& gradient);

	virtual size_t GetNumberOfParameters() const;

	virtual DataType GetResponse(const Tensor<DataType>& data, const size_t* data_pos) const;

	MaxPoolingKernel(const Tensor<DataType>& kernel, const std::vector<size_t>& strides);

	virtual std::vector<size_t> GetOutputTensorDimensions(const std::vector<size_t>& input_dimensions) const ;

	static std::vector<size_t> GetOutputTensorDimensions(const std::vector<size_t>& input_dimensions, const std::vector<size_t>& kernel_dims_sizes, 
														 const std::vector<size_t>& kernel_strides);

	virtual std::string GetType() const;
};

template <class DataType>
std::string MaxPoolingKernel<DataType>::GetType() const
{
	return "MaxPoolingKernel";
}

template <class DataType>
std::vector<size_t> MaxPoolingKernel<DataType>::GetOutputTensorDimensions(const std::vector<size_t>& input_dimensions) const
{
	return GetOutputTensorDimensions(input_dimensions, GetKernelDimensions(), GetStrides());
}

template <class DataType>
std::vector<size_t> MaxPoolingKernel<DataType>::GetOutputTensorDimensions(const std::vector<size_t>& input_dimensions, 
																const std::vector<size_t>& kernel_dims_sizes, const std::vector<size_t>& kernel_strides)
{
	auto input_dims = input_dimensions;
	while (input_dims.size()<kernel_dims_sizes.size())
		input_dims.push_back(1);
	std::vector<size_t> output_dimensions;
	for (size_t dim = 0; dim < kernel_dims_sizes.size(); dim++)
	{
		int full_output_dim_size = input_dims[dim] - kernel_dims_sizes[dim]+1;
		assert(full_output_dim_size>0);
		output_dimensions.push_back(full_output_dim_size/kernel_strides[dim] + (full_output_dim_size%kernel_strides[dim]>0 ? 1 : 0) );
	}
	
	for (size_t dim = kernel_dims_sizes.size(); dim < input_dims.size(); dim++)
		output_dimensions.push_back(input_dims[dim]);

	return output_dimensions;
}

template <class DataType>
size_t MaxPoolingKernel<DataType>::GetNumberOfParameters() const
{
	return 0;
}

template <class DataType>
void MaxPoolingKernel<DataType>::subGetResponse(const Tensor<DataType>& data, size_t dim_num, 
												 size_t* data_current_pos,  size_t* kernel_current_pos, DataType& res) const
{

	const Tensor<DataType>& kernel = GetKernelTensor();
	size_t initial_data_dim_pos = data_current_pos[dim_num];
	for ( size_t i= 0; i<kernel.GetDimensionSize(dim_num); i++ )
	{
		data_current_pos[dim_num] = initial_data_dim_pos+i;
		kernel_current_pos[dim_num] = i;
		if (dim_num==0)
			res = (std::max)(res, *data.GetPtr(data_current_pos));
		else
			subGetResponse(data, dim_num-1, data_current_pos, kernel_current_pos, res);
	}
	data_current_pos[dim_num] = initial_data_dim_pos;
}

template <class DataType>
DataType MaxPoolingKernel<DataType>::GetResponse(const Tensor<DataType>& data,  const size_t* data_pos) const
{
	const Tensor<DataType>& kernel = GetKernelTensor();
	std::vector<size_t> kernel_current_pos(kernel.NumDimensions());
	std::vector<size_t> current_data_pos(data_pos, data_pos+kernel.NumDimensions());
	DataType res = *data.GetPtr(data_pos);
	subGetResponse(data, kernel.NumDimensions()-1, current_data_pos.data(), kernel_current_pos.data(), res);
	return res;
}

template <class DataType>
MaxPoolingKernel<DataType>::MaxPoolingKernel(const Tensor<DataType>& kernel, const std::vector<size_t>& strides) : Kernel<DataType>(kernel, strides)
{
	std::vector<size_t> params_dims; params_dims.push_back(0);
}

template <class DataType>
DataType MaxPoolingKernel<DataType>::GetMaxElement(const DataType* input_ptr, const std::vector<int>& kernel_offsets) const
{
	DataType max_element = *(input_ptr+kernel_offsets[0]);
	for (size_t offset_ind = 1; offset_ind < kernel_offsets.size(); offset_ind++)
	{
		int kernel_offset = kernel_offsets[offset_ind];
		max_element = (std::max)(max_element,*(input_ptr+kernel_offset));
	}
	return max_element;
}

template <class DataType>
size_t MaxPoolingKernel<DataType>::GetMaxElementIndex(const DataType* input_ptr, const std::vector<int>& kernel_offsets) const
{
	size_t max_ind = 0;
	DataType max_element = *(input_ptr+kernel_offsets[0]);
	for (size_t offset_ind = 1; offset_ind < kernel_offsets.size(); offset_ind++)
	{
		int kernel_offset = kernel_offsets[offset_ind];
		if ( *(input_ptr+kernel_offset) > max_element )
		{
			max_element = *(input_ptr+kernel_offset);
			max_ind = offset_ind;
		}
	}
	return max_ind;
}

template <class DataType>
void MaxPoolingKernel<DataType>::fprop(const Tensor<DataType>& input, Tensor<DataType>& output)
{
	output.SetZeros();
	auto input_dims = input.GetDimensions();
	auto& kernel_offsets = GetKernelOffsets(input.GetDimensions());
	auto& valid_tensor_positions = GetValidTensorPositions(input.GetDimensions());
	const size_t* valid_offsets_start_ptr = valid_tensor_positions.data();
	const size_t* valid_offsets_stop_ptr = valid_offsets_start_ptr+valid_tensor_positions.size();
	const DataType* input_start_ptr = input.GetStartPtr();
	DataType* output_pos = output.GetStartPtr();
	for (const size_t* input_offset_ptr = valid_offsets_start_ptr; input_offset_ptr<valid_offsets_stop_ptr; input_offset_ptr++,output_pos++)
	{
		const DataType* current_input_ptr = input_start_ptr+*input_offset_ptr;
		size_t max_index = GetMaxElementIndex(current_input_ptr, kernel_offsets);
		*output_pos = *(current_input_ptr+kernel_offsets[max_index]);
	}
}

template <class DataType>
void MaxPoolingKernel<DataType>::bprop(const Tensor<DataType>& input, const Tensor<DataType>& output, 
										  Tensor<DataType>& input_gradients, const Tensor<DataType>& output_gradients)
{
	auto& kernel_offsets = GetKernelOffsets(input.GetDimensions());
	auto& valid_tensor_positions = GetValidTensorPositions(input.GetDimensions());
	const size_t* valid_offsets_start_ptr = valid_tensor_positions.data();
	const size_t* valid_offsets_stop_ptr = valid_offsets_start_ptr+valid_tensor_positions.size();
	DataType* input_gradients_start_ptr = input_gradients.GetStartPtr();
	const DataType* input_start_ptr = input.GetStartPtr();
	const DataType* output_gradients_pos = output_gradients.GetStartPtr();
	for (const size_t* input_offset_ptr = valid_offsets_start_ptr; input_offset_ptr<valid_offsets_stop_ptr; input_offset_ptr++,output_gradients_pos++)
	{
		DataType* current_input_gradients_ptr = input_gradients_start_ptr+*input_offset_ptr;

		const DataType* current_input_ptr = input_start_ptr+*input_offset_ptr;
		size_t max_index = GetMaxElementIndex(current_input_ptr, kernel_offsets);
		*(current_input_gradients_ptr+kernel_offsets[max_index]) += *output_gradients_pos;
	}
}

template <class DataType>
void MaxPoolingKernel<DataType>::GetGradient(const Tensor<DataType>& input, const Tensor<DataType>& output, 
												 const Tensor<DataType>& upper_gradients, Tensor<DataType>& gradient)
{
}

#endif