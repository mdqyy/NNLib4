#ifndef CONVOLUTIONAL_KERNEL_H
#define CONVOLUTIONAL_KERNEL_H

#include "Kernel.h"

template <class DataType>
class ConvolutionalKernel : public Kernel<DataType>
{
protected:
	void subGetResponse(const Tensor<DataType>& data, size_t dim_num, size_t* data_current_pos, size_t* kernel_current_pos, DataType& res) const;
public:

	virtual void fprop(const Tensor<DataType>& input, Tensor<DataType>& output);

	virtual void bprop(const Tensor<DataType>& input, const Tensor<DataType>& output, 
		Tensor<DataType>& input_gradients, const Tensor<DataType>& upper_gradients);

	virtual void GetGradient(const Tensor<DataType>& input, const Tensor<DataType>& output, 
		const Tensor<DataType>& upper_gradients, Tensor<DataType>& gradient);

	static std::vector<size_t> GetOutputTensorDimensions(const std::vector<size_t>& input_dimensions, 
		const std::vector<size_t>& kernel_dims_sizes, const std::vector<size_t>& kernel_strides);

	virtual DataType GetResponse(const Tensor<DataType>& data, const size_t* data_pos) const;

	virtual size_t GetNumberOfParameters() const;

	ConvolutionalKernel(const Tensor<DataType>& kernel, const std::vector<size_t>& strides);

	virtual std::vector<size_t> GetOutputTensorDimensions(const std::vector<size_t>& input_dimensions) const ;

	virtual std::string GetType() const;

	virtual bool Equals(const Kernel<DataType>& kernel) const;
};


template <class DataType>
bool ConvolutionalKernel<DataType>::Equals(const Kernel<DataType>& kernel) const
{
	if (!Kernel<DataType>::Equals(kernel))
		return false;
	if (GetKernelTensor() != kernel.GetKernelTensor())
		return false;
	return true;
}

template <class DataType>
std::string ConvolutionalKernel<DataType>::GetType() const
{
	return "ConvolutionalKernel";
}

template <class DataType>
std::vector<size_t> ConvolutionalKernel<DataType>::GetOutputTensorDimensions(const std::vector<size_t>& input_dimensions) const
{
	return GetOutputTensorDimensions(input_dimensions, GetKernelDimensions(), GetStrides());
}

template <class DataType>
std::vector<size_t> ConvolutionalKernel<DataType>::GetOutputTensorDimensions(const std::vector<size_t>& input_dimensions, 
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
size_t ConvolutionalKernel<DataType>::GetNumberOfParameters() const
{
	return GetKernelTensor().Numel();
}

template <class DataType>
void ConvolutionalKernel<DataType>::subGetResponse(const Tensor<DataType>& data, size_t dim_num, size_t* data_current_pos, 
												   size_t* kernel_current_pos, DataType& res) const
{
	auto& kernel = GetKernelTensor();
	size_t initial_data_dim_pos = data_current_pos[dim_num];
	for ( size_t i= 0; i<kernel.GetDimensionSize(dim_num); i++ )
	{
		data_current_pos[dim_num] = initial_data_dim_pos+i;
		kernel_current_pos[dim_num] = i;
		if (dim_num==0)
			res += *kernel.GetPtr(kernel_current_pos) * *data.GetPtr(data_current_pos);
		else
			subGetResponse(data, dim_num-1, data_current_pos, kernel_current_pos, res);
	}
	data_current_pos[dim_num] = initial_data_dim_pos;
}

template <class DataType>
DataType ConvolutionalKernel<DataType>::GetResponse(const Tensor<DataType>& data, const size_t* data_pos) const
{
	auto& kernel = GetKernelTensor();
	std::vector<size_t> kernel_current_pos(kernel.NumDimensions());
	std::vector<size_t> current_data_pos(data_pos, data_pos+kernel.NumDimensions());
	DataType res = 0;
	subGetResponse(data, kernel.NumDimensions()-1, current_data_pos.data(), kernel_current_pos.data(), res);
	return res;
}

template <class DataType>
ConvolutionalKernel<DataType>::ConvolutionalKernel(const Tensor<DataType>& kernel, const std::vector<size_t>& strides) : Kernel<DataType>(kernel, strides)
{

}

template <class DataType>
void ConvolutionalKernel<DataType>::fprop(const Tensor<DataType>& input, Tensor<DataType>& output)
{
	output.SetZeros();
	auto& kernel_offsets = GetKernelOffsets(input.GetDimensions());
	auto& valid_tensor_positions = GetValidTensorPositions(input.GetDimensions());
	const auto& kernel = GetKernelTensor();
	const DataType* kernel_start = kernel.GetStartPtr();
	DataType* output_start_ptr = output.GetStartPtr();

	const size_t* valid_offsets_start_ptr = valid_tensor_positions.data();
	const size_t* valid_offsets_stop_ptr = valid_offsets_start_ptr+valid_tensor_positions.size();

	const DataType* input_start_ptr = input.GetStartPtr();

	for (size_t offset_ind = 0; offset_ind < kernel_offsets.size(); offset_ind++)
	{
		DataType* output_pos = output_start_ptr;
		DataType kernel_val = *(kernel_start+offset_ind);
		int kernel_offset = kernel_offsets[offset_ind];
		for (const size_t* input_offset_ptr = valid_offsets_start_ptr; input_offset_ptr<valid_offsets_stop_ptr; input_offset_ptr++,output_pos++)
			*output_pos += *(input_start_ptr + *input_offset_ptr+kernel_offset) * kernel_val;
	}
}

template <class DataType>
void ConvolutionalKernel<DataType>::bprop(const Tensor<DataType>& input, const Tensor<DataType>& output, 
											 Tensor<DataType>& input_gradients, const Tensor<DataType>& output_gradients)
{
	auto& kernel_offsets = GetKernelOffsets(input.GetDimensions());
	auto& valid_tensor_positions = GetValidTensorPositions(input.GetDimensions());
	auto& kernel = GetKernelTensor();
	const DataType* kernel_start = kernel.GetStartPtr();
	const size_t* valid_offsets_start_ptr = valid_tensor_positions.data();
	const size_t* valid_offsets_stop_ptr = valid_offsets_start_ptr+valid_tensor_positions.size();
	DataType* input_gradients_start_ptr = input_gradients.GetStartPtr();
	const DataType* output_gradients_start_ptr = output_gradients.GetStartPtr();

	for (size_t offset_ind = 0; offset_ind < kernel_offsets.size(); offset_ind++)
	{
		const DataType* output_gradients_pos = output_gradients_start_ptr;
		DataType kernel_val = *(kernel_start+offset_ind);
		int kernel_offset = kernel_offsets[offset_ind];
		for (const size_t* input_offset_ptr = valid_offsets_start_ptr; input_offset_ptr<valid_offsets_stop_ptr; input_offset_ptr++,output_gradients_pos++)
			*(input_gradients_start_ptr + *input_offset_ptr+kernel_offset) += *output_gradients_pos * kernel_val;
	}
}

template <class DataType>
void ConvolutionalKernel<DataType>::GetGradient(const Tensor<DataType>& input, const Tensor<DataType>& output, 
													const Tensor<DataType>& output_gradients, Tensor<DataType>& gradient)
{
	auto& kernel_offsets = GetKernelOffsets(input.GetDimensions());
	auto& valid_tensor_positions = GetValidTensorPositions(input.GetDimensions());
	auto& kernel = GetKernelTensor();
	DataType* gradient_start_ptr = gradient.GetStartPtr();
	const DataType* kernel_start = kernel.GetStartPtr();
	const size_t* valid_offsets_start_ptr = valid_tensor_positions.data();
	const size_t* valid_offsets_stop_ptr = valid_offsets_start_ptr+valid_tensor_positions.size();
	const DataType* input_start_ptr = input.GetStartPtr();
	const DataType* output_gradients_start_ptr = output_gradients.GetStartPtr();

	for (size_t offset_ind = 0; offset_ind < kernel_offsets.size(); offset_ind++)
	{
		const DataType* output_gradients_pos = output_gradients_start_ptr;
		DataType* gradient_ptr = gradient_start_ptr+offset_ind;
		DataType kernel_val = *(kernel_start+offset_ind);
		int kernel_offset = kernel_offsets[offset_ind];
		for (const size_t* input_offset_ptr = valid_offsets_start_ptr; input_offset_ptr<valid_offsets_stop_ptr; input_offset_ptr++,output_gradients_pos++)
			*gradient_ptr += *(input_start_ptr + *input_offset_ptr+kernel_offset) * *output_gradients_pos;
	}
}

#endif