#ifndef Tensor_H
#define Tensor_H

#include <vector>
#include <memory>
#include <cassert>

template <class DataType> class Tensor;

template <class DataType>
bool operator==(const Tensor<DataType>& tensor1, const Tensor<DataType>& tensor2);

template <class DataType>
bool operator!=(const Tensor<DataType>& tensor1, const Tensor<DataType>& tensor2);

template <class DataType>
class Tensor
{
private:
	DataType* data_ptr;
	std::vector<size_t> dimensions;
	std::vector<size_t> dim_strides;
	bool owns_data;
	
	static void Tensor<DataType>::AppendValidTensorOffsets(std::vector<size_t>& valid_tensor_offsets, std::vector<size_t>& tensor_dims, 
													std::vector<size_t>& tensor_strides, size_t current_offset, 
													size_t dim_ind, std::vector<size_t> margins_left, std::vector<size_t> margins_right, std::vector<size_t>& strides);
	void Copy(const Tensor& tensor);
	
public:


	// constructors
	Tensor(DataType* data_ptr, const std::vector<size_t>& dimensions);
	Tensor(const std::vector<size_t>& dimensions);
	Tensor(const Tensor<DataType>& tensor);
	Tensor();
	~Tensor();

	// operators
	DataType& operator[] (size_t nIndex);
	const DataType& operator[] (size_t nIndex) const;
	Tensor<DataType>& operator=(const Tensor<DataType>& tensor);
	friend bool operator== <>(const Tensor<DataType>& tensor1, const Tensor<DataType>& tensor2);
	friend bool operator!= <>(const Tensor<DataType>& tensor1, const Tensor<DataType>& tensor2);

	// data manipulation
	void SetZeros();
	DataType* GetPtr(const size_t* dims);
	const DataType* GetPtr(const size_t* dims) const;
	DataType* GetStartPtr();
	const DataType* GetStartPtr() const;
	void SetDataPtr(DataType* data_ptr);

	// kernel Info
	bool OwnsData() const; // whether it is responsible for deallocation of data
	static std::vector<size_t> GetStrides(const std::vector<size_t>& dimensions);
	std::vector<size_t> GetStrides() const;
	size_t Numel() const;
	size_t GetDimStride(const size_t dim_num) const;
	size_t NumDimensions() const;
	bool DimensionsEqual(const std::vector<size_t>& other_dims) const;
	bool DimensionsEqual(const Tensor<DataType>& tensor) const;
	size_t GetDimensionSize(size_t ind) const;
	std::vector<size_t> GetDimensions() const;
	
	// general operations
	// ... number of elements in a tensor with given dimensions
	static size_t Numel(const std::vector<size_t>& dimensions);

	// ... tensor coordinates for the element with the given index
	static std::vector<size_t> IndToPos(const std::vector<size_t>& dimensions, size_t ind);

	// ... positions where a kernel with the given parameters can be safely placed
	static std::vector<size_t> GetValidOffsetsInds(std::vector<size_t> tensor_dims, std::vector<size_t> tensor_strides, 
												   std::vector<size_t> margins_left, std::vector<size_t> margins_right, std::vector<size_t> strides);

	static void GetValidOffsetsInds(std::vector<size_t>& out_offsets, std::vector<size_t> tensor_dims, std::vector<size_t> tensor_strides, 
												   std::vector<size_t> margins_left, std::vector<size_t> margins_right, std::vector<size_t> strides);
};

template <class DataType>
bool operator==(const Tensor<DataType>& tensor1, const Tensor<DataType>& tensor2)
{
	if (tensor1.GetDimensions() != tensor2.GetDimensions())
		return false;
	if ( tensor1.GetStrides() != tensor2.GetStrides() )
		return false;
	if (tensor1.OwnsData() != tensor2.OwnsData())
		return false;

	// When tensors don't own the data and point to other memory locations it is not considered a problem
	// This is required to test save / load tensor functionality
	size_t numel = tensor1.Numel();
	for (size_t i=0; i<numel; i++)
		if ( abs(tensor1[i] - tensor2[i])>0.000001 )
			return false;
	return true;
}

template <class DataType>
bool operator!=(const Tensor<DataType>& tensor1, const Tensor<DataType>& tensor2)
{
	return !(tensor1 == tensor2);
}

template <class DataType>
std::vector<size_t> Tensor<DataType>::GetStrides() const
{
	return Tensor<DataType>::GetStrides(GetDimensions());
}

template <class DataType>
bool Tensor<DataType>::OwnsData() const
{
	return owns_data;
}

template <class DataType>
void Tensor<DataType>::SetDataPtr(DataType* data_ptr)
{
	if (owns_data)
	{
		delete[] data_ptr;
		owns_data = false;
	}
	this->data_ptr = data_ptr;
}

template <class DataType>
void Tensor<DataType>::AppendValidTensorOffsets(std::vector<size_t>& valid_tensor_offsets, std::vector<size_t>& tensor_dims, 
												std::vector<size_t>& tensor_strides, size_t current_offset, 
												size_t dim_ind, std::vector<size_t> margins_left, std::vector<size_t> margins_right, std::vector<size_t>& strides)
{
	if (dim_ind<=2)
	{
		size_t min_ind0 = margins_left[0];
		size_t max_ind0 = tensor_dims[0]-margins_right[0];
		size_t min_ind1 = margins_left[1];
		size_t max_ind1 = tensor_dims[1]-margins_right[1];
		size_t min_ind2 = margins_left[2];
		size_t max_ind2 = tensor_dims[2]-margins_right[2];

		size_t stride0 = strides[0];
		size_t stride1 = strides[1];
		size_t stride2 = strides[2];
		
		size_t tensor_stride0 = tensor_strides[0];
		size_t tensor_stride1 = tensor_strides[1];
		size_t tensor_stride2 = tensor_strides[2];

		size_t current_offset2 = current_offset+tensor_stride2*min_ind2;
		for (size_t pos2 = min_ind2; pos2<max_ind2; pos2+=stride2)
		{
			size_t current_offset1 = current_offset2+tensor_stride1*min_ind1;
			for (size_t pos1 = min_ind1; pos1<max_ind1; pos1+=stride1)
			{
				size_t current_offset0 = current_offset1+tensor_stride0*min_ind0;
				for (size_t pos0 = min_ind0; pos0<max_ind0; pos0+=stride0)
				{
					valid_tensor_offsets.push_back(current_offset0);
					current_offset0 += stride0*tensor_stride0;
				}
				current_offset1 += stride1*tensor_stride1;
			}
			current_offset2 += stride2*tensor_stride2;
		}
	}
	else
	{	
		size_t min_ind = margins_left[dim_ind];
		size_t max_ind = tensor_dims[dim_ind]-margins_right[dim_ind];
		current_offset += min_ind*tensor_strides[dim_ind];
		for (size_t pos = min_ind; pos<max_ind; pos+=strides[dim_ind])
		{
			Tensor<DataType>::AppendValidTensorOffsets(valid_tensor_offsets, tensor_dims, tensor_strides, 
				current_offset, dim_ind-1, margins_left, margins_right, strides);
			current_offset += strides[dim_ind]*tensor_strides[dim_ind];
		}
	}
}

template <class DataType>
std::vector<size_t> Tensor<DataType>::GetValidOffsetsInds(std::vector<size_t> tensor_dims, std::vector<size_t> tensor_strides, 
												   std::vector<size_t> margins_left, std::vector<size_t> margins_right, std::vector<size_t> strides)
{
	while ( tensor_dims.size()<3)
	{
		tensor_dims.push_back(1);
		tensor_strides.push_back(0);
	}

	while ( margins_left.size() < tensor_dims.size())
	{
		margins_left.push_back(0);
		margins_right.push_back(0);
		strides.push_back(strides[strides.size()-1]);
	}

	std::vector<size_t> res;
	Tensor<DataType>::AppendValidTensorOffsets(res, tensor_dims, tensor_strides, 0, tensor_dims.size()-1, margins_left, margins_right, strides);
	return res;
}

template <class DataType>
void Tensor<DataType>::GetValidOffsetsInds(std::vector<size_t>& out_offsets, std::vector<size_t> tensor_dims, std::vector<size_t> tensor_strides, 
												   std::vector<size_t> margins_left, std::vector<size_t> margins_right, std::vector<size_t> strides)
{
	while ( tensor_dims.size()<3)
	{
		tensor_dims.push_back(1);
		tensor_strides.push_back(0);
	}

	while ( margins_left.size() < tensor_dims.size())
	{
		margins_left.push_back(0);
		margins_right.push_back(0);
		strides.push_back(strides[strides.size()-1]);
	}

	Tensor<DataType>::AppendValidTensorOffsets(out_offsets, tensor_dims, tensor_strides, 0, tensor_dims.size()-1, margins_left, margins_right, strides);
}

template <class DataType>
bool Tensor<DataType>::DimensionsEqual(const std::vector<size_t>& other_dims) const
{
	return dimensions == other_dims;
}

template <class DataType>
bool Tensor<DataType>::DimensionsEqual(const Tensor<DataType>& tensor) const
{
	return dimensions == tensor.dimensions;
}

template <class DataType>
std::vector<size_t> Tensor<DataType>::GetStrides(const std::vector<size_t>& dimensions)
{
	std::vector<size_t> res;
	res.push_back(1);
	for (size_t dim_ind = 1; dim_ind<dimensions.size(); dim_ind++)
		res.push_back(res[res.size()-1]*dimensions[dim_ind-1]);
	return res;
}

template <class DataType>
Tensor<DataType>::Tensor(DataType* data_ptr, const std::vector<size_t>& dimensions) : owns_data(false)
{
	this->data_ptr = data_ptr;
	this->dimensions = dimensions;
	dim_strides = Tensor<DataType>::GetStrides(dimensions);
}

template <class DataType>
Tensor<DataType>::Tensor(const std::vector<size_t>& dimensions) : owns_data(true)
{
	this->dimensions = dimensions;
	dim_strides = Tensor<DataType>::GetStrides(dimensions);
	data_ptr = new DataType[this->Numel()];
	this->SetZeros();
}

template <class DataType>
Tensor<DataType>::Tensor() : owns_data(false)
{
	this->data_ptr = 0;
}

template <class DataType>
void Tensor<DataType>::Copy(const Tensor& tensor)
{
	this->dimensions = tensor.dimensions;
	this->dim_strides = tensor.dim_strides;
	if (this->owns_data)
		delete[] this->data_ptr;
	this->owns_data = tensor.owns_data;

	if (this->owns_data)
	{
		this->data_ptr = new DataType[this->Numel()];
		for (size_t i=0; i<tensor.Numel(); i++)
			this->data_ptr[i] = tensor.data_ptr[i];
	}
	else
		this->data_ptr = tensor.data_ptr;
}

template <class DataType>
Tensor<DataType>::Tensor(const Tensor<DataType>& tensor) : owns_data(false), data_ptr(0)
{
	Copy(tensor);
}

template <class DataType>
Tensor<DataType>& Tensor<DataType>::operator=(const Tensor& tensor)
{
	if (&tensor == this)
		return *this;
	Copy(tensor);
	return *this;
}

template <class DataType>
size_t Tensor<DataType>::Numel() const
{
	size_t numel = 1;
	for (size_t dim_ind = 0; dim_ind<dimensions.size(); dim_ind++)
		numel*=dimensions[dim_ind];
	return numel;
}

template <class DataType>
size_t Tensor<DataType>::GetDimStride(size_t dim_num) const
{
	if (dim_num<NumDimensions())
		return dim_strides[dim_num];
	else
		return Numel();
}

template <class DataType>
DataType* Tensor<DataType>::GetPtr(const size_t* dims)
{
	size_t offset = 0;
	for (size_t dim_ind = 0; dim_ind<dimensions.size(); dim_ind++)
		offset+=GetDimStride(dim_ind)*dims[dim_ind];
	return data_ptr+offset;
}

template <class DataType>
const DataType* Tensor<DataType>::GetPtr(const size_t* dims) const
{
	size_t offset = 0;
	for (size_t dim_ind = 0; dim_ind<dimensions.size(); dim_ind++)
		offset+=GetDimStride(dim_ind)*dims[dim_ind];
	return data_ptr+offset;
}

template <class DataType>
size_t Tensor<DataType>::Numel(const std::vector<size_t>& dimensions)
{
	size_t numel = 1;
	for (size_t dim_ind = 0; dim_ind<dimensions.size(); dim_ind++)
		numel*=dimensions[dim_ind];
	return numel;
}

template <class DataType>
size_t Tensor<DataType>::NumDimensions() const
{
	return dimensions.size();
}

template <class DataType>
size_t Tensor<DataType>::GetDimensionSize(size_t ind) const
{
	if (ind<dimensions.size())
		return dimensions[ind];
	else
		return 1;
}

template <class DataType>
std::vector<size_t> Tensor<DataType>::IndToPos(const std::vector<size_t>& dimensions, size_t ind)
{
	std::vector<size_t> res(dimensions.size());
	std::vector<size_t> dims_strides = Tensor<DataType>::GetStrides(dimensions);
	for (int dim_ind = (int)dimensions.size()-1; dim_ind>=0; dim_ind--)
	{
		res[dim_ind]= ind / dims_strides[dim_ind];
		ind -= (ind / dims_strides[dim_ind]) * dims_strides[dim_ind];
	}
	return res;
}

template <class DataType>
void Tensor<DataType>::SetZeros()
{
	std::fill(data_ptr, data_ptr+Numel(), static_cast<DataType>(0));
}

template <class DataType>
std::vector<size_t> Tensor<DataType>::GetDimensions() const
{
	return dimensions;
}

template <class DataType>
DataType* Tensor<DataType>::GetStartPtr()
{
	return data_ptr;
}

template <class DataType>
const DataType* Tensor<DataType>::GetStartPtr() const
{
	return data_ptr;
}

template <class DataType>
inline DataType& Tensor<DataType>::operator[] (const size_t ind)
{
	assert( ind<Numel() );
	return data_ptr[ind];
}

template <class DataType>
inline const DataType& Tensor<DataType>::operator[] (const size_t ind) const
{
	assert( ind<Numel() );
	return data_ptr[ind];
}

template <class DataType>
Tensor<DataType>::~Tensor()
{
	if (owns_data)
		delete[] data_ptr;
}

#endif