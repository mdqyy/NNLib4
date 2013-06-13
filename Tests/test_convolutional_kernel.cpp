#include <boost/test/unit_test.hpp>
#include <vector>
#include "Tensor.h"
#include "ConvolutionalKernel.h"
#include "KernelFactory.h"

#include <memory>
#include "KernelModule.h"
#include "LinearModule.h"
#include "MseCostModule.h"
#include "GaussianInitializer.h"
#include "WeightDecayRegularizer.h"
#include "LinearMixModule.h"
#include "CompositeModule.h"
#include "FullTensorDataLoader.h"
#include "TrainDataset.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(test_kernel_convolution1)
{
	std::vector<size_t> input_dims; input_dims.push_back(5); input_dims.push_back(4); input_dims.push_back(3);
	float input_data[] = { 1, 2, 3, 4, 2,
		             5, 6, 7, 8, 3,
					 2, 4, 1, 3, 1,
					 1, 9, 2, 1, 9,
					 
					 2, 3, 4, 1, 4,
					 1, 5, 7, 2, 8,
					 2, 4, 1, 3, 0,
					 1, 8, 2, 1, 2,
					 
					 2,5,4,1,7,
					 1,4,8,5,5, 
					 0,2,0,1,2,
					 5,3,5,2,1};
	
	std::vector<size_t> output_dims; output_dims.push_back(3); output_dims.push_back(2);output_dims.push_back(1);
	float output_data[100] = {0};
	
	std::vector<size_t> kernel_dims; kernel_dims.push_back(3); kernel_dims.push_back(3); kernel_dims.push_back(3);
	float kernel_data[] = { 1,0,4,
							2,4,6,
							3,1,1,

							2,1,5,
							8,2,2,
							3,9,1,

							8,1,5,
							4,9,9,
							2,5,2};

	Tensor<float> input_tensor(input_data, input_dims);
	Tensor<float> output_tensor(output_data, output_dims);
	Tensor<float> kernel_tensor(kernel_data, kernel_dims);
	
	std::vector<size_t> strides;
	strides.push_back(1);strides.push_back(1);strides.push_back(1);
	ConvolutionalKernel<float> kernel(kernel_tensor, strides);
	kernel.fprop(input_tensor, output_tensor);
	BOOST_CHECK_EQUAL(output_data[0],365);
	BOOST_CHECK_EQUAL(output_data[1],407);
	BOOST_CHECK_EQUAL(output_data[2] , 416);
	BOOST_CHECK_EQUAL(output_data[3] , 323);
	BOOST_CHECK_EQUAL(output_data[4] , 325);
	BOOST_CHECK_EQUAL(output_data[5] , 285);
	BOOST_CHECK(kernel.GetOutputTensorDimensions(input_dims) == output_dims);

	size_t output_pos[3] = {0};
	output_pos[0] = 0;output_pos[1] = 0;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 365);
	output_pos[0] = 1;output_pos[1] = 0;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 407);
	output_pos[0] = 2;output_pos[1] = 0;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 416);
	output_pos[0] = 0;output_pos[1] = 1;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 323);
	output_pos[0] = 1;output_pos[1] = 1;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 325);
	output_pos[0] = 2;output_pos[1] = 1;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 285);

	for (int pos = output_tensor.Numel(); pos<100; pos++)
		BOOST_CHECK_EQUAL(output_data[pos], 0);
}

BOOST_AUTO_TEST_CASE(test_kernel_convolution2)
{
	std::vector<size_t> input_dims; input_dims.push_back(5); input_dims.push_back(4); input_dims.push_back(3);
	float input_data[] = { 1, 2, 3, 4, 2,
		             5, 6, 7, 8, 3,
					 2, 4, 1, 3, 1,
					 1, 9, 2, 1, 9,
					 
					 2, 3, 4, 1, 4,
					 1, 5, 7, 2, 8,
					 2, 4, 1, 3, 0,
					 1, 8, 2, 1, 2,
					 
					 2,5,4,1,7,
					 1,4,8,5,5, 
					 0,2,0,1,2,
					 5,3,5,2,1};
	
	std::vector<size_t> output_dims; output_dims.push_back(3); output_dims.push_back(4);output_dims.push_back(1);
	float output_data[100] = {0};
	
	std::vector<size_t> kernel_dims; kernel_dims.push_back(3); kernel_dims.push_back(1); kernel_dims.push_back(3);
	float kernel_data[] = { 1,0,4,

							2,1,5,

							8,1,5
						  };

	Tensor<float> input_tensor(input_data, input_dims);
	Tensor<float> output_tensor(output_data, output_dims);
	Tensor<float> kernel_tensor(kernel_data, kernel_dims);
	
	std::vector<size_t> strides;
	strides.push_back(1);strides.push_back(1);strides.push_back(1);
	ConvolutionalKernel<float> kernel(kernel_tensor, strides);
	kernel.fprop(input_tensor, output_tensor);
	BOOST_CHECK_EQUAL(output_data[0] , 81);
	BOOST_CHECK_EQUAL(output_data[1] , 82);
	BOOST_CHECK_EQUAL(output_data[2] , 108);
	BOOST_CHECK_EQUAL(output_data[3] , 127);
	BOOST_CHECK_EQUAL(output_data[4] , 130);
	BOOST_CHECK_EQUAL(output_data[5] , 169);
	BOOST_CHECK_EQUAL(output_data[6] , 21);
	BOOST_CHECK_EQUAL(output_data[7] , 61);
	BOOST_CHECK_EQUAL(output_data[8] , 21);
	BOOST_CHECK_EQUAL(output_data[9] , 97);
	BOOST_CHECK_EQUAL(output_data[10] , 75);
	BOOST_CHECK_EQUAL(output_data[11] , 100);
	BOOST_CHECK(kernel.GetOutputTensorDimensions(input_dims) == output_dims);
	
	size_t output_pos[3] = {0};
	output_pos[0] = 0;output_pos[1] = 0;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 81);
	output_pos[0] = 1;output_pos[1] = 0;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 82);
	output_pos[0] = 2;output_pos[1] = 0;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 108);
	output_pos[0] = 0;output_pos[1] = 1;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 127);
	output_pos[0] = 1;output_pos[1] = 1;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 130);
	output_pos[0] = 2;output_pos[1] = 1;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 169);
	output_pos[0] = 0;output_pos[1] = 2;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 21);
	output_pos[0] = 1;output_pos[1] = 2;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 61);
	output_pos[0] = 2;output_pos[1] = 2;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 21);
	output_pos[0] = 0;output_pos[1] = 3;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 97);
	output_pos[0] = 1;output_pos[1] = 3;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 75);
	output_pos[0] = 2;output_pos[1] = 3;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 100);

	for (int pos = output_tensor.Numel(); pos<100; pos++)
		BOOST_CHECK_EQUAL(output_data[pos] , 0);
}

BOOST_AUTO_TEST_CASE(test_kernel_convolution3)
{
	std::vector<size_t> input_dims; input_dims.push_back(5); input_dims.push_back(4); input_dims.push_back(3);
	float input_data[] = { 1, 2, 3, 4, 2,
		             5, 6, 7, 8, 3,
					 2, 4, 1, 3, 1,
					 1, 9, 2, 1, 9,
					 
					 2, 3, 4, 1, 4,
					 1, 5, 7, 2, 8,
					 2, 4, 1, 3, 0,
					 1, 8, 2, 1, 2,
					 
					 2,5,4,1,7,
					 1,4,8,5,5, 
					 0,2,0,1,2,
					 5,3,5,2,1};
	
	std::vector<size_t> output_dims; output_dims.push_back(5); output_dims.push_back(2);output_dims.push_back(1);
	float output_data[100] = {0};
	
	std::vector<size_t> kernel_dims; kernel_dims.push_back(1); kernel_dims.push_back(3); kernel_dims.push_back(3);
	float kernel_data[] = { 1,0,4,

							2,1,5,

							8,1,5
						  };

	Tensor<float> input_tensor(input_data, input_dims);
	Tensor<float> output_tensor(output_data, output_dims);
	Tensor<float> kernel_tensor(kernel_data, kernel_dims);
	
	std::vector<size_t> strides;
	strides.push_back(1);strides.push_back(1);strides.push_back(1);
	ConvolutionalKernel<float> kernel(kernel_tensor, strides);
	kernel.fprop(input_tensor, output_tensor);
	BOOST_CHECK_EQUAL(output_data[0] , 41);
	BOOST_CHECK_EQUAL(output_data[6] , 145);
	BOOST_CHECK(kernel.GetOutputTensorDimensions(input_dims) == output_dims);
	
	size_t output_pos[3] = {0};
	output_pos[0] = 0;output_pos[1] = 0;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 41);
	output_pos[0] = 1;output_pos[1] = 1;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 145);
	
	for (int pos = output_tensor.Numel(); pos<100; pos++)
		BOOST_CHECK_EQUAL(output_data[pos] , 0);
}

BOOST_AUTO_TEST_CASE(test_kernel_convolution4)
{
	std::vector<size_t> input_dims; input_dims.push_back(5); input_dims.push_back(4); input_dims.push_back(3);
	float input_data[] = { 1, 2, 3, 4, 2,
		             5, 6, 7, 8, 3,
					 2, 4, 1, 3, 1,
					 1, 9, 2, 1, 9,
					 
					 2, 3, 4, 1, 4,
					 1, 5, 7, 2, 8,
					 2, 4, 1, 3, 0,
					 1, 8, 2, 1, 2,
					 
					 2,5,4,1,7,
					 1,4,8,5,5, 
					 0,2,0,1,2,
					 5,3,5,2,1};
	
	std::vector<size_t> output_dims; output_dims.push_back(3); output_dims.push_back(2); output_dims.push_back(3);
	float output_data[100] = {0};
	
	std::vector<size_t> kernel_dims; kernel_dims.push_back(3); kernel_dims.push_back(3); kernel_dims.push_back(1);
	float kernel_data[] = { 1,0,4,
							2,1,5,
							8,1,5
						  };

	Tensor<float> input_tensor(input_data, input_dims);
	Tensor<float> output_tensor(output_data, output_dims);
	Tensor<float> kernel_tensor(kernel_data, kernel_dims);
	
	std::vector<size_t> strides;
	strides.push_back(1);strides.push_back(1);strides.push_back(1);
	ConvolutionalKernel<float> kernel(kernel_tensor, strides);
	kernel.fprop(input_tensor, output_tensor);
	BOOST_CHECK_EQUAL(output_data[0] , 89);
	BOOST_CHECK_EQUAL(output_data[17] , 86);
	BOOST_CHECK(kernel.GetOutputTensorDimensions(input_dims) == output_dims);

	size_t output_pos[3];
	output_pos[0] = 0;output_pos[1] = 0;output_pos[2] = 0;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 89);
	output_pos[0] = 2;output_pos[1] = 1;output_pos[2] = 2;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 86);
	for (int pos = output_tensor.Numel(); pos<100; pos++)
		BOOST_CHECK_EQUAL(output_data[pos] , 0);
}

BOOST_AUTO_TEST_CASE(test_kernel_convolution_with_stride1)
{
	std::vector<size_t> input_dims; input_dims.push_back(5); input_dims.push_back(4); input_dims.push_back(3);
	float input_data[] = { 1, 2, 3, 4, 2,
		             5, 6, 7, 8, 3,
					 2, 4, 1, 3, 1,
					 1, 9, 2, 1, 9,
					 
					 2, 3, 4, 1, 4,
					 1, 5, 7, 2, 8,
					 2, 4, 1, 3, 0,
					 1, 8, 2, 1, 2,
					 
					 2,5,4,1,7,
					 1,4,8,5,5, 
					 0,2,0,1,2,
					 5,3,5,2,1};
	
	std::vector<size_t> output_dims; output_dims.push_back(2); output_dims.push_back(2);output_dims.push_back(1);
	float output_data[100] = {0};
	
	std::vector<size_t> kernel_dims; kernel_dims.push_back(3); kernel_dims.push_back(3); kernel_dims.push_back(3);
	float kernel_data[] = { 1,0,4,
							2,4,6,
							3,1,1,

							2,1,5,
							8,2,2,
							3,9,1,

							8,1,5,
							4,9,9,
							2,5,2};

	Tensor<float> input_tensor(input_data, input_dims);
	Tensor<float> output_tensor(output_data, output_dims);
	Tensor<float> kernel_tensor(kernel_data, kernel_dims);
	
	std::vector<size_t> strides;
	strides.push_back(2);strides.push_back(1);strides.push_back(1);
	ConvolutionalKernel<float> kernel(kernel_tensor, strides);
	kernel.fprop(input_tensor, output_tensor);
	BOOST_CHECK_EQUAL(output_data[0] , 365);
	BOOST_CHECK_EQUAL(output_data[1] , 416);
	BOOST_CHECK_EQUAL(output_data[2] , 323);
	BOOST_CHECK_EQUAL(output_data[3] , 285);
	BOOST_CHECK(kernel.GetOutputTensorDimensions(input_dims) == output_dims);
	
	size_t output_pos[3] = {0};
	output_pos[0] = 0;output_pos[1] = 0;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 365);
	output_pos[0] = 1;output_pos[1] = 0;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 416);
	output_pos[0] = 0;output_pos[1] = 1;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 323);
	output_pos[0] = 1;output_pos[1] = 1;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 285);

	for (int pos = output_tensor.Numel(); pos<100; pos++)
		BOOST_CHECK_EQUAL(output_data[pos] , 0);
}

BOOST_AUTO_TEST_CASE(test_kernel_convolution_with_stride2)
{
	std::vector<size_t> input_dims; input_dims.push_back(5); input_dims.push_back(4); input_dims.push_back(3);
	float input_data[] = { 1, 2, 3, 4, 2,
		             5, 6, 7, 8, 3,
					 2, 4, 1, 3, 1,
					 1, 9, 2, 1, 9,
					 
					 2, 3, 4, 1, 4,
					 1, 5, 7, 2, 8,
					 2, 4, 1, 3, 0,
					 1, 8, 2, 1, 2,
					 
					 2,5,4,1,7,
					 1,4,8,5,5, 
					 0,2,0,1,2,
					 5,3,5,2,1};
	
	std::vector<size_t> output_dims; output_dims.push_back(3); output_dims.push_back(2);output_dims.push_back(1);
	float output_data[100] = {0};
	
	std::vector<size_t> kernel_dims; kernel_dims.push_back(3); kernel_dims.push_back(1); kernel_dims.push_back(3);
	float kernel_data[] = { 1,0,4,

							2,1,5,

							8,1,5
						  };

	Tensor<float> input_tensor(input_data, input_dims);
	Tensor<float> output_tensor(output_data, output_dims);
	Tensor<float> kernel_tensor(kernel_data, kernel_dims);
	
	std::vector<size_t> strides;
	strides.push_back(1);strides.push_back(2);strides.push_back(1);
	ConvolutionalKernel<float> kernel(kernel_tensor, strides);
	kernel.fprop(input_tensor, output_tensor);
	BOOST_CHECK_EQUAL(output_data[0] , 81);
	BOOST_CHECK_EQUAL(output_data[1] , 82);
	BOOST_CHECK_EQUAL(output_data[2] , 108);
	BOOST_CHECK_EQUAL(output_data[3] , 21);
	BOOST_CHECK_EQUAL(output_data[4] , 61);
	BOOST_CHECK_EQUAL(output_data[5] , 21);
	BOOST_CHECK(kernel.GetOutputTensorDimensions(input_dims) == output_dims);
	
	size_t output_pos[3] = {0};
	output_pos[0] = 0;output_pos[1] = 0;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 81);
	output_pos[0] = 1;output_pos[1] = 0;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 82);
	output_pos[0] = 2;output_pos[1] = 0;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 108);
	output_pos[0] = 0;output_pos[1] = 1;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 21);
	output_pos[0] = 1;output_pos[1] = 1;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 61);
	output_pos[0] = 2;output_pos[1] = 1;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 21);

	for (int pos = output_tensor.Numel(); pos<100; pos++)
		BOOST_CHECK_EQUAL(output_data[pos] , 0);
}

BOOST_AUTO_TEST_CASE(test_kernel_convolution_with_stride3)
{
	std::vector<size_t> input_dims; input_dims.push_back(5); input_dims.push_back(4); input_dims.push_back(3);
	float input_data[] = { 1, 2, 3, 4, 2,
		             5, 6, 7, 8, 3,
					 2, 4, 1, 3, 1,
					 1, 9, 2, 1, 9,
					 
					 2, 3, 4, 1, 4,
					 1, 5, 7, 2, 8,
					 2, 4, 1, 3, 0,
					 1, 8, 2, 1, 2,
					 
					 2,5,4,1,7,
					 1,4,8,5,5, 
					 0,2,0,1,2,
					 5,3,5,2,1};
	
	std::vector<size_t> output_dims; output_dims.push_back(2); output_dims.push_back(2);output_dims.push_back(1);
	float output_data[100] = {0};
	
	std::vector<size_t> kernel_dims; kernel_dims.push_back(1); kernel_dims.push_back(3); kernel_dims.push_back(3);
	float kernel_data[] = { 1,0,4,

							2,1,5,

							8,1,5
						  };

	Tensor<float> input_tensor(input_data, input_dims);
	Tensor<float> output_tensor(output_data, output_dims);
	Tensor<float> kernel_tensor(kernel_data, kernel_dims);
	
	std::vector<size_t> strides;
	strides.push_back(3);strides.push_back(1);strides.push_back(1);
	ConvolutionalKernel<float> kernel(kernel_tensor, strides);
	kernel.fprop(input_tensor, output_tensor);
	BOOST_CHECK_EQUAL(output_data[0] , 41);
	BOOST_CHECK_EQUAL(output_data[3] , 75);
	BOOST_CHECK(kernel.GetOutputTensorDimensions(input_dims) == output_dims);
	
	size_t output_pos[3] = {0};
	output_pos[0] = 0;output_pos[1] = 0;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 41);
	output_pos[0] = 1;output_pos[1] = 1;
	BOOST_CHECK_EQUAL(*output_tensor.GetPtr(output_pos) , 75);
	
	for (int pos = output_tensor.Numel(); pos<100; pos++)
		BOOST_CHECK_EQUAL(output_data[pos] , 0);
}

BOOST_AUTO_TEST_CASE(test_kernel_deconvolution)
{
	std::vector<size_t> input_dims; input_dims.push_back(5); input_dims.push_back(4); input_dims.push_back(3);
	float input_data[] = { 1, 2, 3, 4, 2,
		             5, 6, 7, 8, 3,
					 2, 4, 1, 3, 1,
					 1, 9, 2, 1, 9,
					 
					 2, 3, 4, 1, 4,
					 1, 5, 7, 2, 8,
					 2, 4, 1, 3, 0,
					 1, 8, 2, 1, 2,
					 
					 2,5,4,1,7,
					 1,4,8,5,5, 
					 0,2,0,1,2,
					 5,3,5,2,1};
	
	std::vector<size_t> output_dims; output_dims.push_back(3); output_dims.push_back(2);output_dims.push_back(1);
	float output_data[100] = {1,5,2,
							  4,8,5};
	
	std::vector<size_t> kernel_dims; kernel_dims.push_back(3); kernel_dims.push_back(3); kernel_dims.push_back(3);
	float kernel_data[] = { 1,0,4,
							2,4,6,
							3,1,1,

							2,1,5,
							8,2,2,
							3,9,1,

							8,1,5,
							4,9,9,
							2,5,2};

	Tensor<float> input_tensor(input_data, input_dims);
	Tensor<float> input_gradients_tensor(input_data, input_dims);
	Tensor<float> output_tensor(output_data, output_dims);
	Tensor<float> kernel_tensor(kernel_data, kernel_dims);
	
	std::vector<size_t> strides;
	strides.push_back(1);strides.push_back(1);strides.push_back(1);
	ConvolutionalKernel<float> kernel(kernel_tensor, strides);
	BOOST_CHECK(kernel.GetOutputTensorDimensions(input_dims) == output_dims);
	input_tensor.SetZeros();
	kernel.bprop(input_tensor, output_tensor, input_gradients_tensor, output_tensor);
	BOOST_CHECK_EQUAL(input_gradients_tensor[0] , 1);
	BOOST_CHECK_EQUAL(input_gradients_tensor[1] , 5);
	BOOST_CHECK_EQUAL(input_gradients_tensor[2] , 6);
	BOOST_CHECK_EQUAL(input_gradients_tensor[3] , 20);
	BOOST_CHECK_EQUAL(input_gradients_tensor[4] , 8);
	BOOST_CHECK_EQUAL(input_gradients_tensor[5] , 6);
	BOOST_CHECK_EQUAL(input_gradients_tensor[6] , 22);
	BOOST_CHECK_EQUAL(input_gradients_tensor[7] , 51);
	BOOST_CHECK_EQUAL(input_gradients_tensor[8] , 70);
	BOOST_CHECK_EQUAL(input_gradients_tensor[9] , 32);
	BOOST_CHECK_EQUAL(input_gradients_tensor[10] , 11);
	BOOST_CHECK_EQUAL(input_gradients_tensor[11] , 48);
	BOOST_CHECK_EQUAL(input_gradients_tensor[12] , 78);
	BOOST_CHECK_EQUAL(input_gradients_tensor[13] , 75);
	BOOST_CHECK_EQUAL(input_gradients_tensor[14] , 32);
	BOOST_CHECK_EQUAL(input_gradients_tensor[15] , 12);
	BOOST_CHECK_EQUAL(input_gradients_tensor[16] , 28);
	BOOST_CHECK_EQUAL(input_gradients_tensor[17] , 27);
	BOOST_CHECK_EQUAL(input_gradients_tensor[18] , 13);
	BOOST_CHECK_EQUAL(input_gradients_tensor[19] , 5);
	BOOST_CHECK_EQUAL(input_gradients_tensor[20] , 2);
	BOOST_CHECK_EQUAL(input_gradients_tensor[21] , 11);
	BOOST_CHECK_EQUAL(input_gradients_tensor[22] , 14);
	BOOST_CHECK_EQUAL(input_gradients_tensor[23] , 27);
	BOOST_CHECK_EQUAL(input_gradients_tensor[24] , 10);
	BOOST_CHECK_EQUAL(input_gradients_tensor[25] , 16);
	BOOST_CHECK_EQUAL(input_gradients_tensor[26] , 62);
	BOOST_CHECK_EQUAL(input_gradients_tensor[27] , 66);
	BOOST_CHECK_EQUAL(input_gradients_tensor[28] , 59);
	BOOST_CHECK_EQUAL(input_gradients_tensor[29] , 29);
	BOOST_CHECK_EQUAL(input_gradients_tensor[30] , 35);
	BOOST_CHECK_EQUAL(input_gradients_tensor[31] , 96);
	BOOST_CHECK_EQUAL(input_gradients_tensor[32] , 116);
	BOOST_CHECK_EQUAL(input_gradients_tensor[33] , 49);
	BOOST_CHECK_EQUAL(input_gradients_tensor[34] , 12);
	BOOST_CHECK_EQUAL(input_gradients_tensor[35] , 12);
	BOOST_CHECK_EQUAL(input_gradients_tensor[36] , 60);
	BOOST_CHECK_EQUAL(input_gradients_tensor[37] , 91);
	BOOST_CHECK_EQUAL(input_gradients_tensor[38] , 53);
	BOOST_CHECK_EQUAL(input_gradients_tensor[39] , 5);
	BOOST_CHECK_EQUAL(input_gradients_tensor[40] , 8);
	BOOST_CHECK_EQUAL(input_gradients_tensor[41] , 41);
	BOOST_CHECK_EQUAL(input_gradients_tensor[42] , 26);
	BOOST_CHECK_EQUAL(input_gradients_tensor[43] , 27);
	BOOST_CHECK_EQUAL(input_gradients_tensor[44] , 10);
	BOOST_CHECK_EQUAL(input_gradients_tensor[45] , 36);
	BOOST_CHECK_EQUAL(input_gradients_tensor[46] , 97);
	BOOST_CHECK_EQUAL(input_gradients_tensor[47] , 130);
	BOOST_CHECK_EQUAL(input_gradients_tensor[48] , 108);
	BOOST_CHECK_EQUAL(input_gradients_tensor[49] , 43);
	BOOST_CHECK_EQUAL(input_gradients_tensor[50] , 18);
	BOOST_CHECK_EQUAL(input_gradients_tensor[51] , 83);
	BOOST_CHECK_EQUAL(input_gradients_tensor[52] , 159);
	BOOST_CHECK_EQUAL(input_gradients_tensor[53] , 137);
	BOOST_CHECK_EQUAL(input_gradients_tensor[54] , 49);
	BOOST_CHECK_EQUAL(input_gradients_tensor[55] , 8);
	BOOST_CHECK_EQUAL(input_gradients_tensor[56] , 36);
	BOOST_CHECK_EQUAL(input_gradients_tensor[57] , 58);
	BOOST_CHECK_EQUAL(input_gradients_tensor[58] , 41);
	BOOST_CHECK_EQUAL(input_gradients_tensor[59] , 10);
}

BOOST_AUTO_TEST_CASE(test_kernel_get_output_dimensions)
{
	std::vector<size_t> input_dims; input_dims.push_back(9); input_dims.push_back(8); input_dims.push_back(4);
	float input_data[288] = {0};
	std::vector<size_t> kernel_dims; kernel_dims.push_back(5); kernel_dims.push_back(2); kernel_dims.push_back(3);
	float kernel_data[30] = {0};
	
	Tensor<float> input_tensor(input_data, input_dims);
	Tensor<float> kernel_tensor(kernel_data, kernel_dims);
	
	std::vector<size_t> strides;
	strides.push_back(2);strides.push_back(3);strides.push_back(4);
	ConvolutionalKernel<float> kernel(kernel_tensor, strides);
	std::vector<size_t>  output_tensor = kernel.GetOutputTensorDimensions(input_tensor.GetDimensions());

	BOOST_CHECK_EQUAL(output_tensor.size() , 3);
	BOOST_CHECK_EQUAL(output_tensor[0] , 3);
	BOOST_CHECK_EQUAL(output_tensor[1] , 3);
	BOOST_CHECK_EQUAL(output_tensor[2] , 1);
}

BOOST_AUTO_TEST_CASE(test_convkernel_gradient)
{
	std::vector< std::shared_ptr< Tensor<double> > > train_input(9);
	std::vector< std::shared_ptr< Tensor<double> > > train_output(9);
	std::vector<double> train_importance(9);

	std::vector< std::shared_ptr< Tensor<double> > > test_input(8);
	std::vector< std::shared_ptr< Tensor<double> > > test_output(8);
	std::vector<double> test_importance(8);

	std::vector<size_t> case_input_dims;case_input_dims.push_back(15); case_input_dims.push_back(14); case_input_dims.push_back(4);
	std::vector<size_t> case_output_dims;case_output_dims.push_back(8);
	for (size_t i=0; i<train_input.size(); i++)
	{
		train_input[i] = GetRandomTensorPtr<double>(case_input_dims);
		train_output[i] = GetRandomTensorPtr<double>(case_output_dims);
		train_importance[i]  = (i>7?1:0);
	}
	
	for (size_t i=0; i<test_input.size(); i++)
	{
		test_input[i] = GetRandomTensorPtr<double>(case_input_dims);
		test_output[i] = GetRandomTensorPtr<double>(case_output_dims);
		test_importance[i]=i+1.0;
	}
	
	std::shared_ptr< ITensorDataLoader<double> > input_data_loader(new FullTensorDataLoader<double,double>(train_input));
	std::shared_ptr< ITensorDataLoader<double> > output_data_loader(new FullTensorDataLoader<double,double>(train_output));
	TrainDataset<double> train_dataset(input_data_loader, output_data_loader, train_importance);

	std::shared_ptr<ParametersInitializer<double>> initializer(new GaussianInitializer<double>());
	std::shared_ptr<Regularizer<double>> regularizer(new WeightDecayRegularizer<double>(0.5));
	
	size_t num_input_kernels = 4;
	size_t num_output_kernels = 5;
	size_t num_samples = train_input.size();
	std::vector<size_t> kernel_dims; kernel_dims.push_back(4); kernel_dims.push_back(5); kernel_dims.push_back(num_input_kernels);
	std::vector<size_t> strides;strides.push_back(4);strides.push_back(3);strides.push_back(1);
	std::shared_ptr< Module<double> > kernel_module(new KernelModule<double>("module1", num_output_kernels,kernel_dims,strides, 
		ConvolutionalKernelFactory<double>(), initializer, regularizer));
	
	size_t num_kernel_outputs = Tensor<double>::Numel(kernel_module->GetPerCaseOutputDims(case_input_dims));
	BOOST_CHECK(num_kernel_outputs == 60);
	std::shared_ptr< Module<double> > m1(new LinearModule<double>("module2", case_input_dims,initializer, regularizer));
	std::shared_ptr< Module<double> > m3(new LinearMixModule<double>("module3", num_kernel_outputs,8,initializer, regularizer));
	std::vector< std::shared_ptr< Module<double> > > modules; modules.push_back(m1); modules.push_back(kernel_module); modules.push_back(m3);
	std::shared_ptr< CompositeModule<double> > main_module(new CompositeModule<double>("module4", modules));
	
	NN<double> net(main_module);
	net.InitializeParameters();
	BOOST_CHECK(NumericalCheckNNGradients(net, MseCostModule<double>(), train_dataset));
	
	BOOST_CHECK( test_save_load_nn_state(net) );
}
