#include <boost/test/unit_test.hpp>
#include <vector>
#include "test_utilities.h"
#include "Tensor.h"
#include "ConstantInitializer.h"
#include "AbsRegularizer.h"
#include "KernelModule.h"
#include "MaxPoolingKernel.h"
#include "KernelFactory.h"

BOOST_AUTO_TEST_CASE(TestKernelModule_max_pooling)
{
	size_t num_input_kernels = 15;
	size_t num_output_kernels = 1;
	size_t num_samples = 10;
	std::vector<size_t> input_dims; input_dims.push_back(50); input_dims.push_back(50); input_dims.push_back(num_input_kernels);input_dims.push_back(num_samples);
	Tensor<float> input_tensor = GetRandomTensor<float>(input_dims);
	std::vector<size_t> kernel_dims; kernel_dims.push_back(4); kernel_dims.push_back(9); kernel_dims.push_back(1);
	
	std::vector<size_t> strides;strides.push_back(4);strides.push_back(5);strides.push_back(1);
	std::shared_ptr<ConstantInitializer<float>> initializer(new ConstantInitializer<float>(0));
	KernelModule<float> kernel_module("module1", num_output_kernels,kernel_dims,strides, MaxPoolingKernelFactory<float>(),initializer);
	std::vector<size_t> output_dims = kernel_module.GetKernel(0)->GetOutputTensorDimensions(input_dims);
	BOOST_CHECK(output_dims.size() == 4 && output_dims[0] == 12 && output_dims[1] == 9 && output_dims[2] == 15 && output_dims[3] == 10);
	kernel_module.train_fprop( std::shared_ptr<Tensor<float> >( new Tensor<float>(input_tensor)) );
	std::shared_ptr<Tensor<float> > output_tensor = kernel_module.GetOutputBuffer();
	BOOST_CHECK(output_tensor->GetDimensions() == output_dims);

	std::vector<size_t> sample_input_dims = input_dims;
	sample_input_dims.pop_back();
	std::vector<size_t> sample_output_dims = output_dims;
	sample_output_dims.pop_back();
	Tensor<float> sample_output_tensor(nullptr, sample_output_dims);
	Tensor<float> sample_input_tensor(nullptr, sample_input_dims);

	for (size_t sample_ind = 0; sample_ind<num_samples; sample_ind++)
	{
		sample_input_tensor.SetDataPtr(input_tensor.GetStartPtr()+sample_ind*sample_input_tensor.Numel());
		sample_output_tensor.SetDataPtr(output_tensor->GetStartPtr()+sample_ind*sample_output_tensor.Numel());
		BOOST_CHECK(test_filter_response<float>(sample_input_tensor, sample_output_tensor, *kernel_module.GetKernel(0), kernel_dims, strides));
	}
	
	BOOST_CHECK_EQUAL(kernel_module.GetNumParams() , 0);
	
	// test set parameters
	size_t num_params = kernel_module.GetNumParams();
	BOOST_CHECK( TestGetSetParameters<float>(kernel_module, kernel_module.GetNumParams()) );
}

BOOST_AUTO_TEST_CASE(TestKernelModule_convolutional)
{
	size_t num_input_kernels = 8;
	size_t num_output_kernels = 16;
	size_t num_samples = 2;
	std::vector<size_t> input_dims; input_dims.push_back(50); input_dims.push_back(50); input_dims.push_back(num_input_kernels);input_dims.push_back(num_samples);
	Tensor<double> input_tensor = GetRandomTensor<double>(input_dims);
	std::vector<size_t> kernel_dims; kernel_dims.push_back(4); kernel_dims.push_back(9); kernel_dims.push_back(num_input_kernels);
	std::vector<double> importances;
	for (size_t i=0; i<num_samples; i++)
		importances.push_back(1);
	
	std::vector<size_t> strides;strides.push_back(4);strides.push_back(5);strides.push_back(1);
	std::shared_ptr<ConstantInitializer<double>> initializer(new ConstantInitializer<double>(0));
	std::shared_ptr<Regularizer<double>> regularizer(new AbsRegularizer<double>(0.5));
	KernelModule<double> kernel_module("module1", num_output_kernels,kernel_dims,strides, ConvolutionalKernelFactory<double>(), initializer, regularizer);
	size_t num_params = kernel_module.GetNumParams();
	std::vector<size_t> all_kernels_dims = kernel_dims;
	all_kernels_dims.push_back(num_output_kernels);
	Tensor<double> all_kernels_tensor = GetRandomTensor<double>(all_kernels_dims);
	double* params = all_kernels_tensor.GetStartPtr();
	kernel_module.SetParameters(params);
	std::vector<size_t> output_dims = kernel_module.GetKernel(0)->GetOutputTensorDimensions(input_dims);
	output_dims[2] = num_output_kernels;
	kernel_module.train_fprop( std::shared_ptr<Tensor<double> >( new Tensor<double>(input_tensor)) );
	std::shared_ptr<Tensor<double> > output_tensor = kernel_module.GetOutputBuffer();
	BOOST_CHECK(output_tensor->GetDimensions() == output_dims);

	// test regularizer
	double reg_cost = kernel_module.GetCost(importances);
	double expected_reg_cost = 0;
	for (size_t i=0; i<all_kernels_tensor.Numel(); i++)
		expected_reg_cost+=num_samples*std::abs(all_kernels_tensor[i]);
	expected_reg_cost /= 2;
	BOOST_CHECK(abs(reg_cost-expected_reg_cost)<0.000001);
	
	std::vector<size_t> sample_output_dims = output_dims;
	sample_output_dims.pop_back();
	sample_output_dims.pop_back();
	std::vector<size_t> sample_input_dims = input_dims;
	sample_input_dims.pop_back();
	Tensor<double> sample_output_tensor(nullptr, sample_output_dims);
	Tensor<double> sample_input_tensor(nullptr, sample_input_dims);

	size_t num_params_per_kernel = Tensor<double>::Numel(kernel_dims);
	for (size_t sample_ind = 0; sample_ind<num_samples; sample_ind++)
	{
		sample_input_tensor.SetDataPtr(input_tensor.GetStartPtr()+sample_ind*sample_input_tensor.Numel());
		for (size_t kernel_ind = 0; kernel_ind<num_output_kernels; kernel_ind++)
		{
			sample_output_tensor.SetDataPtr(output_tensor->GetStartPtr()+(num_output_kernels*sample_ind + kernel_ind)*sample_output_tensor.Numel());
			BOOST_CHECK(test_filter_response<double>(sample_input_tensor, sample_output_tensor, 
				ConvolutionalKernel<double>(Tensor<double>(params + num_params_per_kernel*kernel_ind, kernel_dims), strides), kernel_dims, strides));
		}
	}

	// test initializer
	kernel_module.InitializeParameters();
	std::vector<double> parameters;
	kernel_module.GetParameters(parameters);
	for (size_t i=0; i<all_kernels_tensor.Numel(); i++)
		BOOST_CHECK_EQUAL(parameters[i] , 0);
	
	BOOST_CHECK_EQUAL(kernel_module.GetNumParams() , num_output_kernels*num_input_kernels*kernel_dims[0]*kernel_dims[1]);
	
	TestGetSetParameters<double>(kernel_module, kernel_module.GetNumParams());
}