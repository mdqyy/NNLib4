#include <boost/test/unit_test.hpp>
#include <vector>
#include "Tensor.h"
#include "MaxPoolingKernel.h"

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

// incorrect: max pooling can be performed even when there is missing data (on the borders), but current implementation does not allow this
BOOST_AUTO_TEST_CASE(TestMaxPoolingKernel)
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
	Tensor<float> input_tensor(input_data, input_dims);
	Tensor<float> output_tensor(output_data, output_dims);
	Tensor<float> kernel_tensor(0, kernel_dims);
	
	std::vector<size_t> strides;
	strides.push_back(1);strides.push_back(1);strides.push_back(1);
	MaxPoolingKernel<float> kernel(kernel_tensor, strides);
	BOOST_CHECK(kernel.GetOutputTensorDimensions(input_dims) == output_dims);

	kernel.fprop(input_tensor, output_tensor);
	BOOST_CHECK_EQUAL(output_data[0] , 7);
	BOOST_CHECK_EQUAL(output_data[1] , 8);
	BOOST_CHECK_EQUAL(output_data[2] , 8);
	BOOST_CHECK_EQUAL(output_data[3] , 9);
	BOOST_CHECK_EQUAL(output_data[4] , 9);
	BOOST_CHECK_EQUAL(output_data[5] , 9);
	BOOST_CHECK_EQUAL(output_data[6] , 7);
	BOOST_CHECK_EQUAL(output_data[7] , 7);
	BOOST_CHECK_EQUAL(output_data[8] , 8);
	BOOST_CHECK_EQUAL(output_data[9] , 8);
	BOOST_CHECK_EQUAL(output_data[10] , 8);
	BOOST_CHECK_EQUAL(output_data[11] , 8);
	BOOST_CHECK_EQUAL(output_data[12] , 8);
	BOOST_CHECK_EQUAL(output_data[13] , 8);
	BOOST_CHECK_EQUAL(output_data[14] , 8);
	BOOST_CHECK_EQUAL(output_data[15] , 8);
	BOOST_CHECK_EQUAL(output_data[16] , 8);
	BOOST_CHECK_EQUAL(output_data[17] , 8);

	for (int pos = output_tensor.Numel(); pos<100; pos++)
		BOOST_CHECK_EQUAL(output_data[pos] , 0);
}

BOOST_AUTO_TEST_CASE(test_maxpoolingkernel_gradient)
{
	std::vector< std::shared_ptr< Tensor<double> > > train_input(5);
	std::vector< std::shared_ptr< Tensor<double> > > train_output(5);
	std::vector<double> train_importance(5);

	std::vector< std::shared_ptr< Tensor<double> > > test_input(8);
	std::vector< std::shared_ptr< Tensor<double> > > test_output(8);
	std::vector<double> test_importance(8);

	std::vector<size_t> case_input_dims;case_input_dims.push_back(15); case_input_dims.push_back(14); case_input_dims.push_back(4);
	std::vector<size_t> case_output_dims;case_output_dims.push_back(8);
	for (size_t i=0; i<train_input.size(); i++)
	{
		train_input[i] = GetRandomTensorPtr<double>(case_input_dims);
		train_output[i] = GetRandomTensorPtr<double>(case_output_dims);
		train_importance[i]  = i+1.0;
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
	std::vector<size_t> kernel_dims; kernel_dims.push_back(4); kernel_dims.push_back(5); kernel_dims.push_back(1);
	std::vector<size_t> strides;strides.push_back(4);strides.push_back(3);strides.push_back(1);
	std::shared_ptr< Module<double> > kernel_module(new KernelModule<double>("module1", 1,kernel_dims,strides, 
		MaxPoolingKernelFactory<double>(), initializer, regularizer));
	
	size_t num_kernel_outputs = Tensor<double>::Numel(kernel_module->GetPerCaseOutputDims(case_input_dims));
	assert(num_kernel_outputs == 48);
	std::shared_ptr< Module<double> > m1(new LinearModule<double>("module2", case_input_dims,initializer, regularizer));
	std::shared_ptr< Module<double> > m3(new LinearMixModule<double>("module3", num_kernel_outputs,8,initializer, regularizer));
	std::vector< std::shared_ptr< Module<double> > > modules; modules.push_back(m1); modules.push_back(kernel_module); modules.push_back(m3);
	std::shared_ptr< CompositeModule<double> > main_module(new CompositeModule<double>("module4", modules));
	
	NN<double> net(main_module);
	net.InitializeParameters();
	BOOST_CHECK(NumericalCheckNNGradients(net, MseCostModule<double>(), train_dataset));
	
	BOOST_CHECK( test_save_load_nn_state(net) );
}