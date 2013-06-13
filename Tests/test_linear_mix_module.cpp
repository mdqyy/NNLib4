#include <boost/test/unit_test.hpp>
#include <assert.h>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "MseCostModule.h"
#include "GaussianInitializer.h"
#include "ConstantInitializer.h"
#include "WeightDecayRegularizer.h"
#include "LinearMixModule.h"
#include "CompositeModule.h"
#include "FullTensorDataLoader.h"
#include "TrainDataset.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestLinearMixModule)
{
	float input[] = {0, 0, 1, 2, -1, -2,
					 4, 2,  1,  3, 0, 0};
	float gradients[16] = {0};
	float parameters[16] = {0,0,1,4,2,
								3,2,4,
								5,1,3,
								8,1,2, 0,0};
	float parameters_test[8] = {0,0,1,4,2,3,0,0};
	std::vector<size_t> input_dims; input_dims.push_back(4); input_dims.push_back(2);
	std::vector<size_t> per_case_input_dims = input_dims; per_case_input_dims.pop_back();
	std::vector<size_t> output_dims; output_dims.push_back(3); output_dims.push_back(2);
	std::vector<float> importances;importances.push_back(1); importances.push_back(1);

	Tensor<float> input_tensor(input+2, input_dims);
	float expected_output[] = {-14, 5, 3, 39, 24,25};
	
	std::shared_ptr<ConstantInitializer<float>> initializer(new ConstantInitializer<float>(0));
	std::shared_ptr<WeightDecayRegularizer<float>> regularizer(new WeightDecayRegularizer<float>(0.5));
	LinearMixModule<float> a("module1", 4,3,initializer, regularizer);
	a.SetParameters(parameters+2);
	a.train_fprop( std::shared_ptr<Tensor<float> >( new Tensor<float>(input_tensor) ) );
	std::shared_ptr<Tensor<float> > output_tensor = a.GetOutputBuffer();
	BOOST_CHECK(test_equal_arrays(expected_output, output_tensor->GetStartPtr(), 6));
	
	// test regularizer
	double reg_cost = a.GetCost(importances);
	BOOST_CHECK_EQUAL(reg_cost,2*38.5);
	
	// test initializer
	a.InitializeParameters();
	std::vector<float> parameters_tensor;
	a.GetParameters(parameters_tensor);
	BOOST_CHECK( parameters_tensor.size() == 12);
	for (size_t i=0; i<parameters_tensor.size(); i++)
		BOOST_CHECK_EQUAL(parameters_tensor[i] , 0);

	BOOST_CHECK(a.AlocateOutputBuffer());
	BOOST_CHECK(a.AlocateInputGradientsBuffer());
	BOOST_CHECK_EQUAL(a.GetNumParams() , 12);
	BOOST_CHECK(a.GetPerCaseOutputDims(per_case_input_dims).size() == 1 && a.GetPerCaseOutputDims(per_case_input_dims)[0] == 3);

	BOOST_CHECK( TestGetSetParameters<float>(a, a.GetNumParams()) );
}

BOOST_AUTO_TEST_CASE(test_linear_mix_gradient)
{
	std::vector< std::shared_ptr< Tensor<double> > > train_input(15);
	std::vector< std::shared_ptr< Tensor<double> > > train_output(15);
	std::vector<double> train_importance(15);

	std::vector< std::shared_ptr< Tensor<double> > > test_input(8);
	std::vector< std::shared_ptr< Tensor<double> > > test_output(8);
	std::vector<double> test_importance(8);

	std::vector<size_t> case_input_dims;case_input_dims.push_back(9);
	std::vector<size_t> case_output_dims;case_output_dims.push_back(5);
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
	std::shared_ptr< Module<double> > lmm1(new LinearMixModule<double>("module1", 9,8,initializer, regularizer));
	std::shared_ptr< Module<double> > lmm2(new LinearMixModule<double>("module2", 8,5,initializer, regularizer));
	std::vector< std::shared_ptr< Module<double> > > modules; modules.push_back(lmm1); modules.push_back(lmm2);
	std::shared_ptr< CompositeModule<double> > main_module(new CompositeModule<double>("module3", modules));
	
	NN<double> net(main_module);
	net.InitializeParameters();
	BOOST_CHECK(NumericalCheckNNGradients(net, MseCostModule<double>(), train_dataset));
	
	BOOST_CHECK( test_save_load_nn_state(net) );
}