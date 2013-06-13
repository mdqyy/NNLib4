#include <boost/test/unit_test.hpp>
#include <assert.h>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "TanhModule.h"
#include "BiasModule.h"
#include "LinearModule.h"
#include "MseCostModule.h"
#include "GaussianInitializer.h"
#include "ConstantInitializer.h"
#include "WeightDecayRegularizer.h"
#include "AbsRegularizer.h"
#include "LinearMixModule.h"
#include "CompositeModule.h"
#include "FullTensorDataLoader.h"
#include "TrainDataset.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestLinearModule)
{
	float input[] = {0, 0, 1, 2, -1, -2,
					 4, 2,  1,  3, 0, 0};
	float parameters[8] = {0,0,1,4,2,3,0,0};
	float parameters_test[8] = {0,0,1,4,2,3,0,0};
	std::vector<size_t> input_dims; input_dims.push_back(4); input_dims.push_back(2);
	std::vector<size_t> per_case_input_dims; per_case_input_dims.push_back(4); per_case_input_dims.push_back(1);
	std::vector<size_t> output_dims = input_dims;

	Tensor<float> input_tensor(input+2, input_dims);
	float expected_output[] = {1, 8, -2, -6, 4,8,2,9};
	
	std::vector<size_t> input_case_dims = input_tensor.GetDimensions();
	input_case_dims.pop_back();
	std::shared_ptr<ConstantInitializer<float> > initializer(new ConstantInitializer<float>(0));
	std::shared_ptr<AbsRegularizer<float>> regularizer(new AbsRegularizer<float>(0.5));
	LinearModule<float> a("module1", input_case_dims, initializer, regularizer);
	a.SetParameters(parameters+2);
	a.train_fprop( std::shared_ptr<Tensor<float> >( new Tensor<float>(input_tensor) ) );
	std::shared_ptr<Tensor<float> > output_tensor = a.GetOutputBuffer();
	BOOST_CHECK(test_equal_arrays(expected_output, output_tensor->GetStartPtr(), 8));
	std::vector<float> importances;importances.push_back(1); importances.push_back(1);
	
	// test regularizer
	double reg_cost = a.GetCost(importances);
	BOOST_CHECK(reg_cost==2*5);

	float gradient_expected_output[] = {0, 0, 18, 33, 5, 40, 0, 0};
	auto input_gradients = a.bprop(output_tensor, importances);
	std::vector<float> gradients;
	a.GetGradients(gradients);
	BOOST_CHECK(test_equal_arrays(gradient_expected_output+2, gradients.data(), 4));
	std::vector<float> module_parameters;
	a.GetParameters(module_parameters);
	BOOST_CHECK(test_equal_arrays(module_parameters.data(), parameters_test+2, 4));
	BOOST_CHECK(test_equal_arrays(parameters, parameters_test, 8));
	BOOST_CHECK(a.AlocateOutputBuffer());
	BOOST_CHECK( a.AlocateInputGradientsBuffer() );
	BOOST_CHECK(a.GetNumParams() == 4);
	BOOST_CHECK(a.GetPerCaseOutputDims(per_case_input_dims).size() == 2 && a.GetPerCaseOutputDims(per_case_input_dims)[0] == 4 && 
		a.GetPerCaseOutputDims(per_case_input_dims)[1] == 1);

	// test initializer
	a.InitializeParameters();
	module_parameters.clear();
	a.GetParameters(module_parameters);
	for (size_t i=0; i<a.GetNumParams(); i++)
		BOOST_CHECK(module_parameters[i] == 0);
	
	BOOST_CHECK( TestGetSetParameters<float>(a, a.GetNumParams()) );
}

BOOST_AUTO_TEST_CASE(TestLinearGradient)
{
	std::vector< std::shared_ptr< Tensor<double> > > train_input(15);
	std::vector< std::shared_ptr< Tensor<double> > > train_output(15);
	std::vector<double> train_importance(15);

	std::vector<size_t> case_input_dims;case_input_dims.push_back(9);
	std::vector<size_t> case_output_dims;case_output_dims.push_back(9);
	for (size_t i=0; i<train_input.size(); i++)
	{
		train_input[i] = GetRandomTensorPtr<double>(case_input_dims);
		train_output[i] = GetRandomTensorPtr<double>(case_output_dims);
		train_importance[i]  = i+1.0;
	}
	
	std::shared_ptr< ITensorDataLoader<double> > input_data_loader(new FullTensorDataLoader<double,double>(train_input));
	std::shared_ptr< ITensorDataLoader<double> > output_data_loader(new FullTensorDataLoader<double,double>(train_output));
	TrainDataset<double> train_dataset(input_data_loader, output_data_loader, train_importance);

	std::shared_ptr<ParametersInitializer<double>> initializer(new GaussianInitializer<double>());
	std::shared_ptr<Regularizer<double>> regularizer(new WeightDecayRegularizer<double>(0.5));
	std::shared_ptr< Module<double> > mlinear1(new LinearModule<double>("module1", case_input_dims,initializer, regularizer));
	std::shared_ptr< Module<double> > m_tanh1(new TanhModule<double>("module2"));
	std::shared_ptr< Module<double> > mlinear2(new LinearModule<double>("module3", case_input_dims,initializer, regularizer));
	std::shared_ptr< Module<double> > m_tanh2(new TanhModule<double>("module4"));
	std::vector< std::shared_ptr< Module<double> > > modules; 
	modules.push_back(mlinear1); 
	modules.push_back(m_tanh1); 
	modules.push_back(mlinear2); 
	modules.push_back(m_tanh2);
	std::shared_ptr< CompositeModule<double> > main_module(new CompositeModule<double>("module5", modules));
	
	NN<double> net(main_module);
	net.InitializeParameters();
	BOOST_CHECK(NumericalCheckNNGradients(net, MseCostModule<double>(), train_dataset));
	
	BOOST_CHECK( test_save_load_nn_state(net) );
}