#include <boost/test/unit_test.hpp>
#include <vector>
#include "Tensor.h"
#include "UnsupervisedGroupEntropyCostModule.h"
#include "test_utilities.h"
#include <memory>
#include "SoftmaxModule.h"
#include "GaussianInitializer.h"
#include "WeightDecayRegularizer.h"
#include "LinearMixModule.h"
#include "CompositeModule.h"
#include "FullTensorDataLoader.h"
#include "TrainDataset.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestUnsupervisedGroupEntropyCostModule)
{
	std::vector<size_t> output_dims;output_dims.push_back(4); output_dims.push_back(5);
	float output[] = { 0.1f,  0.15f, 0.5f,  0.21f,
					   0.05f, 0.18f, 0.04f, 0.09f,
					   0.23f, 0.17f, 0.16f, 0.19f,
					   0.35f, 0.45f, 0.21f, 0.4f,
					   0.27f, 0.05f, 0.09f, 0.11f };
	std::vector<float> importance;importance.push_back(1); importance.push_back(2); importance.push_back(3); importance.push_back(4); importance.push_back(5);

	Tensor<float> expected_output_tensor(0, output_dims);
	Tensor<float> output_tensor(output, output_dims);

	double feature1_cost = 3.5923;
	double feature2_cost = 3.6052;
	double feature3_cost = 2.8936;
	double feature4_cost = 3.8615;

	UnsupervisedGroupEntropyCostModule<float> cost_module(3);
	double cost = cost_module.GetCost(output_tensor, expected_output_tensor, importance, false, 0.5);
	BOOST_CHECK( abs(cost-5*0.5*(feature1_cost+feature2_cost+feature3_cost+feature4_cost)/4 )<0.001);
	cost = cost_module.GetCost(output_tensor, expected_output_tensor, importance, true, 0.5);
	BOOST_CHECK( abs(cost-5*0.5*(feature1_cost+feature2_cost+feature3_cost+feature4_cost)/60)<0.001);

	std::shared_ptr< Tensor<float> > gradients_tensor = cost_module.bprop(output_tensor, expected_output_tensor, importance, false, 0.5);

	float numerical_gradients[20]={ 0 };
	
	bool gradient_correct = true;
	float eps = 0.001f;
	for (size_t i=0; i<20; i++)
	{
		float initial_val = output[i];
		output[i]=initial_val-eps;
		double cost1 = cost_module.GetCost(output_tensor, expected_output_tensor, importance, false, 0.5);
		output[i]=initial_val+eps;
		double cost2 = cost_module.GetCost(output_tensor, expected_output_tensor, importance, false, 0.5);
		double numerical_gradient = (cost2-cost1) / 2 / eps;
		if (abs(numerical_gradient-(*gradients_tensor)[i]) / (std::max<double>)(abs(numerical_gradient)+abs((*gradients_tensor)[i]), 1) > 0.0001)
		{
			gradient_correct = false;
			break;
		}
		output[i] = initial_val;
	}
	BOOST_CHECK(gradient_correct);

	gradients_tensor = cost_module.bprop(output_tensor, expected_output_tensor, importance, true, 0.5);
	gradient_correct = true;
	for (size_t i=0; i<20; i++)
	{
		float initial_val = output[i];
		output[i]=initial_val-eps;
		double cost1 = cost_module.GetCost(output_tensor, expected_output_tensor, importance, true, 0.5);
		output[i]=initial_val+eps;
		double cost2 = cost_module.GetCost(output_tensor, expected_output_tensor, importance, true, 0.5);
		double numerical_gradient = (cost2-cost1) / 2 / eps;
		if (abs(numerical_gradient-(*gradients_tensor)[i]) / (std::max<double>)(abs(numerical_gradient)+abs((*gradients_tensor)[i]), 1) > 0.0001)
		{
			gradient_correct = false;
			break;
		}
		output[i] = initial_val;
	}
	BOOST_CHECK(gradient_correct);
}

BOOST_AUTO_TEST_CASE(TestUnsupervisedGroupEntropyCostModuleGradient)
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
		train_output[i] = GetRandomTensorPtr<double>(case_output_dims, 0.005, 0.995);

		double normalizer = 0;
		for (size_t j = 0; j<train_output[i]->Numel(); j++)
			normalizer += (*train_output[i])[j];
		for (size_t j = 0; j<train_output[i]->Numel(); j++)
			(*train_output[i])[j] = (*train_output[i])[j] / normalizer;

		train_importance[i]  = i+1.0;
	}
	
	std::shared_ptr< ITensorDataLoader<double> > input_data_loader(new FullTensorDataLoader<double,double>(train_input));
	std::shared_ptr< ITensorDataLoader<double> > output_data_loader(new FullTensorDataLoader<double,double>(train_output));
	TrainDataset<double> train_dataset(input_data_loader, output_data_loader, train_importance);

	std::shared_ptr<ParametersInitializer<double>> initializer(new GaussianInitializer<double>());
	std::shared_ptr<Regularizer<double>> regularizer(new WeightDecayRegularizer<double>(0.5));
	std::shared_ptr< Module<double> > m1(new LinearMixModule<double>("module1", 9,5,initializer, regularizer));
	std::shared_ptr< Module<double> > m2(new SoftmaxModule<double>("module2"));
	std::vector< std::shared_ptr< Module<double> > > modules; modules.push_back(m1); modules.push_back(m2);
	std::shared_ptr< CompositeModule<double> > main_module(new CompositeModule<double>("module3", modules));
	
	NN<double> net(main_module);
	net.InitializeParameters();
	size_t num_groups = 4;
	BOOST_CHECK(NumericalCheckNNGradients(net, UnsupervisedGroupEntropyCostModule<double>(num_groups), train_dataset, false));
}