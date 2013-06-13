#include <boost/test/unit_test.hpp>
#include <vector>
#include <algorithm>
#include "Tensor.h"
#include "SupervisedGroupEntropyCostModule.h"
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

BOOST_AUTO_TEST_CASE(TestSupervisedGroupEntropyCostModule)
{
	std::vector<size_t> output_dims;output_dims.push_back(2); output_dims.push_back(5);
	float output[] = { 0.1f,  0.15f,
					   0.05f, 0.18f,
					   0.23f, 0.17f,
					   0.35f, 0.45f,
					   0.27f, 0.05f};
	std::vector<float> importance;importance.push_back(1); importance.push_back(2); importance.push_back(3); importance.push_back(4); importance.push_back(5);

	Tensor<float> output_tensor(output, output_dims);
	
	float labels[] = { 0, 0, 1,
					1, 0, 0,
					0, 1, 0,
					1, 0, 0,
					0, 0, 1 };
	std::vector<size_t> labels_dims;labels_dims.push_back(3); labels_dims.push_back(5);
	Tensor<float> labels_tensor(labels, labels_dims);

	double feature1_cost = 5.5258;
	double feature2_cost = 3.6723;

	SupervisedGroupEntropyCostModule<float> cost_module;
	double cost = cost_module.GetCost(output_tensor, labels_tensor, importance, false, 0.5);
	BOOST_CHECK( abs(cost-5*0.5*(feature1_cost+feature2_cost)/2 )<0.001);
	cost = cost_module.GetCost(output_tensor, labels_tensor, importance, true, 0.5);
	BOOST_CHECK( abs(cost-5*0.5*(feature1_cost+feature2_cost)/30)<0.001);

	std::shared_ptr< Tensor<float> > gradients_tensor = cost_module.bprop(output_tensor, labels_tensor, importance, false, 0.5);
	
	bool gradient_correct = true;
	float eps = 0.001f;
	for (size_t i=0; i<10; i++)
	{
		float initial_val = output[i];
		output[i]=initial_val-eps;
		double cost1 = cost_module.GetCost(output_tensor, labels_tensor, importance, false, 0.5);
		output[i]=initial_val+eps;
		double cost2 = cost_module.GetCost(output_tensor, labels_tensor, importance, false, 0.5);
		double numerical_gradient = (cost2-cost1) / 2 / eps;
		if (abs(numerical_gradient-(*gradients_tensor)[i]) / (std::max<double>)(abs(numerical_gradient)+abs((*gradients_tensor)[i]), 1) > 0.0001)
		{
			gradient_correct = false;
			break;
		}
		output[i] = initial_val;
	}
	BOOST_CHECK(gradient_correct);

	gradients_tensor = cost_module.bprop(output_tensor, labels_tensor, importance, true, 0.5);
	gradient_correct = true;
	for (size_t i=0; i<10; i++)
	{
		float initial_val = output[i];
		output[i]=initial_val-eps;
		double cost1 = cost_module.GetCost(output_tensor, labels_tensor, importance, true, 0.5);
		output[i]=initial_val+eps;
		double cost2 = cost_module.GetCost(output_tensor, labels_tensor, importance, true, 0.5);
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

BOOST_AUTO_TEST_CASE(TestSupervisedGroupEntropyCostModuleGradient)
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

		auto max_element_pos = std::max_element(train_output[i]->GetStartPtr(), train_output[i]->GetStartPtr()+train_output[i]->Numel());
		for (size_t j = 0; j<train_output[i]->Numel(); j++)
			(*train_output[i])[j] = 0;
		*max_element_pos = 1;

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
	BOOST_CHECK(NumericalCheckNNGradients(net, SupervisedGroupEntropyCostModule<double>(), train_dataset, false));
}