#include <boost/test/unit_test.hpp>
#include <vector>
#include "Tensor.h"
#include "LinearMixModule.h"
#include "ConstantInitializer.h"
#include "WeightDecayRegularizer.h"
#include "BiasModule.h"
#include "AbsModule.h"
#include "CompositeModule.h"
#include "FullTensorDataLoader.h"
#include "MseCostModule.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestNN)
{
	std::vector< std::shared_ptr< Tensor<float> > > test_input(8);
	std::vector< std::shared_ptr< Tensor<float> > > test_output(8);
	std::vector<float> test_importance(8);

	std::vector<size_t> case_input_dims;case_input_dims.push_back(4);
	std::vector<size_t> case_output_dims;case_output_dims.push_back(3);

	std::vector< std::shared_ptr< Module<float> > > modules;
	float linear_mix_input[] = {1, 2, -1, -2,
								4, 2,  1,  3};
	float linear_mix_gradients[12] = {0};
	float linear_mix_parameters[12] = {1,4,2,
							3,2,4,
							5,1,3,
							8,1,2};
	std::vector<size_t> input_dims; input_dims.push_back(4); input_dims.push_back(2);
	std::vector<size_t> per_case_input_dims; per_case_input_dims.push_back(4);
	std::vector<size_t> output_dims; output_dims.push_back(3); output_dims.push_back(2);

	std::vector< std::shared_ptr< Tensor<float> > > input_tensors;
	input_tensors.push_back(std::shared_ptr< Tensor<float> >(new Tensor<float>(linear_mix_input, per_case_input_dims)) );
	input_tensors.push_back(std::shared_ptr< Tensor<float> >(new Tensor<float>(linear_mix_input+4, per_case_input_dims)) );

	//float expected_output[] = {-14, 5, 3, 39, 24,25};
	
	std::shared_ptr<ConstantInitializer<float>> initializer(new ConstantInitializer<float>(0.5));
	std::shared_ptr<WeightDecayRegularizer<float>> regularizer(new WeightDecayRegularizer<float>(0.5));
	std::shared_ptr< LinearMixModule<float> > lmm(new LinearMixModule<float>("module1", 4,3,initializer, regularizer));
	lmm->SetParameters(linear_mix_parameters);
	modules.push_back(lmm);

	float bias_gradients[3] = {0};
	float bias_parameters[3] = {4,-2,-8};
	std::vector<size_t> bias_input_dims; bias_input_dims.push_back(3);
	//float expected_output[] = {-10, 3, -5, 43, 22, 17};
	
	std::shared_ptr< BiasModule<float> > bias_module(new BiasModule<float>("module2", bias_input_dims, initializer, regularizer));
	bias_module->SetParameters(bias_parameters);
	modules.push_back(bias_module);

	float abs_gradients[1] = {0};
	float abs_parameters[1] = {0};
	std::vector<size_t> abs_input_dims; abs_input_dims.push_back(3);
	//float expected_output[] = {10, 3, 5, 43, 22, 17};
	
	std::shared_ptr< AbsModule<float> > abs_module(new AbsModule<float>("module3"));
	modules.push_back(abs_module);

	float expected_output[] = {10, 3, 5, 43, 22, 17};
	std::vector<size_t> expected_output_dims; expected_output_dims.push_back(3); expected_output_dims.push_back(2);
	std::shared_ptr< Tensor<float> > expected_output_tensor(new Tensor<float>(expected_output, expected_output_dims) );
	std::vector<size_t> expected_per_case_output_dims; expected_per_case_output_dims.push_back(3);
	
	std::vector< std::shared_ptr< Tensor<float> > > expected_output_tensors;
	expected_output_tensors.push_back(std::shared_ptr< Tensor<float> >(new Tensor<float>(expected_output, expected_per_case_output_dims)) );
	expected_output_tensors.push_back(std::shared_ptr< Tensor<float> >(new Tensor<float>(expected_output+3, expected_per_case_output_dims)) );

	std::shared_ptr< CompositeModule<float> > composite_module(new CompositeModule<float>("module4", modules));
	size_t num_params = 15;

	float params[15] = {1,4,2,
						3,2,4,
						5,1,3,
						8,1,2,
						4,-2,-8};
	float gradients[15] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,2,2};
	composite_module->SetParameters(params);
	
	std::vector<float> train_importance;
	train_importance.push_back(1);
	train_importance.push_back(1);
	
	std::shared_ptr< ITensorDataLoader<float> > input_data_loader(new FullTensorDataLoader<float,float>(input_tensors));
	std::shared_ptr< ITensorDataLoader<float> > output_data_loader(new FullTensorDataLoader<float,float>(expected_output_tensors));
	TrainDataset<float> train_dataset(input_data_loader, output_data_loader, train_importance);

	std::shared_ptr< NN<float> > net(new NN<float>(composite_module));
	net->SetParameters(params);

	//test NN
	double cost = net->GetCost(train_dataset, MseCostModule<float>(), std::vector<size_t>(), true, false);
	BOOST_CHECK(cost == 0);
	net->SetMinibatchSize(1);
	BOOST_CHECK( net->GetMinibatchSize() == 1);
	cost = net->GetCost(train_dataset, MseCostModule<float>(), std::vector<size_t>(), true, false);
	BOOST_CHECK(cost == 0);

	// test that we don't always get zero cost
	linear_mix_input[0] = 0;
	cost = net->GetCost(train_dataset, MseCostModule<float>(), std::vector<size_t>(), true, false);
	BOOST_CHECK(cost == 2.25); // 4.5 for 1 sample and 0 for 2

	// test cost with regularization
	cost = net->GetCost(train_dataset, MseCostModule<float>(), std::vector<size_t>(), true, true);
	BOOST_CHECK(cost == 2.25+59.5); // 4.5 for 1 sample and 0 for 2
	
	// test that we can fprop each input separately
	std::vector<size_t> inds;inds.push_back(0);
	cost = net->GetCost(train_dataset, MseCostModule<float>(), inds, true, false);
	BOOST_CHECK(cost == 4.5);
	inds[0] = 1;
	cost = net->GetCost(train_dataset, MseCostModule<float>(), inds, true, false);
	BOOST_CHECK(cost == 0);
	linear_mix_input[1] = 1;
	
	BOOST_CHECK( test_save_load_nn_state(*net) );
}