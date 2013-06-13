#include <boost/test/unit_test.hpp>
#include <vector>
#include <memory>
#include <sstream>
#include "Tensor.h"
#include "CompositeModule.h"
#include "FullTensorDataLoader.h"
#include "WeightDecayRegularizer.h"
#include "ConstantInitializer.h"
#include "test_utilities.h"
#include "LinearModule.h"
#include "BranchModule.h"
#include "BranchSemisupervisedNN.h"
#include "MseCostModule.h"
#include "AbsCostModule.h"

BOOST_AUTO_TEST_CASE(TestBranchModule)
{
	float input[] = {1, 2, -1, -2,
					 4, 2,  1,  3};
	float test_input[] = {1, 2, -1, -2,
					 4, 2,  1,  3};
	float parameters[4] = {1,4,2,3};
	float parameters_test[4] = {1,4,2,3};
	std::vector<size_t> input_dims; input_dims.push_back(4); input_dims.push_back(2);
	std::vector<size_t> per_case_input_dims; per_case_input_dims.push_back(4); per_case_input_dims.push_back(1);
	std::vector<size_t> output_dims = input_dims;
	Tensor<float> input_tensor(input, input_dims);
	
	std::vector<size_t> input_case_dims = input_tensor.GetDimensions();
	input_case_dims.pop_back();
	std::shared_ptr<ConstantInitializer<float> > initializer(new ConstantInitializer<float>(0));
	std::shared_ptr<AbsRegularizer<float>> regularizer(new AbsRegularizer<float>(0.5));
	std::shared_ptr< Module<float> > a1( new LinearModule<float>("module1", input_case_dims, initializer, regularizer) );
	BranchModule<float> a2("module2", a1);
	a2.SetParameters(parameters);

	a2.train_fprop( std::shared_ptr<Tensor<float> >( new Tensor<float>(input_tensor) ) );
	std::shared_ptr<Tensor<float> > output_tensor = a2.GetOutputBuffer();
	BOOST_CHECK(test_equal_arrays(test_input, output_tensor->GetStartPtr(), 8));
	output_tensor = a2.GetBranchModuleOutputBuffer();
	float expected_output[] = { 1, 8, -2, -6, 4, 8, 2, 9 };
	BOOST_CHECK(test_equal_arrays(expected_output, output_tensor->GetStartPtr(), 8));

	output_tensor = a1->GetOutputBuffer();
	BOOST_CHECK(test_equal_arrays(expected_output, output_tensor->GetStartPtr(), 8));
	
	output_tensor = a2.GetOutputBuffer();
	BOOST_CHECK(test_equal_arrays(test_input, output_tensor->GetStartPtr(), 8));

	BOOST_CHECK( a2.GetType() == "BranchModule" );

	std::vector<float> importances(2,1);
	BOOST_CHECK( a2.GetCost(importances) == 10);
	
	BOOST_CHECK( !a2.AlocateOutputBuffer());
	BOOST_CHECK( a2.AlocateInputGradientsBuffer() );
	BOOST_CHECK(a2.GetNumParams() == 4);
	BOOST_CHECK(a2.GetPerCaseOutputDims(per_case_input_dims).size() == 2 && a2.GetPerCaseOutputDims(per_case_input_dims)[0] == 4 && 
		a2.GetPerCaseOutputDims(per_case_input_dims)[1] == 1);

	// test initializer
	a2.InitializeParameters();
	std::vector<float> parameters_tensor;
	a2.GetParameters(parameters_tensor);
	for (size_t i=0; i<a2.GetNumParams(); i++)
		BOOST_CHECK(parameters_tensor[i] == 0);

	BOOST_CHECK( TestGetSetParameters<float>(a2, a2.GetNumParams()) );
}

BOOST_AUTO_TEST_CASE(TestBranchSemisupervisedNN)
{
	float input[] = {1, 2, -1, -2,
					 4, 2,  1,  3};

	float test_input[] = {1, 2, -1, -2,
					 4, 2,  1,  3};
	
	float parameters[16] = {1, 2, 1, 1, 5,1,3,4, 1,4,2,3, 4,1,2,1};

	std::vector<size_t> input_dims; input_dims.push_back(4); input_dims.push_back(2);
	std::vector<size_t> output_dims = input_dims;
	Tensor<float> input_tensor(input, input_dims);
	
	std::vector<size_t> input_case_dims = input_tensor.GetDimensions();
	input_case_dims.pop_back();
	std::shared_ptr<ConstantInitializer<float> > initializer(new ConstantInitializer<float>(0));
	std::shared_ptr<AbsRegularizer<float>> regularizer(new AbsRegularizer<float>(0.5));
	std::shared_ptr< Module<float> > a1( new LinearModule<float>("module1", input_case_dims, initializer, regularizer) );
	a1->SetParameters(parameters);
	std::shared_ptr< Module<float> > a2( new LinearModule<float>("module2", input_case_dims, initializer, regularizer) );
	std::shared_ptr< Module<float> > a3( new BranchModule<float>("module3", a2) );
	a3->SetParameters(parameters+4);
	std::shared_ptr< Module<float> > a4( new LinearModule<float>("module4", input_case_dims, initializer, regularizer) );
	std::shared_ptr< Module<float> > a5( new BranchModule<float>("module5", a4) );
	a5->SetParameters(parameters+8);
	std::shared_ptr< Module<float> > a6( new LinearModule<float>("module6", input_case_dims, initializer, regularizer) );
	a6->SetParameters(parameters+12);

	std::vector< std::shared_ptr< Module<float> > > modules; modules.push_back( a1); modules.push_back( a3 ); modules.push_back( a5 ); modules.push_back( a6 ); 
	std::shared_ptr< CompositeModule<float> > main_module( new CompositeModule<float>("Main", modules) );

	float first_layer_output[] = {1, 4, -1, -2,  4, 4, 1, 3};
	float main_expected_output[] = {4, 4, -2, -2,  16, 4, 2, 3};
	float first_branch_expected_output[] = {5, 4, -3, -8,  20, 4, 3, 12};
	float second_branch_expected_output[] = {1, 16, -2, -6,  4, 16, 2, 9};

	std::shared_ptr<Tensor<float> >  main_output = main_module->train_fprop( std::shared_ptr<Tensor<float> >( new Tensor<float>(input_tensor) ) );
	BOOST_CHECK(test_equal_arrays(main_expected_output, main_output->GetStartPtr(), 8));

	std::shared_ptr<Tensor<float> > first_branch_output_tensor = 
		static_cast< BranchModule<float>* >(&*main_module->GetModule("module3"))->GetBranchModuleOutputBuffer();
	BOOST_CHECK(test_equal_arrays(first_branch_expected_output, first_branch_output_tensor->GetStartPtr(), 8));
	
	std::shared_ptr<Tensor<float> > second_branch_output_tensor = 
		static_cast< BranchModule<float>* >(&*main_module->GetModule("module5"))->GetBranchModuleOutputBuffer();
	BOOST_CHECK(test_equal_arrays(second_branch_expected_output, second_branch_output_tensor->GetStartPtr(), 8));

	float branch_desired_output[] = { 0, 7, 2, -6, 5, 8, 2, 9 };

	BranchCostModuleInfo<float> branch1_cost_module( std::shared_ptr< CostModule<float> >( new MseCostModule<float>()), "module3", 0.5 );
	BranchCostModuleInfo<float> branch2_cost_module( std::shared_ptr< CostModule<float> >( new AbsCostModule<float>()), "module5", 0.25 );

	std::vector< BranchCostModuleInfo<float> > branch_cost_modules;
	branch_cost_modules.push_back(branch1_cost_module);
	branch_cost_modules.push_back(branch2_cost_module);


	BranchSemisupervisedNN<float> nn(main_module, 1000);
	
	std::vector< std::shared_ptr< Tensor<float> > > train_input(2);
	std::vector< std::shared_ptr< Tensor<float> > > train_output(2);
	std::vector<float> train_importance; train_importance.push_back(1); train_importance.push_back(2);
	train_input[0] = std::shared_ptr< Tensor<float> >( new Tensor<float>( input, input_case_dims) );
	train_input[1] = std::shared_ptr< Tensor<float> >( new Tensor<float>( input+4, input_case_dims) );
	train_output[0] = std::shared_ptr< Tensor<float> >( new Tensor<float>( branch_desired_output, input_case_dims) );
	train_output[1] = std::shared_ptr< Tensor<float> >( new Tensor<float>( branch_desired_output+4, input_case_dims) );
	
	std::shared_ptr< ITensorDataLoader<float> > input_data_loader(new FullTensorDataLoader<float,float>(train_input));
	std::shared_ptr< ITensorDataLoader<float> > output_data_loader(new FullTensorDataLoader<float,float>(train_output));
	TrainDataset<float> train_dataset(input_data_loader, output_data_loader, train_importance);

	std::vector<size_t> indices;
	indices.push_back(0);
	indices.push_back(1);
	BranchCostResult res = nn.GetCost(train_dataset, AbsCostModule<float>(), 1, branch_cost_modules, indices, true);
	
	//float input[] =                {1, 2, -1, -2,  4, 2, 1, 3};
	//float main_expected_output[] = {4, 4, -2, -2, 16, 4, 2, 3};
	
	//float branch_desired_output[] =         {0, 7,   2, -6,  5,  8, 2, 9};
	//float first_branch_expected_output[] =  {5, 4,  -3, -8, 20,  4, 3, 12};
	//float second_branch_expected_output[] = {1, 16, -2, -6,  4, 16, 2, 9};
	double expected_cost = ( (6+2*15) + 0.25*(25+9+25+4+2*(225+16+1.0+9)) + 0.25*(1.0+9+4+2*(1+8)) ) / 3;
	BOOST_CHECK( abs( expected_cost - res.main_cost ) < 0.0000001 );
	double expected_branch_cost1 = 0.5*(25+9+25+4+2*(225+16+1.0+9))/ 3.0;
	BOOST_CHECK( abs( res.branch_costs[0] - expected_branch_cost1 ) < 0.0000001 );
	double expected_branch_cost2 = (1.0+9+4+2*(1+8) ) / 3.0;
	BOOST_CHECK( abs( res.branch_costs[1] - expected_branch_cost2 ) < 0.0000001 );

	res = nn.GetCost(train_dataset, AbsCostModule<float>(), 1, branch_cost_modules, indices, true, true);
	expected_cost = ( (6+2*15) + 0.25*(25+9+25+4+2*(225+16+1.0+9)) + 0.25*(1.0+9+4+2*(1+8)) ) / 3.0+18;
	BOOST_CHECK( abs( expected_cost - res.main_cost ) < 0.0000001 );
	expected_branch_cost1 = 0.5*(25+9+25+4+2*(225+16+1.0+9))/ 3.0;
	BOOST_CHECK( abs( res.branch_costs[0] - expected_branch_cost1) < 0.00000001);
	expected_branch_cost2 = (1.0+9+4+2*(1+8) ) / 3.0;
	BOOST_CHECK( abs( res.branch_costs[1] - expected_branch_cost2 ) < 0.00000001 );



	nn.SetMinibatchSize(1);
	res = nn.GetCost(train_dataset, AbsCostModule<float>(), 1, branch_cost_modules, indices, true);
	expected_cost = ( (6+2*15) + 0.25*(25+9+25+4+2*(225+16+1.0+9)) + 0.25*(1.0+9+4+2*(1+8)) ) / 3;
	BOOST_CHECK( abs( expected_cost - res.main_cost ) < 0.0000001 );
	expected_branch_cost1 = 0.5*(25+9+25+4+2*(225+16+1.0+9))/ 3.0;
	BOOST_CHECK( abs( res.branch_costs[0] - expected_branch_cost1 ) < 0.0000001 );
	expected_branch_cost2 = (1.0+9+4+2*(1+8) ) / 3.0;
	BOOST_CHECK( abs( res.branch_costs[1] - expected_branch_cost2 ) < 0.0000001 );

	res = nn.GetCost(train_dataset, AbsCostModule<float>(), 1, branch_cost_modules, indices, true, true);
	expected_cost = ( (6+2*15) + 0.25*(25+9+25+4+2*(225+16+1.0+9)) + 0.25*(1.0+9+4+2*(1+8)) ) / 3.0+18;
	BOOST_CHECK( abs( expected_cost - res.main_cost ) < 0.0000001 );
	expected_branch_cost1 = 0.5*(25+9+25+4+2*(225+16+1.0+9))/ 3.0;
	BOOST_CHECK( abs( res.branch_costs[0] - expected_branch_cost1) < 0.00000001);
	expected_branch_cost2 = (1.0+9+4+2*(1+8) ) / 3.0;
	BOOST_CHECK( abs( res.branch_costs[1] - expected_branch_cost2 ) < 0.00000001 );
	


	BOOST_CHECK( a5->GetNumParams() == 4);
	BOOST_CHECK( a3->GetNumParams() == 4);
	BOOST_CHECK( a1->GetNumParams() == 4);
	BOOST_CHECK( a6->GetNumParams() == 4);
	BOOST_CHECK( main_module->GetNumParams() == 16);

	main_module->InitializeParameters();
	std::vector<float> parameters_vect;
	main_module->GetParameters(parameters_vect);
	for (size_t i=0; i<main_module->GetNumParams(); i++)
		BOOST_CHECK(parameters_vect[i] == 0);
	
	BOOST_CHECK( TestGetSetParameters<float>(*a5, a5->GetNumParams()) );
	BOOST_CHECK( TestGetSetParameters<float>(*main_module, main_module->GetNumParams()) );
}

bool CheckSemisupervisedNNGradientsSameForDifferentBufferSizes(BranchSemisupervisedNN<double>& nn, CostModule<double>& main_cost_module, double main_lambda, 
											 std::vector< BranchCostModuleInfo<double> >& branch_cost_modules, ITrainDataset<double>& dataset)
{
	size_t initial_buffer_size = nn.GetMinibatchSize();
	size_t num_params = nn.GetNumParams();
	std::vector<double> parameters = nn.GetParameters();
	nn.SetMinibatchSize(100);
	CostAndGradients<double> res1 = nn.GetGradientsAndCost(dataset, main_cost_module, main_lambda, branch_cost_modules, std::vector<size_t>(), true).first;
	std::vector<double> gradients1 = res1.gradients;
	double cost1 = res1.cost;

	nn.SetMinibatchSize(4);
	CostAndGradients<double> res2 = nn.GetGradientsAndCost(dataset, main_cost_module, main_lambda, branch_cost_modules, std::vector<size_t>(), true).first;
	std::vector<double> gradients2 = res2.gradients;
	double cost2 = res2.cost;

	if ( abs(cost1 - cost2) > 0.000000001)
		return false;

	for (size_t i=0; i<num_params; i++)
		if( abs(gradients1[i] - gradients2[i]) >0.000000001)
			return false;
	nn.SetMinibatchSize(initial_buffer_size);
	return true;
}

bool NumericalCheckSemisupervisedNNGradients(BranchSemisupervisedNN<double>& nn, CostModule<double>& main_cost_module, double main_lambda, 
											 std::vector< BranchCostModuleInfo<double> >& branch_cost_modules, ITrainDataset<double>& dataset)
{
	// check that each fprop does not affect other fprops and that the cost is consistent
	if (nn.GetCost(dataset, main_cost_module, main_lambda, branch_cost_modules, std::vector<size_t>(), true, true).main_cost != 
		nn.GetCost(dataset, main_cost_module, main_lambda, branch_cost_modules, std::vector<size_t>(), true, true).main_cost)
			return false;

	if (nn.GetCost(dataset, main_cost_module, main_lambda, branch_cost_modules, std::vector<size_t>(), true, true).branch_costs != 
		nn.GetCost(dataset, main_cost_module, main_lambda, branch_cost_modules, std::vector<size_t>(), true, true).branch_costs)
			return false;
	
	if (nn.GetCost(dataset, main_cost_module, main_lambda, branch_cost_modules, std::vector<size_t>(), false, true).main_cost != 
		nn.GetCost(dataset, main_cost_module, main_lambda, branch_cost_modules, std::vector<size_t>(), false, true).main_cost)
			return false;
	
	if (nn.GetCost(dataset, main_cost_module, main_lambda, branch_cost_modules, std::vector<size_t>(), false, false).branch_costs != 
		nn.GetCost(dataset, main_cost_module, main_lambda, branch_cost_modules, std::vector<size_t>(), false, false).branch_costs)
			return false;

	size_t num_params = nn.GetNumParams();
	std::vector<double> parameters = nn.GetParameters();
	CostAndGradients<double> res1 = nn.GetGradientsAndCost(dataset, main_cost_module, main_lambda, branch_cost_modules, std::vector<size_t>(), true).first;
	std::vector<double> gradients1 = res1.gradients;
	// Test that all buffers are cleared
	CostAndGradients<double> res2 = nn.GetGradientsAndCost(dataset, main_cost_module, main_lambda, branch_cost_modules, std::vector<size_t>(), true).first;
	std::vector<double> gradients = res2.gradients;
	if (res1.gradients != gradients)
		return false;

	std::vector<double> numerical_gradients;
	double eps = 1e-5;

	for (size_t i=0; i<num_params; i++)
	{
		double initial_val = parameters[i];
		parameters[i]=initial_val-eps;
		nn.SetParameters(parameters);
		double cost1 = nn.GetCost(dataset, main_cost_module, main_lambda, branch_cost_modules, std::vector<size_t>(), true, true).main_cost;
		parameters[i]=initial_val+eps;
		// Use the array approach for initializing parameters
		nn.SetParameters(parameters.data());
		double cost2 = nn.GetCost(dataset, main_cost_module, main_lambda, branch_cost_modules, std::vector<size_t>(), true, true).main_cost;
		double numerical_gradient = (cost2-cost1) / 2 / eps;
		if (abs(numerical_gradient-gradients[i]) / (std::max<double>)(abs(numerical_gradient)+abs(gradients[i]), 1) > 0.000001)
			return false;
		parameters[i] = initial_val;
		nn.SetParameters(parameters);
	}
	return CheckSemisupervisedNNGradientsSameForDifferentBufferSizes(nn, main_cost_module, main_lambda, branch_cost_modules, dataset);
}
template <class T>
bool test_save_load_branch_nn_state(BranchSemisupervisedNN<T>& net)
{
	std::shared_ptr<IOTreeNode> net_state = net.GetState();
	std::stringstream stream;
	IOXML::save(*net_state, stream);
	net_state = IOXML::load(stream);
	std::shared_ptr< BranchSemisupervisedNN<T> > net2 = BranchSemisupervisedNN<T>::Create(*net_state);
	return net.Equals( *net2 );
}

BOOST_AUTO_TEST_CASE(TestBranchSemisupervisedNNGradient)
{
	std::vector< std::shared_ptr< Tensor<double> > > train_input(15);
	std::vector< std::shared_ptr< Tensor<double> > > train_output(15);
	std::vector<double> train_importance(15);

	std::vector<size_t> case_input_dims;case_input_dims.push_back(5);
	std::vector<size_t> case_output_dims;case_output_dims.push_back(5);
	for (size_t i=0; i<train_input.size(); i++)
	{
		train_input[i] = GetRandomTensorPtr<double>(case_input_dims);
		train_output[i] = GetRandomTensorPtr<double>(case_output_dims);
		train_importance[i]  = i+1.0;
	}
	
	std::shared_ptr< ITensorDataLoader<double> > input_data_loader(new FullTensorDataLoader<double,double>(train_input));
	std::shared_ptr< ITensorDataLoader<double> > output_data_loader(new FullTensorDataLoader<double,double>(train_output));
	TrainDataset<double> train_dataset(input_data_loader, output_data_loader, train_importance);
	
	std::shared_ptr<ParametersInitializer<double> > initializer(new GaussianInitializer<double>());
	std::shared_ptr<Regularizer<double>> regularizer(new WeightDecayRegularizer<double>(0.5));
	std::shared_ptr< Module<double> > m1( new LinearModule<double>("module1", case_input_dims, initializer, regularizer) );
	std::shared_ptr< Module<double> > m2( new LinearModule<double>("module2", case_input_dims, initializer, regularizer) );
	std::shared_ptr< Module<double> > m3( new BranchModule<double>("module3", m2) );
	std::shared_ptr< Module<double> > m4( new LinearModule<double>("module4", case_input_dims, initializer, regularizer) );
	std::shared_ptr< Module<double> > m5( new BranchModule<double>("module5", m4) );
	std::shared_ptr< Module<double> > m6( new LinearModule<double>("module6", case_input_dims, initializer, regularizer) );
	std::vector< std::shared_ptr< Module<double> > > modules; modules.push_back( m1); modules.push_back( m3 ); modules.push_back( m5 ); modules.push_back( m6 ); 
	std::shared_ptr< CompositeModule<double> > main_module( new CompositeModule<double>("Main", modules) );
	BranchCostModuleInfo<double> branch1_cost_module( std::shared_ptr< CostModule<double> >( new MseCostModule<double>()), "module3", 0.5 );
	BranchCostModuleInfo<double> branch2_cost_module( std::shared_ptr< CostModule<double> >( new AbsCostModule<double>()), "module5", 0.25 );
	std::vector< BranchCostModuleInfo<double> > branch_cost_modules;
	branch_cost_modules.push_back(branch1_cost_module);
	branch_cost_modules.push_back(branch2_cost_module);

	BranchSemisupervisedNN<double> net(main_module, 1000);

	double main_lambda = 1;
	net.InitializeParameters();
	BOOST_CHECK(NumericalCheckSemisupervisedNNGradients(net, MseCostModule<double>(), main_lambda, branch_cost_modules, train_dataset));
	
	BOOST_CHECK( test_save_load_branch_nn_state(net) );
}