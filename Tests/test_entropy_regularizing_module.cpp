#include <boost/test/unit_test.hpp>
#include <assert.h>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "EntropyRegularizingModule.h"
#include "SigmoidModule.h"
#include "MseCostModule.h"
#include "GaussianInitializer.h"
#include "WeightDecayRegularizer.h"
#include "LinearModule.h"
#include "CompositeModule.h"
#include "FullTensorDataLoader.h"
#include "TrainDataset.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestEntropyRegularizingModule)
{

	float input[] = {1, 2, 0, 2,
					 4, 2, 1, 3,
	                 5, 6, 9, 5};
	std::vector<size_t> input_dims; input_dims.push_back(4); input_dims.push_back(3);
	std::vector<size_t> per_case_input_dims; per_case_input_dims.push_back(4); per_case_input_dims.push_back(3);
	std::vector<size_t> output_dims = input_dims;
	std::vector<float> importances;importances.push_back(1); importances.push_back(1); importances.push_back(1);

	Tensor<float> input_tensor(input, input_dims);

	EntropyRegularizingModule<float> a("module1", 2, 0.5);
	a.train_fprop( std::shared_ptr<Tensor<float> >( new Tensor<float>(input_tensor) ) );
	std::shared_ptr<Tensor<float> > output_tensor = a.GetOutputBuffer();
	BOOST_CHECK(test_equal_arrays(input, output_tensor->GetStartPtr(), 12));

	double expected_cost = 0.5*0.4782*3;
	double cost = a.GetCost(importances);

	BOOST_CHECK( abs(expected_cost - cost) < 0.001);
	BOOST_CHECK(!a.AlocateOutputBuffer());
	BOOST_CHECK(a.AlocateInputGradientsBuffer());
	BOOST_CHECK(a.GetNumParams() == 0);
	BOOST_CHECK(a.GetPerCaseOutputDims(per_case_input_dims).size() == 2 && a.GetPerCaseOutputDims(per_case_input_dims)[0] == 4 && 
		a.GetPerCaseOutputDims(per_case_input_dims)[1] == 3);
}

BOOST_AUTO_TEST_CASE(TestEntropyRegularizingModuleGradient)
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

	std::shared_ptr<ParametersInitializer<double>> initializer(new GaussianInitializer<double>());
	std::shared_ptr<Regularizer<double>> regularizer(new WeightDecayRegularizer<double>(0.5));
	std::shared_ptr< Module<double> > m1(new LinearModule<double>("module1", case_input_dims,initializer, regularizer));
	std::shared_ptr< Module<double> > m2(new SigmoidModule<double>("module2"));
	std::shared_ptr< Module<double> > m3( new EntropyRegularizingModule<double>("module1", 3, 0.5) );
	std::vector< std::shared_ptr< Module<double> > > modules; modules.push_back(m1); modules.push_back(m2); modules.push_back(m3);
	std::shared_ptr< CompositeModule<double> > main_module(new CompositeModule<double>("module3", modules));
	
	NN<double> net(main_module);
	net.InitializeParameters();
	BOOST_CHECK(NumericalCheckNNGradients(net, MseCostModule<double>(), train_dataset, false));
	
	BOOST_CHECK( test_save_load_nn_state(net) );
}