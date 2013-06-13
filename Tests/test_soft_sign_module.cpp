#include <boost/test/unit_test.hpp>
#include <assert.h>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "SoftSignModule.h"
#include "MseCostModule.h"
#include "GaussianInitializer.h"
#include "WeightDecayRegularizer.h"
#include "LinearMixModule.h"
#include "CompositeModule.h"
#include "FullTensorDataLoader.h"
#include "TrainDataset.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestSoftSignGradient)
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
	std::shared_ptr< Module<double> > m1(new LinearMixModule<double>("module1", 9,5,initializer, regularizer));
	std::shared_ptr< Module<double> > m2(new SoftSignModule<double>("module2"));
	std::vector< std::shared_ptr< Module<double> > > modules; modules.push_back(m1); modules.push_back(m2);
	std::shared_ptr< CompositeModule<double> > main_module(new CompositeModule<double>("module3", modules));
	
	NN<double> net(main_module);
	net.InitializeParameters();
	BOOST_CHECK(NumericalCheckNNGradients(net, MseCostModule<double>(), train_dataset));
	
	BOOST_CHECK( test_save_load_nn_state(net) );
}