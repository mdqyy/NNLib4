#include <boost/test/unit_test.hpp>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "DropoutModule.h"
#include "CompositeModule.h"
#include "FullTensorDataLoader.h"
#include "WeightDecayRegularizer.h"
#include "ConstantInitializer.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestDropoutModule_deterministic)
{
	double input[] = {0, 0, 1, 2, -1, -2,
					 4, 2,  1,  3, 0, 0};
	std::vector<size_t> input_dims; input_dims.push_back(4); input_dims.push_back(2);
	std::vector<size_t> per_case_input_dims; per_case_input_dims.push_back(4); per_case_input_dims.push_back(2);
	std::vector<size_t> output_dims = input_dims;
	std::vector<double> importances;importances.push_back(1); importances.push_back(1);

	Tensor<double> input_tensor(input+2, input_dims);
	double expected_output[] = {0.7, 1.4, -0.7, -1.4, 2.8, 1.4, 0.7, 2.1};
	
	DropoutModule<double> a("module1", 0.7);
	a.predict_fprop( std::shared_ptr<Tensor<double> >( new Tensor<double>(input_tensor) ) );
	std::shared_ptr<Tensor<double> > output_tensor = a.GetOutputBuffer();
	BOOST_CHECK(test_equal_arrays(expected_output, output_tensor->GetStartPtr(), 8));
	BOOST_CHECK(a.AlocateOutputBuffer());
	BOOST_CHECK(a.AlocateInputGradientsBuffer());
	BOOST_CHECK(a.GetNumParams() == 0);
	BOOST_CHECK(a.GetPerCaseOutputDims(per_case_input_dims).size() == 2 && a.GetPerCaseOutputDims(per_case_input_dims)[0] == 4 && 
		a.GetPerCaseOutputDims(per_case_input_dims)[1] == 2);
}

BOOST_AUTO_TEST_CASE(TestDropoutModule_stochastic)
{
	size_t num_inputs = 10000;
	std::vector<size_t> input_dims;input_dims.push_back(num_inputs);input_dims.push_back(2);
	std::shared_ptr< Tensor<double> > train_input = GetRandomTensorPtr<double>(input_dims);
	DropoutModule<double> m1("module1", 0.75);
	std::shared_ptr< Tensor<double> > res = m1.train_fprop(train_input);

	for (size_t train_sample_ind = 0; train_sample_ind < 2; train_sample_ind++)
	{
		size_t num_dropouts = 0;
		for (size_t input_ind = 0; input_ind < num_inputs; input_ind++)
			num_dropouts += static_cast<size_t>((*res)[train_sample_ind*num_inputs + input_ind]==0);
		BOOST_CHECK( abs( static_cast<double>(num_dropouts) / num_inputs -0.75 ) < 0.05 );
	}
}

BOOST_AUTO_TEST_CASE(TestDropoutGradient)
{
	// it is random, not clear how to test it
}

BOOST_AUTO_TEST_CASE(TestDropoutSaveLoadState)
{
	std::vector< std::shared_ptr< Tensor<double> > > train_input(15);
	std::vector< std::shared_ptr< Tensor<double> > > train_output(15);
	std::vector<double> train_importance(15);

	std::vector<size_t> case_input_dims;case_input_dims.push_back(9);
	std::vector<size_t> case_output_dims;case_output_dims.push_back(9);
	for (size_t i=0; i<train_input.size(); i++)
	{
		train_input[i] = GetRandomTensorPtr<double>(case_input_dims);

		// avoid nonlinearity near zero
		for (size_t j=0; j<train_input[i]->Numel(); j++)
			if ( abs((*train_input[i])[j])<0.05 )
				(*train_input[i])[j] = 0.05;

		train_output[i] = GetRandomTensorPtr<double>(case_output_dims);
		train_importance[i]  = i+1.0;
	}
	
	std::shared_ptr< ITensorDataLoader<double> > input_data_loader(new FullTensorDataLoader<double,double>(train_input));
	std::shared_ptr< ITensorDataLoader<double> > output_data_loader(new FullTensorDataLoader<double,double>(train_output));
	TrainDataset<double> train_dataset(input_data_loader, output_data_loader, train_importance);

	// initialize bias to very small weights so that not to get too close to the nonlinearity
	std::shared_ptr<ParametersInitializer<double>> initializer(new ConstantInitializer<double>(2.5));
	std::shared_ptr<Regularizer<double>> regularizer(new WeightDecayRegularizer<double>(0.5));

	std::shared_ptr< Module<double> > m1(new LinearModule<double>("module1", case_input_dims,initializer, regularizer));
	std::shared_ptr< Module<double> > m2(new DropoutModule<double>("module2", 0.5));
	std::vector< std::shared_ptr< Module<double> > > modules; modules.push_back(m1); modules.push_back(m2);
	std::shared_ptr< CompositeModule<double> > main_module(new CompositeModule<double>("module3", modules));

	NN<double> net(main_module);
	net.InitializeParameters();
	BOOST_CHECK( test_save_load_nn_state(net) );
}