#include <boost/test/unit_test.hpp>
#include <assert.h>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "PureSoftmaxModule.h"
#include "LogisticCostModule.h"
#include "EntropyCostModule.h"
#include "UniformInitializer.h"
#include "WeightDecayRegularizer.h"
#include "LinearModule.h"
#include "CompositeModule.h"
#include "FullTensorDataLoader.h"
#include "TrainDataset.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestPureSoftmaxModule)
{
	float input[] = {0, 0, 1, 2, 5, 0,
					 4, 2,  1,  3, 0, 0};
	std::vector<size_t> input_dims; input_dims.push_back(4); input_dims.push_back(2);
	std::vector<size_t> per_case_input_dims; per_case_input_dims.push_back(4); per_case_input_dims.push_back(2);
	std::vector<size_t> output_dims = input_dims;
	std::vector<float> importances;importances.push_back(1); importances.push_back(1);

	Tensor<float> input_tensor(input+2, input_dims);
	Tensor<float> input_gradients_tensor(input_dims);
	float expected_output[] = {1.0f/8, 0.25f, 5.0f/8, 0, 0.4f, 0.2f, 0.1f, 0.3f};
	
	PureSoftmaxModule<float> a("module1");
	a.train_fprop( std::shared_ptr<Tensor<float> >( new Tensor<float>(input_tensor) ) );
	std::shared_ptr<Tensor<float> > output_tensor = a.GetOutputBuffer();
	BOOST_CHECK(test_equal_arrays(expected_output, output_tensor->GetStartPtr(), 8));

	BOOST_CHECK(a.AlocateOutputBuffer());
	BOOST_CHECK(a.AlocateInputGradientsBuffer());
	BOOST_CHECK(a.GetNumParams() == 0);
	BOOST_CHECK(a.GetPerCaseOutputDims(per_case_input_dims).size() == 2 && a.GetPerCaseOutputDims(per_case_input_dims)[0] == 4 && 
		a.GetPerCaseOutputDims(per_case_input_dims)[1] == 2);
}

BOOST_AUTO_TEST_CASE(TestPureSoftmaxGradient)
{
	std::vector< std::shared_ptr< Tensor<double> > > train_input(15);
	std::vector< std::shared_ptr< Tensor<double> > > train_output(15);
	std::vector<double> train_importance(15);

	std::vector< std::shared_ptr< Tensor<double> > > test_input(8);
	std::vector< std::shared_ptr< Tensor<double> > > test_output(8);
	std::vector<double> test_importance(8);

	std::vector<size_t> case_input_dims;case_input_dims.push_back(9);
	std::vector<size_t> case_output_dims;case_output_dims.push_back(9);
	for (size_t i=0; i<train_input.size(); i++)
	{
		train_input[i] = GetRandomTensorPtr<double>(case_input_dims, 0.05, 1);
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

	std::shared_ptr<ParametersInitializer<double>> initializer(new UniformInitializer<double>(0.1, 1));
	std::shared_ptr<Regularizer<double>> regularizer(new WeightDecayRegularizer<double>(0.5));
	std::shared_ptr< Module<double> > m1(new LinearModule<double>("module1", case_input_dims,initializer, regularizer));
	std::shared_ptr< Module<double> > m2(new PureSoftmaxModule<double>("module2"));
	std::vector< std::shared_ptr< Module<double> > > modules; modules.push_back(m1); modules.push_back(m2);
	std::shared_ptr< CompositeModule<double> > main_module(new CompositeModule<double>("module3", modules));
	
	NN<double> net(main_module);
	net.InitializeParameters();
	BOOST_CHECK(NumericalCheckNNGradients(net, LogisticCostModule<double>(), train_dataset));
	BOOST_CHECK(NumericalCheckNNGradients(net, EntropyCostModule<double>(), train_dataset));

	BOOST_CHECK( test_save_load_nn_state(net) );
}