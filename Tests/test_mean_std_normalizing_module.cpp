#include <boost/test/unit_test.hpp>
#include <assert.h>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "SigmoidModule.h"
#include "MeanStdNormalizingModule.h"
#include "LogisticCostModule.h"
#include "GaussianInitializer.h"
#include "WeightDecayRegularizer.h"
#include "LinearMixModule.h"
#include "CompositeModule.h"
#include "FullTensorDataLoader.h"
#include "TrainDataset.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestMeanStdNormalizingModule)
{
	float input[] = {0, 0, 1, 2, -1, -2,
					 4, 2,  1,  3, 0, 0};
	std::vector<size_t> input_dims; input_dims.push_back(4); input_dims.push_back(2);
	std::vector<size_t> per_case_input_dims; per_case_input_dims.push_back(4); per_case_input_dims.push_back(2);
	std::vector<size_t> output_dims = input_dims;
	std::vector<float> importances;importances.push_back(1); importances.push_back(1);

	Tensor<float> input_tensor(input+2, input_dims);
	Tensor<float> input_gradients_tensor(input_dims);
	float expected_output[] = {0.7311f, 0.8808f, 0.2689f, 0.1192f, 0.9820f, 0.8808f, 0.7311f, 0.9526f};
	
	MeanStdNormalizingModule<float> a("module1", 4, 1, 0.5);
	a.predict_fprop( std::shared_ptr<Tensor<float> >( new Tensor<float>(input_tensor) ) );
	std::shared_ptr<Tensor<float> > output_tensor = a.GetOutputBuffer();
	BOOST_CHECK(test_equal_arrays(input+2, output_tensor->GetStartPtr(), 8));
	a.train_fprop( std::shared_ptr<Tensor<float> >( new Tensor<float>(input_tensor) ) );
	std::vector<float> means = a.GetMeans();
	BOOST_CHECK( means.size() == 4);
	float expected_means[] = {1.875f, 1.5f, 0, 0.375f};
	BOOST_CHECK(test_equal_arrays(expected_means, means.data(), 4));
	float expected_stds[] = {1.46875f, 0.625f, 1.0f, 2.12734222f};
	std::vector<float> stds = a.GetStds();
	BOOST_CHECK( stds.size() == 4);
	BOOST_CHECK(test_equal_arrays(expected_stds, stds.data(), 4));
	
	a.predict_fprop( std::shared_ptr<Tensor<float> >( new Tensor<float>(input_tensor) ) );
	output_tensor = a.GetOutputBuffer();

	float expected_output2[] = {0, 0, (1-1.875f) / 1.46875f, 0.5f/0.625f, -1.0f, (-2-0.375f) / 2.12734222f,
					 (4-1.875f) / 1.46875f, (2-1.5f)/0.625f, 1.0f,  (3-0.375f) / 2.12734222f, 0, 0};
	
	BOOST_CHECK(test_equal_arrays(expected_output2+2, output_tensor->GetStartPtr(), 8));
	
	BOOST_CHECK(a.AlocateOutputBuffer());
	BOOST_CHECK(a.AlocateInputGradientsBuffer());
	BOOST_CHECK(a.GetNumParams() == 0);
	BOOST_CHECK(a.GetPerCaseOutputDims(per_case_input_dims).size() == 2 && a.GetPerCaseOutputDims(per_case_input_dims)[0] == 4 && 
		a.GetPerCaseOutputDims(per_case_input_dims)[1] == 2);
}

BOOST_AUTO_TEST_CASE(TestMeanStdNormalizingModuleGradient)
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
		train_importance[i]  = i+1.0;
	}
	
	std::shared_ptr< ITensorDataLoader<double> > input_data_loader(new FullTensorDataLoader<double,double>(train_input));
	std::shared_ptr< ITensorDataLoader<double> > output_data_loader(new FullTensorDataLoader<double,double>(train_output));
	TrainDataset<double> train_dataset(input_data_loader, output_data_loader, train_importance);

	std::shared_ptr<ParametersInitializer<double>> initializer(new GaussianInitializer<double>());
	std::shared_ptr<Regularizer<double>> regularizer(new WeightDecayRegularizer<double>(0.5));
	std::shared_ptr< Module<double> > m1(new LinearMixModule<double>("module1", 9,5,initializer, regularizer));
	std::shared_ptr< Module<double> > m2(new MeanStdNormalizingModule<double>("module2", 5, 1.0, 0.9));
	std::shared_ptr< Module<double> > m3(new SigmoidModule<double>("module3"));
	std::vector< std::shared_ptr< Module<double> > > modules; modules.push_back(m1); modules.push_back(m2); modules.push_back(m3);
	std::shared_ptr< CompositeModule<double> > main_module(new CompositeModule<double>("module4", modules));
	
	NN<double> net(main_module);
	net.GetCost(train_dataset, LogisticCostModule<double>(), std::vector<size_t>(5,1), true, false); // make MeanStdNormalizingModule change its values
	net.InitializeParameters();
	static_cast<MeanStdNormalizingModule<double>*>(&*m2)->DebugFreeze();
	BOOST_CHECK(NumericalCheckNNGradients(net, LogisticCostModule<double>(), train_dataset));
	static_cast<MeanStdNormalizingModule<double>*>(&*m2)->DebugUnfreeze();

	BOOST_CHECK( test_save_load_nn_state(net) );
}