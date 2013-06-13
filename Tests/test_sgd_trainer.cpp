#include <boost/test/unit_test.hpp>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "Preprocessing.h"
#include "LinearMixInitializer.h"
#include "GaussianInitializer.h"
#include "BiasModule.h"
#include "SigmoidModule.h"
#include "CompositeModule.h"
#include "LinearMixModule.h"
#include "LogisticCostModule.h"
#include "SGD_Trainer.h"
#include "FullTensorDataLoader.h"
#include "NN.h"


BOOST_AUTO_TEST_CASE(testSgdTrainer)
{
	// solve simple problem witn neural network (linearly separable)

	double train_cases[] = {1,0,   -1, 1,   1,-1,    0.5,0,    -0.5,0,   -0.67,-0.09,    0.58,-0.83,    -0.47,0.07};
	double train_labels[] = {1, 0, 1, 1, 0, 0, 1, 0};
	
	double validation_cases[] = {-0.78,-0.72,   0.92,0.73,   -0.99,0.15,    0.54,0.09,    0.63,-0.71,   
		0.73,-0.70,    -0.83,0.24,    -0.20,-0.29, 0.56,-0.72,    0.22,0.29,    0.19,-0.29};
	double validation_labels[] = {0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1};

	double test_cases[] = {-0.02,-0.53,   -0.32,-0.29,   -0.80,0.64,    -0.26,-0.69,    -0.77,0.91};

	size_t num_inputs = 2;
	size_t num_outputs = 1;
	size_t num_train_cases = 8;
	size_t num_validation_cases = 11;
	size_t num_test_cases = 5;


	std::vector<size_t> input_dims; input_dims.push_back(num_inputs);
	std::vector<size_t> output_dims; output_dims.push_back(num_outputs);
	
	std::vector< std::shared_ptr< Tensor<double> > > train_input(num_train_cases);
	std::vector< std::shared_ptr< Tensor<double> > > train_output(num_train_cases);
	std::vector<double> train_importance(num_train_cases);
	
	std::vector< std::shared_ptr< Tensor<double> > > test_input(num_test_cases);

	for (size_t i=0; i<num_train_cases; i++)
	{
		train_input[i] = std::shared_ptr< Tensor<double> >( new Tensor<double>(train_cases+num_inputs*i, input_dims));
		train_output[i] = std::shared_ptr< Tensor<double> >( new Tensor<double>(train_labels+num_outputs*i, output_dims));
		train_importance[i]  = 1.0;
	}
	
	for (size_t i=0; i<num_test_cases; i++)
		test_input[i] = std::shared_ptr< Tensor<double> >( new Tensor<double>(test_cases+num_inputs*i, input_dims));
		
	std::shared_ptr< ITensorDataLoader<double> > input_data_loader(new FullTensorDataLoader<double,double>(train_input));
	std::shared_ptr< ITensorDataLoader<double> > output_data_loader(new FullTensorDataLoader<double,double>(train_output));
	std::shared_ptr< ITrainDataset<double> > train_dataset( new TrainDataset<double>(input_data_loader, output_data_loader, train_importance) );

	std::shared_ptr<ParametersInitializer<double>> lmm_initializer(new LinearMixInitializer<double>());
	std::shared_ptr<Regularizer<double>> regularizer(new EmptyRegularizer<double>());
	std::shared_ptr<ParametersInitializer<double>> bias_initializer(new GaussianInitializer<double>(0.001));
	
	std::shared_ptr< Module<double> > m1(new LinearMixModule<double>("module1", num_inputs,num_outputs,lmm_initializer, regularizer));
	std::shared_ptr< Module<double> > m2(new BiasModule<double>("module2", output_dims,bias_initializer, regularizer));
	std::shared_ptr< Module<double> > m3(new SigmoidModule<double>("module3"));
	std::vector< std::shared_ptr< Module<double> > > modules; modules.push_back(m1); modules.push_back(m2); modules.push_back(m3);
	std::shared_ptr< CompositeModule<double> > main_module(new CompositeModule<double>("module4", modules));
	
	NN<double> net(main_module);
	net.InitializeParameters();

	SGD_Trainer<double> trainer(1000, 0.1, 0.99, 1,100, 0.99, 0, 100, 100, 1000, 0.5);
	trainer.Train(net, LogisticCostModule<double>(), LogisticCostModule<double>(), *train_dataset, *train_dataset);
	
	auto predicted_labels = net.Predict( *input_data_loader );
	BOOST_CHECK( (*predicted_labels[0])[0] > 0.5);
	BOOST_CHECK( (*predicted_labels[1])[0] < 0.5);
	BOOST_CHECK( (*predicted_labels[2])[0] > 0.5);
	BOOST_CHECK( (*predicted_labels[3])[0] > 0.5);
	BOOST_CHECK( (*predicted_labels[4])[0] < 0.5);
	BOOST_CHECK( (*predicted_labels[5])[0] < 0.5);
	BOOST_CHECK( (*predicted_labels[6])[0] > 0.5);
	BOOST_CHECK( (*predicted_labels[7])[0] < 0.5);
}