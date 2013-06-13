#ifndef WHHALES_DETECTION_H
#define WHALES_DETECTION_H

#include <fstream>
#include <vector>
#include <memory>
#include <sstream>
#include <istream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "GaussianInitializer.h"
#include "ConstantInitializer.h"
#include "LinearMixInitializer.h"
#include "RandomShiftPartialTensorDataLoader.h"
#include "FixedShiftPartialTensorDataLoader.h"
#include "FullTensorDataLoader.h"
#include "SGD_Trainer.h"
#include "CompositeModule.h"
#include "LogisticCostModule.h"
#include "RectifiedLinearUnitModule.h"
#include "AbsModule.h"
#include "BiasModule.h"
#include "LinearMixModule.h"
#include "SoftSignModule.h"
#include "SigmoidModule.h"
#include "TanhModule.h"
#include "KernelModule.h"
#include "WeightDecayRegularizer.h"
#include "EmptyRegularizer.h"
#include "Utilities.h"
#include "TrainDataset.h"
#include  "ITensorDataLoader.h"
#include "KernelFactory.h"
#include "Preprocessing.h"

template <class T>
class SavedVector
{
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
    {
        ar & data;
    }
public:
	
	std::vector<T> data;
	SavedVector(std::vector<T>& saved_data)
	{
		data = saved_data;
	}
};

std::vector<float> LoadParams(std::string filepath)
{
	std::vector<float> res;
	{
        std::ifstream ifs(filepath);
        boost::archive::binary_iarchive ia(ifs);
        ia >> res;
    }

	return res;
}

void SaveParams(std::string filepath, std::vector<float> vect)
{
	std::ofstream ofs(filepath);
	{
        boost::archive::binary_oarchive oa(ofs);
        oa << vect;
    }
}

std::vector<std::string>& split(const std::string& str, char delimeter, std::vector<std::string>& elems) 
{
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delimeter)) 
        elems.push_back(item);
    return elems;
}

std::vector<std::string> split(const std::string &str, char delimeter) 
{
    std::vector<std::string> elems;
    return split(str, delimeter, elems);
}

void test_save_load_vector()
{
	std::vector<float> x;
	x.push_back(0.5);
	x.push_back(1);
	x.push_back(0.14f);
	x.push_back(-4);

	SavedVector<float> sv(x);
	SaveParams("net_parameters", x);
	std::vector<float> x2 = LoadParams("net_parameters");

	assert(x == x2);
}

template <class T>
std::vector<size_t> AddBiasModule(std::vector< std::shared_ptr< Module<T> > >& modules, std::string name, const std::vector<size_t>& input_dims)
{
	std::shared_ptr< ParametersInitializer<T> > bias_initializer(new GaussianInitializer<T>(0, 0.01));
	std::shared_ptr< Regularizer<T> > bias_regularizer(new EmptyRegularizer<T>());
	std::shared_ptr< Module<T> > bias_module( new BiasModule<T>(name, input_dims, bias_initializer, bias_regularizer) );
	modules.push_back(bias_module);
	return input_dims;
}

template <class T>
std::vector<size_t> AddConvModule(std::vector< std::shared_ptr< Module<T> > >& modules, std::string name,  const std::vector<size_t>& input_dims, 
								  size_t num_output_kernels, const std::vector<size_t>& kernel_dims, const std::vector<size_t>& kernel_strides, double weight_decay)
{
	std::shared_ptr< ParametersInitializer<T> > conv_initializer(new GaussianInitializer<T>(0, 0.1));
	std::shared_ptr< Regularizer<T> > conv_regularizer(new WeightDecayRegularizer<T>(weight_decay));
	std::shared_ptr< Module<T> > conv_module( new KernelModule<T>(name, num_output_kernels,kernel_dims,kernel_strides, 
		ConvolutionalKernelFactory<T>(), conv_initializer, conv_regularizer) );
	modules.push_back(conv_module);
	return conv_module->GetPerCaseOutputDims(input_dims);
}

template <class T>
std::vector<size_t> AddRluModule(std::vector< std::shared_ptr< Module<T> > >& modules, std::string name, const std::vector<size_t>& input_dims)
{
	std::shared_ptr< Module<T> > module( new RectifiedLinearUnitModule<T>(name) );
	modules.push_back(module);
	return input_dims;
}

template <class T>
std::vector<size_t> AddAbsModule(std::vector< std::shared_ptr< Module<T> > >& modules, std::string name, const std::vector<size_t>& input_dims)
{
	std::shared_ptr< Module<T> > module( new AbsModule<T>(name) );
	modules.push_back(module);
	return input_dims;
}

template <class T>
std::vector<size_t> AddSigmoidModule(std::vector< std::shared_ptr< Module<T> > >& modules, std::string name, const std::vector<size_t>& input_dims)
{
	std::shared_ptr< Module<T> > module( new SigmoidModule<T>(name) );
	modules.push_back(module);
	return input_dims;
}

template <class T>
std::vector<size_t> AddTanhModule(std::vector< std::shared_ptr< Module<T> > >& modules, std::string name, const std::vector<size_t>& input_dims)
{
	std::shared_ptr< Module<T> > module( new TanhModule<T>(name) );
	modules.push_back(module);
	return input_dims;
}

template <class T>
std::vector<size_t> AddSoftSignModule(std::vector< std::shared_ptr< Module<T> > >& modules, std::string name, const std::vector<size_t>& input_dims)
{
	std::shared_ptr< Module<T> > module( new SoftSignModule<T>(name) );
	modules.push_back(module);
	return input_dims;
}

template <class T>
std::vector<size_t> AddMaxModule(std::vector< std::shared_ptr< Module<T> > >& modules, std::string name, const std::vector<size_t>& input_dims, 
								 const std::vector<size_t>& kernel_dims, const std::vector<size_t>& kernel_strides)
{
	std::shared_ptr< ParametersInitializer<T> > initializer(new ConstantInitializer<T>(0));
	std::shared_ptr< Regularizer<T> > regularizer(new EmptyRegularizer<T>());
	std::shared_ptr< Module<T> > max_module( new KernelModule<T>(name,1,kernel_dims,kernel_strides, 
		MaxPoolingKernelFactory<T>(), initializer, regularizer) );
	modules.push_back(max_module);
	return max_module->GetPerCaseOutputDims(input_dims);
}

template <class T>
std::vector<size_t> AddLinearMixModule(std::vector< std::shared_ptr< Module<T> > >& modules, std::string name,
									   const std::vector<size_t>& input_dims, size_t num_outputs, double weight_decay)
{
	std::shared_ptr< ParametersInitializer<T> > initializer(new LinearMixInitializer<T>());
	std::shared_ptr< Regularizer<T> > regularizer(new WeightDecayRegularizer<T>(weight_decay));
	std::shared_ptr< Module<T> > lmm_module( new LinearMixModule<T>( name, Tensor<T>::Numel(input_dims),num_outputs,
		initializer, regularizer) );
	modules.push_back(lmm_module);
	return lmm_module->GetPerCaseOutputDims(input_dims);
}

template <class T>
std::shared_ptr< NN<T> > ConstructNeuralNetwork1(const std::vector<size_t>& input_dims)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer1_input_dims = input_dims;

	double weight_decay = 0;
	size_t layer1_num_output_kernels = 20;
	std::vector<size_t> layer1_strides;layer1_strides.push_back(40);layer1_strides.push_back(1);
	std::vector<size_t> layer1_kernel_dims;layer1_kernel_dims.push_back(40);layer1_kernel_dims.push_back(1);
	std::vector<size_t> layer1_output_dims = AddConvModule(modules, "conv1", layer1_input_dims, 
		layer1_num_output_kernels, layer1_kernel_dims, layer1_strides, weight_decay);

	std::vector<size_t> layer2_output_dims = AddBiasModule(modules, "bias1", layer1_output_dims);
	std::vector<size_t> layer3_output_dims = AddAbsModule(modules, "abs1", layer2_output_dims);
	
	size_t layer4_num_output_kernels = 20;
	std::vector<size_t> layer4_strides;layer4_strides.push_back(1);layer4_strides.push_back(1);
	std::vector<size_t> layer4_kernel_dims;layer4_kernel_dims.push_back(3);layer4_kernel_dims.push_back(layer3_output_dims[layer3_output_dims.size()-1]);
	std::vector<size_t> layer4_output_dims = AddConvModule(modules, "conv2", layer3_output_dims, 
		layer4_num_output_kernels, layer4_kernel_dims, layer4_strides, weight_decay);
	
	std::vector<size_t> layer5_output_dims = AddBiasModule(modules, "bias2", layer4_output_dims);
	std::vector<size_t> layer6_output_dims = AddRluModule(modules, "rlu1", layer5_output_dims);

	std::vector<size_t> layer7_strides;layer7_strides.push_back(3);layer7_strides.push_back(1);
	std::vector<size_t> layer7_kernel_dims;layer7_kernel_dims.push_back(3);layer7_kernel_dims.push_back(1);
	std::vector<size_t> layer7_output_dims = AddMaxModule(modules, "max1", layer6_output_dims, layer7_kernel_dims, layer7_strides);

	size_t layer8_num_output_kernels = 20;
	std::vector<size_t> layer8_strides;layer8_strides.push_back(1);layer8_strides.push_back(1);
	std::vector<size_t> layer8_kernel_dims;layer8_kernel_dims.push_back(3);layer8_kernel_dims.push_back(layer7_output_dims[layer7_output_dims.size()-1]);
	std::vector<size_t> layer8_output_dims = AddConvModule(modules, "conv3", layer7_output_dims, 
		layer8_num_output_kernels, layer8_kernel_dims, layer8_strides, weight_decay);

	std::vector<size_t> layer9_output_dims = AddBiasModule(modules, "bias3", layer8_output_dims);
	std::vector<size_t> layer10_output_dims = AddRluModule(modules, "rlu2", layer9_output_dims);
	
	std::vector<size_t> layer11_strides;layer11_strides.push_back(3);layer11_strides.push_back(1);
	std::vector<size_t> layer11_kernel_dims;layer11_kernel_dims.push_back(3);layer11_kernel_dims.push_back(1);
	std::vector<size_t> layer11_output_dims = AddMaxModule(modules, "max2", layer10_output_dims, layer11_kernel_dims, layer11_strides);

	std::vector<size_t> layer12_output_dims = AddLinearMixModule(modules, "linear_mix1", layer11_output_dims, 50, weight_decay);
	std::vector<size_t> layer13_output_dims = AddBiasModule(modules, "bias4", layer12_output_dims);
	std::vector<size_t> layer14_output_dims = AddSoftSignModule(modules, "softsign1", layer13_output_dims);
	std::vector<size_t> layer15_output_dims = AddLinearMixModule(modules, "linear_mix2", layer14_output_dims, 1, weight_decay);
	std::vector<size_t> layer16_output_dims = AddBiasModule(modules, "bias5", layer15_output_dims);
	std::vector<size_t> layer17_output_dims = AddSigmoidModule(modules, "sigmoid1", layer16_output_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
bool CheckNNGradientsSameForDifferentBufferSizes(NN<double>& nn, ITrainDataset<T>& dataset)
{
	size_t initial_buffer_size = nn.GetMinibatchSize();
	size_t num_params = nn.GetNumParams();
	double* parameters = nn.GetParameters();
	nn.SetMinibatchSize(1);
	nn.GetGradientsAndCost(dataset, std::vector<size_t>(), true);
	double* gradients = nn.GetGradients();
	std::vector<double> gradients1(num_params);
	for (size_t i=0; i<num_params; i++)
		gradients1[i] = gradients[i];

	nn.SetMinibatchSize(4);
	nn.GetGradientsAndCost(dataset, std::vector<size_t>(), true);
	std::vector<double> gradients2(num_params);
	for (size_t i=0; i<num_params; i++)
		gradients2[i] = gradients[i];
	for (size_t i=0; i<num_params; i++)
		if( abs(gradients1[i] - gradients2[i]) >0.000000001)
			throw "Different gradient for different buffer sizes";
	nn.SetMinibatchSize(initial_buffer_size);
	return true;
}

template <class T>
bool NumericalCheckNNGradients(NN<double>& nn, ITrainDataset<T>& dataset)
{
	size_t num_params = nn.GetNumParams();
	double* parameters = nn.GetParameters();
	nn.GetGradientsAndCost(dataset, std::vector<size_t>(), true);
	// Ensure that all buffers are cleared
	nn.GetGradientsAndCost(dataset, std::vector<size_t>(), true);
	double* gradients = nn.GetGradients();

	std::vector<double> numerical_gradients;
	double eps = 1e-4;
	// check that each fprop does not affect other fprops
	double c1 = nn.GetCost(dataset, std::vector<size_t>(), true);
	double c2 = nn.GetCost(dataset, std::vector<size_t>(), true);
	if (c1 != c2)
		throw "Wrong gradient";
	for (size_t i=0; i<num_params; i++)
	{
		double initial_val = parameters[i];
		parameters[i]=initial_val-eps;
		double cost1 = nn.GetCost(dataset, std::vector<size_t>(), true, true);
		parameters[i]=initial_val+eps;
		double cost2 = nn.GetCost(dataset, std::vector<size_t>(), true, true);
		double numerical_gradient = (cost2-cost1) / 2 / eps;
		if (abs(numerical_gradient-gradients[i]) / (std::max<double>)(abs(numerical_gradient)+abs(gradients[i]), 1) > 0.000001)
			throw "Wrong gradient";
		parameters[i] = initial_val;
	}
	return CheckNNGradientsSameForDifferentBufferSizes(nn, dataset);
}

// first - train dataset, second - validation dataset
template <class T>
std::pair< std::shared_ptr< ITrainDataset<T> >, std::shared_ptr< ITrainDataset<T> > > LoadData(std::string data_dir, double train_fraction)
{
	std::string train_path = data_dir+"train/";
	std::string labels_path = data_dir + "train.csv";
	std::ifstream labels_stream(labels_path);
	std::string entry_str;
	std::vector<std::string> train_files_paths;
	std::vector< std::shared_ptr< Tensor<float> > > output_labels;
	std::vector<size_t> output_dims; output_dims.push_back(1);
	while (labels_stream>>entry_str)
	{
		std::vector<std::string> entry_fields = split(entry_str, ',');
		train_files_paths.push_back(train_path+entry_fields[0]);
		output_labels.push_back( std::shared_ptr< Tensor<float> >( new Tensor<float>(output_dims) ) );
		(*output_labels[output_labels.size()-1])[0] = (float)std::stoi(entry_fields[1]);
	}

	std::vector< std::shared_ptr< Tensor<float> > > input = ReadAiffFiles(train_files_paths);

	auto means = GetFullMeans(input);
	FullMeanSubtract(input, means);
	auto stds = GetFullStd(input);
	FullStdDivide(input, stds);
	RandomShuffleVectors(input, output_labels);

	size_t num_samples = train_files_paths.size();

	size_t num_train_samples = (size_t)(num_samples * train_fraction);
	std::vector< std::shared_ptr< Tensor<float> > > train_input(num_train_samples);
	std::vector< std::shared_ptr< Tensor<float> > > train_output(num_train_samples);
	std::vector<T> train_importance(num_train_samples, 1);
	for (size_t i=0; i<num_train_samples; i++)
	{
		train_input[i] = input[i];
		train_output[i] = output_labels[i];
	}
	
	std::vector<size_t> max_input_shifts; 
	max_input_shifts.push_back(80);

	std::vector<size_t> left_shifts; 
	left_shifts.push_back(40);
	std::vector<size_t> right_shifts; 
	right_shifts.push_back(40);

	std::shared_ptr< ITensorDataLoader<T> > train_input_loader = 
		std::shared_ptr< ITensorDataLoader<T> >( new RandomShiftPartialTensorDataLoader<T,float>(train_input, max_input_shifts) );
		//std::shared_ptr< ITensorDataLoader<double> >( new FixedShiftPartialTensorDataLoader<double,float>(train_input, left_shifts, right_shifts) );
		//std::shared_ptr< ITensorDataLoader<double> >( new FullTensorDataLoader<double,float>(train_input) );
	std::shared_ptr< ITensorDataLoader<T> > train_output_loader = 
		std::shared_ptr< ITensorDataLoader<T> >( new FullTensorDataLoader<T,float>(train_output) );
	std::shared_ptr< ITrainDataset<T> > train_set(new TrainDataset<T>(train_input_loader, train_output_loader, train_importance));

	std::vector< std::shared_ptr< Tensor<float> > > validation_input(num_samples-num_train_samples);
	std::vector< std::shared_ptr< Tensor<float> > > validation_output(num_samples-num_train_samples);
	std::vector<T> validation_importance(num_samples-num_train_samples, 1);
	for (size_t i=0; i<num_samples-num_train_samples; i++)
	{
		validation_input[i] = input[num_train_samples+i];
		validation_output[i] = output_labels[num_train_samples+i];
	}
	std::shared_ptr< ITensorDataLoader<T> > validation_input_loader = 
		std::shared_ptr< ITensorDataLoader<T> >( new RandomShiftPartialTensorDataLoader<T,float>(validation_input, max_input_shifts) );
	std::shared_ptr< ITensorDataLoader<T> > validation_output_loader = 
		std::shared_ptr< ITensorDataLoader<T> >( new FullTensorDataLoader<T,float>(validation_output) );
	std::shared_ptr< ITrainDataset<T> > validation_set(new TrainDataset<T>(validation_input_loader, validation_output_loader, validation_importance));

	return std::make_pair(train_set, validation_set);
}

template <class DataType>
std::shared_ptr< Tensor<DataType> > GetRandomTensorPtr(std::vector<size_t> tensor_dims, double min_val=-1, double max_val=1)
{
	Tensor<DataType> tensor(tensor_dims);
	for (size_t i=0; i<tensor.Numel(); i++)
		tensor[i] = (DataType)RandomGenerator::GetUniformDouble(min_val, max_val);
	return std::shared_ptr< Tensor<DataType> >( new Tensor<DataType>(tensor) );
}

// first - train dataset, second - validation dataset
template <class T>
std::pair< std::shared_ptr< ITrainDataset<T> >, std::shared_ptr< ITrainDataset<T> > > GetDummyData()
{
	size_t num_train_samples = 18;
	size_t num_validation_samples = 7;

	std::vector<size_t> input_dims(1,4000);
	std::vector<size_t> output_dims(1,1);
	std::vector< std::shared_ptr< Tensor<T> > > train_input(num_train_samples);
	std::vector< std::shared_ptr< Tensor<T> > > train_output(num_train_samples);
	std::vector<T> train_importance(num_train_samples, 1);
	for (size_t i=0; i< num_train_samples; i++)
	{
		train_input[i] = GetRandomTensorPtr<T>(input_dims);
		Tensor<T> output_tensor(output_dims);
		output_tensor[0] = RandomGenerator::GetUniformInt(0, 1);
		train_output[i] = std::shared_ptr< Tensor<T> >( new Tensor<T>(output_tensor) );
	}

	std::vector< std::shared_ptr< Tensor<T> > > validation_input(num_validation_samples);
	std::vector< std::shared_ptr< Tensor<T> > > validation_output(num_validation_samples);
	std::vector<T> validation_importance(num_validation_samples, 1);
	for (size_t i=0; i< num_validation_samples; i++)
	{
		validation_input[i] = GetRandomTensorPtr<T>(input_dims);
		Tensor<T> output_tensor(output_dims);
		output_tensor[0] = RandomGenerator::GetUniformInt(0, 1);
		validation_output[i] = std::shared_ptr< Tensor<T> >( new Tensor<T>(output_tensor) );
	}

	std::shared_ptr< ITensorDataLoader<T> > train_input_loader = 
		std::shared_ptr< ITensorDataLoader<T> >( new FullTensorDataLoader<T,T>(train_input) );
	std::shared_ptr< ITensorDataLoader<T> > train_output_loader = 
		std::shared_ptr< ITensorDataLoader<T> >( new FullTensorDataLoader<T,T>(train_output) );
	std::shared_ptr< ITrainDataset<T> > train_set(new TrainDataset<T>(train_input_loader, train_output_loader, train_importance));

	std::shared_ptr< ITensorDataLoader<T> > validation_input_loader = 
		std::shared_ptr< ITensorDataLoader<T> >( new FullTensorDataLoader<T,T>(validation_input) );
	std::shared_ptr< ITensorDataLoader<T> > validation_output_loader = 
		std::shared_ptr< ITensorDataLoader<T> >( new FullTensorDataLoader<T,T>(validation_output) );
	std::shared_ptr< ITrainDataset<T> > validation_set(new TrainDataset<T>(validation_input_loader, validation_output_loader, validation_importance));

	return std::make_pair(train_set, validation_set);
}

template <class T>
void test_net(NN<T>& net, ITrainDataset<T>& dataset, size_t num_samples)
{
	std::vector<size_t> inds;
	std::vector<T> gradients1(net.GetNumParams());
	double cost1 = 0;
	for (size_t i=0; i<num_samples; i++)
	{
		inds.clear();
		inds.push_back(i);
		CostAndGradients<T> res = net.GetGradientsAndCost(dataset, inds, true);
		for (size_t j=0; j<net.GetNumParams(); j++)
			gradients1[j] += res.gradients[j] * 1/num_samples;
		cost1 += res.cost/num_samples;
	}

	inds.clear();
	for (size_t i=0; i<num_samples; i++)
		inds.push_back(i);

	CostAndGradients<T> res = net.GetGradientsAndCost(dataset, inds, true);
	if ( abs( res.cost - cost1) >0.00000001)
		throw 1;
	
	for (size_t i=0; i<net.GetNumParams(); i++)
		if ( abs( gradients1[i] - res.gradients[i]) >0.00000001)
			throw 1;
}

void whale_detection_main()
{
	typedef float ParamsType;
	std::pair< std::shared_ptr< ITrainDataset<ParamsType> >, std::shared_ptr< ITrainDataset<ParamsType> > > data = 
		LoadData<ParamsType>("C:/Users/Pavel/Desktop/whale_data/data/", 0.75);
		//GetDummyData<ParamsType>();
	std::vector<size_t> input_dims; input_dims.push_back(3920);
	std::shared_ptr< NN<ParamsType> > nn = ConstructNeuralNetwork1<ParamsType>(input_dims);
	//test_net<ParamsType>(*nn, *data.first, 2);

	size_t num_iterations = 1000000; 
	double learning_rate = 0.0005;
	double momentum = 0.9;
	size_t train_batch_size = 1;
	size_t validation_batch_size = 1000;
	double train_decay = 0.999;
	double validation_decay = 0;
	size_t num_batches_before_validation_evaluation = 25000;
	size_t num_batches_before_train_evaluation = 1000;
	size_t num_warmup_batches = 5000;
	double warmup_momentum = 0.5;

	SGD_Trainer<ParamsType> trainer(num_iterations, learning_rate, momentum, train_batch_size,
		validation_batch_size, train_decay, validation_decay, num_batches_before_train_evaluation, 
		num_batches_before_validation_evaluation, num_warmup_batches, warmup_momentum);
	//NumericalCheckNNGradients(*nn, *data.first);
	LogisticCostModule<ParamsType> cost_module;
	trainer.Train(*nn, cost_module, cost_module, *data.first, *data.second);
}

#endif