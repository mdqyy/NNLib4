#ifndef WHHALES_DETECTION_H
#define WHALES_DETECTION_H

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <istream>
#include "GaussianInitializer.h"
#include "ConstantInitializer.h"
#include "LinearMixInitializer.h"
#include "RandomShiftPartialTensorDataLoader.h"
#include "FixedShiftPartialTensorDataLoader.h"
#include "FullTensorDataLoader.h"
#include "SGD_Trainer.h"
#include "LbfgsMinibatchTrainer.h"
#include "CompositeModule.h"
#include "MisclassificationRateCostModule.h"
#include "CrossEntropyCostModule.h"
#include "LogisticCostModule.h"
#include "MseCostModule.h"
#include "RectifiedLinearUnitModule.h"
#include "AbsModule.h"
#include "BiasModule.h"
#include "LinearMixModule.h"
#include "SoftSignModule.h"
#include "SigmoidModule.h"
#include "SoftmaxModule.h"
#include "TanhModule.h"
#include "KernelModule.h"
#include "WeightDecayRegularizer.h"
#include "AbsRegularizer.h"
#include "EmptyRegularizer.h"
#include "Utilities.h"
#include "TrainDataset.h"
#include  "ITensorDataLoader.h"
#include "Preprocessing.h"
#include "Converter.h"
#include "IOTreeNode.h"
#include "IOXML.h"
#include "DropoutModule.h"
#include "MeanStdNormalizingModule.h"
#include "GaussianNoiseModule.h"
#include "EntropyRegularizingModule.h"
#include "ClassificationBalancedPairsTensorDataLoader.h"
#include "ClassificationPairsOutputTensorDataLoader.h"
#include "SemisupervisedCostModule.h"
#include "SemisupervisedTensorDataLoader.h"
#include "EntropyCostModule.h"

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
std::vector<size_t> AddGaussianNoiseModule(std::vector< std::shared_ptr< Module<T> > >& modules, std::string name, 
										   const std::vector<size_t>& input_dims, double gaussian_noise_std)
{
	std::shared_ptr< Module<T> > module( new GaussianNoiseModule<T>(name, gaussian_noise_std) );
	modules.push_back(module);
	return input_dims;
}

template <class T>
std::vector<size_t> AddSoftmaxModule(std::vector< std::shared_ptr< Module<T> > >& modules, std::string name, const std::vector<size_t>& input_dims)
{
	std::shared_ptr< Module<T> > module( new SoftmaxModule<T>(name) );
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
std::vector<size_t> AddDropoutModule(std::vector< std::shared_ptr< Module<T> > >& modules, 
									 std::string name, const std::vector<size_t>& input_dims, double dropout_probability)
{
	std::shared_ptr< Module<T> > module( new DropoutModule<T>(name, dropout_probability) );
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
std::vector<size_t> AddMeanStdNormalizingModule(std::vector< std::shared_ptr< Module<T> > >& modules, std::string name, const std::vector<size_t>& input_dims)
{
	std::shared_ptr< Module<T> > module( new MeanStdNormalizingModule<T>(name, Tensor<T>::Numel(input_dims), 5) );
	modules.push_back(module);
	return input_dims;
}

template <class T>
std::vector<size_t> AddEntropyRegularizingModule(std::vector< std::shared_ptr< Module<T> > >& modules, 
												 std::string name, const std::vector<size_t>& input_dims, 
												 double lambda, size_t num_groups)
{
	std::shared_ptr< Module<T> > module( new EntropyRegularizingModule<T>(name, num_groups, lambda) );
	modules.push_back(module);
	return input_dims;
}

template <class T>
std::shared_ptr< NN<T> > ConstructTanhTanhSoftmaxNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t l2_size, size_t num_outputs, double weight_decay)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer1_input_dims = input_dims;
	std::vector<size_t> layer2_input_dims = AddLinearMixModule(modules, "linear_mix1", layer1_input_dims, l1_size, weight_decay);
	std::vector<size_t> layer3_input_dims = AddBiasModule(modules, "bias1", layer2_input_dims);
	std::vector<size_t> layer4_input_dims = AddTanhModule(modules, "tanh1", layer3_input_dims);
	std::vector<size_t> layer5_input_dims = AddLinearMixModule(modules, "linear_mix2", layer4_input_dims, l2_size, weight_decay);
	std::vector<size_t> layer6_input_dims = AddBiasModule(modules, "bias2", layer5_input_dims);
	std::vector<size_t> layer7_input_dims = AddTanhModule(modules, "tanh2", layer6_input_dims);
	std::vector<size_t> layer8_input_dims = AddLinearMixModule(modules, "linear_mix3", layer7_input_dims, num_outputs, weight_decay);
	std::vector<size_t> layer9_input_dims = AddBiasModule(modules, "bias3", layer8_input_dims);
	std::vector<size_t> layer10_input_dims = AddSoftmaxModule(modules, "softmax1", layer9_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructTanhTanhSoftmaxNormalizedNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t l2_size, size_t num_outputs, double weight_decay)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer1_input_dims = input_dims;
	std::vector<size_t> layer2_input_dims = AddLinearMixModule(modules, "linear_mix1", layer1_input_dims, l1_size, weight_decay);
	std::vector<size_t> layer3_input_dims = AddBiasModule(modules, "bias1", layer2_input_dims);
	std::vector<size_t> layer4_input_dims = AddTanhModule(modules, "tanh1", layer3_input_dims);
	std::vector<size_t> layer5_input_dims = AddMeanStdNormalizingModule(modules, "normalizer1", layer4_input_dims);
	std::vector<size_t> layer6_input_dims = AddLinearMixModule(modules, "linear_mix2", layer5_input_dims, l2_size, weight_decay);
	std::vector<size_t> layer7_input_dims = AddBiasModule(modules, "bias2", layer6_input_dims);
	std::vector<size_t> layer8_input_dims = AddTanhModule(modules, "tanh2", layer7_input_dims);
	std::vector<size_t> layer9_input_dims = AddMeanStdNormalizingModule(modules, "normalizer2", layer8_input_dims);
	std::vector<size_t> layer10_input_dims = AddLinearMixModule(modules, "linear_mix3", layer9_input_dims, num_outputs, weight_decay);
	std::vector<size_t> layer11_input_dims = AddBiasModule(modules, "bias3", layer10_input_dims);
	std::vector<size_t> layer12_input_dims = AddSoftmaxModule(modules, "softmax1", layer11_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructTanhRluTanhRluSoftmaxNormalizedNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t l2_size, size_t num_outputs, double weight_decay)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer1_input_dims = input_dims;
	std::vector<size_t> layer2_input_dims = AddLinearMixModule(modules, "linear_mix1", layer1_input_dims, l1_size, weight_decay);
	std::vector<size_t> layer3_input_dims = AddBiasModule(modules, "bias1", layer2_input_dims);
	std::vector<size_t> layer4_input_dims = AddTanhModule(modules, "tanh1", layer3_input_dims);
	std::vector<size_t> layer5_input_dims = AddRluModule(modules, "rlu1", layer4_input_dims);
	std::vector<size_t> layer6_input_dims = AddMeanStdNormalizingModule(modules, "normalizer1", layer5_input_dims);
	std::vector<size_t> layer7_input_dims = AddLinearMixModule(modules, "linear_mix2", layer6_input_dims, l2_size, weight_decay);
	std::vector<size_t> layer8_input_dims = AddBiasModule(modules, "bias2", layer7_input_dims);
	std::vector<size_t> layer9_input_dims = AddTanhModule(modules, "tanh2", layer8_input_dims);
	std::vector<size_t> layer10_input_dims = AddRluModule(modules, "rlu2", layer9_input_dims);
	std::vector<size_t> layer11_input_dims = AddMeanStdNormalizingModule(modules, "normalizer2", layer10_input_dims);
	std::vector<size_t> layer12_input_dims = AddLinearMixModule(modules, "linear_mix3", layer11_input_dims, num_outputs, weight_decay);
	std::vector<size_t> layer13_input_dims = AddBiasModule(modules, "bias3", layer12_input_dims);
	std::vector<size_t> layer14_input_dims = AddSoftmaxModule(modules, "softmax1", layer13_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructTanhRluTanhRluSoftmaxNormalizeDropoutdNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t l2_size, size_t num_outputs, double weight_decay, double dropout_probability)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer1_input_dims = input_dims;
	std::vector<size_t> layer2_input_dims = AddLinearMixModule(modules, "linear_mix1", layer1_input_dims, l1_size, weight_decay);
	std::vector<size_t> layer3_input_dims = AddBiasModule(modules, "bias1", layer2_input_dims);
	std::vector<size_t> layer4_input_dims = AddTanhModule(modules, "tanh1", layer3_input_dims);
	std::vector<size_t> layer5_input_dims = AddRluModule(modules, "rlu1", layer4_input_dims);
	std::vector<size_t> layer6_input_dims = AddDropoutModule(modules, "dropout1", layer5_input_dims, dropout_probability);
	std::vector<size_t> layer7_input_dims = AddMeanStdNormalizingModule(modules, "normalizer1", layer6_input_dims);
	std::vector<size_t> layer8_input_dims = AddLinearMixModule(modules, "linear_mix2", layer7_input_dims, l2_size, weight_decay);
	std::vector<size_t> layer9_input_dims = AddBiasModule(modules, "bias2", layer8_input_dims);
	std::vector<size_t> layer10_input_dims = AddTanhModule(modules, "tanh2", layer9_input_dims);
	std::vector<size_t> layer11_input_dims = AddRluModule(modules, "rlu2", layer10_input_dims);
	std::vector<size_t> layer12_input_dims = AddDropoutModule(modules, "dropout2", layer11_input_dims, dropout_probability);
	std::vector<size_t> layer13_input_dims = AddMeanStdNormalizingModule(modules, "normalizer2", layer12_input_dims);
	std::vector<size_t> layer14_input_dims = AddLinearMixModule(modules, "linear_mix3", layer13_input_dims, num_outputs, weight_decay);
	std::vector<size_t> layer15_input_dims = AddBiasModule(modules, "bias3", layer14_input_dims);
	std::vector<size_t> layer16_input_dims = AddSoftmaxModule(modules, "softmax1", layer15_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructTanhTanhSoftmaxDropoutNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t l2_size, size_t num_outputs, 
													double weight_decay, double dropout_probability)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer1_input_dims = input_dims;
	//std::vector<size_t> layer2_input_dims = AddDropoutModule(modules, "dropout1", layer1_input_dims, dropout_probability);
	std::vector<size_t> layer3_input_dims = AddLinearMixModule(modules, "linear_mix1", layer1_input_dims, l1_size, weight_decay);
	std::vector<size_t> layer4_input_dims = AddBiasModule(modules, "bias1", layer3_input_dims);
	std::vector<size_t> layer5_input_dims = AddTanhModule(modules, "tanh1", layer4_input_dims);
	std::vector<size_t> layer6_input_dims = AddDropoutModule(modules, "dropout2", layer5_input_dims, dropout_probability);
	std::vector<size_t> layer7_input_dims = AddLinearMixModule(modules, "linear_mix2", layer6_input_dims, l2_size, weight_decay);
	std::vector<size_t> layer8_input_dims = AddBiasModule(modules, "bias2", layer7_input_dims);
	std::vector<size_t> layer9_input_dims = AddTanhModule(modules, "tanh2", layer8_input_dims);
	std::vector<size_t> layer10_input_dims = AddDropoutModule(modules, "dropout3", layer9_input_dims, dropout_probability);
	std::vector<size_t> layer11_input_dims = AddLinearMixModule(modules, "linear_mix3", layer10_input_dims, num_outputs, weight_decay);
	std::vector<size_t> layer12_input_dims = AddBiasModule(modules, "bias3", layer11_input_dims);
	std::vector<size_t> layer13_input_dims = AddSoftmaxModule(modules, "softmax1", layer12_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructRluRluSoftmaxNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t l2_size, size_t num_outputs, 
													double weight_decay)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer1_input_dims = input_dims;
	std::vector<size_t> layer2_input_dims = AddLinearMixModule(modules, "linear_mix1", layer1_input_dims, l1_size, weight_decay);
	std::vector<size_t> layer3_input_dims = AddBiasModule(modules, "bias1", layer2_input_dims);
	std::vector<size_t> layer4_input_dims = AddRluModule(modules, "rlu1", layer3_input_dims);
	std::vector<size_t> layer5_input_dims = AddLinearMixModule(modules, "linear_mix2", layer4_input_dims, l2_size, weight_decay);
	std::vector<size_t> layer6_input_dims = AddBiasModule(modules, "bias2", layer5_input_dims);
	std::vector<size_t> layer7_input_dims = AddRluModule(modules, "rlu2", layer6_input_dims);
	std::vector<size_t> layer8_input_dims = AddLinearMixModule(modules, "linear_mix3", layer7_input_dims, num_outputs, weight_decay);
	std::vector<size_t> layer9_input_dims = AddBiasModule(modules, "bias3", layer8_input_dims);
	std::vector<size_t> layer10_input_dims = AddSoftmaxModule(modules, "softmax1", layer9_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructRluSoftmaxNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t num_outputs, 
													double weight_decay)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer_input_dims = input_dims;
	layer_input_dims = AddLinearMixModule(modules, "linear_mix1", layer_input_dims, l1_size, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias1", layer_input_dims);
	layer_input_dims = AddRluModule(modules, "rlu1", layer_input_dims);
	layer_input_dims = AddLinearMixModule(modules, "linear_mix2", layer_input_dims, num_outputs, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias2", layer_input_dims);
	layer_input_dims = AddSoftmaxModule(modules, "softmax1", layer_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructRluRluSoftmaxNormalizedNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t l2_size, size_t num_outputs, double weight_decay)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer1_input_dims = input_dims;
	std::vector<size_t> layer2_input_dims = AddLinearMixModule(modules, "linear_mix1", layer1_input_dims, l1_size, weight_decay);
	std::vector<size_t> layer3_input_dims = AddBiasModule(modules, "bias1", layer2_input_dims);
	std::vector<size_t> layer4_input_dims = AddRluModule(modules, "rlu1", layer3_input_dims);
	std::vector<size_t> layer5_input_dims = AddMeanStdNormalizingModule(modules, "normalizer1", layer4_input_dims);
	std::vector<size_t> layer6_input_dims = AddLinearMixModule(modules, "linear_mix2", layer5_input_dims, l2_size, weight_decay);
	std::vector<size_t> layer7_input_dims = AddBiasModule(modules, "bias2", layer6_input_dims);
	std::vector<size_t> layer8_input_dims = AddRluModule(modules, "rlu2", layer7_input_dims);
	std::vector<size_t> layer9_input_dims = AddMeanStdNormalizingModule(modules, "normalizer2", layer8_input_dims);
	std::vector<size_t> layer10_input_dims = AddLinearMixModule(modules, "linear_mix3", layer9_input_dims, num_outputs, weight_decay);
	std::vector<size_t> layer11_input_dims = AddBiasModule(modules, "bias3", layer10_input_dims);
	std::vector<size_t> layer12_input_dims = AddSoftmaxModule(modules, "softmax1", layer11_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructRluRluRluSoftmaxNormalizedNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t l2_size, size_t l3_size, size_t num_outputs, double weight_decay)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer1_input_dims = input_dims;
	std::vector<size_t> layer2_input_dims = AddLinearMixModule(modules, "linear_mix1", layer1_input_dims, l1_size, weight_decay);
	std::vector<size_t> layer3_input_dims = AddBiasModule(modules, "bias1", layer2_input_dims);
	std::vector<size_t> layer4_input_dims = AddRluModule(modules, "rlu1", layer3_input_dims);
	std::vector<size_t> layer5_input_dims = AddMeanStdNormalizingModule(modules, "normalizer1", layer4_input_dims);
	std::vector<size_t> layer6_input_dims = AddLinearMixModule(modules, "linear_mix2", layer5_input_dims, l2_size, weight_decay);
	std::vector<size_t> layer7_input_dims = AddBiasModule(modules, "bias2", layer6_input_dims);
	std::vector<size_t> layer8_input_dims = AddRluModule(modules, "rlu2", layer7_input_dims);
	std::vector<size_t> layer9_input_dims = AddMeanStdNormalizingModule(modules, "normalizer2", layer8_input_dims);
	std::vector<size_t> layer10_input_dims = AddLinearMixModule(modules, "linear_mix3", layer9_input_dims, l3_size, weight_decay);
	std::vector<size_t> layer11_input_dims = AddBiasModule(modules, "bias3", layer10_input_dims);
	std::vector<size_t> layer12_input_dims = AddMeanStdNormalizingModule(modules, "normalizer3", layer11_input_dims);
	std::vector<size_t> layer13_input_dims = AddLinearMixModule(modules, "linear_mix4", layer12_input_dims, num_outputs, weight_decay);
	std::vector<size_t> layer14_input_dims = AddBiasModule(modules, "bias4", layer13_input_dims);
	std::vector<size_t> layer15_input_dims = AddSoftmaxModule(modules, "softmax1", layer14_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructRluRluSoftmaxNormalizedDropoutNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t l2_size, size_t num_outputs, 
													double weight_decay, double dropout_probability)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer1_input_dims = input_dims;
	std::vector<size_t> layer2_input_dims = AddLinearMixModule(modules, "linear_mix1", layer1_input_dims, l1_size, weight_decay);
	std::vector<size_t> layer3_input_dims = AddBiasModule(modules, "bias1", layer2_input_dims);
	std::vector<size_t> layer4_input_dims = AddRluModule(modules, "rlu1", layer3_input_dims);
	std::vector<size_t> layer5_input_dims = AddDropoutModule(modules, "dropout2", layer4_input_dims, dropout_probability);
	std::vector<size_t> layer6_input_dims = AddMeanStdNormalizingModule(modules, "normalizer1", layer5_input_dims);
	std::vector<size_t> layer7_input_dims = AddLinearMixModule(modules, "linear_mix2", layer6_input_dims, l2_size, weight_decay);
	std::vector<size_t> layer8_input_dims = AddBiasModule(modules, "bias2", layer7_input_dims);
	std::vector<size_t> layer9_input_dims = AddRluModule(modules, "rlu2", layer8_input_dims);
	std::vector<size_t> layer10_input_dims = AddDropoutModule(modules, "dropout3", layer9_input_dims, dropout_probability);
	std::vector<size_t> layer11_input_dims = AddMeanStdNormalizingModule(modules, "normalizer2", layer10_input_dims);
	std::vector<size_t> layer12_input_dims = AddLinearMixModule(modules, "linear_mix3", layer11_input_dims, num_outputs, weight_decay);
	std::vector<size_t> layer13_input_dims = AddBiasModule(modules, "bias3", layer12_input_dims);
	std::vector<size_t> layer14_input_dims = AddSoftmaxModule(modules, "softmax1", layer13_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructRluLinearNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t num_outputs, 
													double weight_decay)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer_input_dims = input_dims;
	layer_input_dims = AddLinearMixModule(modules, "linear_mix1", layer_input_dims, l1_size, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias1", layer_input_dims);
	layer_input_dims = AddRluModule(modules, "rlu1", layer_input_dims);
	layer_input_dims = AddLinearMixModule(modules, "linear_mix2", layer_input_dims, num_outputs, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias2", layer_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructRluEntropyLinearNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t num_outputs, 
													double weight_decay, double entropy_lambda, size_t entropy_num_groups)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer_input_dims = input_dims;
	layer_input_dims = AddLinearMixModule(modules, "linear_mix1", layer_input_dims, l1_size, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias1", layer_input_dims);
	layer_input_dims = AddRluModule(modules, "rlu1", layer_input_dims);
	layer_input_dims = AddEntropyRegularizingModule(modules, "entropy1", layer_input_dims, entropy_lambda, entropy_num_groups);
	layer_input_dims = AddLinearMixModule(modules, "linear_mix2", layer_input_dims, num_outputs, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias2", layer_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructSigmoidEntropyLinearNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t num_outputs, 
													double weight_decay, double entropy_lambda, size_t entropy_num_groups)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer_input_dims = input_dims;
	layer_input_dims = AddLinearMixModule(modules, "linear_mix1", layer_input_dims, l1_size, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias1", layer_input_dims);
	layer_input_dims = AddSigmoidModule(modules, "sigmoid1", layer_input_dims);
	layer_input_dims = AddEntropyRegularizingModule(modules, "entropy1", layer_input_dims, entropy_lambda, entropy_num_groups);
	layer_input_dims = AddLinearMixModule(modules, "linear_mix2", layer_input_dims, num_outputs, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias2", layer_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructSigmoidLinearNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t num_outputs, 
													double weight_decay)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer_input_dims = input_dims;
	layer_input_dims = AddLinearMixModule(modules, "linear_mix1", layer_input_dims, l1_size, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias1", layer_input_dims);
	layer_input_dims = AddSigmoidModule(modules, "sigmoid1", layer_input_dims);
	layer_input_dims = AddLinearMixModule(modules, "linear_mix2", layer_input_dims, num_outputs, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias2", layer_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructTanhLinearNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t num_outputs, 
													double weight_decay)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer_input_dims = input_dims;
	layer_input_dims = AddLinearMixModule(modules, "linear_mix1", layer_input_dims, l1_size, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias1", layer_input_dims);
	layer_input_dims = AddTanhModule(modules, "tanh1", layer_input_dims);
	layer_input_dims = AddLinearMixModule(modules, "linear_mix2", layer_input_dims, num_outputs, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias2", layer_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructRluRluRluLinearNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t l2_size, size_t l3_size, size_t num_outputs, 
													double weight_decay)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer_input_dims = input_dims;
	layer_input_dims = AddLinearMixModule(modules, "linear_mix1", layer_input_dims, l1_size, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias1", layer_input_dims);
	layer_input_dims = AddRluModule(modules, "rlu1", layer_input_dims);
	layer_input_dims = AddLinearMixModule(modules, "linear_mix2", layer_input_dims, l2_size, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias2", layer_input_dims);
	layer_input_dims = AddRluModule(modules, "rlu2", layer_input_dims);
	layer_input_dims = AddLinearMixModule(modules, "linear_mix3", layer_input_dims, l3_size, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias3", layer_input_dims);
	layer_input_dims = AddRluModule(modules, "rlu3", layer_input_dims);
	layer_input_dims = AddLinearMixModule(modules, "linear_mix4", layer_input_dims, num_outputs, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias4", layer_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructRluRluSoftmaxDropoutNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t l2_size, size_t num_outputs, 
													double weight_decay, double dropout_probability)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer1_input_dims = input_dims;
	std::vector<size_t> layer3_input_dims = AddLinearMixModule(modules, "linear_mix1", layer1_input_dims, l1_size, weight_decay);
	std::vector<size_t> layer4_input_dims = AddBiasModule(modules, "bias1", layer3_input_dims);
	std::vector<size_t> layer5_input_dims = AddRluModule(modules, "rlu1", layer4_input_dims);
	std::vector<size_t> layer6_input_dims = AddDropoutModule(modules, "dropout2", layer5_input_dims, dropout_probability);
	std::vector<size_t> layer7_input_dims = AddLinearMixModule(modules, "linear_mix2", layer6_input_dims, l2_size, weight_decay);
	std::vector<size_t> layer8_input_dims = AddBiasModule(modules, "bias2", layer7_input_dims);
	std::vector<size_t> layer9_input_dims = AddRluModule(modules, "rlu2", layer8_input_dims);
	std::vector<size_t> layer10_input_dims = AddDropoutModule(modules, "dropout3", layer9_input_dims, dropout_probability);
	std::vector<size_t> layer11_input_dims = AddLinearMixModule(modules, "linear_mix3", layer10_input_dims, num_outputs, weight_decay);
	std::vector<size_t> layer12_input_dims = AddBiasModule(modules, "bias3", layer11_input_dims);
	std::vector<size_t> layer13_input_dims = AddSoftmaxModule(modules, "softmax1", layer12_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructRluRluRluSoftmaxDropoutNN(const std::vector<size_t>& input_dims, 
													size_t l1_size, size_t l2_size, size_t l3_size, size_t num_outputs, 
													double weight_decay, double dropout_probability)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer_input_dims = input_dims;
	layer_input_dims = AddLinearMixModule(modules, "linear_mix1", layer_input_dims, l1_size, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias1", layer_input_dims);
	layer_input_dims = AddRluModule(modules, "rlu1", layer_input_dims);
	layer_input_dims = AddDropoutModule(modules, "dropout2", layer_input_dims, dropout_probability);
	layer_input_dims = AddLinearMixModule(modules, "linear_mix2", layer_input_dims, l2_size, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias2", layer_input_dims);
	layer_input_dims = AddRluModule(modules, "rlu2", layer_input_dims);
	layer_input_dims = AddDropoutModule(modules, "dropout3", layer_input_dims, dropout_probability);
	layer_input_dims = AddLinearMixModule(modules, "linear_mix3", layer_input_dims, l3_size, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias3", layer_input_dims);
	layer_input_dims = AddRluModule(modules, "rlu3", layer_input_dims);
	layer_input_dims = AddDropoutModule(modules, "dropout4", layer_input_dims, dropout_probability);
	layer_input_dims = AddLinearMixModule(modules, "linear_mix4", layer_input_dims, num_outputs, weight_decay);
	layer_input_dims = AddBiasModule(modules, "bias4", layer_input_dims);
	layer_input_dims = AddSoftmaxModule(modules, "softmax1", layer_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class T>
std::shared_ptr< NN<T> > ConstructSoftmaxNN(const std::vector<size_t>& input_dims, size_t num_outputs, double weight_decay)
{
	std::vector< std::shared_ptr< Module<T> > > modules;
	std::vector<size_t> layer1_input_dims = input_dims;
	std::vector<size_t> layer2_input_dims = AddLinearMixModule(modules, "linear_mix1", layer1_input_dims, num_outputs, weight_decay);
	std::vector<size_t> layer3_input_dims = AddBiasModule(modules, "bias1", layer2_input_dims);
	std::vector<size_t> layer4_input_dims = AddSoftmaxModule(modules, "softmax1", layer3_input_dims);

	std::shared_ptr< CompositeModule<T> > composite_module( new CompositeModule<T>("composite1", modules) );
	std::shared_ptr< NN<T> > nn( new NN<T>(composite_module, 1000) );
	nn->InitializeParameters();

	return nn;
}

template <class ParamsType, class FeaturesType>
struct SemisupervisedDataset
{
	Tensor<FeaturesType> data_means;
	Tensor<FeaturesType> data_stds;

	std::vector< std::shared_ptr< Tensor<FeaturesType> > > labeled_input;
	std::vector< std::shared_ptr< Tensor<FeaturesType> > > labels;
	std::vector<ParamsType> labeled_importance;
	
	std::vector< std::shared_ptr< Tensor<FeaturesType> > > unlabeled_input;
	std::vector<ParamsType> unlabeled_importance;
};

template <class ParamsType, class FeaturesType>
SemisupervisedDataset<ParamsType, FeaturesType> LoadTrainDataFromFile(std::string labeled_path, 
																	  bool normalize, bool load_unlabeled = false, std::string unlabeled_path = "")
{
	const size_t num_output_clases = 9;
	std::ifstream labeled_stream(labeled_path);
	SemisupervisedDataset<ParamsType, FeaturesType> res;

	std::vector<size_t> labels_dims(1,num_output_clases);
	
	std::string entry_str;
	labeled_stream>>entry_str; // skip header
	std::vector<FeaturesType> features = Converter::StringToVector<FeaturesType>(entry_str, ',');
	std::vector<size_t> input_dims;
	while (labeled_stream>>entry_str)
	{
		std::vector<FeaturesType> features = Converter::StringToVector<FeaturesType>(entry_str, ',');
		if (input_dims.size() == 0)
			input_dims = std::vector<size_t>( 1, features.size()-1);
		if (features.size() != input_dims[0]+1)
			throw 1;
		std::shared_ptr< Tensor<FeaturesType> > features_tensor_ptr( new Tensor<FeaturesType>(input_dims) );
		for (size_t i = 1; i < features.size(); i++)
			(*features_tensor_ptr)[i-1] = features[i];
		res.labeled_input.push_back(features_tensor_ptr);

		std::shared_ptr< Tensor<FeaturesType> > labels_tensor_ptr( new Tensor<FeaturesType>(labels_dims) );
		labels_tensor_ptr->SetZeros();
		(*labels_tensor_ptr)[ static_cast<size_t>(features[0]) - 1 ] = 1;
		res.labels.push_back(labels_tensor_ptr);
	}

	RandomShuffleVectors( res.labeled_input, res.labels );

	res.labeled_importance = std::vector<ParamsType>(res.labels.size(), 1);

	if ( load_unlabeled )
	{
		std::ifstream unlabeled_stream(unlabeled_path);
		unlabeled_stream>>entry_str; // skip header
		while (unlabeled_stream>>entry_str)
		{
			std::vector<FeaturesType> features = Converter::StringToVector<FeaturesType>(entry_str, ',');
			if (features.size() != input_dims[0])
				throw 1;
			std::shared_ptr< Tensor<FeaturesType> > features_tensor_ptr( new Tensor<FeaturesType>(input_dims) );
			for (size_t i = 0; i < features.size(); i++)
				(*features_tensor_ptr)[i] = features[i];
			res.unlabeled_input.push_back(features_tensor_ptr);
		}

		//RandomShuffleVector( res.unlabeled_input );
		
		res.unlabeled_importance = std::vector<ParamsType>(res.unlabeled_input.size(), 1);
	}

	res.data_means = Tensor<FeaturesType>( res.labeled_input[0]->GetDimensions() );
	res.data_means.SetZeros();
	res.data_stds = Tensor<FeaturesType>( res.labeled_input[0]->GetDimensions() );
	for (size_t i = 0; i< res.data_stds.Numel(); i++)
		res.data_stds[i] = 1;

	if (normalize)
	{
		res.data_means = GetFullMeans(res.labeled_input);
		FullMeanSubtract(res.labeled_input, res.data_means);
		res.data_stds = GetFullStd(res.labeled_input);
		FullStdDivide(res.labeled_input, res.data_stds);

		if (load_unlabeled)
		{
			FullMeanSubtract(res.unlabeled_input, res.data_means);
			FullStdDivide(res.unlabeled_input, res.data_stds);
		}
	}

	return res;
}

template <class FeaturesType>
std::vector< std::shared_ptr< Tensor<FeaturesType> > > LoadTestDataFromFile(std::string& filepath, 
																			std::vector<FeaturesType>& means, std::vector<FeaturesType>& stds)
{
	std::ifstream stream(filepath);
	std::vector< std::shared_ptr< Tensor<FeaturesType> > > res;

	stream>>entry_str; // skip header
	while (stream>>entry_str)
	{
		std::vector<FeaturesType> features = Converter::StringToVector<FeaturesType>(entry_str, ',');
		std::vector<size_t> input_dims( 1, features.size());
		std::shared_ptr< Tensor<FeaturesType> > features_tensor_ptr( new Tensor<FeaturesType>(input_dims) );
		for (size_t i = 0; i < features.size(); i++)
			(*features_tensor_ptr)[i] = features[i];
		res.push_back(features_tensor_ptr);
	}

	FullMeanSubtract(res, means);
	FullStdDivide(res, stds);

	return res;
}

template <class T>
bool CheckNNGradientsSameForDifferentBufferSizes(NN<double>& nn, CostModule<double>& cost_module, ITrainDataset<T>& dataset, std::vector<size_t>& inds)
{
	size_t initial_buffer_size = nn.GetMinibatchSize();
	size_t num_params = nn.GetNumParams();
	std::vector<double> parameters = nn.GetParameters();
	nn.SetMinibatchSize(100);
	CostAndGradients<double> res1 = nn.GetGradientsAndCost(dataset, cost_module, inds, true);
	std::vector<double> gradients1 = res1.gradients;
	double cost1 = res1.cost;

	nn.SetMinibatchSize(4);
	CostAndGradients<double> res2 = nn.GetGradientsAndCost(dataset, cost_module, inds, true);
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

template <class T>
bool NumericalCheckNNGradients(NN<double>& nn, CostModule<double>& cost_module, ITrainDataset<T>& dataset, size_t num_samples)
{
	std::vector<size_t> inds;
	for (size_t i=0; i<num_samples; i++)
		inds.push_back(i);

	// check that each fprop does not affect other fprops and that the cost is consistent
	if (nn.GetCost(dataset, cost_module, inds, true, true) != nn.GetCost(dataset, cost_module, inds, true, true))
		return false;
	
	if (nn.GetCost(dataset, cost_module, inds, false, true) != nn.GetCost(dataset, cost_module, inds, false, true))
		return false;
	
	if (nn.GetCost(dataset, cost_module, inds, false, false) != nn.GetCost(dataset, cost_module, inds, false, false))
		return false;

	size_t num_params = nn.GetNumParams();
	std::vector<double> parameters = nn.GetParameters();
	CostAndGradients<double> res1 = nn.GetGradientsAndCost(dataset, cost_module, inds, true);
	std::vector<double> gradients1 = res1.gradients;
	// Test that all buffers are cleared
	CostAndGradients<double> res2 = nn.GetGradientsAndCost(dataset, cost_module, inds, true);
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
		double cost1 = nn.GetCost(dataset, cost_module, inds, true, true);
		parameters[i]=initial_val+eps;
		// Use the array approach for initializing parameters
		nn.SetParameters(parameters.data());
		double cost2 = nn.GetCost(dataset, cost_module, inds, true, true);
		double numerical_gradient = (cost2-cost1) / 2 / eps;
		if (abs(numerical_gradient-gradients[i]) / (std::max<double>)(abs(numerical_gradient)+abs(gradients[i]), 1) > 0.000001)
			return false;
		parameters[i] = initial_val;
		nn.SetParameters(parameters);
	}
	return CheckNNGradientsSameForDifferentBufferSizes(nn, cost_module, dataset, inds);
}

template <class T>
bool TestGetSetParameters(NN<T>& nn)
{
	size_t num_params = nn.GetNumParams();
	std::vector<T> parameters;
	std::vector<T> zero_parameters(num_params);

	std::vector<T> module_parameters;
	for (size_t i = 0; i < num_params; i++)
		module_parameters.push_back(static_cast<T>(i));

	nn.SetParameters(zero_parameters);
	nn.SetParameters(module_parameters);
	parameters = nn.GetParameters();
	if ( parameters != module_parameters )
		return false;
	
	nn.SetParameters(zero_parameters);
	nn.SetParameters(module_parameters.data());
	parameters = nn.GetParameters();
	return parameters == module_parameters;
}

std::vector<size_t> GetIndicesVector(size_t numel)
{
	std::vector<size_t> res(numel);
	for (size_t i=0; i<numel; i++)
		res[i] = i;
	return res;
}

template <class T>
std::vector<T>	GetData(std::vector< T >& input, std::vector<size_t>& indices)
{
	std::vector<T> res(indices.size());
	for (size_t i=0; i<indices.size(); i++)
		res[i] = input[indices[i]];

	return res;
}

template <class ParamsType, class FeaturesType>
std::pair< SemisupervisedDataset<ParamsType, FeaturesType>, SemisupervisedDataset<ParamsType, FeaturesType> > 
	GetTrainValidationDatasets(SemisupervisedDataset<ParamsType, FeaturesType>& dataset, double fraction)
{
	std::pair< SemisupervisedDataset<ParamsType, FeaturesType>, SemisupervisedDataset<ParamsType, FeaturesType> > res;
	SemisupervisedDataset<ParamsType, FeaturesType>& train_dataset = res.first;
	SemisupervisedDataset<ParamsType, FeaturesType>& validation_dataset = res.second;

	size_t num_labeled_cases = dataset.labeled_input.size();
	std::pair< std::vector<size_t>, std::vector<size_t> > labeled_split_inds = StochasticSplitVector( GetIndicesVector(num_labeled_cases), fraction );
	std::vector<size_t> train_labeled_split_inds = labeled_split_inds.first;
	std::cout<<train_labeled_split_inds[0]<<std::endl;
	std::cout<<train_labeled_split_inds[1]<<std::endl;
	std::vector<size_t> validation_labeled_split_inds = labeled_split_inds.second;
	train_dataset.labeled_input = GetData( dataset.labeled_input, train_labeled_split_inds);
	train_dataset.labels = GetData( dataset.labels, train_labeled_split_inds);
	train_dataset.labeled_importance = GetData( dataset.labeled_importance, train_labeled_split_inds);
	validation_dataset.labeled_input = GetData( dataset.labeled_input, validation_labeled_split_inds);
	validation_dataset.labels = GetData( dataset.labels, validation_labeled_split_inds);
	validation_dataset.labeled_importance = GetData( dataset.labeled_importance, validation_labeled_split_inds);
	
	size_t num_unlabeled_cases = dataset.unlabeled_input.size();
	std::pair< std::vector<size_t>, std::vector<size_t> > unlabeled_split_inds = StochasticSplitVector( GetIndicesVector(num_unlabeled_cases), fraction );
	std::vector<size_t> train_unlabeled_split_inds = unlabeled_split_inds.first;
	std::vector<size_t> validation_unlabeled_split_inds = unlabeled_split_inds.second;
	train_dataset.unlabeled_input = GetData( dataset.unlabeled_input, train_unlabeled_split_inds);
	train_dataset.unlabeled_importance = GetData( dataset.unlabeled_importance, train_unlabeled_split_inds);
	validation_dataset.unlabeled_input = GetData( dataset.unlabeled_input, validation_unlabeled_split_inds);
	validation_dataset.unlabeled_importance = GetData( dataset.unlabeled_importance, validation_unlabeled_split_inds);

	train_dataset.data_means = dataset.data_means;
	train_dataset.data_stds = dataset.data_stds;
	validation_dataset.data_means = dataset.data_means;
	validation_dataset.data_stds = dataset.data_stds;

	return res;
}

template <class ParamsType>
struct ProcessValidationFunc
{
private:
	std::string save_net_path_;
public:
	void operator()(ValidationCallbackParams<ParamsType>& result)
	{
		if (result.is_best)
		{
			std::cout<<result.batch_num<<" Train cost = "<<result.train_cost<<" validation cost = "<<result.validation_cost<<" BEST"<<std::endl;
			std::ofstream stream( save_net_path_ );
			std::shared_ptr< IOTreeNode > state = result.net.GetState();
			IOXML::save(*state, stream);
		}
		else
			std::cout<<result.batch_num<<" Train cost = "<<result.train_cost<<" validation cost = "<<result.validation_cost<<std::endl;
	}

	ProcessValidationFunc(const std::string& save_net_path)
	{
		save_net_path_ = save_net_path;
	}
};

struct TrainingParams
{
	size_t num_iterations;
	double learning_rate;
	double momentum;
	size_t train_batch_size;
	size_t validation_batch_size;
	double train_decay;
	double validation_decay;
	size_t num_batches_before_validation_evaluation;
	size_t num_batches_before_train_evaluation;
	size_t num_warmup_batches;
	double warmup_momentum;
	std::string save_best_net_path;
	std::string labeled_train_dataset_path;
	std::string unlabeled_train_dataset_path;
	size_t l1_size;
	size_t l2_size;
	size_t l3_size;
	size_t num_clases;
	double weight_decay;
	double dropout;
	double gaussian_noise_std;
	double entropy_lambda;
	size_t entropy_num_groups;
	size_t lbfgs_num_iterations_per_update;

	void save(std::ostream& stream)
	{
		std::shared_ptr< IOTreeNode > node( new IOTreeNode() );
		node->attributes().AppendEntry("num_iterations", std::to_string(num_iterations) );
		node->attributes().AppendEntry("learning_rate", std::to_string(learning_rate) );
		node->attributes().AppendEntry("momentum", std::to_string(momentum) );
		node->attributes().AppendEntry("train_batch_size", std::to_string(train_batch_size) );
		node->attributes().AppendEntry("validation_batch_size", std::to_string(validation_batch_size) );
		node->attributes().AppendEntry("train_decay", std::to_string(train_decay) );
		node->attributes().AppendEntry("validation_decay", std::to_string(validation_decay) );
		node->attributes().AppendEntry("num_batches_before_validation_evaluation", std::to_string(num_batches_before_validation_evaluation) );
		node->attributes().AppendEntry("num_batches_before_train_evaluation", std::to_string(num_batches_before_train_evaluation) );
		node->attributes().AppendEntry("num_warmup_batches", std::to_string(num_warmup_batches) );
		node->attributes().AppendEntry("warmup_momentum", std::to_string(warmup_momentum) );
		node->attributes().AppendEntry("save_best_net_path", save_best_net_path );
		node->attributes().AppendEntry("labeled_train_dataset_path", labeled_train_dataset_path );
		node->attributes().AppendEntry("unlabeled_train_dataset_path", unlabeled_train_dataset_path );
		node->attributes().AppendEntry("l1_size", std::to_string(l1_size) );
		node->attributes().AppendEntry("l2_size", std::to_string(l2_size) );
		node->attributes().AppendEntry("l3_size", std::to_string(l3_size) );
		node->attributes().AppendEntry("num_clases", std::to_string(num_clases) );
		node->attributes().AppendEntry("weight_decay", std::to_string(weight_decay) );
		node->attributes().AppendEntry("dropout", std::to_string(dropout) );
		node->attributes().AppendEntry("gaussian_noise_std", std::to_string(gaussian_noise_std) );
		node->attributes().AppendEntry("entropy_lambda", std::to_string(entropy_lambda) );
		node->attributes().AppendEntry("entropy_num_groups", std::to_string(entropy_num_groups) );
		node->attributes().AppendEntry("lbfgs_num_iterations_per_update", std::to_string(lbfgs_num_iterations_per_update) );
		IOXML::save(*node, stream);
	}

	static TrainingParams Create(std::istream& stream)
	{
		std::shared_ptr< IOTreeNode > node = IOXML::load(stream);

		TrainingParams params;

		if (node->attributes().HasEntry("num_iterations"))
			params.num_iterations = Converter::ConvertTo<size_t>( node->attributes().GetEntry("num_iterations") );
		else
			params.num_iterations = 0;

		if (node->attributes().HasEntry("learning_rate"))
			params.learning_rate = Converter::ConvertTo<double>( node->attributes().GetEntry("learning_rate") );
		else
			params.learning_rate = 0;

		if (node->attributes().HasEntry("momentum"))
			params.momentum = Converter::ConvertTo<double>( node->attributes().GetEntry("momentum") );
		else
			params.momentum = 0;

		if (node->attributes().HasEntry("train_batch_size"))
			params.train_batch_size = Converter::ConvertTo<size_t>( node->attributes().GetEntry("train_batch_size") );
		else
			params.train_batch_size = 0;

		if (node->attributes().HasEntry("validation_batch_size"))
			params.validation_batch_size = Converter::ConvertTo<size_t>( node->attributes().GetEntry("validation_batch_size") );
		else
			params.validation_batch_size = 0;

		if (node->attributes().HasEntry("train_decay"))
			params.train_decay = Converter::ConvertTo<double>( node->attributes().GetEntry("train_decay") );
		else
			params.train_decay = 0;

		if (node->attributes().HasEntry("validation_decay"))
			params.validation_decay = Converter::ConvertTo<double>( node->attributes().GetEntry("validation_decay") );
		else
			params.validation_decay = 0;

		if (node->attributes().HasEntry("num_batches_before_validation_evaluation"))
			params.num_batches_before_validation_evaluation = Converter::ConvertTo<size_t>( node->attributes().GetEntry("num_batches_before_validation_evaluation") );
		else
			params.num_batches_before_validation_evaluation = 0;

		if (node->attributes().HasEntry("num_batches_before_train_evaluation"))
			params.num_batches_before_train_evaluation = Converter::ConvertTo<size_t>( node->attributes().GetEntry("num_batches_before_train_evaluation") );
		else
			params.num_batches_before_train_evaluation = 0;

		if (node->attributes().HasEntry("num_warmup_batches"))
			params.num_warmup_batches = Converter::ConvertTo<size_t>( node->attributes().GetEntry("num_warmup_batches") );
		else
			params.num_warmup_batches = 0;

		if (node->attributes().HasEntry("warmup_momentum"))
			params.warmup_momentum = Converter::ConvertTo<double>( node->attributes().GetEntry("warmup_momentum") );
		else
			params.warmup_momentum = 0;

		if (node->attributes().HasEntry("save_best_net_path"))
			params.save_best_net_path = node->attributes().GetEntry("save_best_net_path");
		else
			params.save_best_net_path = "";

		if (node->attributes().HasEntry("labeled_train_dataset_path"))
			params.labeled_train_dataset_path = node->attributes().GetEntry("labeled_train_dataset_path");
		else
			params.labeled_train_dataset_path = "";
		
		if (node->attributes().HasEntry("unlabeled_train_dataset_path"))
			params.unlabeled_train_dataset_path = node->attributes().GetEntry("unlabeled_train_dataset_path");
		else
			params.unlabeled_train_dataset_path = "";

		if (node->attributes().HasEntry("l1_size"))
			params.l1_size = Converter::ConvertTo<size_t>( node->attributes().GetEntry("l1_size") );
		else
			params.l1_size = 0;

		if (node->attributes().HasEntry("l2_size"))
			params.l2_size = Converter::ConvertTo<size_t>( node->attributes().GetEntry("l2_size") );
		else
			params.l2_size = 0;

		if (node->attributes().HasEntry("l3_size"))
			params.l3_size = Converter::ConvertTo<size_t>( node->attributes().GetEntry("l3_size") );
		else
			params.l3_size = 0;

		if (node->attributes().HasEntry("num_clases"))
			params.num_clases = Converter::ConvertTo<size_t>( node->attributes().GetEntry("num_clases") );
		else
			params.num_clases = 0;

		if (node->attributes().HasEntry("weight_decay"))
			params.weight_decay = Converter::ConvertTo<double>( node->attributes().GetEntry("weight_decay") );
		else
			params.weight_decay = 0;

		if (node->attributes().HasEntry("dropout"))
			params.dropout = Converter::ConvertTo<double>( node->attributes().GetEntry("dropout") );
		else
			params.dropout = 0;
		
		if (node->attributes().HasEntry("gaussian_noise_std"))
			params.gaussian_noise_std = Converter::ConvertTo<double>( node->attributes().GetEntry("gaussian_noise_std") );
		else
			params.gaussian_noise_std = 0;
		
		if (node->attributes().HasEntry("entropy_lambda"))
			params.entropy_lambda = Converter::ConvertTo<double>( node->attributes().GetEntry("entropy_lambda") );
		else
			params.entropy_lambda = 0;

		if (node->attributes().HasEntry("entropy_num_groups"))
			params.entropy_num_groups = Converter::ConvertTo<size_t>( node->attributes().GetEntry("entropy_num_groups") );
		else
			params.entropy_num_groups = 0;
		
		if (node->attributes().HasEntry("lbfgs_num_iterations_per_update"))
			params.lbfgs_num_iterations_per_update = Converter::ConvertTo<size_t>( node->attributes().GetEntry("lbfgs_num_iterations_per_update") );
		else
			params.lbfgs_num_iterations_per_update = 0;

		return params;
	}

	bool Equals(TrainingParams& params)
	{
		return (params.num_iterations == num_iterations) &&
			(params.learning_rate == learning_rate) &&
			(params.momentum == momentum) &&
			(params.train_batch_size == train_batch_size) &&
			(params.validation_batch_size == validation_batch_size) &&
			(params.validation_decay == validation_decay) &&
			(params.train_decay == train_decay) &&
			(params.num_batches_before_validation_evaluation == num_batches_before_validation_evaluation) &&
			(params.num_batches_before_validation_evaluation == num_batches_before_validation_evaluation) &&
			(params.num_batches_before_train_evaluation == num_batches_before_train_evaluation) &&
			(params.num_warmup_batches == num_warmup_batches) &&
			(params.warmup_momentum == warmup_momentum) &&
			(params.save_best_net_path == save_best_net_path) &&
			(params.labeled_train_dataset_path == labeled_train_dataset_path) &&
			(params.unlabeled_train_dataset_path == unlabeled_train_dataset_path) &&
			(params.l1_size == l1_size) &&
			(params.l2_size == l2_size) &&
			(params.l3_size == l3_size) &&
			(params.num_clases == num_clases) &&
			(params.weight_decay == weight_decay) &&
			(params.dropout == dropout) &&
			(params.gaussian_noise_std == gaussian_noise_std) &&
			(params.entropy_lambda == entropy_lambda) &&
			(params.entropy_num_groups == entropy_num_groups) &&
			(params.lbfgs_num_iterations_per_update == lbfgs_num_iterations_per_update);

	}
};

template <class T>
std::shared_ptr< NN<T> > load_net(std::string filename)
{
	std::ifstream stream(filename);
	std::shared_ptr<IOTreeNode> node = IOXML::load(stream);
	return NN<T>::Create( *node );
}

void DefaultInitializeParams(TrainingParams& params)
{
	params.labeled_train_dataset_path = "D:/train.csv";
	params.unlabeled_train_dataset_path = "D:/KaggleSemisupervisedLearning/extra_unsupervised_data.csv";
	params.save_best_net_path = "D:/best_net.txt";
	params.l1_size = 100;
	params.l2_size = 100;
	params.l3_size = 100;
	params.num_clases = 2;
	params.weight_decay = 0.00000001;
	params.num_iterations = 100000000;
	params.learning_rate = 0.0001;
	params.momentum = 0.99;//0.9;
	params.train_batch_size = 30;
	params.validation_batch_size = 10000;
	params.train_decay = 0.999;
	params.validation_decay = 0;
	params.num_batches_before_validation_evaluation = 5000;
	params.num_batches_before_train_evaluation = 100;
	params.num_warmup_batches = 0;
	params.warmup_momentum = 0.5;
	params.dropout = 0.5;
	params.gaussian_noise_std = 0;
	params.entropy_lambda = 25;
	params.entropy_num_groups = 4;
	params.lbfgs_num_iterations_per_update = 20;
}

void SaveParams(TrainingParams& params, std::string filepath)
{
	std::ofstream out_stream(filepath);
	params.save(out_stream);
	out_stream.close();
}

TrainingParams LoadParams(std::string filepath)
{
	std::ifstream input_stream(filepath);
	return TrainingParams::Create( input_stream );
}

template <class T>
void MergeVectors(std::vector<T>& add_to_vect, std::vector<T>& added_vect)
{
	add_to_vect.insert(add_to_vect.end(), added_vect.data(), added_vect.data()+added_vect.size());
}

void semisupervised_learning_train2()
{
	typedef float ParamsType;
	typedef float FeaturesType;
	
	TrainingParams params;
	DefaultInitializeParams(params);

	//SaveParams(params, "D:/training_params.txt");
	//params = LoadParams("G:/train_net/comp1/training_params.txt");

	std::cout<<"Loading datasets"<<std::endl;
	SemisupervisedDataset<ParamsType, FeaturesType> dataset = 
		LoadTrainDataFromFile<ParamsType, FeaturesType>(params.labeled_train_dataset_path, true, true, params.unlabeled_train_dataset_path);
	std::pair< SemisupervisedDataset<ParamsType, FeaturesType>, SemisupervisedDataset<ParamsType, FeaturesType> > datasets = 
		GetTrainValidationDatasets<ParamsType, FeaturesType>(dataset, 0.75);
	SemisupervisedDataset<ParamsType, FeaturesType> train_data = datasets.first;
	SemisupervisedDataset<ParamsType, FeaturesType> validation_data = datasets.second;

	MergeVectors(train_data.labeled_input, train_data.unlabeled_input);
	MergeVectors(train_data.labeled_importance, train_data.unlabeled_importance);
	std::vector<size_t> labels_dims = dataset.labels[0]->GetDimensions();
	std::shared_ptr< Tensor<FeaturesType> > zero_labels_tensor( new Tensor<FeaturesType>(labels_dims) );
	for (size_t i=0; i<train_data.unlabeled_input.size(); i++)
		train_data.labels.push_back(zero_labels_tensor);

	std::shared_ptr< ITensorDataLoader<ParamsType> > train_labeled_input_loader( 
		new SemisupervisedTensorDataLoader<ParamsType, FeaturesType>( train_data.labeled_input, train_data.labels, 0.5 ) );
	std::shared_ptr< ITensorDataLoader<ParamsType> > train_labeled_output_loader( new FullTensorDataLoader<ParamsType, FeaturesType>( train_data.labels ) );
	std::shared_ptr< ITensorDataLoader<ParamsType> > validation_labeled_input_loader( new FullTensorDataLoader<ParamsType, FeaturesType>( validation_data.labeled_input ) );
	std::shared_ptr< ITensorDataLoader<ParamsType> > validation_labeled_output_loader( new FullTensorDataLoader<ParamsType, FeaturesType>( validation_data.labels ) );

	std::cout<<"Training"<<std::endl;
	std::shared_ptr< ITrainDataset<ParamsType> > train_labeled_dataset( new TrainDataset<ParamsType>(train_labeled_input_loader, 
		train_labeled_output_loader, train_data.labeled_importance) );
	std::shared_ptr< ITrainDataset<ParamsType> > validation_labeled_dataset( new TrainDataset<ParamsType>(validation_labeled_input_loader, 
		validation_labeled_output_loader, validation_data.labeled_importance) );

	std::vector<size_t> input_dims; input_dims.push_back(1875);
	std::shared_ptr< NN<ParamsType> > nn;
	std::ifstream best_net_file(params.save_best_net_path);
	if (best_net_file.good())
	{
		best_net_file.close();
		nn = load_net<ParamsType>(params.save_best_net_path);
	}
	else
	{
		best_net_file.close();
		nn = ConstructRluRluSoftmaxNN<ParamsType>(input_dims, params.l1_size, params.l2_size, params.num_clases, params.weight_decay);

	}

	SGD_Trainer<ParamsType> trainer(params.num_iterations, params.learning_rate, params.momentum, params.train_batch_size,
		params.validation_batch_size, params.train_decay, params.validation_decay, params.num_batches_before_train_evaluation, 
		params.num_batches_before_validation_evaluation, params.num_warmup_batches, params.warmup_momentum);
	SemisupervisedCostModule<ParamsType> train_cost_module( std::shared_ptr< CostModule<ParamsType> >( new CrossEntropyCostModule<ParamsType>()), 
		std::shared_ptr< CostModule<ParamsType> >(new EntropyCostModule<ParamsType>()), 1, 1 );
	MisclassificationRateCostModule<ParamsType> validation_cost_module;
	trainer.Train(*nn, train_cost_module, validation_cost_module, *train_labeled_dataset, *validation_labeled_dataset, 
		DefaultProcessTrainFunc<ParamsType>, ProcessValidationFunc<ParamsType>(params.save_best_net_path));
}


void semisupervised_learning_main()
{
	typedef float ParamsType;
	typedef float FeaturesType;
	
	TrainingParams params;
	DefaultInitializeParams(params);

	//SaveParams(params, "D:/training_params.txt");
	//params = LoadParams("G:/train_net/comp1/training_params.txt");

	std::cout<<"Loading datasets"<<std::endl;
	SemisupervisedDataset<ParamsType, FeaturesType> dataset = 
		LoadTrainDataFromFile<ParamsType, FeaturesType>(params.labeled_train_dataset_path, true, false, params.unlabeled_train_dataset_path);
	std::pair< SemisupervisedDataset<ParamsType, FeaturesType>, SemisupervisedDataset<ParamsType, FeaturesType> > datasets = 
		GetTrainValidationDatasets<ParamsType, FeaturesType>(dataset, 0.75);
	SemisupervisedDataset<ParamsType, FeaturesType> train_data = datasets.first;
	SemisupervisedDataset<ParamsType, FeaturesType> validation_data = datasets.second;

	MergeVectors(train_data.labeled_input, train_data.unlabeled_input);
	MergeVectors(train_data.labeled_importance, train_data.unlabeled_importance);
	std::vector<size_t> labels_dims = dataset.labels[0]->GetDimensions();
	std::shared_ptr< Tensor<FeaturesType> > zero_labels_tensor( new Tensor<FeaturesType>(labels_dims) );
	for (size_t i=0; i<train_data.unlabeled_input.size(); i++)
		train_data.labels.push_back(zero_labels_tensor);

	//std::shared_ptr< ITensorDataLoader<ParamsType> > train_labeled_input_loader( new FullTensorDataLoader<ParamsType, FeaturesType>( train_data.labeled_input ) );
	//std::shared_ptr< ITensorDataLoader<ParamsType> > train_labeled_output_loader( new FullTensorDataLoader<ParamsType, FeaturesType>( train_data.labels ) );
	//std::shared_ptr< ITensorDataLoader<ParamsType> > validation_labeled_input_loader( new FullTensorDataLoader<ParamsType, FeaturesType>( validation_data.labeled_input ) );
	//std::shared_ptr< ITensorDataLoader<ParamsType> > validation_labeled_output_loader( new FullTensorDataLoader<ParamsType, FeaturesType>( validation_data.labels ) );
	//
	//std::shared_ptr< ITensorDataLoader<ParamsType> > train_unlabeled_input_loader( new FullTensorDataLoader<ParamsType, FeaturesType>( train_data.unlabeled_input ) );
	//std::shared_ptr< ITensorDataLoader<ParamsType> > validation_unlabeled_input_loader( new 
	//	FullTensorDataLoader<ParamsType, FeaturesType>( validation_data.unlabeled_input ) );

	std::shared_ptr< ITensorDataLoader<ParamsType> > train_labeled_input_loader( 
		new ClassificationBalancedPairsTensorDataLoader<ParamsType, FeaturesType>( train_data.labeled_input, train_data.labels ) );
	std::shared_ptr< ITensorDataLoader<ParamsType> > train_labeled_output_loader( 
		new ClassificationPairsOutputTensorDataLoader<ParamsType, FeaturesType>( train_data.labels ) );
	std::shared_ptr< ITensorDataLoader<ParamsType> > validation_labeled_input_loader( 
		new ClassificationBalancedPairsTensorDataLoader<ParamsType, FeaturesType>( validation_data.labeled_input, validation_data.labels ) );
	std::shared_ptr< ITensorDataLoader<ParamsType> > validation_labeled_output_loader( 
		new ClassificationPairsOutputTensorDataLoader<ParamsType, FeaturesType>( validation_data.labels ) );

	std::cout<<"Training"<<std::endl;
	std::shared_ptr< ITrainDataset<ParamsType> > train_labeled_dataset( new TrainDatasetPairsWithBatchIndProblem<ParamsType>(train_labeled_input_loader, 
		train_labeled_output_loader, train_data.labeled_importance) );
	std::shared_ptr< ITrainDataset<ParamsType> > validation_labeled_dataset( new TrainDatasetPairsWithBatchIndProblem<ParamsType>(validation_labeled_input_loader, 
		validation_labeled_output_loader, validation_data.labeled_importance) );

	//std::shared_ptr< ITrainDataset<ParamsType> > train_labeled_dataset( new TrainDataset<ParamsType>(train_labeled_input_loader, 
	//	train_labeled_input_loader, train_data.labeled_importance) );
	//std::shared_ptr< ITrainDataset<ParamsType> > validation_labeled_dataset( new TrainDataset<ParamsType>(validation_labeled_input_loader, 
	//	validation_labeled_input_loader, validation_data.labeled_importance) );

	//std::shared_ptr< ITrainDataset<ParamsType> > train_labeled_dataset( new TrainDatasetPairsWithBatchIndProblem<ParamsType>(train_labeled_input_loader, 
	//	train_labeled_output_loader, train_data.labeled_importance) );
	//std::shared_ptr< ITrainDataset<ParamsType> > validation_labeled_dataset( new TrainDatasetPairsWithBatchIndProblem<ParamsType>(validation_labeled_input_loader, 
	//	validation_labeled_output_loader, validation_data.labeled_importance) );

	/*std::shared_ptr< ITrainDataset<ParamsType> > train_unlabeled_dataset( new TrainDataset<ParamsType>(train_unlabeled_input_loader, 
		train_unlabeled_input_loader, train_data.unlabeled_importance) );
	std::shared_ptr< ITrainDataset<ParamsType> > validation_unlabeled_dataset( new TrainDataset<ParamsType>(validation_unlabeled_input_loader, 
		validation_unlabeled_input_loader, validation_data.unlabeled_importance) );*/

	std::vector<size_t> input_dims; input_dims.push_back(2*1875);
	std::shared_ptr< NN<ParamsType> > nn;
	std::ifstream best_net_file(params.save_best_net_path);
	if (best_net_file.good())
	{
		best_net_file.close();
		nn = load_net<ParamsType>(params.save_best_net_path);
	}
	else
	{
		best_net_file.close();
		//nn = ConstructRluLinearNN< ParamsType >(input_dims, params.l1_size, input_dims[0], params.weight_decay);
		//nn = ConstructRluRluRluLinearNN< ParamsType >(input_dims, params.l1_size, params.l2_size, params.l3_size, input_dims[0], params.weight_decay);
		//nn = ConstructRluEntropyLinearNN< ParamsType >(input_dims, params.l1_size, input_dims[0], 
		//	params.weight_decay, params.entropy_lambda, params.entropy_num_groups);
		//nn = ConstructSigmoidEntropyLinearNN< ParamsType >(input_dims, params.l1_size, input_dims[0], 
		//	params.weight_decay, params.entropy_lambda, params.entropy_num_groups);
		//nn = ConstructSigmoidLinearNN< ParamsType >(input_dims, params.l1_size, input_dims[0], params.weight_decay);
		//nn = ConstructTanhLinearNN< ParamsType >(input_dims, params.l1_size, input_dims[0], params.weight_decay);
		//nn = ConstructRluRluRluSoftmaxDropoutNN<ParamsType>(input_dims, params.l1_size, params.l2_size, params.l3_size, params.num_clases, params.weight_decay, params.dropout);
		//nn = ConstructTanhRluTanhRluSoftmaxNormalizeDropoutdNN<ParamsType>(input_dims, params.l1_size, 
		//	params.l2_size, params.num_clases, params.weight_decay, params.dropout);
		//nn = ConstructTanhTanhSoftmaxNN<ParamsType>(input_dims, params.l1_size, params.l2_size, 2, params.weight_decay);
		//nn = ConstructSoftmaxNN< ParamsType >(input_dims, params.num_clases, params.weight_decay);
		nn = ConstructRluSoftmaxNN< ParamsType >(input_dims, params.l1_size, params.num_clases, params.weight_decay);
	}

	SGD_Trainer<ParamsType> trainer(params.num_iterations, params.learning_rate, params.momentum, params.train_batch_size,
		params.validation_batch_size, params.train_decay, params.validation_decay, params.num_batches_before_train_evaluation, 
		params.num_batches_before_validation_evaluation, params.num_warmup_batches, params.warmup_momentum);
	CrossEntropyCostModule<ParamsType> train_cost_module;
	CrossEntropyCostModule<ParamsType> validation_cost_module;
	//MisclassificationRateCostModule<ParamsType> validation_cost_module;
	//MseCostModule<ParamsType> train_cost_module;
	//MseCostModule<ParamsType> validation_cost_module;
	trainer.Train(*nn, train_cost_module, validation_cost_module, *train_labeled_dataset, *validation_labeled_dataset, 
		DefaultProcessTrainFunc<ParamsType>, ProcessValidationFunc<ParamsType>(params.save_best_net_path));

	//LbfgsMinibatchTrainer<ParamsType> trainer(params.num_iterations, params.lbfgs_num_iterations_per_update, 
	//	params.train_batch_size, params.validation_batch_size, params.train_decay, params.validation_decay, params.num_batches_before_train_evaluation, 
	//	params.num_batches_before_validation_evaluation);
	////CrossEntropyCostModule<ParamsType> train_cost_module;
	////MisclassificationRateCostModule<ParamsType> validation_cost_module;
	//MseCostModule<ParamsType> train_cost_module;
	//MseCostModule<ParamsType> validation_cost_module;
	//trainer.Train(*nn, train_cost_module, validation_cost_module, *train_labeled_dataset, *validation_labeled_dataset, 
	//	DefaultProcessTrainFunc<ParamsType>, ProcessValidationFunc<ParamsType>(params.save_best_net_path));
}

// NN tests
	//if (!NumericalCheckNNGradients(*nn, MseCostModule<double>(), *train_dataset, 9))
	//	throw 1;

	//if (!TestGetSetParameters(*nn))
	//	throw 1;

// NN construction
	//nn = ConstructTanhTanhSoftmaxNN<ParamsType>(input_dims, 
	//    params.l1_size, params.l2_size, params.num_clases, params.weight_decay);
	//nn = ConstructRluRluSoftmaxNN<ParamsType>(input_dims, 
	//	params.l1_size, params.l2_size, params.num_clases, params.weight_decay);
	//nn = ConstructRluRluSoftmaxNormalizedNN<ParamsType>(input_dims, 
	//	params.l1_size, params.l2_size, params.num_clases, params.weight_decay);
	//nn = ConstructRluRluSoftmaxDropoutNN<ParamsType>(input_dims, 
	//	params.l1_size, params.l2_size, params.num_clases, params.weight_decay, params.dropout);
	//nn = ConstructTanhTanhSoftmaxNormalizedNN<ParamsType>(input_dims, 
	//	params.l1_size, params.l2_size, params.num_clases, params.weight_decay);
	//nn = ConstructRluRluSoftmaxNormalizedDropoutNN<ParamsType>(input_dims, 
	//	params.l1_size, params.l2_size, params.num_clases, params.weight_decay, params.dropout);
	//nn = ConstructTanhRluTanhRluSoftmaxNormalizedNN<ParamsType>(input_dims, 
	//	params.l1_size, params.l2_size, params.num_clases, params.weight_decay);
	//nn = ConstructTanhRluTanhRluSoftmaxNormalizeDropoutdNN<ParamsType>(input_dims, 
	//	params.l1_size, params.l2_size, params.num_clases, params.weight_decay, params.dropout);
	//nn = ConstructRluRluRluSoftmaxNormalizedNN<ParamsType>(input_dims, 
	//	params.l1_size, params.l2_size, params.l3_size, params.num_clases, params.weight_decay);
	//std::shared_ptr< NN<ParamsType> > nn = ConstructTanhTanhSoftmaxNN<ParamsType>(input_dims, 
	//	l1_size, l2_size, num_clases, weight_decay);
	//nn = ConstructSoftmaxNN< ParamsType >(input_dims, params.num_clases, params.weight_decay);

// calculates classes frequences
	//std::vector<size_t> classes_frequences(9);
	//for (size_t i=0; i<train_output_loader->GetNumSamples(); i++)
	//{
	//	std::vector<size_t> case_ind(1,i);
	//	std::shared_ptr< Tensor<ParamsType> > sample = train_output_loader->GetData(case_ind);
	//	for (size_t i = 0; i < classes_frequences.size(); i++)
	//		classes_frequences[i] += (*sample)[i];
	//}

#endif