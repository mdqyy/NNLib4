#ifndef TEST_UTILITIES_H
#define TEST_UTILITIES_H

#include <vector>
#include <memory>
#include <Windows.h>
#include "Tensor.h"
#include "Kernel.h"
#include "NN.h"
#include "RandomGenerator.h"
#include "IOTreeNode.h"
#include "IOXML.h"

template <class DataType>
Tensor<DataType> GetRandomTensor(std::vector<size_t> tensor_dims, double min_val=-1, double max_val=1)
{
	Tensor<DataType> tensor(tensor_dims);
	for (size_t i=0; i<tensor.Numel(); i++)
		tensor[i] = (DataType)RandomGenerator::GetUniformDouble(min_val, max_val);
	return tensor;
}

template <class DataType>
std::shared_ptr< Tensor<DataType> > GetRandomTensorPtr(std::vector<size_t> tensor_dims, double min_val=-1, double max_val=1)
{
	Tensor<DataType> tensor(tensor_dims);
	for (size_t i=0; i<tensor.Numel(); i++)
		tensor[i] = (DataType)RandomGenerator::GetUniformDouble(min_val, max_val);
	return std::shared_ptr< Tensor<DataType> >( new Tensor<DataType>(tensor) );
}

long long milliseconds_now();


template <class T>
bool test_equal_arrays(T* arr1, T* arr2, int N, T precision = 0.001)
{
	for (int i=0; i<N; i++)
		if ( std::abs(arr1[i] - arr2[i]) > precision)
			return false;
	return true;
}

template <class T>
bool CheckNNGradientsSameForDifferentBufferSizes(NN<double>& nn, CostModule<double>& cost_module, ITrainDataset<T>& dataset)
{
	size_t initial_buffer_size = nn.GetMinibatchSize();
	size_t num_params = nn.GetNumParams();
	std::vector<double> parameters = nn.GetParameters();
	nn.SetMinibatchSize(100);
	CostAndGradients<double> res1 = nn.GetGradientsAndCost(dataset, cost_module, std::vector<size_t>(), true);
	std::vector<double> gradients1 = res1.gradients;
	double cost1 = res1.cost;

	nn.SetMinibatchSize(4);
	CostAndGradients<double> res2 = nn.GetGradientsAndCost(dataset, cost_module, std::vector<size_t>(), true);
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
bool NumericalCheckNNGradients(NN<double>& nn, CostModule<double>& cost_module, ITrainDataset<T>& dataset, bool test_minibatch = true)
{
	// check that each fprop does not affect other fprops and that the cost is consistent
	if (nn.GetCost(dataset, cost_module, std::vector<size_t>(), true, true, 0.5) != nn.GetCost(dataset, cost_module, std::vector<size_t>(), true, true, 0.5))
		return false;
	
	if (nn.GetCost(dataset, cost_module, std::vector<size_t>(), false, true, 0.5) != nn.GetCost(dataset, cost_module, std::vector<size_t>(), false, true, 0.5))
		return false;
	
	if (nn.GetCost(dataset, cost_module, std::vector<size_t>(), false, false, 0.5) != nn.GetCost(dataset, cost_module, std::vector<size_t>(), false, false, 0.5))
		return false;

	size_t num_params = nn.GetNumParams();
	std::vector<double> parameters = nn.GetParameters();
	CostAndGradients<double> res1 = nn.GetGradientsAndCost(dataset, cost_module, std::vector<size_t>(), true, 0.5);
	std::vector<double> gradients1 = res1.gradients;
	// Test that all buffers are cleared
	CostAndGradients<double> res2 = nn.GetGradientsAndCost(dataset, cost_module, std::vector<size_t>(), true, 0.5);
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
		double cost1 = nn.GetCost(dataset, cost_module, std::vector<size_t>(), true, true, 0.5);
		parameters[i]=initial_val+eps;
		// Use the array approach for initializing parameters
		nn.SetParameters(parameters.data());
		double cost2 = nn.GetCost(dataset, cost_module, std::vector<size_t>(), true, true, 0.5);
		double numerical_gradient = (cost2-cost1) / 2 / eps;
		if (abs(numerical_gradient-gradients[i]) / (std::max<double>)(abs(numerical_gradient)+abs(gradients[i]), 1) > 0.000001)
			return false;
		parameters[i] = initial_val;
		nn.SetParameters(parameters);
	}

	if (test_minibatch)
		return CheckNNGradientsSameForDifferentBufferSizes(nn, cost_module, dataset);
	else
		return true;
}

template <class T>
bool TestGetSetParameters(Module<T>& module, size_t num_params)
{
	std::vector<T> parameters;
	std::vector<T> zero_parameters(num_params);

	std::vector<T> module_parameters;
	for (size_t i = 0; i < num_params; i++)
		module_parameters.push_back(static_cast<T>(i));

	parameters.clear();
	module.SetParameters(zero_parameters);
	module.SetParameters(module_parameters);
	module.GetParameters(parameters);
	if ( parameters != module_parameters )
		return false;
	
	parameters.clear();
	module.SetParameters(zero_parameters);
	module.SetParameters(module_parameters.data());
	module.GetParameters(parameters);
	return parameters == module_parameters;
}

template <class T>
bool test_save_load_nn_state(NN<T>& net)
{
	std::shared_ptr<IOTreeNode> net_state = net.GetState();
	std::stringstream stream;
	IOXML::save(*net_state, stream);
	net_state = IOXML::load(stream);
	std::shared_ptr< NN<T> > net2 = NN<T>::Create(*net_state);
	return net.Equals( *net2 );
}

template <class DataType>
bool test_filter_response( Tensor<DataType> input, Tensor<DataType> output, Kernel<DataType>& kernel, 
							  std::vector<size_t> kernel_dims, std::vector<size_t> kernel_strides)
{
	std::vector<size_t> left_margins(kernel_dims.size());
	std::vector<size_t> right_margins=kernel_dims;
	for (size_t i=0; i<right_margins.size(); i++)
			right_margins[i]--;
	std::vector<size_t> valid_input_offsets = Tensor<DataType>::GetValidOffsetsInds(input.GetDimensions(), Tensor<DataType>::GetStrides(input.GetDimensions()), 
												   left_margins, right_margins, kernel_strides);
	size_t output_pos = 0;
	for (size_t i=0; i<valid_input_offsets.size(); i++)
	{
		DataType obtained_output = output[output_pos];
		DataType expected_output = kernel.GetResponse(input, Tensor<DataType>::IndToPos(input.GetDimensions(), valid_input_offsets[i]).data());
		if ( abs(obtained_output - expected_output) > 0.001)
			return false;
		output_pos++;
	}
	return true;
}

#endif