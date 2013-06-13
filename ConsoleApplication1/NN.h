#ifndef NN_H
#define NN_H

#include <numeric>
#include <vector>
#include <memory>
#include <algorithm>
#include "my_math.h"
#include "CompositeModule.h"
#include "CostModule.h"
#include "TrainDataset.h"
#include "IOTreeNode.h"
#include "Converter.h"
#include "ModuleFactory.h"
#include "CostAndGradients.h"
#include "MatrixOperations.h"

template <class ParamsType>
class NN
{
private:
	size_t num_samples_in_buffer_;
	std::shared_ptr< CompositeModule<ParamsType> > nn_module_;
	std::vector<ParamsType> batch_gradients_;
	std::vector<ParamsType> gradients_;
	std::vector<ParamsType> parameters_;
	
	std::pair<double,double> GetCost_(ITrainDataset<ParamsType>& dataset, CostModule<ParamsType>& cost_module, std::vector<size_t>& indices, 
												  bool train_mode, bool with_bprop, bool with_regularization, double cost_module_lambda);

public:

	virtual std::vector< std::shared_ptr< Tensor<ParamsType> > > Predict(
		ITensorDataLoader<ParamsType>& loader, std::vector<size_t>& indices = std::vector<size_t>(), std::string output_module_name = "");

	double GetCost(ITrainDataset<ParamsType>& dataset, CostModule<ParamsType>& cost_module, 
		std::vector<size_t>& indices, bool train_mode, bool with_regularization = false, double cost_module_lambda=1);

	void SetMinibatchSize(size_t new_size)
	{
		num_samples_in_buffer_ = new_size;
	}
	
	size_t GetMinibatchSize()
	{
		return num_samples_in_buffer_;
	}

	CostAndGradients<ParamsType> GetGradientsAndCost(ITrainDataset<ParamsType>& dataset, CostModule<ParamsType>& cost_module, 
		std::vector<size_t>& indices = std::vector<size_t>(), bool with_regularization = false, double cost_module_lambda=1);

	size_t GetNumParams()
	{
		return nn_module_->GetNumParams();
	}

	void SetParameters(const std::vector<ParamsType>& parameters);

	void SetParameters(const ParamsType* parameters);

	void InitializeParameters()
	{
		nn_module_->InitializeParameters();
	}

	NN(std::shared_ptr< CompositeModule<ParamsType> >& nn_module, size_t num_samples_in_buffer = 1000);
	
	std::vector<ParamsType>& GetParameters()
	{
		parameters_.clear();
		nn_module_->GetParameters(parameters_);
		return parameters_;
	}

	std::shared_ptr<IOTreeNode> GetState() const
	{
		std::shared_ptr<IOTreeNode> node( new IOTreeNode() );
		node->attributes().AppendEntry( "Category", "NN" );
		node->attributes().AppendEntry( "Type", "NN1" );
		node->attributes().AppendEntry( "minibatch_size", std::to_string(num_samples_in_buffer_) );
		node->nodes().AppendEntry( "nn_module", nn_module_->GetState() );
		return node;
	}

	static std::shared_ptr< NN< ParamsType> > Create(IOTreeNode& node)
	{
		assert( node.attributes().GetEntry( "Category" ) == "NN" );
		assert( node.attributes().GetEntry( "Type" ) == "NN1" );
		size_t num_samples_in_buffer = Converter::ConvertTo<size_t>( node.attributes().GetEntry( "minibatch_size" ) );
		std::shared_ptr< CompositeModule<ParamsType> > nn_module = 
			std::static_pointer_cast< CompositeModule<ParamsType> >(ModuleFactory::GetModule<ParamsType>( *node.nodes().GetEntry( "nn_module" ) ));
		return std::shared_ptr< NN< ParamsType> >( new NN< ParamsType>(nn_module, num_samples_in_buffer) );
	}

	bool Equals( NN< ParamsType>& nn)
	{
		if (nn.GetMinibatchSize() != GetMinibatchSize())
			return false;

		if ( !nn_module_->Equals(*nn.nn_module_) )
			return false;

		return true;
	}

};

template <class ParamsType>
std::pair<double,double> NN<ParamsType>::GetCost_(ITrainDataset<ParamsType>& dataset, CostModule<ParamsType>& cost_module, std::vector<size_t>& indices, 
												  bool train_mode, bool with_bprop, bool with_regularization, double cost_module_lambda)
{
	assert( train_mode || !with_bprop ); // cannot bprop in predict mode

	std::vector<size_t> batch_sizes = GetBatchSizes(indices.size(), num_samples_in_buffer_);
	size_t num_batches = batch_sizes.size();
	double weighted_num_samples = 0;
	double cost = 0;
	size_t offset = 0;
	for (size_t batch_ind = 0; batch_ind<num_batches; batch_ind++)
	{
		std::vector<size_t> batch_indices(batch_sizes[batch_ind]);
		for (size_t i=0; i<batch_sizes[batch_ind]; i++)
			batch_indices[i]=indices[offset+i];

		std::shared_ptr< Tensor<ParamsType> > input = dataset.GetInput(batch_indices);
		std::shared_ptr< Tensor<ParamsType> > expected_output = dataset.GetOutput(batch_indices);
		std::vector<ParamsType> importance = dataset.GetImportance(batch_indices);

		weighted_num_samples += std::accumulate(importance.begin(),importance.end(),0);
		std::shared_ptr< Tensor<ParamsType> > output = ( train_mode ? nn_module_->train_fprop(input) : nn_module_->predict_fprop(input));

		cost += cost_module_lambda*cost_module.GetCost(*output, *expected_output, importance, false,1);
		if (with_regularization)
			cost+=nn_module_->GetCost(importance);
		if (with_bprop)
		{
			std::shared_ptr< Tensor<ParamsType> > gradient_buffer = cost_module.bprop(*output, *expected_output, importance, false,cost_module_lambda);
			nn_module_->bprop(gradient_buffer, importance);
			if ( batch_ind == 0 )
			{
				gradients_.clear();
				gradients_.reserve(nn_module_->GetNumParams());
				nn_module_->GetGradients(gradients_);
			}
			else
			{
				batch_gradients_.clear();
				nn_module_->GetGradients(batch_gradients_);
				gradients_.resize(batch_gradients_.size());
				axpy<ParamsType>(batch_gradients_.data(), gradients_.data(), batch_gradients_.size(), 1);
			}
		}
		offset += batch_sizes[batch_ind];
	}
	
	if (with_bprop)
	{
		ParamsType normalizer = static_cast<ParamsType>(1/weighted_num_samples);
		scale<ParamsType>(gradients_.data(), gradients_.size(), normalizer);
	}

	return std::make_pair(cost / weighted_num_samples, weighted_num_samples);
}

template <class ParamsType>
std::vector< std::shared_ptr< Tensor<ParamsType> > > NN<ParamsType>::Predict(
	ITensorDataLoader<ParamsType>& loader, std::vector<size_t>& indices, std::string output_module_name)
{
	
	if ( output_module_name == "" )
		output_module_name = nn_module_->GetModule( nn_module_->NumModules()-1 )->GetName();
	std::shared_ptr< Module<ParamsType> > output_module = nn_module_->GetModule(output_module_name);

	if (indices.size() == 0)
	{
		size_t test_numel = loader.GetNumSamples();
		for (size_t i=0; i<test_numel; i++)
			indices.push_back(i);
	}
	std::vector< std::shared_ptr< Tensor<ParamsType> > > output(indices.size());

	std::vector<size_t> batch_sizes = GetBatchSizes(indices.size(), num_samples_in_buffer_);
	size_t num_batches = batch_sizes.size();
	double weighted_num_samples = 0;
	double cost = 0;
	size_t offset = 0;
	for (size_t batch_ind = 0; batch_ind<num_batches; batch_ind++)
	{
		std::vector<size_t> batch_indices(batch_sizes[batch_ind]);
		for (size_t i=0; i<batch_sizes[batch_ind]; i++)
			batch_indices[i]=indices[offset+i];

		std::shared_ptr< Tensor<ParamsType> > input = loader.GetData(batch_indices);
		nn_module_->predict_fprop(input);
			
		std::shared_ptr< Tensor<ParamsType> > batch_output = output_module->GetOutputBuffer();

		std::vector<size_t> per_case_output_dims = batch_output->GetDimensions();
		per_case_output_dims.pop_back();
		size_t per_case_num_elements = batch_output->Numel() / batch_sizes[batch_ind];
		for (size_t i=0; i<batch_sizes[batch_ind]; i++)
		{
			std::shared_ptr< Tensor<ParamsType> > sample_output( new Tensor<ParamsType>(per_case_output_dims));
			for (size_t feature_ind = 0; feature_ind<per_case_num_elements; feature_ind++)
				(*sample_output)[feature_ind] = (*batch_output)[i*per_case_num_elements+feature_ind];
			output[offset+i] = sample_output;
		}

		offset += batch_sizes[batch_ind];
	}

	return output;
}

template <class ParamsType>
double NN<ParamsType>::GetCost(ITrainDataset<ParamsType>& dataset, CostModule<ParamsType>& cost_module, 
							   std::vector<size_t>& indices, bool train_mode, bool with_regularization, double cost_module_lambda)
{
	if (indices.size() == 0)
	{
		size_t test_numel = dataset.GetNumSamples();
		for (size_t i=0; i<test_numel; i++)
			indices.push_back(i);
	}

	auto res = GetCost_(dataset, cost_module, indices, train_mode, false, with_regularization, cost_module_lambda);
	return res.first;
}

template <class ParamsType>
CostAndGradients<ParamsType> NN<ParamsType>::GetGradientsAndCost(ITrainDataset<ParamsType>& dataset,  CostModule<ParamsType>& cost_module, 
	std::vector<size_t>& indices, bool with_regularization, double cost_module_lambda)
{
	if (indices.size() == 0)
	{
		size_t train_numel = dataset.GetNumSamples();
		for (size_t i=0; i<train_numel; i++)
			indices.push_back(i);
	}

	auto res = GetCost_(dataset, cost_module, indices, true, true, with_regularization, cost_module_lambda);
	double cost = res.first;
		
	return CostAndGradients<ParamsType>(cost, gradients_);
}

template <class ParamsType>
void NN<ParamsType>::SetParameters(const std::vector<ParamsType>& parameters)
{
	assert(parameters.size() == nn_module_->GetNumParams());
	nn_module_->SetParameters(parameters);
}

template <class ParamsType>
void NN<ParamsType>::SetParameters(const ParamsType* parameters)
{
	nn_module_->SetParameters(parameters);
}

template <class ParamsType>
NN<ParamsType>::NN(std::shared_ptr< CompositeModule<ParamsType> >& nn_module, size_t num_samples_in_buffer) : 
	nn_module_(nn_module), num_samples_in_buffer_(num_samples_in_buffer)
{
}

#endif