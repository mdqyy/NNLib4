#ifndef BRANCH_SEMISUPERVISED_NN_H
#define BRANCH_SEMISUPERVISED_NN_H

#include <numeric>
#include <vector>
#include <memory>
#include <algorithm>
#include "CompositeModule.h"
#include "CostModule.h"
#include "TrainDataset.h"
#include "IOTreeNode.h"
#include "Converter.h"
#include "ModuleFactory.h"
#include "CostAndGradients.h"
#include "BranchModule.h"

template <class T>
class BranchCostModuleInfo
{
public:
	std::shared_ptr< CostModule<T> > cost_module;
	std::string branch_module_name;
	double lambda;

	BranchCostModuleInfo(std::shared_ptr< CostModule<T> > cost_module, std::string branch_module_name, double lambda) :
		cost_module(cost_module), branch_module_name(branch_module_name), lambda(lambda)
	{
	}
};

class BranchCostResult
{
public:
	double main_cost;
	std::vector<double> branch_costs;

	BranchCostResult() : main_cost(0), branch_costs( std::vector<double>() )
	{
	}
};

template <class ParamsType>
class BranchSemisupervisedNN
{
private:
	size_t num_samples_in_buffer_;
	std::shared_ptr< CompositeModule<ParamsType> > nn_module_;
	std::vector<ParamsType> batch_gradients_;
	std::vector<ParamsType> gradients_;
	std::vector<ParamsType> parameters_;

	std::vector<size_t> GetBatches(size_t num_elements, size_t batch_size);
	
	BranchCostResult GetCost_(ITrainDataset<ParamsType>& dataset,
		CostModule<ParamsType>& main_cost_module, double main_lambda, std::vector< BranchCostModuleInfo<ParamsType> >& branch_cost_modules, 
		std::vector<size_t>& indices, bool train_mode, bool with_bprop, bool with_regularization);

public:

	virtual std::vector< std::shared_ptr< Tensor<ParamsType> > > Predict(
		ITensorDataLoader<ParamsType>& loader, std::vector<size_t>& indices = std::vector<size_t>(), std::string output_module_name = "");

	BranchCostResult GetCost(ITrainDataset<ParamsType>& dataset,  
		CostModule<ParamsType>& main_cost_module, double main_lambda, std::vector< BranchCostModuleInfo<ParamsType> >& branch_cost_modules, 
		std::vector<size_t>& indices, bool train_mode, bool with_regularization = false);

	void SetMinibatchSize(size_t new_size)
	{
		num_samples_in_buffer_ = new_size;
	}
	
	size_t GetMinibatchSize()
	{
		return num_samples_in_buffer_;
	}

	std::pair< CostAndGradients<ParamsType>, BranchCostResult>  GetGradientsAndCost(ITrainDataset<ParamsType>& dataset,  
		CostModule<ParamsType>& main_cost_module, double main_lambda, std::vector< BranchCostModuleInfo<ParamsType> >& branch_cost_modules,
		std::vector<size_t>& indices = std::vector<size_t>(), bool with_regularization = false);

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

	BranchSemisupervisedNN(std::shared_ptr< CompositeModule<ParamsType> >& nn_module, size_t num_samples_in_buffer = 1000);
	
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
		node->attributes().AppendEntry( "Type", "BranchSemisupervisedNN" );
		node->attributes().AppendEntry( "minibatch_size", std::to_string(num_samples_in_buffer_) );
		node->nodes().AppendEntry( "nn_module", nn_module_->GetState() );
		return node;
	}

	static std::shared_ptr< BranchSemisupervisedNN< ParamsType> > Create(IOTreeNode& node)
	{
		assert( node.attributes().GetEntry( "Category" ) == "NN" );
		assert( node.attributes().GetEntry( "Type" ) == "BranchSemisupervisedNN" );
		size_t num_samples_in_buffer = Converter::ConvertTo<size_t>( node.attributes().GetEntry( "minibatch_size" ) );
		std::shared_ptr< CompositeModule<ParamsType> > nn_module = 
			std::static_pointer_cast< CompositeModule<ParamsType> >(ModuleFactory::GetModule<ParamsType>( *node.nodes().GetEntry( "nn_module" ) ));
		return std::shared_ptr< BranchSemisupervisedNN< ParamsType> >( new BranchSemisupervisedNN< ParamsType>(nn_module, num_samples_in_buffer) );
	}

	bool Equals( BranchSemisupervisedNN< ParamsType>& nn)
	{
		if (nn.GetMinibatchSize() != GetMinibatchSize())
			return false;

		if ( !nn_module_->Equals(*nn.nn_module_) )
			return false;

		return true;
	}
};

template <class ParamsType>
BranchCostResult BranchSemisupervisedNN<ParamsType>::GetCost_(ITrainDataset<ParamsType>& dataset,
	CostModule<ParamsType>& main_cost_module, double main_lambda, std::vector< BranchCostModuleInfo<ParamsType> >& branch_cost_modules, 
	std::vector<size_t>& indices, bool train_mode, bool with_bprop, bool with_regularization)
{
	assert( train_mode || !with_bprop ); // cannot bprop in predict mode
	if (with_bprop)
		gradients_.clear();

	BranchCostResult cost_res;
	cost_res.branch_costs = std::vector<double>(branch_cost_modules.size(), 0);
	cost_res.main_cost = 0;

	std::vector<size_t> batch_sizes = GetBatches(indices.size(), num_samples_in_buffer_);
	size_t num_batches = batch_sizes.size();
	double weighted_num_samples = 0;
	size_t offset = 0;
	for (size_t batch_ind = 0; batch_ind<num_batches; batch_ind++)
	{
		std::vector<size_t> batch_indices(batch_sizes[batch_ind]);
		for (size_t i=0; i<batch_sizes[batch_ind]; i++)
			batch_indices[i]=indices[offset+i];

		std::shared_ptr< Tensor<ParamsType> > input = dataset.GetInput(batch_indices);
		std::shared_ptr< Tensor<ParamsType> > output_labels = dataset.GetOutput(batch_indices);
		std::vector<ParamsType> importance = dataset.GetImportance(batch_indices);

		weighted_num_samples += std::accumulate(importance.begin(),importance.end(),0);
		std::shared_ptr< Tensor<ParamsType> > main_net_output = ( train_mode ? nn_module_->train_fprop(input) : nn_module_->predict_fprop(input));

		cost_res.main_cost += main_cost_module.GetCost(*main_net_output, *input, importance, false, main_lambda);
		for (size_t i=0; i<branch_cost_modules.size(); i++)
		{
			BranchModule<ParamsType>& branch_module = *static_cast<BranchModule<ParamsType>*>(&*nn_module_->GetModule(branch_cost_modules[i].branch_module_name));
			Tensor<ParamsType>& branch_module_output = *branch_module.GetBranchModuleOutputBuffer();
			double branch_batch_cost = branch_cost_modules[i].cost_module->GetCost(branch_module_output, *output_labels, importance, false, 1);
			cost_res.main_cost += branch_cost_modules[i].lambda*branch_batch_cost;
			cost_res.branch_costs[i] += branch_batch_cost;
		}

		if (with_regularization)
			cost_res.main_cost+=nn_module_->GetCost(importance);
		if (with_bprop)
		{
			std::shared_ptr< Tensor<ParamsType> > gradient_buffer = main_cost_module.bprop(*main_net_output, *input, importance, false, main_lambda);
			for (size_t i=0; i<branch_cost_modules.size(); i++)
			{
				BranchModule<ParamsType>& branch_module = *static_cast<BranchModule<ParamsType>*>(&*nn_module_->GetModule(branch_cost_modules[i].branch_module_name));
				Tensor<ParamsType>& branch_module_output = *branch_module.GetBranchModuleOutputBuffer();
				std::shared_ptr< Tensor<ParamsType> > branch_gradient_buffer = branch_cost_modules[i].cost_module->bprop(
					*branch_module.GetBranchModuleOutputBuffer(), *output_labels, importance, false, branch_cost_modules[i].lambda);
				branch_module.PushBranchGradients(branch_gradient_buffer);
			}

			nn_module_->bprop(gradient_buffer, importance);
			batch_gradients_.clear();
			nn_module_->GetGradients(batch_gradients_);
			gradients_.resize(batch_gradients_.size());
			std::transform(batch_gradients_.begin(), batch_gradients_.end(), gradients_.begin(), gradients_.begin(), std::plus<double>());
		}
		offset += batch_sizes[batch_ind];
	}
		
	if (with_bprop)
		for (size_t i=0; i<gradients_.size(); i++)
			gradients_[i] /= static_cast<ParamsType>(weighted_num_samples);

	for (size_t i=0; i<branch_cost_modules.size(); i++)
		cost_res.branch_costs[i] /= weighted_num_samples;

	cost_res.main_cost /= weighted_num_samples;

	return cost_res;
}

template <class ParamsType>
std::vector< std::shared_ptr< Tensor<ParamsType> > > BranchSemisupervisedNN<ParamsType>::Predict(
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

	std::vector<size_t> batch_sizes = GetBatches(indices.size(), num_samples_in_buffer_);
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
BranchCostResult BranchSemisupervisedNN<ParamsType>::GetCost(ITrainDataset<ParamsType>& dataset,  
	CostModule<ParamsType>& main_cost_module, double main_lambda, std::vector< BranchCostModuleInfo<ParamsType> >& branch_cost_modules, 
	std::vector<size_t>& indices, bool train_mode, bool with_regularization)
{
	if (indices.size() == 0)
	{
		size_t test_numel = dataset.GetNumSamples();
		for (size_t i=0; i<test_numel; i++)
			indices.push_back(i);
	}

	return GetCost_(dataset, main_cost_module, main_lambda, branch_cost_modules, indices, train_mode, false, with_regularization);
}

template <class ParamsType>
std::pair< CostAndGradients<ParamsType>, BranchCostResult> BranchSemisupervisedNN<ParamsType>::GetGradientsAndCost(ITrainDataset<ParamsType>& dataset,  
	CostModule<ParamsType>& main_cost_module, double main_lambda, std::vector< BranchCostModuleInfo<ParamsType> >& branch_cost_modules,
	std::vector<size_t>& indices, bool with_regularization)
{
	if (indices.size() == 0)
	{
		size_t train_numel = dataset.GetNumSamples();
		for (size_t i=0; i<train_numel; i++)
			indices.push_back(i);
	}

	auto res = GetCost_(dataset, main_cost_module, main_lambda, branch_cost_modules, indices, true, true, with_regularization);
		
	return std::make_pair( CostAndGradients<ParamsType>(res.main_cost, gradients_), res );
}

template <class ParamsType>
void BranchSemisupervisedNN<ParamsType>::SetParameters(const std::vector<ParamsType>& parameters)
{
	assert(parameters.size() == nn_module_->GetNumParams());
	parameters_ = parameters;
	nn_module_->SetParameters(parameters_);
}

template <class ParamsType>
void BranchSemisupervisedNN<ParamsType>::SetParameters(const ParamsType* parameters)
{
	parameters_.clear();
	size_t num_params = nn_module_->GetNumParams();
	for (size_t i=0; i<num_params; i++)
		parameters_.push_back(parameters[i]);

	nn_module_->SetParameters(parameters);
}

template <class ParamsType>
std::vector<size_t> BranchSemisupervisedNN<ParamsType>::GetBatches(size_t num_elements, size_t batch_size)
{
	size_t num_batches = num_elements / batch_size;
	size_t last_batch_size = num_elements - num_batches*batch_size;
	std::vector<size_t> batch_sizes(num_batches+1);
	for (size_t i=0; i<batch_sizes.size(); i++)
		batch_sizes[i] = num_samples_in_buffer_;
	if(last_batch_size == 0)
		batch_sizes.pop_back();
	else
		batch_sizes[batch_sizes.size()-1] = last_batch_size;
	return batch_sizes;
}

template <class ParamsType>
BranchSemisupervisedNN<ParamsType>::BranchSemisupervisedNN(std::shared_ptr< CompositeModule<ParamsType> >& nn_module, size_t num_samples_in_buffer) : 
	nn_module_(nn_module), num_samples_in_buffer_(num_samples_in_buffer)
{
}

#endif