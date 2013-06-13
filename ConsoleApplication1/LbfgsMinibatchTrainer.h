#ifndef LBFGS_MINIBATCH_TRAINER_H
#define LBFGS_MINIBATCH_TRAINER_H

#define LBFGS_FLOAT 32
#include "liblbfgs\lbfgs.h"
#include <functional>
#include "Trainer.h"
#include "MatrixOperations.h"

template <class ParamsType>
class LbfgsMinibatchTrainer : public Trainer<ParamsType>
{
	size_t num_iterations_;
	size_t num_iterations_per_update_;
	size_t train_minibatch_size_;
	size_t validation_batch_size_;
	double train_decay_;
	double validation_decay_;
	size_t num_minibatches_before_validation_evaluation_;
	size_t num_minibatches_before_train_evaluation_;

	bool IsValidationResultBatch(size_t batch_ind)
	{
		if (batch_ind%num_minibatches_before_validation_evaluation_==0)
			return true;
		return false;
	}


	bool IsTrainResultBatch(size_t batch_ind)
	{
		if (batch_ind%num_minibatches_before_train_evaluation_==0)
			return true;
		return false;
	}

	double LbfgsUpdateParameters(NN<ParamsType>& net, ITrainDataset<ParamsType>& train_set, 
		CostModule<ParamsType>& train_cost_module, size_t num_iterations, std::vector<size_t>& minibatch_indices, 
		std::vector<ParamsType>& parameters);

	static lbfgsfloatval_t lbfgs_evaluate(
		void *instance,
		const lbfgsfloatval_t *x,
		lbfgsfloatval_t *g,
		const int n,
		const lbfgsfloatval_t step
	);

	static int lbfgs_progress(
		void *instance,
		const lbfgsfloatval_t *x,
		const lbfgsfloatval_t *g,
		const lbfgsfloatval_t fx,
		const lbfgsfloatval_t xnorm,
		const lbfgsfloatval_t gnorm,
		const lbfgsfloatval_t step,
		int n,
		int k,
		int ls
    );

public:

	size_t GetNumIterations(){return num_iterations_;}
	void SetNumIterations(size_t num_iterations){num_iterations_ = num_iterations;}

	size_t GetNumIterationsPerUpdate(){return num_iterations_per_update_;}
	void SetNumIterationsPerUpdate(size_t num_iterations_per_update){num_iterations_per_update_ = num_iterations_per_update;}

	size_t GetTrainMinibatchSize(){return train_minibatch_size_;}
	void SetTrainMinibatchSize(size_t train_batch_size_){train_batch_size_ = train_minibatch_size_;}
	
	size_t GetValidationBatchSize(){return validation_batch_size_;}
	void SetValidationBatchSize(size_t validation_batch_size){validation_batch_size_ = validation_batch_size;}
	
	double GetTrainDecay(){return train_decay_;}
	void SetTrainDecay(size_t train_decay){train_decay_ = train_decay;}
	
	double GetValidationDecay(){return validation_decay_;}
	void SetValidationDecay(double validation_decay){validation_decay_ = validation_decay;}
	
	size_t GetNumMinibatchesBeforeValidationEvaluation(){return num_minibatches_before_validation_evaluation_;}
	void SetNumMinibatchesBeforeValidationEvaluation(size_t num_minibatches_before_validation_evaluation)
	{num_batches_before_validation_evaluation_ = num_minibatches_before_validation_evaluation;}
	
	size_t GetNumBatchesBeforeTrainEvaluation(){return num_batches_before_train_evaluation_;}
	void SetNumBatchesBeforeTrainEvaluation(size_t num_minibatches_before_train_evaluation)
	{num_minibatches_before_train_evaluation_ = num_minibatches_before_train_evaluation;}

	LbfgsMinibatchTrainer<ParamsType>::LbfgsMinibatchTrainer(size_t num_iterations=100000000, size_t num_iterations_per_update = 20, size_t train_minibatch_size=1000,
		size_t validation_batch_size=100000, double train_decay=0.999, double validation_decay=0, size_t num_minibatches_before_train_evaluation = 100,
		size_t num_minibatches_before_validation_evaluation = 100);

	// return validation cost
	virtual double Train(NN<ParamsType>& net, 
		CostModule<ParamsType>& train_cost_module, CostModule<ParamsType>& validation_cost_module,
		ITrainDataset<ParamsType>& train_set, ITrainDataset<ParamsType>& validation_set, 
		Trainer<ParamsType>::ProcessTrainResultFunc train_result_processor = DefaultProcessTrainFunc<ParamsType>, 
			Trainer<ParamsType>::ProcessValidationResultFunc validation_result_processor = DefaultProcessValidationFunc<ParamsType>);
};

template <class ParamsType>
LbfgsMinibatchTrainer<ParamsType>::LbfgsMinibatchTrainer(size_t num_iterations, size_t num_iterations_per_update, size_t train_minibatch_size,
		size_t validation_batch_size, double train_decay, double validation_decay, size_t num_minibatches_before_train_evaluation,
		size_t num_minibatches_before_validation_evaluation) : 
			num_iterations_(num_iterations), num_iterations_per_update_(num_iterations_per_update), train_minibatch_size_(train_minibatch_size), validation_batch_size_(validation_batch_size), 
			train_decay_(train_decay), validation_decay_(validation_decay), num_minibatches_before_train_evaluation_(num_minibatches_before_train_evaluation), 
			num_minibatches_before_validation_evaluation_(num_minibatches_before_validation_evaluation)
{

}
		
template <class ParamsType>
lbfgsfloatval_t LbfgsMinibatchTrainer<ParamsType>::lbfgs_evaluate(
	void *instance,
	const lbfgsfloatval_t *x,
	lbfgsfloatval_t *g,
	const int n,
	const lbfgsfloatval_t step
)
{
	LbfgsTrainInstance<ParamsType>* train_instance = static_cast< LbfgsTrainInstance<ParamsType>* >(instance);
	train_instance->net.SetParameters(x);
	auto cost_and_gradient = train_instance->net.GetGradientsAndCost(train_instance->train_set, 
		train_instance->train_cost_module, train_instance->minibatch_indices);

	std::copy( cost_and_gradient.gradients.data(), cost_and_gradient.gradients.data()+cost_and_gradient.gradients.size(), g);

	return static_cast<lbfgsfloatval_t>(cost_and_gradient.cost);
}

template <class ParamsType>
int LbfgsMinibatchTrainer<ParamsType>::lbfgs_progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
	std::cout<<"Iteration "<<k<<": f(x)="<<fx<<std::endl;
    return 0;
}

template <class T>
class LbfgsTrainInstance
{
public:
	NN<T>& net;
	ITrainDataset<T>& train_set;
	CostModule<T>& train_cost_module;
	std::vector<size_t>& minibatch_indices;

	LbfgsTrainInstance(NN<T>& net, ITrainDataset<T>& train_set, 
		CostModule<T>& train_cost_module, std::vector<size_t>& minibatch_indices) : 
			net(net), train_set(train_set), train_cost_module(train_cost_module), minibatch_indices(minibatch_indices)
	{
	}
};

template <class ParamsType>
double LbfgsMinibatchTrainer<ParamsType>::LbfgsUpdateParameters(NN<ParamsType>& net, ITrainDataset<ParamsType>& train_set, 
		CostModule<ParamsType>& train_cost_module, size_t num_iterations, std::vector<size_t>& minibatch_indices, 
		std::vector<ParamsType>& parameters)
{
	lbfgs_parameter_t param;
	lbfgs_parameter_init(&param);
	param.max_iterations = num_iterations;

	LbfgsTrainInstance<ParamsType> train_instance(net, train_set, train_cost_module, minibatch_indices);
	
	ParamsType optimization_res = 0;
	lbfgs( static_cast<int>(parameters.size()), parameters.data(), &optimization_res, 
		lbfgs_evaluate, lbfgs_progress, &train_instance, &param);

	net.SetParameters(parameters);

	return optimization_res;
}


template <class ParamsType>
double LbfgsMinibatchTrainer<ParamsType>::Train(NN<ParamsType>& net, 
		CostModule<ParamsType>& train_cost_module, CostModule<ParamsType>& validation_cost_module,
		ITrainDataset<ParamsType>& train_set, ITrainDataset<ParamsType>& validation_set, 
		ProcessTrainResultFunc train_result_processor = DefaultProcessTrainFunc<ParamsType>, 
		ProcessValidationResultFunc validation_result_processor = DefaultProcessValidationFunc<ParamsType>)
{
	size_t num_params = net.GetNumParams();

	size_t num_train_cases = train_set.GetNumSamples();
	size_t num_validation_cases = validation_set.GetNumSamples();

	double best_train_cost = net.GetCost(train_set, train_cost_module, std::vector<size_t>(), true, false);
	double best_validation_cost = net.GetCost(validation_set, validation_cost_module, std::vector<size_t>(), false, false);
	std::vector<ParamsType> best_parameters = net.GetParameters();
	std::vector<ParamsType> parameters = best_parameters;

	double train_cost = best_train_cost;
	double validation_cost = best_validation_cost;
	for (size_t batch_ind = 1; batch_ind<=num_iterations_; batch_ind++)
	{
		auto minibatch_indices = GetRandomSamplesInds(num_train_cases, train_minibatch_size_);
		double cost = LbfgsUpdateParameters(net, train_set, 
			train_cost_module, num_iterations_per_update_, minibatch_indices, parameters);

		train_cost = train_decay_*train_cost+(1-train_decay_)*cost;

		if (IsValidationResultBatch(batch_ind))
		{
			auto validation_indices = GetRandomSamplesInds(num_validation_cases, validation_batch_size_);
			validation_cost = validation_decay_*validation_cost+(1-validation_decay_)*
				net.GetCost(validation_set, validation_cost_module, validation_indices, false, false);
				
			bool is_best = false;
			if ( validation_cost<best_validation_cost )
			{
				is_best = true;
				best_parameters = parameters;
				best_validation_cost = validation_cost;
			}
			validation_result_processor( ValidationCallbackParams<ParamsType>(net, is_best, train_cost, validation_cost,batch_ind) );
		}
		else if (IsTrainResultBatch(batch_ind))
			train_result_processor( TrainCallbackParams<ParamsType>(net, train_cost, batch_ind) );
	}
		
	net.SetParameters(best_parameters);

	return net.GetCost(validation_set, validation_cost_module, std::vector<size_t>(), false, false);
}

#endif