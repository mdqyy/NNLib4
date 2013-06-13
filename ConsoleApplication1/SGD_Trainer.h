#ifndef SGD_TRAINER_H
#define SGD_TRAINER_H

#include <functional>
#include "Trainer.h"
#include "MatrixOperations.h"

template <class ParamsType>
class SGD_Trainer : public Trainer<ParamsType>
{
	size_t num_iterations_;
	ParamsType learning_rate_;
	ParamsType momentum_;
	size_t train_batch_size_;
	size_t validation_batch_size_;
	double train_decay_;
	double validation_decay_;
	size_t num_batches_before_validation_evaluation_;
	size_t num_batches_before_train_evaluation_;
	size_t num_warmup_batches_;
	ParamsType warmup_momentum_;

	ParamsType GetMomentum(size_t batch_ind)
	{
		if (warmup_momentum_<momentum_)
			return momentum_;
		if (batch_ind<num_warmup_batches_)
			return warmup_momentum_;
		return momentum_;
	}

	bool IsValidationResultBatch(size_t batch_ind)
	{
		if (batch_ind%num_batches_before_validation_evaluation_==0)
			return true;
		return false;
	}


	bool IsTrainResultBatch(size_t batch_ind)
	{
		if (batch_ind%num_batches_before_train_evaluation_==0)
			return true;
		return false;
	}

	void UpdateSpeed(std::vector<ParamsType>& speed, ParamsType* gradients, ParamsType momentum )
	{
		scale(speed.data(), speed.size(), momentum);
		axpy<ParamsType>(gradients, speed.data(), speed.size(), 1);
	}
	
	void Move(ParamsType* parameters, ParamsType* speed, size_t num_params )
	{
		axpy(speed, parameters, num_params, -learning_rate_);
	}

public:

	size_t GetNumIterations(){return num_iterations_;}
	void SetNumIterations(size_t num_iterations){num_iterations_ = num_iterations;}
	
	ParamsType GetLearningRate(){return learning_rate_;}
	void SetLearningRate(ParamsType learning_rate){learning_rate_ = learning_rate;}
	
	ParamsType GetMomentum(){return momentum_;}
	void SetNumIterations(ParamsType momentum){momentum_ = momentum;}

	size_t GetTrainBatchSize(){return train_batch_size_;}
	void SetTrainBatchSize(size_t train_batch_size_){train_batch_size_ = train_batch_size;}
	
	size_t GetValidationBatchSize(){return validation_batch_size_;}
	void SetValidationBatchSize(size_t validation_batch_size){validation_batch_size_ = validation_batch_size;}
	
	double GetTrainDecay(){return train_decay_;}
	void SetTrainDecay(size_t train_decay){train_decay_ = train_decay;}
	
	double GetValidationDecay(){return validation_decay_;}
	void SetValidationDecay(double validation_decay){validation_decay_ = validation_decay;}
	
	size_t GetNumBatchesBeforeValidationEvaluation(){return num_batches_before_validation_evaluation_;}
	void SetNumBatchesBeforeValidationEvaluation(size_t num_batches_before_test_evaluation){num_batches_before_validation_evaluation_ = num_batches_before_test_evaluation;}
	
	size_t GetNumBatchesBeforeTrainEvaluation(){return num_batches_before_train_evaluation_;}
	void SetNumBatchesBeforeTrainEvaluation(size_t num_batches_before_train_evaluation){num_batches_before_train_evaluation_ = num_batches_before_train_evaluation;}
	
	size_t GetNumWarmupBatches(){return num_warmup_batches_;}
	void SetNumWarmupBatches(size_t num_warmup_batches){num_warmup_batches_ = num_warmup_batches;}

	ParamsType GetWarmupMomentum_(){return warmup_momentum_;}
	void SetWarmupMomentum_(ParamsType warmup_momentum){warmup_momentum_ = warmup_momentum;}

	SGD_Trainer(size_t num_iterations=100000, ParamsType learning_rate=0.00001, ParamsType momentum=0, size_t train_batch_size=30,
		size_t validation_batch_size=10000000, double train_decay=0.999, double validation_decay=0, size_t num_batches_before_train_evaluation = 100,
		size_t num_batches_before_validation_evaluation = 100, size_t num_warmup_batches=0, ParamsType warmup_momentum=0);

	// return validation cost
	virtual double Train(NN<ParamsType>& net, 
		CostModule<ParamsType>& train_cost_module, CostModule<ParamsType>& validation_cost_module,
		ITrainDataset<ParamsType>& train_set, ITrainDataset<ParamsType>& validation_set, 
		Trainer<ParamsType>::ProcessTrainResultFunc train_result_processor = DefaultProcessTrainFunc<ParamsType>, 
			Trainer<ParamsType>::ProcessValidationResultFunc validation_result_processor = DefaultProcessValidationFunc<ParamsType>);
};

template <class ParamsType>
SGD_Trainer<ParamsType>::SGD_Trainer(size_t num_iterations, ParamsType learning_rate, ParamsType momentum, size_t train_batch_size,
	size_t validation_batch_size, double train_decay, double validation_decay, size_t num_batches_before_train_evaluation,
	size_t num_batches_before_validation_evaluation, size_t num_warmup_batches, ParamsType warmup_momentum) : 
		num_iterations_(num_iterations), learning_rate_(learning_rate), 
		momentum_(momentum), train_batch_size_(train_batch_size), validation_batch_size_(validation_batch_size), train_decay_(train_decay),
		validation_decay_(validation_decay), num_batches_before_train_evaluation_(num_batches_before_train_evaluation), 
		num_batches_before_validation_evaluation_(num_batches_before_validation_evaluation), num_warmup_batches_(num_warmup_batches), warmup_momentum_(warmup_momentum)
{

}

template <class ParamsType>
double SGD_Trainer<ParamsType>::Train(NN<ParamsType>& net, 
		CostModule<ParamsType>& train_cost_module, CostModule<ParamsType>& validation_cost_module,
		ITrainDataset<ParamsType>& train_set, ITrainDataset<ParamsType>& validation_set, 
		ProcessTrainResultFunc train_result_processor = DefaultProcessTrainFunc<ParamsType>, 
		ProcessValidationResultFunc validation_result_processor = DefaultProcessValidationFunc<ParamsType>)
{
	size_t num_params = net.GetNumParams();
	std::vector<ParamsType> move_speed(num_params);

	size_t num_train_cases = train_set.GetNumSamples();
	size_t num_validation_cases = validation_set.GetNumSamples();

	double best_train_cost = net.GetCost(train_set, train_cost_module, train_set.SelectIndices(5000), true, false);
	double best_validation_cost = net.GetCost(validation_set, validation_cost_module, validation_set.SelectIndices(5000), false, false);
	std::vector<ParamsType> best_parameters = net.GetParameters();
	std::vector<ParamsType> parameters = best_parameters;

	double train_cost = best_train_cost;
	double validation_cost = best_validation_cost;
	for (size_t batch_ind = 1; batch_ind<=num_iterations_; batch_ind++)
	{
		auto batch_indices = train_set.SelectIndices(train_batch_size_);
		auto cost_and_gradient = net.GetGradientsAndCost(train_set, train_cost_module, batch_indices);

		train_cost = train_decay_*train_cost+(1-train_decay_)*cost_and_gradient.cost;

		UpdateSpeed(move_speed, cost_and_gradient.gradients.data(), GetMomentum(batch_ind));
		Move(parameters.data(), move_speed.data(), num_params);
		
		net.SetParameters(parameters);

		if (IsValidationResultBatch(batch_ind))
		{
			auto validation_indices = validation_set.SelectIndices(validation_batch_size_);
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