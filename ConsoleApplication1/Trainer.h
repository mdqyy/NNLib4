#ifndef TRAINER_H
#define TRAINER_H

#include <vector>
#include <stdio.h>
#include <functional>
#include "NN.h"
#include "RandomGenerator.h"
#include "TrainDataset.h"

template <class ParamsType>
struct ValidationCallbackParams
{
	NN<ParamsType>& net;
	bool is_best;
	double train_cost;
	double validation_cost;
	size_t batch_num;

	ValidationCallbackParams(NN<ParamsType>& net, bool is_best, double train_cost, double validation_cost, size_t batch_num) :net(net), is_best(is_best), 
		train_cost(train_cost), validation_cost(validation_cost), batch_num(batch_num)
	{

	}
};

template <class ParamsType>
struct TrainCallbackParams
{
	NN<ParamsType>& net;
	double train_cost;
	size_t batch_num;

	TrainCallbackParams(NN<ParamsType>& net, double train_cost, size_t batch_num) :net(net), train_cost(train_cost), batch_num(batch_num)
	{

	}
};

template <class ParamsType>
void DefaultProcessValidationFunc(ValidationCallbackParams<ParamsType>& result)
{
	if (result.is_best)
		std::cout<<result.batch_num<<" Train cost = "<<result.train_cost<<" validation cost = "<<result.validation_cost<<" BEST"<<std::endl;
	else
		std::cout<<result.batch_num<<" Train cost = "<<result.train_cost<<" validation cost = "<<result.validation_cost<<std::endl;
}

template <class ParamsType>
void DefaultProcessTrainFunc(TrainCallbackParams<ParamsType>& result)
{
	std::cout<<result.batch_num<<" Train cost = "<<result.train_cost<<std::endl;
}

template <class ParamsType>
class Trainer
{
public:
	typedef std::tr1::function<void (TrainCallbackParams<ParamsType>&)> ProcessTrainResultFunc;
	typedef std::tr1::function<void (ValidationCallbackParams<ParamsType>&)> ProcessValidationResultFunc;

	// return validation cost
	virtual double Train(NN<ParamsType>& net, 
		CostModule<ParamsType>& train_cost_module, CostModule<ParamsType>& validation_cost_module,
		ITrainDataset<ParamsType>& train_set, ITrainDataset<ParamsType>& validation_set, 
		ProcessTrainResultFunc train_result_processor = DefaultProcessTrain, 
		ProcessValidationResultFunc validation_result_processor = DefaultProcessValidation) = 0;
};

#endif