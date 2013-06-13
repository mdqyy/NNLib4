#ifndef TRAIN_DATASET_H
#define TRAIN_DATASET_H

#include <memory>
#include  "ITensorDataLoader.h"

template <class T>
class ITrainDataset
{
public:
	virtual std::shared_ptr< Tensor<T> > GetInput(std::vector<size_t>& samples_inds) = 0;
	
	virtual std::shared_ptr< Tensor<T> > GetOutput(std::vector<size_t>& samples_inds) = 0;
	
	virtual std::vector<T> GetImportance(std::vector<size_t>& samples_inds) = 0;
	
	virtual std::vector<size_t> SelectIndices(size_t num_samples) = 0;

	virtual size_t GetNumSamples() = 0;
};

template <class T>
class TrainDataset : public ITrainDataset<T>
{
	std::shared_ptr< ITensorDataLoader<T> > input_;
	std::shared_ptr< ITensorDataLoader<T> > output_;
	std::vector<T> importance_;

public:

	TrainDataset(std::shared_ptr< ITensorDataLoader<T> >& input, std::shared_ptr< ITensorDataLoader<T> >& output, 
		std::vector<T>& importance) : input_(input), output_(output), importance_(importance)
	{

	}
	
	virtual std::vector<size_t> SelectIndices(size_t num_samples)
	{
		return input_->SelectIndices(num_samples);
	}
	
	virtual std::shared_ptr< Tensor<T> > GetInput(std::vector<size_t>& samples_inds)
	{
		return input_->GetData(samples_inds);
	}
	
	virtual std::shared_ptr< Tensor<T> > GetOutput(std::vector<size_t>& samples_inds)
	{
		return output_->GetData(samples_inds);
	}
	
	virtual std::vector<T> GetImportance(std::vector<size_t>& samples_inds)
	{
		std::vector<T> res(samples_inds.size());
		for (size_t i=0; i< samples_inds.size(); i++)
			res[i] = importance_[samples_inds[i]];
		return res;
	}

	virtual size_t GetNumSamples()
	{
		return input_->GetNumSamples();
	}
};

template <class T>
class TrainDatasetPairsWithBatchIndProblem : public ITrainDataset<T>
{
	std::shared_ptr< ITensorDataLoader<T> > input_;
	std::shared_ptr< ITensorDataLoader<T> > output_;
	std::vector<T> importance_;

public:

	TrainDatasetPairsWithBatchIndProblem(std::shared_ptr< ITensorDataLoader<T> >& input, std::shared_ptr< ITensorDataLoader<T> >& output, 
		std::vector<T>& importance) : input_(input), output_(output), importance_(importance)
	{

	}
	
	virtual std::shared_ptr< Tensor<T> > GetInput(std::vector<size_t>& samples_inds)
	{
		return input_->GetData(samples_inds);
	}
	
	virtual std::shared_ptr< Tensor<T> > GetOutput(std::vector<size_t>& samples_inds)
	{
		return output_->GetData(samples_inds);
	}
	
	virtual std::vector<T> GetImportance(std::vector<size_t>& samples_inds)
	{
		std::vector<T> res;
		res.reserve(samples_inds.size() / 2);
		for (size_t i=0; i< samples_inds.size(); i+=2)
			res.push_back(importance_[samples_inds[i]]);
		return res;
	}

	virtual std::vector<size_t> SelectIndices(size_t num_samples)
	{
		return input_->SelectIndices(num_samples);
	}

	virtual size_t GetNumSamples()
	{
		return input_->GetNumSamples();
	}
};

#endif