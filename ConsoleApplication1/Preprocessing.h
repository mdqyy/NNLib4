#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <algorithm>
#include <numeric>
#include <math.h>
#include "Utilities.h"
#include "Tensor.h"
#include "exports.h"

template <class T>
void RandomShuffleVectors(std::vector<T>& v1, std::vector<T>& v2)
{
	std::vector<size_t> vectors_inds(v1.size());
	for ( size_t i=0; i<vectors_inds.size(); i++)
		vectors_inds[i] = i;

	std::random_shuffle ( vectors_inds.begin(), vectors_inds.end() );
	std::vector<T> out_v1(v1.size());
	std::vector<T> out_v2(v1.size());
	for ( size_t i=0; i<vectors_inds.size(); i++)
	{
		out_v1[i] = v1[vectors_inds[i]];
		out_v2[i] = v2[vectors_inds[i]];
	}

	v1 = out_v1;
	v2 = out_v2;
}

template <class T>
void RandomShuffleVector(std::vector<T>& v1)
{
	std::vector<size_t> vectors_inds(v1.size());
	std::random_shuffle ( v1.begin(), v1.end() );
}

//returns the mean
template <class T>
void SubtractMean(std::vector< std::shared_ptr< Tensor<T> > >& tensors, double mean)
{
	for (size_t i=0; i<tensors.size(); i++)
	{
		auto& tensor = *tensors[i];
		for (size_t j=0; j<tensor.Numel(); j++)
			(*tensors[i])[j] -= mean;
	}
}

// the last dim is sample index, it is always averaged
//returns the mean
template <class T>
double GetSameMean(std::vector< std::shared_ptr< Tensor<T> > >& tensors)
{
	size_t num_samples = tensors.size();
	size_t num_entries = 0;
	double sum = 0;
	for (size_t i=0; i<num_samples; i++)
	{
		sum += std::accumulate(tensors[i]->GetStartPtr(), tensors[i]->GetStartPtr()+tensors[i]->Numel(), 0.0);
		num_entries += tensors[i]->Numel();
	}
	return sum / num_entries;
}

template <class T>
void DivideByStd(std::vector< std::shared_ptr< Tensor<T> > >& tensors, double stdev)
{
	for (size_t i=0; i<tensors.size(); i++)
	{
		auto& tensor = *tensors[i];
		for (size_t j=0; j<tensor.Numel(); j++)
			tensor[j] /= stdev;
	}
}

struct variance_accumulator
{
	double operator()(double sum, double val)
	{
		return sum+val*val;
	}
};

template <class T>
double GetSameStd(std::vector< std::shared_ptr< Tensor<T> > >& tensors)
{
	size_t num_samples = tensors.size();
	size_t num_entries = 0;
	double sum = 0;
	for (size_t i=0; i<num_samples; i++)
	{
		sum += std::accumulate(tensors[i]->GetStartPtr(), tensors[i]->GetStartPtr()+tensors[i]->Numel(), 0.0, variance_accumulator());
		num_entries += tensors[i]->Numel();
	}
	double stdev = std::sqrt(sum/num_entries)+0.000000001;
	return stdev;
}

template <class T>
void FullMeanSubtract(std::vector< std::shared_ptr< Tensor<T> > >& tensors, Tensor<T>& means)
{
	size_t num_samples = tensors.size();
	size_t num_features = tensors[0]->Numel();
	for (size_t sample_ind=0; sample_ind<num_samples; sample_ind++)
	{
		auto& tensor = (*tensors[sample_ind]);
		for (size_t i=0; i<num_features; i++)
			tensor[i] -= means[i];
	}
}

// the last dim is sample index, it is always averaged
//returns the tensor of means
template <class T>
Tensor<T> GetFullMeans(std::vector< std::shared_ptr< Tensor<T> > >& tensors)
{
	Tensor<double> accumulated_means(tensors[0]->GetDimensions());
	size_t num_samples = tensors.size();
	size_t num_features = tensors[0]->Numel();
	for (size_t sample_ind=0; sample_ind<num_samples; sample_ind++)
	{
		auto& tensor = (*tensors[sample_ind]);
		for (size_t i=0; i<num_features; i++)
			accumulated_means[i] += tensor[i];
	}

	for (size_t i=0; i<accumulated_means.Numel(); i++)
		accumulated_means[i] /= num_samples;
	
	Tensor<T> means(tensors[0]->GetDimensions());
	for (size_t i=0; i< means.Numel(); i++)
		means[i] = static_cast<T>(accumulated_means[i]);

	return means;
}

template <class T>
void FullStdDivide(std::vector< std::shared_ptr< Tensor<T> > >& tensors, Tensor<T>& stds)
{
	size_t num_samples = tensors.size();
	size_t num_features = tensors[0]->Numel();
	for (size_t sample_ind=0; sample_ind<num_samples; sample_ind++)
	{
		auto& tensor = (*tensors[sample_ind]);
		for (size_t i=0; i<num_features; i++)
			tensor[i] /= stds[i];
	}
}

template <class T>
Tensor<T> GetFullStd(std::vector< std::shared_ptr< Tensor<T> > >& tensors)
{
	Tensor<double> accummulated_variances(tensors[0]->GetDimensions());
	size_t num_samples = tensors.size();
	size_t num_features = tensors[0]->Numel();
	for (size_t sample_ind=0; sample_ind<num_samples; sample_ind++)
	{
		auto& tensor = (*tensors[sample_ind]);
		for (size_t i=0; i<num_features; i++)
			accummulated_variances[i] += tensor[i]*tensor[i];
	}
	
	Tensor<T> stds(tensors[0]->GetDimensions());
	for (size_t i=0; i< accummulated_variances.Numel(); i++)
		stds[i] = static_cast<T>( std::sqrt(accummulated_variances[i] / num_samples ) + 0.0000000001 );

	return stds;
}

#endif