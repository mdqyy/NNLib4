#ifndef GAUSSIAN_INITIALIZER_H
#define GAUSSIAN_INITIALIZER_H

#include "ParametersInitializer.h"
#include "RandomGenerator.h"
#include "Converter.h"

template <class T>
class GaussianInitializer: public ParametersInitializer<T>
{
	double mean_;
	double std_;

	virtual void sub_GetState(IOTreeNode& node) const
	{
		node.attributes().AppendEntry( "mean", std::to_string(mean_) );
		node.attributes().AppendEntry( "std", std::to_string(std_) );
	}
public:

	GaussianInitializer(double mean = 0, double std = 1) : ParametersInitializer(), mean_(mean), std_(std)
	{
	}

	virtual void InitializeParameters(Tensor<T>& params_tensor) 
	{
		for (size_t i = 0; i<params_tensor.Numel(); i++)
			params_tensor[i] = static_cast<T>(RandomGenerator::GetNormalDouble(mean_, std_));
	}


	static std::shared_ptr< ParametersInitializer< T> > Create(IOTreeNode& data)
	{
		T mean = Converter::ConvertTo<T>( data.attributes().GetEntry( "mean" ) );
		T std = Converter::ConvertTo<T>( data.attributes().GetEntry( "std" ) );
		return std::shared_ptr< ParametersInitializer< T> >( new GaussianInitializer(mean, std) );
	}

	virtual std::string GetType() const
	{
		return "GaussianInitializer";
	}

	virtual bool Equals(const ParametersInitializer<T>& initializer) const;
};

template <class T>
bool GaussianInitializer<T>::Equals(const ParametersInitializer<T>& initializer) const
{
	if (initializer.GetType() != GetType())
		return false;
	
	const GaussianInitializer<T>* other_initializer = static_cast< const GaussianInitializer<T>* >( &initializer );
	if ( mean_ != other_initializer->mean_ )
		return false;
	if ( std_ != other_initializer->std_ )
		return false;
	return true;
}

#endif