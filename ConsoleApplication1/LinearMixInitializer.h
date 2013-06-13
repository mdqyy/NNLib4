#ifndef LINEARMIX_INITIALIZER_H
#define LINEARMIX_INITIALIZER_H

#include "ParametersInitializer.h"
#include "RandomGenerator.h"
#include "Converter.h"

template <class T>
class LinearMixInitializer: public ParametersInitializer<T>
{
	double multiplier_;

	virtual void sub_GetState(IOTreeNode& node) const
	{
		node.attributes().AppendEntry( "multiplier", std::to_string(multiplier_) );
	}
public:

	LinearMixInitializer(double multiplier = std::sqrt(6)) : ParametersInitializer(), multiplier_(multiplier)
	{
	}

	virtual void InitializeParameters(Tensor<T>& params_tensor) 
	{
		assert(params_tensor.NumDimensions() == 2);
		double coeff = 1.0/std::sqrt(params_tensor.GetDimensionSize(0)+params_tensor.GetDimensionSize(1));
		for (size_t i = 0; i<params_tensor.Numel(); i++)
			params_tensor[i] = static_cast<T>(multiplier_*coeff*RandomGenerator::GetUniformDouble(-1, 1));
	}

	static std::shared_ptr< ParametersInitializer< T> > Create(IOTreeNode& data)
	{
		T multiplier = Converter::ConvertTo<T>( data.attributes().GetEntry( "multiplier" ) );
		return std::shared_ptr< ParametersInitializer< T> >( new LinearMixInitializer(multiplier) );
	}

	virtual std::string GetType() const
	{
		return "LinearMixInitializer";
	}
	
	virtual bool Equals(const ParametersInitializer<T>& initializer) const;
};

template <class T>
bool LinearMixInitializer<T>::Equals(const ParametersInitializer<T>& initializer) const
{
	if (initializer.GetType() != GetType())
		return false;
	
	const LinearMixInitializer<T>* other_initializer = static_cast< const LinearMixInitializer<T>* >( &initializer );
	if ( multiplier_ != other_initializer->multiplier_ )
		return false;
	return true;
}

#endif