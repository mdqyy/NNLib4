#ifndef UNIFORM_INITIALIZER_H
#define UNIFORM_INITIALIZER_H

#include "ParametersInitializer.h"
#include "RandomGenerator.h"
#include "Converter.h"

template <class T>
class UniformInitializer: public ParametersInitializer<T>
{
	double min_value_;
	double max_value_;

	virtual void sub_GetState(IOTreeNode& node) const
	{
		node.attributes().AppendEntry( "min_value", std::to_string(min_value_) );
		node.attributes().AppendEntry( "max_value", std::to_string(max_value_) );
	}
public:

	UniformInitializer(double min_value = -1, double max_value = 1) : ParametersInitializer(), min_value_(min_value), max_value_(max_value)
	{
	}

	virtual void InitializeParameters(Tensor<T>& params_tensor) 
	{
		for (size_t i = 0; i<params_tensor.Numel(); i++)
			params_tensor[i] = static_cast<T>(RandomGenerator::GetUniformDouble(min_value_, max_value_));
	}

	static std::shared_ptr< ParametersInitializer< T> > Create(IOTreeNode& data)
	{
		T min_value = Converter::ConvertTo<T>( data.attributes().GetEntry( "min_value" ) );
		T max_value = Converter::ConvertTo<T>( data.attributes().GetEntry( "max_value" ) );
		return std::shared_ptr< ParametersInitializer< T> >( new UniformInitializer(min_value, max_value) );
	}

	virtual std::string GetType() const
	{
		return "UniformInitializer";
	}

	virtual bool Equals(const ParametersInitializer<T>& initializer) const;
};

template <class T>
bool UniformInitializer<T>::Equals(const ParametersInitializer<T>& initializer) const
{
	if (initializer.GetType() != GetType())
		return false;
	
	const UniformInitializer<T>* other_initializer = static_cast< const UniformInitializer<T>* >( &initializer );
	if ( min_value_ != other_initializer->min_value_ )
		return false;
	if ( max_value_ != other_initializer->max_value_ )
		return false;
	return true;
}

#endif