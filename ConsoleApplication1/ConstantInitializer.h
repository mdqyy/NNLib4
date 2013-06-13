#ifndef CONSTANT_INITIALIZER_H
#define CONSTANT_INITIALIZER_H

#include "ParametersInitializer.h"
#include "Converter.h"

template <class T>
class ConstantInitializer: public ParametersInitializer<T>
{
	T value_;

	virtual void sub_GetState(IOTreeNode& node) const
	{
		node.attributes().AppendEntry( "value", std::to_string(value_) );
	}

public:
	ConstantInitializer(T value) : ParametersInitializer(), value_(value)
	{
	}

	virtual void InitializeParameters(Tensor<T>& params_tensor)  
	{
		for (size_t i = 0; i<params_tensor.Numel(); i++)
			params_tensor[i] = value_;
	}

	static std::shared_ptr< ParametersInitializer< T> > Create(IOTreeNode& data)
	{
		T value = Converter::ConvertTo<T>( data.attributes().GetEntry( "value" ) );
		return std::shared_ptr< ParametersInitializer< T> >( new ConstantInitializer(value) );
	}

	virtual std::string GetType() const
	{
		return "ConstantInitializer";
	}

	virtual bool Equals(const ParametersInitializer<T>& initializer) const;
};

template <class T>
bool ConstantInitializer<T>::Equals(const ParametersInitializer<T>& initializer) const
{
	if (initializer.GetType() != GetType())
		return false;
	
	const ConstantInitializer<T>* other_initializer = static_cast< const ConstantInitializer<T>* >( &initializer );
	if ( value_ != other_initializer->value_ )
		return false;
	return true;
}

#endif