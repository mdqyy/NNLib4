#ifndef EMPTY_REGULARIZER_H
#define EMPTY_REGULARIZER_H

#include "Regularizer.h"

template <class T>
class EmptyRegularizer : public Regularizer<T>
{
public:
	EmptyRegularizer(double base_multiplier = 0) : Regularizer(0){}
	
	virtual ~EmptyRegularizer() {};
	
	virtual double GetCost(const Tensor<T>& data, double multiplier = 1)
	{
		return 0;
	}
	
	virtual void GetGradients(const Tensor<T>& data, Tensor<T>& gradients, double multiplier = 1) {}

	virtual void sub_GetState(IOTreeNode& node) const
	{
	}

	static std::shared_ptr< Regularizer< T> > Create(IOTreeNode& data)
	{
		double base_multiplier = Regularizer<T>::GetBaseMultiplier(data);
		return std::shared_ptr< Regularizer< T> >( new EmptyRegularizer(base_multiplier) );
	}

	virtual std::string GetType() const
	{
		return "EmptyRegularizer";
	}
};

#endif