#ifndef ABS_REGULARIZER
#define ABS_REGULARIZER

#include "Regularizer.h"
#include "my_math.h"

template <class T>
class AbsRegularizer : public Regularizer<T>
{
	virtual void sub_GetState(IOTreeNode& node) const
	{
	}

public:
	AbsRegularizer(double base_multiplier = 0) : Regularizer(base_multiplier){}
	
	virtual ~AbsRegularizer() {};
	
	virtual double GetCost(const Tensor<T>& data, double multiplier = 1)
	{
		T cost = 0;
		size_t numel = data.Numel();
		for (size_t i=0; i<numel; i++)
			cost += abs(data[i]);
		return multiplier*base_multiplier_*cost;
	}

	void GetGradients(const Tensor<T>& data, Tensor<T>& gradients, double multiplier = 1)
	{
		T coeff = static_cast<T>(multiplier*base_multiplier_);
		size_t numel = data.Numel();
		for (size_t i=0; i<numel; i++)
			gradients[i] += coeff*sign(data[i]);
	}

	static std::shared_ptr< Regularizer< T> > Create(IOTreeNode& data)
	{
		double base_multiplier = Regularizer<T>::GetBaseMultiplier(data);
		return std::shared_ptr< Regularizer<T> >( new AbsRegularizer(base_multiplier) );
	}

	virtual std::string GetType() const
	{
		return "AbsRegularizer";
	}
};

#endif