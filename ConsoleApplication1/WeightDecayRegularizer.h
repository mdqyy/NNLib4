#ifndef WEIGHT_DECAY_REGULARIZER
#define WEIGHT_DECAY_REGULARIZER

#include "Regularizer.h"
#include "my_math.h"
#include "MatrixOperations.h"

template <class T>
class WeightDecayRegularizer : public Regularizer<T>
{
public:
	WeightDecayRegularizer(double base_multiplier = 0) : Regularizer(base_multiplier){}
	
	virtual ~WeightDecayRegularizer() {};

	virtual double GetCost(const Tensor<T>& data, double multiplier = 1)
	{
		T cost = 0;
		size_t numel = data.Numel();
		for (size_t i=0; i<numel; i++)
			cost += sqr(data[i]);
		return multiplier*base_multiplier_*cost/2;
	}

	virtual void GetGradients(const Tensor<T>& data, Tensor<T>& gradients, double multiplier = 1)
	{
		T coeff = static_cast<T>(multiplier*base_multiplier_);
		axpy(data.GetStartPtr(), gradients.GetStartPtr(), data.Numel(), coeff);
	}

	virtual void sub_GetState(IOTreeNode& node) const
	{
	}

	static std::shared_ptr< Regularizer< T> > Create(IOTreeNode& data)
	{
		double base_multiplier = Regularizer< T>::GetBaseMultiplier(data);
		return std::shared_ptr< Regularizer< T> >( new WeightDecayRegularizer(base_multiplier) );
	}

	virtual std::string GetType() const
	{
		return "WeightDecayRegularizer";
	}
};
#endif