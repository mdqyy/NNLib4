#ifndef REGULARIZER_H
#define REGULARIZER_H

#include <vector>
#include <numeric>
#include "Tensor.h"
#include "Converter.h"
#include "IOTreeNode.h"

template <class T>
class Regularizer
{
protected:

	double base_multiplier_;
	virtual void sub_GetState(IOTreeNode& node) const = 0;
	static double GetBaseMultiplier(IOTreeNode& node)
	{
		return Converter::ConvertTo<double>( node.attributes().GetEntry( "base_multiplier" ) );
	}

public:

	Regularizer(double multiplier = 0) : base_multiplier_(multiplier){}

	virtual ~Regularizer() {};
	
	
	std::shared_ptr<IOTreeNode> GetState() const
	{
		std::shared_ptr<IOTreeNode> node( new IOTreeNode() );
		node->attributes().AppendEntry( "Category", "Regularizer" );
		node->attributes().AppendEntry( "Type", GetType() );
		node->attributes().AppendEntry( "base_multiplier", std::to_string(base_multiplier_) );
		sub_GetState( *node );
		return node;
	}

	virtual double GetCost(const Tensor<T>& data, double multiplier) = 0;
	virtual void GetGradients(const Tensor<T>& data, Tensor<T>& gradients, double multiplier) = 0;

	virtual std::string GetType() const = 0;
	virtual bool Equals(const Regularizer<T>& regularizer) const
	{
		if (base_multiplier_ != regularizer.base_multiplier_)
			return false;
		return regularizer.GetType() == GetType();
	}
};

#endif