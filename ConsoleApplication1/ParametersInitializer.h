#ifndef PARAMETERS_INITIALIZER_H
#define PARAMETERS_INITIALIZER_H

#include <vector>
#include "Tensor.h"
#include "IOTreeNode.h"

template <class T>
class ParametersInitializer
{
	virtual void sub_GetState(IOTreeNode& node) const = 0;
public:
	virtual void InitializeParameters(Tensor<T>& params = Tensor<T>(std::vector<size_t>(), 0)) =0;
	virtual ~ParametersInitializer() {}

	std::shared_ptr<IOTreeNode> GetState() const
	{
		std::shared_ptr<IOTreeNode> node( new IOTreeNode() );
		node->attributes().AppendEntry( "Category", "Initializer" );
		node->attributes().AppendEntry( "Type", GetType() );
		sub_GetState( *node );
		return node;
	}

	virtual std::string GetType() const = 0;
	virtual bool Equals(const ParametersInitializer<T>& initializer) const = 0;
};

#endif