#ifndef REGULARIZER_FACTORY_H
#define REGULARIZER_FACTORY_H

#include "AbsRegularizer.h"
#include "EmptyRegularizer.h"
#include "WeightDecayRegularizer.h"
#include "IOTreeNode.h"

class UnknownRegularizerType : public std::runtime_error 
{
public:
	UnknownRegularizerType(std::string const& s) : std::runtime_error(s)
    {
	}
};

class RegularizerFactory
{
public:
	
	template <class T>
	static std::shared_ptr< Regularizer<T> > GetRegularizer(IOTreeNode& node)
	{
		auto& attributes = node.attributes();
		assert( attributes.GetEntry("Category") == "Regularizer" );
		std::string type = attributes.GetEntry("Type");
		if (type == "AbsRegularizer")
			return AbsRegularizer<T>::Create(node);
		else if (type == "EmptyRegularizer")
			return EmptyRegularizer<T>::Create(node);
		else if (type == "WeightDecayRegularizer")
			return WeightDecayRegularizer<T>::Create(node);
		else 
			throw UnknownRegularizerType(type);
	}
};

#endif