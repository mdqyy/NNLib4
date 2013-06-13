#ifndef INITIALIZER_FACTORY_H
#define INITIALIZER_FACTORY_H

#include "IOTreeNode.h"
#include "GaussianInitializer.h"
#include "LinearMixInitializer.h"
#include "ConstantInitializer.h"
#include "UniformInitializer.h"

class UnknownInitializerType : public std::runtime_error 
{
public:
	UnknownInitializerType(std::string const& s) : std::runtime_error(s)
    {
	}
};

class InitializerFactory
{
public:
	template <class T>
	static std::shared_ptr< ParametersInitializer<T> > GetInitializer(IOTreeNode& node)
	{
		auto& attributes = node.attributes();
		assert( attributes.GetEntry("Category") == "Initializer" );
		std::string type = attributes.GetEntry("Type");
		if (type == "GaussianInitializer")
			return GaussianInitializer<T>::Create(node);
		else if (type == "LinearMixInitializer")
			return LinearMixInitializer<T>::Create(node);
		else if (type == "ConstantInitializer")
			return ConstantInitializer<T>::Create(node);
		else if (type == "UniformInitializer")
			return UniformInitializer<T>::Create(node);
		else 
			throw UnknownInitializerType(type);
	}
};

#endif