#ifndef MODULE_FACTORY_H
#define MODULE_FACTORY_H

#include <string>
#include "Module.h"
#include "AbsModule.h"
#include "LinearModule.h"
#include "LinearMixModule.h"
#include "BiasModule.h"
#include "KernelModule.h"
#include "CompositeModule.h"
#include "DropoutModule.h"
#include "GaussianNoiseModule.h"
#include "RectifiedLinearUnitModule.h"
#include "SoftmaxModule.h"
#include "SigmoidModule.h"
#include "SoftSignModule.h"
#include "TanhModule.h"
#include "IOTreeNode.h"
#include "BranchModule.h"
#include "PureSoftmaxModule.h"
#include "MeanStdNormalizingModule.h"
#include "BatchPureSoftmaxModule.h"
#include "BatchSoftmaxModule.h"
#include "EntropyRegularizingModule.h"

class UnknownModuleType : public std::runtime_error 
{
public:
	UnknownModuleType(std::string const& s) : std::runtime_error(s)
    {
	}
};

class ModuleFactory
{
public:
	template <class T>
	static std::shared_ptr< Module<T> > GetModule(IOTreeNode& node)
	{
		auto& attributes = node.attributes();
		assert( attributes.GetEntry("Category") == "Module" );
		std::string type = attributes.GetEntry("Type");
		if (type == "AbsModule")
			return AbsModule<T>::Create(node);
		else if (type == "CompositeModule")
			return CompositeModule<T>::Create(node);
		else if (type == "DropoutModule")
			return DropoutModule<T>::Create(node);
		else if (type == "GaussianNoiseModule")
			return GaussianNoiseModule<T>::Create(node);
		else if (type == "RectifiedLinearUnitModule")
			return RectifiedLinearUnitModule<T>::Create(node);
		else if (type == "SoftmaxModule")
			return SoftmaxModule<T>::Create(node);
		else if (type == "PureSoftmaxModule")
			return PureSoftmaxModule<T>::Create(node);
		else if (type == "BatchSoftmaxModule")
			return BatchSoftmaxModule<T>::Create(node);
		else if (type == "BatchPureSoftmaxModule")
			return BatchPureSoftmaxModule<T>::Create(node);
		else if (type == "SigmoidModule")
			return SigmoidModule<T>::Create(node);
		else if (type == "SoftSignModule")
			return SoftSignModule<T>::Create(node);
		else if (type == "LinearModule")
			return LinearModule<T>::Create(node);
		else if (type == "LinearMixModule")
			return LinearMixModule<T>::Create(node);
		else if (type == "BiasModule")
			return BiasModule<T>::Create(node);
		else if (type == "KernelModule")
			return KernelModule<T>::Create(node);
		else if (type == "TanhModule")
			return TanhModule<T>::Create(node);
		else if (type == "BranchModule")
			return BranchModule<T>::Create(node);
		else if (type == "MeanStdNormalizingModule")
			return MeanStdNormalizingModule<T>::Create(node);
		else if (type == "EntropyRegularizingModule")
			return EntropyRegularizingModule<T>::Create(node);
		else 
			throw UnknownModuleType("Unknown module:"+type);
	}
};

#endif