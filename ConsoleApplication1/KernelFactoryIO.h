#ifndef KERNEL_FACTORY_IO
#define KERNEL_FACTORY_IO

#include <memory>
#include <vector>
#include <stdexcept>
#include "KernelFactory.h"
#include "IOTreeNode.h"

class UnknownKernelType : public std::runtime_error 
{
public:
	UnknownKernelType(std::string const& s) : std::runtime_error(s)
    {
	}
};

template< class T>
std::shared_ptr< KernelFactory<T> > GetKernelFactory(std::string kernel_type)
{
	if (kernel_type == "ConvolutionalKernel")
		return std::shared_ptr< KernelFactory<T> >( new ConvolutionalKernelFactory<T>());
	else if (kernel_type == "MaxPoolingKernel")
		return std::shared_ptr< KernelFactory<T> >( new MaxPoolingKernelFactory<T>());
	else
		throw UnknownKernelType(kernel_type);
}

#endif