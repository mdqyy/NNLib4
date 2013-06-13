#ifndef KERNEL_FACTORY_H
#define KERNEL_FACTORY_H

#include <memory>
#include <string>
#include "Kernel.h"
#include "ConvolutionalKernel.h"
#include "MaxPoolingKernel.h"

template <class T>
class KernelFactory
{
public:
	
	virtual std::shared_ptr< KernelFactory<T> > Clone() const = 0;

	virtual std::string GetKernelType() = 0;

	KernelFactory(void)
	{
	}

	virtual ~KernelFactory(void)
	{
	}

	virtual std::shared_ptr< Kernel<T> > GetKernel(const std::vector<size_t>& dims, const std::vector<size_t>& strides, T* params_ptr) const = 0;
};

template <class T>
class ConvolutionalKernelFactory: public KernelFactory<T>
{
public:

	virtual std::shared_ptr< KernelFactory<T> > Clone() const
	{
		return std::shared_ptr< KernelFactory<T> >( new ConvolutionalKernelFactory() );
	}

	virtual std::string GetKernelType()
	{
		return "ConvolutionalKernel";
	}

	virtual std::shared_ptr< Kernel<T> > GetKernel(const std::vector<size_t>& dims, const std::vector<size_t>& strides, T* params_ptr) const
	{
		return std::shared_ptr< Kernel<T> >(new ConvolutionalKernel<T>(Tensor<T>(params_ptr, dims), strides));
	}
};

template <class T>
class MaxPoolingKernelFactory: public KernelFactory<T>
{
public:

	virtual std::shared_ptr< KernelFactory<T> > Clone() const
	{
		return std::shared_ptr< KernelFactory<T> >( new MaxPoolingKernelFactory() );
	}

	virtual std::string GetKernelType()
	{
		return "MaxPoolingKernel";
	}

	virtual std::shared_ptr< Kernel<T> > GetKernel(const std::vector<size_t>& dims, const std::vector<size_t>& strides, T* params_ptr) const
	{
		return std::shared_ptr< Kernel<T> >(new MaxPoolingKernel<T>(Tensor<T>(0, dims), strides));
	}
};

#endif