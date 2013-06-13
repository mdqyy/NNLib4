#ifndef KERNEL_MODULE_H
#define KERNEL_MODULE_H

#include "Module.h"
#include "ConstantInitializer.h"
#include "EmptyRegularizer.h"
#include "Kernel.h"
#include "KernelFactory.h"
#include "IOTreeNode.h"
#include "TensorIO.h"
#include "Converter.h"
#include "KernelFactoryIO.h"

template <class ParamsType>
class KernelModule : public Module<ParamsType>
{
private:
	Tensor<ParamsType> parameters_;
	std::vector< std::shared_ptr< Kernel<ParamsType> > > kernels;
	std::vector<Tensor<ParamsType> > kernels_gradients;
	std::vector<size_t> kernels_dims;
	std::vector<size_t> kernels_strides;
	std::shared_ptr<ParametersInitializer<ParamsType> > params_initializer;
	std::shared_ptr<Regularizer<ParamsType> > regularizer;
	std::shared_ptr<KernelFactory<ParamsType> > kernel_factory_;
public:

	size_t GetNumKernels() const
	{
		return kernels.size();
	}
	
	virtual double GetCost(const std::vector<ParamsType>& samples_importances);

	std::shared_ptr< Kernel<ParamsType> > GetKernel(size_t kernel_ind)
	{
		return kernels[kernel_ind];
	}
	
	KernelModule(std::string name, size_t num_kernels, const std::vector<size_t>& kernels_dims, const std::vector<size_t>& kernels_strides, 
		const KernelFactory<ParamsType>& kernel_factory, 
		const std::shared_ptr<ParametersInitializer<ParamsType> >& params_initializer = std::shared_ptr<ParametersInitializer<ParamsType> >(new ConstantInitializer<ParamsType>(0)),
		const std::shared_ptr<Regularizer<ParamsType> >& regularizer = std::shared_ptr<Regularizer<ParamsType> >(new EmptyRegularizer<ParamsType>()) );

	virtual void InitializeParameters();

	virtual void GetGradients(std::vector<ParamsType>& receiver) const;

	virtual void GetParameters(std::vector<ParamsType>& receiver) const;

	virtual void SetParameters(const ParamsType* parameters);

	virtual void SetParameters(const std::vector<ParamsType>& params);

	virtual void SetParameters(Tensor<ParamsType>& parameters);

	virtual std::vector<size_t> GetPerCaseOutputDims(const std::vector<size_t>& per_case_input_dims) const;

	virtual size_t GetNumParams() const
	{
		return kernels[0]->GetNumberOfParameters()*kernels.size();
	}

	static std::shared_ptr< Module< ParamsType> > Create(IOTreeNode& data);
	
	virtual std::string GetType() const
	{
		return "KernelModule";
	}

	virtual bool Equals(const Module<ParamsType>& module) const;

protected:
	virtual void sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);
	
	virtual void sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& upper_gradients, 
		const std::vector<ParamsType>& samples_importances);
	
	virtual void sub_GetState(IOTreeNode& node) const;
};

template <class ParamsType>
bool KernelModule<ParamsType>::Equals(const Module<ParamsType>& module) const
{
	if (module.GetType() != GetType() || module.GetName() != GetName())
		return false;

	const KernelModule<ParamsType>* other_module = static_cast< const KernelModule<ParamsType>* >( &module );
	if (other_module->parameters_ != parameters_)
		return false;
	if (other_module->kernels.size() != kernels.size())
		return false;
	if (other_module->kernels_dims != kernels_dims)
		return false;
	if (other_module->kernels_strides != kernels_strides)
		return false;
	if (!params_initializer->Equals(*other_module->params_initializer))
		return false;
	if (!regularizer->Equals(*other_module->regularizer))
		return false;
	if (kernel_factory_->GetKernelType() != other_module->kernel_factory_->GetKernelType())
		return false;
	for (size_t i = 0; i< kernels.size(); i++)
		if (!kernels[i]->Equals(*other_module->kernels[i]))
			return false;
	return true;
}

template <class ParamsType>
void KernelModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
	node.attributes().AppendEntry( "num_kernels", std::to_string(kernels.size()) );
	node.attributes().AppendEntry( "kernels_dims", Converter::ConvertVectorToString(kernels_dims) );
	node.attributes().AppendEntry( "kernels_strides", Converter::ConvertVectorToString(kernels_strides) );
	node.attributes().AppendEntry( "kernel_type", kernel_factory_->GetKernelType() );
	node.nodes().AppendEntry( "regularizer", regularizer->GetState() );
	node.nodes().AppendEntry( "initializer", params_initializer->GetState() );
	node.nodes().AppendEntry( "Parameters", GetTensorState(parameters_) );
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > KernelModule<ParamsType>::Create(IOTreeNode& data)
{
	size_t num_kernels = Converter::ConvertTo<size_t>(data.attributes().GetEntry( "num_kernels" ));
	std::shared_ptr< Regularizer<ParamsType> > regularizer = RegularizerFactory::GetRegularizer<ParamsType>(*data.nodes().GetEntry("regularizer"));
	std::shared_ptr< ParametersInitializer<ParamsType> > initializer = InitializerFactory::GetInitializer<ParamsType>(*data.nodes().GetEntry("initializer"));
	std::shared_ptr<KernelFactory<ParamsType> > kernel_factory = GetKernelFactory<ParamsType>( data.attributes().GetEntry("kernel_type") );
	std::shared_ptr< Tensor<ParamsType> > parameters_tensor = CreateTensor<ParamsType>(*data.nodes().GetEntry("Parameters"));
	std::vector<size_t> kernels_dims = Converter::StringToVector<size_t>( data.attributes().GetEntry("kernels_dims") );
	std::vector<size_t> kernels_strides = Converter::StringToVector<size_t>( data.attributes().GetEntry("kernels_strides") );

	std::shared_ptr< KernelModule<ParamsType> > module = 
		std::shared_ptr< KernelModule< ParamsType> >( new KernelModule<ParamsType>(data.attributes().GetEntry( "Name" ), 
		num_kernels, kernels_dims, kernels_strides, *kernel_factory, 
		initializer, regularizer));

	module->SetParameters(*parameters_tensor);

	return module;
}

template <class ParamsType>
void KernelModule<ParamsType>::GetParameters(std::vector<ParamsType>& receiver) const
{
	for (size_t i=0; i < parameters_.Numel(); i++)
		receiver.push_back( parameters_[i] );
}

template <class ParamsType>
void KernelModule<ParamsType>::GetGradients(std::vector<ParamsType>& receiver) const
{
	for (size_t kernel_ind=0; kernel_ind<kernels_gradients.size(); kernel_ind++)
	{
		const Tensor<ParamsType>& kernel_gradients = kernels_gradients[kernel_ind];
		for (size_t i=0; i < kernel_gradients.Numel(); i++)
			receiver.push_back( kernel_gradients[i] );
	}
}

template <class ParamsType>
void KernelModule<ParamsType>::SetParameters(const ParamsType* parameters)
{
	size_t numel = parameters_.Numel();
	for (size_t i = 0; i < numel; i++)
		parameters_[i] = parameters[i];
}

template <class ParamsType>
void KernelModule<ParamsType>::SetParameters(const std::vector<ParamsType>& params)
{
	assert(parameters_.Numel() == params.size());
	SetParameters(params.data());
}

template <class ParamsType>
void KernelModule<ParamsType>::SetParameters(Tensor<ParamsType>& parameters)
{
	assert( parameters_.Numel() == parameters.Numel() );
	size_t numel = parameters_.Numel();
	for (size_t i = 0; i < numel; i++)
		parameters_[i] = parameters[i];
}

template <class ParamsType>
double KernelModule<ParamsType>::GetCost(const std::vector<ParamsType>& samples_importances)
{
	size_t num_params_per_kernel = GetNumParams() / GetNumKernels();
	Tensor<ParamsType> kernel_params_tensor(0, std::vector<size_t>(1, num_params_per_kernel) );
	ParamsType importance_sum = static_cast<ParamsType>(std::accumulate(samples_importances.begin(),samples_importances.end(),0.0));
	double cost = 0;
	for (size_t kernel_ind=0; kernel_ind<GetNumKernels(); kernel_ind++)
	{
		kernel_params_tensor.SetDataPtr( parameters_.GetStartPtr() + num_params_per_kernel*kernel_ind);
		cost+=regularizer->GetCost(kernel_params_tensor, importance_sum);
	}
	return cost;
}

template <class ParamsType>
KernelModule<ParamsType>::KernelModule(std::string name, size_t num_kernels, const std::vector<size_t>& kernels_dims, const std::vector<size_t>& kernels_strides, 
		const KernelFactory<ParamsType>& kernel_factory, const std::shared_ptr<ParametersInitializer<ParamsType> >& params_initializer,
		const std::shared_ptr<Regularizer<ParamsType> >& regularizer ) 
		: Module(name), kernels_dims(kernels_dims), kernels_strides(kernels_strides),  	params_initializer(params_initializer), regularizer(regularizer)
{
	kernel_factory_ = kernel_factory.Clone();
	std::shared_ptr< Kernel<ParamsType> > kernel = kernel_factory.GetKernel(kernels_dims, kernels_strides, 0);
	size_t num_params_per_kernel = kernel->GetNumberOfParameters();
	size_t num_params = num_params_per_kernel*num_kernels;
	std::vector<size_t> params_dims(1, num_params);
	parameters_ = Tensor<ParamsType>(params_dims);
	ParamsType* parameters_start = parameters_.GetStartPtr();
	
	std::vector<size_t> kernel_params_dims(1, num_params_per_kernel);
	for (size_t kernel_ind = 0; kernel_ind<num_kernels; kernel_ind++)
	{
		kernels.push_back( kernel_factory.GetKernel(kernels_dims, kernels_strides, parameters_start+num_params_per_kernel*kernel_ind) );
		kernels_gradients.push_back( Tensor<ParamsType>( kernel_params_dims ) );
	}
}

template <class ParamsType>
void KernelModule<ParamsType>::InitializeParameters()
{
	size_t num_params_per_kernel = GetNumParams() / GetNumKernels();
	Tensor<ParamsType> kernel_params_tensor(0, std::vector<size_t>(1, num_params_per_kernel) );
	for (size_t kernel_ind=0; kernel_ind<GetNumKernels(); kernel_ind++)
	{
		kernel_params_tensor.SetDataPtr( parameters_.GetStartPtr() + num_params_per_kernel*kernel_ind);
		params_initializer->InitializeParameters(kernel_params_tensor);
	}
}

template <class ParamsType>
std::vector<size_t> KernelModule<ParamsType>::GetPerCaseOutputDims(const std::vector<size_t>& per_case_input_dims) const
{
	auto dims = kernels[0]->GetOutputTensorDimensions(per_case_input_dims);
	assert( !(dims[dims.size()-1] >1 && GetNumKernels()>1) ); // Nontypical use. Possibly a mistake
	dims[dims.size()-1]*=GetNumKernels();
	return dims;
}


template <class ParamsType>
void KernelModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	std::vector<size_t> per_case_input_dims = input->GetDimensions();
	per_case_input_dims.pop_back(); // remove minibatch dimension
	std::vector<size_t> per_case_kernel_output_dims = kernels[0]->GetOutputTensorDimensions(per_case_input_dims);
	size_t num_kernel_dims_per_kernel = per_case_kernel_output_dims[per_case_kernel_output_dims.size()-1];
	size_t minibatch_size = input->GetDimensionSize(input->NumDimensions()-1);
	std::vector<size_t> input_pos(input->NumDimensions());
	std::vector<size_t> output_pos(per_case_kernel_output_dims.size()+1);

	Tensor<ParamsType> input_tensor(0, per_case_input_dims);
	Tensor<ParamsType> output_tensor(0, per_case_kernel_output_dims);
	for (size_t case_ind = 0; case_ind<minibatch_size; case_ind++ )
	{
		input_pos[input_pos.size()-1] = case_ind;
		output_pos[output_pos.size()-1] = case_ind;
		for (size_t kernel_ind = 0; kernel_ind<kernels.size(); kernel_ind++)
		{
			output_pos[output_pos.size()-2] = kernel_ind*num_kernel_dims_per_kernel;
			input_tensor.SetDataPtr(input->GetPtr(input_pos.data()));
			output_tensor.SetDataPtr(output->GetPtr(output_pos.data()));
			kernels[kernel_ind]->fprop(input_tensor, output_tensor);
		}
	}
}

template <class ParamsType>
void KernelModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
		const std::vector<ParamsType>& samples_importances)
{
	for (size_t i=0; i< kernels_gradients.size(); i++)
		kernels_gradients[i].SetZeros();
	std::vector<size_t> per_case_input_dims = input->GetDimensions();
	per_case_input_dims.pop_back(); // remove minibatch dimension

	std::vector<size_t> per_case_kernel_output_dims = kernels[0]->GetOutputTensorDimensions(per_case_input_dims);
	size_t num_kernel_dims_per_kernel = per_case_kernel_output_dims[per_case_kernel_output_dims.size()-1];

	size_t minibatch_size = input->GetDimensionSize(input->NumDimensions()-1);
	std::vector<size_t> input_pos(per_case_input_dims.size()+1);
	std::vector<size_t> output_pos(per_case_kernel_output_dims.size()+1);
	
	size_t num_params_per_kernel = GetNumParams() / GetNumKernels();
	Tensor<ParamsType> kernel_params_tensor(0, std::vector<size_t>(1, num_params_per_kernel) );
	ParamsType importance_sum = static_cast<ParamsType>(std::accumulate(samples_importances.begin(),samples_importances.end(),0.0));
	for (size_t kernel_ind=0; kernel_ind<GetNumKernels(); kernel_ind++)
	{
		kernel_params_tensor.SetDataPtr( parameters_.GetStartPtr() + num_params_per_kernel*kernel_ind);
		regularizer->GetGradients(kernel_params_tensor, kernels_gradients[kernel_ind], importance_sum);
	}

	Tensor<ParamsType> input_tensor(0, per_case_input_dims);
	Tensor<ParamsType> input_gradients_tensor(0, per_case_input_dims);
	Tensor<ParamsType> output_tensor(0, per_case_kernel_output_dims);
	Tensor<ParamsType> output_gradients_tensor(0, per_case_kernel_output_dims);

	// update gradient
	for (size_t case_ind = 0; case_ind<minibatch_size; case_ind++ )
	{
		input_pos[input_pos.size()-1] = case_ind;
		output_pos[output_pos.size()-1] = case_ind;
		for (size_t kernel_ind = 0; kernel_ind<kernels.size(); kernel_ind++)
		{
			output_pos[output_pos.size()-2] = kernel_ind*num_kernel_dims_per_kernel;
			input_tensor.SetDataPtr(input->GetPtr(input_pos.data()));
			output_tensor.SetDataPtr(output->GetPtr(output_pos.data()));
			output_gradients_tensor.SetDataPtr(output_gradients->GetPtr(output_pos.data()));
			kernels[kernel_ind]->GetGradient(input_tensor, output_tensor, output_gradients_tensor, kernels_gradients[kernel_ind]);
		}
	}

	// bprop
	input_gradients->SetZeros();
	for (size_t case_ind = 0; case_ind<minibatch_size; case_ind++ )
	{
		input_pos[input_pos.size()-1] = case_ind;
		output_pos[output_pos.size()-1] = case_ind;
		for (size_t kernel_ind = 0; kernel_ind<kernels.size(); kernel_ind++)
		{
			output_pos[output_pos.size()-2] = kernel_ind*num_kernel_dims_per_kernel;
			input_tensor.SetDataPtr(input->GetPtr(input_pos.data()));
			output_tensor.SetDataPtr(output->GetPtr(output_pos.data()));
			output_gradients_tensor.SetDataPtr(output_gradients->GetPtr(output_pos.data()));
			input_gradients_tensor.SetDataPtr(input_gradients->GetPtr(input_pos.data()));
			kernels[kernel_ind]->bprop(input_tensor, output_tensor, input_gradients_tensor, output_gradients_tensor);

		}
	}
}

#endif