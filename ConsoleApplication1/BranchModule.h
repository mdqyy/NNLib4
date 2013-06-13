#ifndef BRANCH_MODULE_H
#define BRANCH_MODULE_H

#include "Module.h"

template <class ParamsType>
class BranchModule : public Module<ParamsType>
{	
	std::shared_ptr< Module<ParamsType> > branch_module_;
	std::shared_ptr< Tensor<ParamsType> > branch_module_output_gradients_;
public:

	virtual double GetCost(const std::vector<ParamsType>& samples_importances)
	{
		return branch_module_->GetCost(samples_importances);
	}

	virtual void GetParameters(std::vector<ParamsType>& receiver) const
	{
		branch_module_->GetParameters(receiver);
	}

	virtual void SetParameters(const ParamsType* params)
	{
		branch_module_->SetParameters(params);
	}

	virtual void SetParameters(const std::vector<ParamsType>& params)
	{
		branch_module_->SetParameters(params);
	}

	virtual void SetParameters(const Tensor<ParamsType>& params)
	{
		branch_module_->SetParameters(params);
	}

	virtual void GetGradients(std::vector<ParamsType>& receiver) const
	{
		branch_module_->GetGradients(receiver);
	}

	virtual void InitializeParameters()
	{
		Module<ParamsType>::InitializeParameters();
		branch_module_->InitializeParameters();
	}

	virtual size_t GetNumParams() const
	{
		return branch_module_->GetNumParams();
	}

	std::shared_ptr< Module<ParamsType> > GetBranchModule()
	{
		return branch_module_;
	}
	std::shared_ptr< const Module<ParamsType> > GetBranchModule() const
	{
		return branch_module_;
	}

	std::shared_ptr< Tensor<ParamsType> > GetBranchModuleOutputBuffer()
	{
		return branch_module_->GetOutputBuffer();
	}

	std::shared_ptr< const Tensor<ParamsType> > GetBranchModuleOutputBuffer() const
	{
		return branch_module_->GetOutputBuffer();
	}

	void PushBranchGradients(std::shared_ptr< Tensor<ParamsType> > branch_module_output_gradients)
	{
		branch_module_output_gradients_ = branch_module_output_gradients;
	}

	virtual bool AlocateOutputBuffer() const
	{
		return false;
	}

	virtual bool AlocateInputGradientsBuffer() const
	{
		return true;
	}

	BranchModule( std::string name, std::shared_ptr< Module<ParamsType> > branch_module) : Module<ParamsType>(name), branch_module_(branch_module)
	{
	}

	virtual std::string GetType() const
	{
		return "BranchModule";
	}
	
	virtual bool Equals(const Module<ParamsType>& module) const
	{
		return module.GetType() == GetType() && module.GetName() == GetName() && 
			branch_module_->Equals( *static_cast<const BranchModule<ParamsType>*>(&module)->GetBranchModule() );
	}
	
	static std::shared_ptr< Module< ParamsType> > Create(IOTreeNode& data);

protected:
	virtual void sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);
	virtual void sub_predict_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);
	
	virtual void sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
		const std::vector<ParamsType>& samples_importances);

	virtual void sub_GetState(IOTreeNode& node) const;

};

template <class ParamsType>
void BranchModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
	node.nodes().AppendEntry( "branch_module", branch_module_->GetState() );
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > BranchModule<ParamsType>::Create(IOTreeNode& data)
{
	std::shared_ptr< Module< ParamsType> > branch_module = ModuleFactory::GetModule<ParamsType>( *data.nodes().GetEntry("branch_module") );
	return std::shared_ptr< Module< ParamsType> >( new BranchModule<ParamsType>( data.attributes().GetEntry( "Name" ), branch_module ) );
}

template <class ParamsType>
void BranchModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	branch_module_->train_fprop(input);
	branch_module_output_gradients_ = nullptr;
	output = input;
}

template <class ParamsType>
void BranchModule<ParamsType>::sub_predict_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	branch_module_->predict_fprop(input);
	branch_module_output_gradients_ = nullptr;
	output = input;
}

template <class ParamsType>
void BranchModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
	std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
	const std::vector<ParamsType>& samples_importances)
{
	std::shared_ptr< Tensor<ParamsType> > branch_module_input_gradients_ = branch_module_->bprop( branch_module_output_gradients_, samples_importances );
	Tensor<ParamsType>& branch_module_input_gradients_tensor = *branch_module_input_gradients_;
	Tensor<ParamsType>& input_gradients_tensor = *input_gradients;
	const Tensor<ParamsType>& output_gradients_tensor = *output_gradients;
	for (size_t i = 0; i<input_gradients_tensor.Numel(); i++)
		input_gradients_tensor[i] = output_gradients_tensor[i]+branch_module_input_gradients_tensor[i];
}

#endif