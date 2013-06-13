#ifndef COMPOSITE_MODULE_H
#define COMPOSITE_MODULE_H

#include <map>
#include "Module.h"
#include "ModuleFactory.h"

template <class ParamsType>
class CompositeModule : public Module<ParamsType>
{
	std::vector<std::shared_ptr<Module<ParamsType> > > modules_;

	std::map< std::string, std::shared_ptr<Module<ParamsType> > > modules_map_;

public:

	CompositeModule(std::string name, std::vector<std::shared_ptr<Module<ParamsType> > >& modules) : Module<ParamsType>(name), modules_(modules)
	{
		for (size_t i=0; i< modules_.size(); i++)
			modules_map_[modules_[i]->GetName()] = modules_[i];
	}
	
	virtual size_t GetNumParams() const;

	virtual bool AlocateOutputBuffer() const
	{
		return false;
	}

	virtual bool AlocateInputGradientsBuffer() const
	{
		return false;
	}

	std::shared_ptr<Module<ParamsType> > GetModule(std::string name)
	{
		if ( modules_map_.find( name ) == modules_map_.end() )
			throw "CompositeModule: no module with name " + name;
		return modules_map_[name];
	}
	
	size_t NumModules()
	{
		return modules_.size();
	}

	std::shared_ptr<Module<ParamsType> > GetModule(size_t module_ind)
	{
		if ( modules_.size() <= module_ind )
			throw "CompositeModule: index out of range " + std::to_string(module_ind);
		return modules_[module_ind];
	}

	virtual void GetParameters(std::vector<ParamsType>& receiver) const
	{
		for (size_t i=0; i < modules_.size(); i++)
			modules_[i]->GetParameters(receiver);
	}

	virtual void SetParameters(const ParamsType* params)
	{
		size_t offset = 0;
		for (size_t i=0; i < modules_.size(); i++)
		{
			modules_[i]->SetParameters(params+offset);
			offset += modules_[i]->GetNumParams();
		}
		assert( offset == GetNumParams() );
	}

	virtual void SetParameters(const std::vector<ParamsType>& params)
	{
		assert(GetNumParams() == params.size());
		SetParameters(params.data());
	}

	virtual void SetParameters(const Tensor<ParamsType>& params)
	{
		assert( GetNumParams() == params.Numel() );
		SetParameters(params.GetStartPtr());
	}

	virtual void GetGradients(std::vector<ParamsType>& receiver) const
	{
		for (size_t i=0; i < modules_.size(); i++)
			modules_[i]->GetGradients(receiver);
	}

	virtual void InitializeParameters();
	
	virtual double GetCost(const std::vector<ParamsType>& samples_importances);

	virtual std::vector<size_t> GetPerCaseOutputDims(const std::vector<size_t>& per_case_input_dims) const;

	virtual void sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);
	virtual void sub_predict_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output);
	
	virtual void sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& upper_gradients, 
		const std::vector<ParamsType>& samples_importances);

	virtual void sub_GetState(IOTreeNode& node) const;
	static std::shared_ptr< Module< ParamsType> > Create(IOTreeNode& data);
	
	virtual std::string GetType() const
	{
		return "CompositeModule";
	}

	virtual bool Equals(const Module<ParamsType>& module) const;
};

template <class ParamsType>
bool CompositeModule<ParamsType>::Equals(const Module<ParamsType>& module) const
{
	if (module.GetType() != GetType() || module.GetName() != GetName())
		return false;
	
	const CompositeModule<ParamsType>* other_module = static_cast< const CompositeModule<ParamsType>* >( &module );
	for (size_t i = 0; i< modules_.size(); i++)
		if (!modules_[i]->Equals(*other_module->modules_[i]))
			return false;
	return true;
}

template <class ParamsType>
void CompositeModule<ParamsType>::sub_GetState(IOTreeNode& node) const
{
	for (size_t module_ind = 0; module_ind<modules_.size(); module_ind++)
		node.nodes().AppendEntry( "module" + std::to_string(module_ind), modules_[module_ind]->GetState() );
}

template <class ParamsType>
std::shared_ptr< Module< ParamsType> > CompositeModule<ParamsType>::Create(IOTreeNode& data)
{
	std::vector< std::shared_ptr< Module<ParamsType> > > modules;
	for (auto iter = data.nodes().begin(); iter != data.nodes().end(); iter++)
		modules.push_back( ModuleFactory::GetModule<ParamsType>( *data.nodes().GetEntry(*iter) ) );
	return std::shared_ptr< Module<ParamsType> >( new CompositeModule<ParamsType>(data.attributes().GetEntry( "Name" ), modules) );
}

template <class ParamsType>
size_t CompositeModule<ParamsType>::GetNumParams() const
{
	size_t num_params = 0;
	for (size_t module_ind = 0; module_ind<modules_.size(); module_ind++)
		num_params += modules_[module_ind]->GetNumParams();
	return num_params;
}

template <class ParamsType>
void CompositeModule<ParamsType>::InitializeParameters()
{
	size_t offset = 0;
	for (size_t module_ind = 0; module_ind<modules_.size(); module_ind++)
	{
		modules_[module_ind]->InitializeParameters();
		offset += modules_[module_ind]->GetNumParams();
	}
}

template <class ParamsType>
double CompositeModule<ParamsType>::GetCost(const std::vector<ParamsType>& samples_importances)
{
	double cost = 0;
	for (size_t module_ind = 0; module_ind<modules_.size(); module_ind++)
		cost+=modules_[module_ind]->GetCost(samples_importances);
	return cost;
}

template <class ParamsType>
std::vector<size_t> CompositeModule<ParamsType>::GetPerCaseOutputDims(const std::vector<size_t>& per_case_input_dims) const
{
	std::vector<size_t> per_case_output_dims = per_case_input_dims;
	for (size_t module_ind = 0; module_ind<modules_.size(); module_ind++)
		per_case_output_dims = modules_[module_ind]->GetPerCaseOutputDims(per_case_output_dims);
	return per_case_output_dims;
}

template <class ParamsType>
void CompositeModule<ParamsType>::sub_train_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	std::shared_ptr< Tensor<ParamsType> > buffer = input;
	for (size_t module_ind = 0; module_ind<modules_.size(); module_ind++)
		buffer = modules_[module_ind]->train_fprop(buffer);
	output = buffer;
}

template <class ParamsType>
void CompositeModule<ParamsType>::sub_predict_fprop(const std::shared_ptr< Tensor<ParamsType> >& input, std::shared_ptr< Tensor<ParamsType> >& output)
{
	std::shared_ptr< Tensor<ParamsType> > buffer = input;
	for (size_t module_ind = 0; module_ind<modules_.size(); module_ind++)
		buffer = modules_[module_ind]->predict_fprop(buffer);
	output = buffer;
}

template <class ParamsType>
void CompositeModule<ParamsType>::sub_bprop(const std::shared_ptr< Tensor<ParamsType> >& input, const std::shared_ptr< Tensor<ParamsType> >& output, 
		std::shared_ptr< Tensor<ParamsType> >& input_gradients, const std::shared_ptr< Tensor<ParamsType> >& output_gradients, 
		const std::vector<ParamsType>& samples_importances)
{
	std::shared_ptr< Tensor<ParamsType> > output_gradients_buffer = output_gradients;
	for (int module_ind = modules_.size()-1; module_ind>=0; module_ind--)
		output_gradients_buffer = modules_[module_ind]->bprop(output_gradients_buffer, samples_importances);
	input_gradients = output_gradients_buffer;
}

#endif