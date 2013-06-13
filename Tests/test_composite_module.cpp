#include <boost/test/unit_test.hpp>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "AbsModule.h"
#include "BiasModule.h"
#include "ConstantInitializer.h"
#include "WeightDecayRegularizer.h"
#include "LinearMixModule.h"
#include "CompositeModule.h"
#include "FullTensorDataLoader.h"
#include "TrainDataset.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestCompositeModule)
{
	std::vector< std::shared_ptr< Module<float> > > modules;
	float linear_mix_input[] = {1, 2, -1, -2,
								4, 2,  1,  3};
	float linear_mix_parameters[12] = {1,4,2,
							3,2,4,
							5,1,3,
							8,1,2};
	std::vector<size_t> input_dims; input_dims.push_back(4); input_dims.push_back(2);
	std::vector<size_t> per_case_input_dims; per_case_input_dims.push_back(4);
	std::vector<size_t> output_dims; output_dims.push_back(3); output_dims.push_back(2);
	std::vector<float> importances;importances.push_back(1); importances.push_back(1);

	std::shared_ptr< Tensor<float> > input_tensor(new Tensor<float>(linear_mix_input, input_dims) );
	//float expected_output[] = {-14, 5, 3, 39, 24,25};
	
	std::shared_ptr<ConstantInitializer<float>> initializer(new ConstantInitializer<float>(0));
	std::shared_ptr<WeightDecayRegularizer<float>> regularizer(new WeightDecayRegularizer<float>(0.5));
	std::shared_ptr< LinearMixModule<float> > lmm(new LinearMixModule<float>("module1", 4,3,initializer, regularizer));
	lmm->SetParameters(linear_mix_parameters);
	modules.push_back(lmm);

	float bias_parameters[3] = {4,-2,-8};
	std::vector<size_t> bias_input_dims; bias_input_dims.push_back(3);
	//float expected_output[] = {-10, 3, -8, 43, 22, 17};
	
	std::shared_ptr< BiasModule<float> > bias_module(new BiasModule<float>("module2", bias_input_dims, initializer, regularizer));
	bias_module->SetParameters(bias_parameters);
	modules.push_back(bias_module);

	float abs_gradients[1] = {0};
	float abs_parameters[1] = {0};
	std::vector<size_t> abs_input_dims; abs_input_dims.push_back(3);
	//float expected_output[] = {10, 3, 8, 43, 22, 17};
	
	std::shared_ptr< Module<float> > abs_module(new AbsModule<float>("module3"));
	modules.push_back(abs_module);

	float expected_output[] = {10, 3, 5, 43, 22, 17};
	std::vector<size_t> expected_output_dims; expected_output_dims.push_back(3); expected_output_dims.push_back(2);
	std::vector<size_t> expected_per_case_output_dims; expected_per_case_output_dims.push_back(3);

	CompositeModule<float> module("module4", modules);
	size_t num_params = 15;
	BOOST_CHECK(module.GetNumParams() == 15);
	BOOST_CHECK(module.GetPerCaseOutputDims(per_case_input_dims) == expected_per_case_output_dims);
	BOOST_CHECK(module.GetCost(importances) == 2*59.5);
	float params[15] = {1,4,2,
						3,2,4,
						5,1,3,
						8,1,2,
						4,-2,-8};

	// test composite module get / set params
	module.SetParameters(params);
	std::vector<float> parameters_tensor;
	module.GetParameters(parameters_tensor);
	BOOST_CHECK( parameters_tensor.size() == 15 );
	BOOST_CHECK(test_equal_arrays( parameters_tensor.data(), params, 15));
	
	// test composite module fprop
	module.train_fprop(input_tensor);
	assert(module.GetOutputBuffer()->GetDimensions() == expected_output_dims);
	BOOST_CHECK( module.GetOutputBuffer()->Numel() == 6);
	BOOST_CHECK(test_equal_arrays(module.GetOutputBuffer()->GetStartPtr(), expected_output, 6));

	// test initialization
	module.InitializeParameters();
	float linear_mix_parameters_expected[12] = {0};
	std::vector<float> lmm_params;
	lmm->GetParameters(lmm_params);
	BOOST_CHECK( lmm_params.size() == 12 );
	BOOST_CHECK(test_equal_arrays( lmm_params.data(), linear_mix_parameters_expected, 12));

	float bias_parameters_expected[3] = {0};
	std::vector<float> bias_module_params;
	bias_module->GetParameters(bias_module_params);
	BOOST_CHECK( bias_module_params.size() == 3 );
	BOOST_CHECK(test_equal_arrays( bias_module_params.data(), bias_parameters_expected, 3));

	BOOST_CHECK(module.GetCost(importances) == 0);

	// test initializer
	module.InitializeParameters();
	parameters_tensor.clear();
	module.GetParameters(parameters_tensor);
	for (size_t i=0; i<module.GetNumParams(); i++)
		BOOST_CHECK(parameters_tensor[i] == 0);
	
	BOOST_CHECK( TestGetSetParameters<float>(module, num_params) );
}