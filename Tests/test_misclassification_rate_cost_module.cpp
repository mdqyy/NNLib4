#include <boost/test/unit_test.hpp>
#include <vector>
#include "Tensor.h"
#include "MisclassificationRateCostModule.h"
#include "test_utilities.h"
#include "FullTensorDataLoader.h"
#include "GaussianInitializer.h"
#include "WeightDecayRegularizer.h"
#include "LinearMixModule.h"
#include "CompositeModule.h"

BOOST_AUTO_TEST_CASE(TestMisclassificationRateCostModule)
{
	std::vector<size_t> output_dims;output_dims.push_back(4); output_dims.push_back(5);
	float output[] = {0.45f, 0.5f, 0.8f, -1.0f, 
					  0.49f, 0.5f, -0.4f, -0.1f, 
					  1.0f, 2.0f, -3.0f, 1.0f, 
					  4.0f, 1.0f, 5.0f, 2.0f, 
					  -9.0f, 8.0f, 1.0f, 4.0f};
	float expected_output[] = {1.0f, 0, 0, 0,     0, 1, 0, 0,     0, 0, 1, 0,     0, 0, 1, 0,     0, 0, 0, 1};
	std::vector<float> importance;importance.push_back(1); importance.push_back(2); importance.push_back(3); importance.push_back(4);importance.push_back(5);

	Tensor<float> expected_output_tensor(expected_output, output_dims);
	Tensor<float> output_tensor(output, output_dims);

	MisclassificationRateCostModule<float> cost_module;
	double cost = cost_module.GetCost(output_tensor, expected_output_tensor, importance, false);
	BOOST_CHECK( cost == 9);
	cost = cost_module.GetCost(output_tensor, expected_output_tensor, importance, true);
	BOOST_CHECK( abs(cost-9.0/15)<0.00001);
}