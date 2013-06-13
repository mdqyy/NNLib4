#include <boost/test/unit_test.hpp>
#include <vector>
#include "Tensor.h"
#include "EntropyCostModule.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestEntropyCostModule)
{
	std::vector<size_t> output_dims;output_dims.push_back(4); output_dims.push_back(2);
	float output[] = {0.4f, 0.2f, 0.3f, 0.1f, 0.5f, 0.125f, 0.125f, 0.25f};
	float expected_output[] = {1, 0, 0, 0, 0, 1, 0, 0};
	std::vector<float> importance;importance.push_back(1); importance.push_back(2);

	Tensor<float> expected_output_tensor(expected_output, output_dims);
	Tensor<float> output_tensor(output, output_dims);

	EntropyCostModule<float> cost_module;
	double cost = cost_module.GetCost(output_tensor, expected_output_tensor, importance, false, 0.5);
	BOOST_CHECK( abs(cost-0.5*5.3464)<0.001);
	cost = cost_module.GetCost(output_tensor, expected_output_tensor, importance, true, 0.5);
	BOOST_CHECK( abs(cost-0.5*5.3464/3)<0.001);
}