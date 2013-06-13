#include <boost/test/unit_test.hpp>
#include <vector>
#include "Tensor.h"
#include "MseCostModule.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestMseCostModule)
{
	std::vector<size_t> output_dims;output_dims.push_back(4); output_dims.push_back(2);
	float output[] = {0.95f, 0.5f, 0.2f, 0.4f, 0.1f, 0.5f, 0.98f, 0.05f};
	float expected_output[] = {1, 0, 0, 1, 0, 1, 0, 1};
	std::vector<float> importance;importance.push_back(1); importance.push_back(2);

	Tensor<float> expected_output_tensor(expected_output, output_dims);
	Tensor<float> output_tensor(output, output_dims);

	MseCostModule<float> mse_module;
	double cost = mse_module.GetCost(output_tensor, expected_output_tensor, importance, false);
	BOOST_CHECK( abs(cost-2.4491)<0.001);
	cost = mse_module.GetCost(output_tensor, expected_output_tensor, importance, true);
	BOOST_CHECK( abs(cost-2.4491/3)<0.001);
}