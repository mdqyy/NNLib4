#include <boost/test/unit_test.hpp>
#include <vector>
#include "Tensor.h"
#include "CrossEntropyCostModule.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestCrossEntropyCostModule)
{
	std::vector<size_t> output_dims;output_dims.push_back(4); output_dims.push_back(2);
	float output[] = {0.4147f, 0.2645f, 0.1604f, 0.1604f, 0.1708f, 0.2548f, 0.4118f, 0.1625f};
	float expected_output[] = {1, 0, 0, 0, 0, 1, 0, 0};
	std::vector<float> importance;importance.push_back(1); importance.push_back(2);

	Tensor<float> expected_output_tensor(expected_output, output_dims);
	Tensor<float> output_tensor(output, output_dims);

	CrossEntropyCostModule<float> cost_module;
	double cost = cost_module.GetCost(output_tensor, expected_output_tensor, importance, false);
	BOOST_CHECK( abs(cost-5.2150)<0.001);
	cost = cost_module.GetCost(output_tensor, expected_output_tensor, importance, true);
	BOOST_CHECK( abs(cost-5.2150/3)<0.001);
}