#include <boost/test/unit_test.hpp>
#include <vector>
#include "Tensor.h"
#include "AbsCostModule.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestAbsCostModule)
{
	std::vector<size_t> output_dims;output_dims.push_back(4); output_dims.push_back(2);
	float output[] = {0.95f, 0.5f, 0.2f, 0.4f, 0.1f, 0.5f, 0.98f, 0.05f};
	float expected_output[] = {1, 0, 0, -1, 0, 1, 0, 1};
	std::vector<float> importance;importance.push_back(1); importance.push_back(2);

	Tensor<float> expected_output_tensor(expected_output, output_dims);
	Tensor<float> output_tensor(output, output_dims);

	AbsCostModule<float> abs_module;
	double cost = abs_module.GetCost(output_tensor, expected_output_tensor, importance, false);
	BOOST_CHECK( abs(cost-7.21)<0.001);
	cost = abs_module.GetCost(output_tensor, expected_output_tensor, importance, true);
	BOOST_CHECK( abs(cost-7.21/3)<0.001);

	std::shared_ptr< Tensor<float> > gradients = abs_module.bprop(output_tensor, expected_output_tensor, importance, false, 0.5);
	float gradients1[] = {-0.5, 0.5, 0.5, 0.5, 1, -1, 1, -1};
	BOOST_CHECK(test_equal_arrays(gradients1, gradients->GetStartPtr(), 8));

	gradients = abs_module.bprop(output_tensor, expected_output_tensor, importance, true, 0.5);
	float gradients2[] = {-0.5f / 3 , 0.5f / 3, 0.5f / 3, 0.5f / 3, 1.0f / 3, -1.0f / 3, 1.0f / 3, -1.0f / 3};
	BOOST_CHECK(test_equal_arrays(gradients2, gradients->GetStartPtr(), 8));

}