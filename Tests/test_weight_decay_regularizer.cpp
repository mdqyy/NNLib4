#include <boost/test/unit_test.hpp>
#include "WeightDecayRegularizer.h"

BOOST_AUTO_TEST_CASE(TestWeughtDecayRegularizer)
{

	float params[] = {0, 0, 1, 2, -1, -2,
							4, 2,  1,  3, 0, 0};
	std::vector<size_t> params_dims; params_dims.push_back(4); params_dims.push_back(2);
	Tensor<float> params_tensor(params+2, params_dims);
	std::shared_ptr<Regularizer<float>> regularizer(new WeightDecayRegularizer<float>(0.5));
	double cost = regularizer->GetCost(params_tensor, 1);
	BOOST_CHECK_EQUAL(cost, 10);
	
	float gradients[] = {1,1,1,1,1,1,1,1};
	Tensor<float> gradients_tensor(gradients, params_dims);
	regularizer->GetGradients(params_tensor, gradients_tensor, 1);
	BOOST_CHECK_EQUAL(gradients_tensor[0], 1.5);
	BOOST_CHECK_EQUAL(gradients_tensor[1], 2);
	BOOST_CHECK_EQUAL(gradients_tensor[2], 0.5);
	BOOST_CHECK_EQUAL(gradients_tensor[3], 0);
	BOOST_CHECK_EQUAL(gradients_tensor[4], 3);
	BOOST_CHECK_EQUAL(gradients_tensor[5], 2);
	BOOST_CHECK_EQUAL(gradients_tensor[6], 1.5);
	BOOST_CHECK_EQUAL(gradients_tensor[7], 2.5);
}