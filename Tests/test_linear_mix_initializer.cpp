#include <boost/test/unit_test.hpp>
#include <assert.h>
#include <vector>
#include "Tensor.h"
#include "LinearMixInitializer.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestLinearMixInitializer)
{
	LinearMixInitializer<float> initializer;
	std::vector<size_t> dims;dims.push_back(19);dims.push_back(25);
	Tensor<float> tensor = GetRandomTensor<float>(dims);
	initializer.InitializeParameters(tensor);
}