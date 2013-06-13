#include <boost/test/unit_test.hpp>
#include <assert.h>
#include <vector>
#include "Tensor.h"
#include "ConstantInitializer.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestConstantInitializer)
{
	ConstantInitializer<float> initializer(5);
	std::vector<size_t> dims;dims.push_back(5);dims.push_back(5);dims.push_back(4);dims.push_back(4);dims.push_back(8);
	Tensor<float> tensor = GetRandomTensor<float>(dims);
	initializer.InitializeParameters(tensor);
	for (size_t i=0; i<tensor.Numel(); i++)
		BOOST_CHECK_EQUAL(tensor[i] , 5);
}