#include <boost/test/unit_test.hpp>
#include <assert.h>
#include <vector>
#include "Tensor.h"
#include "GaussianInitializer.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(TestGaussianInitializer)
{
	GaussianInitializer<float> initializer(0,1);
	std::vector<size_t> dims;dims.push_back(50);dims.push_back(5);dims.push_back(4);dims.push_back(4);dims.push_back(8);
	Tensor<float> tensor = GetRandomTensor<float>(dims);
	initializer.InitializeParameters(tensor);
	double std = 0;
	double mean = 0;
	for (size_t i=0; i< tensor.Numel(); i++)
	{
		mean += tensor[i];
		std += tensor[i]*tensor[i];
	}
	
	mean /= tensor.Numel();
	std /= tensor.Numel();

	BOOST_CHECK( abs(mean) < 0.05);
	BOOST_CHECK( abs(std-1) < 0.05);

}