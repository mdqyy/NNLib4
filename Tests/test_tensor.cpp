#include <boost/test/unit_test.hpp>
#include <vector>
#include "Tensor.h"

BOOST_AUTO_TEST_CASE(TestTensor)
{
	float data[100];
	for (int i=0; i<100; i++)
		data[i] = (float)i;
	std::vector<size_t> dims; dims.push_back(3); dims.push_back(4);	dims.push_back(5);
	Tensor<float> data_buffer(data, dims);

	BOOST_CHECK_EQUAL(Tensor<float>::Numel(dims),60);
	BOOST_CHECK_EQUAL(data_buffer.GetDimStride(0),1);
	BOOST_CHECK_EQUAL(data_buffer.GetDimStride(1),3);
	BOOST_CHECK_EQUAL(data_buffer.GetDimStride(2),12);
	BOOST_CHECK_EQUAL(data_buffer.Numel(),60);
	size_t pos[3] = {2,1,3};
	BOOST_CHECK_EQUAL(data_buffer.GetPtr(pos), data+41);

	data_buffer.SetZeros();
	for (int i=0; i<60; i++)
		BOOST_CHECK_EQUAL(data[i],0);
	for (int i=60; i<100; i++)
		BOOST_CHECK_EQUAL(data[i],i);

	std::vector<size_t> output_pos = Tensor<float>::IndToPos(dims, 41);
	for (size_t i =0; i<3; i++)
		BOOST_CHECK_EQUAL(output_pos[i],pos[i]);

	// test that when a tensor that owns data is created it sets its parameters to zeros
	std::vector<size_t> dims2; dims2.push_back(6); dims.push_back(9);
	data_buffer = Tensor<float>(dims2);
	for (size_t i=0; i<data_buffer.Numel(); i++)
		BOOST_CHECK( data_buffer[i] == 0 );
}

BOOST_AUTO_TEST_CASE(TestTensorComparison)
{
	std::vector<size_t> dims1; dims1.push_back(5); dims1.push_back(9);
	Tensor<float> tensor1(dims1);
	std::vector<size_t> dims2; dims2.push_back(2); dims2.push_back(9);
	Tensor<float> tensor2(dims2);
	BOOST_CHECK( tensor1 != tensor2 );

	tensor2 = Tensor<float>(dims1);
	BOOST_CHECK( tensor1 == tensor2 );

	tensor1[0] = 1;
	BOOST_CHECK( tensor1 != tensor2 );
	tensor1[0] = 0;
	tensor2[0] = 1;
	BOOST_CHECK( tensor1 != tensor2 );
	tensor1[tensor1.Numel()-1] = 1;
	tensor2[0] = 0;
	BOOST_CHECK( tensor1 != tensor2 );
	tensor1[tensor1.Numel()-1] = 0;
	tensor2[tensor1.Numel()-1] = 1;
	BOOST_CHECK( tensor1 != tensor2 );
	tensor2[tensor1.Numel()-1] = 0;

	float data1[100] = {0};
	tensor1 = Tensor<float>(data1, dims1);
	BOOST_CHECK( tensor1 != tensor2 );

	float data2[100] = {0};
	tensor2 = Tensor<float>(data1, dims1);
	BOOST_CHECK( tensor1 == tensor2 );
	tensor2 = Tensor<float>(data2, dims1);
	BOOST_CHECK( tensor1 == tensor2 );
}