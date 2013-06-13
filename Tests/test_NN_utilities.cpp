#include <boost/test/unit_test.hpp>
#include <assert.h>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "Preprocessing.h"

BOOST_AUTO_TEST_CASE(testSameMeanAndStd)
{
	double data[] = { 1, 5, 3,
	                   2, 9, 2,
	
	                   5, 1, 1,
	                   4, 4, 5,};

	std::vector<size_t> data_dims;data_dims.push_back(3), data_dims.push_back(2);

	std::vector< std::shared_ptr< Tensor<double> > > tensors;
	tensors.push_back( std::shared_ptr< Tensor<double> >( new Tensor<double>(data, data_dims)) );
	tensors.push_back( std::shared_ptr< Tensor<double> >( new Tensor<double>(data+6, data_dims)) );

	auto mean = GetSameMean(tensors);
	SubtractMean(tensors, mean);
	BOOST_CHECK( abs(std::accumulate(data, data+12, 0.0)) < 0.0000001);

	auto stdev = GetSameStd(tensors);
	DivideByStd(tensors, stdev);
	BOOST_CHECK( abs(data[0]*data[0]+data[1]*data[1]+data[2]*data[2]+data[3]*data[3]+
			data[4]*data[4]+data[5]*data[5]+data[6]*data[6]+data[7]*data[7]+
			data[8]*data[8]+data[9]*data[9]+data[10]*data[10]+data[11]*data[11] - 12)<0.000001);
}

BOOST_AUTO_TEST_CASE(testFullMeanAndStd)
{	
	
	double data[] = { 1, 5, 3,
	                   2, 9, 2,
	
	                   5, 1, 1,
	                   4, 4, 5,};

	std::vector<size_t> data_dims;data_dims.push_back(3), data_dims.push_back(2);

	std::vector< std::shared_ptr< Tensor<double> > > tensors;
	tensors.push_back( std::shared_ptr< Tensor<double> >( new Tensor<double>(data, data_dims)) );
	tensors.push_back( std::shared_ptr< Tensor<double> >( new Tensor<double>(data+6, data_dims)) );

	auto means = GetFullMeans(tensors);
	FullMeanSubtract(tensors, means);
	BOOST_CHECK_EQUAL( data[0]+data[6] , 0);
	BOOST_CHECK_EQUAL( data[1]+data[7] , 0);
	BOOST_CHECK_EQUAL( data[2]+data[8] , 0);
	BOOST_CHECK_EQUAL( data[3]+data[9] , 0);
	BOOST_CHECK_EQUAL( data[4]+data[10] , 0);
	BOOST_CHECK_EQUAL( data[5]+data[11] , 0);

	auto stds = GetFullStd(tensors);
	FullStdDivide(tensors, stds);
	BOOST_CHECK( abs(data[0]*data[0]+data[6]*data[6]-2) <0.000000001);
	BOOST_CHECK( abs(data[1]*data[1]+data[7]*data[7]-2) <0.000000001);
	BOOST_CHECK( abs(data[2]*data[2]+data[8]*data[8]-2) <0.000000001);
	BOOST_CHECK( abs(data[3]*data[3]+data[9]*data[9]-2) <0.000000001);
	BOOST_CHECK( abs(data[4]*data[4]+data[10]*data[10]-2) <0.000000001);
	BOOST_CHECK( abs(data[5]*data[5]+data[11]*data[11]-2) <0.000000001);
}