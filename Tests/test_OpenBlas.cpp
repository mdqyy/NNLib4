#include <boost/test/unit_test.hpp>
#include <vector>
#include <cblas.h>

BOOST_AUTO_TEST_CASE(test_openBLAS)
{
	float a[] = {1, 2, 3, 4, 
				 5, 2, 4, 1, 
				 2, 8, 8, 4};
	float b[] = {1,2,3};
	float c[] = {0,0,0,0};

	cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, 1, 3, 1, a, 4, b, 3, 0,c, 4);
	BOOST_CHECK_EQUAL(c[0] , 17);
	BOOST_CHECK_EQUAL(c[1] , 30);
	BOOST_CHECK_EQUAL(c[2] , 35);
	BOOST_CHECK_EQUAL(c[3] , 18);
}