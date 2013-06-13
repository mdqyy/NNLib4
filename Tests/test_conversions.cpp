#include <boost/test/unit_test.hpp>
#include <vector>
#include "Converter.h"

BOOST_AUTO_TEST_CASE(TestConversions)
{
	std::vector<double> vect;
	vect.push_back(1.4);
	vect.push_back(5.1);
	vect.push_back(-1.5);
	vect.push_back(0);
	vect.push_back(0.05);
	std::string str = Converter::ConvertVectorToString(vect);
	std::vector<double> vect2 = Converter::StringToVector<double>(str);
	BOOST_CHECK( vect == vect2 );

	str = Converter::ConvertArrayToString(vect.data(), vect.size());
	vect2 = Converter::StringToVector<double>(str);
	BOOST_CHECK( vect == vect2 );

	str = Converter::ConvertArrayToString(vect.data(), 0);
	BOOST_CHECK( str == "" );
}