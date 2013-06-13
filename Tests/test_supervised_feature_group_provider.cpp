#include <boost/test/unit_test.hpp>
#include <vector>
#include "Tensor.h"
#include "SupervisedFeatureGroupProvider.h"

BOOST_AUTO_TEST_CASE(TestSupervisedFeatureGroupProvider)
{
	std::vector<size_t> output_dims;output_dims.push_back(4); output_dims.push_back(5);
	float output[] = { 0.1f,  0.15f, 0.5f,  0.21f,
					   0.05f, 0.18f, 0.04f, 0.09f,
					   0.23f, 0.17f, 0.16f, 0.19f,
					   0.35f, 0.45f, 0.21f, 0.4f,
					   0.27f, 0.05f, 0.09f, 0.11f };
	Tensor<float> output_tensor(output, output_dims);

	float labels[] = { 0, 0, 1,
					1, 0, 0,
					0, 1, 0,
					1, 0, 0,
					0, 0, 1 };
	std::vector<size_t> labels_dims;labels_dims.push_back(3); labels_dims.push_back(5);
	Tensor<float> labels_tensor(labels, labels_dims);

	SupervisedFeatureGroupProvider<float> group_provider;

	FeatureGroups groups = group_provider.GetGroups(&output_tensor, &labels_tensor, 0);

	BOOST_CHECK( groups.size() == 3);
	
	BOOST_CHECK( (groups.GetGroup(0).size() == 2) && (groups.GetGroup(0)[0] == 1) && (groups.GetGroup(0)[1] == 3));
	BOOST_CHECK( (groups.GetGroup(1).size() == 1) && (groups.GetGroup(1)[0] == 2));
	BOOST_CHECK( (groups.GetGroup(2).size() == 2) && (groups.GetGroup(2)[0] == 0) && (groups.GetGroup(2)[1] == 4));

	BOOST_CHECK( group_provider.IsFeatureIndependent() );
	
	groups = group_provider.GetGroups(&output_tensor, &labels_tensor, 1);
	
	BOOST_CHECK( (groups.GetGroup(0).size() == 2) && (groups.GetGroup(0)[0] == 1) && (groups.GetGroup(0)[1] == 3));
	BOOST_CHECK( (groups.GetGroup(1).size() == 1) && (groups.GetGroup(1)[0] == 2));
	BOOST_CHECK( (groups.GetGroup(2).size() == 2) && (groups.GetGroup(2)[0] == 0) && (groups.GetGroup(2)[1] == 4));
	
	groups = group_provider.GetGroups(&output_tensor, &labels_tensor, 2);
	
	BOOST_CHECK( (groups.GetGroup(0).size() == 2) && (groups.GetGroup(0)[0] == 1) && (groups.GetGroup(0)[1] == 3));
	BOOST_CHECK( (groups.GetGroup(1).size() == 1) && (groups.GetGroup(1)[0] == 2));
	BOOST_CHECK( (groups.GetGroup(2).size() == 2) && (groups.GetGroup(2)[0] == 0) && (groups.GetGroup(2)[1] == 4));
}