#include <boost/test/unit_test.hpp>
#include <sstream>
#include "test_utilities.h"
#include "TensorIO.h"

BOOST_AUTO_TEST_CASE(TestSaveLoadTensor)
{
	std::vector<size_t> case_dims;case_dims.push_back(9); case_dims.push_back(8);
	std::shared_ptr< Tensor<double> > input = GetRandomTensorPtr<double>(case_dims);
	
	std::shared_ptr<IOTreeNode> tensor_state = GetTensorState(*input);
	std::shared_ptr< Tensor<double> > input2 = CreateTensor<double>(*tensor_state);

	BOOST_CHECK( *input == *input2 );
}

BOOST_AUTO_TEST_CASE(TestSaveLoadTensorDataset)
{
	std::vector<size_t> case_dims; case_dims.push_back(9); case_dims.push_back(8);
	std::vector< std::shared_ptr< Tensor<double> > > input(15);

	for (size_t i=0; i<input.size(); i++)
		input[i] = GetRandomTensorPtr<double>(case_dims);

	std::stringstream stream;

	// write twice. If tensor reads to the end of stream, we will see it 
	SaveDataset( input, stream);
	SaveDataset( input, stream);

	std::vector< std::shared_ptr< Tensor<double> > > input2;

	LoadDataset(stream, input2);

	BOOST_CHECK( input2.size() == input.size() );

	bool all_tensors_equal = true;
	for (size_t i=0; i < input.size(); i++)
		if ( *input[i] != *input2[i] )
		{
			all_tensors_equal = false;
			break;
		}
	BOOST_CHECK( all_tensors_equal );

	input2.clear();
	LoadDataset(stream, input2);
	BOOST_CHECK( input2.size() == input.size() );
	all_tensors_equal = true;
	for (size_t i=0; i < input.size(); i++)
		if ( *input[i] != *input2[i] )
		{
			all_tensors_equal = false;
			break;
		}
	BOOST_CHECK( all_tensors_equal );
}