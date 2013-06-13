#include <boost/test/unit_test.hpp>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "FullTensorDataLoader.h"
#include "test_utilities.h"
#include "DataLoaderFactory.h"

BOOST_AUTO_TEST_CASE(test_full_tensor_data_loader)
{
	float input[] = {1,4,6,2,4,
					 7,9,7,8,0,
					 3,8,3,6,9,
					 3,5,8,9,0,
					 4,6,8,1,2,
					 8,9,1,4,1,
					 
					 3,2,5,8,9,
					 4,1,1,4,6,
					 8,9,4,6,3,
					 7,1,4,9,5,
					 9,1,8,2,5,
					 1,5,2,4,8};

	std::vector<size_t> input_dims; input_dims.push_back(5); input_dims.push_back(6); input_dims.push_back(2);
	std::vector<size_t> per_case_input_dims=input_dims;per_case_input_dims.pop_back();
	std::vector< std::shared_ptr< Tensor<float> > > input_cases;
	input_cases.push_back(std::shared_ptr< Tensor<float> >(new Tensor<float>(input, per_case_input_dims)));
	input_cases.push_back(std::shared_ptr< Tensor<float> >(new Tensor<float>(input+30, per_case_input_dims)));

	FullTensorDataLoader<float,float> data_loader(input_cases);

	std::vector<size_t> inds;inds.push_back(0);
	std::vector<size_t> full_per_case_input_dims = per_case_input_dims; full_per_case_input_dims.push_back(1);
	std::shared_ptr< Tensor<float> > samples = data_loader.GetData(inds);
	BOOST_CHECK(samples->GetDimensions() == full_per_case_input_dims);
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), input, 30));

	inds.clear();
	inds.push_back(1);
	samples = data_loader.GetData(inds);
	BOOST_CHECK(samples->GetDimensions() == full_per_case_input_dims);
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), input+30, 30));
	
	inds.clear();
	inds.push_back(0);
	inds.push_back(1);
	samples = data_loader.GetData(inds);
	BOOST_CHECK(samples->GetDimensions() == input_dims);
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), input, 60));
}

BOOST_AUTO_TEST_CASE(test_full_tensor_data_loader_save_load)
{
	std::vector< std::shared_ptr< Tensor<float> > > input(15);
	std::vector<size_t> case_dims;case_dims.push_back(9);case_dims.push_back(5);case_dims.push_back(4);
	
	for (size_t i=0; i<input.size(); i++)
		input[i] = GetRandomTensorPtr<float>(case_dims);
	
	FullTensorDataLoader<double,float> data_loader(input);

	std::stringstream stream;
	data_loader.Save(stream);

	std::shared_ptr< FullTensorDataLoader<double,float> > data_loader2 = 
		std::static_pointer_cast<  FullTensorDataLoader<double,float> >(DataLoaderFactory::GetDataLoader<double, float>(stream));

	BOOST_CHECK(data_loader.Equals(*data_loader2));
}