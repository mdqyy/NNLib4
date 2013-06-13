#include <boost/test/unit_test.hpp>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "FixedShiftPartialTensorDataLoader.h"
#include "test_utilities.h"
#include "DataLoaderFactory.h"

BOOST_AUTO_TEST_CASE(test_fixed_shift_partial_tensor_data_loader)
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
	
	std::vector<size_t> input_dims; input_dims.push_back(5); input_dims.push_back(6);
	std::vector<size_t> full_sample_dims = input_dims; full_sample_dims.push_back(1);
	std::vector< std::shared_ptr< Tensor<float> > > input_cases;
	input_cases.push_back(std::shared_ptr< Tensor<float> >(new Tensor<float>(input, input_dims)));
	input_cases.push_back(std::shared_ptr< Tensor<float> >(new Tensor<float>(input+30, input_dims)));

	// test with zero offsets
	std::vector<size_t> left_offsets; left_offsets.push_back(1); left_offsets.push_back(2);
	std::vector<size_t> right_offsets; right_offsets.push_back(0); right_offsets.push_back(1);
	FixedShiftPartialTensorDataLoader<float,float> data_loader1(input_cases, left_offsets, right_offsets);
	
	full_sample_dims[0] -= 1;
	full_sample_dims[1] -= 3;

	std::vector<size_t> inds;inds.push_back(0);
	std::shared_ptr< Tensor<float> > samples = data_loader1.GetData(inds);
	BOOST_CHECK(samples->GetDimensions() == full_sample_dims);
	float q1[] = {8,3,6,9, 5,8,9,0, 6,8,1,2};
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), q1, 12));

	inds.clear();
	inds.push_back(1);
	samples = data_loader1.GetData(inds);
	float q2[] = {9,4,6,3, 1,4,9,5, 1,8,2,5};

	BOOST_CHECK(samples->GetDimensions() == full_sample_dims);
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), q2, 12));
	
	full_sample_dims[2] = 2;
	inds.clear();
	inds.push_back(0);
	inds.push_back(1);
	samples = data_loader1.GetData(inds);
	float q3[] = {8,3,6,9, 5,8,9,0, 6,8,1,2,  9,4,6,3, 1,4,9,5, 1,8,2,5};
	BOOST_CHECK(samples->GetDimensions() == full_sample_dims);
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), q3, 24));
	
	inds.clear();
	inds.push_back(1);
	inds.push_back(0);
	samples = data_loader1.GetData(inds);
	float q4[] = {9,4,6,3, 1,4,9,5, 1,8,2,5,  8,3,6,9, 5,8,9,0, 6,8,1,2};
	BOOST_CHECK(samples->GetDimensions() == full_sample_dims);
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), q4, 24));
}

BOOST_AUTO_TEST_CASE(test_fixed_shift_partial_tensor_data_loader_save_load)
{
	std::vector< std::shared_ptr< Tensor<float> > > input(15);
	std::vector<size_t> case_dims;case_dims.push_back(9);case_dims.push_back(5);case_dims.push_back(4);
	
	for (size_t i=0; i<input.size(); i++)
		input[i] = GetRandomTensorPtr<float>(case_dims);
	
	std::vector<size_t> left_offsets; left_offsets.push_back(2); left_offsets.push_back(1); left_offsets.push_back(0);
	std::vector<size_t> right_offsets; right_offsets.push_back(3); right_offsets.push_back(2); right_offsets.push_back(1);
	std::vector<size_t> max_offsets; max_offsets.push_back(4); max_offsets.push_back(3); max_offsets.push_back(2);
	FixedShiftPartialTensorDataLoader<double,float> data_loader(input, left_offsets, right_offsets);

	std::stringstream stream;
	data_loader.Save(stream);

	std::shared_ptr< FixedShiftPartialTensorDataLoader<double,float> > data_loader2 = 
		std::static_pointer_cast<  FixedShiftPartialTensorDataLoader<double,float> >(DataLoaderFactory::GetDataLoader<double, float>(stream));

	BOOST_CHECK(data_loader.Equals(*data_loader2));
}