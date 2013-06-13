#include <boost/test/unit_test.hpp>
#include <vector>
#include <memory>
#include <sstream>
#include "Tensor.h"
#include "RandomShiftPartialTensorDataLoader.h"
#include "test_utilities.h"
#include "DataLoaderFactory.h"

BOOST_AUTO_TEST_CASE(test_random_shift_partial_tensor_data_loader)
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
	std::vector<size_t> full_per_case_input_dims = per_case_input_dims; full_per_case_input_dims.push_back(1);
	std::vector< std::shared_ptr< Tensor<float> > > input_cases;
	input_cases.push_back(std::shared_ptr< Tensor<float> >(new Tensor<float>(input, per_case_input_dims)));
	input_cases.push_back(std::shared_ptr< Tensor<float> >(new Tensor<float>(input+30, per_case_input_dims)));

	// test with zero offsets
	std::vector<size_t> max_offsets; max_offsets.push_back(0); max_offsets.push_back(0);
	RandomShiftPartialTensorDataLoader<float,float> data_loader1(input_cases, max_offsets);

	std::vector<size_t> inds;inds.push_back(0);
	std::shared_ptr< Tensor<float> > samples = data_loader1.GetData(inds);
	BOOST_CHECK(samples->GetDimensions() == full_per_case_input_dims);
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), input, 30));

	inds.clear();
	inds.push_back(1);
	samples = data_loader1.GetData(inds);
	BOOST_CHECK(samples->GetDimensions() == full_per_case_input_dims);
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), input+30, 30));
		
	inds.clear();
	inds.push_back(0);
	inds.push_back(1);
	samples = data_loader1.GetData(inds);
	BOOST_CHECK(samples->GetDimensions() == input_dims);
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), input, 60));
	
	inds.clear();
	inds.push_back(1);
	inds.push_back(0);
	samples = data_loader1.GetData(inds);
	BOOST_CHECK(samples->GetDimensions() == input_dims);
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr()+30, input, 30));
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), input+30, 30));
	
	// test with nonzero offsets along the first dimension
	max_offsets[0] = 1;
	max_offsets[1] = 0;
	RandomShiftPartialTensorDataLoader<float,float> data_loader2(input_cases, max_offsets);

	input_dims[0]-=1;
	per_case_input_dims[0]-=1;
	full_per_case_input_dims[0] -= 1;

	inds.clear();inds.push_back(0);
	samples = data_loader2.GetData(inds);
	BOOST_CHECK(samples->GetDimensions() == full_per_case_input_dims);

	std::vector< std::vector<size_t> > samples_left_offsets(1);samples_left_offsets[0].push_back(1);samples_left_offsets[0].push_back(0);
	std::vector< std::vector<size_t> > samples_right_offsets(1);samples_right_offsets[0].push_back(0);samples_right_offsets[0].push_back(0);
	samples = data_loader2.GetSamplesData(inds, samples_left_offsets, samples_right_offsets);
	BOOST_CHECK(samples->GetDimensions() == full_per_case_input_dims);

	float q1[] = {4,6,2,4, 9,7,8,0, 8,3,6,9, 5,8,9,0, 6,8,1,2, 9,1,4,1};
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), q1, 24));

	inds.clear();
	inds.push_back(1);
	samples = data_loader2.GetData(inds);
	BOOST_CHECK(samples->GetDimensions() == full_per_case_input_dims);
	samples_left_offsets[0][0] = 0;samples_right_offsets[0][0] = 1;
	samples = data_loader2.GetSamplesData(inds, samples_left_offsets, samples_right_offsets);
	BOOST_CHECK(samples->GetDimensions() == full_per_case_input_dims);
	float q2[] = {3,2,5,8, 4,1,1,4, 8,9,4,6, 7,1,4,9, 9,1,8,2, 1,5,2,4};
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), q2, 24));
	
	inds.clear(); inds.push_back(1);inds.push_back(0);
	samples = data_loader2.GetData(inds);
	BOOST_CHECK(samples->GetDimensions() == input_dims);

	samples_left_offsets.clear();
	samples_right_offsets.clear();
	samples_left_offsets.push_back(std::vector<size_t>(2));
	samples_left_offsets[0][0]=0;
	samples_left_offsets[0][1]=0;
	samples_left_offsets.push_back(std::vector<size_t>(2));
	samples_left_offsets[1][0]=1;
	samples_left_offsets[1][1]=0;
	samples_right_offsets.clear();
	samples_right_offsets.push_back(std::vector<size_t>(2));
	samples_right_offsets[0][0]=1;
	samples_right_offsets[0][1]=0;
	samples_right_offsets.push_back(std::vector<size_t>(2));
	samples_right_offsets[1][0]=0;
	samples_right_offsets[1][1]=0;

	samples = data_loader2.GetSamplesData(inds, samples_left_offsets, samples_right_offsets);
	BOOST_CHECK(samples->GetDimensions() == input_dims);
	float q3[] = {3,2,5,8, 4,1,1,4, 8,9,4,6, 7,1,4,9, 9,1,8,2, 1,5,2,4, 4,6,2,4, 9,7,8,0, 8,3,6,9, 5,8,9,0, 6,8,1,2, 9,1,4,1};
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), q3, 48));
	
	// along the second dimension
	input_dims[0]+=1;
	per_case_input_dims[0]+=1;
	full_per_case_input_dims[0] += 1;
	input_dims[1]-=1;
	per_case_input_dims[1]-=1;
	full_per_case_input_dims[1] -= 1;
	inds.clear(); inds.push_back(0);inds.push_back(1);
	samples_left_offsets.clear();
	samples_right_offsets.clear();
	samples_left_offsets.push_back(std::vector<size_t>(2));
	samples_left_offsets[0][0]=0;
	samples_left_offsets[0][1]=0;
	samples_left_offsets.push_back(std::vector<size_t>(2));
	samples_left_offsets[1][0]=0;
	samples_left_offsets[1][1]=1;
	samples_right_offsets.clear();
	samples_right_offsets.push_back(std::vector<size_t>(2));
	samples_right_offsets[0][0]=0;
	samples_right_offsets[0][1]=1;
	samples_right_offsets.push_back(std::vector<size_t>(2));
	samples_right_offsets[1][0]=0;
	samples_right_offsets[1][1]=0;
	
	inds.clear(); inds.push_back(0);inds.push_back(1);

	max_offsets[0] = 0;
	max_offsets[1] = 1;
	RandomShiftPartialTensorDataLoader<float,float> data_loader3(input_cases, max_offsets);
	samples = data_loader3.GetData(inds);
	BOOST_CHECK(samples->GetDimensions() == input_dims);

	float q4[] = {1,4,6,2,4, 7,9,7,8,0, 3,8,3,6,9, 3,5,8,9,0, 4,6,8,1,2, 4,1,1,4,6, 8,9,4,6,3, 7,1,4,9,5, 9,1,8,2,5, 1,5,2,4,8};
	samples = data_loader2.GetSamplesData(inds, samples_left_offsets, samples_right_offsets);
	BOOST_CHECK(samples->GetDimensions() == input_dims);
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), q4, 50));
	
	// along both dimensions
	input_dims[0]-=1;
	per_case_input_dims[0]-=1;
	full_per_case_input_dims[0] -= 1;
	inds.clear(); inds.push_back(1);inds.push_back(0);
	samples_left_offsets.clear();
	samples_right_offsets.clear();
	samples_left_offsets.push_back(std::vector<size_t>(2));
	samples_left_offsets[0][0]=1;
	samples_left_offsets[0][1]=1;
	samples_left_offsets.push_back(std::vector<size_t>(2));
	samples_left_offsets[1][0]=0;
	samples_left_offsets[1][1]=0;
	samples_right_offsets.clear();
	samples_right_offsets.push_back(std::vector<size_t>(2));
	samples_right_offsets[0][0]=0;
	samples_right_offsets[0][1]=0;
	samples_right_offsets.push_back(std::vector<size_t>(2));
	samples_right_offsets[1][0]=1;
	samples_right_offsets[1][1]=1;
	
	inds.clear(); inds.push_back(1);inds.push_back(0);

	max_offsets[0] = 1;
	max_offsets[1] = 1;
	RandomShiftPartialTensorDataLoader<float,float> data_loader4(input_cases, max_offsets);
	samples = data_loader4.GetData(inds);
	BOOST_CHECK(samples->GetDimensions() == input_dims);

	float q5[] = { 1,1,4,6, 9,4,6,3, 1,4,9,5, 1,8,2,5, 5,2,4,8, 1,4,6,2, 7,9,7,8, 3,8,3,6, 3,5,8,9, 4,6,8,1};
	samples = data_loader2.GetSamplesData(inds, samples_left_offsets, samples_right_offsets);
	BOOST_CHECK(samples->GetDimensions() == input_dims);
	BOOST_CHECK(test_equal_arrays(samples->GetStartPtr(), q5, 40));
}

BOOST_AUTO_TEST_CASE(test_random_shift_partial_tensor_data_loader_save_load)
{
	std::vector< std::shared_ptr< Tensor<float> > > input(15);
	std::vector<size_t> case_dims;case_dims.push_back(9);case_dims.push_back(5);case_dims.push_back(4);
	
	for (size_t i=0; i<input.size(); i++)
		input[i] = GetRandomTensorPtr<float>(case_dims);
	
	std::vector<size_t> max_offsets; max_offsets.push_back(4); max_offsets.push_back(3); max_offsets.push_back(2);
	RandomShiftPartialTensorDataLoader<double,float> data_loader(input, max_offsets);

	std::stringstream stream;
	data_loader.Save(stream);

	std::shared_ptr< RandomShiftPartialTensorDataLoader<double,float> > data_loader2 = 
		std::static_pointer_cast<  RandomShiftPartialTensorDataLoader<double,float> >(DataLoaderFactory::GetDataLoader<double, float>(stream));

	BOOST_CHECK(data_loader.Equals(*data_loader2));
}