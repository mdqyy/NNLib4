#include <boost/test/unit_test.hpp>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "FullTensorDataLoader.h"
#include "test_utilities.h"

BOOST_AUTO_TEST_CASE(test_train_dataset)
{
	float train_input[] = {1,4,6,2,4,
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

	float train_output[] = { 1, 2, 3, 4,
	                   5, 3, 2, 1,
	                   1, 8, 9, 2,
	
	                   5, 8, 1, 5,
	                   2, 4, 9, 8,
	                   5, 1, 2, 4};

	std::vector<float> train_importance;train_importance.push_back(1); train_importance.push_back(2);
	
	std::vector<size_t> train_input_dims; train_input_dims.push_back(5); train_input_dims.push_back(6); train_input_dims.push_back(2);
	std::vector<size_t> train_per_case_input_dims=train_input_dims;train_per_case_input_dims.pop_back();
	std::vector<size_t> train_output_dims; train_output_dims.push_back(4); train_output_dims.push_back(3); train_output_dims.push_back(2);
	std::vector<size_t> train_per_case_output_dims=train_output_dims;train_per_case_output_dims.pop_back();
	std::vector< std::shared_ptr< Tensor<float> > > train_input_cases;
	train_input_cases.push_back(std::shared_ptr< Tensor<float> >(new Tensor<float>(train_input, train_per_case_input_dims)));
	train_input_cases.push_back(std::shared_ptr< Tensor<float> >(new Tensor<float>(train_input+30, train_per_case_input_dims)));
	std::vector< std::shared_ptr< Tensor<float> > > train_output_cases;
	train_output_cases.push_back(std::shared_ptr< Tensor<float> >(new Tensor<float>(train_output, train_per_case_output_dims)));
	train_output_cases.push_back(std::shared_ptr< Tensor<float> >(new Tensor<float>(train_output+12, train_per_case_output_dims)));
	
	std::shared_ptr< ITensorDataLoader<float> > input_data_loader( new FullTensorDataLoader<float,float>(train_input_cases) );
	std::shared_ptr< ITensorDataLoader<float> > output_data_loader( new FullTensorDataLoader<float,float>(train_output_cases) );
	TrainDataset<float> dataset(input_data_loader, output_data_loader, train_importance);

	std::vector<size_t> inds;inds.push_back(0);
	std::vector<size_t> full_train_per_case_input_dims = train_per_case_input_dims; full_train_per_case_input_dims.push_back(1);
	std::vector<size_t> full_train_per_case_output_dims = train_per_case_output_dims; full_train_per_case_output_dims.push_back(1);

	BOOST_CHECK(dataset.GetInput(inds)->GetDimensions() == full_train_per_case_input_dims);
	BOOST_CHECK(dataset.GetOutput(inds)->GetDimensions() == full_train_per_case_output_dims);
	BOOST_CHECK(test_equal_arrays(dataset.GetInput(inds)->GetStartPtr(), train_input, 30));
	BOOST_CHECK(test_equal_arrays(dataset.GetOutput(inds)->GetStartPtr(), train_output, 12));
	BOOST_CHECK(dataset.GetImportance(inds).size() == 1 && dataset.GetImportance(inds)[0] == 1);

	inds.clear();
	inds.push_back(1);
	BOOST_CHECK(dataset.GetInput(inds)->GetDimensions() == full_train_per_case_input_dims);
	BOOST_CHECK(dataset.GetOutput(inds)->GetDimensions() == full_train_per_case_output_dims);
	BOOST_CHECK(test_equal_arrays(dataset.GetInput(inds)->GetStartPtr(), train_input+30, 30));
	BOOST_CHECK(test_equal_arrays(dataset.GetOutput(inds)->GetStartPtr(), train_output+12, 12));
	BOOST_CHECK(dataset.GetImportance(inds).size() == 1 && dataset.GetImportance(inds)[0] == 2);
		
	inds.clear();
	inds.push_back(0);
	inds.push_back(1);
	BOOST_CHECK(dataset.GetInput(inds)->GetDimensions() == train_input_dims);
	BOOST_CHECK(dataset.GetOutput(inds)->GetDimensions() == train_output_dims);
	BOOST_CHECK(test_equal_arrays(dataset.GetInput(inds)->GetStartPtr(), train_input, 60));
	BOOST_CHECK(test_equal_arrays(dataset.GetOutput(inds)->GetStartPtr(), train_output, 24));
	BOOST_CHECK(dataset.GetImportance(inds).size() == 2 && dataset.GetImportance(inds)[0] == 1 && dataset.GetImportance(inds)[1]==2);

	inds.clear();
	inds.push_back(1);
	inds.push_back(0);
	BOOST_CHECK(dataset.GetInput(inds)->GetDimensions() == train_input_dims);
	BOOST_CHECK(dataset.GetOutput(inds)->GetDimensions() == train_output_dims);
	BOOST_CHECK(test_equal_arrays(dataset.GetInput(inds)->GetStartPtr(), train_input+30, 30));
	BOOST_CHECK(test_equal_arrays(dataset.GetInput(inds)->GetStartPtr()+30, train_input, 30));
	BOOST_CHECK(test_equal_arrays(dataset.GetOutput(inds)->GetStartPtr(), train_output+12, 12));
	BOOST_CHECK(test_equal_arrays(dataset.GetOutput(inds)->GetStartPtr()+12, train_output, 12));
	BOOST_CHECK(dataset.GetImportance(inds).size() == 2 && dataset.GetImportance(inds)[0] == 2 && dataset.GetImportance(inds)[1]==1);
}