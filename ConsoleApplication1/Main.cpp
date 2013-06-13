#include <cblas.h>
#include <iostream>
#include <Windows.h>
#include "Utilities.h"
#include "SemisupervisedLearning2013.h"
#include "UnsupervisedFeatureGroupProvider.h"

//void test_openBLAS()
//{
//	const int N = 5000;
//	float* big_matrix1 = new float[N*N];
//	float* big_matrix2 = new float[N*N];
//	float* big_matrix3 = new float[N*N];
//
//	for (int row = 0; row<N; row++)
//		for (int col=0; col<N; col++)
//		{
//			big_matrix1[row+N*col] = (row+col)/10000;
//			big_matrix2[row+N*col] = (row+col)/1000;
//		}
//	long long start = milliseconds_now();
//	//cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, big_matrix1, N, big_matrix2, N, 0,big_matrix3, N);
//	axpy(big_matrix1, big_matrix2, N*N, 0.5f);
//	long long elapsed = milliseconds_now() - start;
//	std::cout<<elapsed;
//}

int main( int argc, char* argv[])
{
	//test_openBLAS();
	semisupervised_learning_main();
}