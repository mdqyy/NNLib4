#ifndef MATRIX_OPERATIONS_H_
#define MATRIX_OPERATIONS_H_

#include <algorithm>
#include "my_math.h"
#include "cblas.h"

template <class T>
void MatrixMultiply(CBLAS_ORDER order, CBLAS_TRANSPOSE transpose_A, CBLAS_TRANSPOSE transpose_B, size_t M, size_t N, size_t K, 
	T alpha, const T *A, size_t lda, const T *B, size_t ldb, T beta, T* C, size_t ldc)
{
	throw "Not implemented";
}

template <>
inline void MatrixMultiply<float>(CBLAS_ORDER order, CBLAS_TRANSPOSE transpose_A, CBLAS_TRANSPOSE transpose_B, size_t M, size_t N, size_t K, 
	float alpha, const float *A, size_t lda, const float *B, size_t ldb, float beta, float* C, size_t ldc)
{
	cblas_sgemm(order, transpose_A, transpose_B, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline void MatrixMultiply<double>(CBLAS_ORDER order, CBLAS_TRANSPOSE transpose_A, CBLAS_TRANSPOSE transpose_B, size_t M, size_t N, size_t K, 
	double alpha, const double *A, size_t lda, const double *B, size_t ldb, double beta, double* C, size_t ldc)
{
	cblas_dgemm(order, transpose_A, transpose_B, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <class T>
T dot_product(const T* x1, const T* x2, size_t num_elements)
{
	T res = 0;
	for (size_t i=0; i<num_elements; i++)
		res += x1[i]*x2[i];
	return res;
}

template <>
inline double dot_product<double>(const double* x1, const double* x2, size_t num_elements)
{
	return cblas_ddot(num_elements, x1, 1, x2, 1);
}

template <>
inline float dot_product<float>(const float* x1, const float* x2, size_t num_elements)
{
	return cblas_sdot(num_elements, x1, 1, x2, 1);
}

template <class T>
void swap(T* x1, T* x2, size_t num_elements)
{
	std::swap_ranges(x1, x1+num_elements, x2)
}

template <>
inline void swap<double>(double* x1, double* x2, size_t num_elements)
{
	cblas_dswap(num_elements, x1, 1, x2, 1);
}

template <>
inline void swap<float>(float* x1, float* x2, size_t num_elements)
{
	cblas_sswap(num_elements, x1, 1, x2, 1);
}

// y <- alpha*x+y
template <class T>
void axpy(const T* x, T* y, size_t num_elements, T alpha)
{
	for (size_t i=0; i<num_elements; i++)
		y[i] = alpha*x[i]+y[i];
}

// y <- alpha*x+y
template <>
inline void axpy<double>(const double* x, double* y, size_t num_elements, double alpha)
{
	cblas_daxpy(num_elements, alpha, x, 1, y, 1);
}

// y <- alpha*x+y
template <>
inline void axpy<float>(const float* x, float* y, size_t num_elements, float alpha)
{
	cblas_saxpy(num_elements, alpha, x, 1, y, 1);
}

// y <- alpha*x+y
template <class T>
void scale(T* x, size_t num_elements, T alpha)
{
	for (size_t i=0; i<num_elements; i++)
		x[i] *= alpha;
}

// y <- alpha*x+y
template <>
inline void scale<double>(double* x, size_t num_elements, double alpha)
{
	cblas_dscal(num_elements, alpha, x, 1);
}

// y <- alpha*x+y
template <>
inline void scale<float>(float* x, size_t num_elements, float alpha)
{
	cblas_sscal(num_elements, alpha, x, 1);
}

// euclidian norm
template <class T>
T norm(const T* x, size_t num_elements)
{
	T res = 0;
	for (size_t i=0; i<num_elements; i++)
		res += sqr(x[i]);
	return res;
}

// euclidian norm
template <>
inline double norm<double>(const double* x, size_t num_elements)
{
	return cblas_dnrm2(num_elements, x, 1);
}

// euclidian norm
template <>
inline float norm<float>(const float* x, size_t num_elements)
{
	return cblas_snrm2(num_elements, x, 1);
}

template <class T>
T abs_sum(const T* x, size_t num_elements)
{
	T res = 0;
	for (size_t i=0; i<num_elements; i++)
		res += abs(x[i]);
	return res;
}

template <>
inline double abs_sum<double>(const double* x, size_t num_elements)
{
	return cblas_dasum(num_elements, x, 1);
}

template <>
inline float abs_sum<float>(const float* x, size_t num_elements)
{
	return cblas_sasum(num_elements, x, 1);
}

template <class T>
void copy(const T* from, T* to, size_t num_elements)
{
	std::copy(from, from + num_elements, to);
}

template <>
inline void copy<double>(const double* from, double* to, size_t num_elements)
{
	return cblas_dcopy(num_elements, from, 1, to, 1);
}

template <>
inline void copy<float>(const float* from, float* to, size_t num_elements)
{
	return cblas_scopy(num_elements, from, 1, to, 1);
}

#endif