#ifndef MY_MATH_H
#define MY_MATH_H

#include <math.h>

const double MY_MATH_LOG2 = std::log(2);

template <typename OutputIterator, typename T>
void my_iota(std::size_t count, T start_value, OutputIterator out)
{
	T value = start_value;
    while (count!=0)
    {
        (*out) = value;
        ++out;
        --count;
		value++;
    }
}

template <typename T> int sign(T val) 
{
    return (T(0) < val) - (val < T(0));
}

template <typename T> T sqr(T x)
{
	return x*x;
}

template <class T>
T log2(T x)
{
	return (T)(std::log(x) / MY_MATH_LOG2);
}

template <typename T>
std::vector<size_t> GetOrderedIndices(std::vector<T> const& values) {
	std::vector<size_t> indices(values.size());
	my_iota(indices.size(), 0, indices.begin());

	std::sort(
		begin(indices), end(indices),
		[&](size_t a, size_t b) { return values[a] < values[b]; }
	);
	return indices;
}

inline std::vector<size_t> GetBatchSizes(size_t num_elements, size_t batch_size)
{
	size_t num_batches = num_elements / batch_size;
	size_t last_batch_size = num_elements - num_batches*batch_size;
	std::vector<size_t> batch_sizes(num_batches+1);
	for (size_t i=0; i<batch_sizes.size(); i++)
		batch_sizes[i] = batch_size;
	if(last_batch_size == 0)
		batch_sizes.pop_back();
	else
		batch_sizes[batch_sizes.size()-1] = last_batch_size;
	return batch_sizes;
}

inline std::vector<size_t> GetEqualSplitBatchSizes(size_t num_elements, size_t num_groups)
{
	size_t smalles_batch_size = num_elements/num_groups;
	std::vector<size_t> batch_sizes(num_groups, smalles_batch_size);
	size_t residual = num_elements - smalles_batch_size*num_groups;
	while (residual>0)
	{
		batch_sizes[residual-1]++;
		residual--;
	}
	return batch_sizes;
}

#endif