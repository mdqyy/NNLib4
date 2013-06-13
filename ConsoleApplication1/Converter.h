#ifndef CONVERT_H
#define CONVERT_H

#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <limits>
#include "strtk\strtk.hpp"

class Converter
{
	static void split(const std::string& str, char delimeter, std::vector<std::string>& elems) 
	{
		std::stringstream ss(str);
		std::string item;
		while (std::getline(ss, item, delimeter)) 
			elems.push_back(item);
	}

public:
	
	template <class T>
	static T ConvertTo(std::string const& s)
	{
	  std::istringstream i(s);
	  T x;
	  if (!(i >> x))
		throw "Error: convertion of (\"" + s + "\")";
	  return x;
	}

	static std::vector<std::string> split(const std::string &str, char delimeter) 
	{
		std::vector<std::string> elems;
		split(str, delimeter, elems);
		return elems;
	}
	
	template <class T>
	static std::vector<T> StringToVector(std::string const& s, char delimeter=' ')
	{
		std::vector<T> res;
		std::vector<std::string> strings = split(s, delimeter);
		for (size_t i=0; i<strings.size(); i++)
			res.push_back( ConvertTo<T>(strings[i]) );
		return res;
	}

	template <>
	static std::vector<double> StringToVector<double>(std::string const& s, char delimeter)
	{
		std::string str_delimeter( 1, delimeter );
		std::vector<double> res;
		strtk::parse(s, str_delimeter ,res);
		return res;
	}

	template <>
	static std::vector<float> StringToVector<float>(std::string const& s, char delimeter)
	{
		std::string str_delimeter( 1, delimeter );
		std::vector<float> res;
		strtk::parse(s, str_delimeter ,res);
		return res;
	}
	
	template <>
	static std::vector<size_t> StringToVector<size_t>(std::string const& s, char delimeter)
	{
		std::string str_delimeter( 1, delimeter );
		std::vector<size_t> res;
		strtk::parse(s, str_delimeter ,res);
		return res;
	}

	template <>
	static std::vector<int> StringToVector<int>(std::string const& s, char delimeter)
	{
		std::string str_delimeter( 1, delimeter );
		std::vector<int> res;
		strtk::parse(s, str_delimeter ,res);
		return res;
	}

	template <class T>
	static std::string ConvertVectorToString(const std::vector<T>& vect, char delimeter=' ')
	{
		std::ostringstream oss;
		oss.flags (std::ios::scientific);
		oss.precision (std::numeric_limits<T>::digits10 + 1);
		if (!vect.empty())
		{
			std::string str_delimeter(1, delimeter);
			std::copy(vect.begin(), vect.end()-1, std::ostream_iterator<T>(oss, str_delimeter.c_str()));
			oss << vect.back();
		}
		return oss.str();
	}
	
	template <class T>
	static std::string ConvertArrayToString(const T* arr, size_t num_elements, char delimeter=' ')
	{
		std::ostringstream oss;
		oss.flags (std::ios::scientific);
		oss.precision (std::numeric_limits<T>::digits10 + 1);
		if ( num_elements != 0 )
		{
			std::string str_delimeter(1, delimeter);
			std::copy(arr, arr+num_elements-1, std::ostream_iterator<T>(oss, str_delimeter.c_str()));
			oss << arr[num_elements-1];
		}
		return oss.str();
	}
};

#endif