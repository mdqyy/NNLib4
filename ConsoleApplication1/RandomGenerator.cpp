#include "RandomGenerator.h"


// I could not make Visual Studio link to the original cpp file. 
// It is the only cpp that has to be imported, therefore I copied it here

#include <random>
std::mt19937 gen = std::mt19937();

// max inclusive
int RandomGenerator::GetUniformInt(int min_val, int max_val)
{
	std::uniform_int_distribution<int> dist(min_val, max_val);
	return dist(gen);
}

double RandomGenerator::GetUniformDouble(double min_val, double max_val)
{
	std::uniform_real_distribution<double> dist(min_val, max_val);
	return dist(gen);
}

double RandomGenerator::GetNormalDouble(double mean, double std)
{
	std::normal_distribution<double> dist(mean, std);
	return dist(gen);
}