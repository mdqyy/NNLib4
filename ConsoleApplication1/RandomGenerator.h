#ifndef RANDOMGENERATOR_H
#define RANDOMGENERATOR_H

class RandomGenerator
{
public:
	static int GetUniformInt(int min_val, int max_val);

	static double GetUniformDouble(double min_val, double max_val);

	static double GetNormalDouble(double mean, double std);
};

#endif