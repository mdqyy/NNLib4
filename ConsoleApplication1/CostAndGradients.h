#ifndef COST_AND_GRADIENTS_H
#define COST_AND_GRADIENTS_H

template <class ParamsType>
class CostAndGradients
{
public:
	double cost;
	std::vector<ParamsType>& gradients;
	CostAndGradients(double cost, std::vector<ParamsType>& gradients) : cost(cost), gradients(gradients)
	{

	}
};

#endif