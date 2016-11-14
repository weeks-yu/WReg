#pragma once

#include "PairwiseRegister.h"

class OrbRegister : public PairwiseRegister
{
public:
	OrbRegister();
	OrbRegister(int min_matches, float inlier_percentage, float inlier_dist);
	virtual ~OrbRegister();

	virtual bool getTransformation(void *prev, void *now, Eigen::Matrix4f &tran);
	virtual float getCorrespondencePercent(void *last, void *now, Eigen::Matrix4f &estimatedTran);
	virtual void setParameters(void **parameters);

private:
	int min_matches;
	float inlier_percentage;
	float inlier_dist;
};