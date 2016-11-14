#pragma once

#include "PairwiseRegister.h"
#include "ICPOdometry.h"

class IcpcudaRegister : public PairwiseRegister
{
public:
	IcpcudaRegister();
	IcpcudaRegister(ICPOdometry *icpcuda, int threads, int blocks, float depthCutOff);
	virtual ~IcpcudaRegister();

	virtual bool getTransformation(void *prev, void *now, Eigen::Matrix4f &tran);
	virtual float getCorrespondencePercent(void *last, void *now, Eigen::Matrix4f &estimatedTran);
	virtual void setParameters(void **parameters);

private:
	ICPOdometry *icpcuda;
	int threads;
	int blocks;
	float depthCutOff;
};