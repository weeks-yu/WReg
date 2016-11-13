#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

class PairwiseRegister
{
public:
	PairwiseRegister();
	virtual ~PairwiseRegister();

	virtual bool getTransformation(void *prev, void *now, Eigen::Matrix4f &tran) = 0;
	virtual float getCorrespondencePercent(void *last, void *now, Eigen::Matrix4f &estimatedTran) = 0;
	virtual void setParameters(void **parameters) = 0;
};