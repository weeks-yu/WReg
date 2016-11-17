#pragma once

#include "PairwiseRegister.h"
#include "PointCloud.h"
#include <pcl/registration/gicp.h>

class GeneralizedIcpRegister : public PairwiseRegister
{
public:
	GeneralizedIcpRegister();
	GeneralizedIcpRegister(float dist);
	virtual ~GeneralizedIcpRegister();

	virtual bool getTransformation(void *prev, void *now, Eigen::Matrix4f &tran);
	virtual float getCorrespondencePercent(void *last, void *now, Eigen::Matrix4f &estimatedTran);
	virtual void setParameters(void **parameters);

private:
	float dist;
};