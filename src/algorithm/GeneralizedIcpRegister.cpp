#include "GeneralizedIcpRegister.h"
#include "Config.h"

GeneralizedIcpRegister::GeneralizedIcpRegister()
{
	this->dist = Config::instance()->get<float>("dist_threshold");
}

GeneralizedIcpRegister::GeneralizedIcpRegister(float dist)
{
	this->dist = dist;
}

GeneralizedIcpRegister::~GeneralizedIcpRegister()
{
}

bool GeneralizedIcpRegister::getTransformation(void *prev, void *now, Eigen::Matrix4f &tran)
{
	PointCloudPtr cloud_prev = *static_cast<PointCloudPtr *>(prev);
	PointCloudPtr cloud_now = *static_cast<PointCloudPtr *>(now);

	pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
	gicp.setInputSource(cloud_now);
	gicp.setInputTarget(cloud_prev);
	PointCloudT final;
	gicp.align(final);

	tran = gicp.getFinalTransformation();

	return true;
}

float GeneralizedIcpRegister::getCorrespondencePercent(void *last, void *now, Eigen::Matrix4f &estimatedTran)
{
	throw std::exception("Exception : not implemented.");
}

void GeneralizedIcpRegister::setParameters(void **parameters)
{
	dist = *static_cast<float *>(parameters[0]);
}