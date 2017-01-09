#pragma once

#include "PointCloud.h"

class PointCloudModel
{
public:
	PointCloudModel();
	PointCloudModel(float dr);
	~PointCloudModel();

	void dataFusion(PointCloudPtr cloud);
	PointCloudPtr getModel();

	void setDownsampleRate(float dr);

private:
	PointCloudPtr temp;
	PointCloudPtr model;

	float downsample_rate;
};