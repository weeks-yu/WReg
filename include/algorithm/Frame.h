#pragma once

#include "Feature.h"

class Frame
{
public:

	Feature f;
	Eigen::Matrix4f tran;

public:

	Frame()
	{
		tran = Eigen::Matrix4f::Identity();
	}

	Frame(const cv::Mat &imgRGB, const cv::Mat &imgDepth, string type = "SURF", const Eigen::Matrix4f &tran = Eigen::Matrix4f::Identity())
	{
		this->f.extract(imgRGB, imgDepth, type);
		this->f.updateFeaturePoints3D(tran);
		this->tran = tran;
	}
};