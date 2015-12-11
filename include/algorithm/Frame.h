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

	Frame(const cv::Mat &imgRGB, const cv::Mat &imgDepth, const Eigen::Matrix4f &tran = Eigen::Matrix4f::Identity(), string type = "SURF")
	{
		this->f.extract(imgRGB, imgDepth, tran, type);
		this->tran = tran;
	}
};