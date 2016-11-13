#pragma once

#include "Feature.h"

typedef Eigen::Matrix<double, 6, 6> InformationMatrix;

class Frame
{
public:

	Feature *f;
	Eigen::Matrix4f tran;
	Eigen::Matrix4f relative_tran;
	bool ransac_failed;

public:

	Frame()
	{
		tran = Eigen::Matrix4f::Identity();
		f = nullptr;
		ransac_failed = false;
	}

	Frame(const cv::Mat &imgRGB, const cv::Mat &imgDepth, string type = "surf", const Eigen::Matrix4f &tran = Eigen::Matrix4f::Identity())
	{
		f = new Feature();
		f->extract(imgRGB, imgDepth, type);
		ransac_failed = false;
		this->tran = tran;
	}

	~Frame()
	{
		if (f)
		{
			delete f;
		}
	}
};