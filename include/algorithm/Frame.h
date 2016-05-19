#pragma once

#include "Feature.h"

class Frame
{
public:

	Feature *f;
	cv::Mat *depth;
	Eigen::Matrix4f tran;
	Eigen::Matrix4f relative_tran;

public:

	Frame()
	{
		tran = Eigen::Matrix4f::Identity();
		f = nullptr;
	}

	Frame(const cv::Mat &imgRGB, const cv::Mat &imgDepth, string type = "SURF", const Eigen::Matrix4f &tran = Eigen::Matrix4f::Identity())
	{
		this->f = new Feature();
		this->f->extract(imgRGB, imgDepth, type);
		this->f->updateFeaturePoints3D(tran);
		this->depth = new cv::Mat();
		imgDepth.copyTo(*this->depth);
		this->tran = tran;
	}

	~Frame()
	{
		if (f)
		{
			delete f;
		}

		if (depth)
		{
			delete depth;
		}
	}
};