#include "RGBDDatas.h"

std::vector<RGBDFrame> RGBDDatas::frames;

RGBDFrame::RGBDFrame(cv::Mat color, cv::Mat depth, double timestamp)
{
	color.copyTo(this->color);
	depth.copyTo(this->depth);
	this->timestamp = timestamp;
}

void RGBDDatas::push(const cv::Mat &r, const cv::Mat &d, double ts)
{
	RGBDFrame f;
	r.copyTo(f.color);
	d.copyTo(f.depth);
	f.timestamp = ts;
	frames.push_back(f);
}

double RGBDDatas::getTimestamp(int k)
{
	if (k >= 0 && k < RGBDDatas::frames.size())
	{
		return RGBDDatas::frames[k].timestamp;
	}
	return 0;
}

RGBDFrame RGBDDatas::getFrame(int k)
{
	if (k >= 0 && k < RGBDDatas::frames.size())
	{
		return RGBDDatas::frames[k];
	}
	return RGBDFrame();
}

void RGBDDatas::getFrame(int k, cv::Mat &c, cv::Mat &d)
{
	if (k >= 0 && k < RGBDDatas::frames.size())
	{
		RGBDFrame f = RGBDDatas::frames[k];
		f.color.copyTo(c);
		f.depth.copyTo(d);
	}
}

PointCloudPtr RGBDDatas::getCloud(int k)
{
	if (k >= 0 && k < RGBDDatas::frames.size())
	{
		RGBDFrame f = RGBDDatas::frames[k];
		PointCloudPtr cloud = ConvertToPointCloudWithoutMissingData(f.depth, f.color, f.timestamp, k);
		return cloud;
	}
	return PointCloudPtr(new PointCloudT());
}

PointCloudPtr RGBDDatas::getOrganizedCloud(int k)
{
	if (k >= 0 && k < RGBDDatas::frames.size())
	{
		RGBDFrame f = RGBDDatas::frames[k];
		PointCloudPtr cloud = ConvertToOrganizedPointCloud(f.depth, f.color, f.timestamp, k);
		return cloud;
	}
	return PointCloudPtr(new PointCloudT());
}

void RGBDDatas::getCloudWithNormal(int k, PointCloudPtr &cloud, PointCloudNormalPtr &normal)
{
	if (k >= 0 && k < RGBDDatas::frames.size())
	{
		RGBDFrame f = RGBDDatas::frames[k];
		cloud = ConvertToPointCloudWithoutMissingData(f.depth, f.color, f.timestamp, k);
		normal = ComputeNormal(cloud);
	}
}

void RGBDDatas::getOrganizedCloudWithNormal(int k, PointCloudPtr &cloud, PointCloudNormalPtr &normal)
{
	if (k >= 0 && k < RGBDDatas::frames.size())
	{
		RGBDFrame f = RGBDDatas::frames[k];
		cloud = ConvertToOrganizedPointCloud(f.depth, f.color, f.timestamp, k);
		normal = ComputeOrganizedNormal(cloud);
	}
}

PointCloudWithNormalPtr RGBDDatas::getOrganizedCloudWithNormal_cuda(int k)
{
	PointCloudWithNormalPtr cloud;
	if (k >= 0 && k < RGBDDatas::frames.size())
	{
		RGBDFrame f = RGBDDatas::frames[k];
		ConvertToPointCloudWithNormalCuda(cloud, f.depth, f.color, f.timestamp, k);
		return cloud;
	}
	return PointCloudWithNormalPtr(new PointCloudWithNormalT);
}

int RGBDDatas::getFrameCount()
{
	return frames.size();
}