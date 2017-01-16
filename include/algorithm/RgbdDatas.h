#pragma once

#include <opencv2/opencv.hpp>
#include "PointCloud.h"
#include "RGBDReader.h"

struct RGBDFrame
{
	double timestamp;
	cv::Mat color;
	cv::Mat depth;

	RGBDFrame() { }
	RGBDFrame(cv::Mat color, cv::Mat depth, double timestamp);
};

class RGBDDatas
{
public:
	static void push(const cv::Mat &r, const cv::Mat &d, double ts);

	static double getTimestamp(int k);

	static RGBDFrame getFrame(int k);

	static void getFrame(int k, cv::Mat &c, cv::Mat &d);

	static PointCloudPtr getCloud(int k);

	static PointCloudPtr getOrganizedCloud(int k);

	static void getCloudWithNormal(int k, PointCloudPtr &cloud, PointCloudNormalPtr &normal);

	static void getOrganizedCloudWithNormal(int k, PointCloudPtr &cloud, PointCloudNormalPtr &normal);

	static PointCloudWithNormalPtr getOrganizedCloudWithNormal_cuda(int k);

	static int getFrameCount();

public:

	static std::vector<RGBDFrame> frames;
	static Intrinsic intr;
};