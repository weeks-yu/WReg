#include "PointCloud.h"
#include "Config.h"

PointCloudPtr ConvertToPointCloud(const cv::Mat &depth, const cv::Mat &rgb, double timestamp, int frameID)
{
	PointCloudPtr cloud(new PointCloudT);
	cloud->header.stamp = timestamp * 10000;
	cloud->header.seq = frameID;

	cloud->height = rgb.size().height;
	cloud->width = rgb.size().width;
	cloud->points.resize(cloud->height * cloud->width);

	double fx = Config::instance()->get<double>("camera_fx");  // focal length x
	double fy = Config::instance()->get<double>("camera_fy");  // focal length y
	double cx = Config::instance()->get<double>("camera_cx");  // optical center x
	double cy = Config::instance()->get<double>("camera_cy");  // optical center y

	double factor = Config::instance()->get<double>("depth_factor");	// for the 16-bit PNG files
	// OR: factor = 1 # for the 32-bit float images in the ROS bag files

	for (int j = 0; j < cloud->height; j++)
	{
		for (int i = 0; i < cloud->width; i++)
		{
			PointT& pt = cloud->points[j * cloud->width + i];
			pt.z = ((double)depth.at<ushort>(j, i)) / factor;
			pt.x = (i - cx) * pt.z / fx;
			pt.y = (j - cy) * pt.z / fy;
			pt.b = rgb.at<cv::Vec3b>(j, i)[0];
			pt.g = rgb.at<cv::Vec3b>(j, i)[1];
			pt.r = rgb.at<cv::Vec3b>(j, i)[2];
		}
	}

	return cloud;
}

PointCloudPtr ConvertToPointCloudWithoutMissingData(const cv::Mat &depth, const cv::Mat &rgb, double timestamp, int frameID)
{
	PointCloudPtr cloud(new PointCloudT);
	cloud->header.stamp = timestamp * 10000;
	cloud->header.seq = frameID;

	// 	cloud->height = rgb.size().height;
	// 	cloud->width = rgb.size().width;
	//	cloud->points.resize(cloud->height * cloud->width);

	double fx = Config::instance()->get<double>("camera_fx");  // focal length x
	double fy = Config::instance()->get<double>("camera_fy");  // focal length y
	double cx = Config::instance()->get<double>("camera_cx");  // optical center x
	double cy = Config::instance()->get<double>("camera_cy");  // optical center y

	double factor = Config::instance()->get<double>("depth_factor");	// for the 16-bit PNG files
	// OR: factor = 1 # for the 32-bit float images in the ROS bag files

	for (int j = 0; j < rgb.size().height; j++)
	{
		for (int i = 0; i < rgb.size().width; i++)
		{
			ushort temp = depth.at<ushort>(j, i);
			if (depth.at<ushort>(j, i) != 0)
			{
				PointT pt;
				pt.z = ((double)depth.at<ushort>(j, i)) / factor;
				pt.x = (i - cx) * pt.z / fx;
				pt.y = (j - cy) * pt.z / fy;
				pt.b = rgb.at<cv::Vec3b>(j, i)[0];
				pt.g = rgb.at<cv::Vec3b>(j, i)[1];
				pt.r = rgb.at<cv::Vec3b>(j, i)[2];
				cloud->push_back(pt);
			}
		}
	}

	return cloud;
}

Eigen::Vector3f ConvertPointTo3D(int i, int j, const cv::Mat &depth)
{
	Eigen::Vector3f pt;

	ushort temp = depth.at<ushort>(j, i);
	if (temp != 0)
	{
		pt(2) = ((double)temp) / Config::instance()->get<double>("depth_factor");
		pt(0) = (i - Config::instance()->get<double>("camera_cx")) * pt(2) / Config::instance()->get<double>("camera_fx");
		pt(1) = (j - Config::instance()->get<double>("camera_cy")) * pt(2) / Config::instance()->get<double>("camera_fy");
	}
	else
	{
		pt(2) = 0;
		pt(0) = 0;
		pt(1) = 0;
	}

	return pt;
}

PointCloudPtr DownSamplingByVoxelGrid(PointCloudPtr cloud, float lx, float ly, float lz)
{
	pcl::VoxelGrid<PointT> vg;
	vg.setInputCloud(cloud);
	vg.setLeafSize(lx, ly, lz);
	PointCloudPtr result(new PointCloudT);
	vg.filter(*result);
	return result;
}