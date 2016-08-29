#include "PointCloud.h"
#include "Config.h"

#ifdef SHOW_Z_INDEX
float now_max_z, now_min_z;
#endif

PointCloudPtr ConvertToPointCloudWithoutMissingData(const cv::Mat &depth, const cv::Mat &rgb, double timestamp, int frameID)
{
	PointCloudPtr cloud(new PointCloudT);
	cloud->header.stamp = timestamp * 10000;
	cloud->header.seq = frameID;

#ifdef SHOW_Z_INDEX
	float min_z = 5000.0, max_z = 0.0;
#endif

	float fx = Config::instance()->get<float>("camera_fx");  // focal length x
	float fy = Config::instance()->get<float>("camera_fy");  // focal length y
	float cx = Config::instance()->get<float>("camera_cx");  // optical center x
	float cy = Config::instance()->get<float>("camera_cy");  // optical center y

	float factor = Config::instance()->get<float>("depth_factor");	// for the 16-bit PNG files

	for (int j = 0; j < rgb.size().height; j++)
	{
		for (int i = 0; i < rgb.size().width; i++)
		{
			ushort temp = depth.at<ushort>(j, i);
			if (temp > 0)
			{
				PointT pt;
				pt.z = ((double)temp) / factor;
				pt.x = (i - cx) * pt.z / fx;
				pt.y = (j - cy) * pt.z / fy;
				pt.b = rgb.at<cv::Vec3b>(j, i)[0];
				pt.g = rgb.at<cv::Vec3b>(j, i)[1];
				pt.r = rgb.at<cv::Vec3b>(j, i)[2];
				cloud->push_back(pt);

#ifdef SHOW_Z_INDEX
				if (pt.z < min_z) min_z = pt.z;
				if (pt.z > max_z) max_z = pt.z;
#endif
			}
		}
	}

#ifdef SHOW_Z_INDEX
	now_min_z = min_z;
	now_max_z = max_z;
#endif
	return cloud;
}

Eigen::Vector3f ConvertPointTo3D(int i, int j, const cv::Mat &depth)
{
	Eigen::Vector3f pt;

	ushort temp = depth.at<ushort>(j, i);
	if (temp != 0)
	{
		pt(2) = ((float)temp) / Config::instance()->get<float>("depth_factor");
		pt(0) = (i - Config::instance()->get<float>("camera_cx")) * pt(2) / Config::instance()->get<float>("camera_fx");
		pt(1) = (j - Config::instance()->get<float>("camera_cy")) * pt(2) / Config::instance()->get<float>("camera_fy");
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