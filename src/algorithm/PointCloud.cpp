#include "PointCloud.h"
#include "Config.h"
#include "CommonFunction.h"
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>

#ifdef SHOW_Z_INDEX
extern float now_max_z, now_min_z;
#endif

PointCloudPtr ConvertToOrganizedPointCloud(const Intrinsic &intr,
	const cv::Mat &depth, const cv::Mat &rgb,
	double timestamp, int frameID)
{
	PointCloudPtr cloud(new PointCloudT);
	cloud->width = rgb.size().width;
	cloud->height = rgb.size().height;
	cloud->is_dense = false;
	cloud->resize(cloud->height * cloud->width);
	cloud->header.stamp = timestamp * 10000;
	cloud->header.seq = frameID;

#ifdef SHOW_Z_INDEX
	float min_z = 5000.0, max_z = 0.0;
#endif

	float fx = intr.fx;  // focal length x
	float fy = intr.fy;  // focal length y
	float cx = intr.cx;  // optical center x
	float cy = intr.cy;  // optical center y

	float factor = intr.zFactor;

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

#ifdef SHOW_Z_INDEX
			if (pt.z < min_z) min_z = pt.z;
			if (pt.z > max_z) max_z = pt.z;
#endif
		}
	}

#ifdef SHOW_Z_INDEX
	now_min_z = min_z;
	now_max_z = max_z;
#endif
	return cloud;
}

PointCloudPtr ConvertToPointCloudWithoutMissingData(const Intrinsic &intr,
	const cv::Mat &depth, const cv::Mat &rgb,
	double timestamp, int frameID)
{
	PointCloudPtr cloud(new PointCloudT);
	cloud->header.stamp = timestamp * 10000;
	cloud->header.seq = frameID;

#ifdef SHOW_Z_INDEX
	float min_z = 5000.0, max_z = 0.0;
#endif

	float fx = intr.fx;  // focal length x
	float fy = intr.fy;  // focal length y
	float cx = intr.cx;  // optical center x
	float cy = intr.cy;  // optical center y

	float factor = intr.zFactor;

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

PointCloudNormalPtr ComputeNormal(PointCloudPtr cloud)
{
	PointCloudNormalPtr normal(new PointCloudNormalT);
	pcl::NormalEstimation<PointT, NormalT> ne;
	ne.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(0.03);
	ne.compute(*normal);

	return normal;
}

PointCloudNormalPtr ComputeOrganizedNormal(PointCloudPtr cloud)
{
	PointCloudNormalPtr normal(new PointCloudNormalT);
	pcl::IntegralImageNormalEstimation<PointT, NormalT> ne;
	ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
	ne.setMaxDepthChangeFactor(0.03f);
	ne.setNormalSmoothingSize(20.0f);
	ne.setInputCloud(cloud);
	ne.compute(*normal);

	return normal;
}

void ConvertToPointCloudWithNormalCuda(	PointCloudWithNormalPtr &cloud,	const Intrinsic &intr,
	const cv::Mat &depth, const cv::Mat &rgb,
	double timestamp, int frameID)
{
	cloud = PointCloudWithNormalPtr(new PointCloudWithNormalT);
	cloud->width = rgb.size().width;
	cloud->height = rgb.size().height;
	cloud->is_dense = false;
	cloud->resize(cloud->height * cloud->width);
	
	float fx = intr.fx;  // focal length x
	float fy = intr.fy;  // focal length y
	float cx = intr.cx;  // optical center x
	float cy = intr.cy;  // optical center y

	float factor = intr.zFactor;

	PointCloudCuda *pcc = new PointCloudCuda(rgb.size().width, rgb.size().height,
		cx, cy, fx, fy, factor);
	pcc->init((unsigned short *)depth.data);
	cv::Mat vmap, nmap;
	pcc->getVMap(vmap, 20.0f);
	pcc->getNMap(nmap);

	for (int j = 0; j < cloud->height; j++)
	{
		for (int i = 0; i < cloud->width; i++)
		{
			PointWithNormalT& pt = cloud->points[j * cloud->width + i];
			pt.z = vmap.at<cv::Vec3f>(j, i)[2];
			pt.x = vmap.at <cv::Vec3f>(j, i)[0];
			pt.y = vmap.at <cv::Vec3f>(j, i)[1];
			pt.b = rgb.at<cv::Vec3b>(j, i)[0];
			pt.g = rgb.at<cv::Vec3b>(j, i)[1];
			pt.r = rgb.at<cv::Vec3b>(j, i)[2];

			pt.normal_x = nmap.at<cv::Vec3f>(j, i)[0];
			pt.normal_y = nmap.at<cv::Vec3f>(j, i)[1];
			pt.normal_z = nmap.at<cv::Vec3f>(j, i)[2];
		}
	}
	delete pcc;
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
