#include "PointCloud.h"
#include "Config.h"

PointCloudPtr ConvertToPointCloud(const cv::Mat &depth, const cv::Mat &rgb, double timestamp, int frameID)
{
	PointCloudPtr cloud(new PointCloudType());
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
			pcl::PointXYZRGB& pt = cloud->points[j * cloud->width + i];
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
	PointCloudPtr cloud(new PointCloudType());
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
				pcl::PointXYZRGB pt;
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

// PointCloudPtr convertToPointCloud(xn::DepthGenerator& rDepthGen,
// 	const XnDepthPixel *dm, const XnRGB24Pixel *im,
// 	XnUInt64 timestamp, XnInt32 frameID)		
// {
// 	PointCloudPtr cloud(new PointCloudType());
// 
// 	// Not supported in file yet:
// 	cloud->header.stamp = timestamp;
// 	cloud->header.seq = frameID;
// 	// End not supported in file yet
// 
// 	cloud->height = Config::instance()->get<int>("image_height");
// 	cloud->width = Config::instance()->get<int>("image_width");
// 	cloud->is_dense = false;
// 
// 	cloud->points.resize(cloud->height * cloud->width);
// 
// 	register int centerX = cloud->width >> 1;
// 	int centerY = cloud->height >> 1;
// 
// 	register const XnDepthPixel* depth_map = dm;
// 	register const XnRGB24Pixel* rgb_map = im;
// 
// 	unsigned int uPointNum = cloud->width * cloud->height;
// 
// 
// 	XnPoint3D* pDepthPointSet = new XnPoint3D[ uPointNum ];
// 	unsigned int i, j, idxShift, idx;
// 	for( j = 0; j < cloud->height; ++j )
// 	{
// 		idxShift = j * cloud->width;
// 		for( i = 0; i < cloud->width; ++i )
// 		{
// 			idx = idxShift + i;
// 			pDepthPointSet[idx].X = i;
// 			pDepthPointSet[idx].Y = j;
// 			pDepthPointSet[idx].Z = depth_map[idx];
// 		}
// 	}
// 
// 	XnPoint3D* p3DPointSet = new XnPoint3D[ uPointNum ];
// 	rDepthGen.ConvertProjectiveToRealWorld( uPointNum, pDepthPointSet, p3DPointSet );
// 	delete[] pDepthPointSet;
// 	register int depth_idx = 0;
// 	PointCloudType::iterator iter = cloud->begin();
// 	for (int v = -centerY; v < centerY; ++v)
// 	{
// 		for (register int u = -centerX; u < centerX; ++u, ++depth_idx)
// 		{
// 			pcl::PointXYZRGB& pt = cloud->points[depth_idx];
// 
// 			pt.z = p3DPointSet[depth_idx].Z * 0.001f;
// 			pt.x = p3DPointSet[depth_idx].X * 0.001f;
// 			pt.y = p3DPointSet[depth_idx].Y * 0.001f;
// 			pt.r = (float) rgb_map[depth_idx].nRed;
// 			pt.g = (float) rgb_map[depth_idx].nGreen;
// 			pt.b = (float) rgb_map[depth_idx].nBlue;
// 		}
// 	}
// 	return cloud;
// }

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
	pcl::VoxelGrid<pcl::PointXYZRGB> vg;
	vg.setInputCloud(cloud);
	vg.setLeafSize(lx, ly, lz);
	PointCloudPtr result(new pcl::PointCloud<pcl::PointXYZRGB>);
	vg.filter(*result);
	return result;
}