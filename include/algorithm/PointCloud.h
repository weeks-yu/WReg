#pragma once

#include <pcl/common/common_headers.h>
#include <pcl/filters/voxel_grid.h>
#include <opencv2/core/core.hpp>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudT;
typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudPtr;

typedef pcl::Normal NormalT;
typedef pcl::PointCloud<pcl::Normal> PointCloudNormalT;
typedef pcl::PointCloud<pcl::Normal>::Ptr PointCloudNormalPtr;

PointCloudPtr ConvertToPointCloud(const cv::Mat &depth, const cv::Mat &rgb, double timestamp, int frameID);

PointCloudPtr ConvertToPointCloudWithoutMissingData(const cv::Mat &depth, const cv::Mat &rgb, double timestamp, int frameID);

void ConvertToPointCloudWithNormalCuda(PointCloudPtr &cloud, PointCloudNormalPtr &normal,
	const cv::Mat &depth, const cv::Mat &rgb, double timestamp, int frameID);

Eigen::Vector3f ConvertPointTo3D(int i, int j, const cv::Mat &depth);

PointCloudPtr DownSamplingByVoxelGrid(PointCloudPtr cloud, float lx, float ly, float lz);
