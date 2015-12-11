#pragma once

#include <pcl/common/common_headers.h>
#include <XnCppWrapper.h>
#include <opencv2/opencv.hpp>

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudType;
typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloudPtr;

PointCloudPtr convertToPointCloud(const cv::Mat &depth, const cv::Mat &rgb, double timestamp, int frameID);

PointCloudPtr convertToPointCloudWithoutMissingData(const cv::Mat &depth, const cv::Mat &rgb, double timestamp, int frameID);

PointCloudPtr convertToPointCloud(xn::DepthGenerator& rDepthGen, const XnDepthPixel *dm, const XnRGB24Pixel *im, XnUInt64 timestamp, XnInt32 frameID);

Eigen::Vector3f convertPointTo3D(int i, int j, const cv::Mat &depth);
