#include "CommonFunction.h"

PointCloudCuda::PointCloudCuda(int width,
                         int height,
                         float cx, float cy, float fx, float fy,
						 float depthFactor)
: width(width),
  height(height),
  depthFactor_(depthFactor)
{
    intr.cx = cx;
    intr.cy = cy;
    intr.fx = fx;
    intr.fy = fy;

	depth_tmp.create(height, width);

	vmaps_.create(height * 3, width);
	nmaps_.create(height * 3, width);
}

PointCloudCuda::~PointCloudCuda()
{
	depth_tmp.release();
	vmaps_.release();
	nmaps_.release();
}

void PointCloudCuda::init(unsigned short * depth)
{
    depth_tmp.upload(depth, sizeof(unsigned short) * width, height, width);
}

void PointCloudCuda::getVMap(cv::Mat &mat, const float depthCutoff)
{
	createVMap(intr, depth_tmp, vmaps_, depthCutoff, depthFactor_);
	cudaDeviceSynchronize();

	DeviceArray2D<float> dst;
	rearrangeMap(vmaps_, dst);
	mat = cv::Mat(height, width, CV_32FC3);
	dst.download(mat.data, mat.step[0]);
	dst.release();
}

void PointCloudCuda::getNMap(cv::Mat &mat)
{
	createNMap(vmaps_, nmaps_);
	cudaDeviceSynchronize();

	DeviceArray2D<float> dst;
	rearrangeMap(nmaps_, dst);
	mat = cv::Mat(height, width, CV_32FC3);
	dst.download(mat.data, mat.step[0]);
	dst.release();
}
