#ifndef POINTCLOUDCUDA_H_
#define POINTCLOUDCUDA_H_

#include "Cuda/internal.h"
#include "OdometryProvider.h"

class PointCloudCuda
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	PointCloudCuda(int width,
		int height,
		float cx, float cy, float fx, float fy,
		float depthFactor,
		float distThresh = 0.10f,
		float angleThresh = sin(20.f * 3.14159254f / 180.f));

	virtual ~PointCloudCuda();

	void initCurr(unsigned short * depth, const float depthCutoff);

	void initPrev(unsigned short * depth, const float depthCutoff);

	void getCoresp(const Eigen::Vector3f & trans, const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
		Eigen::Matrix<double, 6, 6> &information,
		int & point_count_curr, int & corr, int threads, int blocks);

	void getCorespPairs(const Eigen::Vector3f & trans, const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
		Eigen::Matrix<double, 6, 6> &information,
		cv::Mat &pairs,
		int & point_count_curr, int & corr, int threads, int blocks);

private:
	DeviceArray2D<unsigned short> depth_tmp;
	DeviceArray2D<float> vmaps_g_prev_;
	DeviceArray2D<float> nmaps_g_prev_;

	DeviceArray2D<float> vmaps_curr_;
	DeviceArray2D<float> nmaps_curr_;

	Intr intr;

	DeviceArray<jtj> sumData;
	DeviceArray<jtj> outData;

	float distThres_;
	float angleThres_;

	const int width;
	const int height;
	const float depthFactor_;
};

#endif