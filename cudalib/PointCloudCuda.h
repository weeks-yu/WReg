#ifndef POINTCLOUDCUDA_H_
#define POINTCLOUDCUDA_H_

#include "Cuda/internal.h"
#include "OdometryProvider.h"

class PointCloudCuda
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	PointCloudCuda(unsigned short * depth,
		int width,
		int height,
		float cx, float cy, float fx, float fy,
		float depthFactor);

	virtual ~PointCloudCuda();

	void getVmap();

private:
	DeviceArray2D<unsigned short> depth_tmp;
	DeviceArray2D<float> vmap;
	DeviceArray2D<float> nmap;

	Intr intr;

	const int width;
	const int height;
	const float depthFactor_;
};

#endif