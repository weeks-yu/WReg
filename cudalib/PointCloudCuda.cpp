#include "PointCloudCuda.h"

PointCloudCuda::PointCloudCuda(unsigned short * depth,
	int width,
	int height,
	float cx, float cy, float fx, float fy,
	float depthFactor)
:	width(width),
	height(height),
	depthFactor_(depthFactor)
{
	intr.cx = cx;
	intr.cy = cy;
	intr.fx = fx;
	intr.fy = fy;

	depth_tmp.create(height, width);
	depth_tmp.upload(depth, sizeof(unsigned short)* width, height, width);
}