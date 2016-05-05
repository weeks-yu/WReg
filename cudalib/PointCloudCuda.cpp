#include "PointCloudCuda.h"

PointCloudCuda::PointCloudCuda(int width,
	int height,
	float cx, float cy, float fx, float fy,
	float depthFactor,
	float distThresh /* = 0.10f */,
	float angleThresh /* = sin(20.f * 3.14159254f / 180.f) */)
:	width(width),
	height(height),
	depthFactor_(depthFactor),
	distThres_(distThresh),
	angleThres_(angleThresh)
{
	sumData.create(MAX_THREADS);
	outData.create(1);

	intr.cx = cx;
	intr.cy = cy;
	intr.fx = fx;
	intr.fy = fy;

	depth_tmp.create(height, width);

	vmaps_g_prev_.create(height * 3, width);
	nmaps_g_prev_.create(height * 3, width);

	vmaps_curr_.create(height * 3, width);
	nmaps_curr_.create(height * 3, width);
}

PointCloudCuda::~PointCloudCuda()
{

}

void PointCloudCuda::initPrev(unsigned short * depth, const float depthCutoff)
{
	depth_tmp.upload(depth, sizeof(unsigned short)* width, height, width);

	createVMap(intr, depth_tmp, vmaps_g_prev_, depthCutoff, depthFactor_);
	createNMap(vmaps_g_prev_, nmaps_g_prev_);

	cudaDeviceSynchronize();
}

void PointCloudCuda::initCurr(unsigned short * depth, const float depthCutoff)
{
	depth_tmp.upload(depth, sizeof(unsigned short)* width, height, width);

	createVMap(intr, depth_tmp, vmaps_curr_, depthCutoff, depthFactor_);
	createNMap(vmaps_curr_, nmaps_curr_);

	cudaDeviceSynchronize();
}

void PointCloudCuda::getCoresp(const Eigen::Vector3f & trans, const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
	Eigen::Matrix<float, 6, 6, Eigen::RowMajor> &information,
	int & point_count_curr, int & corr, int threads, int blocks)
{
	//test();

	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = rot;
	Eigen::Vector3f tcurr = trans;

	Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
	float3& device_tcurr = device_cast<float3>(tcurr);
	
	int result[2];

	calcCorr(device_Rcurr,
		device_tcurr,
		vmaps_curr_,
		nmaps_curr_,
		intr,
		vmaps_g_prev_,
		nmaps_g_prev_,
		distThres_,
		angleThres_,
		sumData,
		outData,
		information.data(),
		&result[0],
		threads,
		blocks);

	point_count_curr = result[0];
	corr = result[1];
}